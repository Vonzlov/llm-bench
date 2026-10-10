"""Качество модели на наборе: метрики с 95-процентными бутстреп-интервалами.

    uv run bench-quality --dataset data/massive-ru-intent.jsonl
    uv run bench-quality --dataset data/massive-ru-intent.jsonl --model llama3.2:3b --limit 50
    uv run bench-quality --dataset data/massive-ru-slots.jsonl --constrained

Умеет классификацию и извлечение в JSON; ответы по тексту появятся следующими. Перед
вопросом модель видит три примера с ответами из manifest.json (few-shot); --shots 0
спрашивает без примеров. Ответ на каждый пример пишется в
data/results/<набор>--<модель>.jsonl, итог прогона — рядом, в .summary.json. У прогона
с --constrained к имени добавляется --schema, а при числе примеров не по умолчанию —
например, --0shot. Ключ API, если он нужен, берётся из переменной окружения BENCH_API_KEY.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from bench_core import classification, extraction
from bench_core.client import CHAT_PATH
from bench_core.quality import (
    ItemResult,
    Metric,
    Task,
    load_dataset,
    run_task,
    summarize,
    write_results,
)
from bench_core.stats import SEED, percentile

TASKS = {"classification": classification.TASK, "extraction": extraction.TASK}
# Сколько примеров с ответами показываем перед вопросом по умолчанию.
DEFAULT_SHOTS = 3
TITLES = {
    "accuracy": "точность",
    "macro_f1": "macro-F1",
    "out_of_list": "вне списка",
    "valid_json": "валидный JSON",
    "schema_ok": "по схеме",
    "slot_f1": "F1 по слотам",
}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bench-quality", description="Качество модели на наборе данных."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="набор в JSONL, например data/massive-ru-intent.jsonl",
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="адрес API с /v1")
    parser.add_argument("--model", default="fake-llm", help="имя модели на сервере")
    parser.add_argument("--concurrency", type=int, default=4, help="сколько запросов одновременно")
    parser.add_argument(
        "--max-tokens", type=int, help="потолок длины ответа; по умолчанию свой у задачи"
    )
    parser.add_argument("--limit", type=int, help="только первые N примеров — для быстрой проверки")
    parser.add_argument("--timeout", type=float, default=120.0, help="таймаут запроса, секунды")
    parser.add_argument("--out", type=Path, help="файл для ответов; по умолчанию в data/results/")
    parser.add_argument(
        "--constrained",
        action="store_true",
        help="ограниченная генерация: движок получает JSON-схему ответа",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_SHOTS,
        help="сколько примеров с ответами показать перед вопросом; 0 — без примеров",
    )
    args = parser.parse_args(argv)
    if args.concurrency < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--concurrency и --limit — целые числа от 1")
    if args.shots < 0:
        parser.error("--shots не может быть меньше нуля")
    return args


def slug(name: str) -> str:
    """Имя модели для имени файла: всё, кроме букв, цифр, точки и дефиса, — в дефис."""
    return re.sub(r"[^\w.-]+", "-", name).strip("-")


def format_metric(name: str, value: Metric) -> str:
    title = TITLES.get(name, name)
    return f"  {title:<13} {value.value:.3f}   интервал [{value.low:.3f}; {value.high:.3f}]"


async def run(
    args: argparse.Namespace,
    task: Task,
    items: list[dict[str, Any]],
    labels: list[str],
    examples: list[dict[str, Any]],
    max_tokens: int,
) -> list[ItemResult]:
    headers = {}
    api_key = os.environ.get("BENCH_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    limits = httpx.Limits(
        max_connections=args.concurrency, max_keepalive_connections=args.concurrency
    )
    # Прогресс — примерно каждые 10% набора.
    step = max(1, len(items) // 10)

    def progress(done: int) -> None:
        if done % step == 0 or done == len(items):
            print(f"  готово {done} из {len(items)}", flush=True)

    async with httpx.AsyncClient(
        base_url=args.base_url, headers=headers, timeout=args.timeout, limits=limits
    ) as client:
        return await run_task(
            client,
            task,
            model=args.model,
            items=items,
            labels=labels,
            concurrency=args.concurrency,
            max_tokens=max_tokens,
            examples=examples,
            constrained=args.constrained,
            on_done=progress,
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        items, entry = load_dataset(args.dataset)
    except (OSError, ValueError) as error:
        print(f"ошибка: {error}", file=sys.stderr)
        sys.exit(1)
    task = TASKS.get(entry["task_type"])
    if task is None:
        print(f"ошибка: задачу {entry['task_type']} раннер пока не умеет", file=sys.stderr)
        sys.exit(1)
    if args.constrained and task.schema is None:
        print(f"ошибка: у задачи {task.name} нет режима с JSON-схемой", file=sys.stderr)
        sys.exit(1)

    # Меньше примеров, чем просили, — ошибка, а не тихий прогон с теми, что есть: иначе набор,
    # собранный до появления примеров, незаметно подменит прогон с примерами прогоном без них.
    available = entry.get("examples", [])
    if args.shots > len(available):
        print(
            f"ошибка: в наборе {args.dataset.stem} примеров с ответами {len(available)},"
            f" а просили {args.shots}: пересоберите наборы через make datasets"
            " или уменьшите --shots",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.limit:
        items = items[: args.limit]
    labels = entry.get("labels", [])
    examples = available[: args.shots]
    max_tokens = args.max_tokens or task.max_tokens
    mode = "--schema" if args.constrained else ""
    shots = f"--{args.shots}shot" if args.shots != DEFAULT_SHOTS else ""
    name = f"{args.dataset.stem}--{slug(args.model)}{mode}{shots}.jsonl"
    out = args.out or Path("data/results") / name
    url = f"{args.base_url.rstrip('/')}{CHAT_PATH}"
    print(f"POST {url}, модель {args.model}")
    print(f"набор {args.dataset.stem}: {task.name}, примеров {len(items)}")
    generation = "по JSON-схеме" if args.constrained else "свободная"
    print(
        f"одновременных запросов {args.concurrency}, температура 0, ответ до {max_tokens} токенов,"
        f" генерация {generation}, примеров с ответами в промпте {len(examples)}"
    )

    started = time.perf_counter()
    results = asyncio.run(run(args, task, items, labels, examples, max_tokens))
    elapsed_s = time.perf_counter() - started
    metrics = summarize(task, results)
    errors = Counter(result.error for result in results if result.error is not None)
    # Ответы, которые модель не закончила сама, а оборвал лимит токенов.
    truncated = sum(1 for result in results if result.finish_reason == "length")
    latency_p50_s = percentile([r.latency_s for r in results if r.error is None], 50)

    print("метрики; интервал — 95-процентный, бутстреп по 1000 выборкам:")
    for name, value in metrics.items():
        print(format_metric(name, value))
    if errors:
        details = ", ".join(f"{kind} ×{count}" for kind, count in sorted(errors.items()))
        print(f"ошибки запросов, засчитаны как неверные ответы: {details}")
    if truncated:
        print(f"обрезано лимитом в {max_tokens} токенов: {truncated} ответов из {len(results)}")
    if latency_p50_s is not None:
        print(f"время ответа p50: {latency_p50_s:.2f} с, весь прогон: {elapsed_s:.0f} с")

    write_results(results, out)
    summary = {
        "dataset": args.dataset.stem,
        "dataset_version": entry["version_hash"],
        "task_type": task.name,
        "model": args.model,
        "base_url": args.base_url,
        "n_items": len(items),
        "concurrency": args.concurrency,
        "max_tokens": max_tokens,
        "constrained": args.constrained,
        "shots": len(examples),
        "example_ids": [example["id"] for example in examples],
        "temperature": 0,
        "seed": SEED,
        "metrics": {name: asdict(value) for name, value in metrics.items()},
        "errors": dict(errors),
        "truncated": truncated,
        "latency_p50_s": latency_p50_s,
        "elapsed_s": round(elapsed_s, 1),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"ответы: {out}\nитог: {summary_path}")

    # Код выхода 1, если не ответил ни один запрос: метрики тогда ничего не значат.
    if sum(errors.values()) == len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
