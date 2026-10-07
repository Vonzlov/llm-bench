"""Качество модели на наборе: метрики с 95-процентными бутстреп-интервалами.

    uv run bench-quality --dataset data/massive-ru-intent.jsonl
    uv run bench-quality --dataset data/massive-ru-intent.jsonl --model llama3.2:3b --limit 50

Пока умеет классификацию; извлечение и ответы по тексту появятся следующими. Ответ на
каждый пример пишется в data/results/<набор>--<модель>.jsonl, итог прогона — рядом, в
<набор>--<модель>.summary.json. Ключ API, если он нужен, берётся из BENCH_API_KEY.
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

from bench_core import classification
from bench_core.client import CHAT_PATH
from bench_core.quality import ItemResult, Metric, Task, load_dataset, run_task, write_results
from bench_core.stats import SEED, percentile

TASKS = {"classification": classification.TASK}
TITLES = {"accuracy": "точность", "macro_f1": "macro-F1", "out_of_list": "вне списка"}


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
    args = parser.parse_args(argv)
    if args.concurrency < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--concurrency и --limit — целые числа от 1")
    return args


def slug(name: str) -> str:
    """Имя модели для имени файла: всё, кроме букв, цифр, точки и дефиса, — в дефис."""
    return re.sub(r"[^\w.-]+", "-", name).strip("-")


def format_metric(name: str, value: Metric) -> str:
    title = TITLES.get(name, name)
    return f"  {title:<11} {value.value:.3f}   интервал [{value.low:.3f}; {value.high:.3f}]"


async def run(
    args: argparse.Namespace,
    task: Task,
    items: list[dict[str, Any]],
    labels: list[str],
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

    if args.limit:
        items = items[: args.limit]
    labels = entry.get("labels", [])
    max_tokens = args.max_tokens or task.max_tokens
    out = args.out or Path("data/results") / f"{args.dataset.stem}--{slug(args.model)}.jsonl"
    url = f"{args.base_url.rstrip('/')}{CHAT_PATH}"
    print(f"POST {url}, модель {args.model}")
    print(f"набор {args.dataset.stem}: {task.name}, примеров {len(items)}")
    print(
        f"одновременных запросов {args.concurrency}, температура 0, ответ до {max_tokens} токенов"
    )

    started = time.perf_counter()
    results = asyncio.run(run(args, task, items, labels, max_tokens))
    elapsed_s = time.perf_counter() - started
    metrics = task.summarize(results)
    errors = Counter(result.error for result in results if result.error is not None)
    latency_p50_s = percentile([r.latency_s for r in results if r.error is None], 50)

    print("метрики; интервал — 95-процентный, бутстреп по 1000 выборкам:")
    for name, value in metrics.items():
        print(format_metric(name, value))
    if errors:
        details = ", ".join(f"{kind} ×{count}" for kind, count in sorted(errors.items()))
        print(f"ошибки запросов, засчитаны как неверные ответы: {details}")
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
        "temperature": 0,
        "seed": SEED,
        "metrics": {name: asdict(value) for name, value in metrics.items()},
        "errors": dict(errors),
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
