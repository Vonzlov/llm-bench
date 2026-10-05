"""Ступенчатая нагрузка на модель и таблица итогов по каждой ступени.

    uv run bench-load
    uv run bench-load --levels 1,2,4,8 --warmup 2 --measure 10

Без аргументов прогон идёт по методике: ступени 1–32, 20 секунд прогрева и 120 секунд
замера на каждой, это около 14 минут. Вторая команда — быстрый прогон, как make load.
По умолчанию нагрузка идёт в фейковую модель на localhost:8000.

Промпты пока короткие и встроенные: пул из абзацев XQuAD появится вместе с загрузкой
наборов данных. Ключ API, если он нужен, берётся из переменной окружения BENCH_API_KEY.
"""

import argparse
import asyncio
import os
import sys

import httpx

from bench_core.client import CHAT_PATH
from bench_core.load import DEFAULT_LEVELS, LevelResult, run_level

DEFAULT_PROMPTS = [
    "Коротко расскажи, что такое Kubernetes.",
    "Объясни разницу между процессом и потоком.",
    "Чем отличается TCP от UDP?",
    "Что такое индекс в базе данных и зачем он нужен?",
    "Опиши, как работает кеширование в браузере.",
]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bench-load",
        description="Ступенчатая нагрузка на модель: закрытая модель, N воркеров.",
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="адрес API с /v1")
    parser.add_argument("--model", default="fake-llm", help="имя модели на сервере")
    default_levels = ",".join(str(level) for level in DEFAULT_LEVELS)
    parser.add_argument("--levels", default=default_levels, help="ступени через запятую")
    parser.add_argument("--warmup", type=float, default=20.0, help="прогрев ступени, секунды")
    parser.add_argument("--measure", type=float, default=120.0, help="окно замера, секунды")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300.0, help="таймаут запроса, секунды")
    args = parser.parse_args(argv)

    args.levels = [int(level) for level in args.levels.split(",")]
    if min(args.levels) < 1 or args.warmup < 0 or args.measure <= 0:
        parser.error("ступени — целые числа от 1, прогрев не меньше 0, замер больше 0")
    return args


def pair(p50: float | None, p95: float | None, digits: int) -> str:
    """Два перцентиля в миллисекундах через косую черту."""
    if p50 is None or p95 is None:
        return "—"
    return f"{p50 * 1000:.{digits}f} / {p95 * 1000:.{digits}f}"


def row(*cells: str) -> str:
    """Строка таблицы. Ширины колонок подобраны под заголовки."""
    widths = (4, 8, 6, 16, 15, 17, 7, 5)
    return "  ".join(cell.rjust(width) for cell, width in zip(cells, widths, strict=True))


HEADER = row(
    "N",
    "запросов",
    "ошибок",
    "TTFT p50/p95, мс",
    "ITL p50/p95, мс",
    "время p50/p95, мс",
    "ток/с",
    "зап/с",
)


def format_level(result: LevelResult) -> str:
    lines = [
        row(
            str(result.concurrency),
            str(result.n_requests),
            str(result.n_errors),
            pair(result.ttft_p50_s, result.ttft_p95_s, 0),
            pair(result.itl_p50_s, result.itl_p95_s, 1),
            pair(result.latency_p50_s, result.latency_p95_s, 0),
            f"{result.tokens_per_s:.1f}",
            f"{result.requests_per_s:.2f}",
        )
    ]
    if result.errors:
        details = ", ".join(f"{kind} ×{count}" for kind, count in sorted(result.errors.items()))
        lines.append(f"      ошибки: {details}")
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> list[LevelResult]:
    headers = {}
    api_key = os.environ.get("BENCH_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    # Соединений держим столько, сколько воркеров на самой высокой ступени. Иначе httpx
    # закрывает лишние после ответа, и установка соединения попадает в замер.
    top = max(args.levels)
    limits = httpx.Limits(max_connections=top, max_keepalive_connections=top)

    results = []
    async with httpx.AsyncClient(
        base_url=args.base_url, headers=headers, timeout=args.timeout, limits=limits
    ) as client:
        for concurrency in args.levels:
            result = await run_level(
                client,
                model=args.model,
                prompts=DEFAULT_PROMPTS,
                concurrency=concurrency,
                max_tokens=args.max_tokens,
                warmup_s=args.warmup,
                measure_s=args.measure,
            )
            print(format_level(result), flush=True)
            results.append(result)
    return results


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    minutes = len(args.levels) * (args.warmup + args.measure) / 60
    url = f"{args.base_url.rstrip('/')}{CHAT_PATH}"
    print(f"POST {url}, модель {args.model}")
    print(
        f"ступени {args.levels}, прогрев {args.warmup:g} с, замер {args.measure:g} с"
        f" — около {minutes:.1f} мин"
    )
    print(HEADER, flush=True)
    results = asyncio.run(run(args))
    # Код выхода 1, если хоть на одной ступени не было ни одного успешного запроса.
    all_levels_worked = all(result.n_requests > result.n_errors for result in results)
    sys.exit(0 if all_levels_worked else 1)


if __name__ == "__main__":
    main()
