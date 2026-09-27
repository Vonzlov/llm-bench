"""Проба: один запрос к модели и все замеры на экране.

    uv run bench-probe
    uv run bench-probe --base-url http://localhost:11434/v1 --model qwen2.5:0.5b

По умолчанию проба идёт в фейковую модель на localhost:8000 (make fake-llm).
Ключ API, если он нужен, берётся из переменной окружения BENCH_API_KEY, а не из
аргумента: так он не попадёт в историю терминала.
"""

import argparse
import asyncio
import os
import statistics
import sys

import httpx

from bench_core.client import CHAT_PATH, Measurement, stream_chat

DEFAULT_PROMPT = "Коротко расскажи, что такое Kubernetes."


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bench-probe", description="Один потоковый запрос к модели с замерами времени."
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="адрес API с /v1")
    parser.add_argument("--model", default="fake-llm", help="имя модели на сервере")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=60.0, help="таймаут, секунды")
    parser.add_argument("--warmup", type=int, default=1, help="запросов до замера")
    return parser.parse_args(argv)


async def probe(args: argparse.Namespace) -> Measurement:
    headers = {}
    api_key = os.environ.get("BENCH_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    messages = [{"role": "user", "content": args.prompt}]
    async with httpx.AsyncClient(
        base_url=args.base_url, headers=headers, timeout=args.timeout
    ) as client:
        # Первый запрос в процессе медленнее на десятки миллисекунд: прогреваются библиотеки
        # клиента и соединение. Поэтому сначала шлём прогревочные запросы и их не меряем.
        for _ in range(args.warmup):
            await stream_chat(
                client, model=args.model, messages=messages, max_tokens=args.max_tokens
            )
        return await stream_chat(
            client, model=args.model, messages=messages, max_tokens=args.max_tokens
        )


def ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f} мс"


def report(result: Measurement) -> str:
    """Замеры одного запроса в виде таблички для терминала."""
    lines = [f"статус             {'ok' if result.ok else result.error}"]
    if result.ttft_s is not None:
        lines.append(f"до первого токена  {ms(result.ttft_s)}")
    lines.append(f"полное время       {ms(result.latency_s)}")
    if result.chunk_times_s:
        chunks = len(result.chunk_times_s)
        lines.append(f"токенов            {result.output_tokens} (чанков {chunks})")
    if result.itl_s:
        lines.append(f"между токенами     {ms(statistics.median(result.itl_s))} (медиана)")
    if result.tpot_s:
        lines.append(f"скорость           {1 / result.tpot_s:.1f} ток/с")
    if result.retry_after_s is not None:
        lines.append(f"повторить через    {result.retry_after_s:g} с")
    if result.text:
        preview = result.text if len(result.text) <= 80 else result.text[:80] + "…"
        lines.append(f"ответ              «{preview}»")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = asyncio.run(probe(args))
    url = f"{args.base_url.rstrip('/')}{CHAT_PATH}"
    print(f"POST {url}, модель {args.model}, прогревочных запросов: {args.warmup}")
    print(report(result))
    sys.exit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
