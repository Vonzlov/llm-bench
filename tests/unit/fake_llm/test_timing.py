"""Проверка «линейки»: фейковая модель выдаёт ровно те задержки, что заданы.

Если эти тесты зелёные, то любое расхождение в генераторе нагрузки — ошибка
генератора, а не модели. Допуски небольшие, но с запасом на шум ОС в CI.
Нижние границы строгие: сервер никогда не должен отвечать быстрее заданного.
"""

import asyncio
import json
import time

import httpx
import pytest

from tests.unit.fake_llm.helpers import CHAT_URL, StartServer, chat_request


async def test_ttft_and_token_rate_match_settings(start_server: StartServer) -> None:
    base_url = start_server(ttft_ms=200, tokens_per_second=50, output_tokens=21)
    arrivals: list[float] = []

    async with httpx.AsyncClient(base_url=base_url) as client:
        started = time.perf_counter()
        async with client.stream("POST", CHAT_URL, json=chat_request(stream=True)) as response:
            async for line in response.aiter_lines():
                if not line.startswith("data: {"):
                    continue
                chunk = json.loads(line.removeprefix("data: "))
                if chunk["choices"] and chunk["choices"][0]["delta"].get("content"):
                    arrivals.append(time.perf_counter())

    ttft = arrivals[0] - started
    tokens_per_second = (len(arrivals) - 1) / (arrivals[-1] - arrivals[0])
    assert len(arrivals) == 21
    assert 0.2 <= ttft < 0.25
    assert tokens_per_second == pytest.approx(50, rel=0.1)


async def test_regular_answer_takes_ttft_plus_generation(start_server: StartServer) -> None:
    # 100 мс до первого токена и ещё 9 токенов по 20 мс: всего 280 мс.
    base_url = start_server(ttft_ms=100, tokens_per_second=50, output_tokens=10)
    async with httpx.AsyncClient(base_url=base_url) as client:
        started = time.perf_counter()
        await client.post(CHAT_URL, json=chat_request())
        elapsed = time.perf_counter() - started

    assert 0.28 <= elapsed < 0.33


async def test_queue_adds_waiting_time(start_server: StartServer) -> None:
    """При max_concurrency=1 запросы идут по одному, как у Ollama с OLLAMA_NUM_PARALLEL=1."""
    base_url = start_server(
        ttft_ms=100, tokens_per_second=100_000, output_tokens=1, max_concurrency=1
    )
    async with httpx.AsyncClient(base_url=base_url) as client:
        started = time.perf_counter()
        await asyncio.gather(*(client.post(CHAT_URL, json=chat_request()) for _ in range(3)))
        elapsed = time.perf_counter() - started

    assert 0.3 <= elapsed < 0.4


async def test_without_limit_requests_run_in_parallel(start_server: StartServer) -> None:
    base_url = start_server(ttft_ms=100, tokens_per_second=100_000, output_tokens=1)
    async with httpx.AsyncClient(base_url=base_url) as client:
        started = time.perf_counter()
        await asyncio.gather(*(client.post(CHAT_URL, json=chat_request()) for _ in range(3)))
        elapsed = time.perf_counter() - started

    assert 0.1 <= elapsed < 0.2
