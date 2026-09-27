"""Измерительный клиент против фейковой модели.

Фейковая модель выдаёт ровно те задержки, что ей заданы (это проверяют её собственные
тесты). Значит, клиент обязан намерить те же числа — это первая половина «проверки
линейки» из плана. Допуск 10%, как в плане, нижние границы строгие.
"""

import asyncio
import socket
import statistics
from typing import Any

import httpx
import pytest

from bench_core.client import Measurement, stream_chat
from tests.unit.fake_llm.helpers import MODEL, StartServer

# Три слова: фейковая модель считает токены промпта по словам.
MESSAGES = [{"role": "user", "content": "Привет, как дела?"}]
# Для тестов, где время не проверяется: сервер отвечает мгновенно.
INSTANT: dict[str, Any] = {"ttft_ms": 0, "tokens_per_second": 100_000}


async def measure(base_url: str, max_tokens: int = 10, http_timeout: float = 5.0) -> Measurement:
    async with httpx.AsyncClient(base_url=f"{base_url}/v1", timeout=http_timeout) as client:
        return await stream_chat(client, model=MODEL, messages=MESSAGES, max_tokens=max_tokens)


async def warm_up(client: httpx.AsyncClient) -> None:
    """Первый запрос в процессе медленнее на десятки миллисекунд: прогреваются библиотеки
    клиента и соединение. Поэтому в тестах на время, как и в методике, его не меряем."""
    await stream_chat(client, model=MODEL, messages=MESSAGES, max_tokens=1)


async def test_measures_ttft_and_token_intervals(start_server: StartServer) -> None:
    # 200 мс до первого токена, потом 20 токенов по 20 мс: всего около 600 мс.
    base_url = start_server(ttft_ms=200, tokens_per_second=50, output_tokens=64)
    async with httpx.AsyncClient(base_url=f"{base_url}/v1", timeout=5.0) as client:
        await warm_up(client)
        result = await stream_chat(client, model=MODEL, messages=MESSAGES, max_tokens=21)

    assert result.ok
    assert result.ttft_s is not None
    assert 0.2 <= result.ttft_s < 0.25
    assert len(result.chunk_times_s) == 21
    assert statistics.median(result.itl_s) == pytest.approx(0.02, rel=0.1)
    assert result.tpot_s == pytest.approx(0.02, rel=0.1)
    assert result.latency_s == pytest.approx(0.6, rel=0.1)


async def test_reads_text_usage_and_finish_reason(start_server: StartServer) -> None:
    base_url = start_server(**INSTANT, output_tokens=64)
    result = await measure(base_url, max_tokens=10)

    assert result.ok
    assert result.status_code == 200
    assert result.text.strip() != ""
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 10
    assert result.output_tokens == 10
    assert result.finish_reason == "length"


async def test_short_answer_finishes_with_stop(start_server: StartServer) -> None:
    base_url = start_server(**INSTANT, output_tokens=5)
    result = await measure(base_url, max_tokens=10)

    assert result.ok
    assert result.completion_tokens == 5
    assert result.finish_reason == "stop"


async def test_waiting_in_queue_counts_as_ttft(start_server: StartServer) -> None:
    """При одном слоте, как у Ollama с OLLAMA_NUM_PARALLEL=1, второй запрос ждёт первый."""
    # Каждый запрос занимает слот на 100 мс до первого токена и ещё 4 токена по 10 мс.
    base_url = start_server(ttft_ms=100, tokens_per_second=100, output_tokens=5, max_concurrency=1)
    async with httpx.AsyncClient(base_url=f"{base_url}/v1", timeout=5.0) as client:
        await warm_up(client)
        results = await asyncio.gather(
            *(stream_chat(client, model=MODEL, messages=MESSAGES, max_tokens=5) for _ in range(2))
        )

    ttfts = sorted(result.ttft_s for result in results if result.ttft_s is not None)
    assert len(ttfts) == 2
    assert 0.1 <= ttfts[0] < 0.13
    # Второй ждал, пока первый освободит слот (0,14 с), и только потом получил свои 100 мс.
    assert 0.24 <= ttfts[1] < 0.29


@pytest.mark.parametrize(
    ("settings", "error", "status"),
    [
        ({"rate_limit_rate": 1.0}, "http_429", 429),
        ({"error_rate": 1.0}, "http_500", 500),
        ({"served_model_name": "other-model"}, "http_404", 404),
    ],
)
async def test_http_errors_become_results(
    start_server: StartServer, settings: dict[str, Any], error: str, status: int
) -> None:
    base_url = start_server(**INSTANT, **settings)
    result = await measure(base_url)

    assert not result.ok
    assert result.error == error
    assert result.status_code == status
    assert result.ttft_s is None


async def test_rate_limit_reports_retry_after(start_server: StartServer) -> None:
    base_url = start_server(**INSTANT, rate_limit_rate=1.0)
    result = await measure(base_url)

    assert result.retry_after_s == 1.0


async def test_stream_without_done_is_truncated(start_server: StartServer) -> None:
    base_url = start_server(**INSTANT, output_tokens=10, truncate_rate=1.0)
    result = await measure(base_url, max_tokens=10)

    assert not result.ok
    assert result.error == "truncated"
    # Фейковая модель обрывает стрим на середине: пять токенов из десяти успевают прийти.
    assert len(result.chunk_times_s) == 5
    assert result.ttft_s is not None


async def test_slow_server_gives_timeout(start_server: StartServer) -> None:
    base_url = start_server(ttft_ms=500, tokens_per_second=100_000, output_tokens=1)
    result = await measure(base_url, http_timeout=0.1)

    assert not result.ok
    assert result.error == "timeout"
    assert result.latency_s < 0.5


async def test_closed_port_gives_connect_error() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    # Сокет закрыт, на порту никто не слушает.
    result = await measure(f"http://127.0.0.1:{port}")

    assert not result.ok
    assert result.error == "connect"
    assert result.status_code is None
