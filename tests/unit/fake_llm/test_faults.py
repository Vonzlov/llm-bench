"""Сбои: ошибки 500 и 429, оборванный стрим и доли сбоев по настройкам."""

import httpx
import pytest
from pydantic import ValidationError

from fake_llm.settings import Settings
from tests.unit.fake_llm.helpers import CHAT_URL, StartServer, chat_request, content_of, read_events

INSTANT = {"ttft_ms": 0, "tokens_per_second": 100_000}


async def test_error_rate_one_always_returns_500(start_server: StartServer) -> None:
    base_url = start_server(**INSTANT, error_rate=1)
    async with httpx.AsyncClient(base_url=base_url) as client:
        response = await client.post(CHAT_URL, json=chat_request())

    assert response.status_code == 500
    assert response.json()["error"]["type"] == "InternalServerError"


async def test_rate_limit_returns_429_with_retry_after(start_server: StartServer) -> None:
    base_url = start_server(**INSTANT, rate_limit_rate=1)
    async with httpx.AsyncClient(base_url=base_url) as client:
        response = await client.post(CHAT_URL, json=chat_request())

    assert response.status_code == 429
    assert response.headers["retry-after"] == "1"


async def test_truncated_stream_has_no_finish_and_no_done(start_server: StartServer) -> None:
    base_url = start_server(**INSTANT, output_tokens=10, truncate_rate=1)
    async with (
        httpx.AsyncClient(base_url=base_url) as client,
        client.stream("POST", CHAT_URL, json=chat_request(stream=True)) as response,
    ):
        events = await read_events(response)

    assert "[DONE]" not in events
    assert not any(e["choices"] and e["choices"][0]["finish_reason"] for e in events)
    # Обрыв ровно на середине: пришла половина токенов.
    assert len(content_of(events).split()) == 5


async def test_error_shares_follow_settings(start_server: StartServer) -> None:
    base_url = start_server(**INSTANT, output_tokens=1, error_rate=0.2, rate_limit_rate=0.1, seed=7)
    async with httpx.AsyncClient(base_url=base_url) as client:
        statuses = [
            (await client.post(CHAT_URL, json=chat_request())).status_code for _ in range(500)
        ]

    assert statuses.count(500) / len(statuses) == pytest.approx(0.2, abs=0.05)
    assert statuses.count(429) / len(statuses) == pytest.approx(0.1, abs=0.04)


def test_error_rates_together_cannot_exceed_one() -> None:
    with pytest.raises(ValidationError):
        Settings(error_rate=0.7, rate_limit_rate=0.5)
