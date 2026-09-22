"""Стрим SSE в формате OpenAI: чанки, финальный чанк, usage и [DONE]."""

from typing import Any

import httpx

from tests.unit.fake_llm.helpers import CHAT_URL, StartServer, chat_request, content_of, read_events

INSTANT = {"ttft_ms": 0, "tokens_per_second": 100_000}


async def stream(base_url: str, body: dict[str, Any]) -> list[Any]:
    async with (
        httpx.AsyncClient(base_url=base_url) as client,
        client.stream("POST", CHAT_URL, json=body) as response,
    ):
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        return await read_events(response)


async def test_stream_follows_openai_format(start_server: StartServer) -> None:
    base_url = start_server(**INSTANT, output_tokens=5)
    body = chat_request(stream=True, stream_options={"include_usage": True})
    events = await stream(base_url, body)

    assert events[-1] == "[DONE]"
    chunks = events[:-1]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert len(content_of(chunks).split()) == 5

    finish_reasons = [c["choices"][0]["finish_reason"] for c in chunks if c["choices"]]
    assert [reason for reason in finish_reasons if reason] == ["stop"]

    # Последний чанк перед [DONE] — usage без choices, как у OpenAI и vLLM.
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["completion_tokens"] == 5


async def test_usage_chunk_only_when_requested(start_server: StartServer) -> None:
    base_url = start_server(**INSTANT, output_tokens=5)
    events = await stream(base_url, chat_request(stream=True))

    assert not any("usage" in event for event in events if isinstance(event, dict))


async def test_stream_text_matches_regular_answer(start_server: StartServer) -> None:
    base_url = start_server(**INSTANT, output_tokens=12)
    events = await stream(base_url, chat_request(stream=True))
    async with httpx.AsyncClient(base_url=base_url) as client:
        response = await client.post(CHAT_URL, json=chat_request())

    assert content_of(events[:-1]) == response.json()["choices"][0]["message"]["content"]
