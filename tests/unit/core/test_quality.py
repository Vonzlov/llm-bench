"""Раннер качества: порядок ответов, повторы временных сбоев и ограничение параллельности.

Вместо модели — httpx.MockTransport: он отвечает на каждый пример заранее заданным текстом,
поэтому видно, что раннер делает с ответами, без настоящего сервера.
"""

import asyncio
import json
from collections import Counter
from typing import Any

import httpx
import pytest

from bench_core import classification, extraction, quality
from bench_core.quality import run_task
from bench_core.stats import SEED

LABELS = ["alarm_set", "iot_hue_lighton", "weather_query"]
ITEMS = [
    {"id": "1", "input": "разбуди меня в семь", "expected": "alarm_set"},
    {"id": "2", "input": "какая погода", "expected": "weather_query"},
    {"id": "3", "input": "включи свет", "expected": "iot_hue_lighton"},
    {"id": "4", "input": "сломанный", "expected": "alarm_set"},
]
ANSWERS = {
    "разбуди меня в семь": "alarm_set",
    "какая погода": "`Weather_Query`.",
    "включи свет": "Наверное, iot_hue_lighton",
}


def sse(text: str) -> bytes:
    chunk = {"choices": [{"delta": {"content": text}, "finish_reason": "stop"}]}
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\ndata: [DONE]\n\n".encode()


def phrase(body: dict[str, Any]) -> str:
    """Текст примера из сообщения пользователя «Текст: …\\nКатегория:»."""
    content: str = body["messages"][1]["content"]
    return content.removeprefix("Текст: ").removesuffix("\nКатегория:")


async def test_runner_scores_answers_and_retries_temporary_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(quality, "RETRY_PAUSE_S", 0.0)
    calls: Counter[str] = Counter()
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        text = phrase(body)
        calls[text] += 1
        if text == "сломанный":
            return httpx.Response(500)
        # Первая попытка падает, вторая проходит.
        if text == "какая погода" and calls[text] == 1:
            return httpx.Response(503)
        return httpx.Response(200, content=sse(ANSWERS[text]))

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(base_url="http://test/v1", transport=transport) as client:
        results = await run_task(
            client,
            classification.TASK,
            model="m",
            items=ITEMS,
            labels=LABELS,
            concurrency=2,
            max_tokens=8,
        )

    assert [r.id for r in results] == ["1", "2", "3", "4"]
    assert [r.score for r in results] == [1.0, 1.0, 0.0, 0.0]
    # Кавычки, точка и регистр сняты — ответ засчитан.
    assert results[1].prediction == "weather_query"
    assert calls["какая погода"] == 2
    # Сервер ответил, но не по формату: не ошибка запроса, а ответ вне списка.
    assert results[2].prediction is None
    assert results[2].error is None
    # Постоянная ошибка — три попытки, потом пример засчитан как неверный.
    assert results[3].error == "http_500"
    assert calls["сломанный"] == quality.ATTEMPTS
    assert all(body["temperature"] == 0 and body["seed"] == SEED for body in bodies)
    # Свободная генерация: схему движку не передаём.
    assert all("response_format" not in body for body in bodies)


async def test_constrained_mode_sends_json_schema() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=sse('{"slots": [{"type": "time", "value": "семь"}]}'))

    expected = [{"type": "time", "value": "семь"}]
    items = [{"id": "1", "input": "разбуди меня в семь", "expected": expected}]
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(base_url="http://test/v1", transport=transport) as client:
        results = await run_task(
            client,
            extraction.TASK,
            model="m",
            items=items,
            labels=["date", "time"],
            concurrency=1,
            max_tokens=64,
            constrained=True,
        )

    response_format = bodies[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["schema"] == extraction.schema(["date", "time"])
    assert results[0].score == 1.0
    assert results[0].checks == {"valid_json": True, "schema_ok": True}


def test_constrained_mode_needs_a_schema() -> None:
    with pytest.raises(ValueError, match="JSON-схемой"):
        quality.response_format(classification.TASK, LABELS)


async def test_runner_keeps_at_most_concurrency_requests_in_flight() -> None:
    in_flight = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return httpx.Response(200, content=sse("alarm_set"))

    items = [{"id": str(i), "input": f"фраза {i}", "expected": "alarm_set"} for i in range(10)]
    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(base_url="http://test/v1", transport=transport) as client:
        results = await run_task(
            client,
            classification.TASK,
            model="m",
            items=items,
            labels=LABELS,
            concurrency=3,
            max_tokens=8,
        )

    assert peak == 3
    assert all(result.score == 1.0 for result in results)
