"""Базовое поведение API: здоровье, список моделей и обычный (не потоковый) ответ."""

import httpx

from tests.unit.fake_llm.helpers import CHAT_URL, StartServer, chat_request

# Настройки для тестов, где время не проверяется: отвечаем мгновенно.
INSTANT = {"ttft_ms": 0, "tokens_per_second": 100_000}


async def test_health(start_server: StartServer) -> None:
    base_url = start_server(**INSTANT)
    async with httpx.AsyncClient(base_url=base_url) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_models_endpoint_lists_served_model(start_server: StartServer) -> None:
    base_url = start_server(**INSTANT, served_model_name="qwen-fake")
    async with httpx.AsyncClient(base_url=base_url) as client:
        response = await client.get("/v1/models")

    assert [model["id"] for model in response.json()["data"]] == ["qwen-fake"]


async def test_max_tokens_cuts_the_answer(start_server: StartServer) -> None:
    base_url = start_server(**INSTANT, output_tokens=50)
    async with httpx.AsyncClient(base_url=base_url) as client:
        response = await client.post(CHAT_URL, json=chat_request(max_tokens=10))

    body = response.json()
    choice = body["choices"][0]
    assert response.status_code == 200
    assert body["object"] == "chat.completion"
    assert choice["finish_reason"] == "length"
    assert len(choice["message"]["content"].split()) == 10
    # «Привет, как дела?» — три слова в промпте.
    assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 10, "total_tokens": 13}


async def test_answer_stops_on_its_own_without_limit(start_server: StartServer) -> None:
    base_url = start_server(**INSTANT, output_tokens=5)
    async with httpx.AsyncClient(base_url=base_url) as client:
        response = await client.post(CHAT_URL, json=chat_request())

    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert len(choice["message"]["content"].split()) == 5


async def test_max_completion_tokens_is_respected(start_server: StartServer) -> None:
    base_url = start_server(**INSTANT, output_tokens=50)
    async with httpx.AsyncClient(base_url=base_url) as client:
        response = await client.post(CHAT_URL, json=chat_request(max_completion_tokens=3))

    assert response.json()["usage"]["completion_tokens"] == 3


async def test_unknown_model_returns_404_in_openai_format(start_server: StartServer) -> None:
    base_url = start_server(**INSTANT)
    async with httpx.AsyncClient(base_url=base_url) as client:
        response = await client.post(CHAT_URL, json={**chat_request(), "model": "gpt-9"})

    assert response.status_code == 404
    assert response.json()["error"]["type"] == "NotFoundError"


async def test_unknown_openai_parameters_are_ignored(start_server: StartServer) -> None:
    base_url = start_server(**INSTANT)
    body = chat_request(temperature=0, seed=42, response_format={"type": "json_object"})
    async with httpx.AsyncClient(base_url=base_url) as client:
        response = await client.post(CHAT_URL, json=body)

    assert response.status_code == 200
