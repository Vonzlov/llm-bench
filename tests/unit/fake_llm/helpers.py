"""Общие помощники для тестов фейковой модели."""

import json
from collections.abc import Callable
from typing import Any

import httpx

MODEL = "fake-llm"
CHAT_URL = "/v1/chat/completions"

# Фикстура start_server: принимает настройки как именованные аргументы и возвращает адрес сервера.
StartServer = Callable[..., str]


def chat_request(**extra: Any) -> dict[str, Any]:
    """Тело запроса к фейковой модели. Промпт — три слова, это важно для проверки usage."""
    return {"model": MODEL, "messages": [{"role": "user", "content": "Привет, как дела?"}], **extra}


async def read_events(response: httpx.Response) -> list[Any]:
    """Разбирает SSE-стрим в список: словари-чанки и строка «[DONE]» в конце."""
    events: list[Any] = []
    async for line in response.aiter_lines():
        if not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ")
        events.append(data if data == "[DONE]" else json.loads(data))
    return events


def content_of(chunks: list[dict[str, Any]]) -> str:
    """Склеивает текст ответа из чанков стрима."""
    return "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
