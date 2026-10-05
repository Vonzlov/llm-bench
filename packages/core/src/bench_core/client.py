"""Измерительный клиент: один потоковый запрос к модели и время каждого его шага.

Протокол — OpenAI Chat Completions со стримингом. На нём говорят vLLM, Ollama
(по адресу /v1), фейковая модель и большинство облачных API, поэтому клиент один
на всех. base_url задаётся вместе с /v1, как в OpenAI SDK: http://localhost:8000/v1.

Клиент ничего не повторяет и не бросает исключений на ответах сервера. Ошибка 429,
обрыв стрима или таймаут — тоже результат замера: генератор нагрузки должен видеть
их как есть и считать долю ошибок, а не падать.
"""

import json
import time
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import httpx

CHAT_PATH = "/chat/completions"


@dataclass
class Measurement:
    """Результат одного запроса. Все времена — в секундах от момента отправки."""

    ok: bool = False
    # Что пошло не так: http_<код>, truncated, timeout, connect, protocol, bad_chunk, stream_error.
    error: str | None = None
    # Момент отправки по time.perf_counter(). По нему генератор нагрузки кладёт запросы
    # разных воркеров на общую шкалу времени.
    sent_at: float = 0.0
    status_code: int | None = None
    # Время до первого непустого токена. None, если ни один токен не пришёл.
    ttft_s: float | None = None
    # Полное время: от отправки до конца стрима или до ошибки.
    latency_s: float = 0.0
    # Когда пришёл каждый чанк с текстом. Обычно один чанк — один токен.
    chunk_times_s: list[float] = field(default_factory=list)
    text: str = ""
    finish_reason: str | None = None
    # Счётчики токенов от сервера (usage). None, если сервер их не прислал.
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    # Сколько сервер просит подождать перед повтором: заголовок Retry-After у ответа 429.
    retry_after_s: float | None = None

    @property
    def itl_s(self) -> list[float]:
        """Интервалы между соседними чанками с текстом — ITL (inter-token latency)."""
        return [later - earlier for earlier, later in pairwise(self.chunk_times_s)]

    @property
    def output_tokens(self) -> int:
        """Сколько токенов сгенерировано: по usage, а если сервер его не прислал — по чанкам."""
        if self.completion_tokens is not None:
            return self.completion_tokens
        return len(self.chunk_times_s)

    @property
    def tpot_s(self) -> float | None:
        """Среднее время на один токен после первого (TPOT, time per output token)."""
        if self.ttft_s is None or self.output_tokens < 2:
            return None
        return (self.latency_s - self.ttft_s) / (self.output_tokens - 1)


async def stream_chat(
    client: httpx.AsyncClient,
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    extra_body: dict[str, Any] | None = None,
) -> Measurement:
    """Отправляет потоковый запрос и замеряет время до первого токена, между токенами и до конца.

    client создаёт вызывающий код: в нём base_url с /v1, ключ API и таймауты. Один клиент
    на много запросов — так соединения переиспользуются, и установка TCP-соединения
    не попадает в замер каждого запроса.
    """
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        # Просим сервер прислать счётчики токенов последним чанком.
        "stream_options": {"include_usage": True},
        **(extra_body or {}),
    }
    started = time.perf_counter()
    result = Measurement(sent_at=started)
    try:
        async with client.stream("POST", CHAT_PATH, json=body) as response:
            result.status_code = response.status_code
            if response.status_code == 200:
                await read_stream(response, result, started)
            else:
                await response.aread()
                result.error = f"http_{response.status_code}"
                result.retry_after_s = parse_retry_after(response.headers.get("retry-after"))
    except httpx.TimeoutException:
        result.error = "timeout"
    except httpx.ConnectError:
        result.error = "connect"
    except httpx.HTTPError:
        # Сервер оборвал соединение посреди ответа или нарушил протокол HTTP.
        result.error = "protocol"
    result.latency_s = time.perf_counter() - started
    result.ok = result.error is None
    return result


async def read_stream(response: httpx.Response, result: Measurement, started: float) -> None:
    """Разбирает SSE-стрим и записывает время каждого чанка с текстом.

    Нормальный стрим заканчивается строкой «data: [DONE]». Если её нет, ответ считается
    оборванным, даже когда соединение закрылось чисто.
    """
    parts: list[str] = []
    done = False
    try:
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue  # пустые строки между событиями и комментарии SSE
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                done = True
                break
            try:
                chunk: dict[str, Any] = json.loads(data)
            except json.JSONDecodeError:
                result.error = "bad_chunk"
                return
            # vLLM сообщает об ошибке посреди стрима отдельным событием с полем error.
            if "error" in chunk:
                result.error = "stream_error"
                return
            usage = chunk.get("usage")
            if usage:
                result.prompt_tokens = usage.get("prompt_tokens")
                result.completion_tokens = usage.get("completion_tokens")
            choices = chunk.get("choices") or []
            if not choices:
                continue  # чанк только с usage
            content = (choices[0].get("delta") or {}).get("content")
            if content:
                # Пустые чанки не считаем: vLLM первым шлёт чанк с ролью и пустым текстом.
                arrived = time.perf_counter() - started
                if result.ttft_s is None:
                    result.ttft_s = arrived
                result.chunk_times_s.append(arrived)
                parts.append(content)
            if choices[0].get("finish_reason"):
                result.finish_reason = choices[0]["finish_reason"]
    finally:
        result.text = "".join(parts)
    if not done:
        result.error = "truncated"


def parse_retry_after(value: str | None) -> float | None:
    """Retry-After бывает числом секунд или HTTP-датой. Дату не разбираем: API шлют секунды."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None
