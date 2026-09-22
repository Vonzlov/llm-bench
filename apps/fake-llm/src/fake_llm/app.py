"""HTTP-приложение фейковой модели.

Отвечает на /v1/chat/completions в формате OpenAI — целым ответом или стримом
SSE, как vLLM. Время строится по расписанию: первый токен через ttft_ms, каждый
следующий через 1 / tokens_per_second. Расписание отсчитывается от момента,
когда запрос получил свободный слот, поэтому ожидание в очереди попадает в TTFT.
Сроки токенов абсолютные, а не «спим после каждого токена», так что задержки
не накапливаются и скорость на длинном ответе остаётся точной.
"""

import asyncio
import json
import random
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from fake_llm import __version__
from fake_llm.schemas import ChatCompletionRequest
from fake_llm.settings import Settings
from fake_llm.tokens import make_tokens

# При max_concurrency = 0 очереди нет: берём заведомо недостижимое число слотов.
UNLIMITED_SLOTS = 1_000_000


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    rng = random.Random(settings.seed)
    slots = asyncio.Semaphore(settings.max_concurrency or UNLIMITED_SLOTS)
    started_at = int(time.time())

    app = FastAPI(title="fake-llm", version=__version__)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        model = {
            "id": settings.served_model_name,
            "object": "model",
            "created": started_at,
            "owned_by": "fake-llm",
        }
        return {"object": "list", "data": [model]}

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: ChatCompletionRequest) -> JSONResponse | StreamingResponse:
        if request.model != settings.served_model_name:
            message = f"The model '{request.model}' does not exist."
            return error_response(404, message, "NotFoundError")

        # Один бросок на запрос решает, будет ли ошибка, поэтому доли 500 и 429
        # точно соответствуют настройкам и не перекрываются.
        roll = rng.random()
        if roll < settings.error_rate:
            return error_response(500, "Simulated internal server error.", "InternalServerError")
        if roll < settings.error_rate + settings.rate_limit_rate:
            return error_response(
                429, "Simulated rate limit.", "RateLimitError", headers={"Retry-After": "1"}
            )

        limit = request.token_limit()
        n_tokens = min(limit, settings.output_tokens) if limit else settings.output_tokens
        finish_reason = "length" if limit is not None and limit < settings.output_tokens else "stop"
        tokens = make_tokens(n_tokens)
        prompt_tokens = request.prompt_tokens()
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": n_tokens,
            "total_tokens": prompt_tokens + n_tokens,
        }
        header = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "created": int(time.time()),
            "model": settings.served_model_name,
        }

        if request.stream:
            truncate = rng.random() < settings.truncate_rate
            events = stream_events(
                slots=slots,
                settings=settings,
                header=header,
                tokens=tokens,
                finish_reason=finish_reason,
                usage=usage if request.include_usage() else None,
                truncate=truncate,
            )
            return StreamingResponse(events, media_type="text/event-stream")

        async with slots:
            start = asyncio.get_running_loop().time()
            await sleep_until(token_deadline(start, n_tokens - 1, settings))

        body = {
            **header,
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(tokens)},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }
        return JSONResponse(body)

    return app


async def stream_events(
    *,
    slots: asyncio.Semaphore,
    settings: Settings,
    header: dict[str, Any],
    tokens: list[str],
    finish_reason: str,
    usage: dict[str, int] | None,
    truncate: bool,
) -> AsyncIterator[str]:
    """Выдаёт стрим в формате OpenAI: чанки с токенами, финальный чанк, usage и [DONE]."""
    async with slots:
        start = asyncio.get_running_loop().time()
        # Обрыв имитируем на середине ответа: без финального чанка и без [DONE].
        cut_at = len(tokens) // 2 if truncate else None

        for index, token in enumerate(tokens):
            await sleep_until(token_deadline(start, index, settings))
            if index == cut_at:
                return
            # Роль приходит вместе с первым токеном: первый чанк и есть первый токен.
            delta = {"role": "assistant", "content": token} if index == 0 else {"content": token}
            yield sse(chunk(header, delta))

        yield sse(chunk(header, {}, finish_reason))
        if usage is not None:
            yield sse({**header, "object": "chat.completion.chunk", "choices": [], "usage": usage})
        yield "data: [DONE]\n\n"


def token_deadline(start: float, index: int, settings: Settings) -> float:
    """Момент (по часам event loop), к которому должен быть готов токен с номером index."""
    return start + settings.ttft_ms / 1000 + index / settings.tokens_per_second


async def sleep_until(deadline: float) -> None:
    delay = deadline - asyncio.get_running_loop().time()
    if delay > 0:
        await asyncio.sleep(delay)


def chunk(
    header: dict[str, Any], delta: dict[str, str], finish_reason: str | None = None
) -> dict[str, Any]:
    return {
        **header,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def error_response(
    status: int, message: str, error_type: str, headers: dict[str, str] | None = None
) -> JSONResponse:
    """Ошибка в формате OpenAI: клиенты и SDK разбирают именно такое тело."""
    body = {"error": {"message": message, "type": error_type, "code": status}}
    return JSONResponse(status_code=status, content=body, headers=headers)
