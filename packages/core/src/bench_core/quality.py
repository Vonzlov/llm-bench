"""Раннер качества: задаёт модели вопросы из набора и оценивает ответы.

Как спрашивать и как оценивать, решает задача (Task): у классификации, извлечения и
ответов по тексту свои промпты и метрики, а раннер у всех один. Запросы идут
параллельно, но не больше concurrency сразу. Температура 0 и фиксированный seed —
чтобы повторный прогон давал те же ответы.

Временные сбои — 429, ошибки 5xx, таймауты, обрывы стрима — повторяем до трёх раз.
Если не помогло, пример засчитывается как неверный ответ, а ошибка показывается в
итоге отдельно: модель, которая не отвечает, для пользователя хуже той, что ошибается.
"""

import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from bench_core.client import Measurement, stream_chat
from bench_core.stats import SEED, bootstrap_ci

ATTEMPTS = 3
# Пауза перед повтором растёт с номером попытки, если сервер сам не назвал её в Retry-After.
RETRY_PAUSE_S = 2.0


@dataclass
class ItemResult:
    """Ответ модели на один пример — то, что ляжет в таблицу item_results."""

    id: str
    expected: Any
    output: str
    # Разобранный ответ; None — модель ответила не по формату или запрос не удался.
    prediction: Any
    # Балл от 0 до 1: у классификации это 1 за верный ответ и 0 за неверный.
    score: float
    error: str | None
    finish_reason: str | None
    latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None


@dataclass
class Metric:
    """Значение метрики и её 95-процентный бутстреп-интервал."""

    value: float
    low: float
    high: float


@dataclass
class Task:
    """Как спрашивать модель и как оценивать ответ — своё у каждого типа задач."""

    name: str
    # Потолок длины ответа по умолчанию.
    max_tokens: int
    # Пример и варианты ответа → сообщения для модели.
    build_messages: Callable[[dict[str, Any], list[str]], list[dict[str, str]]]
    # Пример, текст ответа и варианты → разобранный ответ и балл.
    score: Callable[[dict[str, Any], str, list[str]], tuple[Any, float]]
    # Ответы на все примеры → метрики по названиям.
    summarize: Callable[[list[ItemResult]], dict[str, Metric]]


def metric(results: list[ItemResult], statistic: Callable[[Sequence[ItemResult]], float]) -> Metric:
    """Метрика по всем ответам и её бутстреп-интервал."""
    low, high = bootstrap_ci(results, statistic)
    return Metric(statistic(results), low, high)


def is_retryable(error: str | None) -> bool:
    """Сбои, которые проходят сами: перегрузка, ошибки сервера, сеть и обрывы."""
    if error is None:
        return False
    temporary = {"http_429", "timeout", "connect", "protocol", "truncated"}
    return error in temporary or error.startswith("http_5")


async def ask(
    client: httpx.AsyncClient, *, model: str, messages: list[dict[str, str]], max_tokens: int
) -> Measurement:
    """Один пример с повторами временных сбоев."""
    attempt = 1
    while True:
        result = await stream_chat(
            client,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            extra_body={"temperature": 0, "seed": SEED},
        )
        if not is_retryable(result.error) or attempt == ATTEMPTS:
            return result
        await asyncio.sleep(result.retry_after_s or RETRY_PAUSE_S * attempt)
        attempt += 1


async def run_task(
    client: httpx.AsyncClient,
    task: Task,
    *,
    model: str,
    items: list[dict[str, Any]],
    labels: list[str],
    concurrency: int,
    max_tokens: int,
    on_done: Callable[[int], None] | None = None,
) -> list[ItemResult]:
    """Прогоняет все примеры набора, не больше concurrency запросов одновременно.

    Ответы возвращаются в порядке примеров. on_done получает число готовых — для прогресса.
    """
    semaphore = asyncio.Semaphore(concurrency)
    done = 0

    async def one(item: dict[str, Any]) -> ItemResult:
        nonlocal done
        messages = task.build_messages(item, labels)
        async with semaphore:
            answer = await ask(client, model=model, messages=messages, max_tokens=max_tokens)
        prediction, score = (None, 0.0)
        if answer.ok:
            prediction, score = task.score(item, answer.text, labels)
        done += 1
        if on_done is not None:
            on_done(done)
        return ItemResult(
            id=item["id"],
            expected=item["expected"],
            output=answer.text,
            prediction=prediction,
            score=score,
            error=answer.error,
            finish_reason=answer.finish_reason,
            latency_s=answer.latency_s,
            prompt_tokens=answer.prompt_tokens,
            completion_tokens=answer.completion_tokens,
        )

    return list(await asyncio.gather(*(one(item) for item in items)))


def load_dataset(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Примеры набора и его строка из manifest.json, который лежит в той же папке."""
    manifest_path = path.parent / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry: dict[str, Any] | None = manifest.get(path.stem)
    if entry is None:
        raise ValueError(
            f"набора {path.stem} нет в {manifest_path}: соберите наборы через make datasets"
        )
    lines = path.read_text(encoding="utf-8").splitlines()
    items = [json.loads(line) for line in lines if line.strip()]
    return items, entry


def write_results(results: list[ItemResult], path: Path) -> None:
    """Ответы на все примеры в JSONL — по строке на пример."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(asdict(result), ensure_ascii=False) + "\n" for result in results)
    path.write_text(text, encoding="utf-8")
