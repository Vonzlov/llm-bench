"""Раннер качества: задаёт модели вопросы из набора и оценивает ответы.

Как спрашивать и как оценивать, решает задача (Task): у классификации, извлечения и
ответов по тексту свои промпты и метрики, а раннер у всех один. Запросы идут
параллельно, но не больше concurrency сразу. Температура 0 и фиксированный seed —
чтобы повторный прогон давал те же ответы.

Временные сбои — 429, ошибки 5xx, таймауты, обрывы стрима — повторяем до трёх раз.
Если не помогло, пример засчитывается как неверный ответ, а ошибка показывается в
итоге отдельно: модель, которая не отвечает, для пользователя хуже той, что ошибается.

У задач с ответом в JSON есть второй режим — ограниченная генерация (constrained): в
запрос уходит response_format с JSON-схемой, и движок не даёт модели выйти за неё.
"""

import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from bench_core.client import Measurement, stream_chat
from bench_core.stats import SEED, bootstrap_ci

ATTEMPTS = 3
# Пауза перед повтором растёт с номером попытки, если сервер сам не назвал её в Retry-After.
RETRY_PAUSE_S = 2.0


@dataclass
class Scored:
    """Оценка одного ответа: разобранный ответ, балл и проверки формата."""

    # Разобранный ответ; None — модель ответила не по формату.
    prediction: Any
    # Балл от 0 до 1: у классификации это 1 за верный ответ и 0 за неверный.
    score: float
    # Проверки формата, свои у каждой задачи: например, valid_json у извлечения.
    checks: dict[str, bool] = field(default_factory=dict)


@dataclass
class ItemResult:
    """Ответ модели на один пример — то, что ляжет в таблицу item_results."""

    id: str
    expected: Any
    output: str
    # Разобранный ответ; None — модель ответила не по формату или запрос не удался.
    prediction: Any
    score: float
    checks: dict[str, bool]
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
    # Варианты ответа → системный промпт с инструкцией.
    system_prompt: Callable[[list[str]], str]
    # Пример → сообщение пользователя с вопросом.
    user_message: Callable[[dict[str, Any]], str]
    # Эталонный ответ → текст, каким его должна написать модель. Нужен для примеров в промпте.
    render: Callable[[Any], str]
    # Пример, текст ответа и варианты → оценка ответа.
    score: Callable[[dict[str, Any], str, list[str]], Scored]
    # Ответы на все примеры → метрики по названиям.
    summarize: Callable[[list[ItemResult]], dict[str, Metric]]
    # Варианты ответа → JSON-схема для ограниченной генерации; None — такого режима нет.
    schema: Callable[[list[str]], dict[str, Any]] | None = None


def build_messages(
    task: Task, item: dict[str, Any], labels: list[str], examples: list[dict[str, Any]]
) -> list[dict[str, str]]:
    """Диалог, который видит модель: инструкция, примеры с ответами и сам вопрос.

    Примеры идут как прошлые реплики: вопрос пользователя и ответ ассистента ровно в том
    формате, который нужен (few-shot). Так модель видит не только инструкцию, но и то,
    как именно надо отвечать: где у значения границы, что пустой ответ тоже бывает.
    """
    messages = [{"role": "system", "content": task.system_prompt(labels)}]
    for example in examples:
        messages.append({"role": "user", "content": task.user_message(example)})
        messages.append({"role": "assistant", "content": task.render(example["expected"])})
    messages.append({"role": "user", "content": task.user_message(item)})
    return messages


def response_format(task: Task, labels: list[str]) -> dict[str, Any]:
    """Поле response_format в формате OpenAI: так схему понимают и vLLM, и Ollama."""
    if task.schema is None:
        raise ValueError(f"у задачи {task.name} нет режима с JSON-схемой")
    json_schema = {"name": task.name, "strict": True, "schema": task.schema(labels)}
    return {"type": "json_schema", "json_schema": json_schema}


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
    client: httpx.AsyncClient,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    extra_body: dict[str, Any],
) -> Measurement:
    """Один пример с повторами временных сбоев."""
    attempt = 1
    while True:
        result = await stream_chat(
            client,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            extra_body={"temperature": 0, "seed": SEED, **extra_body},
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
    examples: list[dict[str, Any]] | None = None,
    constrained: bool = False,
    on_done: Callable[[int], None] | None = None,
) -> list[ItemResult]:
    """Прогоняет все примеры набора, не больше concurrency запросов одновременно.

    examples — примеры с ответами для промпта, одинаковые для всех вопросов. Ответы
    возвращаются в порядке примеров набора. on_done получает число готовых — для прогресса.
    """
    semaphore = asyncio.Semaphore(concurrency)
    extra_body = {"response_format": response_format(task, labels)} if constrained else {}
    done = 0

    async def one(item: dict[str, Any]) -> ItemResult:
        nonlocal done
        messages = build_messages(task, item, labels, examples or [])
        async with semaphore:
            answer = await ask(
                client, model=model, messages=messages, max_tokens=max_tokens, extra_body=extra_body
            )
        scored = task.score(item, answer.text, labels) if answer.ok else Scored(None, 0.0)
        done += 1
        if on_done is not None:
            on_done(done)
        return ItemResult(
            id=item["id"],
            expected=item["expected"],
            output=answer.text,
            prediction=scored.prediction,
            score=scored.score,
            checks=scored.checks,
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
