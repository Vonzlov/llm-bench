"""Извлечение в JSON: модель находит в тексте сущности заданных типов.

Ответ — JSON-объект {"slots": [{"type": "тип", "value": "значение"}]}, где значение —
дословный фрагмент текста. Корень — объект, а не массив: такую схему принимают и vLLM,
и Ollama, и облачные API с ограниченной генерацией.

Задача идёт в двух режимах. В свободном модель пишет ответ сама, а мы его разбираем;
снисхождение одно — разрешаем обернуть JSON в блок кода Markdown. В ограниченном
(--constrained) движок получает JSON-схему и не даёт модели выйти за неё. Два прогона
рядом показывают, сколько валидности даёт схема и сколько она стоит в скорости.

Метрики:
- валидный JSON — доля ответов, которые разбираются как JSON;
- по схеме — доля ответов, которые ещё и подходят под схему: объект с одним полем
  slots, у каждого слота ровно поля type и value, type из списка, value — строка;
- F1 по слотам — micro-F1 по парам «тип — значение» на всём наборе. Значения сравниваем
  без учёта регистра, лишних пробелов и знаков препинания по краям. Ответ не по схеме
  не находит ни одного слота: все эталонные слоты примера становятся пропущенными.
"""

import json
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from bench_core.quality import ItemResult, Metric, Scored, Task, metric

SYSTEM_PROMPT = (
    "Найди в тексте все сущности перечисленных типов. Ответь только JSON-объектом вида "
    '{"slots": [{"type": "тип", "value": "значение"}]}, где значение — дословный фрагмент '
    'текста. Если сущностей нет, ответь {"slots": []}.\n\n'
    "Типы:\n"
)
# Блок кода Markdown вокруг JSON: ```json … ``` — модели часто так оборачивают ответ.
FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
# Что срезаем по краям значения перед сравнением.
EDGE_CHARS = " \t\r\n.,!?;:\"'«»"


def build_messages(item: dict[str, Any], labels: list[str]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT + "\n".join(labels)},
        {"role": "user", "content": f"Текст: {item['input']}"},
    ]


def schema(labels: list[str]) -> dict[str, Any]:
    """JSON-схема ответа для ограниченной генерации: тип слота — только из списка."""
    slot = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": labels},
            "value": {"type": "string"},
        },
        "required": ["type", "value"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"slots": {"type": "array", "items": slot}},
        "required": ["slots"],
        "additionalProperties": False,
    }


def parse_json(output: str) -> tuple[bool, Any]:
    """Разбирает ответ как JSON. Первое значение — получилось ли разобрать."""
    text = output.strip()
    fenced = FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        return True, json.loads(text)
    except json.JSONDecodeError:
        return False, None


def slots_from(data: Any, labels: list[str]) -> list[dict[str, str]] | None:
    """Слоты, если разобранный JSON подходит под схему; иначе None."""
    if not isinstance(data, dict) or set(data) != {"slots"} or not isinstance(data["slots"], list):
        return None
    allowed = set(labels)
    for slot in data["slots"]:
        if not isinstance(slot, dict) or set(slot) != {"type", "value"}:
            return None
        if slot["type"] not in allowed or not isinstance(slot["value"], str):
            return None
    return [{"type": slot["type"], "value": slot["value"]} for slot in data["slots"]]


def normalize(value: str) -> str:
    """Значение для сравнения: без регистра, лишних пробелов и знаков препинания по краям."""
    return " ".join(value.casefold().strip(EDGE_CHARS).split())


def counts(
    expected: list[dict[str, str]], predicted: list[dict[str, str]] | None
) -> tuple[int, int, int]:
    """Верно найденные, лишние и пропущенные слоты одного ответа."""
    expected_pairs = Counter((slot["type"], normalize(slot["value"])) for slot in expected)
    predicted_pairs = Counter((slot["type"], normalize(slot["value"])) for slot in predicted or [])
    found = sum((expected_pairs & predicted_pairs).values())
    extra = sum(predicted_pairs.values()) - found
    missed = sum(expected_pairs.values()) - found
    return found, extra, missed


def f1(found: int, extra: int, missed: int) -> float:
    """F1 = 2·TP / (2·TP + FP + FN). Нет слотов ни в эталоне, ни в ответе — это 1."""
    total = 2 * found + extra + missed
    return 2 * found / total if total else 1.0


def score(item: dict[str, Any], output: str, labels: list[str]) -> Scored:
    valid_json, data = parse_json(output)
    slots = slots_from(data, labels) if valid_json else None
    checks = {"valid_json": valid_json, "schema_ok": slots is not None}
    # Ответ не по схеме получает ноль, даже если в эталоне слотов нет.
    item_f1 = f1(*counts(item["expected"], slots)) if slots is not None else 0.0
    return Scored(slots, item_f1, checks)


def valid_json(results: Sequence[ItemResult]) -> float:
    return sum(result.checks.get("valid_json", False) for result in results) / len(results)


def schema_ok(results: Sequence[ItemResult]) -> float:
    return sum(result.checks.get("schema_ok", False) for result in results) / len(results)


def slot_f1(results: Sequence[ItemResult]) -> float:
    """micro-F1: верно найденные, лишние и пропущенные слоты суммируются по всем примерам."""
    found = extra = missed = 0
    for result in results:
        item_found, item_extra, item_missed = counts(result.expected, result.prediction)
        found += item_found
        extra += item_extra
        missed += item_missed
    return f1(found, extra, missed)


def summarize(results: list[ItemResult]) -> dict[str, Metric]:
    return {
        "valid_json": metric(results, valid_json),
        "schema_ok": metric(results, schema_ok),
        "slot_f1": metric(results, slot_f1),
    }


TASK = Task(
    name="extraction",
    max_tokens=128,
    build_messages=build_messages,
    score=score,
    summarize=summarize,
    schema=schema,
)
