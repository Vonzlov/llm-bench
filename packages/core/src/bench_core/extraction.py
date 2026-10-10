"""Извлечение в JSON: модель находит в тексте сущности заданных типов.

Ответ — JSON-объект {"slots": [{"type": "тип", "value": "значение"}]}, где значение —
дословный фрагмент текста. Корень — объект, а не массив: такую схему принимают и vLLM,
и Ollama, и облачные API с ограниченной генерацией.

Границы значения промпт задаёт явно: только сама сущность, без предлогов и соседних слов.
Так размечен эталон MASSIVE: предлог стоит перед значением у трети слотов, а внутрь
значения попадает только у 3%. Без этого правила модель не может угадать, где у
значения границы, и теряет верные ответы на «на этой неделе» вместо «этой неделе».

Задача идёт в двух режимах. В свободном модель пишет ответ сама, а мы его разбираем;
снисхождение одно — разрешаем обернуть JSON в блок кода Markdown. В ограниченном
(--constrained) движок получает JSON-схему и не даёт модели выйти за неё. Два прогона
рядом показывают, сколько валидности даёт схема и сколько она стоит в скорости.

Метрики:
- валидный JSON — доля ответов, которые разбираются как JSON;
- по схеме — доля ответов, которые ещё и подходят под схему: объект с одним полем
  slots, в нём не больше MAX_SLOTS слотов, у каждого ровно поля type и value, type из
  списка, value — непустая строка. Свободный режим проверяем по той же схеме;
- F1 по слотам — micro-F1 по парам «тип — значение» на всём наборе. Значения сравниваем
  без учёта регистра, лишних пробелов и знаков препинания по краям, е и ё не различаем.
  Ответ не по схеме не находит ни одного слота: все эталонные слоты примера становятся
  пропущенными.
"""

import json
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from bench_core.quality import ItemResult, Scored, Task

SYSTEM_PROMPT = (
    "Найди в тексте сущности перечисленных типов. Включай в ответ только сущности, которые "
    "действительно есть в тексте; типы, которых в тексте нет, не перечисляй. Значение — "
    "дословный фрагмент текста, как можно короче: только сама сущность, без предлогов и "
    "соседних слов. Ответь только JSON-объектом вида "
    '{"slots": [{"type": "тип", "value": "значение"}]}. Если подходящих сущностей нет, '
    'ответь {"slots": []}.\n\n'
    "Типы:\n"
)
# Потолок числа слотов в ответе. В эталоне MASSIVE их не больше пяти на фразу; без потолка
# маленькая модель в режиме со схемой перечисляет все типы подряд, пока не упрётся в лимит
# токенов, и JSON обрывается на середине.
MAX_SLOTS = 10
# Блок кода Markdown вокруг JSON: ```json … ``` — модели часто так оборачивают ответ.
FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
# Что срезаем по краям значения перед сравнением.
EDGE_CHARS = " \t\r\n.,!?;:\"'«»"


def system_prompt(labels: list[str]) -> str:
    return SYSTEM_PROMPT + "\n".join(labels)


def user_message(item: dict[str, Any]) -> str:
    return f"Текст: {item['input']}"


def render(expected: Any) -> str:
    """Ответ в примерах — тот же JSON, который мы ждём от модели."""
    return json.dumps({"slots": expected}, ensure_ascii=False)


def schema(labels: list[str]) -> dict[str, Any]:
    """JSON-схема ответа: тип слота — из списка, значение непустое, слотов не больше MAX_SLOTS."""
    slot = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": labels},
            "value": {"type": "string", "minLength": 1},
        },
        "required": ["type", "value"],
        "additionalProperties": False,
    }
    slots = {"type": "array", "items": slot, "maxItems": MAX_SLOTS}
    return {
        "type": "object",
        "properties": {"slots": slots},
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
    """Слоты, если разобранный JSON подходит под схему из schema(); иначе None."""
    if not isinstance(data, dict) or set(data) != {"slots"} or not isinstance(data["slots"], list):
        return None
    if len(data["slots"]) > MAX_SLOTS:
        return None
    allowed = set(labels)
    for slot in data["slots"]:
        if not isinstance(slot, dict) or set(slot) != {"type", "value"}:
            return None
        if slot["type"] not in allowed or not isinstance(slot["value"], str):
            return None
        if not slot["value"]:
            return None
    return [{"type": slot["type"], "value": slot["value"]} for slot in data["slots"]]


def normalize(value: str) -> str:
    """Значение для сравнения: без регистра, лишних пробелов и знаков по краям, ё как е."""
    return " ".join(value.casefold().replace("ё", "е").strip(EDGE_CHARS).split())


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


TASK = Task(
    name="extraction",
    max_tokens=128,
    system_prompt=system_prompt,
    user_message=user_message,
    render=render,
    score=score,
    metrics={"valid_json": valid_json, "schema_ok": schema_ok, "slot_f1": slot_f1},
    schema=schema,
)
