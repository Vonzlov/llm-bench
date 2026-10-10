"""Извлечение в JSON: разбор ответа, проверка схемы и F1 по слотам, посчитанный руками."""

import json
from typing import Any

import pytest

from bench_core.extraction import (
    MAX_SLOTS,
    TASK,
    normalize,
    parse_json,
    render,
    schema,
    score,
    slot_f1,
    slots_from,
    system_prompt,
    user_message,
)
from bench_core.quality import ItemResult, summarize

LABELS = ["date", "place_name", "time"]
EXPECTED = [{"type": "time", "value": "пять утра"}, {"type": "date", "value": "этой неделе"}]


def result(expected: list[dict[str, str]], output: str) -> ItemResult:
    scored = score({"expected": expected}, output, LABELS)
    return ItemResult(
        id="1",
        expected=expected,
        output=output,
        prediction=scored.prediction,
        score=scored.score,
        checks=scored.checks,
        error=None,
        finish_reason="stop",
        latency_s=0.1,
        prompt_tokens=None,
        completion_tokens=None,
    )


@pytest.mark.parametrize(
    "output",
    ['{"slots": []}', '```json\n{"slots": []}\n```', '```\n{"slots": []}\n```', '  {"slots": []} '],
)
def test_parse_json_allows_markdown_code_block(output: str) -> None:
    assert parse_json(output) == (True, {"slots": []})


def test_parse_json_reports_failure_separately_from_null() -> None:
    assert parse_json("Вот слоты: time") == (False, None)
    assert parse_json("null") == (True, None)


def test_slots_follow_the_schema() -> None:
    slots = [{"type": "time", "value": "пять утра"}]
    assert slots_from({"slots": slots}, LABELS) == slots


@pytest.mark.parametrize(
    "data",
    [
        [{"type": "time", "value": "пять утра"}],
        {"result": []},
        {"slots": [{"type": "alarm", "value": "пять утра"}]},
        {"slots": [{"type": "time", "value": 5}]},
        {"slots": [{"type": "time", "value": "пять утра", "confidence": 0.9}]},
        {"slots": [], "comment": "ничего не нашёл"},
        # Пустое значение: так маленькая модель перечисляет типы, которых в тексте нет.
        {"slots": [{"type": "time", "value": ""}]},
        # Больше MAX_SLOTS слотов.
        {"slots": [{"type": "time", "value": f"{hour} утра"} for hour in range(11)]},
    ],
)
def test_slots_off_schema_are_rejected(data: Any) -> None:
    assert slots_from(data, LABELS) is None


def test_values_compare_without_case_spaces_edge_punctuation_and_yo() -> None:
    assert normalize("  Пять   Утра. ") == "пять утра"
    assert normalize("«этой неделе»") == "этой неделе"
    assert normalize("пятизвёздочных") == normalize("ПЯТИЗВЕЗДОЧНЫХ")


def test_score_counts_pairs_of_type_and_value() -> None:
    """Время найдено, дата найдена с неверным типом: TP = 1, FP = 1, FN = 1, F1 = 2/4."""
    output = (
        '{"slots": [{"type": "time", "value": "Пять утра"},'
        ' {"type": "place_name", "value": "этой неделе"}]}'
    )

    scored = score({"expected": EXPECTED}, output, LABELS)

    assert scored.score == 0.5
    assert scored.checks == {"valid_json": True, "schema_ok": True}


def test_off_format_answer_scores_zero_even_without_expected_slots() -> None:
    scored = score({"expected": []}, "слотов нет", LABELS)

    assert scored.prediction is None
    assert scored.score == 0.0
    assert scored.checks == {"valid_json": False, "schema_ok": False}
    assert score({"expected": []}, '{"slots": []}', LABELS).score == 1.0


def test_micro_f1_sums_slots_over_all_examples() -> None:
    """Пример 1: оба слота верно. Пример 2: ответ не по схеме — два слота пропущены.

    TP = 2, FP = 0, FN = 2, micro-F1 = 2·2 / (2·2 + 0 + 2) = 2/3.
    """
    all_right = json.dumps({"slots": EXPECTED}, ensure_ascii=False)
    results = [result(EXPECTED, all_right), result(EXPECTED, "не знаю")]

    assert slot_f1(results) == pytest.approx(2 / 3)
    metrics = summarize(TASK, results)
    assert metrics["valid_json"].value == 0.5
    assert metrics["schema_ok"].value == 0.5
    assert metrics["slot_f1"].value == pytest.approx(2 / 3)


def test_schema_allows_only_listed_types_and_bounds_the_answer() -> None:
    slots = schema(LABELS)["properties"]["slots"]
    slot = slots["items"]

    assert slot["properties"]["type"]["enum"] == LABELS
    assert slot["properties"]["value"]["minLength"] == 1
    assert slot["additionalProperties"] is False
    assert slot["required"] == ["type", "value"]
    assert slots["maxItems"] == MAX_SLOTS


def test_prompt_lists_types_and_text() -> None:
    prompt = system_prompt(LABELS)

    assert prompt.endswith("Типы:\ndate\nplace_name\ntime")
    assert '{"slots": []}' in prompt
    assert "без предлогов" in prompt
    assert user_message({"input": "разбуди меня в пять утра"}) == "Текст: разбуди меня в пять утра"


def test_answer_in_examples_is_exactly_what_we_parse() -> None:
    """Пример в промпте показывает ровно тот формат, который разбор принимает за верный."""
    text = render(EXPECTED)

    assert parse_json(text) == (True, {"slots": EXPECTED})
    assert score({"expected": EXPECTED}, text, LABELS).score == 1.0
