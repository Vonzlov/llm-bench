"""Классификация: промпт, строгий разбор ответа и метрики на примерах, посчитанных руками."""

import pytest

from bench_core.classification import (
    accuracy,
    build_messages,
    macro_f1,
    out_of_list,
    parse,
    summarize,
)
from bench_core.quality import ItemResult

LABELS = ["alarm_set", "iot_hue_lighton", "weather_query"]


def result(expected: str, prediction: str | None, error: str | None = None) -> ItemResult:
    return ItemResult(
        id="1",
        expected=expected,
        output="",
        prediction=prediction,
        score=float(prediction == expected),
        checks={},
        error=error,
        finish_reason="stop",
        latency_s=0.1,
        prompt_tokens=None,
        completion_tokens=None,
    )


@pytest.mark.parametrize(
    "output", ["alarm_set", "  `Alarm_Set`.\n", "«alarm_set»", "**alarm_set**", '"alarm_set"']
)
def test_parse_strips_quotes_markdown_dot_and_case(output: str) -> None:
    assert parse(output, LABELS) == "alarm_set"


@pytest.mark.parametrize(
    "output", ["Категория: alarm_set", "alarm", "alarm_set, weather_query", ""]
)
def test_parse_rejects_answers_not_from_the_list(output: str) -> None:
    assert parse(output, LABELS) is None


def test_prompt_lists_every_label_on_its_own_line() -> None:
    system, user = build_messages({"input": "разбуди меня в семь"}, LABELS)

    assert system["role"] == "system"
    assert system["content"].endswith("alarm_set\niot_hue_lighton\nweather_query")
    assert user == {"role": "user", "content": "Текст: разбуди меня в семь\nКатегория:"}


def test_metrics_match_hand_computed_example() -> None:
    """Эталоны a, a, b, b; ответы a, b, b и один вне списка.

    F1(a) = 2·1 / (2·1 + 0 + 1) = 2/3, F1(b) = 2·1 / (2·1 + 1 + 1) = 1/2,
    macro-F1 = (2/3 + 1/2) / 2 = 7/12 — столько же даёт sklearn с average="macro".
    """
    results = [result("a", "a"), result("a", "b"), result("b", "b"), result("b", None)]

    assert accuracy(results) == 0.5
    assert macro_f1(results) == pytest.approx(7 / 12)
    assert out_of_list(results) == 0.25


def test_failed_request_is_wrong_but_not_out_of_list() -> None:
    results = [result("a", None, error="timeout"), result("a", "a")]

    assert accuracy(results) == 0.5
    assert out_of_list(results) == 0.0


def test_summary_gives_value_and_interval_for_each_metric() -> None:
    results = [result("a", "a")] * 30 + [result("a", "b")] * 10

    summary = summarize(results)

    assert set(summary) == {"accuracy", "macro_f1", "out_of_list"}
    assert summary["accuracy"].value == 0.75
    assert summary["accuracy"].low < 0.75 < summary["accuracy"].high
    assert summary["out_of_list"].value == summary["out_of_list"].high == 0.0
