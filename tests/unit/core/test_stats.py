"""Статистика: перцентиль как в numpy, бутстреп-интервал и парный бутстреп для разницы."""

from collections.abc import Sequence

import pytest

from bench_core.stats import bootstrap_ci, paired_bootstrap_ci, percentile


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def test_percentile_interpolates_like_numpy() -> None:
    assert percentile([], 50) is None
    assert percentile([5.0], 95) == 5.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    # numpy.percentile(range(1, 11), 95) == 9.55
    assert percentile([float(x) for x in range(1, 11)], 95) == pytest.approx(9.55)


def test_bootstrap_interval_is_repeatable_and_has_textbook_width() -> None:
    """400 бросков монеты, половина орлов. Теория даёт интервал 0,5 ± 1,96 × 0,025."""
    flips = [1.0, 0.0] * 200

    low, high = bootstrap_ci(flips, mean)

    assert bootstrap_ci(flips, mean) == (low, high)
    assert low == pytest.approx(0.5 - 0.049, abs=0.01)
    assert high == pytest.approx(0.5 + 0.049, abs=0.01)


def test_bootstrap_of_constant_is_a_point() -> None:
    assert bootstrap_ci([1.0] * 10, mean) == (1.0, 1.0)


def test_bootstrap_needs_data() -> None:
    with pytest.raises(ValueError, match="пустому"):
        bootstrap_ci([], mean)


def test_paired_bootstrap_sees_consistent_gain_that_separate_intervals_miss() -> None:
    """Второй прогон исправил 10 ошибок из 40 и не сломал ни одного верного ответа.

    Раздельные интервалы точностей 0,6 и 0,7 перекрываются. Парный интервал разницы её
    видит: все изменения в одну сторону. Теория для него: 0,1 ± 1,96 × √(0,1 · 0,9 / 100),
    то есть примерно от 0,04 до 0,16.
    """
    first = [1.0] * 60 + [0.0] * 40
    second = [1.0] * 70 + [0.0] * 30

    low, high = paired_bootstrap_ci(first, second, mean)

    assert low == pytest.approx(0.04, abs=0.02)
    assert high == pytest.approx(0.16, abs=0.02)
    _, first_high = bootstrap_ci(first, mean)
    second_low, _ = bootstrap_ci(second, mean)
    assert second_low < first_high


def test_paired_bootstrap_of_identical_runs_is_zero() -> None:
    answers = [1.0, 0.0] * 10
    assert paired_bootstrap_ci(answers, answers, mean) == (0.0, 0.0)


def test_paired_bootstrap_needs_runs_of_equal_length() -> None:
    with pytest.raises(ValueError, match="одинаковое число"):
        paired_bootstrap_ci([1.0], [1.0, 0.0], mean)
    with pytest.raises(ValueError, match="пустому"):
        paired_bootstrap_ci([], [], mean)
