"""Статистика: перцентиль как в numpy и бутстреп-интервал."""

from collections.abc import Sequence

import pytest

from bench_core.stats import bootstrap_ci, percentile


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
