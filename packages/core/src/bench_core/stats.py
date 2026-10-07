"""Статистика для отчётов: перцентили и бутстреп-интервалы.

Бутстреп отвечает на вопрос «насколько метрика гуляла бы на другом наборе того же
размера». Из n примеров много раз выбираем n с возвращением — одни примеры попадают
дважды, другие ни разу, — считаем метрику на каждой такой выборке и смотрим, в каком
диапазоне лежат 95% значений. Формулы под конкретную метрику не нужны: так же
считается интервал и для точности, и для macro-F1.
"""

import math
import random
from collections.abc import Callable, Sequence

# Одно зерно на весь проект: повторный прогон даёт те же выборки и те же интервалы.
SEED = 2026
RESAMPLES = 1000


def percentile(values: Sequence[float], q: float) -> float | None:
    """Перцентиль с линейной интерполяцией между соседними значениями, как numpy.percentile."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bootstrap_ci[T](
    items: Sequence[T],
    statistic: Callable[[Sequence[T]], float],
    *,
    resamples: int = RESAMPLES,
    level: float = 0.95,
    seed: int = SEED,
) -> tuple[float, float]:
    """Перцентильный бутстреп-интервал: 2,5-й и 97,5-й перцентили статистики по выборкам."""
    if not items:
        raise ValueError("бутстреп по пустому набору не считается")
    rng = random.Random(seed)
    values = [statistic(rng.choices(items, k=len(items))) for _ in range(resamples)]
    tail = (1 - level) / 2 * 100
    low = percentile(values, tail)
    high = percentile(values, 100 - tail)
    if low is None or high is None:
        raise ValueError("нужна хотя бы одна повторная выборка")
    return low, high
