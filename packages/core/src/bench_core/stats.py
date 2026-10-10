"""Статистика для отчётов: перцентили и бутстреп-интервалы.

Бутстреп отвечает на вопрос «насколько метрика гуляла бы на другом наборе того же
размера». Из n примеров много раз выбираем n с возвращением — одни примеры попадают
дважды, другие ни разу, — считаем метрику на каждой такой выборке и смотрим, в каком
диапазоне лежат 95% значений. Формулы под конкретную метрику не нужны: так же
считается интервал и для точности, и для macro-F1. Парный бутстреп так же отвечает на
вопрос «лучше ли второй прогон первого», когда оба сделаны на одних и тех же примерах.
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


def paired_bootstrap_ci[T](
    first: Sequence[T],
    second: Sequence[T],
    statistic: Callable[[Sequence[T]], float],
    *,
    resamples: int = RESAMPLES,
    level: float = 0.95,
    seed: int = SEED,
) -> tuple[float, float]:
    """Интервал разницы statistic(second) − statistic(first) двух прогонов на одних примерах.

    first[i] и second[i] — ответы на один и тот же пример. В каждой повторной выборке оба
    прогона берут одни и те же номера примеров, поэтому разница в трудности примеров, общая
    для обоих прогонов, в разницу не попадает. Интервал выходит уже, чем если сравнивать
    два раздельных интервала, и видит небольшой, но устойчивый выигрыш.
    """
    if len(first) != len(second):
        raise ValueError("в парном бутстрепе у прогонов должно быть одинаковое число примеров")
    if not first:
        raise ValueError("бутстреп по пустому набору не считается")
    rng = random.Random(seed)
    positions = range(len(first))
    differences = []
    for _ in range(resamples):
        sample = rng.choices(positions, k=len(first))
        differences.append(
            statistic([second[i] for i in sample]) - statistic([first[i] for i in sample])
        )
    tail = (1 - level) / 2 * 100
    low = percentile(differences, tail)
    high = percentile(differences, 100 - tail)
    if low is None or high is None:
        raise ValueError("нужна хотя бы одна повторная выборка")
    return low, high
