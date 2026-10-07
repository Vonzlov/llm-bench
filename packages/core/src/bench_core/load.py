"""Генератор нагрузки: закрытая модель, ступени параллельности, прогрев и окно замера.

Закрытая модель: N воркеров, каждый шлёт следующий запрос сразу после ответа на
предыдущий, поэтому на сервере всегда ровно N запросов. Первые warmup_s секунд
ступени — прогрев, их не считаем. Следующие measure_s секунд — окно замера. После
окна новые запросы не отправляются, а начатые дожидаемся, чтобы следующая ступень
стартовала на пустом сервере.

Промпты идут по кругу из пула, но каждый запрос начинается со своего номера: так сервер
не возьмёт из кеша префилл промпта, который уже видел.

Что считается в окне:
- задержки (TTFT, ITL, полное время) и ошибки — по запросам, отправленным в окне,
  сколько бы они ни длились: после окна мы их дожидаемся. Запрос, начатый во время
  прогрева, в задержки не попадает;
- токены в секунду — по всем токенам, пришедшим в окне, от любых запросов. Так
  пропускная способность не теряет запросы, разрезанные краями окна;
- успешные запросы в секунду — по запросам, которые закончились в окне.

Почему задержки отбираются по моменту отправки, а не «начался и закончился в окне».
Длина запроса не зависит от того, когда он отправлен, поэтому такой отбор не смещает
перцентили. А в момент конца окна в полёте чаще оказываются длинные запросы (парадокс
инспекции): если требовать, чтобы запрос закончился в окне, выпадают как раз они,
и p95 занижается. Запрос длиннее окна, в том числе таймаут, не попал бы в итог никогда.
Так же отбирает запросы NVIDIA AIPerf: окно --benchmark-duration и ожидание начатых
--benchmark-grace-period.

Цена этого правила: последние N запросов доезжают, когда новые уже не отправляются и
нагрузка падает. У движков с батчингом их хвост от этого чуть быстрее, а у очереди,
где запросы обслуживаются по одному, не меняется ничего.
"""

import asyncio
import itertools
import time
import uuid
from collections import Counter
from dataclasses import dataclass

import httpx

from bench_core.client import Measurement, stream_chat
from bench_core.stats import percentile

# Ступени из методики: сколько запросов одновременно держим на сервере.
DEFAULT_LEVELS = (1, 2, 4, 8, 16, 32)
# Если сервер недоступен, воркер не долбит его в цикле, а ждёт перед следующей попыткой.
CONNECT_ERROR_PAUSE_S = 1.0


@dataclass
class LevelResult:
    """Итоги одной ступени — то, что ляжет в таблицу load_levels. Времена в секундах."""

    concurrency: int
    measure_s: float
    # Запросы, отправленные в окне замера, и сколько из них с ошибкой.
    n_requests: int
    n_errors: int
    errors: dict[str, int]
    ttft_p50_s: float | None
    ttft_p95_s: float | None
    itl_p50_s: float | None
    itl_p95_s: float | None
    latency_p50_s: float | None
    latency_p95_s: float | None
    tokens_per_s: float
    requests_per_s: float


async def run_level(
    client: httpx.AsyncClient,
    *,
    model: str,
    prompts: list[str],
    concurrency: int,
    max_tokens: int,
    warmup_s: float,
    measure_s: float,
) -> LevelResult:
    """Держит на сервере concurrency запросов в течение прогрева и окна замера."""
    results: list[Measurement] = []
    level_start = time.perf_counter()
    stop_at = level_start + warmup_s + measure_s
    # Общий счётчик на всех воркеров: промпты идут по кругу, без повторов подряд.
    prompt_numbers = itertools.count()
    # Метка ступени и номер запроса в начале делают каждый промпт уникальным, даже между
    # прогонами. Иначе кеш префиксов сервера (в vLLM он включён по умолчанию) узнал бы
    # повторный промпт и пропустил префилл, и время до первого токена вышло бы лучше,
    # чем у живых пользователей с разными запросами.
    level_tag = uuid.uuid4().hex[:8]

    async def worker() -> None:
        while time.perf_counter() < stop_at:
            number = next(prompt_numbers)
            prompt = prompts[number % len(prompts)]
            result = await stream_chat(
                client,
                model=model,
                messages=[{"role": "user", "content": f"Запрос {level_tag}-{number}.\n\n{prompt}"}],
                max_tokens=max_tokens,
            )
            results.append(result)
            # Сервер просит подождать (429 с Retry-After) — ждём, как сделал бы настоящий клиент.
            if result.retry_after_s:
                await asyncio.sleep(result.retry_after_s)
            elif result.error == "connect":
                await asyncio.sleep(CONNECT_ERROR_PAUSE_S)

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return summarize(
        results,
        level_start=level_start,
        concurrency=concurrency,
        warmup_s=warmup_s,
        measure_s=measure_s,
    )


def summarize(
    results: list[Measurement],
    *,
    level_start: float,
    concurrency: int,
    warmup_s: float,
    measure_s: float,
) -> LevelResult:
    """Сводит замеры ступени в итог по правилам окна из описания модуля."""
    window_start = level_start + warmup_s
    window_end = window_start + measure_s

    # Задержки и ошибки — по запросам, отправленным в окне, даже если они закончились позже.
    measured = [r for r in results if window_start <= r.sent_at < window_end]
    ok = [r for r in measured if r.ok]
    errors = Counter(r.error for r in measured if r.error is not None)

    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    itls = [itl for r in ok for itl in r.itl_s]
    latencies = [r.latency_s for r in ok]

    # Токены считаем по чанкам: у vLLM и Ollama один чанк стрима — один токен.
    tokens_in_window = 0
    for r in results:
        for arrived in r.chunk_times_s:
            if window_start <= r.sent_at + arrived < window_end:
                tokens_in_window += 1
    finished_ok = [
        r for r in results if r.ok and window_start <= r.sent_at + r.latency_s < window_end
    ]

    return LevelResult(
        concurrency=concurrency,
        measure_s=measure_s,
        n_requests=len(measured),
        n_errors=len(measured) - len(ok),
        errors=dict(errors),
        ttft_p50_s=percentile(ttfts, 50),
        ttft_p95_s=percentile(ttfts, 95),
        itl_p50_s=percentile(itls, 50),
        itl_p95_s=percentile(itls, 95),
        latency_p50_s=percentile(latencies, 50),
        latency_p95_s=percentile(latencies, 95),
        tokens_per_s=tokens_in_window / measure_s,
        requests_per_s=len(finished_ok) / measure_s,
    )
