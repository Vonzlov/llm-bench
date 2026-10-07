"""Генератор нагрузки: правила окна замера и поведение на двух режимах фейковой модели.

Два режима — это главный эксперимент проекта в миниатюре. Без ограничения параллельности
фейковая модель ведёт себя как идеальный батчинг vLLM: пропускная способность растёт с N,
время до первого токена стоит на месте. С одним слотом — как Ollama при
OLLAMA_NUM_PARALLEL=1: пропускная способность стоит, время до первого токена растёт.
Генератор обязан показать обе картины.

Времена в тестах нарочно крупные: сотня миллисекунд до первого токена и 20 мс между
токенами. Сервер и клиент живут в одном процессе, и на загруженной машине или в WSL2
поток может просыпаться с опозданием на несколько миллисекунд. При миллисекундных
задержках это опоздание сравнимо с тем, что мы меряем, и тесты начинают врать.
Поэтому проверяется форма поведения — что растёт, а что стоит, — а не точные числа:
точность линейки проверяют тесты клиента в test_client.py.
"""

import httpx
import pytest

from bench_core.client import Measurement
from bench_core.load import LevelResult, percentile, run_level, summarize
from tests.unit.fake_llm.helpers import MODEL, StartServer

# Каждый ответ: 100 мс до первого токена и ещё 2 токена по 20 мс — 140 мс на запрос.
SERVER = {"ttft_ms": 100, "tokens_per_second": 50, "output_tokens": 3}
PROMPTS = ["раз", "два", "три"]


async def load(
    base_url: str, concurrency: int, warmup_s: float = 0.3, measure_s: float = 1.5
) -> LevelResult:
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=f"{base_url}/v1", timeout=5.0, limits=limits) as client:
        return await run_level(
            client,
            model=MODEL,
            prompts=PROMPTS,
            concurrency=concurrency,
            max_tokens=3,
            warmup_s=warmup_s,
            measure_s=measure_s,
        )


async def test_without_limit_throughput_grows_and_ttft_stays(start_server: StartServer) -> None:
    base_url = start_server(**SERVER)
    one = await load(base_url, concurrency=1)
    four = await load(base_url, concurrency=4)

    assert one.n_errors == four.n_errors == 0
    assert one.ttft_p50_s is not None and four.ttft_p50_s is not None
    # Быстрее заданного сервер ответить не может, а запас сверху — на опоздания потоков.
    assert 0.1 <= one.ttft_p50_s < 0.2
    # Время до первого токена почти не растёт: запросы не ждут друг друга...
    assert four.ttft_p50_s < one.ttft_p50_s + 0.08
    # ...а токенов в секунду становится в разы больше: в идеале вчетверо.
    assert four.tokens_per_s > 2.5 * one.tokens_per_s


async def test_single_slot_queues_requests_like_ollama(start_server: StartServer) -> None:
    base_url = start_server(**SERVER, max_concurrency=1)
    one = await load(base_url, concurrency=1)
    four = await load(base_url, concurrency=4)

    assert one.ttft_p50_s is not None and four.ttft_p50_s is not None
    # Каждый запрос ждёт, пока обслужат трёх соседей: около 3 × 140 + 100 мс вместо 100.
    assert four.ttft_p50_s > 3 * one.ttft_p50_s
    assert four.ttft_p50_s >= 0.4
    # А токенов в секунду больше не становится: сервер всё равно работает по одному.
    assert four.tokens_per_s < 1.5 * one.tokens_per_s


@pytest.mark.parametrize("max_concurrency", [0, 1])
async def test_closed_model_keeps_n_requests_on_server(
    start_server: StartServer, max_concurrency: int
) -> None:
    """Закон Литтла: запросов в секунду × время запроса = сколько их одновременно на сервере."""
    base_url = start_server(**SERVER, max_concurrency=max_concurrency)
    result = await load(base_url, concurrency=4)

    assert result.latency_p50_s is not None
    assert result.requests_per_s * result.latency_p50_s == pytest.approx(4, rel=0.25)


async def test_errors_are_counted_not_raised(start_server: StartServer) -> None:
    """Ошибки — это результат замера: их считают, а воркеры продолжают работать.

    Точную долю ошибок фейковой модели проверяет её собственный тест на 500 запросах.
    Здесь окно короткое, и доля в нём честно гуляет, поэтому проверяем другое: с этим
    зерном в любых 15 подряд идущих запросах есть и ошибки, и успехи.
    """
    base_url = start_server(
        ttft_ms=0, tokens_per_second=100_000, output_tokens=1, error_rate=0.5, seed=7
    )
    result = await load(base_url, concurrency=2, warmup_s=0.1, measure_s=1.0)

    assert result.n_requests >= 15
    assert set(result.errors) == {"http_500"}
    assert 0 < result.n_errors < result.n_requests


def test_window_rules() -> None:
    """Ступень началась в момент 100, прогрев 1 с, окно замера 2 с — то есть [101, 103)."""
    # Отправлен в прогреве: в задержки не идёт. Но его токен в 101,4 входит в пропускную
    # способность, а закончился он в окне — значит, считается и в запросах в секунду.
    warmup = Measurement(
        ok=True, sent_at=100.5, latency_s=1.0, ttft_s=0.1, chunk_times_s=[0.1, 0.9]
    )
    # Целиком в окне: идёт и в задержки, и в пропускную способность.
    inside = Measurement(
        ok=True, sent_at=101.2, latency_s=0.5, ttft_s=0.2, chunk_times_s=[0.2, 0.4]
    )
    # Ошибка в окне: считается в запросах и ошибках, но не в задержках.
    failed = Measurement(ok=False, error="http_500", sent_at=101.5, latency_s=0.01)
    # Отправлен в окне и длится дольше самого окна: в задержки идёт.
    # Из его токенов в окно попадает только первый, в 102,5.
    long = Measurement(ok=True, sent_at=102.0, latency_s=4.0, ttft_s=0.5, chunk_times_s=[0.5, 3.5])
    # Таймаут тоже длиннее окна, и в ошибки он попадает.
    timeout = Measurement(ok=False, error="timeout", sent_at=102.5, latency_s=300.0)
    # Отправлен после окна: не считается нигде.
    after = Measurement(ok=True, sent_at=103.05, latency_s=0.2, ttft_s=0.1, chunk_times_s=[0.1])

    result = summarize(
        [warmup, inside, failed, long, timeout, after],
        level_start=100.0,
        concurrency=2,
        warmup_s=1.0,
        measure_s=2.0,
    )

    assert result.n_requests == 4
    assert result.n_errors == 2
    assert result.errors == {"http_500": 1, "timeout": 1}
    # Задержки — по inside и long.
    assert result.ttft_p50_s == pytest.approx(0.35)
    assert result.latency_p50_s == pytest.approx(2.25)
    # Токены в окне: один от warmup, два от inside, один от long — 4 за 2 секунды.
    assert result.tokens_per_s == 2.0
    # Успешно закончились в окне warmup и inside — 2 за 2 секунды.
    assert result.requests_per_s == 1.0


async def test_request_sent_in_window_counts_even_if_it_ends_later(
    start_server: StartServer,
) -> None:
    """Каждый ответ идёт 0,3 с, окно замера — [0,1; 0,5) от начала ступени.

    Первый запрос воркера уходит в прогреве. Второй уходит около 0,3 с, внутри окна,
    а заканчивается около 0,6 с, уже после окна. Третьего нет: окно к тому времени
    кончилось. По правилу «начался и закончился в окне» второй запрос выпадал бы,
    и ступень показала бы ноль запросов.
    """
    base_url = start_server(ttft_ms=300, tokens_per_second=1000, output_tokens=3)
    result = await load(base_url, concurrency=2, warmup_s=0.1, measure_s=0.4)

    assert result.n_requests == 2
    assert result.n_errors == 0
    assert result.latency_p50_s is not None and result.latency_p50_s >= 0.3


def test_percentile_interpolates_like_numpy() -> None:
    assert percentile([], 50) is None
    assert percentile([5.0], 95) == 5.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    # numpy.percentile(range(1, 11), 95) == 9.55
    assert percentile([float(x) for x in range(1, 11)], 95) == pytest.approx(9.55)
