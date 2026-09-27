"""Проба bench-probe: печатает замеры и возвращает код выхода по результату."""

import pytest

from bench_core.probe import main
from tests.unit.fake_llm.helpers import StartServer


def test_probe_prints_measurements(
    start_server: StartServer, capsys: pytest.CaptureFixture[str]
) -> None:
    base_url = start_server(ttft_ms=50, tokens_per_second=1000, output_tokens=8)

    with pytest.raises(SystemExit) as exit_info:
        main(["--base-url", f"{base_url}/v1", "--max-tokens", "8"])

    output = capsys.readouterr().out
    assert exit_info.value.code == 0
    assert "статус             ok" in output
    assert "до первого токена" in output
    assert "токенов            8" in output


def test_probe_exits_with_error_code(
    start_server: StartServer, capsys: pytest.CaptureFixture[str]
) -> None:
    base_url = start_server(ttft_ms=0, tokens_per_second=100_000, rate_limit_rate=1.0)

    with pytest.raises(SystemExit) as exit_info:
        main(["--base-url", f"{base_url}/v1"])

    output = capsys.readouterr().out
    assert exit_info.value.code == 1
    assert "статус             http_429" in output
    assert "повторить через    1 с" in output
