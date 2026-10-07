"""Команда bench-load: строка таблицы на каждую ступень и код выхода по результату."""

import json
from pathlib import Path

import pytest

from bench_core.load_cli import main
from tests.unit.fake_llm.helpers import StartServer


def levels_printed(output: str) -> list[str]:
    """Номера ступеней из строк таблицы: только они начинаются с числа."""
    rows = [line.split() for line in output.splitlines()]
    return [parts[0] for parts in rows if parts and parts[0].isdigit()]


def test_prints_a_row_per_level(
    start_server: StartServer, capsys: pytest.CaptureFixture[str]
) -> None:
    base_url = start_server(ttft_ms=10, tokens_per_second=1000, output_tokens=3)
    args = [
        "--base-url",
        f"{base_url}/v1",
        "--levels",
        "1,2",
        "--warmup",
        "0.1",
        "--measure",
        "0.4",
    ]

    with pytest.raises(SystemExit) as exit_info:
        main([*args, "--max-tokens", "3"])

    output = capsys.readouterr().out
    assert exit_info.value.code == 0
    assert "TTFT p50/p95, мс" in output
    assert levels_printed(output) == ["1", "2"]


def test_takes_prompts_from_dataset_file(
    start_server: StartServer, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    prompts = tmp_path / "prompts.jsonl"
    lines = [{"id": "1", "input": "Первый промпт"}, {"id": "2", "input": "Второй промпт"}]
    prompts.write_text("".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines))
    base_url = start_server(ttft_ms=10, tokens_per_second=1000, output_tokens=3)
    args = ["--base-url", f"{base_url}/v1", "--levels", "1", "--warmup", "0.1", "--measure", "0.3"]

    with pytest.raises(SystemExit) as exit_info:
        main([*args, "--prompts", str(prompts)])

    output = capsys.readouterr().out
    assert exit_info.value.code == 0
    assert f"промпты: {prompts}, 2 шт." in output
    assert levels_printed(output) == ["1"]


def test_rejects_prompts_that_are_not_strings(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Вопросы XQuAD с абзацем на входе — объект, а не строка: для нагрузки не годятся."""
    prompts = tmp_path / "qa.jsonl"
    prompts.write_text(json.dumps({"id": "1", "input": {"context": "абзац", "question": "?"}}))

    with pytest.raises(SystemExit) as exit_info:
        main(["--prompts", str(prompts)])

    assert exit_info.value.code == 2
    assert "input должно быть строкой" in capsys.readouterr().err


def test_exits_with_error_when_nothing_succeeds(
    start_server: StartServer, capsys: pytest.CaptureFixture[str]
) -> None:
    base_url = start_server(ttft_ms=0, tokens_per_second=100_000, error_rate=1.0)
    args = ["--base-url", f"{base_url}/v1", "--levels", "1", "--warmup", "0.05", "--measure", "0.2"]

    with pytest.raises(SystemExit) as exit_info:
        main(args)

    output = capsys.readouterr().out
    assert exit_info.value.code == 1
    assert "ошибки: http_500" in output
