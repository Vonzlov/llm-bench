"""Команда bench-quality на фейковой модели: метрики, файлы с ответами и коды выхода.

Фейковая модель отвечает словами «альфа бета гамма…», то есть всегда не по формату, —
этого хватает, чтобы проверить всю цепочку от набора до файла с итогом.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from bench_core import quality
from bench_core.quality_cli import main
from tests.unit.fake_llm.helpers import StartServer

INTENTS = [
    {"id": "1", "input": "разбуди меня в семь", "expected": "alarm_set"},
    {"id": "2", "input": "какая погода завтра", "expected": "weather_query"},
]
SLOTS = [
    {"id": "1", "input": "разбуди меня в семь", "expected": [{"type": "time", "value": "семь"}]},
    {"id": "2", "input": "какая погода завтра", "expected": [{"type": "date", "value": "завтра"}]},
]
# Три примера с ответами — столько модель видит перед вопросом по умолчанию.
INTENT_EXAMPLES: list[dict[str, Any]] = [
    {"id": "e1", "input": "поставь будильник на шесть", "expected": "alarm_set"},
    {"id": "e2", "input": "будет ли дождь", "expected": "weather_query"},
    {"id": "e3", "input": "разбуди меня в восемь", "expected": "alarm_set"},
]
SLOT_EXAMPLES: list[dict[str, Any]] = [
    {"id": "e1", "input": "будет ли дождь", "expected": []},
    {"id": "e2", "input": "разбуди в шесть", "expected": [{"type": "time", "value": "шесть"}]},
    {"id": "e3", "input": "дождь сегодня", "expected": [{"type": "date", "value": "сегодня"}]},
]


def make_dataset(
    tmp_path: Path,
    task_type: str = "classification",
    items: list[dict[str, Any]] = INTENTS,
    labels: tuple[str, ...] = ("alarm_set", "weather_query"),
    examples: list[dict[str, Any]] = INTENT_EXAMPLES,
) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    path = data / "demo.jsonl"
    path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items))
    entry = {
        "task_type": task_type,
        "n_items": len(items),
        "version_hash": "ab" * 32,
        "source": "тест",
        "license": "CC0",
        "labels": list(labels),
        "examples": examples,
    }
    (data / "manifest.json").write_text(json.dumps({"demo": entry}, ensure_ascii=False))
    return path


def test_writes_answers_and_summary(
    start_server: StartServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = make_dataset(tmp_path)
    base_url = start_server(ttft_ms=0, tokens_per_second=100_000)
    out = tmp_path / "results" / "demo.jsonl"

    main(["--dataset", str(dataset), "--base-url", f"{base_url}/v1", "--out", str(out)])

    output = capsys.readouterr().out
    assert "точность" in output
    assert "вне списка" in output
    assert "примеров с ответами в промпте 3" in output
    answers = [json.loads(line) for line in out.read_text().splitlines()]
    assert [answer["id"] for answer in answers] == ["1", "2"]
    summary = json.loads((tmp_path / "results" / "demo.summary.json").read_text())
    assert summary["metrics"]["accuracy"]["value"] == 0.0
    assert summary["metrics"]["out_of_list"]["value"] == 1.0
    assert summary["dataset_version"] == "ab" * 32
    assert summary["truncated"] == 0
    assert summary["shots"] == 3
    assert summary["example_ids"] == ["e1", "e2", "e3"]


def test_run_without_examples_goes_to_its_own_file(
    start_server: StartServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Прогон без примеров не затирает прогон с ними: к имени файла добавляется --0shot."""
    dataset = make_dataset(tmp_path)
    base_url = start_server(ttft_ms=0, tokens_per_second=100_000)
    monkeypatch.chdir(tmp_path)

    main(["--dataset", str(dataset), "--base-url", f"{base_url}/v1", "--shots", "0"])

    summary_path = tmp_path / "data" / "results" / "demo--fake-llm--0shot.summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["shots"] == 0
    assert summary["example_ids"] == []


def test_rejects_more_examples_than_dataset_has(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Набор, собранный до появления примеров, не подменяет прогон с примерами прогоном без них."""
    dataset = make_dataset(tmp_path, examples=[])

    with pytest.raises(SystemExit) as exit_info:
        main(["--dataset", str(dataset)])

    assert exit_info.value.code == 1
    assert "make datasets" in capsys.readouterr().err


def test_reports_answers_cut_by_token_limit(
    start_server: StartServer, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Фейковая модель хочет ответить 64 токенами, а лимит — 8: оба ответа обрезаны."""
    dataset = make_dataset(tmp_path)
    base_url = start_server(ttft_ms=0, tokens_per_second=100_000, output_tokens=64)
    out = tmp_path / "results" / "demo.jsonl"
    args = ["--dataset", str(dataset), "--base-url", f"{base_url}/v1", "--out", str(out)]

    main([*args, "--max-tokens", "8"])

    assert "обрезано лимитом в 8 токенов: 2 ответов из 2" in capsys.readouterr().out
    summary = json.loads((tmp_path / "results" / "demo.summary.json").read_text())
    assert summary["truncated"] == 2


def test_extraction_with_schema_goes_to_its_own_file(
    start_server: StartServer,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Прогон со схемой не затирает свободный: к имени файла добавляется --schema."""
    dataset = make_dataset(tmp_path, "extraction", SLOTS, ("date", "time"), SLOT_EXAMPLES)
    base_url = start_server(ttft_ms=0, tokens_per_second=100_000)
    monkeypatch.chdir(tmp_path)

    main(["--dataset", str(dataset), "--base-url", f"{base_url}/v1", "--constrained"])

    assert "валидный JSON" in capsys.readouterr().out
    summary_path = tmp_path / "data" / "results" / "demo--fake-llm--schema.summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["constrained"] is True
    assert summary["metrics"]["valid_json"]["value"] == 0.0


def test_classification_has_no_schema_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = make_dataset(tmp_path)

    with pytest.raises(SystemExit) as exit_info:
        main(["--dataset", str(dataset), "--constrained"])

    assert exit_info.value.code == 1
    assert "нет режима с JSON-схемой" in capsys.readouterr().err


def test_exits_with_error_when_no_request_succeeds(
    start_server: StartServer,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(quality, "RETRY_PAUSE_S", 0.0)
    dataset = make_dataset(tmp_path)
    base_url = start_server(ttft_ms=0, tokens_per_second=100_000, error_rate=1.0)
    out = tmp_path / "results" / "demo.jsonl"

    with pytest.raises(SystemExit) as exit_info:
        main(["--dataset", str(dataset), "--base-url", f"{base_url}/v1", "--out", str(out)])

    assert exit_info.value.code == 1
    assert "http_500 ×2" in capsys.readouterr().out


def test_rejects_dataset_missing_from_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = make_dataset(tmp_path)
    renamed = dataset.with_name("other.jsonl")
    dataset.rename(renamed)

    with pytest.raises(SystemExit) as exit_info:
        main(["--dataset", str(renamed)])

    assert exit_info.value.code == 1
    assert "make datasets" in capsys.readouterr().err


def test_rejects_task_it_cannot_run_yet(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset = make_dataset(tmp_path, task_type="qa")

    with pytest.raises(SystemExit) as exit_info:
        main(["--dataset", str(dataset)])

    assert exit_info.value.code == 1
    assert "qa" in capsys.readouterr().err
