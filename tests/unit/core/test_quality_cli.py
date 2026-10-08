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


def make_dataset(
    tmp_path: Path,
    task_type: str = "classification",
    items: list[dict[str, Any]] = INTENTS,
    labels: tuple[str, ...] = ("alarm_set", "weather_query"),
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
    answers = [json.loads(line) for line in out.read_text().splitlines()]
    assert [answer["id"] for answer in answers] == ["1", "2"]
    summary = json.loads((tmp_path / "results" / "demo.summary.json").read_text())
    assert summary["metrics"]["accuracy"]["value"] == 0.0
    assert summary["metrics"]["out_of_list"]["value"] == 1.0
    assert summary["dataset_version"] == "ab" * 32


def test_extraction_with_schema_goes_to_its_own_file(
    start_server: StartServer,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Прогон со схемой не затирает свободный: к имени файла добавляется --schema."""
    dataset = make_dataset(tmp_path, "extraction", SLOTS, ("date", "time"))
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
