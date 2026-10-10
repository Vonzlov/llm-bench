"""Команда bench-compare: разница метрик двух прогонов по их файлам ответов и итогов."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from bench_core.compare_cli import main
from bench_core.quality import ItemResult, write_results


def write_run(directory: Path, predictions: Sequence[str | None], **summary_fields: Any) -> Path:
    """Прогон классификации, как его пишет bench-quality: ответы и итог рядом."""
    results = [
        ItemResult(
            id=str(index),
            expected="a",
            output=prediction or "",
            prediction=prediction,
            score=float(prediction == "a"),
            checks={},
            error=None,
            finish_reason="stop",
            latency_s=0.1,
            prompt_tokens=None,
            completion_tokens=None,
        )
        for index, prediction in enumerate(predictions)
    ]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "demo--m.jsonl"
    write_results(results, path)
    summary = {
        "dataset": "demo",
        "dataset_version": "ab" * 32,
        "task_type": "classification",
        "model": "m",
        **summary_fields,
    }
    path.with_suffix(".summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return path


def test_reports_significant_gain(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Второй прогон исправил 10 ошибок из 20 и ничего не сломал."""
    first = write_run(tmp_path / "0shot", ["a"] * 20 + ["b"] * 20)
    second = write_run(tmp_path, ["a"] * 30 + ["b"] * 10, shots=3)

    main([str(first), str(second)])

    output = capsys.readouterr().out
    assert "общих примеров 40" in output
    # У первого прогона в итоге нет поля shots: он сделан до появления примеров.
    assert "примеров с ответами 0" in output
    assert "примеров с ответами 3" in output
    accuracy = next(line for line in output.splitlines() if "точность" in line)
    assert "0.500 → 0.750" in accuracy
    assert "разница +0.250" in accuracy
    assert accuracy.endswith("значимо")


def test_same_answers_are_within_noise(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    first = write_run(tmp_path / "first", ["a", "b"] * 10)
    second = write_run(tmp_path / "second", ["a", "b"] * 10, constrained=True)

    main([str(first), str(second)])

    output = capsys.readouterr().out
    assert "генерация по JSON-схеме" in output
    accuracy = next(line for line in output.splitlines() if "точность" in line)
    assert accuracy.endswith("разницы не видно")


def test_rejects_runs_on_different_datasets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = write_run(tmp_path / "first", ["a"])
    second = write_run(tmp_path / "second", ["a"], dataset="other")

    with pytest.raises(SystemExit) as exit_info:
        main([str(first), str(second)])

    assert exit_info.value.code == 1
    assert "разных наборах" in capsys.readouterr().err


def test_rejects_run_without_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    first = write_run(tmp_path / "first", ["a"])
    second = write_run(tmp_path / "second", ["a"])
    second.with_suffix(".summary.json").unlink()

    with pytest.raises(SystemExit) as exit_info:
        main([str(first), str(second)])

    assert exit_info.value.code == 1
    assert ".summary.json" in capsys.readouterr().err
