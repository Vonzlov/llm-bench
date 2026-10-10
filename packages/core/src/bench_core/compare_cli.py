"""Сравнение двух прогонов на одних и тех же примерах: разница метрик и её интервал.

    uv run bench-compare data/results/0shot/intent.jsonl data/results/intent.jsonl

Первым идёт прогон, с которым сравниваем, вторым — новый; разница считается как «второй
минус первый». Ответы сопоставляются по id примера, интервал — парный бутстреп по 1000
выборкам: в каждой выборке оба прогона берут одни и те же примеры. Разница значима, если
её интервал не накрывает ноль. Рядом с каждым файлом ответов должен лежать его
.summary.json: оттуда берутся задача, набор и настройки прогона.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from bench_core.quality import Difference, compare, read_results
from bench_core.quality_cli import TASKS, TITLES


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bench-compare", description="Сравнение двух прогонов на одних и тех же примерах."
    )
    parser.add_argument("first", type=Path, help="ответы прогона, с которым сравниваем")
    parser.add_argument("second", type=Path, help="ответы нового прогона")
    return parser.parse_args(argv)


def read_summary(results_path: Path) -> dict[str, Any]:
    path = results_path.with_suffix(".summary.json")
    if not path.exists():
        raise ValueError(f"нет файла {path}: настройки прогона берутся из него")
    summary: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return summary


def describe(summary: dict[str, Any]) -> str:
    """Прогон одной строкой: модель, число примеров с ответами в промпте и режим генерации."""
    # Прогоны, сделанные до few-shot и режима со схемой, этих полей не пишут: примеров в них
    # не было, а генерация была свободной.
    shots = summary.get("shots", 0)
    generation = "по JSON-схеме" if summary.get("constrained", False) else "свободная"
    return f"{summary['model']}, примеров с ответами {shots}, генерация {generation}"


def format_difference(name: str, difference: Difference) -> str:
    title = TITLES.get(name, name)
    verdict = "значимо" if difference.significant else "разницы не видно"
    return (
        f"  {title:<13} {difference.first:.3f} → {difference.second:.3f}"
        f"   разница {difference.delta:+.3f}"
        f"   интервал [{difference.low:+.3f}; {difference.high:+.3f}]   {verdict}"
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        first_summary = read_summary(args.first)
        second_summary = read_summary(args.second)
        for key in ("dataset", "dataset_version", "task_type"):
            if first_summary[key] != second_summary[key]:
                raise ValueError(
                    f"прогоны сделаны на разных наборах: {key} — {first_summary[key]}"
                    f" и {second_summary[key]}"
                )
        task = TASKS.get(first_summary["task_type"])
        if task is None:
            raise ValueError(f"задачу {first_summary['task_type']} сравнение пока не умеет")
        first = read_results(args.first)
        differences = compare(task, first, read_results(args.second))
    except (OSError, ValueError) as error:
        print(f"ошибка: {error}", file=sys.stderr)
        sys.exit(1)

    version = first_summary["dataset_version"][:12]
    print(f"набор {first_summary['dataset']}, версия {version}, общих примеров {len(first)}")
    print(f"первый: {args.first} — {describe(first_summary)}")
    print(f"второй: {args.second} — {describe(second_summary)}")
    print("разница — второй минус первый; интервал — 95-процентный, парный бутстреп:")
    for name, difference in differences.items():
        print(format_difference(name, difference))


if __name__ == "__main__":
    main()
