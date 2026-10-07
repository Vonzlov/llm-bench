"""Скачивает открытые наборы и собирает из них JSONL для оценки качества и нагрузки.

    uv run bench-datasets
    make datasets

Сырые файлы ложатся в data/raw, готовые наборы и manifest.json — в data/. В git эта
папка не попадает: в репозитории лежат только id выбранных примеров, в datasets/.
Если файлы уже скачаны, повторный запуск их не качает.
"""

import argparse
import json
import sys
from pathlib import Path

import httpx

from bench_core import datasets


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bench-datasets",
        description="Скачивает MASSIVE и XQuAD и собирает наборы в JSONL.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="куда класть данные")
    parser.add_argument(
        "--ids-dir", type=Path, default=Path("datasets"), help="где лежат id выборки"
    )
    return parser.parse_args(argv)


def fetch(client: httpx.Client, url: str, raw_dir: Path) -> Path:
    """Путь к сырому файлу; если его ещё нет — скачивает."""
    path = raw_dir / url.rsplit("/", 1)[1]
    if not path.exists():
        print(f"скачиваю {url}", flush=True)
        datasets.download(client, url, path)
    return path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raw_dir = args.data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    args.ids_dir.mkdir(parents=True, exist_ok=True)

    try:
        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            xquad_path = fetch(client, datasets.XQUAD_URL, raw_dir)
            massive_path = fetch(client, datasets.MASSIVE_URL, raw_dir)
        xquad = datasets.read_xquad(xquad_path, datasets.XQUAD_SHA256)
        massive = datasets.read_massive(massive_path, datasets.MASSIVE_SHA256)
        built = datasets.build(massive, xquad, args.ids_dir)
    except (httpx.HTTPError, ValueError) as error:
        print(f"ошибка: {error}", file=sys.stderr)
        sys.exit(1)

    manifest = {}
    print(f"{'набор':<18}  {'задача':<14}  {'примеров':>8}  версия")
    for dataset in built:
        version_hash = datasets.write_jsonl(dataset, args.data_dir)
        manifest[dataset.name] = datasets.manifest_entry(dataset, version_hash)
        print(
            f"{dataset.name:<18}  {dataset.task_type:<14}  {len(dataset.items):>8}"
            f"  {version_hash[:12]}"
        )
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    (args.data_dir / "manifest.json").write_text(manifest_text, encoding="utf-8")


if __name__ == "__main__":
    main()
