"""Наборы данных: разбор разметки, выборка, сборка и скачивание — без сети.

Настоящие файлы MASSIVE и XQuAD в тесты не попадают. Фикстуры повторяют их формат
на нескольких придуманных строках.
"""

import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import httpx
import pytest

from bench_core import datasets
from bench_core.datasets_cli import main


def massive_row(massive_id: str, partition: str, intent: str, annot_utt: str) -> dict[str, Any]:
    return {
        "id": massive_id,
        "locale": "ru-RU",
        "partition": partition,
        "intent": intent,
        "utt": datasets.SLOT.sub(r"\2", annot_utt),
        "annot_utt": annot_utt,
    }


MASSIVE_ROWS = [
    massive_row("1", "test", "alarm_set", "поставь будильник на [time : семь утра]"),
    massive_row(
        "2", "test", "weather_query", "какая погода [date : завтра] в [place_name : казани]"
    ),
    massive_row("3", "test", "general_joke", "расскажи анекдот"),
    massive_row("4", "train", "cooking_recipe", "как испечь [food_type : блины]"),
    massive_row("5", "train", "general_joke", "пошути"),
    massive_row("6", "train", "weather_query", "погода [date : сегодня] в [place_name : москве]"),
]


def paragraph(context: str, question_id: str, answer: str) -> dict[str, Any]:
    qa = {
        "id": question_id,
        "question": f"Вопрос {question_id}?",
        "answers": [{"answer_start": 0, "text": answer}],
    }
    return {"context": context, "qas": [qa]}


XQUAD_DOC = {
    "version": "1.1",
    "data": [
        {
            "title": "Первая статья",
            # В настоящем XQuAD у семи абзацев в начале стоит BOM.
            "paragraphs": [
                paragraph("\ufeffПервый абзац.", "q1", "Первый"),
                paragraph("Второй абзац.", "q2", "Второй"),
            ],
        },
        {"title": "Вторая статья", "paragraphs": [paragraph("Третий абзац.", "q3", "Третий")]},
    ],
}


def massive_archive(path: Path, rows: list[dict[str, Any]]) -> str:
    """Архив в формате MASSIVE с одним файлом ru-RU.jsonl. Возвращает SHA-256 этого файла."""
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
    info = tarfile.TarInfo("1.1/data/ru-RU.jsonl")
    info.size = len(data)
    with tarfile.open(path, "w:gz") as tar:
        tar.addfile(info, io.BytesIO(data))
    return hashlib.sha256(data).hexdigest()


def test_parse_slots_returns_type_and_verbatim_value() -> None:
    row = MASSIVE_ROWS[1]
    assert datasets.parse_slots(row["annot_utt"], row["utt"]) == [
        {"type": "date", "value": "завтра"},
        {"type": "place_name", "value": "казани"},
    ]
    assert datasets.parse_slots("расскажи анекдот", "расскажи анекдот") == []


def test_parse_slots_rejects_markup_that_differs_from_phrase() -> None:
    with pytest.raises(ValueError, match="не совпадает"):
        datasets.parse_slots("разбуди в [time : семь утра]", "разбуди в восемь утра")


def test_select_ids_is_repeatable_and_ignores_candidate_order() -> None:
    candidates = [str(i) for i in range(100)]
    first = datasets.select_ids(candidates, 10)

    assert datasets.select_ids(list(reversed(candidates)), 10) == first
    assert len(set(first)) == 10
    assert datasets.select_ids(candidates, 10, seed=1) != first


def test_ids_are_selected_once_and_then_read_from_file(tmp_path: Path) -> None:
    path = tmp_path / "sample.ids"
    candidates = ["1", "2", "3", "4"]

    selected = datasets.ids_for(path, candidates, 2)
    assert path.read_text().split() == selected

    # Файл в репозитории главнее кода выборки: что в нём записано, то и берём.
    path.write_text("4\n1\n")
    assert datasets.ids_for(path, candidates, 2) == ["4", "1"]

    path.write_text("999\n")
    with pytest.raises(ValueError, match="999"):
        datasets.ids_for(path, candidates, 2)


def test_examples_come_from_train_with_zero_one_and_two_slots() -> None:
    assert datasets.select_examples(MASSIVE_ROWS) == ["5", "4", "6"]


def test_example_ids_are_pinned_in_a_file(tmp_path: Path) -> None:
    path = tmp_path / "examples.ids"

    assert datasets.example_ids_for(path, MASSIVE_ROWS) == ["5", "4", "6"]
    assert path.read_text().split() == ["5", "4", "6"]

    # Пример из test недопустим: он попал бы и в промпт, и в оценку.
    path.write_text("1\n")
    with pytest.raises(ValueError, match="train"):
        datasets.example_ids_for(path, MASSIVE_ROWS)


def test_massive_gives_two_datasets_on_the_same_phrases() -> None:
    intent, slots = datasets.massive_datasets(MASSIVE_ROWS, ["2", "3"], ["5", "4", "6"])

    assert [item["id"] for item in intent.items] == ["massive-ru-2", "massive-ru-3"]
    assert [item["input"] for item in slots.items] == [item["input"] for item in intent.items]
    assert intent.items[0]["expected"] == "weather_query"
    assert slots.items[1]["expected"] == []
    # Варианты ответа — по всему файлу, а не по выборке и не только по test.
    assert intent.labels == ["alarm_set", "cooking_recipe", "general_joke", "weather_query"]
    assert slots.labels == ["date", "food_type", "place_name", "time"]
    # Примеры — те же три фразы в двух видах.
    assert [example["id"] for example in intent.examples] == [
        "massive-ru-5",
        "massive-ru-4",
        "massive-ru-6",
    ]
    assert intent.examples[0]["expected"] == "general_joke"
    assert slots.examples[0]["expected"] == []
    assert slots.examples[2]["expected"] == [
        {"type": "date", "value": "сегодня"},
        {"type": "place_name", "value": "москве"},
    ]


def test_xquad_questions_drop_bom_and_keep_answers() -> None:
    questions = datasets.xquad_questions(XQUAD_DOC)

    assert sorted(questions) == ["q1", "q2", "q3"]
    assert questions["q1"]["input"] == {"context": "Первый абзац.", "question": "Вопрос q1?"}
    assert questions["q1"]["expected"] == ["Первый"]


def test_load_prompt_is_paragraph_next_paragraph_and_question() -> None:
    prompts = datasets.xquad_load_prompts(XQUAD_DOC)

    assert [prompt["id"] for prompt in prompts] == ["q1", "q2", "q3"]
    assert "Первый абзац.\n\nВторой абзац." in prompts[0]["input"]
    assert prompts[0]["input"].endswith("Вопрос: Вопрос q1?")
    # За последним абзацем статьи идёт первый.
    assert "Второй абзац.\n\nПервый абзац." in prompts[1]["input"]


def test_read_massive_takes_one_file_from_archive_and_checks_hash(tmp_path: Path) -> None:
    archive = tmp_path / "massive.tar.gz"
    sha256 = massive_archive(archive, MASSIVE_ROWS)

    assert datasets.read_massive(archive, sha256) == MASSIVE_ROWS
    with pytest.raises(ValueError, match="SHA-256"):
        datasets.read_massive(archive, "0" * 64)


def test_write_jsonl_keeps_cyrillic_and_returns_file_hash(tmp_path: Path) -> None:
    item = {"id": "1", "input": "поставь будильник", "expected": "alarm_set"}
    dataset = datasets.Dataset("demo", "classification", "тест", "CC BY 4.0", [item])

    version_hash = datasets.write_jsonl(dataset, tmp_path)

    data = (tmp_path / "demo.jsonl").read_bytes()
    assert "поставь будильник" in data.decode()
    assert version_hash == hashlib.sha256(data).hexdigest()


def test_download_leaves_no_file_when_server_fails(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/ok.json":
            return httpx.Response(200, content=b'{"ok": true}')
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        datasets.download(client, "https://example.org/ok.json", tmp_path / "ok.json")
        with pytest.raises(httpx.HTTPStatusError):
            datasets.download(client, "https://example.org/missing.json", tmp_path / "missing.json")

    assert (tmp_path / "ok.json").read_bytes() == b'{"ok": true}'
    assert sorted(path.name for path in tmp_path.iterdir()) == ["ok.json"]


def raw_files(data_dir: Path) -> tuple[str, str]:
    """Кладёт фикстуры туда, куда bench-datasets скачал бы настоящие файлы."""
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)
    massive_sha256 = massive_archive(raw_dir / "amazon-massive-dataset-1.1.tar.gz", MASSIVE_ROWS)
    xquad = json.dumps(XQUAD_DOC, ensure_ascii=False).encode()
    (raw_dir / "xquad.ru.json").write_bytes(xquad)
    return massive_sha256, hashlib.sha256(xquad).hexdigest()


def test_cli_builds_all_datasets_from_downloaded_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    massive_sha256, xquad_sha256 = raw_files(tmp_path / "data")
    monkeypatch.setattr(datasets, "MASSIVE_SHA256", massive_sha256)
    monkeypatch.setattr(datasets, "XQUAD_SHA256", xquad_sha256)
    monkeypatch.setattr(datasets, "SAMPLE_SIZE", 2)
    args = ["--data-dir", str(tmp_path / "data"), "--ids-dir", str(tmp_path / "ids")]

    main(args)
    first = json.loads((tmp_path / "data" / "manifest.json").read_text())
    main(args)
    second = json.loads((tmp_path / "data" / "manifest.json").read_text())

    assert list(first) == ["massive-ru-intent", "massive-ru-slots", "xquad-ru-qa", "xquad-ru-load"]
    assert first["massive-ru-intent"]["n_items"] == 2
    assert first["xquad-ru-load"]["n_items"] == 3
    assert len((tmp_path / "ids" / "massive-ru.ids").read_text().split()) == 2
    assert (tmp_path / "ids" / "massive-ru-examples.ids").read_text().split() == ["5", "4", "6"]
    assert len(first["massive-ru-slots"]["examples"]) == 3
    assert "examples" not in first["xquad-ru-qa"]
    # Второй запуск берёт id из файлов и собирает те же наборы байт в байт.
    assert first == second
    assert "massive-ru-slots" in capsys.readouterr().out


def test_cli_stops_when_file_hash_does_not_match(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_files(tmp_path / "data")

    with pytest.raises(SystemExit) as exit_info:
        main(["--data-dir", str(tmp_path / "data"), "--ids-dir", str(tmp_path / "ids")])

    assert exit_info.value.code == 1
    assert "SHA-256" in capsys.readouterr().err
