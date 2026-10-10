"""Открытые наборы для оценки качества и пул промптов для нагрузки.

Оба набора скачиваются с первоисточников и сверяются по SHA-256, поэтому у всех
получаются одни и те же данные.

MASSIVE 1.1 от Amazon (CC BY 4.0) — команды голосовому ассистенту. Каждая фраза
размечена интентом — одним из 60 — и слотами, то есть кусками фразы с типом:
«разбуди меня в [time : пять утра]». Из 300 русских фраз раздела test собираются два
набора: классификация интента и извлечение слотов в JSON.

XQuAD от DeepMind (CC BY-SA 4.0) — абзацы Википедии с вопросами и ответами,
переведённые на русский. Из него — 300 вопросов для ответов по тексту и пул промптов
для нагрузки.

Выборка простая случайная с фиксированным seed, поэтому точность на 300 примерах
оценивает точность на всём разделе. Выбранные id лежат в репозитории (datasets/*.ids),
и сборка берёт их оттуда: набор не изменится, даже если изменится код выборки.

Формат у всех наборов один — строка JSONL с полями id, input и expected. В нём же
будут загружаться свои наборы.

Для промпта с примерами (few-shot) у наборов MASSIVE есть три фразы из раздела train:
без слотов, с одним и с двумя слотами. Их id лежат в datasets/massive-ru-examples.ids,
а сами примеры — в manifest.json, рядом с вариантами ответа.
"""

import hashlib
import json
import random
import re
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

SEED = 2026
SAMPLE_SIZE = 300

XQUAD_URL = "https://raw.githubusercontent.com/google-deepmind/xquad/master/xquad.ru.json"
XQUAD_SHA256 = "208d5b1aa154c52b1b5c5eda16281e455e8fd198cdb9af3f469f0d6037d973bf"
MASSIVE_URL = (
    "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
)
# Из архива на 40 МБ нужен один файл — его хеш и сверяем.
MASSIVE_FILE = "ru-RU.jsonl"
MASSIVE_SHA256 = "af7367861ea21c1d69a084b290145815f7eaf82dcdc43b7ab572b993f9b2a69d"

# Слот в разметке MASSIVE: [тип : значение].
SLOT = re.compile(r"\[(\w+) : ([^\]]+)\]")

LOAD_PROMPT = "Ответь на вопрос по тексту.\n\nТекст:\n{context}\n\nВопрос: {question}"


@dataclass
class Dataset:
    """Набор, готовый к записи в JSONL."""

    name: str
    # classification, extraction, qa или load.
    task_type: str
    source: str
    license: str
    items: list[dict[str, Any]]
    # Из чего модель выбирает ответ: интенты для классификации, типы слотов для извлечения.
    labels: list[str] = field(default_factory=list)
    # Примеры с ответами для промпта (few-shot) — в том же формате, что и items.
    examples: list[dict[str, Any]] = field(default_factory=list)


def check_sha256(data: bytes, expected: str, name: str) -> None:
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(
            f"{name}: SHA-256 {actual}, а ожидали {expected}. Файл повреждён или "
            "источник изменился: удалите его из data/raw и запустите сборку заново."
        )


def download(client: httpx.Client, url: str, path: Path) -> None:
    """Скачивает файл во временный и только потом переименовывает.

    Так оборванная загрузка не оставит на месте битый файл, который приняли бы за целый.
    """
    part = path.with_name(path.name + ".part")
    with client.stream("GET", url) as response:
        response.raise_for_status()
        with part.open("wb") as file:
            for chunk in response.iter_bytes():
                file.write(chunk)
    part.replace(path)


def read_massive(archive: Path, sha256: str) -> list[dict[str, Any]]:
    """Достаёт из архива MASSIVE русские фразы, не распаковывая остальные языки."""
    data: bytes | None = None
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if member.name.rsplit("/", 1)[-1] == MASSIVE_FILE:
                file = tar.extractfile(member)
                data = file.read() if file else None
                break
    if data is None:
        raise ValueError(f"{archive.name}: внутри нет {MASSIVE_FILE}")
    check_sha256(data, sha256, MASSIVE_FILE)
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def read_xquad(path: Path, sha256: str) -> dict[str, Any]:
    data = path.read_bytes()
    check_sha256(data, sha256, path.name)
    doc: dict[str, Any] = json.loads(data)
    return doc


def parse_slots(annot_utt: str, utt: str) -> list[dict[str, str]]:
    """Слоты из разметки вида «разбуди меня в [time : пять утра]».

    Без скобок разметка должна совпасть с фразой: значит, каждое значение слота дословно
    есть во фразе. В русской части MASSIVE так у всех 16 521 фразы.
    """
    if SLOT.sub(r"\2", annot_utt) != utt:
        raise ValueError(f"разметка {annot_utt!r} не совпадает с фразой {utt!r}")
    return [{"type": kind, "value": value} for kind, value in SLOT.findall(annot_utt)]


def select_ids(candidates: list[str], k: int, seed: int = SEED) -> list[str]:
    """Простая случайная выборка без повторов. Порядок кандидатов на неё не влияет."""
    return sorted(random.Random(seed).sample(sorted(candidates), k))


def ids_for(path: Path, candidates: list[str], k: int) -> list[str]:
    """id выборки из файла в репозитории. Если файла ещё нет — выбирает и записывает."""
    if not path.exists():
        path.write_text("\n".join(select_ids(candidates, k)) + "\n", encoding="utf-8")
    ids = path.read_text(encoding="utf-8").split()
    missing = set(ids) - set(candidates)
    if missing:
        raise ValueError(f"{path}: в источнике нет id {sorted(missing)[:5]}")
    return ids


def select_examples(rows: list[dict[str, Any]], seed: int = SEED) -> list[str]:
    """id трёх фраз из train для примеров в промпте: без слотов, с одним и с двумя слотами.

    Так примеры показывают модели все случаи, в том числе пустой ответ. Интенты у них
    разные, чтобы примеры не тянули ответы классификации к одному классу. Примеры берутся
    из train, а оцениваем на test, поэтому в оценку они не попадают.
    """
    rng = random.Random(seed)
    train = sorted((row for row in rows if row["partition"] == "train"), key=lambda r: r["id"])
    chosen: list[str] = []
    used_intents: set[str] = set()
    for slot_count in (0, 1, 2):
        candidates = [
            row
            for row in train
            if len(SLOT.findall(row["annot_utt"])) == slot_count
            and row["intent"] not in used_intents
        ]
        row = rng.choice(candidates)
        chosen.append(row["id"])
        used_intents.add(row["intent"])
    return chosen


def example_ids_for(path: Path, rows: list[dict[str, Any]]) -> list[str]:
    """id примеров из файла в репозитории. Если файла ещё нет — выбирает и записывает."""
    if not path.exists():
        path.write_text("\n".join(select_examples(rows)) + "\n", encoding="utf-8")
    ids = path.read_text(encoding="utf-8").split()
    train_ids = {row["id"] for row in rows if row["partition"] == "train"}
    missing = set(ids) - train_ids
    if missing:
        raise ValueError(f"{path}: в разделе train нет id {sorted(missing)}")
    return ids


def massive_items(
    rows: list[dict[str, Any]], ids: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Одни и те же фразы в двух видах: с интентом и со слотами."""
    by_id = {row["id"]: row for row in rows}
    intents = []
    slots = []
    for massive_id in ids:
        row = by_id[massive_id]
        item_id = f"massive-ru-{massive_id}"
        intents.append({"id": item_id, "input": row["utt"], "expected": row["intent"]})
        expected_slots = parse_slots(row["annot_utt"], row["utt"])
        slots.append({"id": item_id, "input": row["utt"], "expected": expected_slots})
    return intents, slots


def massive_datasets(
    rows: list[dict[str, Any]], ids: list[str], example_ids: list[str]
) -> list[Dataset]:
    """Классификация интента и извлечение слотов на одних и тех же фразах."""
    intents, slots = massive_items(rows, ids)
    intent_examples, slot_examples = massive_items(rows, example_ids)
    # Варианты ответа — по всему файлу, а не по выборке: модель выбирает из всех 60 интентов.
    all_intents = sorted({row["intent"] for row in rows})
    all_slot_types = sorted({kind for row in rows for kind, _ in SLOT.findall(row["annot_utt"])})
    source = "MASSIVE 1.1, ru-RU, test"
    license_ = "CC BY 4.0"
    return [
        Dataset(
            "massive-ru-intent",
            "classification",
            source,
            license_,
            intents,
            all_intents,
            intent_examples,
        ),
        Dataset(
            "massive-ru-slots",
            "extraction",
            source,
            license_,
            slots,
            all_slot_types,
            slot_examples,
        ),
    ]


def clean(text: str) -> str:
    """В семи абзацах русского XQuAD есть невидимый символ BOM — убираем его."""
    return text.replace("\ufeff", "")


def xquad_questions(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Все вопросы XQuAD по id: на входе абзац и вопрос, на выходе варианты ответа."""
    questions = {}
    for article in doc["data"]:
        for paragraph in article["paragraphs"]:
            context = clean(paragraph["context"])
            for qa in paragraph["qas"]:
                questions[qa["id"]] = {
                    "id": qa["id"],
                    "input": {"context": context, "question": qa["question"]},
                    "expected": [answer["text"] for answer in qa["answers"]],
                }
    return questions


def xquad_load_prompts(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Пул промптов для нагрузки: абзац, следующий абзац той же статьи и вопрос к первому.

    Два абзаца — в среднем около 1 600 знаков, по нашей оценке порядка 400–500 токенов;
    точное число у каждого токенизатора своё. Так выглядит типичный запрос в RAG с
    найденными фрагментами текста. За последним абзацем статьи идёт первый, поэтому
    промптов столько же, сколько абзацев, — 240.
    """
    prompts = []
    for article in doc["data"]:
        paragraphs = article["paragraphs"]
        for i, first in enumerate(paragraphs):
            second = paragraphs[(i + 1) % len(paragraphs)]
            qa = first["qas"][0]
            context = clean(first["context"]) + "\n\n" + clean(second["context"])
            prompt = LOAD_PROMPT.format(context=context, question=qa["question"])
            prompts.append({"id": qa["id"], "input": prompt})
    return prompts


def build(
    massive_rows: list[dict[str, Any]], xquad_doc: dict[str, Any], ids_dir: Path
) -> list[Dataset]:
    """Все четыре набора. id выборки берутся из ids_dir или выбираются при первом запуске."""
    test_ids = [row["id"] for row in massive_rows if row["partition"] == "test"]
    massive_ids = ids_for(ids_dir / "massive-ru.ids", test_ids, SAMPLE_SIZE)
    example_ids = example_ids_for(ids_dir / "massive-ru-examples.ids", massive_rows)

    questions = xquad_questions(xquad_doc)
    qa_ids = ids_for(ids_dir / "xquad-ru-qa.ids", list(questions), SAMPLE_SIZE)
    source = "XQuAD, ru"
    qa = Dataset("xquad-ru-qa", "qa", source, "CC BY-SA 4.0", [questions[i] for i in qa_ids])
    load = Dataset("xquad-ru-load", "load", source, "CC BY-SA 4.0", xquad_load_prompts(xquad_doc))

    return [*massive_datasets(massive_rows, massive_ids, example_ids), qa, load]


def write_jsonl(dataset: Dataset, data_dir: Path) -> str:
    """Пишет набор в data_dir/<name>.jsonl и возвращает SHA-256 файла — версию набора."""
    text = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in dataset.items)
    data = text.encode("utf-8")
    (data_dir / f"{dataset.name}.jsonl").write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def manifest_entry(dataset: Dataset, version_hash: str) -> dict[str, Any]:
    """Строка manifest.json: то, что ляжет в таблицу datasets."""
    entry: dict[str, Any] = {
        "task_type": dataset.task_type,
        "n_items": len(dataset.items),
        "version_hash": version_hash,
        "source": dataset.source,
        "license": dataset.license,
    }
    if dataset.labels:
        entry["labels"] = dataset.labels
    if dataset.examples:
        entry["examples"] = dataset.examples
    return entry
