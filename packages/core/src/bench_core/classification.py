"""Классификация: модель выбирает один вариант из списка, например интент фразы.

Ответ разбираем строго: срезаем по краям пробелы, кавычки, звёздочки Markdown и точку
и сравниваем без учёта регистра. Если такого варианта в списке нет, ответ неверный и
вдобавок попадает в долю ответов вне списка. Так видно отдельно, где модель не знает
ответ, а где не следует формату.

Промпт общий для любых наборов классификации, в том числе своих: в нём нет ничего про
голосовых ассистентов, только текст и список вариантов.
"""

from collections import Counter
from collections.abc import Sequence
from typing import Any

from bench_core.quality import ItemResult, Metric, Scored, Task, metric

SYSTEM_PROMPT = (
    "Определи, к какой категории относится текст. Выбери ровно одну категорию из списка "
    "и ответь только её названием — без пояснений и других слов.\n\n"
    "Категории:\n{labels}"
)
# Что срезаем по краям ответа: пробелы, кавычки, обратные кавычки, звёздочки и точку.
EDGE_CHARS = " \t\r\n\"'`*«»."


def build_messages(item: dict[str, Any], labels: list[str]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(labels="\n".join(labels))},
        {"role": "user", "content": f"Текст: {item['input']}\nКатегория:"},
    ]


def parse(output: str, labels: list[str]) -> str | None:
    """Вариант из списка, которым ответила модель; None, если ответ не из списка."""
    by_lowercase = {label.lower(): label for label in labels}
    return by_lowercase.get(output.strip(EDGE_CHARS).lower())


def score(item: dict[str, Any], output: str, labels: list[str]) -> Scored:
    prediction = parse(output, labels)
    return Scored(prediction, float(prediction == item["expected"]))


def accuracy(results: Sequence[ItemResult]) -> float:
    return sum(result.score for result in results) / len(results)


def macro_f1(results: Sequence[ItemResult]) -> float:
    """Среднее F1 по классам, которые есть среди эталонов, как f1_score(average="macro").

    F1 класса — 2·TP / (2·TP + FP + FN). Ответ вне списка не попадает ни в один класс:
    для эталонного класса это пропуск (FN), а ложным срабатыванием (FP) он не считается.
    """
    true_positive: Counter[str] = Counter()
    false_positive: Counter[str] = Counter()
    false_negative: Counter[str] = Counter()
    for result in results:
        if result.prediction == result.expected:
            true_positive[result.expected] += 1
        else:
            false_negative[result.expected] += 1
            if result.prediction is not None:
                false_positive[result.prediction] += 1
    classes = {result.expected for result in results}
    f1 = [
        2 * true_positive[c] / (2 * true_positive[c] + false_positive[c] + false_negative[c])
        for c in classes
    ]
    return sum(f1) / len(f1)


def out_of_list(results: Sequence[ItemResult]) -> float:
    """Доля ответов не по формату: сервер ответил, но такого варианта в списке нет."""
    misses = sum(1 for result in results if result.error is None and result.prediction is None)
    return misses / len(results)


def summarize(results: list[ItemResult]) -> dict[str, Metric]:
    return {
        "accuracy": metric(results, accuracy),
        "macro_f1": metric(results, macro_f1),
        "out_of_list": metric(results, out_of_list),
    }


TASK = Task(
    name="classification",
    max_tokens=64,
    build_messages=build_messages,
    score=score,
    summarize=summarize,
)
