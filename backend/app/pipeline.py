"""Опциональный синтетический контур для изолированной разработки интерфейса.

Production API использует сохраняемые analysis jobs и проверенный
``analysis-response-v1``. Этот модуль доступен только при явном
``NEXTWAVE_ENABLE_SYNTHETIC_DEMO=1`` и не подменяет продуктовый результат.
"""

import asyncio
import json
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

from .models import BUCKETS, Bucket, Coverage, SourceStat, Stage, Trend

DATA = Path(__file__).parent / "data"

CORPUS_VERSION = "synthetic-ui-demo-2026.09.11"
METHOD_VERSION = "weighted-rules-demo-0.1"
TOP_N = 15

# Эти веса нужны только для проверки UI-контракта. Обученная модель заменит их в V1-09.
FACTOR_WEIGHTS = {"growth": 0.35, "novelty": 0.25, "independence": 0.2, "evidence": 0.2}

# Демонстрационные пороги меняются вместе с METHOD_VERSION и не являются измеренными метриками.
THRESHOLDS = {
    "novelty_min": 0.6,
    "growth_min": 0.5,
    "evidence_min": 0.6,
    "independent_min": 5,
    "documents_min": 10,
}

def pct(value: float) -> int:
    return round(value * 100)

STAGES = [
    Stage(key="collect", label="Загрузка документов из источников"),
    Stage(key="clean", label="Очистка и удаление дубликатов"),
    Stage(key="select", label="Отбор по направлению"),
    Stage(key="cluster", label="Кластеризация тем"),
    Stage(key="history", label="Сопоставление с историческим корпусом"),
    Stage(key="rank", label="Ранжирование и сборка карточек"),
]

_demo = json.loads((DATA / "demo_trends.json").read_text(encoding="utf-8"))
# Демонстрационный каркас поддерживает одно направление.
DIRECTIONS = [_demo["direction"]]
KEYWORDS = {"ии", "ai", "искусственн", "машинн", "llm", "нейросет", "языков", "интеллект"}


def score(trend: dict) -> float:
    return round(sum(f["value"] * FACTOR_WEIGHTS[f["key"]] for f in trend["factors"]), 4)


def is_covered(query: str) -> bool:
    q = query.casefold()
    return any(word in q for word in KEYWORDS)


def coverage() -> Coverage:
    documents = sum(t["document_count"] for t in _demo["trends"])
    return Coverage(
        directions=DIRECTIONS,
        examples=["технологии в ИИ", "агенты и LLM", "инференс языковых моделей"],
        documents_from=date(2024, 1, 1),
        documents_to=date(2026, 9, 10),
        document_count=documents,
        sources=[
            SourceStat(name="arXiv", source_type="preprint", documents=int(documents * 0.62)),
            SourceStat(name="Отраслевые отчёты", source_type="report", documents=int(documents * 0.18)),
            SourceStat(name="Материалы конференций", source_type="conference", documents=int(documents * 0.2)),
        ],
        corpus_version=CORPUS_VERSION,
        method_version=METHOD_VERSION,
        thresholds=THRESHOLDS,
        updated_at=datetime(2026, 9, 11, 9, 0, tzinfo=UTC),
    )


def classify(trend: dict) -> tuple[Bucket, str]:
    """Корзина кандидата и причина попадания. Пороги — часть версии метода."""
    f = {factor["key"]: factor["value"] for factor in trend["factors"]}
    documents, independent = trend["document_count"], trend["independent_sources"]

    if f["novelty"] < THRESHOLDS["novelty_min"]:
        return "excluded", (
            f"Не прошёл проверку на новизну: {pct(f['novelty'])} при пороге "
            f"{pct(THRESHOLDS['novelty_min'])}. Тема присутствует в корпусе с {trend['first_seen']} "
            "и развивается как устоявшееся направление, а не как зарождающееся."
        )
    if f["growth"] < THRESHOLDS["growth_min"]:
        return "excluded", (
            f"Не прошёл проверку на зарождаемость: рост {pct(f['growth'])} при пороге "
            f"{pct(THRESHOLDS['growth_min'])}. Доля темы в корпусе направления не увеличивается "
            "между сопоставимыми временными окнами."
        )
    if f["evidence"] < THRESHOLDS["evidence_min"]:
        return "watchlist", (
            f"Доказательная база {pct(f['evidence'])} при пороге {pct(THRESHOLDS['evidence_min'])}: "
            "в найденных материалах преобладают предположения без воспроизводимых измерений."
        )
    if independent < THRESHOLDS["independent_min"]:
        return "watchlist", (
            f"Подтверждения сводятся к {independent} независимым источникам при пороге "
            f"{THRESHOLDS['independent_min']}. Признаки роста есть, независимость подтверждений "
            "пока недостаточна."
        )
    if documents < THRESHOLDS["documents_min"]:
        return "watchlist", (
            f"В корпусе {documents} документов по теме при пороге {THRESHOLDS['documents_min']}. "
            "Выводы о динамике на такой выборке неустойчивы."
        )
    return "main", (
        f"Прошёл все пороги отбора: новизна {pct(f['novelty'])}, рост {pct(f['growth'])}, "
        f"доказательная база {pct(f['evidence'])}, {independent} независимых источников, "
        f"{documents} документов."
    )


def rank(trends: list[dict]) -> list[Trend]:
    """Классифицирует кандидатов по корзинам и нумерует каждую корзину отдельно."""
    ordered = sorted(trends, key=lambda t: (-score(t), t["id"]))
    classified = [[trend, *classify(trend)] for trend in ordered]

    qualified = 0
    for item in classified:
        if item[1] != "main":
            continue
        qualified += 1
        # Прошедшие пороги сверх ТОП-15 не отбрасываются, а уходят в наблюдение
        if qualified > TOP_N:
            item[1] = "watchlist"
            item[2] = (
                f"Прошёл пороги отбора, но не вошёл в ТОП-15: рейтинг {pct(score(item[0]))} ниже, "
                "чем у пятнадцатого кандидата основного списка."
            )

    counters = dict.fromkeys(BUCKETS, 0)
    result = []
    for trend, bucket, reason in classified:
        counters[bucket] += 1
        result.append(
            Trend(**trend, score=score(trend), rank=counters[bucket], bucket=bucket, bucket_reason=reason)
        )
    return result


async def run(query: str, on_stage: Callable[[Stage, float], None]) -> list[Trend]:
    """Прогоняет этапы, сообщая прогресс через on_stage. Возвращает отранжированных кандидатов."""
    covered = is_covered(query)
    for i, stage in enumerate(STAGES, start=1):
        on_stage(stage, i / len(STAGES))
        # Небольшая задержка позволяет проверить состояния прогресса в интерфейсе.
        await asyncio.sleep(0.45)
        if not covered and stage.key == "select":
            return []
    return rank(_demo["trends"])
