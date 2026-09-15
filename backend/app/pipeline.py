"""Шов между API и анализом.

Сейчас `run` отдаёт зафиксированный демо-корпус. Реальный пайплайн
(сбор → очистка → кластеризация → динамика → ранжирование) подставляется
внутрь `run`, не меняя контракт из models.py и фронтенд.
"""

import asyncio
import json
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

from nextwave.contracts import (
    CandidateFeatures,
    CandidateStatus,
    DevelopmentStage,
    Evidence,
    ExclusionReason,
    ModelPrediction,
    SourceType,
    TrustLevel,
)

from .models import BUCKETS, Bucket, Coverage, SourceStat, Stage, Trend

DATA = Path(__file__).parent / "data"

CORPUS_VERSION = "demo-2026.09.11"
METHOD_VERSION = "tfidf-baseline-0.1"
MODEL_VERSION = "demo-weighted-baseline-0.1"
FEATURE_VERSION = "candidate-features-1"
TOP_N = 15

# Веса фиксируются версией метода и проверяются на данных (methodology.md, «Основания рейтинга»)
FACTOR_WEIGHTS = {"growth": 0.35, "novelty": 0.25, "independence": 0.2, "evidence": 0.2}

# Пороги отбора по корзинам. Тоже часть версии метода: меняются вместе с METHOD_VERSION.
THRESHOLDS = {
    "novelty_min": 0.6,
    "growth_min": 0.5,
    "evidence_min": 0.6,
    "independent_min": 5,
    "documents_min": 10,
}

pct = lambda value: round(value * 100)  # noqa: E731 — короткая подпись для текста причин

STAGES = [
    Stage(key="collect", label="Загрузка документов из источников"),
    Stage(key="clean", label="Очистка и удаление дубликатов"),
    Stage(key="select", label="Отбор по направлению"),
    Stage(key="cluster", label="Кластеризация тем"),
    Stage(key="history", label="Сопоставление с историческим корпусом"),
    Stage(key="rank", label="Ранжирование и сборка карточек"),
]

_demo = json.loads((DATA / "demo_trends.json").read_text(encoding="utf-8"))
# ponytail: покрытие — один демо-корпус; при подключении второго направления это станет словарём
DIRECTIONS = [_demo["direction"]]
KEYWORDS = {"ии", "ai", "искусственн", "машинн", "llm", "нейросет", "языков", "интеллект"}
UPDATED_AT = datetime(2026, 9, 11, 9, 0, tzinfo=UTC)

SOURCE_TYPES = {
    "preprint": SourceType.SCIENTIFIC_PUBLICATION,
    "journal": SourceType.SCIENTIFIC_PUBLICATION,
    "patent": SourceType.PATENT,
    "vendor": SourceType.COMPANY,
    "conference": SourceType.CONFERENCE,
    "report": SourceType.ANALYTICAL_REPORT,
}


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
            SourceStat(
                name="arXiv",
                source_type=SourceType.SCIENTIFIC_PUBLICATION,
                documents=int(documents * 0.62),
            ),
            SourceStat(
                name="Отраслевые отчёты",
                source_type=SourceType.ANALYTICAL_REPORT,
                documents=int(documents * 0.18),
            ),
            SourceStat(
                name="Материалы конференций",
                source_type=SourceType.CONFERENCE,
                documents=int(documents * 0.2),
            ),
        ],
        corpus_version=CORPUS_VERSION,
        method_version=METHOD_VERSION,
        thresholds=THRESHOLDS,
        updated_at=UPDATED_AT,
    )


def classify(trend: dict) -> tuple[Bucket, str]:
    """Корзина кандидата и причина попадания. Пороги — часть версии метода."""
    f = {factor["key"]: factor["value"] for factor in trend["factors"]}
    documents, independent = trend["document_count"], trend["independent_sources"]

    if f["novelty"] < THRESHOLDS["novelty_min"]:
        return CandidateStatus.EXCLUDED, (
            f"Не прошёл проверку на новизну: {pct(f['novelty'])} при пороге "
            f"{pct(THRESHOLDS['novelty_min'])}. Тема присутствует в корпусе "
            f"с {trend['first_seen']} "
            "и развивается как устоявшееся направление, а не как зарождающееся."
        )
    if f["growth"] < THRESHOLDS["growth_min"]:
        return CandidateStatus.EXCLUDED, (
            f"Не прошёл проверку на зарождаемость: рост {pct(f['growth'])} при пороге "
            f"{pct(THRESHOLDS['growth_min'])}. Доля темы в корпусе направления не увеличивается "
            "между сопоставимыми временными окнами."
        )
    if f["evidence"] < THRESHOLDS["evidence_min"]:
        return CandidateStatus.WATCHLIST, (
            f"Доказательная база {pct(f['evidence'])} при пороге "
            f"{pct(THRESHOLDS['evidence_min'])}: "
            "в найденных материалах преобладают предположения без воспроизводимых измерений."
        )
    if independent < THRESHOLDS["independent_min"]:
        return CandidateStatus.WATCHLIST, (
            f"Подтверждения сводятся к {independent} независимым источникам при пороге "
            f"{THRESHOLDS['independent_min']}. Признаки роста есть, независимость подтверждений "
            "пока недостаточна."
        )
    if documents < THRESHOLDS["documents_min"]:
        return CandidateStatus.WATCHLIST, (
            f"В корпусе {documents} документов по теме при пороге {THRESHOLDS['documents_min']}. "
            "Выводы о динамике на такой выборке неустойчивы."
        )
    return CandidateStatus.MAIN, (
        f"Прошёл все пороги отбора: новизна {pct(f['novelty'])}, рост {pct(f['growth'])}, "
        f"доказательная база {pct(f['evidence'])}, {independent} независимых источников, "
        f"{documents} документов."
    )


def _evidence(trend: dict) -> tuple[Evidence, ...]:
    items = []
    for index, source in enumerate(trend["sources"], start=1):
        source_type = SOURCE_TYPES.get(source["source_type"], SourceType.OTHER)
        trust = (
            TrustLevel.HIGH
            if source_type
            in {
                SourceType.SCIENTIFIC_PUBLICATION,
                SourceType.PATENT,
                SourceType.REGULATOR,
                SourceType.CONFERENCE,
            }
            else TrustLevel.MEDIUM
        )
        items.append(
            Evidence(
                evidence_id=f"{trend['id']}-source-{index}",
                title=source["title"],
                url=source["url"],
                source_type=source_type,
                language="en",
                trust_level=trust,
                published_at=source.get("published_at"),
                retrieved_at=UPDATED_AT,
            )
        )
    return tuple(items)


def _features(trend: dict, evidence: tuple[Evidence, ...]) -> CandidateFeatures:
    factors = {item["key"]: item["value"] for item in trend["factors"]}
    dates = [item.published_at for item in evidence if item.published_at]
    recency = (UPDATED_AT.date() - max(dates)).days if dates else None
    promotional = sum(item.source_type is SourceType.COMPANY for item in evidence)
    return CandidateFeatures(
        stage=DevelopmentStage.UNKNOWN,
        independent_source_count=trend["independent_sources"],
        source_type_diversity=len({item.source_type for item in evidence}),
        # Демо-корпус пока не разделяет источники и участников; это явный временный proxy.
        independent_actor_count=trend["independent_sources"],
        publication_momentum=factors["growth"],
        evidence_recency_days=recency,
        promotional_source_share=promotional / len(evidence) if evidence else None,
    )


def _exclusion_reason(status: CandidateStatus, trend: dict) -> ExclusionReason | None:
    if status is not CandidateStatus.EXCLUDED:
        return None
    factors = {item["key"]: item["value"] for item in trend["factors"]}
    if factors["novelty"] < THRESHOLDS["novelty_min"]:
        return ExclusionReason.MATURE
    return ExclusionReason.IRRELEVANT


def rank(trends: list[dict], query: str | None = None) -> list[Trend]:
    """Классифицирует кандидатов по корзинам и нумерует каждую корзину отдельно."""
    ordered = sorted(trends, key=lambda t: (-score(t), t["id"]))
    classified = [[trend, *classify(trend)] for trend in ordered]

    qualified = 0
    for item in classified:
        if item[1] is not CandidateStatus.MAIN:
            continue
        qualified += 1
        # Прошедшие пороги сверх ТОП-15 не отбрасываются, а уходят в наблюдение
        if qualified > TOP_N:
            item[1] = CandidateStatus.WATCHLIST
            item[2] = (
                f"Прошёл пороги отбора, но не вошёл в ТОП-15: рейтинг {pct(score(item[0]))} ниже, "
                "чем у пятнадцатого кандидата основного списка."
            )

    counters = dict.fromkeys(BUCKETS, 0)
    result = []
    for trend, bucket, reason in classified:
        counters[bucket] += 1
        evidence = _evidence(trend)
        ranking_score = score(trend)
        result.append(
            Trend(
                candidate_id=trend["id"],
                canonical_name=trend["title"],
                query=query or _demo["direction"],
                status=bucket,
                features=_features(trend, evidence),
                prediction=ModelPrediction(
                    weak_signal_score=ranking_score,
                    model_version=MODEL_VERSION,
                    feature_version=FEATURE_VERSION,
                    calibrated=False,
                ),
                explanation=reason,
                evidence=evidence,
                exclusion_reason=_exclusion_reason(bucket, trend),
                priority_score=(
                    ranking_score * 100 if bucket is not CandidateStatus.EXCLUDED else None
                ),
                rank=counters[bucket],
                summary=trend["summary"],
                factors=trend["factors"],
                problem=trend["problem"],
                advantage=trend["advantage"],
                hypothesis=trend.get("hypothesis"),
                use_case=trend["use_case"],
                first_seen=trend["first_seen"],
                timeline=trend["timeline"],
                document_count=trend["document_count"],
                limitations=trend["limitations"],
            )
        )
    return result


async def run(query: str, on_stage: Callable[[Stage, float], None]) -> list[Trend]:
    """Прогоняет этапы, сообщая прогресс через on_stage. Возвращает отранжированных кандидатов."""
    covered = is_covered(query)
    for i, stage in enumerate(STAGES, start=1):
        on_stage(stage, i / len(STAGES))
        # ponytail: задержка имитирует работу пайплайна, чтобы фронт показывал реальные состояния
        await asyncio.sleep(0.45)
        if not covered and stage.key == "select":
            return []
    return rank(_demo["trends"], query=query)
