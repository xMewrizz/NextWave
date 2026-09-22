"""Validate final ML assessments and add presentation ranks; never run demo data."""

import asyncio
import importlib
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from nextwave.analysis import AnalysisMetadata, Analyzer
from nextwave.contracts import CandidateAssessment
from pydantic import ValidationError

from .models import BUCKETS, Coverage, Stage, Trend

TOP_N = 15
STAGES = [
    Stage(key="analyze", label="Выполнение анализа"),
    Stage(key="validate", label="Проверка результата модели"),
    Stage(key="save", label="Сохранение результата"),
]


class IntegrationUnavailable(RuntimeError):
    """No usable real analyzer is configured."""


class InvalidAssessment(RuntimeError):
    """Analyzer output does not satisfy the shared contract."""


def load_analyzer() -> tuple[Analyzer, AnalysisMetadata]:
    """NEXTWAVE_ANALYZER_FACTORY=package.module:build_analyzer (no arguments).

    Only server configuration controls the import, never a user request.
    Metadata is validated and persisted before calling analyze(query).
    """
    target = os.getenv("NEXTWAVE_ANALYZER_FACTORY", "")
    try:
        module, name = target.split(":")
        analyzer = getattr(importlib.import_module(module), name)()
        raw = analyzer.metadata
        metadata = AnalysisMetadata.model_validate(
            raw.model_dump() if isinstance(raw, AnalysisMetadata) else raw
        )
        if not callable(analyzer.analyze):
            raise TypeError("analyze must be callable")
    except Exception:  # noqa: BLE001 — factory errors may contain credentials
        # Import/validation exceptions may contain secrets; do not expose their text.
        raise IntegrationUnavailable from None
    return analyzer, metadata


def coverage() -> Coverage:
    # Global source statistics cannot be inferred from a candidate list or demo file.
    return Coverage(
        directions=[],
        examples=[],
        documents_from=None,
        documents_to=None,
        document_count=None,
        sources=[],
        corpus_version=None,
        method_version=None,
        thresholds={},
        updated_at=None,
        notice="Статистика покрытия пока недоступна. Версии сохраняются в каждом анализе.",
    )


def rank(
    payload: Sequence[CandidateAssessment | Mapping[str, Any]],
    query: str,
    metadata: AnalysisMetadata,
) -> list[Trend]:
    """Keep every final decision unchanged; UI selects the first TOP_N main ranks."""
    try:
        if not isinstance(payload, (list, tuple)):
            raise TypeError("expected a list of CandidateAssessment")
        candidates = [
            CandidateAssessment.model_validate(
                item.model_dump() if isinstance(item, CandidateAssessment) else item
            )
            for item in payload
        ]
        ids = [item.candidate_id for item in candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate candidate_id")
        for item in candidates:
            if item.query != query:
                raise ValueError("candidate query differs from analysis query")
            if (
                item.prediction.model_version != metadata.model_version
                or item.prediction.feature_version != metadata.feature_version
            ):
                raise ValueError("prediction versions differ from run metadata")
    except (ValidationError, ValueError, TypeError):
        raise InvalidAssessment from None

    # Model score first, independent evidence second, stable id last (ARCHITECTURE.md).
    ordered = sorted(
        candidates,
        key=lambda item: (
            -item.prediction.weak_signal_score,
            -item.features.independent_source_count,
            item.candidate_id,
        ),
    )
    counters = dict.fromkeys(BUCKETS, 0)
    result = []
    for item in ordered:
        counters[item.status] += 1
        result.append(Trend(**item.model_dump(), rank=counters[item.status]))
    return result


async def run(
    query: str,
    on_stage: Callable[[Stage, float], None],
    on_metadata: Callable[[AnalysisMetadata], None],
) -> list[Trend]:
    on_stage(STAGES[0], 0.0)
    analyzer, metadata = await asyncio.to_thread(load_analyzer)
    on_metadata(metadata)
    payload = await asyncio.to_thread(analyzer.analyze, query)
    on_stage(STAGES[1], 0.8)
    trends = rank(payload, query, metadata)
    on_stage(STAGES[2], 0.95)
    return trends
