"""Candidate-level contracts shared by ML and application components."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from urllib.parse import urlparse


class SourceType(StrEnum):
    SCIENTIFIC_PUBLICATION = "scientific_publication"
    PATENT = "patent"
    REGULATOR = "regulator"
    UNIVERSITY = "university"
    COMPANY = "company"
    INDUSTRY_MEDIA = "industry_media"
    ANALYTICAL_REPORT = "analytical_report"
    CONFERENCE = "conference"
    SOCIAL_OR_BLOG = "social_or_blog"
    OTHER = "other"


class TrustLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class DevelopmentStage(StrEnum):
    RESEARCH = "research"
    PROTOTYPE = "prototype"
    PILOT = "pilot"
    EARLY_ADOPTION = "early_adoption"
    MASS_ADOPTION = "mass_adoption"
    UNKNOWN = "unknown"


class CandidateStatus(StrEnum):
    MAIN = "main"
    WATCHLIST = "watchlist"
    EXCLUDED = "excluded"


class ExclusionReason(StrEnum):
    MATURE = "mature"
    MASS_ADOPTION = "mass_adoption"
    INDUSTRY_STANDARD = "industry_standard"
    MARKETING_HYPE = "marketing_hype"
    INSUFFICIENT_TRUST = "insufficient_trust"
    IRRELEVANT = "irrelevant"
    DUPLICATE = "duplicate"


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def _require_non_negative(value: int | float | None, field_name: str) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _require_ratio(value: float | None, field_name: str) -> None:
    if value is not None and not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class Evidence:
    evidence_id: str
    title: str
    url: str
    source_type: SourceType
    language: str
    trust_level: TrustLevel
    published_at: date | None = None
    retrieved_at: datetime | None = None
    excerpt: str | None = None
    generated_summary: bool = False

    def __post_init__(self) -> None:
        _require_text(self.evidence_id, "evidence_id")
        _require_text(self.title, "title")
        _require_text(self.language, "language")
        parsed = urlparse(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("url must be an absolute HTTP or HTTPS URL")


@dataclass(frozen=True, slots=True)
class CandidateFeatures:
    stage: DevelopmentStage
    independent_source_count: int
    source_type_diversity: int
    independent_actor_count: int
    publication_momentum: float | None = None
    patent_momentum: float | None = None
    evidence_recency_days: int | None = None
    mass_adoption: bool | None = None
    formed_market: bool | None = None
    industry_standard: bool | None = None
    promotional_source_share: float | None = None

    def __post_init__(self) -> None:
        _require_non_negative(self.independent_source_count, "independent_source_count")
        _require_non_negative(self.source_type_diversity, "source_type_diversity")
        _require_non_negative(self.independent_actor_count, "independent_actor_count")
        _require_non_negative(self.evidence_recency_days, "evidence_recency_days")
        _require_ratio(self.promotional_source_share, "promotional_source_share")


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    weak_signal_score: float
    model_version: str
    feature_version: str
    calibrated: bool = False

    def __post_init__(self) -> None:
        _require_ratio(self.weak_signal_score, "weak_signal_score")
        _require_text(self.model_version, "model_version")
        _require_text(self.feature_version, "feature_version")


@dataclass(frozen=True, slots=True)
class CandidateAssessment:
    candidate_id: str
    canonical_name: str
    query: str
    status: CandidateStatus
    features: CandidateFeatures
    prediction: ModelPrediction
    explanation: str
    evidence: tuple[Evidence, ...] = field(default_factory=tuple)
    exclusion_reason: ExclusionReason | None = None
    priority_score: float | None = None

    def __post_init__(self) -> None:
        _require_text(self.candidate_id, "candidate_id")
        _require_text(self.canonical_name, "canonical_name")
        _require_text(self.query, "query")
        _require_text(self.explanation, "explanation")

        if self.status is CandidateStatus.EXCLUDED and self.exclusion_reason is None:
            raise ValueError("excluded candidates require an exclusion_reason")
        if self.status is not CandidateStatus.EXCLUDED and self.exclusion_reason is not None:
            raise ValueError("only excluded candidates may have an exclusion_reason")
        if self.status is CandidateStatus.MAIN and not self.evidence:
            raise ValueError("main candidates require at least one evidence item")
        if self.priority_score is not None and not 0.0 <= self.priority_score <= 100.0:
            raise ValueError("priority_score must be between 0 and 100")
