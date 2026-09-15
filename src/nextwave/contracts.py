"""Canonical candidate contracts shared by ML, backend and API layers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SourceType(StrEnum):
    SCIENTIFIC_PUBLICATION = "scientific_publication"
    PATENT = "patent"
    STANDARD = "standard"
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


class TrustTier(StrEnum):
    """Source-ingestion trust grade retained for connector compatibility."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
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


class ContractModel(BaseModel):
    """Strict base for data that crosses component boundaries."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """Normalized document emitted by source connectors before candidate scoring."""

    document_id: str
    connector_id: str
    external_id: str
    snapshot_id: str
    title: str
    url: str
    canonical_url: str
    source_type: SourceType
    language: str
    trust_tier: TrustTier
    origin_id: str
    doi: str | None = None
    authors: tuple[str, ...] = ()
    organizations: tuple[str, ...] = ()
    published_at: date | None = None
    observed_at: datetime | None = None
    retrieved_at: datetime | None = None
    publisher: str | None = None
    excerpt: str | None = None
    automatic_translation: bool = False
    generated_summary: bool = False
    origin_method: str | None = None
    origin_confidence: float | None = None

    def __post_init__(self) -> None:
        required = {
            "document_id": self.document_id,
            "connector_id": self.connector_id,
            "external_id": self.external_id,
            "snapshot_id": self.snapshot_id,
            "title": self.title,
            "language": self.language,
            "origin_id": self.origin_id,
        }
        for name, value in required.items():
            if not value.strip():
                raise ValueError(f"{name} must not be blank")

        for name, value in (("url", self.url), ("canonical_url", self.canonical_url)):
            parsed = urlparse(value)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(f"{name} must be an absolute HTTP or HTTPS URL")

        if not isinstance(self.source_type, SourceType):
            raise ValueError("source_type must be a SourceType")
        if not isinstance(self.trust_tier, TrustTier):
            raise ValueError("trust_tier must be a TrustTier")
        if self.origin_confidence is not None and not 0 <= self.origin_confidence <= 1:
            raise ValueError("origin_confidence must be between 0 and 1")
        for name, value in (("observed_at", self.observed_at), ("retrieved_at", self.retrieved_at)):
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError(f"{name} must include a timezone")


class Evidence(ContractModel):
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

    @field_validator("evidence_id", "title", "language")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("url")
    @classmethod
    def require_absolute_http_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("url must be an absolute HTTP or HTTPS URL")
        return value


class CandidateFeatures(ContractModel):
    stage: DevelopmentStage
    independent_source_count: int = Field(ge=0)
    source_type_diversity: int = Field(ge=0)
    independent_actor_count: int = Field(ge=0)
    publication_momentum: float | None = None
    patent_momentum: float | None = None
    evidence_recency_days: int | None = Field(default=None, ge=0)
    mass_adoption: bool | None = None
    formed_market: bool | None = None
    industry_standard: bool | None = None
    promotional_source_share: float | None = Field(default=None, ge=0, le=1)


class ModelPrediction(ContractModel):
    weak_signal_score: float = Field(ge=0, le=1)
    model_version: str
    feature_version: str
    calibrated: bool = False

    @field_validator("model_version", "feature_version")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value:
            raise ValueError("must not be blank")
        return value


class CandidateAssessment(ContractModel):
    candidate_id: str
    canonical_name: str
    query: str
    status: CandidateStatus
    features: CandidateFeatures
    prediction: ModelPrediction
    explanation: str
    evidence: tuple[Evidence, ...] = ()
    exclusion_reason: ExclusionReason | None = None
    priority_score: float | None = Field(default=None, ge=0, le=100)

    @field_validator("candidate_id", "canonical_name", "query", "explanation")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def validate_decision(self) -> CandidateAssessment:
        if self.status is CandidateStatus.EXCLUDED and self.exclusion_reason is None:
            raise ValueError("excluded candidates require an exclusion_reason")
        if self.status is not CandidateStatus.EXCLUDED and self.exclusion_reason is not None:
            raise ValueError("only excluded candidates may have an exclusion_reason")
        if self.status is CandidateStatus.MAIN and not self.evidence:
            raise ValueError("main candidates require at least one evidence item")
        return self
