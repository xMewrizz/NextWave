"""Domain contracts shared by training, inference and API integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from urllib.parse import urlparse


class SourceType(StrEnum):
    SCIENTIFIC_PUBLICATION = "scientific_publication"
    PATENT = "patent"
    STANDARD = "standard"
    REGULATOR = "regulator"
    UNIVERSITY = "university"
    COMPANY_TECHNICAL = "company_technical"
    ANALYTICAL_REPORT = "analytical_report"
    INDUSTRY_MEDIA = "industry_media"
    PRESS_RELEASE = "press_release"
    SOCIAL_OR_BLOG = "social_or_blog"
    OTHER = "other"


class TrustTier(StrEnum):
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


class ClaimType(StrEnum):
    NOVELTY = "novelty"
    GROWTH = "growth"
    RESEARCH = "research"
    PATENT = "patent"
    PROTOTYPE = "prototype"
    PILOT = "pilot"
    INVESTMENT = "investment"
    ADOPTION = "adoption"
    STANDARD = "standard"
    MARKET = "market"
    PROMOTIONAL_CLAIM = "promotional_claim"


class EvidenceDirection(StrEnum):
    SUPPORT = "support"
    COUNTER = "counter"


class CandidateStatus(StrEnum):
    MAIN = "main"
    WATCHLIST = "watchlist"
    EXCLUDED = "excluded"


class ExclusionReason(StrEnum):
    LOW_RELEVANCE = "low_relevance"
    NOT_A_TECHNOLOGY = "not_a_technology"
    DUPLICATE = "duplicate"
    MATURE = "mature"
    MASS_ADOPTION = "mass_adoption"
    INDUSTRY_STANDARD = "industry_standard"
    MARKETING_HYPE = "marketing_hype"
    SOURCE_ECHO = "source_echo"
    MODEL_BELOW_THRESHOLD = "model_below_threshold"


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def _require_non_negative(value: int | float | None, field_name: str) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _require_ratio(value: float | None, field_name: str) -> None:
    if value is not None and not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")


def _require_signed_ratio(value: float | None, field_name: str) -> None:
    if value is not None and not -1.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be between -1 and 1")


def _require_url(value: str, field_name: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{field_name} must be an absolute HTTP or HTTPS URL")


@dataclass(frozen=True, slots=True)
class SourceDocument:
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
    published_at: date | None = None
    retrieved_at: datetime | None = None
    publisher: str | None = None
    excerpt: str | None = None
    automatic_translation: bool = False
    generated_summary: bool = False
    origin_method: str | None = None
    origin_confidence: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("document_id", self.document_id),
            ("connector_id", self.connector_id),
            ("external_id", self.external_id),
            ("snapshot_id", self.snapshot_id),
            ("title", self.title),
            ("language", self.language),
            ("origin_id", self.origin_id),
        ):
            _require_text(value, name)
        _require_url(self.url, "url")
        _require_url(self.canonical_url, "canonical_url")
        _require_ratio(self.origin_confidence, "origin_confidence")
        if self.origin_method is not None:
            _require_text(self.origin_method, "origin_method")


@dataclass(frozen=True, slots=True)
class EvidenceClaim:
    claim_id: str
    document_id: str
    claim_type: ClaimType
    direction: EvidenceDirection
    text: str
    locator: str | None = None
    organization: str | None = None
    extraction_confidence: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("claim_id", self.claim_id),
            ("document_id", self.document_id),
            ("text", self.text),
        ):
            _require_text(value, name)
        _require_ratio(self.extraction_confidence, "extraction_confidence")
        if self.locator is not None:
            _require_text(self.locator, "locator")


@dataclass(frozen=True, slots=True)
class TemporalFeatures:
    source_id: str
    analysis_scope_id: str
    previous_document_count: int | None
    recent_document_count: int | None
    previous_origin_count: int | None
    recent_origin_count: int | None
    previous_query_share: float | None = None
    recent_query_share: float | None = None
    query_share_change: float | None = None
    first_seen_at: date | None = None
    coverage_complete: bool = False
    burst_score: float | None = None

    def __post_init__(self) -> None:
        _require_text(self.source_id, "source_id")
        _require_text(self.analysis_scope_id, "analysis_scope_id")
        for name, value in (
            ("previous_document_count", self.previous_document_count),
            ("recent_document_count", self.recent_document_count),
            ("previous_origin_count", self.previous_origin_count),
            ("recent_origin_count", self.recent_origin_count),
        ):
            _require_non_negative(value, name)
        _require_ratio(self.previous_query_share, "previous_query_share")
        _require_ratio(self.recent_query_share, "recent_query_share")
        _require_signed_ratio(self.query_share_change, "query_share_change")
        _require_non_negative(self.burst_score, "burst_score")


@dataclass(frozen=True, slots=True)
class CandidateFeatures:
    stage: DevelopmentStage
    temporal: TemporalFeatures
    query_relevance: float
    independent_origin_count: int
    source_type_diversity: int
    independent_actor_count: int
    echo_share: float | None = None
    promotional_source_share: float | None = None
    mass_adoption: bool | None = None
    formed_market: bool | None = None
    industry_standard: bool | None = None

    def __post_init__(self) -> None:
        _require_ratio(self.query_relevance, "query_relevance")
        _require_non_negative(self.independent_origin_count, "independent_origin_count")
        _require_non_negative(self.source_type_diversity, "source_type_diversity")
        _require_non_negative(self.independent_actor_count, "independent_actor_count")
        _require_ratio(self.echo_share, "echo_share")
        _require_ratio(self.promotional_source_share, "promotional_source_share")


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    model_score: float
    decision_threshold: float
    model_version: str
    feature_version: str
    calibrated: bool = False

    def __post_init__(self) -> None:
        _require_ratio(self.model_score, "model_score")
        _require_ratio(self.decision_threshold, "decision_threshold")
        _require_text(self.model_version, "model_version")
        _require_text(self.feature_version, "feature_version")


@dataclass(frozen=True, slots=True)
class CandidateAssessment:
    candidate_id: str
    group_id: str
    canonical_name: str
    aliases: tuple[str, ...]
    query: str
    analysis_scope_id: str
    cutoff_date: date
    status: CandidateStatus
    features: CandidateFeatures
    prediction: ModelPrediction | None
    explanation: str
    documents: tuple[SourceDocument, ...] = field(default_factory=tuple)
    claims: tuple[EvidenceClaim, ...] = field(default_factory=tuple)
    exclusion_reason: ExclusionReason | None = None
    rank: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("candidate_id", self.candidate_id),
            ("group_id", self.group_id),
            ("canonical_name", self.canonical_name),
            ("query", self.query),
            ("analysis_scope_id", self.analysis_scope_id),
            ("explanation", self.explanation),
        ):
            _require_text(value, name)

        if len(set(self.aliases)) != len(self.aliases):
            raise ValueError("aliases must be unique")
        for alias in self.aliases:
            _require_text(alias, "alias")
        if self.features.temporal.analysis_scope_id != self.analysis_scope_id:
            raise ValueError("temporal features must use the assessment analysis_scope_id")

        documents_by_id = {document.document_id: document for document in self.documents}
        if len(documents_by_id) != len(self.documents):
            raise ValueError("document_id must be unique within an assessment")

        claim_ids = {claim.claim_id for claim in self.claims}
        if len(claim_ids) != len(self.claims):
            raise ValueError("claim_id must be unique within an assessment")
        if any(claim.document_id not in documents_by_id for claim in self.claims):
            raise ValueError("every claim must reference a document in the assessment")
        if any(
            document.published_at is not None and document.published_at > self.cutoff_date
            for document in self.documents
        ):
            raise ValueError("documents published after cutoff_date are not allowed")
        if (
            self.features.temporal.first_seen_at is not None
            and self.features.temporal.first_seen_at > self.cutoff_date
        ):
            raise ValueError("first_seen_at must not be after cutoff_date")

        if self.status is CandidateStatus.EXCLUDED and self.exclusion_reason is None:
            raise ValueError("excluded candidates require an exclusion_reason")
        if self.status is not CandidateStatus.EXCLUDED and self.exclusion_reason is not None:
            raise ValueError("only excluded candidates may have an exclusion_reason")
        if self.status is CandidateStatus.MAIN:
            if self.prediction is None:
                raise ValueError("main candidates require a model prediction")
            if self.prediction.model_score < self.prediction.decision_threshold:
                raise ValueError("main candidates must pass the model threshold")
            if self.rank is None or self.rank < 1:
                raise ValueError("main candidates require a positive rank")
            if not self.features.temporal.coverage_complete:
                raise ValueError("main candidates require complete temporal coverage")
            if self.features.independent_origin_count < 2:
                raise ValueError("main candidates require at least two independent origins")
            if self.features.independent_actor_count < 2:
                raise ValueError("main candidates require at least two independent actors")
            if any(
                flag is True
                for flag in (
                    self.features.mass_adoption,
                    self.features.formed_market,
                    self.features.industry_standard,
                )
            ):
                raise ValueError("main candidates must not have critical maturity evidence")

            grounded_support = [
                claim
                for claim in self.claims
                if claim.direction is EvidenceDirection.SUPPORT
                and claim.locator is not None
                and documents_by_id[claim.document_id].trust_tier in {TrustTier.A, TrustTier.B}
            ]
            if not grounded_support:
                raise ValueError("main candidates require grounded A/B supporting evidence")
        elif self.rank is not None:
            raise ValueError("only main candidates may have a rank")
