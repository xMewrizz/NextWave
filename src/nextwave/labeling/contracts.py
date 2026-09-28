"""Contracts for negative examples and candidate-gate controls."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from enum import Enum, StrEnum
from typing import Any
from urllib.parse import urlparse

from ..datasets.contracts import ORGANIZER_SCOPE_KEYS

RUBRIC_VERSION = "labeling-v1"
LABELING_CUTOFF_DATE = date(2026, 9, 15)

NEGATIVE_CANDIDATE_SCHEMA_VERSION = "negative-candidate-v1"
NOISE_CONTROL_SCHEMA_VERSION = "noise-control-v1"
MODEL_DECISION_SCHEMA_VERSION = "model-label-decision-v1"
GATE_DECISION_SCHEMA_VERSION = "gate-label-decision-v1"

_NEGATIVE_ID = re.compile(r"team-negative-(\d{3})\Z")
_NOISE_ID = re.compile(r"noise-(\d{3})\Z")
_STABLE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{2,79}\Z")


class NegativeClass(StrEnum):
    MATURE = "mature"
    MARKETING_HYPE = "marketing_hype"


class NoiseType(StrEnum):
    BROAD_CONCEPT = "broad_concept"
    IRRELEVANT = "irrelevant"
    NOT_TECHNOLOGY = "not_technology"
    EXTRACTION_ERROR = "extraction_error"
    DUPLICATE = "duplicate"


class ReviewRound(StrEnum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    ADJUDICATION = "adjudication"


class ReviewStatus(StrEnum):
    DRAFT = "draft"
    REVIEWED = "reviewed"
    ADJUDICATED = "adjudicated"


class TrustLevel(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class EvidenceDirection(StrEnum):
    SUPPORT = "support"
    COUNTER = "counter"


class EvidenceKind(StrEnum):
    STANDARD_ADOPTION = "standard_adoption"
    SERIAL_DEPLOYMENT = "serial_deployment"
    ESTABLISHED_MARKET = "established_market"
    INFRASTRUCTURE_ADOPTION = "infrastructure_adoption"
    PUBLICITY_WAVE = "publicity_wave"
    TECHNICAL_VALIDATION = "technical_validation"
    PILOT = "pilot"


class SourceType(StrEnum):
    RESEARCH = "research"
    PATENT = "patent"
    STANDARD = "standard"
    REGULATOR = "regulator"
    OFFICIAL_TECHNICAL = "official_technical"
    ANALYTICAL_REPORT = "analytical_report"
    INDUSTRY_MEDIA = "industry_media"
    PRESS_RELEASE = "press_release"
    SOCIAL = "social"
    OTHER = "other"


class SearchSourceClass(StrEnum):
    SCIENTIFIC = "scientific"
    OFFICIAL = "official"
    INDUSTRY = "industry"


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def _require_stable_id(value: str, field_name: str) -> None:
    if _STABLE_ID.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase stable identifier")


def _require_http_url(value: str, field_name: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{field_name} must be an absolute HTTP(S) URL")


def _validate_identity(
    domain: str,
    analysis_scope_key: str,
    aliases: tuple[str, ...],
    canonical_name: str,
    cutoff_date: date,
) -> None:
    for name, value in (
        ("canonical_name", canonical_name),
        ("domain", domain),
        ("analysis_scope_key", analysis_scope_key),
    ):
        _require_text(value, name)
    expected_scope = ORGANIZER_SCOPE_KEYS.get(domain)
    if expected_scope is None:
        raise ValueError(f"unsupported domain: {domain}")
    if analysis_scope_key != expected_scope:
        raise ValueError("analysis_scope_key does not match domain")
    normalized = [alias.strip().casefold() for alias in aliases]
    if any(not alias for alias in normalized):
        raise ValueError("aliases must not contain blank values")
    if len(set(normalized)) != len(normalized):
        raise ValueError("aliases must be unique")
    if canonical_name.strip().casefold() in normalized:
        raise ValueError("aliases must not repeat canonical_name")
    if cutoff_date != LABELING_CUTOFF_DATE:
        raise ValueError("cutoff_date does not match labeling-v1")


def _validate_review_round(review_round: ReviewRound, status: ReviewStatus) -> None:
    if review_round is ReviewRound.ADJUDICATION:
        if status is not ReviewStatus.ADJUDICATED:
            raise ValueError("adjudication round must have adjudicated status")
    elif status is ReviewStatus.ADJUDICATED:
        raise ValueError("only adjudication round may have adjudicated status")


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


@dataclass(frozen=True, slots=True)
class NegativeCandidateRecord:
    candidate_id: str
    canonical_name: str
    aliases: tuple[str, ...]
    group_id: str
    source_query: str
    domain: str
    analysis_scope_key: str
    cutoff_date: date
    schema_version: str = field(default=NEGATIVE_CANDIDATE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        if _NEGATIVE_ID.fullmatch(self.candidate_id) is None:
            raise ValueError("candidate_id must match team-negative-NNN")
        _require_stable_id(self.group_id, "group_id")
        _require_text(self.source_query, "source_query")
        _validate_identity(
            self.domain,
            self.analysis_scope_key,
            self.aliases,
            self.canonical_name,
            self.cutoff_date,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "canonical_name": self.canonical_name,
            "aliases": list(self.aliases),
            "group_id": self.group_id,
            "source_query": self.source_query,
            "domain": self.domain,
            "analysis_scope_key": self.analysis_scope_key,
            "cutoff_date": self.cutoff_date.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class NoiseControlRecord:
    noise_id: str
    source_query: str
    extracted_text: str
    source_document_url: str
    domain: str
    analysis_scope_key: str
    cutoff_date: date
    duplicate_of_candidate_id: str | None = None
    schema_version: str = field(default=NOISE_CONTROL_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        if _NOISE_ID.fullmatch(self.noise_id) is None:
            raise ValueError("noise_id must match noise-NNN")
        _require_text(self.source_query, "source_query")
        _require_text(self.extracted_text, "extracted_text")
        _require_http_url(self.source_document_url, "source_document_url")
        _validate_identity(
            self.domain,
            self.analysis_scope_key,
            (),
            self.extracted_text,
            self.cutoff_date,
        )
        if self.duplicate_of_candidate_id is not None:
            _require_stable_id(self.duplicate_of_candidate_id, "duplicate_of_candidate_id")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True, slots=True)
class LabelEvidence:
    evidence_id: str
    direction: EvidenceDirection
    kind: EvidenceKind
    source_type: SourceType
    trust_level: TrustLevel
    title: str
    url: str
    published_at: date
    organization: str
    origin_id: str
    claim: str
    locator: str

    def __post_init__(self) -> None:
        _require_stable_id(self.evidence_id, "evidence_id")
        _require_http_url(self.url, "url")
        _require_stable_id(self.origin_id, "origin_id")
        for name, value in (
            ("title", self.title),
            ("organization", self.organization),
            ("claim", self.claim),
            ("locator", self.locator),
        ):
            _require_text(value, name)


@dataclass(frozen=True, slots=True)
class SearchCoverage:
    queries: tuple[str, ...]
    source_classes: tuple[SearchSourceClass, ...]
    searched_at: date
    notes: str

    def __post_init__(self) -> None:
        if not self.queries:
            raise ValueError("search coverage must contain at least one query")
        normalized = [query.strip().casefold() for query in self.queries]
        if any(not query for query in normalized):
            raise ValueError("search queries must not be blank")
        if len(set(normalized)) != len(normalized):
            raise ValueError("search queries must be unique")
        if len(set(self.source_classes)) != len(self.source_classes):
            raise ValueError("search source classes must be unique")
        _require_text(self.notes, "notes")


_MATURE_EVIDENCE_KINDS = {
    EvidenceKind.STANDARD_ADOPTION,
    EvidenceKind.SERIAL_DEPLOYMENT,
    EvidenceKind.ESTABLISHED_MARKET,
    EvidenceKind.INFRASTRUCTURE_ADOPTION,
}
_TECHNICAL_EVIDENCE_KINDS = {
    EvidenceKind.TECHNICAL_VALIDATION,
    EvidenceKind.PILOT,
}
_REQUIRED_SEARCH_CLASSES = {
    SearchSourceClass.SCIENTIFIC,
    SearchSourceClass.INDUSTRY,
}


@dataclass(frozen=True, slots=True)
class ModelLabelDecision:
    decision_id: str
    candidate_id: str
    review_round: ReviewRound
    status: ReviewStatus
    label: NegativeClass
    rationale: str
    reviewer_id: str
    annotated_at: date
    cutoff_date: date
    evidence: tuple[LabelEvidence, ...]
    search_coverage: SearchCoverage | None = None
    rubric_version: str = field(default=RUBRIC_VERSION, init=False)
    schema_version: str = field(default=MODEL_DECISION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        _require_stable_id(self.decision_id, "decision_id")
        if _NEGATIVE_ID.fullmatch(self.candidate_id) is None:
            raise ValueError("candidate_id must match team-negative-NNN")
        _validate_review_round(self.review_round, self.status)
        _require_text(self.rationale, "rationale")
        _require_stable_id(self.reviewer_id, "reviewer_id")
        if self.cutoff_date != LABELING_CUTOFF_DATE:
            raise ValueError("cutoff_date does not match labeling-v1")
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("evidence IDs must be unique within a decision")
        if any(item.published_at > self.cutoff_date for item in self.evidence):
            raise ValueError("evidence published after cutoff is not allowed")
        if self.status is not ReviewStatus.DRAFT:
            self._validate_complete_decision()

    def _validate_complete_decision(self) -> None:
        supporting = [
            item for item in self.evidence if item.direction is EvidenceDirection.SUPPORT
        ]
        if self.label is NegativeClass.MATURE:
            qualifies = any(
                item.kind in _MATURE_EVIDENCE_KINDS
                and item.trust_level in {TrustLevel.A, TrustLevel.B}
                for item in supporting
            )
            if not qualifies:
                raise ValueError("reviewed mature decision requires A/B maturity evidence")
            return

        wave_start = self.cutoff_date - timedelta(days=365)
        wave = [
            item
            for item in supporting
            if item.kind is EvidenceKind.PUBLICITY_WAVE
            and wave_start <= item.published_at <= self.cutoff_date
        ]
        if len(wave) < 3 or len({item.origin_id for item in wave}) < 2:
            raise ValueError("reviewed marketing_hype requires a documented publicity wave")
        if self.search_coverage is None:
            raise ValueError("reviewed marketing_hype requires search coverage")
        if not _REQUIRED_SEARCH_CLASSES <= set(self.search_coverage.source_classes):
            raise ValueError(
                "search coverage must include scientific and industry sources"
            )
        technical_origins = {
            item.origin_id
            for item in supporting
            if item.kind in _TECHNICAL_EVIDENCE_KINDS
            and item.trust_level in {TrustLevel.A, TrustLevel.B}
        }
        if len(technical_origins) >= 2:
            raise ValueError("marketing_hype cannot have two independent A/B technical origins")
        if any(
            item.kind in {EvidenceKind.PILOT, EvidenceKind.SERIAL_DEPLOYMENT}
            for item in supporting
        ):
            raise ValueError(
                "marketing_hype cannot have confirmed pilot or deployment evidence"
            )
        if any(
            item.kind in _MATURE_EVIDENCE_KINDS
            and item.trust_level in {TrustLevel.A, TrustLevel.B}
            for item in supporting
        ):
            raise ValueError("marketing_hype cannot have confirmed A/B maturity evidence")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True, slots=True)
class GateLabelDecision:
    decision_id: str
    noise_id: str
    review_round: ReviewRound
    status: ReviewStatus
    noise_type: NoiseType
    rationale: str
    reviewer_id: str
    annotated_at: date
    rubric_version: str = field(default=RUBRIC_VERSION, init=False)
    schema_version: str = field(default=GATE_DECISION_SCHEMA_VERSION, init=False)
    outcome: str = field(default="reject", init=False)

    def __post_init__(self) -> None:
        _require_stable_id(self.decision_id, "decision_id")
        if _NOISE_ID.fullmatch(self.noise_id) is None:
            raise ValueError("noise_id must match noise-NNN")
        _validate_review_round(self.review_round, self.status)
        _require_text(self.rationale, "rationale")
        _require_stable_id(self.reviewer_id, "reviewer_id")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))
