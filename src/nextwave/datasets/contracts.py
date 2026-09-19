"""Contracts for deterministic organizer dataset artifacts."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any

POSITIVE_SCHEMA_VERSION = "organizer-positive-v1"
ANNOTATION_SCHEMA_VERSION = "organizer-annotation-v1"
MANIFEST_SCHEMA_VERSION = "organizer-manifest-v1"

ORGANIZER_SCOPE_KEYS = {
    "Edge": "edge-v1",
    "Защита ИИ": "ai-security-v1",
    "Индустриальный ИИ": "industrial-ai-v1",
    "Инфраструктура ИИ": "ai-infrastructure-v1",
    "Роботы": "robotics-v1",
    "Финтех": "fintech-v1",
}

_RECORD_ID = re.compile(r"organizer-(\d{3})\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class IdentityStatus(StrEnum):
    """Review state of aliases and grouping, independent of the class label."""

    PENDING_REVIEW = "pending_review"
    REVIEWED = "reviewed"


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def _record_number(record_id: str) -> int:
    match = _RECORD_ID.fullmatch(record_id)
    if match is None:
        raise ValueError("record_id must match organizer-NNN")
    return int(match.group(1))


@dataclass(frozen=True, slots=True)
class PositiveCandidateRecord:
    """Leakage-safe identity and label for one organizer row."""

    record_id: str
    source_row: int
    canonical_name: str
    domain: str
    analysis_scope_key: str
    aliases: tuple[str, ...]
    group_id: str
    identity_status: IdentityStatus
    cutoff_date: date
    schema_version: str = field(default=POSITIVE_SCHEMA_VERSION, init=False)
    label: str = field(default="weak_signal", init=False)
    target: int = field(default=1, init=False)
    label_origin: str = field(default="organizer_confirmed", init=False)

    def __post_init__(self) -> None:
        number = _record_number(self.record_id)
        if self.source_row < 3:
            raise ValueError("source_row must point to a data row")
        if number != self.source_row - 2:
            raise ValueError("record_id number must match source_row")
        for name, value in (
            ("canonical_name", self.canonical_name),
            ("domain", self.domain),
            ("analysis_scope_key", self.analysis_scope_key),
            ("group_id", self.group_id),
        ):
            _require_text(value, name)
        expected_scope = ORGANIZER_SCOPE_KEYS.get(self.domain)
        if expected_scope is None:
            raise ValueError(f"unsupported organizer domain: {self.domain}")
        if self.analysis_scope_key != expected_scope:
            raise ValueError("analysis_scope_key does not match domain")
        if not isinstance(self.identity_status, IdentityStatus):
            raise ValueError("identity_status must be an IdentityStatus")
        normalized_aliases = [alias.strip().casefold() for alias in self.aliases]
        if any(not alias for alias in normalized_aliases):
            raise ValueError("aliases must not contain blank values")
        if len(set(normalized_aliases)) != len(normalized_aliases):
            raise ValueError("aliases must be unique")
        if self.canonical_name.strip().casefold() in normalized_aliases:
            raise ValueError("aliases must not repeat canonical_name")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "source_row": self.source_row,
            "canonical_name": self.canonical_name,
            "domain": self.domain,
            "analysis_scope_key": self.analysis_scope_key,
            "aliases": list(self.aliases),
            "group_id": self.group_id,
            "identity_status": self.identity_status.value,
            "label": self.label,
            "target": self.target,
            "label_origin": self.label_origin,
            "cutoff_date": self.cutoff_date.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class OrganizerAnnotationRecord:
    """Organizer-provided expert fields kept outside model input."""

    record_id: str
    source_number: int
    companies_raw: str
    expert_rationale: str
    expert_stage: str
    expert_mention_trend: str
    expert_score: int
    sources_raw: str
    schema_version: str = field(default=ANNOTATION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        number = _record_number(self.record_id)
        if self.source_number != number:
            raise ValueError("source_number must match record_id")
        for name, value in (
            ("companies_raw", self.companies_raw),
            ("expert_rationale", self.expert_rationale),
            ("expert_stage", self.expert_stage),
            ("expert_mention_trend", self.expert_mention_trend),
            ("sources_raw", self.sources_raw),
        ):
            _require_text(value, name)
        if not 3 <= self.expert_score <= 7:
            raise ValueError("expert_score must be between 3 and 7")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "source_number": self.source_number,
            "companies_raw": self.companies_raw,
            "expert_rationale": self.expert_rationale,
            "expert_stage": self.expert_stage,
            "expert_mention_trend": self.expert_mention_trend,
            "expert_score": self.expert_score,
            "sources_raw": self.sources_raw,
        }


@dataclass(frozen=True, slots=True)
class FileDigest:
    """Name, byte size and content digest of an input or output file."""

    filename: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _require_text(self.filename, "filename")
        if "/" in self.filename or "\\" in self.filename:
            raise ValueError("filename must not contain a directory")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("sha256 must contain 64 lowercase hexadecimal characters")


@dataclass(frozen=True, slots=True)
class HeaderMapping:
    """Explicit mapping from an XLSX header to an artifact field."""

    source_header: str
    output_field: str

    def __post_init__(self) -> None:
        _require_text(self.source_header, "source_header")
        _require_text(self.output_field, "output_field")


@dataclass(frozen=True, slots=True)
class OrganizerDatasetManifest:
    """Lineage and validation summary for one deterministic adapter run."""

    dataset_version: str
    adapter_version: str
    cutoff_date: date
    source: FileDigest
    sheet_name: str
    header_row: int
    data_start_row: int
    data_end_row: int
    input_record_count: int
    accepted_record_count: int
    rejected_record_count: int
    header_mapping: tuple[HeaderMapping, ...]
    outputs: tuple[FileDigest, ...]
    validation_errors: tuple[str, ...] = ()
    schema_version: str = field(default=MANIFEST_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("dataset_version", self.dataset_version),
            ("adapter_version", self.adapter_version),
            ("sheet_name", self.sheet_name),
        ):
            _require_text(value, name)
        if self.header_row < 1:
            raise ValueError("header_row must be positive")
        if self.data_start_row <= self.header_row:
            raise ValueError("data_start_row must be after header_row")
        if self.data_end_row < self.data_start_row:
            raise ValueError("data_end_row must not precede data_start_row")
        counts = (
            self.input_record_count,
            self.accepted_record_count,
            self.rejected_record_count,
        )
        if any(count < 0 for count in counts):
            raise ValueError("record counts must be non-negative")
        if self.input_record_count != self.accepted_record_count + self.rejected_record_count:
            raise ValueError("accepted and rejected counts must add up to input count")
        source_headers = [item.source_header for item in self.header_mapping]
        output_fields = [item.output_field for item in self.header_mapping]
        if len(set(source_headers)) != len(source_headers):
            raise ValueError("source headers must be unique")
        if len(set(output_fields)) != len(output_fields):
            raise ValueError("mapped output fields must be unique")
        output_names = [item.filename for item in self.outputs]
        if len(set(output_names)) != len(output_names):
            raise ValueError("output filenames must be unique")
        for error in self.validation_errors:
            _require_text(error, "validation_error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_version": self.dataset_version,
            "adapter_version": self.adapter_version,
            "cutoff_date": self.cutoff_date.isoformat(),
            "source": asdict(self.source),
            "sheet_name": self.sheet_name,
            "header_row": self.header_row,
            "data_start_row": self.data_start_row,
            "data_end_row": self.data_end_row,
            "input_record_count": self.input_record_count,
            "accepted_record_count": self.accepted_record_count,
            "rejected_record_count": self.rejected_record_count,
            "header_mapping": [asdict(item) for item in self.header_mapping],
            "outputs": [asdict(item) for item in self.outputs],
            "validation_errors": list(self.validation_errors),
        }
