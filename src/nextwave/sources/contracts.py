"""Contracts for source queries, connector runs and immutable snapshots."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum, StrEnum
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse

SOURCE_QUERY_SCHEMA_VERSION = "source-query-v1"
CONNECTOR_REQUEST_SCHEMA_VERSION = "connector-request-v1"
SNAPSHOT_MANIFEST_SCHEMA_VERSION = "source-snapshot-v1"

_STABLE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{2,99}\Z")
_PARAMETER_NAME = re.compile(r"[a-zA-Z][a-zA-Z0-9_.-]{0,79}\Z")
_LANGUAGE = re.compile(r"[a-z]{2,3}(?:-[a-z0-9]{2,8})*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ConnectorId(StrEnum):
    OPENALEX = "openalex"
    CROSSREF = "crossref"
    GDELT = "gdelt"
    MEDIACLOUD = "mediacloud"


class QueryPurpose(StrEnum):
    DISCOVERY = "discovery"
    HISTORICAL_ENRICHMENT = "historical_enrichment"
    METADATA_RESOLUTION = "metadata_resolution"


class RetrievalChannel(StrEnum):
    TEXT = "text"
    SEMANTIC = "semantic"
    TAXONOMY = "taxonomy"
    IDENTIFIER = "identifier"


class ConnectorStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class SnapshotStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def _require_stable_id(value: str, field_name: str) -> None:
    if _STABLE_ID.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase stable identifier")


def _require_enum(value: Enum, enum_type: type[Enum], field_name: str) -> None:
    if not isinstance(value, enum_type):
        raise ValueError(f"{field_name} must be a {enum_type.__name__}")


def _require_aware_datetime(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")


def _require_https_url(value: str, field_name: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"{field_name} must be an absolute HTTPS URL")


def _require_unique_text(values: tuple[str, ...], field_name: str) -> None:
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    normalized = [value.strip().casefold() for value in values]
    if any(not value for value in normalized):
        raise ValueError(f"{field_name} must not contain blank values")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must contain unique values")


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


@dataclass(frozen=True, slots=True)
class SourceQuery:
    """Provider-independent meaning and time boundary of one source search."""

    query_id: str
    analysis_scope_id: str
    purpose: QueryPurpose
    raw_query: str
    normalized_query: str
    search_texts: tuple[str, ...]
    published_from: date
    published_until: date
    cutoff_date: date
    languages: tuple[str, ...] = ("en",)
    topic_ids: tuple[str, ...] = ()
    subfield_ids: tuple[str, ...] = ()
    schema_version: str = field(default=SOURCE_QUERY_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        _require_stable_id(self.query_id, "query_id")
        _require_stable_id(self.analysis_scope_id, "analysis_scope_id")
        _require_enum(self.purpose, QueryPurpose, "purpose")
        _require_text(self.raw_query, "raw_query")
        if len(self.raw_query) > 200:
            raise ValueError("raw_query must not exceed 200 characters")
        _require_text(self.normalized_query, "normalized_query")
        _require_unique_text(self.search_texts, "search_texts")
        _require_unique_text(self.languages, "languages")
        if any(_LANGUAGE.fullmatch(language) is None for language in self.languages):
            raise ValueError("languages must contain lowercase language tags")
        if self.published_from > self.published_until:
            raise ValueError("published_from must not be after published_until")
        if self.published_until > self.cutoff_date:
            raise ValueError("published_until must not be after cutoff_date")
        normalized_topics = [topic_id.strip().casefold() for topic_id in self.topic_ids]
        if any(not topic_id for topic_id in normalized_topics):
            raise ValueError("topic_ids must not contain blank values")
        if len(set(normalized_topics)) != len(normalized_topics):
            raise ValueError("topic_ids must contain unique values")
        normalized_subfields = [
            subfield_id.strip().casefold() for subfield_id in self.subfield_ids
        ]
        if any(not subfield_id for subfield_id in normalized_subfields):
            raise ValueError("subfield_ids must not contain blank values")
        if len(set(normalized_subfields)) != len(normalized_subfields):
            raise ValueError("subfield_ids must contain unique values")
        if self.topic_ids and self.subfield_ids:
            raise ValueError("topic_ids and subfield_ids are mutually exclusive")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True, slots=True)
class QueryParameter:
    """One exact query-string parameter sent to a source API."""

    name: str
    value: str

    def __post_init__(self) -> None:
        if _PARAMETER_NAME.fullmatch(self.name) is None:
            raise ValueError("parameter name contains unsupported characters")
        _require_text(self.value, "parameter value")


@dataclass(frozen=True, slots=True)
class ConnectorRequest:
    """Exact provider request derived from a provider-independent SourceQuery."""

    request_id: str
    query_id: str
    connector_id: ConnectorId
    channel: RetrievalChannel
    endpoint: str
    parameters: tuple[QueryParameter, ...]
    page_index: int = 1
    attempt: int = 1
    schema_version: str = field(default=CONNECTOR_REQUEST_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        _require_stable_id(self.request_id, "request_id")
        _require_stable_id(self.query_id, "query_id")
        _require_enum(self.connector_id, ConnectorId, "connector_id")
        _require_enum(self.channel, RetrievalChannel, "channel")
        _require_https_url(self.endpoint, "endpoint")
        if not self.parameters:
            raise ValueError("parameters must not be empty")
        names = [parameter.name for parameter in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError("parameter names must be unique")
        if names != sorted(names):
            raise ValueError("parameters must be sorted by name")
        if self.page_index < 1:
            raise ValueError("page_index must be positive")
        if self.attempt < 1:
            raise ValueError("attempt must be positive")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True, slots=True)
class RawResponseArtifact:
    """Exact bytes received from a source API and stored inside a snapshot."""

    uri: str
    media_type: str
    size_bytes: int
    sha256: str
    retrieved_at: datetime

    def __post_init__(self) -> None:
        _require_text(self.uri, "uri")
        path = PurePosixPath(self.uri)
        if path.is_absolute() or ".." in path.parts or "\\" in self.uri:
            raise ValueError("uri must be a safe relative POSIX path")
        _require_text(self.media_type, "media_type")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("sha256 must contain 64 lowercase hexadecimal characters")
        _require_aware_datetime(self.retrieved_at, "retrieved_at")


@dataclass(frozen=True, slots=True)
class ConnectorError:
    code: str
    message: str
    retryable: bool

    def __post_init__(self) -> None:
        _require_stable_id(self.code, "error code")
        _require_text(self.message, "error message")


@dataclass(frozen=True, slots=True)
class ConnectorRun:
    """Outcome of one connector attempt without confusing failure with zero results."""

    run_id: str
    request: ConnectorRequest
    status: ConnectorStatus
    started_at: datetime
    finished_at: datetime
    http_status: int | None
    returned_records: int | None
    artifact: RawResponseArtifact | None
    error: ConnectorError | None = None

    def __post_init__(self) -> None:
        _require_stable_id(self.run_id, "run_id")
        _require_enum(self.status, ConnectorStatus, "status")
        _require_aware_datetime(self.started_at, "started_at")
        _require_aware_datetime(self.finished_at, "finished_at")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not be before started_at")
        if self.http_status is not None and not 100 <= self.http_status <= 599:
            raise ValueError("http_status must be a valid HTTP status code")
        if self.returned_records is not None and self.returned_records < 0:
            raise ValueError("returned_records must be non-negative")

        if self.status is ConnectorStatus.SUCCESS:
            if self.artifact is None or self.returned_records is None:
                raise ValueError("successful connector run requires an artifact and record count")
            if self.error is not None:
                raise ValueError("successful connector run must not contain an error")
            if self.http_status is not None and not 200 <= self.http_status <= 299:
                raise ValueError("successful connector run requires a 2xx HTTP status")
        elif self.status is ConnectorStatus.PARTIAL:
            if self.artifact is None or self.returned_records is None or self.error is None:
                raise ValueError("partial connector run requires data, count and an error")
        else:
            if self.returned_records is not None:
                raise ValueError("failed connector run must not report a record count")
            if self.error is None:
                raise ValueError("failed connector run requires an error")

    @property
    def coverage_complete(self) -> bool:
        return self.status is ConnectorStatus.SUCCESS

    @property
    def duration_ms(self) -> int:
        return round((self.finished_at - self.started_at).total_seconds() * 1000)

    def to_dict(self) -> dict[str, Any]:
        payload = _json_value(asdict(self))
        payload["coverage_complete"] = self.coverage_complete
        payload["duration_ms"] = self.duration_ms
        return payload


@dataclass(frozen=True, slots=True)
class SnapshotManifest:
    """Reproducible manifest joining a source query with all connector runs."""

    snapshot_id: str
    snapshot_version: str
    analysis_id: str
    created_at: datetime
    query: SourceQuery
    runs: tuple[ConnectorRun, ...]
    schema_version: str = field(default=SNAPSHOT_MANIFEST_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        for field_name, value in (
            ("snapshot_id", self.snapshot_id),
            ("snapshot_version", self.snapshot_version),
            ("analysis_id", self.analysis_id),
        ):
            _require_stable_id(value, field_name)
        _require_aware_datetime(self.created_at, "created_at")
        if not self.runs:
            raise ValueError("snapshot must contain at least one connector run")
        run_ids = [run.run_id for run in self.runs]
        request_ids = [run.request.request_id for run in self.runs]
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("connector run IDs must be unique")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("connector request IDs must be unique")
        if any(run.request.query_id != self.query.query_id for run in self.runs):
            raise ValueError("all connector runs must reference the snapshot query")
        artifact_uris = [run.artifact.uri for run in self.runs if run.artifact is not None]
        if len(set(artifact_uris)) != len(artifact_uris):
            raise ValueError("raw response artifact URIs must be unique")

    @property
    def status(self) -> SnapshotStatus:
        statuses = {run.status for run in self.runs}
        if statuses == {ConnectorStatus.SUCCESS}:
            return SnapshotStatus.COMPLETE
        if statuses == {ConnectorStatus.FAILED}:
            return SnapshotStatus.FAILED
        return SnapshotStatus.PARTIAL

    def to_dict(self) -> dict[str, Any]:
        payload = _json_value(asdict(self))
        payload["status"] = self.status.value
        return payload
