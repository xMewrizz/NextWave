"""Normalization of raw OpenAlex work records into domain documents."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from urllib.parse import urlparse

from nextwave.contracts import SourceDocument, SourceType, TrustTier

from .identifiers import doi_url, normalize_doi

_OPENALEX_WORK_ID = re.compile(r"W[0-9]+\Z", re.IGNORECASE)
_OPENALEX_TOPIC_ID = re.compile(r"T[0-9]+\Z", re.IGNORECASE)
_SCIENTIFIC_WORK_TYPES = {
    "article",
    "book",
    "book-chapter",
    "dissertation",
    "editorial",
    "letter",
    "peer-review",
    "preprint",
    "report",
    "review",
}


@dataclass(frozen=True, slots=True)
class OpenAlexParseIssue:
    """One rejected OpenAlex record that must not silently enter the domain model."""

    record_index: int
    external_id: str | None
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class OpenAlexTopicHint:
    topic_id: str
    display_name: str
    score: float
    primary: bool
    subfield_id: str
    subfield_name: str
    field_id: str
    field_name: str
    domain_id: str
    domain_name: str


@dataclass(frozen=True, slots=True)
class OpenAlexKeywordHint:
    keyword_id: str
    display_name: str
    score: float


@dataclass(frozen=True, slots=True)
class OpenAlexDiscoveryHints:
    document_id: str
    topics: tuple[OpenAlexTopicHint, ...]
    keywords: tuple[OpenAlexKeywordHint, ...]


@dataclass(frozen=True, slots=True)
class OpenAlexHintIssue:
    record_index: int
    external_id: str | None
    document_id: str
    message: str


@dataclass(frozen=True, slots=True)
class OpenAlexParseResult:
    """Valid normalized documents plus explicit per-record rejections."""

    documents: tuple[SourceDocument, ...]
    hints: tuple[OpenAlexDiscoveryHints, ...]
    issues: tuple[OpenAlexParseIssue, ...]
    hint_issues: tuple[OpenAlexHintIssue, ...]
    total_records: int

    @property
    def accepted_records(self) -> int:
        return len(self.documents)

    @property
    def rejected_records(self) -> int:
        return len(self.issues)


def reconstruct_openalex_abstract(value: object) -> str | None:
    """Restore readable text from OpenAlex's word-to-position inverted index."""

    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("abstract_inverted_index must be an object or null")

    words_by_position: dict[int, str] = {}
    for word, positions in value.items():
        if not isinstance(word, str) or not word.strip():
            raise ValueError("abstract words must be non-blank strings")
        if not isinstance(positions, list):
            raise ValueError("abstract positions must be lists")
        for position in positions:
            if not isinstance(position, int) or isinstance(position, bool) or position < 0:
                raise ValueError("abstract positions must be non-negative integers")
            if position in words_by_position:
                raise ValueError("abstract contains duplicate word positions")
            words_by_position[position] = word

    if not words_by_position:
        return None
    return " ".join(words_by_position[position] for position in sorted(words_by_position))


def parse_openalex_response(
    payload: bytes,
    *,
    snapshot_id: str,
    retrieved_at: datetime,
    cutoff_date: date,
) -> OpenAlexParseResult:
    """Parse one saved OpenAlex response without allowing bad rows or future data through."""

    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"OpenAlex response is not valid UTF-8 JSON: {error}") from error
    if not isinstance(decoded, dict) or not isinstance(decoded.get("results"), list):
        raise ValueError("OpenAlex response must contain a results list")

    documents: list[SourceDocument] = []
    hints: list[OpenAlexDiscoveryHints] = []
    issues: list[OpenAlexParseIssue] = []
    hint_issues: list[OpenAlexHintIssue] = []
    for index, record in enumerate(decoded["results"]):
        external_id = _best_effort_external_id(record)
        try:
            document = parse_openalex_work(
                record,
                snapshot_id=snapshot_id,
                retrieved_at=retrieved_at,
                cutoff_date=cutoff_date,
            )
            documents.append(document)
        except ValueError as error:
            issues.append(
                OpenAlexParseIssue(
                    record_index=index,
                    external_id=external_id,
                    code=_issue_code(error),
                    message=str(error),
                )
            )
            continue
        try:
            hints.append(parse_openalex_discovery_hints(record, document.document_id))
        except ValueError as error:
            hints.append(OpenAlexDiscoveryHints(document.document_id, (), ()))
            hint_issues.append(
                OpenAlexHintIssue(
                    record_index=index,
                    external_id=external_id,
                    document_id=document.document_id,
                    message=str(error),
                )
            )

    return OpenAlexParseResult(
        documents=tuple(documents),
        hints=tuple(hints),
        issues=tuple(issues),
        hint_issues=tuple(hint_issues),
        total_records=len(decoded["results"]),
    )


def parse_openalex_discovery_hints(
    record: object,
    document_id: str,
) -> OpenAlexDiscoveryHints:
    """Preserve scored OpenAlex aboutness fields outside the universal document model."""

    if not isinstance(record, dict):
        raise ValueError("record must be an object")
    primary_record = record.get("primary_topic")
    primary_id = _optional_topic_id(primary_record)
    topic_records = record.get("topics")
    if topic_records is None:
        topic_records = []
    if not isinstance(topic_records, list):
        raise ValueError("topics must be a list or null")
    topics = [_topic_hint(value, primary_id) for value in topic_records]
    if primary_record is not None and primary_id not in {topic.topic_id for topic in topics}:
        topics.insert(0, _topic_hint(primary_record, primary_id))

    keyword_records = record.get("keywords")
    if keyword_records is None:
        keyword_records = []
    if not isinstance(keyword_records, list):
        raise ValueError("keywords must be a list or null")
    keywords = tuple(_keyword_hint(value) for value in keyword_records)
    return OpenAlexDiscoveryHints(
        document_id=document_id,
        topics=tuple(topics),
        keywords=keywords,
    )


def parse_openalex_work(
    record: object,
    *,
    snapshot_id: str,
    retrieved_at: datetime,
    cutoff_date: date,
) -> SourceDocument:
    """Normalize one OpenAlex work into a reproducible SourceDocument."""

    if not isinstance(record, dict):
        raise ValueError("record must be an object")
    work_url, external_id = _work_identity(record.get("id"))
    title = _required_title(record)
    published_at = _publication_date(record.get("publication_date"))
    if published_at is not None and published_at > cutoff_date:
        raise ValueError("publication_date is after cutoff_date")

    doi = normalize_doi(record.get("doi"))
    landing_page_url = _landing_page_url(record.get("primary_location"))
    url = landing_page_url or work_url
    canonical_url = doi_url(doi) if doi is not None else url
    origin_id = f"doi:{doi}" if doi is not None else f"url:{canonical_url.casefold()}"
    origin_method = "doi" if doi is not None else "canonical_url"
    source_type, trust_tier = _source_classification(record.get("type"))

    document_digest = hashlib.sha256(
        f"{snapshot_id}|openalex|{external_id.casefold()}".encode()
    ).hexdigest()[:16]
    return SourceDocument(
        document_id=f"document-openalex-{document_digest}",
        connector_id="openalex",
        external_id=external_id,
        snapshot_id=snapshot_id,
        title=title,
        url=url,
        canonical_url=canonical_url,
        source_type=source_type,
        language=_language(record.get("language")),
        trust_tier=trust_tier,
        origin_id=origin_id,
        doi=doi,
        authors=_authors(record.get("authorships")),
        organizations=_organizations(record.get("authorships")),
        published_at=published_at,
        retrieved_at=retrieved_at,
        publisher=_publisher(record.get("primary_location")),
        excerpt=reconstruct_openalex_abstract(record.get("abstract_inverted_index")),
        origin_method=origin_method,
        origin_confidence=1.0,
    )


def _work_identity(value: object) -> tuple[str, str]:
    if not isinstance(value, str) or not _is_http_url(value):
        raise ValueError("record id must be an absolute OpenAlex work URL")
    external_id = value.rstrip("/").rsplit("/", 1)[-1]
    if not _OPENALEX_WORK_ID.fullmatch(external_id):
        raise ValueError("record id must end with an OpenAlex W identifier")
    return value, external_id.upper()


def _required_title(record: dict[str, Any]) -> str:
    for field_name in ("title", "display_name"):
        value = record.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("record must contain a non-blank title")


def _publication_date(value: object) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("publication_date must be an ISO date or null")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError("publication_date must be an ISO date or null") from error


def _landing_page_url(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    landing_page_url = value.get("landing_page_url")
    if isinstance(landing_page_url, str) and _is_http_url(landing_page_url):
        return landing_page_url
    return None


def _publisher(value: object) -> str | None:
    if not isinstance(value, dict) or not isinstance(value.get("source"), dict):
        return None
    display_name = value["source"].get("display_name")
    if isinstance(display_name, str) and display_name.strip():
        return display_name.strip()
    return None


def _authors(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    names = []
    for authorship in value:
        if not isinstance(authorship, dict) or not isinstance(authorship.get("author"), dict):
            continue
        name = authorship["author"].get("display_name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return _unique(names)


def _organizations(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    names = []
    for authorship in value:
        if not isinstance(authorship, dict) or not isinstance(authorship.get("institutions"), list):
            continue
        for institution in authorship["institutions"]:
            if not isinstance(institution, dict):
                continue
            name = institution.get("display_name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
    return _unique(names)


def _unique(values: list[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result)


def _language(value: object) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip().casefold()
    return "und"


def _source_classification(value: object) -> tuple[SourceType, TrustTier]:
    if isinstance(value, str) and value.casefold() in _SCIENTIFIC_WORK_TYPES:
        return SourceType.SCIENTIFIC_PUBLICATION, TrustTier.A
    return SourceType.OTHER, TrustTier.UNKNOWN


def _optional_topic_id(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("primary_topic must be an object or null")
    return _openalex_hint_id(value.get("id"), "T", "primary_topic.id")


def _topic_hint(value: object, primary_id: str | None) -> OpenAlexTopicHint:
    if not isinstance(value, dict):
        raise ValueError("topic hint must be an object")
    topic_id = _openalex_hint_id(value.get("id"), "T", "topic.id")
    return OpenAlexTopicHint(
        topic_id=topic_id,
        display_name=_hint_name(value.get("display_name"), "topic.display_name"),
        score=_hint_score(value.get("score"), "topic.score"),
        primary=topic_id == primary_id,
        subfield_id=_hierarchy_id(value.get("subfield"), "subfields", "topic.subfield"),
        subfield_name=_hierarchy_name(value.get("subfield"), "topic.subfield"),
        field_id=_hierarchy_id(value.get("field"), "fields", "topic.field"),
        field_name=_hierarchy_name(value.get("field"), "topic.field"),
        domain_id=_hierarchy_id(value.get("domain"), "domains", "topic.domain"),
        domain_name=_hierarchy_name(value.get("domain"), "topic.domain"),
    )


def _keyword_hint(value: object) -> OpenAlexKeywordHint:
    if not isinstance(value, dict):
        raise ValueError("keyword hint must be an object")
    keyword_id = _openalex_hint_id(value.get("id"), "keywords", "keyword.id")
    return OpenAlexKeywordHint(
        keyword_id=keyword_id,
        display_name=_hint_name(value.get("display_name"), "keyword.display_name"),
        score=_hint_score(value.get("score"), "keyword.score"),
    )


def _openalex_hint_id(value: object, expected_parent: str, field_name: str) -> str:
    if not isinstance(value, str) or not _is_http_url(value):
        raise ValueError(f"{field_name} must be an absolute OpenAlex URL")
    path_parts = [part for part in urlparse(value).path.split("/") if part]
    if not path_parts:
        raise ValueError(f"{field_name} has no identifier")
    identifier = path_parts[-1]
    if expected_parent == "T":
        if _OPENALEX_TOPIC_ID.fullmatch(identifier) is None:
            raise ValueError(f"{field_name} must end with an OpenAlex T identifier")
    elif len(path_parts) < 2 or path_parts[-2] != expected_parent:
        raise ValueError(f"{field_name} must use the {expected_parent} path")
    return identifier


def _hint_name(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")
    return value.strip()


def _hint_score(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be a number")
    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return score


def _hierarchy_id(value: object, parent: str, field_name: str) -> str:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return _openalex_hint_id(value.get("id"), parent, f"{field_name}.id")


def _hierarchy_name(value: object, field_name: str) -> str:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return _hint_name(value.get("display_name"), f"{field_name}.display_name")


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _best_effort_external_id(record: object) -> str | None:
    if not isinstance(record, dict) or not isinstance(record.get("id"), str):
        return None
    return record["id"].rstrip("/").rsplit("/", 1)[-1] or None


def _issue_code(error: ValueError) -> str:
    message = str(error)
    if "after cutoff_date" in message:
        return "after_cutoff"
    if "title" in message:
        return "missing_title"
    if "record id" in message:
        return "invalid_identifier"
    return "invalid_record"
