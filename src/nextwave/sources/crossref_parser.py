"""Normalization of raw Crossref work records into domain documents."""

from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from urllib.parse import urlparse

from nextwave.contracts import SourceDocument, SourceType, TrustTier

from .identifiers import doi_url, normalize_doi

_SCIENTIFIC_TYPES = {
    "book",
    "book-chapter",
    "book-part",
    "book-section",
    "book-series",
    "dissertation",
    "edited-book",
    "journal-article",
    "monograph",
    "peer-review",
    "posted-content",
    "proceedings",
    "proceedings-article",
    "reference-book",
    "reference-entry",
}
_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class CrossrefParseIssue:
    record_index: int
    external_id: str | None
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class CrossrefParseResult:
    documents: tuple[SourceDocument, ...]
    issues: tuple[CrossrefParseIssue, ...]
    total_records: int

    @property
    def accepted_records(self) -> int:
        return len(self.documents)

    @property
    def rejected_records(self) -> int:
        return len(self.issues)


def parse_crossref_response(
    payload: bytes,
    *,
    snapshot_id: str,
    retrieved_at: datetime,
    cutoff_date: date,
) -> CrossrefParseResult:
    """Parse a saved Crossref response and isolate invalid individual records."""

    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Crossref response is not valid UTF-8 JSON: {error}") from error
    if not isinstance(decoded, dict) or not isinstance(decoded.get("message"), dict):
        raise ValueError("Crossref response must contain a message object")
    items = decoded["message"].get("items")
    if not isinstance(items, list):
        raise ValueError("Crossref response message must contain an items list")

    documents: list[SourceDocument] = []
    issues: list[CrossrefParseIssue] = []
    for index, record in enumerate(items):
        external_id = _best_effort_external_id(record)
        try:
            documents.append(
                parse_crossref_work(
                    record,
                    snapshot_id=snapshot_id,
                    retrieved_at=retrieved_at,
                    cutoff_date=cutoff_date,
                )
            )
        except ValueError as error:
            issues.append(
                CrossrefParseIssue(
                    record_index=index,
                    external_id=external_id,
                    code=_issue_code(error),
                    message=str(error),
                )
            )

    return CrossrefParseResult(
        documents=tuple(documents),
        issues=tuple(issues),
        total_records=len(items),
    )


def parse_crossref_work(
    record: object,
    *,
    snapshot_id: str,
    retrieved_at: datetime,
    cutoff_date: date,
) -> SourceDocument:
    """Normalize one Crossref work into the same document contract as OpenAlex."""

    if not isinstance(record, dict):
        raise ValueError("record must be an object")
    doi = normalize_doi(record.get("DOI"))
    if doi is None:
        raise ValueError("record must contain a DOI")
    title = _title(record)
    published_at = _publication_date(record)
    if published_at is not None and published_at > cutoff_date:
        raise ValueError("publication_date is after cutoff_date")

    canonical_url = doi_url(doi)
    url_value = record.get("URL")
    url = url_value if isinstance(url_value, str) and _is_http_url(url_value) else canonical_url
    source_type, trust_tier = _source_classification(record.get("type"))
    document_digest = hashlib.sha256(
        f"{snapshot_id}|crossref|{doi}".encode("utf-8")
    ).hexdigest()[:16]
    return SourceDocument(
        document_id=f"document-crossref-{document_digest}",
        connector_id="crossref",
        external_id=doi,
        snapshot_id=snapshot_id,
        title=title,
        url=url,
        canonical_url=canonical_url,
        source_type=source_type,
        language=_language(record.get("language")),
        trust_tier=trust_tier,
        origin_id=f"doi:{doi}",
        doi=doi,
        authors=_authors(record.get("author")),
        organizations=_organizations(record.get("author")),
        published_at=published_at,
        retrieved_at=retrieved_at,
        publisher=_optional_text(record.get("publisher")),
        excerpt=_abstract(record.get("abstract")),
        origin_method="doi",
        origin_confidence=1.0,
    )


def _title(record: dict[str, object]) -> str:
    for field_name in ("title", "subtitle"):
        value = record.get(field_name)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return _clean_text(item)
        elif isinstance(value, str) and value.strip():
            return _clean_text(value)
    raise ValueError("record must contain a non-blank title")


def _publication_date(record: dict[str, object]) -> date | None:
    for field_name in ("published", "published-online", "published-print", "issued"):
        value = record.get(field_name)
        if not isinstance(value, dict) or not isinstance(value.get("date-parts"), list):
            continue
        date_parts = value["date-parts"]
        if not date_parts or not isinstance(date_parts[0], list):
            continue
        first = date_parts[0]
        if len(first) < 3:
            continue
        year, month, day = first[:3]
        if any(not isinstance(part, int) or isinstance(part, bool) for part in (year, month, day)):
            raise ValueError("publication date parts must be integers")
        try:
            return date(year, month, day)
        except ValueError as error:
            raise ValueError("publication date parts do not form a valid date") from error
    return None


def _authors(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    names: list[str] = []
    for author in value:
        if not isinstance(author, dict):
            continue
        collective_name = _optional_text(author.get("name"))
        given = _optional_text(author.get("given"))
        family = _optional_text(author.get("family"))
        name = collective_name or " ".join(part for part in (given, family) if part)
        if name:
            names.append(name)
    return _unique(names)


def _organizations(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    names: list[str] = []
    for author in value:
        if not isinstance(author, dict) or not isinstance(author.get("affiliation"), list):
            continue
        for affiliation in author["affiliation"]:
            if not isinstance(affiliation, dict):
                continue
            name = _optional_text(affiliation.get("name"))
            if name:
                names.append(name)
    return _unique(names)


def _abstract(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return _clean_text(value)


def _clean_text(value: str) -> str:
    without_tags = _TAG.sub(" ", value)
    return _WHITESPACE.sub(" ", html.unescape(without_tags)).strip()


def _optional_text(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return _clean_text(value)
    return None


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
    if not isinstance(value, str):
        return SourceType.OTHER, TrustTier.UNKNOWN
    normalized = value.casefold()
    if normalized == "standard":
        return SourceType.STANDARD, TrustTier.A
    if normalized == "report":
        return SourceType.ANALYTICAL_REPORT, TrustTier.B
    if normalized in _SCIENTIFIC_TYPES:
        return SourceType.SCIENTIFIC_PUBLICATION, TrustTier.A
    return SourceType.OTHER, TrustTier.UNKNOWN


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _best_effort_external_id(record: object) -> str | None:
    if not isinstance(record, dict):
        return None
    value = record.get("DOI")
    return value if isinstance(value, str) and value.strip() else None


def _issue_code(error: ValueError) -> str:
    message = str(error)
    if "after cutoff_date" in message:
        return "after_cutoff"
    if "DOI" in message or "doi" in message:
        return "invalid_identifier"
    if "title" in message:
        return "missing_title"
    return "invalid_record"
