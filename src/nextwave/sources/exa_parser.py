"""Normalization of saved Exa news-search responses."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from urllib.parse import urlparse

from nextwave.contracts import SourceDocument, SourceType, TrustTier

from .gdelt_parser import canonicalize_article_url


@dataclass(frozen=True, slots=True)
class ExaParseIssue:
    record_index: int
    external_id: str | None
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ExaParseResult:
    documents: tuple[SourceDocument, ...]
    issues: tuple[ExaParseIssue, ...]
    total_records: int

    @property
    def accepted_records(self) -> int:
        return len(self.documents)

    @property
    def rejected_records(self) -> int:
        return len(self.issues)


def parse_exa_response(
    payload: bytes,
    *,
    snapshot_id: str,
    retrieved_at: datetime,
    cutoff_date: date,
) -> ExaParseResult:
    try:
        decoded = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Exa response is not valid UTF-8 JSON") from error
    if not isinstance(decoded, dict) or not isinstance(decoded.get("results"), list):
        raise ValueError("Exa response must contain a results list")

    documents: list[SourceDocument] = []
    issues: list[ExaParseIssue] = []
    for index, record in enumerate(decoded["results"]):
        external_id = _best_effort_id(record)
        try:
            documents.append(
                parse_exa_result(
                    record,
                    snapshot_id=snapshot_id,
                    retrieved_at=retrieved_at,
                    cutoff_date=cutoff_date,
                )
            )
        except (TypeError, ValueError) as error:
            issues.append(
                ExaParseIssue(index, external_id, _issue_code(error), str(error))
            )
    return ExaParseResult(tuple(documents), tuple(issues), len(decoded["results"]))


def parse_exa_result(
    record: object,
    *,
    snapshot_id: str,
    retrieved_at: datetime,
    cutoff_date: date,
) -> SourceDocument:
    if not isinstance(record, dict):
        raise ValueError("record must be an object")
    external_id = _required_text(record.get("id"), "result id")
    url = _required_text(record.get("url"), "result url")
    canonical_url = canonicalize_article_url(url)
    title = _required_text(record.get("title"), "result title")
    published_at = _published_at(record.get("publishedDate"))
    if published_at is not None and published_at > cutoff_date:
        raise ValueError("publishedDate is after cutoff_date")
    excerpt = _excerpt(record)
    author = record.get("author")
    authors = (author.strip(),) if isinstance(author, str) and author.strip() else ()
    publisher = urlparse(canonical_url).hostname
    digest = hashlib.sha256(
        f"{snapshot_id}|exa|{external_id}".encode()
    ).hexdigest()[:16]
    return SourceDocument(
        document_id=f"document-exa-{digest}",
        connector_id="exa",
        external_id=external_id,
        snapshot_id=snapshot_id,
        title=title,
        url=url,
        canonical_url=canonical_url,
        source_type=SourceType.INDUSTRY_MEDIA,
        language="und",
        trust_tier=TrustTier.UNKNOWN,
        origin_id=f"url:{canonical_url}",
        authors=authors,
        published_at=published_at,
        observed_at=retrieved_at.astimezone(UTC),
        retrieved_at=retrieved_at,
        publisher=publisher.casefold() if publisher else None,
        excerpt=excerpt,
        origin_method="canonical_url",
        origin_confidence=1.0,
    )


def _excerpt(record: dict[str, object]) -> str | None:
    highlights = record.get("highlights")
    if highlights is None:
        return None
    if not isinstance(highlights, list):
        raise ValueError("highlights must be a list")
    parts = [item.strip() for item in highlights if isinstance(item, str) and item.strip()]
    return "\n\n".join(parts) or None


def _published_at(value: object) -> date | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("publishedDate must be a timestamp or null")
    normalized = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).date()
    except ValueError as error:
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError:
            raise ValueError("publishedDate has an unsupported format") from error


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")
    return value.strip()


def _best_effort_id(record: object) -> str | None:
    if not isinstance(record, dict):
        return None
    value = record.get("id")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _issue_code(error: Exception) -> str:
    message = str(error)
    if "after cutoff_date" in message:
        return "after_cutoff"
    if "title" in message:
        return "missing_title"
    if "url" in message or "result id" in message:
        return "invalid_identifier"
    if "publishedDate" in message:
        return "invalid_publication_date"
    return "invalid_record"
