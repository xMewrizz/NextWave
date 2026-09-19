"""Normalization of GDELT DOC article records into domain documents."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from nextwave.contracts import SourceDocument, SourceType, TrustTier

_TRACKING_PARAMETERS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
}
_LANGUAGE_CODES = {
    "arabic": "ar",
    "chinese": "zh",
    "english": "en",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "portuguese": "pt",
    "russian": "ru",
    "spanish": "es",
}


@dataclass(frozen=True, slots=True)
class GdeltParseIssue:
    record_index: int
    external_id: str | None
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class GdeltParseResult:
    documents: tuple[SourceDocument, ...]
    issues: tuple[GdeltParseIssue, ...]
    total_records: int

    @property
    def accepted_records(self) -> int:
        return len(self.documents)

    @property
    def rejected_records(self) -> int:
        return len(self.issues)


def canonicalize_article_url(value: object) -> str:
    """Normalize an article URL while retaining content-changing parameters."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("article url must be a non-blank string")
    parsed = urlsplit(value.strip())
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("article url must be an absolute HTTP or HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("article url must not contain credentials")

    scheme = parsed.scheme.casefold()
    hostname = parsed.hostname.casefold()
    port = parsed.port
    if port is None or (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        netloc = hostname
    else:
        netloc = f"{hostname}:{port}"
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")

    query_items = []
    for name, item_value in parse_qsl(parsed.query, keep_blank_values=True):
        normalized_name = name.casefold()
        if normalized_name.startswith("utm_") or normalized_name in _TRACKING_PARAMETERS:
            continue
        query_items.append((name, item_value))
    query_items.sort()
    return urlunsplit((scheme, netloc, path, urlencode(query_items), ""))


def parse_gdelt_response(
    payload: bytes,
    *,
    snapshot_id: str,
    retrieved_at: datetime,
    cutoff_date: date,
) -> GdeltParseResult:
    """Parse a saved GDELT article-list response and isolate invalid records."""

    try:
        decoded = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"GDELT response is not valid UTF-8 JSON: {error}") from error
    if not isinstance(decoded, dict) or not isinstance(decoded.get("articles"), list):
        raise ValueError("GDELT response must contain an articles list")

    documents: list[SourceDocument] = []
    issues: list[GdeltParseIssue] = []
    for index, record in enumerate(decoded["articles"]):
        external_id = _best_effort_external_id(record)
        try:
            documents.append(
                parse_gdelt_article(
                    record,
                    snapshot_id=snapshot_id,
                    retrieved_at=retrieved_at,
                    cutoff_date=cutoff_date,
                )
            )
        except (TypeError, ValueError) as error:
            issues.append(
                GdeltParseIssue(
                    record_index=index,
                    external_id=external_id,
                    code=_issue_code(error),
                    message=str(error),
                )
            )

    return GdeltParseResult(
        documents=tuple(documents),
        issues=tuple(issues),
        total_records=len(decoded["articles"]),
    )


def parse_gdelt_article(
    record: object,
    *,
    snapshot_id: str,
    retrieved_at: datetime,
    cutoff_date: date,
) -> SourceDocument:
    """Normalize one GDELT article without treating it as verified evidence."""

    if not isinstance(record, dict):
        raise ValueError("record must be an object")
    canonical_url = canonicalize_article_url(record.get("url"))
    title = _required_text(record.get("title"), "article title")
    observed_at = _observed_at(record.get("seendate"))
    if observed_at.date() > cutoff_date:
        raise ValueError("seendate is after cutoff_date")

    external_id = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:24]
    document_digest = hashlib.sha256(
        f"{snapshot_id}|gdelt|{external_id}".encode("utf-8")
    ).hexdigest()[:16]
    return SourceDocument(
        document_id=f"document-gdelt-{document_digest}",
        connector_id="gdelt",
        external_id=external_id,
        snapshot_id=snapshot_id,
        title=title,
        url=_required_text(record.get("url"), "article url"),
        canonical_url=canonical_url,
        source_type=SourceType.OTHER,
        language=_language(record.get("language")),
        trust_tier=TrustTier.UNKNOWN,
        origin_id=f"url:{canonical_url}",
        published_at=None,
        observed_at=observed_at,
        retrieved_at=retrieved_at,
        publisher=_publisher(record),
        origin_method="canonical_url",
        origin_confidence=1.0,
    )


def _observed_at(value: object) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("seendate must be a non-blank timestamp")
    normalized = value.strip()
    formats = (
        "%Y%m%dT%H%M%SZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y%m%d%H%M%S",
    )
    for timestamp_format in formats:
        try:
            return datetime.strptime(normalized, timestamp_format).replace(tzinfo=UTC)
        except ValueError:
            continue
    raise ValueError("seendate has an unsupported timestamp format")


def _publisher(record: dict[str, object]) -> str | None:
    domain = record.get("domain")
    if isinstance(domain, str) and domain.strip():
        return domain.strip().casefold()
    try:
        return urlsplit(_required_text(record.get("url"), "article url")).hostname
    except ValueError:
        return None


def _language(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return "und"
    normalized = value.strip().casefold()
    if len(normalized) in {2, 3} and normalized.isalpha():
        return normalized
    return _LANGUAGE_CODES.get(normalized, "und")


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")
    return value.strip()


def _best_effort_external_id(record: object) -> str | None:
    if not isinstance(record, dict):
        return None
    value = record.get("url")
    if not isinstance(value, str):
        return None
    try:
        canonical_url = canonicalize_article_url(value)
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:24]


def _issue_code(error: Exception) -> str:
    message = str(error)
    if "after cutoff_date" in message:
        return "after_cutoff"
    if "url" in message:
        return "invalid_identifier"
    if "title" in message:
        return "missing_title"
    if "seendate" in message:
        return "invalid_observation_time"
    return "invalid_record"
