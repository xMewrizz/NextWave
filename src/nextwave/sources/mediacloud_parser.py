"""Normalization of Media Cloud stories and attention timelines."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime

from nextwave.contracts import SourceDocument, SourceType, TrustTier

from .gdelt_parser import canonicalize_article_url


@dataclass(frozen=True, slots=True)
class MediaCloudParseIssue:
    record_index: int
    external_id: str | None
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class MediaCloudParseResult:
    documents: tuple[SourceDocument, ...]
    issues: tuple[MediaCloudParseIssue, ...]
    total_records: int
    pagination_token: str | None

    @property
    def accepted_records(self) -> int:
        return len(self.documents)

    @property
    def rejected_records(self) -> int:
        return len(self.issues)


@dataclass(frozen=True, slots=True)
class MediaCloudTimelinePoint:
    interval_start: date
    matched_articles: int
    monitored_articles: int

    def __post_init__(self) -> None:
        if self.matched_articles < 0 or self.monitored_articles < 0:
            raise ValueError("timeline counts must be non-negative")
        if self.matched_articles > self.monitored_articles:
            raise ValueError("matched articles must not exceed monitored articles")

    @property
    def share(self) -> float | None:
        if self.monitored_articles == 0:
            return None
        return self.matched_articles / self.monitored_articles


@dataclass(frozen=True, slots=True)
class MediaCloudTimelineResult:
    points: tuple[MediaCloudTimelinePoint, ...]

    @property
    def matched_articles(self) -> int:
        return sum(point.matched_articles for point in self.points)

    @property
    def monitored_articles(self) -> int:
        return sum(point.monitored_articles for point in self.points)

    @property
    def share(self) -> float | None:
        if self.monitored_articles == 0:
            return None
        return self.matched_articles / self.monitored_articles


def parse_mediacloud_response(
    payload: bytes,
    *,
    snapshot_id: str,
    retrieved_at: datetime,
    cutoff_date: date,
) -> MediaCloudParseResult:
    """Parse one saved story-list page and isolate invalid records."""

    decoded = _decode_object(payload)
    stories = decoded.get("stories")
    if not isinstance(stories, list):
        raise ValueError("Media Cloud response must contain a stories list")
    pagination_token = decoded.get("pagination_token")
    if pagination_token is not None and not isinstance(pagination_token, str):
        raise ValueError("Media Cloud pagination_token must be a string or null")

    documents: list[SourceDocument] = []
    issues: list[MediaCloudParseIssue] = []
    for index, record in enumerate(stories):
        external_id = _best_effort_external_id(record)
        try:
            documents.append(
                parse_mediacloud_story(
                    record,
                    snapshot_id=snapshot_id,
                    retrieved_at=retrieved_at,
                    cutoff_date=cutoff_date,
                )
            )
        except (TypeError, ValueError) as error:
            issues.append(
                MediaCloudParseIssue(
                    record_index=index,
                    external_id=external_id,
                    code=_issue_code(error),
                    message=str(error),
                )
            )

    return MediaCloudParseResult(
        documents=tuple(documents),
        issues=tuple(issues),
        total_records=len(stories),
        pagination_token=pagination_token,
    )


def parse_mediacloud_story(
    record: object,
    *,
    snapshot_id: str,
    retrieved_at: datetime,
    cutoff_date: date,
) -> SourceDocument:
    """Normalize metadata without pretending the unavailable full text is evidence."""

    if not isinstance(record, dict):
        raise ValueError("record must be an object")
    external_id = _required_text(record.get("id"), "story id")
    canonical_url = canonicalize_article_url(record.get("url"))
    title = _required_text(record.get("title"), "story title")
    published_at = _published_at(record.get("publish_date"))
    if published_at is not None and published_at > cutoff_date:
        raise ValueError("publish_date is after cutoff_date")
    observed_at = _observed_at(record.get("indexed_date"))
    if observed_at.date() > cutoff_date:
        raise ValueError("indexed_date is after cutoff_date")

    document_digest = hashlib.sha256(
        f"{snapshot_id}|mediacloud|{external_id}".encode("utf-8")
    ).hexdigest()[:16]
    return SourceDocument(
        document_id=f"document-mediacloud-{document_digest}",
        connector_id="mediacloud",
        external_id=external_id,
        snapshot_id=snapshot_id,
        title=title,
        url=_required_text(record.get("url"), "story url"),
        canonical_url=canonical_url,
        source_type=SourceType.OTHER,
        language=_language(record.get("language")),
        trust_tier=TrustTier.UNKNOWN,
        origin_id=f"url:{canonical_url}",
        published_at=published_at,
        observed_at=observed_at,
        retrieved_at=retrieved_at,
        publisher=_publisher(record),
        origin_method="canonical_url",
        origin_confidence=1.0,
    )


def parse_mediacloud_timeline_response(
    payload: bytes,
    *,
    cutoff_date: date,
) -> MediaCloudTimelineResult:
    """Parse daily counts and recompute normalized news attention."""

    decoded = _decode_object(payload)
    timeline = decoded.get("count_over_time")
    if not isinstance(timeline, dict) or not isinstance(timeline.get("counts"), list):
        raise ValueError("Media Cloud timeline response must contain a counts list")

    points: list[MediaCloudTimelinePoint] = []
    seen_intervals: set[date] = set()
    for item in timeline["counts"]:
        if not isinstance(item, dict):
            raise ValueError("Media Cloud timeline point must be an object")
        interval_start = _required_date(item.get("date"), "timeline date")
        if interval_start > cutoff_date:
            raise ValueError("timeline interval is after cutoff_date")
        if interval_start in seen_intervals:
            raise ValueError("Media Cloud timeline contains duplicate intervals")
        seen_intervals.add(interval_start)
        points.append(
            MediaCloudTimelinePoint(
                interval_start=interval_start,
                matched_articles=_count(item.get("count"), "count"),
                monitored_articles=_count(item.get("total_count"), "total_count"),
            )
        )
    points.sort(key=lambda point: point.interval_start)
    return MediaCloudTimelineResult(points=tuple(points))


def _decode_object(payload: bytes) -> dict[str, object]:
    try:
        decoded = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Media Cloud response is not valid UTF-8 JSON: {error}") from error
    if not isinstance(decoded, dict):
        raise ValueError("Media Cloud response must be an object")
    return decoded


def _published_at(value: object) -> date | None:
    if value is None or value == "":
        return None
    return _required_date(value, "publish_date")


def _required_date(value: object, field_name: str) -> date:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank date")
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError as error:
        raise ValueError(f"{field_name} has an unsupported date format") from error


def _observed_at(value: object) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("indexed_date must be a non-blank timestamp")
    normalized = value.strip().replace("Z", "+00:00")
    try:
        result = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError("indexed_date has an unsupported timestamp format") from error
    if result.tzinfo is None or result.utcoffset() is None:
        result = result.replace(tzinfo=UTC)
    return result.astimezone(UTC)


def _count(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"timeline {field_name} must be a non-negative integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, float) and value.is_integer():
        result = int(value)
    elif isinstance(value, str) and value.isdigit():
        result = int(value)
    else:
        raise ValueError(f"timeline {field_name} must be a non-negative integer")
    if result < 0:
        raise ValueError(f"timeline {field_name} must be a non-negative integer")
    return result


def _publisher(record: dict[str, object]) -> str | None:
    for field_name in ("media_name", "media_url"):
        value = record.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip().casefold()
    return None


def _language(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return "und"
    normalized = value.strip().casefold()
    if len(normalized) in {2, 3} and normalized.isalpha():
        return normalized
    return "und"


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-blank string")
    return value.strip()


def _best_effort_external_id(record: object) -> str | None:
    if not isinstance(record, dict):
        return None
    value = record.get("id")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _issue_code(error: Exception) -> str:
    message = str(error)
    if "after cutoff_date" in message:
        return "after_cutoff"
    if "url" in message or "story id" in message:
        return "invalid_identifier"
    if "title" in message:
        return "missing_title"
    if "indexed_date" in message:
        return "invalid_observation_time"
    if "publish_date" in message:
        return "invalid_publication_date"
    return "invalid_record"
