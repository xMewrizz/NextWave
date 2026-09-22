"""Bounded information discovery with Media Cloud primary and GDELT fallback."""

from __future__ import annotations

import calendar
import hashlib
import re
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

from nextwave.contracts import SourceDocument
from nextwave.sources import (
    ConnectorId,
    ConnectorRun,
    ConnectorStatus,
    GdeltConnector,
    HttpTransport,
    MediaCloudConnector,
    NewsContentStatus,
    NewsDocumentEnricher,
    NewsDocumentEnrichment,
    SnapshotManifest,
    SnapshotStatus,
    SnapshotWriter,
    SourceQuery,
    parse_gdelt_response,
    parse_mediacloud_response,
)

from .contracts import DiscoveryBudget, DiscoveryPlan
from .executor import DiscoveryBudgetUsage, DiscoveryParseIssue, DiscoveryStopReason

MEDIA_DISCOVERY_VERSION = "media-discovery-v1"
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")


class MediaFallbackReason(StrEnum):
    NOT_NEEDED = "not_needed"
    PRIMARY_UNCONFIGURED = "primary_unconfigured"
    PRIMARY_NOT_PLANNED = "primary_not_planned"
    PRIMARY_FAILED = "primary_failed"
    FALLBACK_NOT_PLANNED = "fallback_not_planned"


@dataclass(frozen=True, slots=True)
class MediaSearchStep:
    search_text: str
    languages: tuple[str, ...]
    page_index: int = 1
    pagination_token: str | None = None


@dataclass(frozen=True, slots=True)
class MediaDiscoveryResult:
    plan_id: str
    manifest: SnapshotManifest
    provider_used: ConnectorId | None
    fallback_reason: MediaFallbackReason
    documents: tuple[SourceDocument, ...]
    enrichment: tuple[NewsDocumentEnrichment, ...]
    issues: tuple[DiscoveryParseIssue, ...]
    usage: tuple[DiscoveryBudgetUsage, ...]
    snapshot_path: Path

    @property
    def status(self) -> SnapshotStatus:
        return self.manifest.status

    @property
    def title_only_documents(self) -> int:
        return sum(
            item.status is NewsContentStatus.TITLE_ONLY for item in self.enrichment
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "status": self.status.value,
            "provider_used": self.provider_used.value if self.provider_used else None,
            "fallback_reason": self.fallback_reason.value,
            "manifest": self.manifest.to_dict(),
            "documents": [_json_value(asdict(document)) for document in self.documents],
            "enrichment": [item.to_dict() for item in self.enrichment],
            "issues": [issue.to_dict() for issue in self.issues],
            "usage": [item.to_dict() for item in self.usage],
            "title_only_documents": self.title_only_documents,
            "snapshot_path": self.snapshot_path.as_posix(),
        }


def build_media_search_schedule(query: SourceQuery) -> tuple[MediaSearchStep, ...]:
    """Create separate bounded language searches without multiplying every alias."""

    cyrillic = next(
        (text for text in query.search_texts if _CYRILLIC.search(text) is not None),
        None,
    )
    steps: list[MediaSearchStep] = []
    seen: set[tuple[str, str]] = set()
    for language in query.languages:
        base_language = language.split("-", 1)[0]
        search_text = cyrillic if base_language == "ru" and cyrillic else query.normalized_query
        identity = (search_text.casefold(), base_language)
        if identity in seen:
            continue
        seen.add(identity)
        steps.append(MediaSearchStep(search_text=search_text, languages=(language,)))
    return tuple(steps)


class MediaDiscoveryExecutor:
    """Retrieve one news pool, enrich its pages, and preserve source coverage."""

    def __init__(
        self,
        snapshot_root: Path = Path("runtime") / "snapshots",
        *,
        mediacloud_api_key: str | None = None,
        mediacloud_collection_ids: tuple[int, ...] = (),
        mediacloud_transport: HttpTransport | None = None,
        gdelt_transport: HttpTransport | None = None,
        news_enricher: NewsDocumentEnricher | None = None,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
        mediacloud_min_interval_seconds: float = 30.0,
        gdelt_min_interval_seconds: float = 5.0,
    ) -> None:
        self._snapshot_root = snapshot_root
        self._mediacloud_api_key = (
            mediacloud_api_key.strip()
            if mediacloud_api_key and mediacloud_api_key.strip()
            else None
        )
        self._mediacloud_collection_ids = _validated_collection_ids(
            mediacloud_collection_ids
        )
        self._mediacloud_transport = mediacloud_transport
        self._gdelt_transport = gdelt_transport
        self._news_enricher = news_enricher or NewsDocumentEnricher()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._monotonic = monotonic or time.monotonic
        self._sleeper = sleeper
        self._mediacloud_min_interval_seconds = mediacloud_min_interval_seconds
        self._gdelt_min_interval_seconds = gdelt_min_interval_seconds

    def execute(self, plan: DiscoveryPlan) -> MediaDiscoveryResult:
        mediacloud_budget = plan.budget_for(ConnectorId.MEDIACLOUD)
        gdelt_budget = plan.budget_for(ConnectorId.GDELT)
        if mediacloud_budget is None and gdelt_budget is None:
            raise ValueError("discovery plan contains no media connector budget")

        created_at = self._clock()
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError("clock must return timezone-aware datetimes")
        snapshot_id = _media_snapshot_id(plan.plan_id, created_at)
        writer = SnapshotWriter(self._snapshot_root, snapshot_id)
        runs: list[ConnectorRun] = []
        documents_by_origin: dict[str, SourceDocument] = {}
        issues: list[DiscoveryParseIssue] = []
        usages: list[DiscoveryBudgetUsage] = []
        provider_used: ConnectorId | None = None

        if mediacloud_budget is None:
            fallback_reason = MediaFallbackReason.PRIMARY_NOT_PLANNED
            primary_success = False
        elif self._mediacloud_api_key is None:
            fallback_reason = MediaFallbackReason.PRIMARY_UNCONFIGURED
            primary_success = False
        else:
            primary = self._execute_mediacloud(
                plan,
                mediacloud_budget,
                writer,
                snapshot_id,
            )
            runs.extend(primary.runs)
            documents_by_origin.update(
                (document.origin_id, document) for document in primary.documents
            )
            issues.extend(primary.issues)
            usages.append(primary.usage)
            primary_success = primary.successful_requests > 0
            if primary_success:
                provider_used = ConnectorId.MEDIACLOUD
                fallback_reason = MediaFallbackReason.NOT_NEEDED
            else:
                fallback_reason = MediaFallbackReason.PRIMARY_FAILED

        if not primary_success:
            if gdelt_budget is None:
                if runs:
                    fallback_reason = MediaFallbackReason.FALLBACK_NOT_PLANNED
                else:
                    raise ValueError("media discovery has no executable connector")
            else:
                fallback = self._execute_gdelt(
                    plan,
                    gdelt_budget,
                    writer,
                    snapshot_id,
                )
                runs.extend(fallback.runs)
                documents_by_origin.update(
                    (document.origin_id, document) for document in fallback.documents
                )
                issues.extend(fallback.issues)
                usages.append(fallback.usage)
                provider_used = ConnectorId.GDELT

        if not runs:
            raise RuntimeError("media discovery produced no connector runs")
        manifest = SnapshotManifest(
            snapshot_id=snapshot_id,
            snapshot_version=MEDIA_DISCOVERY_VERSION,
            analysis_id=plan.analysis_id,
            created_at=created_at,
            query=plan.query,
            runs=tuple(runs),
        )
        snapshot_path = writer.finalize(manifest)
        enrichment = self._news_enricher.enrich_many(
            tuple(documents_by_origin.values())
        )
        enriched_documents = tuple(item.document for item in enrichment)
        return MediaDiscoveryResult(
            plan_id=plan.plan_id,
            manifest=manifest,
            provider_used=provider_used,
            fallback_reason=fallback_reason,
            documents=enriched_documents,
            enrichment=enrichment,
            issues=tuple(issues),
            usage=tuple(usages),
            snapshot_path=snapshot_path,
        )

    def _execute_mediacloud(
        self,
        plan: DiscoveryPlan,
        budget: DiscoveryBudget,
        writer: SnapshotWriter,
        snapshot_id: str,
    ) -> _ProviderExecution:
        connector = MediaCloudConnector(
            api_key=self._mediacloud_api_key or "",
            transport=self._mediacloud_transport,
            timeout_seconds=budget.request_timeout_seconds,
            min_interval_seconds=self._mediacloud_min_interval_seconds,
            clock=self._clock,
            monotonic_clock=self._monotonic,
            sleeper=self._sleeper,
        )
        queue = deque(build_media_search_schedule(plan.query))
        started = self._monotonic()
        runs: list[ConnectorRun] = []
        documents_by_origin: dict[str, SourceDocument] = {}
        issues: list[DiscoveryParseIssue] = []
        returned_records = 0
        accepted_records = 0
        duplicates = 0
        successful_requests = 0
        stop_reason = DiscoveryStopReason.CHANNELS_EXHAUSTED

        while queue:
            if len(runs) >= budget.max_requests:
                stop_reason = DiscoveryStopReason.REQUEST_BUDGET
                break
            if len(runs) >= budget.max_pages:
                stop_reason = DiscoveryStopReason.PAGE_BUDGET
                break
            if len(documents_by_origin) >= budget.max_documents:
                stop_reason = DiscoveryStopReason.DOCUMENT_BUDGET
                break
            if runs and self._monotonic() - started >= budget.max_elapsed_seconds:
                stop_reason = DiscoveryStopReason.ELAPSED_BUDGET
                break

            step = queue.popleft()
            remaining = budget.max_documents - len(documents_by_origin)
            run = connector.run_page(
                plan.query,
                writer,
                search_text=step.search_text,
                languages=step.languages,
                collection_ids=self._mediacloud_collection_ids,
                page_size=min(100, remaining),
                pagination_token=step.pagination_token,
                page_index=step.page_index,
            )
            runs.append(run)
            if run.status is not ConnectorStatus.SUCCESS or run.artifact is None:
                continue
            successful_requests += 1
            parsed = parse_mediacloud_response(
                writer.read_response(run.artifact),
                snapshot_id=snapshot_id,
                retrieved_at=run.artifact.retrieved_at,
                cutoff_date=plan.query.cutoff_date,
            )
            returned_records += parsed.total_records
            accepted_records += parsed.accepted_records
            issues.extend(
                _parse_issue(ConnectorId.MEDIACLOUD, run, issue)
                for issue in parsed.issues
            )
            for document in parsed.documents:
                if document.origin_id in documents_by_origin:
                    duplicates += 1
                else:
                    documents_by_origin[document.origin_id] = document
            if parsed.pagination_token and len(queue) == 0:
                queue.append(
                    replace(
                        step,
                        page_index=step.page_index + 1,
                        pagination_token=parsed.pagination_token,
                    )
                )

        usage = _usage(
            ConnectorId.MEDIACLOUD,
            runs,
            returned_records,
            accepted_records,
            len(documents_by_origin),
            duplicates,
            len(issues),
            started,
            self._monotonic,
            stop_reason,
        )
        return _ProviderExecution(
            runs=tuple(runs),
            documents=tuple(documents_by_origin.values()),
            issues=tuple(issues),
            usage=usage,
            successful_requests=successful_requests,
        )

    def _execute_gdelt(
        self,
        plan: DiscoveryPlan,
        budget: DiscoveryBudget,
        writer: SnapshotWriter,
        snapshot_id: str,
    ) -> _ProviderExecution:
        connector = GdeltConnector(
            transport=self._gdelt_transport,
            timeout_seconds=budget.request_timeout_seconds,
            min_interval_seconds=self._gdelt_min_interval_seconds,
            clock=self._clock,
            monotonic_clock=self._monotonic,
            sleeper=self._sleeper,
        )
        started = self._monotonic()
        recent_query = _gdelt_recent_query(plan.query)
        run = connector.run_window(
            recent_query,
            writer,
            search_text=plan.query.normalized_query,
            max_records=min(250, budget.max_documents),
        )
        runs = (run,)
        documents: tuple[SourceDocument, ...] = ()
        issues: tuple[DiscoveryParseIssue, ...] = ()
        returned_records = 0
        accepted_records = 0
        successful_requests = 0
        if run.status is ConnectorStatus.SUCCESS and run.artifact is not None:
            successful_requests = 1
            parsed = parse_gdelt_response(
                writer.read_response(run.artifact),
                snapshot_id=snapshot_id,
                retrieved_at=run.artifact.retrieved_at,
                cutoff_date=plan.query.cutoff_date,
            )
            returned_records = parsed.total_records
            accepted_records = parsed.accepted_records
            documents_by_origin: dict[str, SourceDocument] = {}
            duplicate_documents = 0
            for document in parsed.documents:
                if document.origin_id in documents_by_origin:
                    duplicate_documents += 1
                else:
                    documents_by_origin[document.origin_id] = document
            documents = tuple(documents_by_origin.values())
            issues = tuple(
                _parse_issue(ConnectorId.GDELT, run, issue) for issue in parsed.issues
            )
        usage = _usage(
            ConnectorId.GDELT,
            runs,
            returned_records,
            accepted_records,
            len(documents),
            duplicate_documents if successful_requests else 0,
            len(issues),
            started,
            self._monotonic,
            DiscoveryStopReason.CHANNELS_EXHAUSTED,
        )
        return _ProviderExecution(
            runs=runs,
            documents=documents,
            issues=issues,
            usage=usage,
            successful_requests=successful_requests,
        )


@dataclass(frozen=True, slots=True)
class _ProviderExecution:
    runs: tuple[ConnectorRun, ...]
    documents: tuple[SourceDocument, ...]
    issues: tuple[DiscoveryParseIssue, ...]
    usage: DiscoveryBudgetUsage
    successful_requests: int


def parse_mediacloud_collection_ids(environment: Mapping[str, str]) -> tuple[int, ...]:
    value = environment.get("NEXTWAVE_MEDIACLOUD_COLLECTION_IDS", "").strip()
    if not value:
        return ()
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise ValueError(
            "NEXTWAVE_MEDIACLOUD_COLLECTION_IDS must contain comma-separated integers"
        ) from error
    return _validated_collection_ids(parsed)


def _validated_collection_ids(values: tuple[int, ...]) -> tuple[int, ...]:
    if any(isinstance(value, bool) or value <= 0 for value in values):
        raise ValueError("Media Cloud collection IDs must be positive integers")
    if len(set(values)) != len(values):
        raise ValueError("Media Cloud collection IDs must be unique")
    return values


def _parse_issue(connector_id: ConnectorId, run: ConnectorRun, issue: Any) -> DiscoveryParseIssue:
    return DiscoveryParseIssue(
        connector_id=connector_id,
        request_id=run.request.request_id,
        record_index=issue.record_index,
        external_id=issue.external_id,
        code=issue.code,
        message=issue.message,
    )


def _usage(
    connector_id: ConnectorId,
    runs: tuple[ConnectorRun, ...] | list[ConnectorRun],
    returned_records: int,
    accepted_records: int,
    unique_documents: int,
    duplicate_documents: int,
    rejected_records: int,
    started: float,
    monotonic: Callable[[], float],
    stop_reason: DiscoveryStopReason,
) -> DiscoveryBudgetUsage:
    return DiscoveryBudgetUsage(
        connector_id=connector_id,
        requests_used=len(runs),
        pages_used=len(runs),
        returned_records=returned_records,
        accepted_records=accepted_records,
        unique_documents=unique_documents,
        duplicate_documents=duplicate_documents,
        rejected_records=rejected_records,
        elapsed_ms=max(0, round((monotonic() - started) * 1000)),
        stop_reason=stop_reason,
    )


def _media_snapshot_id(plan_id: str, created_at: datetime) -> str:
    identity = f"{plan_id}|media|{created_at.isoformat()}"
    return f"snapshot-{hashlib.sha256(identity.encode()).hexdigest()[:16]}"


def _gdelt_recent_query(query: SourceQuery) -> SourceQuery:
    earliest_supported = _subtract_months(query.published_until, 3)
    if query.published_from >= earliest_supported:
        return query
    return replace(query, published_from=earliest_supported)


def _subtract_months(value: date, months: int) -> date:
    month_index = value.month - 1 - months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, date | datetime):
        return value.isoformat()
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value
