"""Bounded OpenAlex discovery execution and document-pool construction."""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable
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
    HttpTransport,
    OpenAlexConnector,
    OpenAlexDiscoveryHints,
    RetrievalChannel,
    SnapshotManifest,
    SnapshotStatus,
    SnapshotWriter,
    SourceQuery,
    normalize_openalex_api_key,
    parse_openalex_response,
)

from .contracts import DiscoveryBudget, DiscoveryPlan

OPENALEX_DISCOVERY_VERSION = "openalex-discovery-v1"
_LATIN_LETTER = re.compile(r"[A-Za-z]")


class DiscoveryStopReason(StrEnum):
    CHANNELS_EXHAUSTED = "channels_exhausted"
    REQUEST_BUDGET = "request_budget"
    PAGE_BUDGET = "page_budget"
    DOCUMENT_BUDGET = "document_budget"
    ELAPSED_BUDGET = "elapsed_budget"


@dataclass(frozen=True, slots=True)
class OpenAlexSearchStep:
    channel: RetrievalChannel
    search_text: str | None = None
    page_index: int = 1

    def __post_init__(self) -> None:
        if self.channel in {RetrievalChannel.TEXT, RetrievalChannel.SEMANTIC}:
            if self.search_text is None or not self.search_text.strip():
                raise ValueError("text and semantic steps require search_text")
        elif self.channel is RetrievalChannel.TAXONOMY:
            if self.search_text is not None:
                raise ValueError("taxonomy step cannot contain search_text")
        else:
            raise ValueError("identifier retrieval is not a discovery search step")
        if self.page_index < 1:
            raise ValueError("page_index must be positive")


@dataclass(frozen=True, slots=True)
class DiscoveryParseIssue:
    connector_id: ConnectorId
    request_id: str
    record_index: int
    external_id: str | None
    code: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True, slots=True)
class DiscoveryHintIssue:
    connector_id: ConnectorId
    request_id: str
    record_index: int
    external_id: str | None
    document_id: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True, slots=True)
class DiscoveryBudgetUsage:
    connector_id: ConnectorId
    requests_used: int
    pages_used: int
    returned_records: int
    accepted_records: int
    unique_documents: int
    duplicate_documents: int
    rejected_records: int
    elapsed_ms: int
    stop_reason: DiscoveryStopReason

    def __post_init__(self) -> None:
        for name, value in (
            ("requests_used", self.requests_used),
            ("pages_used", self.pages_used),
            ("returned_records", self.returned_records),
            ("accepted_records", self.accepted_records),
            ("unique_documents", self.unique_documents),
            ("duplicate_documents", self.duplicate_documents),
            ("rejected_records", self.rejected_records),
            ("elapsed_ms", self.elapsed_ms),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.unique_documents + self.duplicate_documents != self.accepted_records:
            raise ValueError("accepted_records must equal unique plus duplicate documents")
        if self.returned_records != self.accepted_records + self.rejected_records:
            raise ValueError("returned_records must equal accepted plus rejected records")
        if self.pages_used > self.requests_used:
            raise ValueError("pages_used must not exceed requests_used")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True, slots=True)
class OpenAlexDiscoveryResult:
    plan_id: str
    manifest: SnapshotManifest
    documents: tuple[SourceDocument, ...]
    hints: tuple[OpenAlexDiscoveryHints, ...]
    issues: tuple[DiscoveryParseIssue, ...]
    hint_issues: tuple[DiscoveryHintIssue, ...]
    usage: DiscoveryBudgetUsage
    snapshot_path: Path

    @property
    def status(self) -> SnapshotStatus:
        return self.manifest.status

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "status": self.status.value,
            "manifest": self.manifest.to_dict(),
            "documents": [_json_value(asdict(document)) for document in self.documents],
            "hints": [_json_value(asdict(hint)) for hint in self.hints],
            "issues": [issue.to_dict() for issue in self.issues],
            "hint_issues": [issue.to_dict() for issue in self.hint_issues],
            "usage": self.usage.to_dict(),
            "snapshot_path": self.snapshot_path.as_posix(),
        }


# Повторы при временных сбоях (сеть, 429, 5xx): живой прогон 22.09.2026 показал,
# что OpenAlex режет частые запросы лимитом. Все попытки попадают в опись.
OPENALEX_MAX_ATTEMPTS = 3
OPENALEX_RETRY_BACKOFF_SECONDS = (5.0, 15.0)
MAX_OPENALEX_RETRY_DELAY_SECONDS = 120.0


def build_openalex_search_schedule(query: SourceQuery) -> tuple[OpenAlexSearchStep, ...]:
    """Prioritize complementary channels without multiplying every synonym."""

    canonical = query.normalized_query
    steps = [
        OpenAlexSearchStep(RetrievalChannel.TEXT, canonical),
        OpenAlexSearchStep(RetrievalChannel.SEMANTIC, canonical),
    ]
    if query.topic_ids or query.subfield_ids:
        steps.append(OpenAlexSearchStep(RetrievalChannel.TAXONOMY))
    for value in query.search_texts:
        if value.casefold() == canonical.casefold():
            continue
        if not value.isascii() or _LATIN_LETTER.search(value) is None:
            continue
        steps.append(OpenAlexSearchStep(RetrievalChannel.TEXT, value))
    return tuple(steps)


class OpenAlexDiscoveryExecutor:
    """Execute the scientific discovery part of one immutable plan."""

    def __init__(
        self,
        snapshot_root: Path = Path("runtime") / "snapshots",
        *,
        transport: HttpTransport | None = None,
        contact_email: str | None = None,
        api_key: str | None = None,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        self._snapshot_root = snapshot_root
        self._transport = transport
        self._contact_email = contact_email
        self._api_key = normalize_openalex_api_key(api_key)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._monotonic = monotonic or time.monotonic
        self._sleeper = sleeper

    def _run_with_retries(
        self,
        make_run: Callable[[int], ConnectorRun],
    ) -> tuple[tuple[ConnectorRun, ...], ConnectorRun]:
        """Repeat one connector call while the failure is retryable.

        Every attempt is returned: attempts are real requests, so the caller
        appends all of them to the snapshot manifest and usage counters.
        """

        attempts: list[ConnectorRun] = []
        for index in range(OPENALEX_MAX_ATTEMPTS):
            run = make_run(index + 1)
            attempts.append(run)
            if index >= OPENALEX_MAX_ATTEMPTS - 1:
                return tuple(attempts), run
            if (
                run.status is ConnectorStatus.SUCCESS
                or run.error is None
                or not run.error.retryable
            ):
                return tuple(attempts), run
            backoff = OPENALEX_RETRY_BACKOFF_SECONDS[
                min(index, len(OPENALEX_RETRY_BACKOFF_SECONDS) - 1)
            ]
            if run.error.retry_after_seconds is not None:
                if run.error.retry_after_seconds > MAX_OPENALEX_RETRY_DELAY_SECONDS:
                    # The failure remains retryable in a later run, but this
                    # bounded execution cannot wait out a long server delay.
                    return tuple(attempts), run
                backoff = max(backoff, run.error.retry_after_seconds)
            (self._sleeper or time.sleep)(backoff)
        return tuple(attempts), attempts[-1]

    def _page_fetch(
        self,
        connector: OpenAlexConnector,
        query: SourceQuery,
        writer: SnapshotWriter,
        *,
        channel: RetrievalChannel,
        search_text: str | None,
        page_index: int,
        per_page: int,
    ) -> Callable[[int], ConnectorRun]:
        def fetch(attempt: int) -> ConnectorRun:
            return connector.run_page(
                query,
                writer,
                channel=channel,
                search_text=search_text,
                page_index=page_index,
                per_page=per_page,
                attempt=attempt,
            )

        return fetch

    def execute(self, plan: DiscoveryPlan) -> OpenAlexDiscoveryResult:
        budget = _openalex_budget(plan)
        created_at = self._clock()
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError("clock must return timezone-aware datetimes")
        snapshot_id = _snapshot_id(plan.plan_id, created_at)
        writer = SnapshotWriter(self._snapshot_root, snapshot_id)
        connector = OpenAlexConnector(
            transport=self._transport,
            contact_email=self._contact_email,
            api_key=self._api_key,
            timeout_seconds=budget.request_timeout_seconds,
            clock=self._clock,
        )
        started = self._monotonic()
        runs: list[ConnectorRun] = []
        documents_by_origin: dict[str, SourceDocument] = {}
        hints_by_document: dict[str, OpenAlexDiscoveryHints] = {}
        issues: list[DiscoveryParseIssue] = []
        hint_issues: list[DiscoveryHintIssue] = []
        returned_records = 0
        accepted_records = 0
        duplicate_documents = 0
        pages_fetched = 0
        stop_reason = DiscoveryStopReason.CHANNELS_EXHAUSTED

        schedule = list(build_openalex_search_schedule(plan.query))
        schedule_index = 0
        while schedule_index < len(schedule):
            step = schedule[schedule_index]
            schedule_index += 1
            if len(runs) >= budget.max_requests:
                stop_reason = DiscoveryStopReason.REQUEST_BUDGET
                break
            if pages_fetched >= budget.max_pages:
                stop_reason = DiscoveryStopReason.PAGE_BUDGET
                break
            if len(documents_by_origin) >= budget.max_documents:
                stop_reason = DiscoveryStopReason.DOCUMENT_BUDGET
                break
            if runs and self._monotonic() - started >= budget.max_elapsed_seconds:
                stop_reason = DiscoveryStopReason.ELAPSED_BUDGET
                break

            remaining_documents = budget.max_documents - len(documents_by_origin)
            attempts, run = self._run_with_retries(
                self._page_fetch(
                    connector,
                    plan.query,
                    writer,
                    channel=step.channel,
                    search_text=step.search_text,
                    page_index=step.page_index,
                    per_page=min(100, remaining_documents),
                )
            )
            runs.extend(attempts)
            elapsed_exhausted = self._monotonic() - started >= budget.max_elapsed_seconds
            if run.status is not ConnectorStatus.SUCCESS or run.artifact is None:
                if elapsed_exhausted:
                    stop_reason = DiscoveryStopReason.ELAPSED_BUDGET
                    break
                continue
            pages_fetched += 1

            parsed = parse_openalex_response(
                writer.read_response(run.artifact),
                snapshot_id=snapshot_id,
                retrieved_at=run.artifact.retrieved_at,
                cutoff_date=plan.query.cutoff_date,
            )
            returned_records += parsed.total_records
            accepted_records += parsed.accepted_records
            if parsed.total_records >= min(100, remaining_documents):
                schedule.append(replace(step, page_index=step.page_index + 1))
            hint_issues.extend(
                DiscoveryHintIssue(
                    connector_id=ConnectorId.OPENALEX,
                    request_id=run.request.request_id,
                    record_index=issue.record_index,
                    external_id=issue.external_id,
                    document_id=issue.document_id,
                    message=issue.message,
                )
                for issue in parsed.hint_issues
            )
            for issue in parsed.issues:
                issues.append(
                    DiscoveryParseIssue(
                        connector_id=ConnectorId.OPENALEX,
                        request_id=run.request.request_id,
                        record_index=issue.record_index,
                        external_id=issue.external_id,
                        code=issue.code,
                        message=issue.message,
                    )
                )
            parsed_hints = {hint.document_id: hint for hint in parsed.hints}
            for document in parsed.documents:
                hints = parsed_hints[document.document_id]
                retained_document = documents_by_origin.get(document.origin_id)
                if retained_document is not None:
                    duplicate_documents += 1
                    hints_by_document[retained_document.document_id] = _merge_hints(
                        hints_by_document[retained_document.document_id],
                        hints,
                        retained_document.document_id,
                    )
                    continue
                documents_by_origin[document.origin_id] = document
                hints_by_document[document.document_id] = hints
            if self._monotonic() - started >= budget.max_elapsed_seconds:
                stop_reason = DiscoveryStopReason.ELAPSED_BUDGET
                break

        if not runs:
            raise RuntimeError("OpenAlex discovery produced no connector runs")
        finished = self._clock()
        if finished < created_at:
            raise ValueError("clock moved backwards during discovery")
        manifest = SnapshotManifest(
            snapshot_id=snapshot_id,
            snapshot_version=OPENALEX_DISCOVERY_VERSION,
            analysis_id=plan.analysis_id,
            created_at=created_at,
            query=plan.query,
            runs=tuple(runs),
        )
        snapshot_path = writer.finalize(manifest)
        usage = DiscoveryBudgetUsage(
            connector_id=ConnectorId.OPENALEX,
            requests_used=len(runs),
            pages_used=pages_fetched,
            returned_records=returned_records,
            accepted_records=accepted_records,
            unique_documents=len(documents_by_origin),
            duplicate_documents=duplicate_documents,
            rejected_records=len(issues),
            elapsed_ms=max(0, round((self._monotonic() - started) * 1000)),
            stop_reason=stop_reason,
        )
        return OpenAlexDiscoveryResult(
            plan_id=plan.plan_id,
            manifest=manifest,
            documents=tuple(documents_by_origin.values()),
            hints=tuple(hints_by_document.values()),
            issues=tuple(issues),
            hint_issues=tuple(hint_issues),
            usage=usage,
            snapshot_path=snapshot_path,
        )


def _openalex_budget(plan: DiscoveryPlan) -> DiscoveryBudget:
    budget = plan.budget_for(ConnectorId.OPENALEX)
    if budget is None:
        raise ValueError("discovery plan does not contain an OpenAlex budget")
    return budget


def _snapshot_id(plan_id: str, created_at: datetime) -> str:
    identity = f"{plan_id}|{created_at.isoformat()}"
    return f"snapshot-{hashlib.sha256(identity.encode()).hexdigest()[:16]}"


def _merge_hints(
    current: OpenAlexDiscoveryHints,
    incoming: OpenAlexDiscoveryHints,
    document_id: str,
) -> OpenAlexDiscoveryHints:
    topics = {topic.topic_id: topic for topic in current.topics}
    for topic in incoming.topics:
        existing = topics.get(topic.topic_id)
        if existing is None or topic.score > existing.score:
            topics[topic.topic_id] = replace(
                topic,
                primary=topic.primary or (existing.primary if existing else False),
            )
        elif topic.primary and not existing.primary:
            topics[topic.topic_id] = replace(existing, primary=True)
    keywords = {keyword.keyword_id: keyword for keyword in current.keywords}
    for keyword in incoming.keywords:
        existing = keywords.get(keyword.keyword_id)
        if existing is None or keyword.score > existing.score:
            keywords[keyword.keyword_id] = keyword
    return OpenAlexDiscoveryHints(
        document_id=document_id,
        topics=tuple(topics.values()),
        keywords=tuple(keywords.values()),
    )


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
