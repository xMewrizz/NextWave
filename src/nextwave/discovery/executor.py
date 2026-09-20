"""Bounded OpenAlex discovery execution and document-pool construction."""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
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
    RetrievalChannel,
    SnapshotManifest,
    SnapshotStatus,
    SnapshotWriter,
    SourceQuery,
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
    issues: tuple[DiscoveryParseIssue, ...]
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
            "issues": [issue.to_dict() for issue in self.issues],
            "usage": self.usage.to_dict(),
            "snapshot_path": self.snapshot_path.as_posix(),
        }


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
        api_key: str | None = None,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self._snapshot_root = snapshot_root
        self._transport = transport
        self._api_key = api_key
        self._clock = clock or (lambda: datetime.now(UTC))
        self._monotonic = monotonic or time.monotonic

    def execute(self, plan: DiscoveryPlan) -> OpenAlexDiscoveryResult:
        budget = _openalex_budget(plan)
        created_at = self._clock()
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError("clock must return timezone-aware datetimes")
        snapshot_id = _snapshot_id(plan.plan_id, created_at)
        writer = SnapshotWriter(self._snapshot_root, snapshot_id)
        connector = OpenAlexConnector(
            transport=self._transport,
            api_key=self._api_key,
            timeout_seconds=budget.request_timeout_seconds,
            clock=self._clock,
        )
        started = self._monotonic()
        runs: list[ConnectorRun] = []
        documents_by_origin: dict[str, SourceDocument] = {}
        issues: list[DiscoveryParseIssue] = []
        returned_records = 0
        accepted_records = 0
        duplicate_documents = 0
        stop_reason = DiscoveryStopReason.CHANNELS_EXHAUSTED

        schedule = build_openalex_search_schedule(plan.query)
        for step in schedule:
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

            remaining_documents = budget.max_documents - len(documents_by_origin)
            run = connector.run_page(
                plan.query,
                writer,
                channel=step.channel,
                search_text=step.search_text,
                page_index=step.page_index,
                per_page=min(100, remaining_documents),
            )
            runs.append(run)
            elapsed_exhausted = self._monotonic() - started >= budget.max_elapsed_seconds
            if run.status is not ConnectorStatus.SUCCESS or run.artifact is None:
                if elapsed_exhausted:
                    stop_reason = DiscoveryStopReason.ELAPSED_BUDGET
                    break
                continue

            parsed = parse_openalex_response(
                writer.read_response(run.artifact),
                snapshot_id=snapshot_id,
                retrieved_at=run.artifact.retrieved_at,
                cutoff_date=plan.query.cutoff_date,
            )
            returned_records += parsed.total_records
            accepted_records += parsed.accepted_records
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
            for document in parsed.documents:
                if document.origin_id in documents_by_origin:
                    duplicate_documents += 1
                    continue
                documents_by_origin[document.origin_id] = document
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
            pages_used=len(runs),
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
            issues=tuple(issues),
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
