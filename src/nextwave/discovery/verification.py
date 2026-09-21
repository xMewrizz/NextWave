"""Bounded scientific verification searches for accepted alias groups."""

from __future__ import annotations

import hashlib
import re
import time
import unicodedata
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

from nextwave.contracts import SourceDocument
from nextwave.sources import (
    ConnectorStatus,
    HttpTransport,
    OpenAlexConnector,
    OpenAlexDiscoveryHints,
    QueryPurpose,
    RetrievalChannel,
    SnapshotManifest,
    SnapshotWriter,
    SourceQuery,
    parse_openalex_response,
)

from .alias_resolution import AliasResolutionResult, ResolvedAliasGroup
from .contracts import DiscoveryPlan

VERIFICATION_VERSION = "candidate-verification-v1"
DEFAULT_VERIFICATION_MAX_GROUPS = 30
DEFAULT_VERIFICATION_MAX_RECORDS = 20


class VerificationStatus(StrEnum):
    SEARCHED = "searched"
    FAILED = "failed"
    SKIPPED_BUDGET = "skipped_budget"


@dataclass(frozen=True, slots=True)
class GroupVerification:
    group_id: str
    status: VerificationStatus
    query_text: str
    returned_records: int | None
    matching_documents: tuple[SourceDocument, ...]
    snapshot_path: Path | None
    error_code: str | None = None
    rejected_record_codes: tuple[str, ...] = ()

    @property
    def matching_origin_count(self) -> int:
        return len({document.origin_id for document in self.matching_documents})

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "status": self.status.value,
            "query_text": self.query_text,
            "returned_records": self.returned_records,
            "matching_documents": [
                _json_value(asdict(document)) for document in self.matching_documents
            ],
            "matching_origin_count": self.matching_origin_count,
            "snapshot_path": self.snapshot_path.as_posix() if self.snapshot_path else None,
            "error_code": self.error_code,
            "rejected_record_codes": list(self.rejected_record_codes),
        }


@dataclass(frozen=True, slots=True)
class CandidateVerificationResult:
    plan_id: str
    results: tuple[GroupVerification, ...]
    requests_used: int
    version: str = VERIFICATION_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "results": [item.to_dict() for item in self.results],
            "requests_used": self.requests_used,
            "version": self.version,
        }


class CandidateVerificationExecutor:
    """Run one auditable OpenAlex text query per accepted group within a global cap."""

    def __init__(
        self,
        snapshot_root: Path = Path("runtime") / "snapshots",
        *,
        transport: HttpTransport | None = None,
        api_key: str | None = None,
        max_groups: int = DEFAULT_VERIFICATION_MAX_GROUPS,
        max_records_per_group: int = DEFAULT_VERIFICATION_MAX_RECORDS,
        request_timeout_seconds: float = 20.0,
        max_elapsed_seconds: float = 120.0,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        if not 1 <= max_groups <= 100:
            raise ValueError("max_groups must be between 1 and 100")
        if not 1 <= max_records_per_group <= 100:
            raise ValueError("max_records_per_group must be between 1 and 100")
        if request_timeout_seconds <= 0 or max_elapsed_seconds < request_timeout_seconds:
            raise ValueError("verification time budgets are invalid")
        self._snapshot_root = snapshot_root
        self._transport = transport
        self._api_key = api_key
        self._max_groups = max_groups
        self._max_records = max_records_per_group
        self._request_timeout = request_timeout_seconds
        self._max_elapsed = max_elapsed_seconds
        self._clock = clock or (lambda: datetime.now(UTC))
        self._monotonic = monotonic or time.monotonic

    def execute(
        self,
        plan: DiscoveryPlan,
        aliases: AliasResolutionResult,
    ) -> CandidateVerificationResult:
        if aliases.analysis_scope_id != plan.scope.scope_id:
            raise ValueError("alias groups must match the discovery scope")
        order = {value: index for index, value in enumerate(aliases.input_proposal_ids)}
        groups = sorted(
            aliases.groups,
            key=lambda group: min(order[proposal_id] for proposal_id in group.proposal_ids),
        )
        started = self._monotonic()
        results: list[GroupVerification] = []
        requests_used = 0
        for group in groups:
            remaining_seconds = self._max_elapsed - (self._monotonic() - started)
            if requests_used >= self._max_groups or remaining_seconds <= 0:
                results.append(GroupVerification(
                    group.group_id,
                    VerificationStatus.SKIPPED_BUDGET,
                    group.canonical_name,
                    None,
                    (),
                    None,
                ))
                continue
            results.append(self._search_one(
                plan, group, timeout_seconds=min(self._request_timeout, remaining_seconds)
            ))
            requests_used += 1
        return CandidateVerificationResult(plan.plan_id, tuple(results), requests_used)

    def _search_one(
        self,
        plan: DiscoveryPlan,
        group: ResolvedAliasGroup,
        *,
        timeout_seconds: float,
    ) -> GroupVerification:
        created_at = self._clock()
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError("clock must return timezone-aware datetimes")
        query = _verification_query(plan, group)
        snapshot_id = "snapshot-" + hashlib.sha256(
            f"{query.query_id}|{created_at.isoformat()}".encode()
        ).hexdigest()[:16]
        writer = SnapshotWriter(self._snapshot_root, snapshot_id)
        connector = OpenAlexConnector(
            transport=self._transport,
            api_key=self._api_key,
            timeout_seconds=timeout_seconds,
            clock=self._clock,
        )
        run = connector.run_page(
            query,
            writer,
            channel=RetrievalChannel.TEXT,
            search_text=group.canonical_name,
            per_page=self._max_records,
        )
        manifest = SnapshotManifest(
            snapshot_id=snapshot_id,
            snapshot_version=VERIFICATION_VERSION,
            analysis_id=plan.analysis_id,
            created_at=created_at,
            query=query,
            runs=(run,),
        )
        snapshot_path = writer.finalize(manifest)
        if run.status is not ConnectorStatus.SUCCESS or run.artifact is None:
            return GroupVerification(
                group.group_id,
                VerificationStatus.FAILED,
                group.canonical_name,
                None,
                (),
                snapshot_path,
                run.error.code if run.error else "unknown_error",
            )
        try:
            parsed = parse_openalex_response(
                (snapshot_path / run.artifact.uri).read_bytes(),
                snapshot_id=snapshot_id,
                retrieved_at=run.artifact.retrieved_at,
                cutoff_date=plan.query.cutoff_date,
            )
        except (OSError, ValueError, TypeError):
            return GroupVerification(
                group.group_id,
                VerificationStatus.FAILED,
                group.canonical_name,
                run.returned_records,
                (),
                snapshot_path,
                "parse_error",
            )
        hints_by_id = {hint.document_id: hint for hint in parsed.hints}
        matched_by_origin: dict[str, SourceDocument] = {}
        names = (group.canonical_name, *group.aliases)
        for document in parsed.documents:
            if _matches_name(document, hints_by_id.get(document.document_id), names):
                matched_by_origin.setdefault(document.origin_id, document)
        return GroupVerification(
            group.group_id,
            VerificationStatus.SEARCHED,
            group.canonical_name,
            parsed.total_records,
            tuple(matched_by_origin.values()),
            snapshot_path,
            rejected_record_codes=tuple(issue.code for issue in parsed.issues),
        )


def _verification_query(plan: DiscoveryPlan, group: ResolvedAliasGroup) -> SourceQuery:
    identity = f"{plan.plan_id}|{group.group_id}|{VERIFICATION_VERSION}"
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return SourceQuery(
        query_id=f"query-verification-{digest}",
        analysis_scope_id=plan.scope.scope_id,
        purpose=QueryPurpose.VERIFICATION,
        raw_query=group.canonical_name,
        normalized_query=group.canonical_name,
        search_texts=(group.canonical_name,),
        published_from=plan.query.published_from,
        published_until=plan.query.published_until,
        cutoff_date=plan.query.cutoff_date,
        languages=plan.query.languages,
    )


def _matches_name(
    document: SourceDocument,
    hints: OpenAlexDiscoveryHints | None,
    names: tuple[str, ...],
) -> bool:
    fields = [document.title, document.excerpt or ""]
    if hints is not None:
        fields.extend(item.display_name for item in hints.keywords)
        fields.extend(item.display_name for item in hints.topics)
    normalized_fields = [_normalized_phrase(field) for field in fields]
    return any(
        re.search(rf"(?<!\w){re.escape(_normalized_phrase(name))}(?!\w)", field)
        for name in names
        for field in normalized_fields
    )


def _normalized_phrase(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.sub(r"[-‐‑‒–—−_]+", " ", normalized).split())


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value
