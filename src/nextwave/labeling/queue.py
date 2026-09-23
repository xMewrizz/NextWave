"""Deterministic review-queue construction from persisted discovery runs.

Pipeline proposes (objects + links), humans dispose (labels). This module never
assigns mature/marketing_hype: candidate slots (template-owned IDs, domains and
quota buckets) are filled from merged discovery groups, noise slots from mapped
discovery exhaust. Shortages become deficits that drive targeted searches.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

from ..datasets.artifacts import render_jsonl
from ..datasets.contracts import ORGANIZER_SCOPE_KEYS
from ..discovery.run_store import DiscoveryRun
from .contracts import (
    LABELING_CUTOFF_DATE,
    EvidenceDirection,
    EvidenceKind,
    NegativeCandidateRecord,
    NoiseControlRecord,
    NoiseType,
    SourceType,
    TrustLevel,
)

QUEUE_SCHEMA_VERSION = "labeling-queue-v1"
MAX_EVIDENCE_ROWS = 400

_EXCLUSION_NOISE = {
    "scope_term": NoiseType.BROAD_CONCEPT,
    "organization": NoiseType.NOT_TECHNOLOGY,
    "invalid_name": NoiseType.EXTRACTION_ERROR,
}
_GATE_REJECT_NOISE = {
    "generic_area": NoiseType.BROAD_CONCEPT,
    "organization": NoiseType.NOT_TECHNOLOGY,
    "promotional_claim": NoiseType.NOT_TECHNOLOGY,
    "irrelevant": NoiseType.IRRELEVANT,
}
_CLAIM_KIND = {
    "standard": EvidenceKind.STANDARD_ADOPTION,
    "adoption": EvidenceKind.SERIAL_DEPLOYMENT,
    "market": EvidenceKind.ESTABLISHED_MARKET,
    "pilot": EvidenceKind.PILOT,
    "research": EvidenceKind.TECHNICAL_VALIDATION,
    "patent": EvidenceKind.TECHNICAL_VALIDATION,
    "prototype": EvidenceKind.TECHNICAL_VALIDATION,
    "novelty": EvidenceKind.TECHNICAL_VALIDATION,
    "investment": EvidenceKind.PUBLICITY_WAVE,
    "growth": EvidenceKind.PUBLICITY_WAVE,
    "promotional_claim": EvidenceKind.PUBLICITY_WAVE,
}
_SOURCE_TYPE = {
    "scientific_publication": SourceType.RESEARCH,
    "patent": SourceType.PATENT,
    "standard": SourceType.STANDARD,
    "regulator": SourceType.REGULATOR,
    "university": SourceType.OFFICIAL_TECHNICAL,
    "company_technical": SourceType.OFFICIAL_TECHNICAL,
    "analytical_report": SourceType.ANALYTICAL_REPORT,
    "industry_media": SourceType.INDUSTRY_MEDIA,
    "press_release": SourceType.PRESS_RELEASE,
    "social_or_blog": SourceType.SOCIAL,
    "other": SourceType.OTHER,
}
_TRUST_LEVEL = {
    "A": TrustLevel.A,
    "B": TrustLevel.B,
    "C": TrustLevel.C,
    "D": TrustLevel.D,
}


@dataclass(frozen=True, slots=True)
class CandidateSlot:
    """One template row awaiting a technology: ID, domain and quota bucket."""

    candidate_id: str
    domain: str
    planned_class: str


@dataclass(frozen=True, slots=True)
class NoiseSlot:
    """One template noise row awaiting a control example."""

    noise_id: str
    planned_noise_type: str
    domain: str


@dataclass(frozen=True, slots=True)
class QueuedCandidate:
    candidate_id: str
    planned_class: str
    canonical_name: str
    aliases: tuple[str, ...]
    group_id: str
    source_query: str
    domain: str
    analysis_scope_key: str
    run_id: str


@dataclass(frozen=True, slots=True)
class QueuedNoise:
    noise_id: str
    planned_noise_type: str
    source_query: str
    extracted_text: str
    source_document_url: str
    domain: str
    analysis_scope_key: str
    duplicate_of_candidate_id: str | None
    origin_kind: str


@dataclass(frozen=True, slots=True)
class QueuedEvidence:
    evidence_id: str
    candidate_id: str
    direction: EvidenceDirection
    kind: EvidenceKind
    source_type: SourceType
    trust_level: TrustLevel | None
    title: str
    url: str
    published_at: date | None
    organization: str
    origin_id: str
    claim: str
    locator: str
    extraction_confidence: float | None


@dataclass(frozen=True, slots=True)
class QueueDeficit:
    area: str
    need: str
    missing: int


@dataclass(frozen=True, slots=True)
class QueueOverflow:
    kind: str
    key: str
    run_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class QueueDropped:
    reason: str
    detail: str


@dataclass(frozen=True, slots=True)
class LabelingQueue:
    candidates: tuple[QueuedCandidate, ...]
    noise: tuple[QueuedNoise, ...]
    evidence: tuple[QueuedEvidence, ...]
    deficits: tuple[QueueDeficit, ...]
    overflow: tuple[QueueOverflow, ...]
    dropped: tuple[QueueDropped, ...]
    unknown_trust_count: int
    missing_date_count: int
    future_evidence_count: int
    schema_version: str = QUEUE_SCHEMA_VERSION


def _scope_key(domain: str) -> str:
    try:
        return ORGANIZER_SCOPE_KEYS[domain]
    except KeyError as error:
        raise ValueError(f"queue slot domain is not an organizer domain: {domain!r}") from error


def _check_cutoff(manifest: Mapping[str, Any], run_id: str) -> None:
    if manifest.get("cutoff_date") != LABELING_CUTOFF_DATE.isoformat():
        raise ValueError(
            f"run {run_id} cutoff {manifest.get('cutoff_date')!r} "
            "is not eligible for labeling"
        )


def _clean_aliases(canonical_name: str, aliases: Any) -> tuple[str, ...]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for alias in aliases or ():
        text = alias.strip() if isinstance(alias, str) else ""
        if not text or text.casefold() == canonical_name.casefold():
            continue
        if text.casefold() in seen:
            continue
        seen.add(text.casefold())
        cleaned.append(text)
    return tuple(cleaned)


def _parse_date(value: Any) -> date | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def _documents_by_id(result: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    scientific = result.get("scientific") or {}
    media = result.get("media") or {}
    for section in (scientific, media):
        for document in section.get("documents") or []:
            if isinstance(document, dict) and document.get("document_id"):
                documents.setdefault(document["document_id"], document)
    verification = result.get("verification") or {}
    for item in verification.get("results") or []:
        for document in item.get("matching_documents") or []:
            if isinstance(document, dict) and document.get("document_id"):
                documents.setdefault(document["document_id"], document)
    return documents


def _merged_groups(
    runs: tuple[DiscoveryRun, ...],
) -> list[tuple[dict[str, Any], str, str]]:
    """Merge accepted alias groups across runs: (group, run_id, raw_query)."""

    merged: dict[str, tuple[dict[str, Any], str, str]] = {}
    for run in runs:
        _check_cutoff(run.manifest, run.run_id)
        raw_query = ((run.plan.get("scope") or {}).get("raw_query") or "").strip()
        groups = (run.result.get("alias_resolution") or {}).get("groups") or []
        for group in groups:
            key = group.get("normalization_key") or group.get("group_id")
            if not key or key in merged:
                continue
            merged[key] = (group, run.run_id, raw_query)
    ordered = sorted(
        merged.values(),
        key=lambda item: (
            -len(item[0].get("origin_ids") or []),
            -len(item[0].get("document_ids") or []),
            (item[0].get("canonical_name") or "").casefold(),
            item[0].get("group_id") or "",
        ),
    )
    return ordered


def _fill_candidates(
    merged: list[tuple[dict[str, Any], str, str]],
    slots: tuple[CandidateSlot, ...],
    overflow: list[QueueOverflow],
    deficits: list[QueueDeficit],
) -> list[QueuedCandidate]:
    queued: list[QueuedCandidate] = []
    for slot, (group, run_id, raw_query) in zip(slots, merged, strict=False):
        queued.append(
            QueuedCandidate(
                candidate_id=slot.candidate_id,
                planned_class=slot.planned_class,
                canonical_name=(group.get("canonical_name") or "").strip(),
                aliases=_clean_aliases(
                    (group.get("canonical_name") or ""), group.get("aliases")
                ),
                group_id=(group.get("group_id") or "").strip(),
                source_query=raw_query,
                domain=slot.domain,
                analysis_scope_key=_scope_key(slot.domain),
                run_id=run_id,
            )
        )
    for group, run_id, _ in merged[len(slots) :]:
        overflow.append(
            QueueOverflow(
                kind="candidate",
                key=str(group.get("group_id") or group.get("canonical_name")),
                run_id=run_id,
                reason="no free candidate slot",
            )
        )
    missing: dict[tuple[str, str], int] = {}
    for slot in slots[len(merged) :]:
        key = (slot.domain, slot.planned_class)
        missing[key] = missing.get(key, 0) + 1
    for (domain, planned_class), count in sorted(missing.items()):
        deficits.append(QueueDeficit(area=domain, need=planned_class, missing=count))
    return queued


def _noise_pool(
    runs: tuple[DiscoveryRun, ...],
    dropped: list[QueueDropped],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Collect noise candidates per type plus the gate_review backfill pool."""

    pool: dict[str, list[dict[str, Any]]] = {}
    review_pool: list[dict[str, Any]] = []

    def documents_of(run: DiscoveryRun) -> dict[str, dict[str, Any]]:
        return _documents_by_id(run.result)

    def raw_query_of(run: DiscoveryRun) -> str:
        return ((run.plan.get("scope") or {}).get("raw_query") or "").strip()

    def add(
        noise_type: NoiseType,
        *,
        run: DiscoveryRun,
        text: Any,
        url: Any,
        origin_kind: str,
        duplicate_of: str | None = None,
    ) -> None:
        cleaned = text.strip() if isinstance(text, str) else ""
        link = url.strip() if isinstance(url, str) else ""
        if not cleaned:
            dropped.append(QueueDropped(reason="blank_text", detail=origin_kind))
            return
        if not link:
            dropped.append(
                QueueDropped(reason="no_url", detail=f"{origin_kind}:{cleaned[:60]}")
            )
            return
        pool.setdefault(noise_type.value, []).append(
            {
                "text": cleaned,
                "url": link,
                "run_id": run.run_id,
                "raw_query": raw_query_of(run),
                "origin_kind": origin_kind,
                "duplicate_of": duplicate_of,
            }
        )

    for run in runs:
        _check_cutoff(run.manifest, run.run_id)
        documents = documents_of(run)
        proposals = {
        item.get("proposal_id"): item
        for item in ((run.result.get("candidate_proposals") or {}).get("proposals") or [])
    }
        gate = run.result.get("candidate_gate") or {}
        for exclusion in (run.result.get("candidate_proposals") or {}).get("exclusions") or []:
            reason = exclusion.get("reason")
            if reason not in _EXCLUSION_NOISE:
                raise ValueError(f"unknown exclusion reason: {reason!r}")
            document = documents.get(exclusion.get("document_id") or "")
            add(
                _EXCLUSION_NOISE[reason],
                run=run,
                text=exclusion.get("display_name"),
                url=(document or {}).get("url"),
                origin_kind="exclusion",
            )
        for decision in gate.get("decisions") or []:
            if decision.get("decision") == "accept":
                continue
            proposal = proposals.get(decision.get("proposal_id")) or {}
            basis = decision.get("basis_document_ids") or []
            document = documents.get(basis[0]) if basis else None
            if decision.get("decision") == "review":
                review_pool.append(
                    {
                        "text": (proposal.get("canonical_name") or "").strip(),
                        "url": ((document or {}).get("url") or ""),
                        "run_id": run.run_id,
                        "raw_query": raw_query_of(run),
                        "origin_kind": "gate_review",
                        "duplicate_of": None,
                    }
                )
                continue
            reason = decision.get("reason")
            if reason not in _GATE_REJECT_NOISE:
                raise ValueError(f"unknown gate reject reason: {reason!r}")
            add(
                _GATE_REJECT_NOISE[reason],
                run=run,
                text=proposal.get("canonical_name"),
                url=(document or {}).get("url"),
                origin_kind="gate_reject",
            )
        for issue in (run.result.get("text_extraction") or {}).get("issues") or []:
            document = documents.get(issue.get("document_id") or "")
            add(
                NoiseType.EXTRACTION_ERROR,
                run=run,
                text=issue.get("text"),
                url=(document or {}).get("url"),
                origin_kind="extraction_issue",
            )
        for suggestion in (
            (run.result.get("alias_resolution") or {}).get("review_suggestions") or []
        ):
            left = suggestion.get("left_group_id")
            right = suggestion.get("right_group_id")
            if not left or not right:
                dropped.append(
                    QueueDropped(reason="bad_suggestion_shape", detail=run.run_id)
                )
                continue
            right_name = None
            right_run = run.run_id
            for candidate_run in runs:
                groups = (candidate_run.result.get("alias_resolution") or {}).get("groups") or []
                for group in groups:
                    if group.get("group_id") == right:
                        right_name = group.get("canonical_name")
                        right_run = candidate_run.run_id
                        break
            if not right_name:
                dropped.append(
                    QueueDropped(reason="unknown_alias_side", detail=str(right))
                )
                continue
            pool.setdefault(NoiseType.DUPLICATE.value, []).append(
                {
                    "text": str(right_name).strip(),
                    "url": None,
                    "run_id": right_run,
                    "raw_query": raw_query_of(run),
                    "origin_kind": "alias_suggestion",
                    "duplicate_of": None,
                }
            )
    return pool, review_pool


def _fill_noise(
    pool: dict[str, list[dict[str, Any]]],
    review_pool: list[dict[str, Any]],
    slots: tuple[NoiseSlot, ...],
    dropped: list[QueueDropped],
    overflow: list[QueueOverflow],
    deficits: list[QueueDeficit],
) -> list[QueuedNoise]:
    for items in pool.values():
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for item in items:
            identity = item["text"].casefold()
            if identity in seen:
                continue
            seen.add(identity)
            unique.append(item)
        items[:] = unique
    queued: list[QueuedNoise] = []
    by_type: dict[str, list[dict[str, Any]]] = {}
    for slot in slots:
        by_type.setdefault(slot.planned_noise_type, []).append(slot)
    for noise_type, type_slots in by_type.items():
        items = pool.get(noise_type, [])
        for slot in type_slots:
            if not items:
                break
            item = items.pop(0)
            if item["origin_kind"] == "alias_suggestion" and not item["url"]:
                document_url = ""
            else:
                document_url = item["url"]
            if not document_url:
                dropped.append(
                    QueueDropped(reason="no_url", detail=f"alias_suggestion:{item['text'][:60]}")
                )
                continue
            queued.append(
                QueuedNoise(
                    noise_id=slot.noise_id,
                    planned_noise_type=noise_type,
                    source_query=item["raw_query"],
                    extracted_text=item["text"],
                    source_document_url=document_url,
                    domain=slot.domain,
                    analysis_scope_key=_scope_key(slot.domain),
                    duplicate_of_candidate_id=item["duplicate_of"],
                    origin_kind=item["origin_kind"],
                )
            )
    short: dict[str, int] = {}
    for noise_type, type_slots in by_type.items():
        filled = sum(1 for item in queued if item.planned_noise_type == noise_type)
        if filled < len(type_slots):
            short[noise_type] = len(type_slots) - filled
    for noise_type in sorted(short):
        while short[noise_type] > 0 and review_pool:
            item = review_pool.pop(0)
            if not item["text"] or not item["url"]:
                dropped.append(
                    QueueDropped(reason="no_url", detail=f"gate_review:{item['text'][:60]}")
                )
                continue
            slot = next(
                slot
                for slot in by_type[noise_type]
                if slot.noise_id
                not in {item_queued.noise_id for item_queued in queued}
            )
            queued.append(
                QueuedNoise(
                    noise_id=slot.noise_id,
                    planned_noise_type=noise_type,
                    source_query=item["raw_query"],
                    extracted_text=item["text"],
                    source_document_url=item["url"],
                    domain=slot.domain,
                    analysis_scope_key=_scope_key(slot.domain),
                    duplicate_of_candidate_id=None,
                    origin_kind="gate_review",
                )
            )
            short[noise_type] -= 1
        if short[noise_type] > 0:
            deficits.append(
                QueueDeficit(area=noise_type, need="noise", missing=short[noise_type])
            )
    for noise_type, items in pool.items():
        for item in items:
            overflow.append(
                QueueOverflow(
                    kind="noise",
                    key=f"{noise_type}:{item['text'][:60]}",
                    run_id=item["run_id"],
                    reason="no free noise slot",
                )
            )
    return queued


def _fill_evidence(
    runs: tuple[DiscoveryRun, ...],
    queued_candidates: list[QueuedCandidate],
    counters: dict[str, int],
) -> list[QueuedEvidence]:
    collected: list[QueuedEvidence] = []
    evidence_number = 0

    def organization_of(document: Mapping[str, Any], claim: Mapping[str, Any]) -> str:
        claimed = claim.get("organization")
        if isinstance(claimed, str) and claimed.strip():
            return claimed.strip()
        organizations = document.get("organizations") or []
        for entry in organizations:
            if isinstance(entry, str) and entry.strip():
                return entry.strip()
        return ""

    for candidate in queued_candidates:
        support: list[QueuedEvidence] = []
        counter: list[QueuedEvidence] = []
        for run in runs:
            _check_cutoff(run.manifest, run.run_id)
            documents = _documents_by_id(run.result)
            proposals = (run.result.get("evidence_extraction") or {}).get("proposals") or []
            for proposal in proposals:
                if proposal.get("alias_group_id") != candidate.group_id:
                    continue
                claim = proposal.get("claim") or {}
                direction = claim.get("direction")
                if direction not in ("support", "counter"):
                    raise ValueError(f"unknown evidence direction: {direction!r}")
                kind_key = claim.get("claim_type")
                if kind_key not in _CLAIM_KIND:
                    raise ValueError(f"unknown claim kind: {kind_key!r}")
                document = documents.get(claim.get("document_id") or "") or {}
                source_key = document.get("source_type") or "other"
                if source_key not in _SOURCE_TYPE:
                    raise ValueError(f"unknown source type: {source_key!r}")
                trust_key = document.get("trust_tier")
                trust_level = _TRUST_LEVEL.get(trust_key)
                if trust_level is None:
                    counters["unknown_trust"] += 1
                published = _parse_date(document.get("published_at"))
                if published is None:
                    counters["missing_date"] += 1
                elif published > LABELING_CUTOFF_DATE:
                    counters["future_evidence"] += 1
                    continue
                confidence = claim.get("extraction_confidence")
                entry = QueuedEvidence(
                    evidence_id="",
                    candidate_id=candidate.candidate_id,
                    direction=(
                        EvidenceDirection.SUPPORT
                        if direction == "support"
                        else EvidenceDirection.COUNTER
                    ),
                    kind=_CLAIM_KIND[kind_key],
                    source_type=_SOURCE_TYPE[source_key],
                    trust_level=trust_level,
                    title=str(document.get("title") or ""),
                    url=str(
                        proposal.get("source_url") or document.get("url") or ""
                    ),
                    published_at=published,
                    organization=organization_of(document, claim),
                    origin_id=str(
                        proposal.get("origin_id") or document.get("origin_id") or ""
                    ),
                    claim=str(claim.get("text") or ""),
                    locator=str(claim.get("locator") or ""),
                    extraction_confidence=(
                        float(confidence) if isinstance(confidence, (int, float)) else None
                    ),
                )
                (support if direction == "support" else counter).append(entry)
        for pair in zip(support, counter, strict=False):
            for entry in pair:
                if len(collected) >= MAX_EVIDENCE_ROWS:
                    break
                evidence_number += 1
                collected.append(_replace_evidence_id(entry, evidence_number))
            if len(collected) >= MAX_EVIDENCE_ROWS:
                break
        leftovers = support[len(counter) :] + counter[len(support) :]
        for entry in leftovers:
            if len(collected) >= MAX_EVIDENCE_ROWS:
                break
            evidence_number += 1
            collected.append(_replace_evidence_id(entry, evidence_number))
        if len(collected) >= MAX_EVIDENCE_ROWS:
            break
    return collected


def _replace_evidence_id(entry: QueuedEvidence, number: int) -> QueuedEvidence:
    from dataclasses import replace

    return replace(entry, evidence_id=f"evidence-{number:03d}")


def build_labeling_queue(
    runs: tuple[DiscoveryRun, ...],
    *,
    candidate_slots: tuple[CandidateSlot, ...] = (),
    noise_slots: tuple[NoiseSlot, ...] = (),
) -> LabelingQueue:
    """Build the deterministic review queue from persisted runs and template slots."""

    ordered_runs = tuple(sorted(runs, key=lambda run: run.run_id))
    overflow: list[QueueOverflow] = []
    deficits: list[QueueDeficit] = []
    dropped: list[QueueDropped] = []
    merged = _merged_groups(ordered_runs)
    queued_candidates = _fill_candidates(merged, candidate_slots, overflow, deficits)
    pool, review_pool = _noise_pool(ordered_runs, dropped)
    queued_noise = _fill_noise(pool, review_pool, noise_slots, dropped, overflow, deficits)
    counters = {"unknown_trust": 0, "missing_date": 0, "future_evidence": 0}
    queued_evidence = _fill_evidence(ordered_runs, queued_candidates, counters)
    return LabelingQueue(
        candidates=tuple(queued_candidates),
        noise=tuple(queued_noise),
        evidence=tuple(queued_evidence),
        deficits=tuple(deficits),
        overflow=tuple(overflow),
        dropped=tuple(dropped),
        unknown_trust_count=counters["unknown_trust"],
        missing_date_count=counters["missing_date"],
        future_evidence_count=counters["future_evidence"],
    )


def queue_to_jsonl(queue: LabelingQueue) -> tuple[bytes, bytes]:
    """Render candidate and noise records through the labeling contracts."""

    negative = [
        NegativeCandidateRecord(
            candidate_id=item.candidate_id,
            canonical_name=item.canonical_name,
            aliases=item.aliases,
            group_id=item.group_id,
            source_query=item.source_query,
            domain=item.domain,
            analysis_scope_key=item.analysis_scope_key,
            cutoff_date=LABELING_CUTOFF_DATE,
        )
        for item in queue.candidates
    ]
    noise = [
        NoiseControlRecord(
            noise_id=item.noise_id,
            source_query=item.source_query,
            extracted_text=item.extracted_text,
            source_document_url=item.source_document_url,
            domain=item.domain,
            analysis_scope_key=item.analysis_scope_key,
            cutoff_date=LABELING_CUTOFF_DATE,
            duplicate_of_candidate_id=item.duplicate_of_candidate_id,
        )
        for item in queue.noise
    ]
    negative_bytes = render_jsonl(negative) if negative else b""
    noise_bytes = render_jsonl(noise) if noise else b""
    return negative_bytes, noise_bytes
