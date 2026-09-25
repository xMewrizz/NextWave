"""Deterministic review-queue construction from persisted discovery runs.

Pipeline proposes objects and links; humans assign labels. Candidate slots are
neutral review rows with template-owned IDs and domains. Class quotas are
checked against expert decisions, never used to pre-classify queue entries.
"""

from __future__ import annotations

import re
import unicodedata
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

QUEUE_SCHEMA_VERSION = "labeling-queue-v3"
MAX_EVIDENCE_ROWS = 400

_SCIENTIFIC_CONNECTORS = frozenset({"openalex", "crossref"})
_MEDIA_CONNECTORS = frozenset({"mediacloud", "gdelt"})
_ALLOWED_MEDIA_PROVIDERS = ("mediacloud", "gdelt")
_STRATA_ORDER = ("cross_source", "media_present", "multi_origin", "single_origin")
_QUEUE_IDENTITY_SEPARATORS = re.compile(r"[-‐‑‒–—−_]+")
_QUEUE_IDENTITY_WORD = re.compile(r"[^\W_]+", re.UNICODE)

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
    """One neutral template row awaiting a technology from the same domain."""

    candidate_id: str
    domain: str


@dataclass(frozen=True, slots=True)
class NoiseSlot:
    """One template noise row awaiting a control example."""

    noise_id: str
    planned_noise_type: str
    domain: str


@dataclass(frozen=True, slots=True)
class QueuedCandidate:
    candidate_id: str
    canonical_name: str
    aliases: tuple[str, ...]
    group_id: str
    source_query: str
    domain: str
    analysis_scope_key: str
    run_id: str
    selection_stratum: str = "single_origin"


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
class QueueDuplicateProposal:
    """One queue-level spelling duplicate awaiting manual review.

    A proposal, not a label: the dropped spelling may fill a ``duplicate``
    noise slot pointing at the queued primary, or stay in dropped/overflow
    when the link cannot be built honestly.
    """

    dropped_group_id: str
    dropped_canonical: str
    dropped_document_ids: tuple[str, ...]
    primary_group_id: str
    run_id: str
    domain: str
    source_query: str
    overflow_key: str = ""
    reason: str = "equivalent queue identity"


@dataclass(frozen=True, slots=True)
class RunSearchCoverage:
    """Completed source classes searched for one discovery run."""

    run_id: str
    raw_query: str
    source_classes: tuple[str, ...]
    notes: str = ""


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
    search_coverage: tuple[RunSearchCoverage, ...] = ()
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


def resolve_run_domains(
    runs: tuple[DiscoveryRun, ...],
    run_domains: Mapping[str, str] | None,
) -> dict[str, str]:
    """Map every run to one controlled organizer domain.

    The run vault stores a free-text domain, so an explicit audited mapping
    is required. The resolved domain — never the template slot domain —
    is stamped onto every queued row.
    """
    overrides = dict(run_domains or {})
    resolved: dict[str, str] = {}
    for run in runs:
        vault_domain = run.manifest.get("domain")
        if isinstance(vault_domain, str) and vault_domain in ORGANIZER_SCOPE_KEYS:
            resolved[run.run_id] = vault_domain
            continue
        override = overrides.get(run.run_id)
        if override in ORGANIZER_SCOPE_KEYS:
            resolved[run.run_id] = override
            continue
        raise ValueError(
            f"run {run.run_id} has no controlled domain "
            f"(vault has {vault_domain!r}); re-run discovery with "
            f"--domain set to one of {sorted(ORGANIZER_SCOPE_KEYS)} or pass "
            f"--domain-map {run.run_id}=<Domain>"
        )
    return resolved


def _merged_groups(
    runs: tuple[DiscoveryRun, ...],
    domains: Mapping[str, str],
) -> list[tuple[dict[str, Any], str, str, str]]:
    """Merge accepted alias groups across runs: (group, run_id, raw_query, domain)."""

    merged: dict[tuple[str, str], tuple[dict[str, Any], str, str, str]] = {}
    for run in runs:
        _check_cutoff(run.manifest, run.run_id)
        raw_query = ((run.plan.get("scope") or {}).get("raw_query") or "").strip()
        groups = (run.result.get("alias_resolution") or {}).get("groups") or []
        for group in groups:
            group_key = group.get("normalization_key") or group.get("group_id")
            key = (domains[run.run_id], group_key)
            if not group_key or key in merged:
                continue
            merged[key] = (group, run.run_id, raw_query, domains[run.run_id])
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


def _selection_stratum(group: Mapping[str, Any]) -> str:
    """One observable stratum per accepted group (no evidence/labels)."""

    raw_connectors = group.get("connector_ids") or []
    connectors = {
        str(item).casefold()
        for item in raw_connectors
        if isinstance(item, str) and str(item).strip()
    }
    has_scientific = bool(connectors & _SCIENTIFIC_CONNECTORS)
    has_media = bool(connectors & _MEDIA_CONNECTORS)
    if has_scientific and has_media:
        return "cross_source"
    if has_media:
        return "media_present"
    raw_origins = group.get("origin_ids") or []
    unique_origins = {
        str(item).strip()
        for item in raw_origins
        if isinstance(item, str) and str(item).strip()
    }
    if len(unique_origins) >= 2:
        return "multi_origin"
    return "single_origin"


def _rank_key(item: tuple[dict[str, Any], str, str, str]) -> tuple[int, int, str, str]:
    group = item[0]
    origin_count = len(
        {
            str(value).strip()
            for value in (group.get("origin_ids") or [])
            if isinstance(value, str) and value.strip()
        }
    )
    document_count = len(
        {
            str(value).strip()
            for value in (group.get("document_ids") or [])
            if isinstance(value, str) and value.strip()
        }
    )
    return (
        -origin_count,
        -document_count,
        (group.get("canonical_name") or "").casefold(),
        group.get("group_id") or "",
    )


def _singularize_token(token: str) -> str:
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def _queue_identity_key(name: str) -> str:
    """Conservative queue identity: case/space/_/- plus simple plural.

    Only strips a trailing ``s`` (never for ``ss``/``us``/``is``). RAG versus
    Retrieval-Augmented Generation keeps different token sequences, so
    abbreviations and semantic synonyms never collapse here.
    """

    normalized = unicodedata.normalize("NFKC", name or "").casefold()
    normalized = _QUEUE_IDENTITY_SEPARATORS.sub(" ", normalized)
    tokens = _QUEUE_IDENTITY_WORD.findall(normalized)
    return " ".join(_singularize_token(token) for token in tokens)


def _suppress_queue_duplicates(
    merged: list[tuple[dict[str, Any], str, str, str]],
    overflow: list[QueueOverflow],
) -> tuple[list[tuple[dict[str, Any], str, str, str]], list[QueueDuplicateProposal]]:
    """Drop obvious spelling variants before slot filling (queue-level only)."""

    seen: set[tuple[str, str]] = set()
    primary_of: dict[tuple[str, str], str] = {}
    kept: list[tuple[dict[str, Any], str, str, str]] = []
    duplicates: list[QueueDuplicateProposal] = []
    for item in merged:
        group = item[0]
        identity = _queue_identity_key(str(group.get("canonical_name") or ""))
        scoped_identity = (item[3], identity)
        if identity and scoped_identity in seen:
            overflow_key = str(group.get("group_id") or group.get("canonical_name"))
            overflow.append(
                QueueOverflow(
                    kind="candidate_duplicate",
                    key=overflow_key,
                    run_id=item[1],
                    reason="equivalent queue identity",
                )
            )
            duplicates.append(
                QueueDuplicateProposal(
                    dropped_group_id=str(group.get("group_id") or ""),
                    dropped_canonical=str(group.get("canonical_name") or "").strip(),
                    dropped_document_ids=tuple(
                        str(document_id)
                        for document_id in (group.get("document_ids") or [])
                        if isinstance(document_id, str) and document_id.strip()
                    ),
                    primary_group_id=primary_of[scoped_identity],
                    run_id=item[1],
                    domain=item[3],
                    source_query=item[2],
                    overflow_key=overflow_key,
                )
            )
            continue
        if identity:
            seen.add(scoped_identity)
            primary_of[scoped_identity] = str(group.get("group_id") or "")
        kept.append(item)
    return kept, duplicates


def _fill_candidates(
    merged: list[tuple[dict[str, Any], str, str, str]],
    slots: tuple[CandidateSlot, ...],
    overflow: list[QueueOverflow],
    deficits: list[QueueDeficit],
) -> list[QueuedCandidate]:
    # Candidates keep the domain of the run that produced them; a slot is
    # filled only from its own domain, otherwise it stays an honest deficit.
    # Within one domain slots are filled round-robin across non-empty strata
    # (cross_source, media_present, multi_origin, single_origin); an empty
    # stratum simply donates its turn to the next non-empty one.
    by_domain: dict[str, list[tuple[dict[str, Any], str, str, str]]] = {}
    for item in merged:
        by_domain.setdefault(item[3], []).append(item)
    ordered_by_domain: dict[str, list[tuple[dict[str, Any], str, str, str]]] = {}
    for domain, items in by_domain.items():
        strata: dict[str, list[tuple[dict[str, Any], str, str, str]]] = {
            key: [] for key in _STRATA_ORDER
        }
        for item in items:
            strata[_selection_stratum(item[0])].append(item)
        for key in _STRATA_ORDER:
            strata[key].sort(key=_rank_key)
        positions = {key: 0 for key in _STRATA_ORDER}
        ordered: list[tuple[dict[str, Any], str, str, str]] = []
        while True:
            progressed = False
            for key in _STRATA_ORDER:
                bucket = strata[key]
                index = positions[key]
                if index >= len(bucket):
                    continue
                ordered.append(bucket[index])
                positions[key] = index + 1
                progressed = True
            if not progressed:
                break
        ordered_by_domain[domain] = ordered
    queued: list[QueuedCandidate] = []
    missing_counts: dict[str, int] = {}
    for slot in slots:
        bucket = ordered_by_domain.get(slot.domain) or []
        if not bucket:
            missing_counts[slot.domain] = missing_counts.get(slot.domain, 0) + 1
            continue
        group, run_id, raw_query, domain = bucket.pop(0)
        queued.append(
            QueuedCandidate(
                candidate_id=slot.candidate_id,
                canonical_name=(group.get("canonical_name") or "").strip(),
                aliases=_clean_aliases(
                    (group.get("canonical_name") or ""), group.get("aliases")
                ),
                group_id=(group.get("group_id") or "").strip(),
                source_query=raw_query,
                domain=domain,
                analysis_scope_key=_scope_key(domain),
                run_id=run_id,
                selection_stratum=_selection_stratum(group),
            )
        )
    for domain in sorted(ordered_by_domain):
        for group, run_id, _raw_query, _domain in ordered_by_domain[domain]:
            overflow.append(
                QueueOverflow(
                    kind="candidate",
                    key=str(group.get("group_id") or group.get("canonical_name")),
                    run_id=run_id,
                    reason="no free candidate slot",
                )
            )
    deficits.extend(
        sorted(
            (
                QueueDeficit(area=area, need="candidate_review", missing=count)
                for area, count in missing_counts.items()
            ),
            key=lambda item: (item.area, item.need, item.missing),
        )
    )
    return queued


def _noise_pool(
    runs: tuple[DiscoveryRun, ...],
    domains: Mapping[str, str],
    dropped: list[QueueDropped],
    overflow: list[QueueOverflow],
) -> dict[str, list[dict[str, Any]]]:
    """Collect typed noise without converting unresolved Gate reviews."""

    pool: dict[str, list[dict[str, Any]]] = {}

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
                "domain": domains[run.run_id],
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
            for item in (
                (run.result.get("candidate_proposals") or {}).get("proposals") or []
            )
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
            document_ids = (*basis, *(proposal.get("document_ids") or []))
            document = next(
                (documents[item] for item in document_ids if item in documents),
                None,
            )
            if decision.get("decision") == "review":
                overflow.append(
                    QueueOverflow(
                        kind="gate_review",
                        key=str(decision.get("proposal_id") or "unknown"),
                        run_id=run.run_id,
                        reason="requires human resolution before noise typing",
                    )
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
            if not isinstance(issue.get("text"), str) or not issue["text"].strip():
                continue
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
            right_group = None
            right_run = run.run_id
            for candidate_run in runs:
                groups = (candidate_run.result.get("alias_resolution") or {}).get("groups") or []
                for group in groups:
                    if group.get("group_id") == right:
                        right_group = group
                        right_run = candidate_run.run_id
                        break
            if not right_group:
                dropped.append(
                    QueueDropped(reason="unknown_alias_side", detail=str(right))
                )
                continue
            right_documents = documents_of(
                next(item for item in runs if item.run_id == right_run)
            )
            right_document = next(
                (
                    right_documents[item]
                    for item in right_group.get("document_ids") or []
                    if item in right_documents
                ),
                None,
            )
            pool.setdefault(NoiseType.DUPLICATE.value, []).append(
                {
                    "text": str(right_group.get("canonical_name") or "").strip(),
                    "url": (right_document or {}).get("url"),
                    "run_id": right_run,
                    "domain": domains[right_run],
                    "raw_query": raw_query_of(run),
                    "origin_kind": "alias_suggestion",
                    "duplicate_of": None,
                }
            )
    return pool


def _transfer_candidate_duplicates(
    pool: dict[str, list[dict[str, Any]]],
    duplicates: list[QueueDuplicateProposal],
    primary_index: Mapping[str, str],
    runs_by_id: Mapping[str, DiscoveryRun],
    noise_slots: tuple[NoiseSlot, ...],
    dropped: list[QueueDropped],
    overflow: list[QueueOverflow],
) -> None:
    """Offer queue-level spelling duplicates as ``duplicate`` noise rows.

    Each entry stays a manual-review proposal. A duplicate moves into the
    duplicate pool — and its suppression-time ``candidate_duplicate``
    overflow entry is withdrawn as resolved — only when the primary group
    actually received a candidate_id, a source document with a non-blank
    URL exists (the first document_id carrying one wins), and a
    ``duplicate`` noise slot covers the duplicate's domain. Otherwise the
    reason is recorded in dropped while the overflow entry keeps the
    variant itself from being lost silently; a pool leftover without a
    free slot is reported by the standard noise overflow.
    """

    duplicate_domains = {
        slot.domain
        for slot in noise_slots
        if slot.planned_noise_type == NoiseType.DUPLICATE.value
    }
    resolved: set[tuple[str, str]] = set()
    for proposal in duplicates:
        primary_candidate_id = primary_index.get(proposal.primary_group_id)
        if primary_candidate_id is None:
            dropped.append(
                QueueDropped(
                    reason="primary_not_queued",
                    detail=proposal.dropped_group_id or proposal.dropped_canonical,
                )
            )
            continue
        text = proposal.dropped_canonical.strip()
        if not text:
            dropped.append(
                QueueDropped(reason="blank_text", detail="candidate_duplicate")
            )
            continue
        run = runs_by_id.get(proposal.run_id)
        documents = _documents_by_id(run.result) if run is not None else {}
        url: str | None = None
        for document_id in proposal.dropped_document_ids:
            document = documents.get(document_id)
            if document is None:
                continue
            candidate_url = document.get("url")
            if isinstance(candidate_url, str) and candidate_url.strip():
                url = candidate_url.strip()
                break
        if url is None:
            dropped.append(
                QueueDropped(
                    reason="no_url",
                    detail=f"candidate_duplicate:{text[:60]}",
                )
            )
            continue
        if proposal.domain not in duplicate_domains:
            continue
        pool.setdefault(NoiseType.DUPLICATE.value, []).append(
            {
                "text": text,
                "url": url,
                "run_id": proposal.run_id,
                "domain": proposal.domain,
                "raw_query": proposal.source_query,
                "origin_kind": "candidate_duplicate",
                "duplicate_of": primary_candidate_id,
            }
        )
        resolved.add((proposal.overflow_key, proposal.run_id))
    if resolved:
        overflow[:] = [
            item
            for item in overflow
            if not (
                item.kind == "candidate_duplicate"
                and (item.key, item.run_id) in resolved
            )
        ]


def _fill_noise(
    pool: dict[str, list[dict[str, Any]]],
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
    by_type: dict[str, list[NoiseSlot]] = {}
    for slot in slots:
        by_type.setdefault(slot.planned_noise_type, []).append(slot)

    def take_matching(
        items: list[dict[str, Any]], domain: str
    ) -> dict[str, Any] | None:
        for index, item in enumerate(items):
            if item.get("domain") == domain:
                return items.pop(index)
        return None

    for noise_type, type_slots in by_type.items():
        items = pool.get(noise_type, [])
        for slot in type_slots:
            item = take_matching(items, slot.domain)
            if item is None:
                continue
            document_url = item["url"]
            if not document_url:
                dropped.append(
                    QueueDropped(
                        reason="no_url",
                        detail=f"{item['origin_kind']}:{item['text'][:60]}",
                    )
                )
                continue
            queued.append(
                QueuedNoise(
                    noise_id=slot.noise_id,
                    planned_noise_type=noise_type,
                    source_query=item["raw_query"],
                    extracted_text=item["text"],
                    source_document_url=document_url,
                    domain=item["domain"],
                    analysis_scope_key=_scope_key(item["domain"]),
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


def _usage_requests_and_stop(value: Any) -> tuple[int | None, str | None, bool]:
    """Return (requests_used, stop_reason, has_usage) for one usage payload."""

    if isinstance(value, Mapping):
        requests = value.get("requests_used")
        stop = value.get("stop_reason")
        requests_used = (
            requests if isinstance(requests, int) and not isinstance(requests, bool) else 0
        )
        stop_reason = str(stop).strip() if isinstance(stop, str) and str(stop).strip() else None
        return requests_used, stop_reason, True
    if isinstance(value, list):
        total = 0
        stops: list[str] = []
        seen_any = False
        for entry in value:
            if not isinstance(entry, Mapping):
                continue
            seen_any = True
            requests = entry.get("requests_used")
            if isinstance(requests, int) and not isinstance(requests, bool):
                total += requests
            stop = entry.get("stop_reason")
            if isinstance(stop, str) and stop.strip() and stop.strip() not in stops:
                stops.append(stop.strip())
        if not seen_any:
            return None, None, False
        stop_reason = ",".join(stops) if stops else None
        return total, stop_reason, True
    return None, None, False


def _search_coverage(runs: tuple[DiscoveryRun, ...]) -> tuple[RunSearchCoverage, ...]:
    """Record searched source classes with honest machine notes.

    ``complete`` with executed requests counts as covered; ``partial`` with
    ``requests_used > 0`` also counts but keeps ``status(stop_reason)`` in the
    note. ``failed``, missing runs and zero-request classes never count.
    Unknown media providers are never counted as Media Cloud.
    """

    coverage: list[RunSearchCoverage] = []
    for run in runs:
        raw_query = ((run.plan.get("scope") or {}).get("raw_query") or "").strip()
        source_classes: list[str] = []
        notes: list[str] = []
        scientific = run.result.get("scientific") or {}
        scientific_status = scientific.get("status")
        scientific_requests, scientific_stop, scientific_has_usage = (
            _usage_requests_and_stop(scientific.get("usage"))
        )
        scientific_verified = False
        if scientific_status == "complete":
            if scientific_has_usage and (scientific_requests or 0) > 0:
                scientific_verified = True
        elif scientific_status == "partial":
            if (scientific_requests or 0) > 0:
                scientific_verified = True
        if scientific_verified:
            source_classes.append("scientific")
            if scientific_stop:
                notes.append(f"scientific={scientific_status}({scientific_stop})")
            else:
                notes.append(f"scientific={scientific_status}")
        media = run.result.get("media") or {}
        media_status = media.get("status")
        provider = media.get("provider_used")
        provider_ok = provider in _ALLOWED_MEDIA_PROVIDERS
        media_requests, media_stop, media_has_usage = _usage_requests_and_stop(
            media.get("usage")
        )
        media_verified = False
        if provider_ok and media_status in ("complete", "partial"):
            if media_has_usage and (media_requests or 0) > 0:
                media_verified = True
        if media_verified:
            source_classes.append("industry")
            if media_status == "partial" and media_stop:
                notes.append(f"industry={media_status}({provider},{media_stop})")
            elif provider:
                notes.append(f"industry={media_status}({provider})")
            else:
                notes.append(f"industry={media_status}")
        coverage.append(
            RunSearchCoverage(
                run_id=run.run_id,
                raw_query=raw_query,
                source_classes=tuple(source_classes),
                notes=";".join(notes),
            )
        )
    return tuple(coverage)


def build_labeling_queue(
    runs: tuple[DiscoveryRun, ...],
    *,
    candidate_slots: tuple[CandidateSlot, ...] = (),
    noise_slots: tuple[NoiseSlot, ...] = (),
    run_domains: Mapping[str, str] | None = None,
) -> LabelingQueue:
    """Build the deterministic review queue from persisted runs and template slots.

    Every row keeps the domain of the run that produced it; slots from other
    domains stay honest deficits instead of being filled with foreign data.
    """

    ordered_runs = tuple(sorted(runs, key=lambda run: run.run_id))
    domains = resolve_run_domains(ordered_runs, run_domains)
    overflow: list[QueueOverflow] = []
    deficits: list[QueueDeficit] = []
    dropped: list[QueueDropped] = []
    merged = _merged_groups(ordered_runs, domains)
    merged, duplicate_proposals = _suppress_queue_duplicates(merged, overflow)
    queued_candidates = _fill_candidates(merged, candidate_slots, overflow, deficits)
    pool = _noise_pool(ordered_runs, domains, dropped, overflow)
    _transfer_candidate_duplicates(
        pool,
        duplicate_proposals,
        {item.group_id: item.candidate_id for item in queued_candidates},
        {run.run_id: run for run in ordered_runs},
        noise_slots,
        dropped,
        overflow,
    )
    queued_noise = _fill_noise(pool, noise_slots, dropped, overflow, deficits)
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
        search_coverage=_search_coverage(ordered_runs),
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
