"""Cross-source discovery pipeline through the auditable candidate gate."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nextwave.contracts import SourceDocument

from .alias_resolution import AliasResolutionResult, resolve_candidate_aliases
from .candidate_gate import (
    DEFAULT_GATE_MAX_PROPOSALS,
    CandidateGateResult,
    StructuredCandidateGate,
    build_candidate_gate_from_environment,
)
from .candidates import (
    CandidateMention,
    CandidateProposal,
    CandidateProposalBatch,
    build_candidate_proposals,
    build_openalex_candidate_mentions,
)
from .contracts import DiscoveryPlan
from .evidence_extractor import (
    EvidenceExtractionResult,
    StructuredEvidenceExtractor,
    build_evidence_extractor_from_environment,
)
from .executor import OpenAlexDiscoveryExecutor, OpenAlexDiscoveryResult
from .media_executor import MediaDiscoveryExecutor, MediaDiscoveryResult
from .mention_extractor import (
    CandidateMentionExtractionResult,
    StructuredCandidateMentionExtractor,
    build_candidate_text_extractor_from_environment,
)
from .origins import OriginResolutionResult, resolve_candidate_origins
from .verification import (
    CandidateVerificationExecutor,
    CandidateVerificationResult,
)

DISCOVERY_PIPELINE_VERSION = "discovery-pipeline-v8"


def split_gate_batch(
    batch: CandidateProposalBatch, max_gate_proposals: int
) -> tuple[CandidateProposalBatch, int]:
    """Select a deterministic mix of proposal visibility strata for the gate.

    Cross-source, two-to-three-origin, single-origin, and four-plus-origin
    proposals take turns. Empty strata donate their places to the remaining
    ones, so the cap is always filled without letting prevalence alone decide
    which candidates are judged.
    """

    if max_gate_proposals < 1:
        raise ValueError("max_gate_proposals must be positive")
    if len(batch.proposals) <= max_gate_proposals:
        selected = batch.proposals
    else:
        strata: tuple[list[CandidateProposal], ...] = ([], [], [], [])
        for proposal in batch.proposals:
            strata[_gate_stratum(proposal)].append(proposal)
        positions = [0] * len(strata)
        selected_items: list[CandidateProposal] = []
        while len(selected_items) < max_gate_proposals:
            for index, stratum in enumerate(strata):
                if positions[index] >= len(stratum):
                    continue
                selected_items.append(stratum[positions[index]])
                positions[index] += 1
                if len(selected_items) == max_gate_proposals:
                    break
        selected = tuple(selected_items)
    return (
        CandidateProposalBatch(
            analysis_scope_id=batch.analysis_scope_id,
            proposals=selected,
            exclusions=batch.exclusions,
        ),
        len(batch.proposals) - len(selected),
    )


def _gate_stratum(proposal: CandidateProposal) -> int:
    if len(proposal.connector_ids) > 1:
        return 0
    if 2 <= proposal.origin_count <= 3:
        return 1
    if proposal.origin_count <= 1:
        return 2
    return 3


@dataclass(frozen=True, slots=True)
class DiscoveryPipelineResult:
    plan_id: str
    scientific: OpenAlexDiscoveryResult
    media: MediaDiscoveryResult
    documents: tuple[SourceDocument, ...]
    mentions: tuple[CandidateMention, ...]
    text_extraction: CandidateMentionExtractionResult | None
    candidate_proposals: CandidateProposalBatch
    candidate_gate: CandidateGateResult
    alias_resolution: AliasResolutionResult
    verification: CandidateVerificationResult
    origin_resolution: OriginResolutionResult
    evidence_extraction: EvidenceExtractionResult
    pipeline_version: str = DISCOVERY_PIPELINE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "pipeline_version": self.pipeline_version,
            "scientific": self.scientific.to_dict(),
            "media": self.media.to_dict(),
            "documents": [document.document_id for document in self.documents],
            "mentions": [mention.to_dict() for mention in self.mentions],
            "text_extraction": (self.text_extraction.to_dict() if self.text_extraction else None),
            "candidate_proposals": self.candidate_proposals.to_dict(),
            "candidate_gate": self.candidate_gate.to_dict(),
            "alias_resolution": self.alias_resolution.to_dict(),
            "verification": self.verification.to_dict(),
            "origin_resolution": self.origin_resolution.to_dict(),
            "evidence_extraction": self.evidence_extraction.to_dict(),
        }


class DiscoveryPipeline:
    """Run independent sources concurrently, then build one proposal pool."""

    def __init__(
        self,
        scientific_executor: OpenAlexDiscoveryExecutor,
        media_executor: MediaDiscoveryExecutor,
        text_extractor: StructuredCandidateMentionExtractor,
        candidate_gate: StructuredCandidateGate,
        verification_executor: CandidateVerificationExecutor,
        evidence_extractor: StructuredEvidenceExtractor,
        *,
        max_gate_proposals: int = DEFAULT_GATE_MAX_PROPOSALS,
    ) -> None:
        if max_gate_proposals < 1:
            raise ValueError("max_gate_proposals must be positive")
        self._scientific_executor = scientific_executor
        self._media_executor = media_executor
        self._text_extractor = text_extractor
        self._candidate_gate = candidate_gate
        self._max_gate_proposals = max_gate_proposals
        self._verification_executor = verification_executor
        self._evidence_extractor = evidence_extractor

    def execute(
        self,
        plan: DiscoveryPlan,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> DiscoveryPipelineResult:
        """Execute the pipeline, optionally reporting timed stage lines.

        The callback receives one ``[discovery] <stage>: <detail> (<s>s)`` line
        per stage, so a live run shows where minutes go instead of silence.
        Timing never affects results and defaults to off (tests stay silent).
        """

        def report(stage: str, detail: str, started_at: float) -> None:
            if progress is not None:
                elapsed = time.monotonic() - started_at
                progress(f"[discovery] {stage}: {detail} ({elapsed:.1f}s)")

        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as pool:
            scientific_future = pool.submit(self._scientific_executor.execute, plan)
            media_future = pool.submit(self._media_executor.execute, plan)
            scientific = scientific_future.result()
            media = media_future.result()

        documents = _merge_documents(scientific.documents, media.documents)
        report(
            "sources",
            f"{len(documents)} documents "
            f"({len(scientific.documents)} scientific, {len(media.documents)} media)",
            started,
        )
        started = time.monotonic()
        text_extraction = (
            self._text_extractor.extract_many(plan.scope, documents) if documents else None
        )
        grounded_mentions = text_extraction.mentions if text_extraction is not None else ()
        mentions_by_id = {
            mention.mention_id: mention for mention in grounded_mentions
        }
        mentions_by_id.update(
            (
                mention.mention_id,
                mention,
            )
            for mention in build_openalex_candidate_mentions(
                scientific.documents,
                scientific.hints,
                grounded_mentions=grounded_mentions,
            )
        )
        mentions = tuple(sorted(mentions_by_id.values(), key=lambda mention: mention.mention_id))
        report("extraction", f"{len(mentions)} mentions", started)
        started = time.monotonic()
        candidate_proposals = build_candidate_proposals(
            plan.scope,
            documents,
            mentions,
        )
        report(
            "proposals",
            f"{len(candidate_proposals.proposals)} proposals, "
            f"{len(candidate_proposals.exclusions)} excluded",
            started,
        )
        started = time.monotonic()
        gated_proposals, gate_skipped = split_gate_batch(
            candidate_proposals, self._max_gate_proposals
        )
        candidate_gate = self._candidate_gate.evaluate(
            plan.scope,
            gated_proposals,
            documents,
        )
        report(
            "gate",
            f"{len(candidate_gate.accepted_proposal_ids)} accepted "
            f"({gate_skipped} skipped by cap {self._max_gate_proposals})",
            started,
        )
        started = time.monotonic()
        alias_resolution = resolve_candidate_aliases(gated_proposals, candidate_gate)
        report("aliases", f"{len(alias_resolution.groups)} groups", started)
        started = time.monotonic()
        verification = self._verification_executor.execute(plan, alias_resolution)
        report("verification", f"{verification.requests_used} requests", started)
        started = time.monotonic()
        origin_resolution = resolve_candidate_origins(
            plan, alias_resolution, verification, documents
        )
        report("origins", f"{len(origin_resolution.candidates)} candidates", started)
        started = time.monotonic()
        evidence_extraction = self._evidence_extractor.extract(
            alias_resolution, origin_resolution, media.enrichment
        )
        report("evidence", f"{len(evidence_extraction.proposals)} proposals", started)
        return DiscoveryPipelineResult(
            plan_id=plan.plan_id,
            scientific=scientific,
            media=media,
            documents=documents,
            mentions=mentions,
            text_extraction=text_extraction,
            candidate_proposals=candidate_proposals,
            candidate_gate=candidate_gate,
            alias_resolution=alias_resolution,
            verification=verification,
            origin_resolution=origin_resolution,
            evidence_extraction=evidence_extraction,
        )


def build_discovery_pipeline_from_environment(
    environment: Mapping[str, str],
    *,
    snapshot_root: Path = Path("runtime") / "snapshots",
) -> DiscoveryPipeline:
    """Build the live pipeline from backend-only runtime configuration."""

    from .media_executor import parse_mediacloud_collection_ids

    return DiscoveryPipeline(
        OpenAlexDiscoveryExecutor(
            snapshot_root,
            contact_email=environment.get("NEXTWAVE_OPENALEX_MAILTO") or None,
        ),
        MediaDiscoveryExecutor(
            snapshot_root,
            mediacloud_api_key=environment.get("NEXTWAVE_MEDIACLOUD_API_KEY") or None,
            mediacloud_collection_ids=parse_mediacloud_collection_ids(environment),
        ),
        build_candidate_text_extractor_from_environment(environment),
        build_candidate_gate_from_environment(environment),
        CandidateVerificationExecutor(
            snapshot_root,
            contact_email=environment.get("NEXTWAVE_OPENALEX_MAILTO") or None,
        ),
        build_evidence_extractor_from_environment(environment),
    )


def _merge_documents(
    scientific: tuple[SourceDocument, ...],
    media: tuple[SourceDocument, ...],
) -> tuple[SourceDocument, ...]:
    documents = (*scientific, *media)
    document_ids = [document.document_id for document in documents]
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("source executors produced duplicate document_id values")
    return documents
