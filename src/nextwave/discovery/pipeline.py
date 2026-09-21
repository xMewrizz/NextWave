"""Cross-source discovery pipeline through the auditable candidate gate."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nextwave.contracts import SourceDocument

from .candidate_gate import (
    CandidateGateResult,
    StructuredCandidateGate,
    build_candidate_gate_from_environment,
)
from .candidates import (
    CandidateMention,
    CandidateProposalBatch,
    build_candidate_proposals,
    build_openalex_candidate_mentions,
)
from .contracts import DiscoveryPlan
from .executor import OpenAlexDiscoveryExecutor, OpenAlexDiscoveryResult
from .media_executor import MediaDiscoveryExecutor, MediaDiscoveryResult
from .mention_extractor import (
    CandidateMentionExtractionResult,
    StructuredCandidateMentionExtractor,
    build_candidate_text_extractor_from_environment,
)

DISCOVERY_PIPELINE_VERSION = "discovery-pipeline-v2"


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
    pipeline_version: str = DISCOVERY_PIPELINE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "pipeline_version": self.pipeline_version,
            "scientific": self.scientific.to_dict(),
            "media": self.media.to_dict(),
            "documents": [document.document_id for document in self.documents],
            "mentions": [mention.to_dict() for mention in self.mentions],
            "text_extraction": (
                self.text_extraction.to_dict() if self.text_extraction else None
            ),
            "candidate_proposals": self.candidate_proposals.to_dict(),
            "candidate_gate": self.candidate_gate.to_dict(),
        }


class DiscoveryPipeline:
    """Run independent sources concurrently, then build one proposal pool."""

    def __init__(
        self,
        scientific_executor: OpenAlexDiscoveryExecutor,
        media_executor: MediaDiscoveryExecutor,
        text_extractor: StructuredCandidateMentionExtractor,
        candidate_gate: StructuredCandidateGate,
    ) -> None:
        self._scientific_executor = scientific_executor
        self._media_executor = media_executor
        self._text_extractor = text_extractor
        self._candidate_gate = candidate_gate

    def execute(self, plan: DiscoveryPlan) -> DiscoveryPipelineResult:
        with ThreadPoolExecutor(max_workers=2) as pool:
            scientific_future = pool.submit(self._scientific_executor.execute, plan)
            media_future = pool.submit(self._media_executor.execute, plan)
            scientific = scientific_future.result()
            media = media_future.result()

        documents = _merge_documents(scientific.documents, media.documents)
        text_extraction = (
            self._text_extractor.extract_many(plan.scope, documents)
            if documents
            else None
        )
        mentions_by_id = {
            mention.mention_id: mention
            for mention in build_openalex_candidate_mentions(
                scientific.documents,
                scientific.hints,
            )
        }
        if text_extraction is not None:
            mentions_by_id.update(
                (mention.mention_id, mention) for mention in text_extraction.mentions
            )
        mentions = tuple(
            sorted(mentions_by_id.values(), key=lambda mention: mention.mention_id)
        )
        candidate_proposals = build_candidate_proposals(
            plan.scope,
            documents,
            mentions,
        )
        candidate_gate = self._candidate_gate.evaluate(
            plan.scope,
            candidate_proposals,
            documents,
        )
        return DiscoveryPipelineResult(
            plan_id=plan.plan_id,
            scientific=scientific,
            media=media,
            documents=documents,
            mentions=mentions,
            text_extraction=text_extraction,
            candidate_proposals=candidate_proposals,
            candidate_gate=candidate_gate,
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
            api_key=environment.get("NEXTWAVE_OPENALEX_API_KEY") or None,
        ),
        MediaDiscoveryExecutor(
            snapshot_root,
            mediacloud_api_key=environment.get("NEXTWAVE_MEDIACLOUD_API_KEY") or None,
            mediacloud_collection_ids=parse_mediacloud_collection_ids(environment),
        ),
        build_candidate_text_extractor_from_environment(environment),
        build_candidate_gate_from_environment(environment),
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
