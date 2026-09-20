"""Auditable high-recall candidate proposals built from provider discovery hints."""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import asdict, dataclass
from enum import Enum, StrEnum
from typing import Any

from nextwave.contracts import SourceDocument
from nextwave.sources import OpenAlexDiscoveryHints

from .contracts import AnalysisScope

CANDIDATE_PROPOSAL_VERSION = "candidate-proposal-v1"


class CandidateHintKind(StrEnum):
    KEYWORD = "keyword"
    TOPIC = "topic"


class ProposalExclusionReason(StrEnum):
    SCOPE_TERM = "scope_term"
    ORGANIZATION = "organization"
    INVALID_NAME = "invalid_name"


@dataclass(frozen=True, slots=True)
class CandidateProposal:
    proposal_id: str
    analysis_scope_id: str
    canonical_name: str
    normalized_name: str
    aliases: tuple[str, ...]
    source_kinds: tuple[CandidateHintKind, ...]
    provider_term_ids: tuple[str, ...]
    document_ids: tuple[str, ...]
    origin_ids: tuple[str, ...]
    max_score: float
    primary_topic: bool
    proposal_version: str = CANDIDATE_PROPOSAL_VERSION

    @property
    def document_count(self) -> int:
        return len(self.document_ids)

    @property
    def origin_count(self) -> int:
        return len(self.origin_ids)

    def to_dict(self) -> dict[str, Any]:
        payload = _json_value(asdict(self))
        payload["document_count"] = self.document_count
        payload["origin_count"] = self.origin_count
        return payload


@dataclass(frozen=True, slots=True)
class CandidateProposalExclusion:
    display_name: str
    normalized_name: str
    provider_term_id: str
    document_id: str
    reason: ProposalExclusionReason

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True, slots=True)
class CandidateProposalBatch:
    analysis_scope_id: str
    proposals: tuple[CandidateProposal, ...]
    exclusions: tuple[CandidateProposalExclusion, ...]
    proposal_version: str = CANDIDATE_PROPOSAL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_scope_id": self.analysis_scope_id,
            "proposal_version": self.proposal_version,
            "proposals": [proposal.to_dict() for proposal in self.proposals],
            "exclusions": [exclusion.to_dict() for exclusion in self.exclusions],
        }


@dataclass(slots=True)
class _ProposalAccumulator:
    canonical_name: str
    canonical_score: float
    names: set[str]
    source_kinds: set[CandidateHintKind]
    provider_term_ids: set[str]
    document_ids: set[str]
    origin_ids: set[str]
    max_score: float
    primary_topic: bool


def build_candidate_proposals(
    scope: AnalysisScope,
    documents: tuple[SourceDocument, ...],
    hints: tuple[OpenAlexDiscoveryHints, ...],
) -> CandidateProposalBatch:
    """Aggregate OpenAlex keyword/topic labels without claiming they are technologies yet."""

    documents_by_id = {document.document_id: document for document in documents}
    if len(documents_by_id) != len(documents):
        raise ValueError("documents must contain unique document_id values")
    if len({hint.document_id for hint in hints}) != len(hints):
        raise ValueError("hints must contain unique document_id values")
    if any(hint.document_id not in documents_by_id for hint in hints):
        raise ValueError("every discovery hint must reference a supplied document")

    scope_terms = {_normalize_name(scope.normalized_query)}
    scope_terms.update(_normalize_name(value) for value in scope.search_texts)
    organizations = {
        _normalize_name(name)
        for document in documents
        for name in document.organizations
    }
    accumulators: dict[str, _ProposalAccumulator] = {}
    exclusions: list[CandidateProposalExclusion] = []

    for hint in hints:
        document = documents_by_id[hint.document_id]
        for keyword in hint.keywords:
            _add_term(
                accumulators,
                exclusions,
                scope_terms,
                organizations,
                display_name=keyword.display_name,
                provider_term_id=f"openalex:keyword:{keyword.keyword_id}",
                kind=CandidateHintKind.KEYWORD,
                score=keyword.score,
                primary=False,
                document=document,
            )
        for topic in hint.topics:
            _add_term(
                accumulators,
                exclusions,
                scope_terms,
                organizations,
                display_name=topic.display_name,
                provider_term_id=f"openalex:topic:{topic.topic_id}",
                kind=CandidateHintKind.TOPIC,
                score=topic.score,
                primary=topic.primary,
                document=document,
            )

    proposals = tuple(
        sorted(
            (_build_proposal(scope.scope_id, name, value) for name, value in accumulators.items()),
            key=lambda proposal: (
                -proposal.origin_count,
                -proposal.document_count,
                -proposal.max_score,
                proposal.normalized_name,
            ),
        )
    )
    return CandidateProposalBatch(
        analysis_scope_id=scope.scope_id,
        proposals=proposals,
        exclusions=tuple(exclusions),
    )


def _add_term(
    accumulators: dict[str, _ProposalAccumulator],
    exclusions: list[CandidateProposalExclusion],
    scope_terms: set[str],
    organizations: set[str],
    *,
    display_name: str,
    provider_term_id: str,
    kind: CandidateHintKind,
    score: float,
    primary: bool,
    document: SourceDocument,
) -> None:
    normalized = _normalize_name(display_name)
    reason: ProposalExclusionReason | None = None
    if len(normalized) < 2 or not any(character.isalnum() for character in normalized):
        reason = ProposalExclusionReason.INVALID_NAME
    elif normalized in scope_terms:
        reason = ProposalExclusionReason.SCOPE_TERM
    elif normalized in organizations:
        reason = ProposalExclusionReason.ORGANIZATION
    if reason is not None:
        exclusions.append(
            CandidateProposalExclusion(
                display_name=display_name,
                normalized_name=normalized,
                provider_term_id=provider_term_id,
                document_id=document.document_id,
                reason=reason,
            )
        )
        return

    accumulator = accumulators.get(normalized)
    if accumulator is None:
        accumulators[normalized] = _ProposalAccumulator(
            canonical_name=display_name,
            canonical_score=score,
            names={display_name},
            source_kinds={kind},
            provider_term_ids={provider_term_id},
            document_ids={document.document_id},
            origin_ids={document.origin_id},
            max_score=score,
            primary_topic=primary,
        )
        return
    accumulator.names.add(display_name)
    accumulator.source_kinds.add(kind)
    accumulator.provider_term_ids.add(provider_term_id)
    accumulator.document_ids.add(document.document_id)
    accumulator.origin_ids.add(document.origin_id)
    accumulator.max_score = max(accumulator.max_score, score)
    accumulator.primary_topic = accumulator.primary_topic or primary
    if score > accumulator.canonical_score:
        accumulator.canonical_name = display_name
        accumulator.canonical_score = score


def _build_proposal(
    scope_id: str,
    normalized_name: str,
    value: _ProposalAccumulator,
) -> CandidateProposal:
    digest = hashlib.sha256(f"{scope_id}|{normalized_name}".encode()).hexdigest()[:16]
    aliases = tuple(
        sorted(
            (name for name in value.names if name != value.canonical_name),
            key=str.casefold,
        )
    )
    return CandidateProposal(
        proposal_id=f"proposal-{digest}",
        analysis_scope_id=scope_id,
        canonical_name=value.canonical_name,
        normalized_name=normalized_name,
        aliases=aliases,
        source_kinds=tuple(sorted(value.source_kinds, key=lambda item: item.value)),
        provider_term_ids=tuple(sorted(value.provider_term_ids)),
        document_ids=tuple(sorted(value.document_ids)),
        origin_ids=tuple(sorted(value.origin_ids)),
        max_score=value.max_score,
        primary_topic=value.primary_topic,
    )


def _normalize_name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value
