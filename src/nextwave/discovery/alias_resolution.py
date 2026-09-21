"""Prepare accepted proposal identities and uncertain alias review suggestions."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from .candidate_gate import CandidateGateResult, GateDecision
from .candidates import CandidateProposalBatch, orthographic_candidate_key

ALIAS_RESOLUTION_VERSION = "alias-resolution-v1"
_ASCII_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_ACRONYM = re.compile(r"[A-Z0-9]{3,8}\Z")


class AliasSuggestionReason(StrEnum):
    SHARED_ORIGIN_ACRONYM = "shared_origin_acronym"


@dataclass(frozen=True, slots=True)
class ResolvedAliasGroup:
    group_id: str
    analysis_scope_id: str
    canonical_name: str
    aliases: tuple[str, ...]
    proposal_ids: tuple[str, ...]
    document_ids: tuple[str, ...]
    origin_ids: tuple[str, ...]
    connector_ids: tuple[str, ...]
    normalization_key: str
    resolution_version: str = ALIAS_RESOLUTION_VERSION

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for field_name in (
            "aliases", "proposal_ids", "document_ids", "origin_ids", "connector_ids"
        ):
            value[field_name] = list(value[field_name])
        return value


@dataclass(frozen=True, slots=True)
class AliasSuggestion:
    left_group_id: str
    right_group_id: str
    reason: AliasSuggestionReason
    shared_origin_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["reason"] = self.reason.value
        value["shared_origin_ids"] = list(self.shared_origin_ids)
        return value


@dataclass(frozen=True, slots=True)
class AliasResolutionResult:
    analysis_scope_id: str
    input_proposal_ids: tuple[str, ...]
    groups: tuple[ResolvedAliasGroup, ...]
    review_suggestions: tuple[AliasSuggestion, ...]
    resolution_version: str = ALIAS_RESOLUTION_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_scope_id": self.analysis_scope_id,
            "input_proposal_ids": list(self.input_proposal_ids),
            "groups": [group.to_dict() for group in self.groups],
            "review_suggestions": [item.to_dict() for item in self.review_suggestions],
            "resolution_version": self.resolution_version,
        }


def resolve_candidate_aliases(
    proposals: CandidateProposalBatch,
    gate: CandidateGateResult,
) -> AliasResolutionResult:
    """Preserve pre-gate groups and queue shared-origin acronyms for review."""

    if proposals.analysis_scope_id != gate.analysis_scope_id:
        raise ValueError("candidate gate and proposals must use the same analysis scope")
    proposal_ids = tuple(item.proposal_id for item in proposals.proposals)
    if proposal_ids != gate.input_proposal_ids:
        raise ValueError("candidate gate must cover these proposals in the same order")
    if tuple(item.proposal_id for item in gate.decisions) != proposal_ids:
        raise ValueError("candidate gate must return one ordered decision per proposal")

    accepted = tuple(
        proposal
        for proposal, decision in zip(proposals.proposals, gate.decisions, strict=True)
        if decision.decision is GateDecision.ACCEPT
    )
    keys = tuple(orthographic_candidate_key(item.canonical_name) for item in accepted)
    if len(set(keys)) != len(keys):
        raise ValueError("orthographic variants must be grouped before candidate gate")
    groups: list[ResolvedAliasGroup] = []
    for proposal, key in zip(accepted, keys, strict=True):
        group_digest = hashlib.sha256(
            f"{proposals.analysis_scope_id}|{ALIAS_RESOLUTION_VERSION}|{key}".encode()
        ).hexdigest()[:16]
        groups.append(
            ResolvedAliasGroup(
                group_id=f"alias-group-{group_digest}",
                analysis_scope_id=proposals.analysis_scope_id,
                canonical_name=proposal.canonical_name,
                aliases=proposal.aliases,
                proposal_ids=(proposal.proposal_id,),
                document_ids=proposal.document_ids,
                origin_ids=proposal.origin_ids,
                connector_ids=proposal.connector_ids,
                normalization_key=key,
            )
        )

    suggestions: list[AliasSuggestion] = []
    for index, left in enumerate(groups):
        for right in groups[index + 1 :]:
            shared_origins = tuple(sorted(set(left.origin_ids) & set(right.origin_ids)))
            if shared_origins and _acronym_match(left, right):
                suggestions.append(AliasSuggestion(
                    left_group_id=left.group_id,
                    right_group_id=right.group_id,
                    reason=AliasSuggestionReason.SHARED_ORIGIN_ACRONYM,
                    shared_origin_ids=shared_origins,
                ))
    return AliasResolutionResult(
        analysis_scope_id=proposals.analysis_scope_id,
        input_proposal_ids=tuple(item.proposal_id for item in accepted),
        groups=tuple(groups),
        review_suggestions=tuple(suggestions),
    )


def _acronym_match(left: ResolvedAliasGroup, right: ResolvedAliasGroup) -> bool:
    return any(
        _is_acronym_of(short, long)
        for short in (left.canonical_name, *left.aliases)
        for long in (right.canonical_name, *right.aliases)
    ) or any(
        _is_acronym_of(short, long)
        for short in (right.canonical_name, *right.aliases)
        for long in (left.canonical_name, *left.aliases)
    )


def _is_acronym_of(short: str, long: str) -> bool:
    if _ACRONYM.fullmatch(short) is None:
        return False
    words = _ASCII_WORD.findall(long)
    if not 2 <= len(words) <= 8:
        return False
    return short == "".join(word[0].upper() for word in words)
