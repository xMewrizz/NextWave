"""Conservative origin grouping for discovered and verification documents."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Any

from nextwave.contracts import SourceDocument, SourceType

from .alias_resolution import AliasResolutionResult
from .contracts import DiscoveryPlan
from .verification import CandidateVerificationResult

ORIGIN_RESOLUTION_VERSION = "origin-resolution-v1"
_NEWS_TYPES = {
    SourceType.INDUSTRY_MEDIA,
    SourceType.PRESS_RELEASE,
    SourceType.SOCIAL_OR_BLOG,
}


class EchoSuggestionReason(StrEnum):
    SAME_NEWS_TITLE_NEAR_DATE = "same_news_title_near_date"


@dataclass(frozen=True, slots=True)
class OriginGroup:
    origin_id: str
    documents: tuple[SourceDocument, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin_id": self.origin_id,
            "document_ids": [item.document_id for item in self.documents],
            "connector_ids": sorted({item.connector_id for item in self.documents}),
            "urls": sorted({item.canonical_url for item in self.documents}),
            "dois": sorted({item.doi for item in self.documents if item.doi}),
            "origin_methods": sorted({
                item.origin_method for item in self.documents if item.origin_method
            }),
        }


@dataclass(frozen=True, slots=True)
class EchoSuggestion:
    left_origin_id: str
    right_origin_id: str
    left_document_id: str
    right_document_id: str
    reason: EchoSuggestionReason
    date_gap_days: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_origin_id": self.left_origin_id,
            "right_origin_id": self.right_origin_id,
            "left_document_id": self.left_document_id,
            "right_document_id": self.right_document_id,
            "reason": self.reason.value,
            "date_gap_days": self.date_gap_days,
        }


@dataclass(frozen=True, slots=True)
class CandidateOrigins:
    alias_group_id: str
    origin_groups: tuple[OriginGroup, ...]
    echo_suggestions: tuple[EchoSuggestion, ...]

    @property
    def document_count(self) -> int:
        return sum(len(group.documents) for group in self.origin_groups)

    @property
    def exact_origin_count(self) -> int:
        return len(self.origin_groups)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alias_group_id": self.alias_group_id,
            "document_count": self.document_count,
            "exact_origin_count": self.exact_origin_count,
            "origin_groups": [group.to_dict() for group in self.origin_groups],
            "echo_suggestions": [item.to_dict() for item in self.echo_suggestions],
        }


@dataclass(frozen=True, slots=True)
class OriginResolutionResult:
    plan_id: str
    candidates: tuple[CandidateOrigins, ...]
    version: str = ORIGIN_RESOLUTION_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "candidates": [item.to_dict() for item in self.candidates],
            "version": self.version,
        }


def resolve_candidate_origins(
    plan: DiscoveryPlan,
    aliases: AliasResolutionResult,
    verification: CandidateVerificationResult,
    discovery_documents: tuple[SourceDocument, ...],
) -> OriginResolutionResult:
    """Group exact source identities; flag plausible news echoes without merging them."""
    if aliases.analysis_scope_id != plan.scope.scope_id:
        raise ValueError("alias groups must match the discovery scope")
    if verification.plan_id != plan.plan_id:
        raise ValueError("verification must match the discovery plan")
    verification_by_id = {item.group_id: item for item in verification.results}
    alias_ids = {item.group_id for item in aliases.groups}
    if len(verification_by_id) != len(verification.results) or set(verification_by_id) != alias_ids:
        raise ValueError("verification must cover each accepted alias group exactly once")
    documents_by_id = {item.document_id: item for item in discovery_documents}
    if len(documents_by_id) != len(discovery_documents):
        raise ValueError("discovery documents must have unique identifiers")

    candidates: list[CandidateOrigins] = []
    for alias_group in aliases.groups:
        candidate_documents: dict[str, SourceDocument] = {}
        for document_id in alias_group.document_ids:
            document = documents_by_id.get(document_id)
            if document is None:
                raise ValueError("alias group refers to an unavailable discovery document")
            candidate_documents[document_id] = document
        for document in verification_by_id[alias_group.group_id].matching_documents:
            previous = candidate_documents.setdefault(document.document_id, document)
            if previous != document:
                raise ValueError("document_id refers to conflicting source documents")

        by_origin: dict[str, list[SourceDocument]] = {}
        for document in candidate_documents.values():
            by_origin.setdefault(document.origin_id, []).append(document)
        groups = tuple(
            OriginGroup(origin_id, tuple(sorted(items, key=lambda item: item.document_id)))
            for origin_id, items in sorted(by_origin.items())
        )
        candidates.append(CandidateOrigins(
            alias_group_id=alias_group.group_id,
            origin_groups=groups,
            echo_suggestions=_news_echo_suggestions(tuple(candidate_documents.values())),
        ))
    return OriginResolutionResult(plan.plan_id, tuple(candidates))


def _news_echo_suggestions(documents: tuple[SourceDocument, ...]) -> tuple[EchoSuggestion, ...]:
    by_title: dict[str, list[SourceDocument]] = {}
    for document in documents:
        if document.source_type not in _NEWS_TYPES or document.published_at is None:
            continue
        title = _normalized_title(document.title)
        if title:
            by_title.setdefault(title, []).append(document)

    suggestions: dict[tuple[str, str], EchoSuggestion] = {}
    for same_title in by_title.values():
        ordered = sorted(same_title, key=lambda item: (item.origin_id, item.document_id))
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                if left.origin_id == right.origin_id:
                    continue
                gap = _date_gap(left.published_at, right.published_at)
                if gap > 3:
                    continue
                key = (left.origin_id, right.origin_id)
                suggestion = EchoSuggestion(
                    left.origin_id,
                    right.origin_id,
                    left.document_id,
                    right.document_id,
                    EchoSuggestionReason.SAME_NEWS_TITLE_NEAR_DATE,
                    gap,
                )
                previous = suggestions.get(key)
                if previous is None or gap < previous.date_gap_days:
                    suggestions[key] = suggestion
    return tuple(suggestions[key] for key in sorted(suggestions))


def _normalized_title(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", normalized).split())


def _date_gap(left: date | None, right: date | None) -> int:
    if left is None or right is None:
        raise ValueError("echo comparison requires publication dates")
    return abs((left - right).days)
