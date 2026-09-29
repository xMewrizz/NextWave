"""Provider-neutral candidate mentions and auditable proposal aggregation."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import asdict, dataclass
from enum import Enum, StrEnum
from typing import Any

from nextwave.contracts import SourceDocument
from nextwave.sources import OpenAlexDiscoveryHints

from .contracts import AnalysisScope

CANDIDATE_MENTION_VERSION = "candidate-mention-v1"
CANDIDATE_PROPOSAL_VERSION = "candidate-proposal-v4"
OPENALEX_HINT_EXTRACTOR_ID = "openalex-hints-v1"
_ORTHOGRAPHIC_SEPARATORS = re.compile(r"[-‐‑‒–—−_]+")


class CandidateMentionKind(StrEnum):
    """Location or provider field from which a possible technology name was extracted."""

    PROVIDER_KEYWORD = "provider_keyword"
    PROVIDER_TOPIC = "provider_topic"
    PROVIDER_SUBJECT = "provider_subject"
    TITLE = "title"
    EXCERPT = "excerpt"
    HEADLINE = "headline"


class ProposalExclusionReason(StrEnum):
    SCOPE_TERM = "scope_term"
    ORGANIZATION = "organization"
    INVALID_NAME = "invalid_name"


@dataclass(frozen=True, slots=True)
class CandidateMention:
    """One source-grounded suggestion before alias resolution and candidate gating."""

    mention_id: str
    document_id: str
    connector_id: str
    text: str
    normalized_text: str
    kind: CandidateMentionKind
    locator: str
    extractor_id: str
    provider_term_id: str | None = None
    provider_score: float | None = None
    primary_provider_topic: bool = False
    mention_version: str = CANDIDATE_MENTION_VERSION

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.mention_id, "mention_id"),
            (self.document_id, "document_id"),
            (self.connector_id, "connector_id"),
            (self.text, "text"),
            (self.normalized_text, "normalized_text"),
            (self.locator, "locator"),
            (self.extractor_id, "extractor_id"),
            (self.mention_version, "mention_version"),
        ):
            _require_text(value, field_name)
        if not isinstance(self.kind, CandidateMentionKind):
            raise ValueError("kind must be a CandidateMentionKind")
        if self.normalized_text != _normalize_name(self.text):
            raise ValueError("normalized_text must match normalized text")
        if self.provider_term_id is not None:
            _require_text(self.provider_term_id, "provider_term_id")
        if self.kind in {
            CandidateMentionKind.PROVIDER_KEYWORD,
            CandidateMentionKind.PROVIDER_TOPIC,
            CandidateMentionKind.PROVIDER_SUBJECT,
        } and self.provider_term_id is None:
            raise ValueError("provider mentions require provider_term_id")
        if self.provider_score is not None:
            if isinstance(self.provider_score, bool) or not isinstance(
                self.provider_score, int | float
            ):
                raise ValueError("provider_score must be a number")
            if not 0.0 <= self.provider_score <= 1.0:
                raise ValueError("provider_score must be between 0 and 1")
        if (
            self.primary_provider_topic
            and self.kind is not CandidateMentionKind.PROVIDER_TOPIC
        ):
            raise ValueError("primary_provider_topic requires a provider topic")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True, slots=True)
class CandidateProposal:
    proposal_id: str
    analysis_scope_id: str
    canonical_name: str
    normalized_name: str
    aliases: tuple[str, ...]
    mention_ids: tuple[str, ...]
    source_kinds: tuple[CandidateMentionKind, ...]
    connector_ids: tuple[str, ...]
    provider_term_ids: tuple[str, ...]
    document_ids: tuple[str, ...]
    origin_ids: tuple[str, ...]
    max_provider_score: float | None
    primary_provider_topic: bool
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
    mention_id: str
    display_name: str
    normalized_name: str
    connector_id: str
    provider_term_id: str | None
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
    names: dict[str, float | None]
    mention_ids: set[str]
    source_kinds: set[CandidateMentionKind]
    connector_ids: set[str]
    provider_term_ids: set[str]
    document_ids: set[str]
    origin_ids: set[str]
    max_provider_score: float | None
    primary_provider_topic: bool


def build_candidate_mention(
    document: SourceDocument,
    *,
    text: str,
    kind: CandidateMentionKind,
    locator: str,
    extractor_id: str,
    provider_term_id: str | None = None,
    provider_score: float | None = None,
    primary_provider_topic: bool = False,
) -> CandidateMention:
    """Create one stable mention from a source-specific extractor."""

    normalized_text = _normalize_name(text)
    identity = "|".join(
        (
            document.document_id,
            kind.value,
            locator,
            extractor_id,
            provider_term_id or "",
            normalized_text,
        )
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
    return CandidateMention(
        mention_id=f"mention-{digest}",
        document_id=document.document_id,
        connector_id=document.connector_id,
        text=text,
        normalized_text=normalized_text,
        kind=kind,
        locator=locator,
        extractor_id=extractor_id,
        provider_term_id=provider_term_id,
        provider_score=provider_score,
        primary_provider_topic=primary_provider_topic,
    )


def build_openalex_candidate_mentions(
    documents: tuple[SourceDocument, ...],
    hints: tuple[OpenAlexDiscoveryHints, ...],
    *,
    grounded_mentions: tuple[CandidateMention, ...],
) -> tuple[CandidateMention, ...]:
    """Attach OpenAlex metadata only to names already grounded in document text.

    OpenAlex topics and keywords describe what a publication is about.  They are
    useful corroborating metadata, but are not proof that the term names a
    concrete technology.  A provider term therefore becomes a mention only
    when the text extractor found the same normalized name in the same
    document title or abstract.
    """

    documents_by_id = _documents_by_id(documents)
    if len({hint.document_id for hint in hints}) != len(hints):
        raise ValueError("hints must contain unique document_id values")
    grounded_names = {
        (mention.document_id, mention.normalized_text)
        for mention in grounded_mentions
        if mention.kind
        in {
            CandidateMentionKind.TITLE,
            CandidateMentionKind.EXCERPT,
        }
    }
    mentions: list[CandidateMention] = []
    for hint in hints:
        document = documents_by_id.get(hint.document_id)
        if document is None:
            raise ValueError("every discovery hint must reference a supplied document")
        if document.connector_id != "openalex":
            raise ValueError("OpenAlex hints must reference an OpenAlex document")
        for keyword in hint.keywords:
            if (hint.document_id, _normalize_name(keyword.display_name)) not in grounded_names:
                continue
            mentions.append(
                build_candidate_mention(
                    document,
                    text=keyword.display_name,
                    kind=CandidateMentionKind.PROVIDER_KEYWORD,
                    locator="keywords",
                    extractor_id=OPENALEX_HINT_EXTRACTOR_ID,
                    provider_term_id=f"openalex:keyword:{keyword.keyword_id}",
                    provider_score=keyword.score,
                )
            )
        for topic in hint.topics:
            if (hint.document_id, _normalize_name(topic.display_name)) not in grounded_names:
                continue
            mentions.append(
                build_candidate_mention(
                    document,
                    text=topic.display_name,
                    kind=CandidateMentionKind.PROVIDER_TOPIC,
                    locator="topics",
                    extractor_id=OPENALEX_HINT_EXTRACTOR_ID,
                    provider_term_id=f"openalex:topic:{topic.topic_id}",
                    provider_score=topic.score,
                    primary_provider_topic=topic.primary,
                )
            )
    return tuple(mentions)


def build_candidate_proposals(
    scope: AnalysisScope,
    documents: tuple[SourceDocument, ...],
    mentions: tuple[CandidateMention, ...],
) -> CandidateProposalBatch:
    """Aggregate mentions from every connector without accepting technologies yet."""

    documents_by_id = _documents_by_id(documents)
    if len({mention.mention_id for mention in mentions}) != len(mentions):
        raise ValueError("mentions must contain unique mention_id values")
    for mention in mentions:
        document = documents_by_id.get(mention.document_id)
        if document is None:
            raise ValueError("every candidate mention must reference a supplied document")
        if mention.connector_id != document.connector_id:
            raise ValueError("candidate mention connector must match its document")

    scope_terms = {orthographic_candidate_key(scope.normalized_query)}
    scope_terms.update(orthographic_candidate_key(value) for value in scope.search_texts)
    organizations = {
        orthographic_candidate_key(name)
        for document in documents
        for name in document.organizations
    }
    accumulators: dict[str, _ProposalAccumulator] = {}
    exclusions: list[CandidateProposalExclusion] = []

    for mention in mentions:
        _add_mention(
            accumulators,
            exclusions,
            scope_terms,
            organizations,
            mention=mention,
            document=documents_by_id[mention.document_id],
        )

    proposals = tuple(
        sorted(
            (_build_proposal(scope.scope_id, name, value) for name, value in accumulators.items()),
            key=lambda proposal: (
                -proposal.origin_count,
                -proposal.document_count,
                -_sortable_score(proposal.max_provider_score),
                proposal.normalized_name,
            ),
        )
    )
    return CandidateProposalBatch(
        analysis_scope_id=scope.scope_id,
        proposals=proposals,
        exclusions=tuple(exclusions),
    )


def _add_mention(
    accumulators: dict[str, _ProposalAccumulator],
    exclusions: list[CandidateProposalExclusion],
    scope_terms: set[str],
    organizations: set[str],
    *,
    mention: CandidateMention,
    document: SourceDocument,
) -> None:
    normalized = orthographic_candidate_key(mention.text)
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
                mention_id=mention.mention_id,
                display_name=mention.text,
                normalized_name=normalized,
                connector_id=mention.connector_id,
                provider_term_id=mention.provider_term_id,
                document_id=document.document_id,
                reason=reason,
            )
        )
        return

    accumulator = accumulators.get(normalized)
    if accumulator is None:
        accumulators[normalized] = _ProposalAccumulator(
            names={mention.text: mention.provider_score},
            mention_ids={mention.mention_id},
            source_kinds={mention.kind},
            connector_ids={mention.connector_id},
            provider_term_ids={mention.provider_term_id} if mention.provider_term_id else set(),
            document_ids={document.document_id},
            origin_ids={document.origin_id},
            max_provider_score=mention.provider_score,
            primary_provider_topic=mention.primary_provider_topic,
        )
        return
    current_name_score = accumulator.names.get(mention.text)
    if current_name_score is None or (
        mention.provider_score is not None and mention.provider_score > current_name_score
    ):
        accumulator.names[mention.text] = mention.provider_score
    accumulator.mention_ids.add(mention.mention_id)
    accumulator.source_kinds.add(mention.kind)
    accumulator.connector_ids.add(mention.connector_id)
    if mention.provider_term_id is not None:
        accumulator.provider_term_ids.add(mention.provider_term_id)
    accumulator.document_ids.add(document.document_id)
    accumulator.origin_ids.add(document.origin_id)
    if mention.provider_score is not None:
        accumulator.max_provider_score = max(
            accumulator.max_provider_score if accumulator.max_provider_score is not None else -1.0,
            mention.provider_score,
        )
    accumulator.primary_provider_topic = (
        accumulator.primary_provider_topic or mention.primary_provider_topic
    )


def _build_proposal(
    scope_id: str,
    normalized_name: str,
    value: _ProposalAccumulator,
) -> CandidateProposal:
    digest = hashlib.sha256(f"{scope_id}|{normalized_name}".encode()).hexdigest()[:16]
    canonical_name = min(
        value.names,
        key=lambda name: (
            -_sortable_score(value.names[name]),
            name.casefold(),
            name,
        ),
    )
    aliases = tuple(
        sorted(
            (name for name in value.names if name != canonical_name),
            key=lambda name: (name.casefold(), name),
        )
    )
    return CandidateProposal(
        proposal_id=f"proposal-{digest}",
        analysis_scope_id=scope_id,
        canonical_name=canonical_name,
        normalized_name=normalized_name,
        aliases=aliases,
        mention_ids=tuple(sorted(value.mention_ids)),
        source_kinds=tuple(sorted(value.source_kinds, key=lambda item: item.value)),
        connector_ids=tuple(sorted(value.connector_ids)),
        provider_term_ids=tuple(sorted(value.provider_term_ids)),
        document_ids=tuple(sorted(value.document_ids)),
        origin_ids=tuple(sorted(value.origin_ids)),
        max_provider_score=value.max_provider_score,
        primary_provider_topic=value.primary_provider_topic,
    )


def _documents_by_id(
    documents: tuple[SourceDocument, ...],
) -> dict[str, SourceDocument]:
    result = {document.document_id: document for document in documents}
    if len(result) != len(documents):
        raise ValueError("documents must contain unique document_id values")
    return result


def _normalize_name(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("candidate mention text must be a string")
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def orthographic_candidate_key(value: str) -> str:
    """Group only obvious spelling variants before the candidate gate.

    Besides case, whitespace, underscores, and Unicode dashes, a conservative
    English plural rule folds a final ``s``.  Token order stays intact and no
    abbreviations or semantic synonyms are expanded, so RAG and
    Retrieval-Augmented Generation remain separate proposals.
    """

    normalized = " ".join(
        _ORTHOGRAPHIC_SEPARATORS.sub(" ", _normalize_name(value)).split()
    )
    return " ".join(_singularize_candidate_token(token) for token in normalized.split())


def _singularize_candidate_token(token: str) -> str:
    if (
        token.isascii()
        and token.isalpha()
        and len(token) > 3
        and token.endswith("s")
        and not token.endswith(("ss", "us", "is"))
    ):
        return token[:-1]
    return token


def _sortable_score(value: float | None) -> float:
    return value if value is not None else -1.0


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value
