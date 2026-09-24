from __future__ import annotations

import json
import unittest
from datetime import date

from nextwave.contracts import SourceDocument, SourceType, TrustTier
from nextwave.discovery import (
    CandidateMention,
    CandidateMentionKind,
    ProposalExclusionReason,
    ScopeGranularity,
    build_analysis_scope,
    build_candidate_mention,
    build_candidate_proposals,
    build_openalex_candidate_mentions,
)
from nextwave.sources import (
    OpenAlexDiscoveryHints,
    OpenAlexKeywordHint,
    OpenAlexTopicHint,
)


def scope():
    return build_analysis_scope(
        raw_query="Технологии в ИИ",
        normalized_query="artificial intelligence",
        search_texts=("AI",),
        languages=("en", "ru"),
        granularity=ScopeGranularity.DIRECTION,
        subfield_ids=("1702",),
    )


def document(
    number: int,
    *,
    connector_id: str = "openalex",
    origin_id: str | None = None,
    organizations: tuple[str, ...] = (),
) -> SourceDocument:
    return SourceDocument(
        document_id=f"document-{number}",
        connector_id=connector_id,
        external_id=f"W{number}",
        snapshot_id="snapshot-ai-001",
        title=f"Research work {number}",
        url=f"https://example.org/work-{number}",
        canonical_url=f"https://example.org/work-{number}",
        source_type=SourceType.SCIENTIFIC_PUBLICATION,
        language="en",
        trust_tier=TrustTier.A,
        origin_id=origin_id or f"doi:10.1234/work-{number}",
        organizations=organizations,
        published_at=date(2026, 8, number),
    )


def keyword(
    keyword_id: str,
    name: str,
    score: float,
) -> OpenAlexKeywordHint:
    return OpenAlexKeywordHint(
        keyword_id=keyword_id,
        display_name=name,
        score=score,
    )


def topic(
    topic_id: str,
    name: str,
    score: float,
    *,
    primary: bool = False,
) -> OpenAlexTopicHint:
    return OpenAlexTopicHint(
        topic_id=topic_id,
        display_name=name,
        score=score,
        primary=primary,
        subfield_id="1702",
        subfield_name="Artificial Intelligence",
        field_id="17",
        field_name="Computer Science",
        domain_id="3",
        domain_name="Physical Sciences",
    )


def grounded(document: SourceDocument, *names: str) -> tuple[CandidateMention, ...]:
    return tuple(
        build_candidate_mention(
            document,
            text=name,
            kind=CandidateMentionKind.TITLE,
            locator=f"title[{index}]",
            extractor_id="text-v1",
        )
        for index, name in enumerate(names)
    )


class CandidateProposalTests(unittest.TestCase):
    def test_aggregates_same_term_across_independent_origins(self) -> None:
        documents = (document(1), document(2))
        hints = (
            OpenAlexDiscoveryHints(
                document_id="document-1",
                topics=(topic("T100", "Speculative Decoding", 0.79, primary=True),),
                keywords=(keyword("speculative-decoding", "Speculative decoding", 0.91),),
            ),
            OpenAlexDiscoveryHints(
                document_id="document-2",
                topics=(),
                keywords=(keyword("speculative-decoding", "SPECULATIVE DECODING", 0.84),),
            ),
        )

        mentions = build_openalex_candidate_mentions(
            documents,
            hints,
            grounded_mentions=(
                *grounded(documents[0], "Speculative decoding"),
                *grounded(documents[1], "Speculative decoding"),
            ),
        )
        batch = build_candidate_proposals(scope(), documents, mentions)

        self.assertEqual(len(batch.proposals), 1)
        proposal = batch.proposals[0]
        self.assertEqual(proposal.canonical_name, "Speculative decoding")
        self.assertEqual(proposal.normalized_name, "speculative decoding")
        self.assertEqual(proposal.document_count, 2)
        self.assertEqual(proposal.origin_count, 2)
        self.assertEqual(proposal.max_provider_score, 0.91)
        self.assertTrue(proposal.primary_provider_topic)
        self.assertEqual(
            proposal.source_kinds,
            (
                CandidateMentionKind.PROVIDER_KEYWORD,
                CandidateMentionKind.PROVIDER_TOPIC,
            ),
        )
        self.assertEqual(proposal.connector_ids, ("openalex",))
        self.assertEqual(len(proposal.mention_ids), 3)
        self.assertEqual(len(proposal.provider_term_ids), 2)
        json.dumps(batch.to_dict(), ensure_ascii=False)

    def test_groups_hyphen_and_space_spellings_before_gate(self) -> None:
        documents = (document(1), document(2))
        mentions = (
            build_candidate_mention(
                documents[0],
                text="Speculative-decoding",
                kind=CandidateMentionKind.TITLE,
                locator="title[0:20]",
                extractor_id="text-v1",
            ),
            build_candidate_mention(
                documents[1],
                text="speculative decoding",
                kind=CandidateMentionKind.TITLE,
                locator="title[0:20]",
                extractor_id="text-v1",
            ),
        )
        batch = build_candidate_proposals(scope(), documents, mentions)

        self.assertEqual(len(batch.proposals), 1)
        self.assertEqual(batch.proposals[0].normalized_name, "speculative decoding")
        self.assertEqual(batch.proposals[0].origin_count, 2)
        self.assertEqual(
            {batch.proposals[0].canonical_name, *batch.proposals[0].aliases},
            {"Speculative-decoding", "speculative decoding"},
        )

    def test_excludes_scope_terms_and_organization_names_with_reasons(self) -> None:
        documents = (document(1, organizations=("Example University",)),)
        hints = (
            OpenAlexDiscoveryHints(
                document_id="document-1",
                topics=(topic("T100", "Artificial Intelligence", 0.99),),
                keywords=(keyword("example-university", "Example University", 0.74),),
            ),
        )

        mentions = build_openalex_candidate_mentions(
            documents,
            hints,
            grounded_mentions=grounded(
                documents[0],
                "Artificial Intelligence",
                "Example University",
            ),
        )
        batch = build_candidate_proposals(scope(), documents, mentions)

        self.assertEqual(batch.proposals, ())
        self.assertEqual(
            {exclusion.reason for exclusion in batch.exclusions},
            {
                ProposalExclusionReason.SCOPE_TERM,
                ProposalExclusionReason.ORGANIZATION,
            },
        )

    def test_ranking_uses_independent_origins_before_provider_score(self) -> None:
        documents = (document(1), document(2), document(3))
        hints = (
            OpenAlexDiscoveryHints(
                "document-1",
                (),
                (
                    keyword("high-score-single", "High score single", 0.99),
                    keyword("repeated", "Repeated mechanism", 0.61),
                ),
            ),
            OpenAlexDiscoveryHints(
                "document-2",
                (),
                (keyword("repeated", "Repeated mechanism", 0.62),),
            ),
            OpenAlexDiscoveryHints(
                "document-3",
                (),
                (keyword("other", "Other mechanism", 0.70),),
            ),
        )

        mentions = build_openalex_candidate_mentions(
            documents,
            hints,
            grounded_mentions=(
                *grounded(documents[0], "High score single", "Repeated mechanism"),
                *grounded(documents[1], "Repeated mechanism"),
                *grounded(documents[2], "Other mechanism"),
            ),
        )
        batch = build_candidate_proposals(scope(), documents, mentions)

        self.assertEqual(batch.proposals[0].canonical_name, "Repeated mechanism")
        self.assertEqual(batch.proposals[0].origin_count, 2)

    def test_does_not_count_duplicate_documents_as_independent_origins(self) -> None:
        documents = (
            document(1, origin_id="doi:10.1234/shared"),
            document(2, origin_id="doi:10.1234/shared"),
        )
        hints = tuple(
            OpenAlexDiscoveryHints(
                f"document-{number}",
                (),
                (keyword("mechanism", "Shared mechanism", 0.8),),
            )
            for number in (1, 2)
        )

        mentions = build_openalex_candidate_mentions(
            documents,
            hints,
            grounded_mentions=(
                *grounded(documents[0], "Shared mechanism"),
                *grounded(documents[1], "Shared mechanism"),
            ),
        )
        proposal = build_candidate_proposals(scope(), documents, mentions).proposals[0]

        self.assertEqual(proposal.document_count, 2)
        self.assertEqual(proposal.origin_count, 1)

    def test_rejects_hints_that_reference_an_unknown_document(self) -> None:
        hints = (
            OpenAlexDiscoveryHints(
                "document-missing",
                (),
                (keyword("mechanism", "Unknown mechanism", 0.8),),
            ),
        )

        with self.assertRaisesRegex(ValueError, "reference a supplied document"):
            build_openalex_candidate_mentions(
                (document(1),),
                hints,
                grounded_mentions=(),
            )

    def test_does_not_promote_ungrounded_provider_terms(self) -> None:
        documents = (document(1),)
        hints = (
            OpenAlexDiscoveryHints(
                document_id="document-1",
                topics=(topic("T100", "Artificial Intelligence", 0.99),),
                keywords=(keyword("inference", "Inference", 0.95),),
            ),
        )

        mentions = build_openalex_candidate_mentions(
            documents,
            hints,
            grounded_mentions=(),
        )

        self.assertEqual(mentions, ())

    def test_combines_mentions_from_scientific_and_media_connectors(self) -> None:
        openalex_document = document(1)
        media_document = document(2, connector_id="mediacloud")
        openalex_mention = build_candidate_mention(
            openalex_document,
            text="Speculative Decoding",
            kind=CandidateMentionKind.PROVIDER_KEYWORD,
            locator="keywords",
            extractor_id="openalex-hints-v1",
            provider_term_id="openalex:keyword:speculative-decoding",
            provider_score=0.81,
        )
        media_mention = build_candidate_mention(
            media_document,
            text="speculative decoding",
            kind=CandidateMentionKind.HEADLINE,
            locator="title[0:20]",
            extractor_id="yandex-yandexgpt-lite-5-candidate-text-v1",
        )

        proposal = build_candidate_proposals(
            scope(),
            (openalex_document, media_document),
            (openalex_mention, media_mention),
        ).proposals[0]

        self.assertEqual(proposal.origin_count, 2)
        self.assertEqual(proposal.connector_ids, ("mediacloud", "openalex"))
        self.assertEqual(
            proposal.source_kinds,
            (
                CandidateMentionKind.HEADLINE,
                CandidateMentionKind.PROVIDER_KEYWORD,
            ),
        )
        self.assertEqual(proposal.max_provider_score, 0.81)

    def test_rejects_a_mention_with_a_connector_mismatch(self) -> None:
        source_document = document(1)
        mention = build_candidate_mention(
            source_document,
            text="Speculative Decoding",
            kind=CandidateMentionKind.TITLE,
            locator="title[0:20]",
            extractor_id="yandex-yandexgpt-lite-5-candidate-text-v1",
        )
        mismatched = type(mention)(
            mention_id=mention.mention_id,
            document_id=mention.document_id,
            connector_id="gdelt",
            text=mention.text,
            normalized_text=mention.normalized_text,
            kind=mention.kind,
            locator=mention.locator,
            extractor_id=mention.extractor_id,
        )

        with self.assertRaisesRegex(ValueError, "connector must match"):
            build_candidate_proposals(scope(), (source_document,), (mismatched,))


if __name__ == "__main__":
    unittest.main()
