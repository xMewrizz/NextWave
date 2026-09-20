from __future__ import annotations

import json
import unittest
from datetime import date

from nextwave.contracts import SourceDocument, SourceType, TrustTier
from nextwave.discovery import (
    CandidateHintKind,
    ProposalExclusionReason,
    ScopeGranularity,
    build_analysis_scope,
    build_candidate_proposals,
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
    origin_id: str | None = None,
    organizations: tuple[str, ...] = (),
) -> SourceDocument:
    return SourceDocument(
        document_id=f"document-{number}",
        connector_id="openalex",
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

        batch = build_candidate_proposals(scope(), documents, hints)

        self.assertEqual(len(batch.proposals), 1)
        proposal = batch.proposals[0]
        self.assertEqual(proposal.canonical_name, "Speculative decoding")
        self.assertEqual(proposal.normalized_name, "speculative decoding")
        self.assertEqual(proposal.document_count, 2)
        self.assertEqual(proposal.origin_count, 2)
        self.assertEqual(proposal.max_score, 0.91)
        self.assertTrue(proposal.primary_topic)
        self.assertEqual(
            proposal.source_kinds,
            (CandidateHintKind.KEYWORD, CandidateHintKind.TOPIC),
        )
        self.assertEqual(len(proposal.provider_term_ids), 2)
        json.dumps(batch.to_dict(), ensure_ascii=False)

    def test_excludes_scope_terms_and_organization_names_with_reasons(self) -> None:
        documents = (document(1, organizations=("Example University",)),)
        hints = (
            OpenAlexDiscoveryHints(
                document_id="document-1",
                topics=(topic("T100", "Artificial Intelligence", 0.99),),
                keywords=(keyword("example-university", "Example University", 0.74),),
            ),
        )

        batch = build_candidate_proposals(scope(), documents, hints)

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

        batch = build_candidate_proposals(scope(), documents, hints)

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

        proposal = build_candidate_proposals(scope(), documents, hints).proposals[0]

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
            build_candidate_proposals(scope(), (document(1),), hints)


if __name__ == "__main__":
    unittest.main()
