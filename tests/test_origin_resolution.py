from __future__ import annotations

import json
import unittest
from datetime import date

from nextwave.contracts import SourceDocument, SourceType, TrustTier
from nextwave.discovery import (
    AliasResolutionResult,
    CandidateVerificationResult,
    DiscoveryBudget,
    EchoSuggestionReason,
    GroupVerification,
    ResolvedAliasGroup,
    ScopeGranularity,
    VerificationStatus,
    build_analysis_scope,
    build_discovery_plan,
    resolve_candidate_origins,
)
from nextwave.sources import ConnectorId


def plan():
    scope = build_analysis_scope(
        raw_query="Технологии в ИИ",
        normalized_query="artificial intelligence",
        search_texts=("AI",),
        languages=("en", "ru"),
        granularity=ScopeGranularity.DIRECTION,
        subfield_ids=("1702",),
    )
    return build_discovery_plan(
        analysis_id="analysis-ai-001",
        scope=scope,
        published_from=date(2025, 9, 15),
        cutoff_date=date(2026, 9, 15),
        budgets=(DiscoveryBudget(ConnectorId.OPENALEX, True, 1, 1, 20, 10, 20),),
    )


def document(
    number: int,
    *,
    origin_id: str,
    title: str = "Speculative decoding research",
    source_type: SourceType = SourceType.SCIENTIFIC_PUBLICATION,
    published_at: date = date(2026, 8, 10),
    connector_id: str = "openalex",
) -> SourceDocument:
    url = f"https://example.org/work-{number}"
    return SourceDocument(
        document_id=f"document-{number}",
        connector_id=connector_id,
        external_id=f"W{number}",
        snapshot_id=f"snapshot-{number}",
        title=title,
        url=url,
        canonical_url=url,
        source_type=source_type,
        language="en",
        trust_tier=TrustTier.A,
        origin_id=origin_id,
        doi=origin_id.removeprefix("doi:") if origin_id.startswith("doi:") else None,
        published_at=published_at,
    )


def inputs(discovery_documents, verification_documents=()):
    discovery_plan = plan()
    group = ResolvedAliasGroup(
        group_id="alias-group-1",
        analysis_scope_id=discovery_plan.scope.scope_id,
        canonical_name="Speculative decoding",
        aliases=(),
        proposal_ids=("proposal-1",),
        document_ids=tuple(item.document_id for item in discovery_documents),
        origin_ids=tuple(sorted({item.origin_id for item in discovery_documents})),
        connector_ids=tuple(sorted({item.connector_id for item in discovery_documents})),
        normalization_key="speculative decoding",
    )
    aliases = AliasResolutionResult(
        discovery_plan.scope.scope_id, ("proposal-1",), (group,), ()
    )
    verification = CandidateVerificationResult(
        discovery_plan.plan_id,
        (GroupVerification(
            group.group_id,
            VerificationStatus.SEARCHED,
            group.canonical_name,
            len(verification_documents),
            tuple(verification_documents),
            None,
        ),),
        1,
    )
    return discovery_plan, aliases, verification


class OriginResolutionTests(unittest.TestCase):
    def test_same_doi_across_snapshots_is_one_exact_origin(self) -> None:
        first = document(1, origin_id="doi:10.1234/shared")
        second = document(2, origin_id="doi:10.1234/shared")
        discovery_plan, aliases, verification = inputs((first,), (second,))

        result = resolve_candidate_origins(
            discovery_plan, aliases, verification, (first,)
        )
        candidate = result.candidates[0]
        self.assertEqual(candidate.document_count, 2)
        self.assertEqual(candidate.exact_origin_count, 1)
        self.assertEqual(
            candidate.origin_groups[0].to_dict()["document_ids"],
            ["document-1", "document-2"],
        )
        json.dumps(result.to_dict(), ensure_ascii=False)

    def test_same_news_title_near_date_is_flagged_without_merging(self) -> None:
        first = document(
            1,
            origin_id="url:https://news.example/story-1",
            title="Startup announces photonic inference chip",
            source_type=SourceType.INDUSTRY_MEDIA,
        )
        second = document(
            2,
            origin_id="url:https://other.example/story-2",
            title="Startup announces photonic inference chip",
            source_type=SourceType.INDUSTRY_MEDIA,
            published_at=date(2026, 8, 11),
        )
        discovery_plan, aliases, verification = inputs((first, second))

        candidate = resolve_candidate_origins(
            discovery_plan, aliases, verification, (first, second)
        ).candidates[0]
        self.assertEqual(candidate.exact_origin_count, 2)
        self.assertEqual(len(candidate.echo_suggestions), 1)
        self.assertEqual(
            candidate.echo_suggestions[0].reason,
            EchoSuggestionReason.SAME_NEWS_TITLE_NEAR_DATE,
        )
        self.assertEqual(candidate.echo_suggestions[0].date_gap_days, 1)

    def test_same_research_title_does_not_merge_independent_dois(self) -> None:
        first = document(1, origin_id="doi:10.1234/first")
        second = document(2, origin_id="doi:10.1234/second")
        discovery_plan, aliases, verification = inputs((first, second))

        candidate = resolve_candidate_origins(
            discovery_plan, aliases, verification, (first, second)
        ).candidates[0]
        self.assertEqual(candidate.exact_origin_count, 2)
        self.assertEqual(candidate.echo_suggestions, ())

    def test_missing_discovery_document_is_not_silently_ignored(self) -> None:
        first = document(1, origin_id="doi:10.1234/first")
        discovery_plan, aliases, verification = inputs((first,))
        with self.assertRaisesRegex(ValueError, "unavailable discovery document"):
            resolve_candidate_origins(discovery_plan, aliases, verification, ())


if __name__ == "__main__":
    unittest.main()
