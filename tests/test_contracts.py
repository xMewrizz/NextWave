from __future__ import annotations

import unittest
from datetime import UTC, date, datetime

from nextwave.contracts import (
    CandidateAssessment,
    CandidateFeatures,
    CandidateStatus,
    ClaimType,
    DevelopmentStage,
    EvidenceClaim,
    EvidenceDirection,
    ExclusionReason,
    ModelPrediction,
    SourceDocument,
    SourceType,
    TemporalFeatures,
    TrustTier,
)


def make_temporal() -> TemporalFeatures:
    return TemporalFeatures(
        source_id="openalex",
        analysis_scope_id="scope-industrial-ai-v1",
        previous_document_count=3,
        recent_document_count=9,
        previous_origin_count=2,
        recent_origin_count=6,
        previous_query_share=0.01,
        recent_query_share=0.03,
        query_share_change=0.02,
        first_seen_at=date(2024, 3, 1),
        coverage_complete=True,
    )


def make_features() -> CandidateFeatures:
    return CandidateFeatures(
        stage=DevelopmentStage.PILOT,
        temporal=make_temporal(),
        query_relevance=0.88,
        independent_origin_count=3,
        source_type_diversity=2,
        independent_actor_count=2,
        echo_share=0.2,
        promotional_source_share=0.1,
    )


def make_prediction() -> ModelPrediction:
    return ModelPrediction(
        model_score=0.78,
        decision_threshold=0.62,
        model_version="baseline-1",
        feature_version="candidate-features-1",
        calibrated=False,
    )


def make_document() -> SourceDocument:
    return SourceDocument(
        document_id="document-1",
        connector_id="openalex",
        external_id="W123456789",
        snapshot_id="snapshot-1",
        title="Prototype evaluation",
        url="https://example.org/prototype",
        canonical_url="https://example.org/prototype",
        source_type=SourceType.SCIENTIFIC_PUBLICATION,
        language="en",
        trust_tier=TrustTier.A,
        origin_id="doi:10.0000/example",
        published_at=date(2026, 8, 1),
    )


def make_claim() -> EvidenceClaim:
    return EvidenceClaim(
        claim_id="claim-1",
        document_id="document-1",
        claim_type=ClaimType.PILOT,
        direction=EvidenceDirection.SUPPORT,
        text="The prototype was evaluated in a limited pilot.",
        locator="abstract",
        extraction_confidence=0.91,
    )


class ContractTests(unittest.TestCase):
    def test_main_candidate_requires_and_accepts_supporting_evidence(self) -> None:
        assessment = CandidateAssessment(
            candidate_id="candidate-1",
            group_id="technology-family-1",
            canonical_name="Example technology",
            aliases=("Example mechanism",),
            query="industrial AI",
            analysis_scope_id="scope-industrial-ai-v1",
            cutoff_date=date(2026, 9, 15),
            status=CandidateStatus.MAIN,
            features=make_features(),
            prediction=make_prediction(),
            explanation="Early pilot with independent evidence.",
            documents=(make_document(),),
            claims=(make_claim(),),
            rank=1,
        )

        self.assertEqual(assessment.status, CandidateStatus.MAIN)
        self.assertEqual(assessment.rank, 1)

    def test_main_candidate_requires_supporting_claim(self) -> None:
        with self.assertRaisesRegex(ValueError, "grounded A/B supporting evidence"):
            CandidateAssessment(
                candidate_id="candidate-2",
                group_id="technology-family-2",
                canonical_name="Unverified technology",
                aliases=(),
                query="industrial AI",
                analysis_scope_id="scope-industrial-ai-v1",
                cutoff_date=date(2026, 9, 15),
                status=CandidateStatus.MAIN,
                features=make_features(),
                prediction=make_prediction(),
                explanation="No grounded support.",
                rank=1,
            )

    def test_main_candidate_must_pass_model_threshold(self) -> None:
        prediction = ModelPrediction(
            model_score=0.4,
            decision_threshold=0.62,
            model_version="baseline-1",
            feature_version="candidate-features-1",
        )
        with self.assertRaisesRegex(ValueError, "model threshold"):
            CandidateAssessment(
                candidate_id="candidate-low-score",
                group_id="technology-family-low-score",
                canonical_name="Low-score technology",
                aliases=(),
                query="industrial AI",
                analysis_scope_id="scope-industrial-ai-v1",
                cutoff_date=date(2026, 9, 15),
                status=CandidateStatus.MAIN,
                features=make_features(),
                prediction=prediction,
                explanation="Evidence exists, but the model score is below threshold.",
                documents=(make_document(),),
                claims=(make_claim(),),
                rank=1,
            )

    def test_claim_must_reference_included_document(self) -> None:
        unknown_document_claim = EvidenceClaim(
            claim_id="claim-missing-document",
            document_id="document-not-in-assessment",
            claim_type=ClaimType.PILOT,
            direction=EvidenceDirection.SUPPORT,
            text="A pilot was reported.",
            locator="abstract",
        )
        with self.assertRaisesRegex(ValueError, "reference a document"):
            CandidateAssessment(
                candidate_id="candidate-broken-evidence",
                group_id="technology-family-broken-evidence",
                canonical_name="Broken evidence graph",
                aliases=(),
                query="industrial AI",
                analysis_scope_id="scope-industrial-ai-v1",
                cutoff_date=date(2026, 9, 15),
                status=CandidateStatus.WATCHLIST,
                features=make_features(),
                prediction=None,
                explanation="The evidence graph is incomplete.",
                documents=(make_document(),),
                claims=(unknown_document_claim,),
            )

    def test_future_document_is_rejected_at_historical_cutoff(self) -> None:
        future_document = SourceDocument(
            document_id="document-future",
            connector_id="openalex",
            external_id="W987654321",
            snapshot_id="snapshot-historical",
            title="Later confirmation",
            url="https://example.org/future",
            canonical_url="https://example.org/future",
            source_type=SourceType.SCIENTIFIC_PUBLICATION,
            language="en",
            trust_tier=TrustTier.A,
            origin_id="doi:10.0000/future",
            published_at=date(2026, 10, 1),
        )
        with self.assertRaisesRegex(ValueError, "after cutoff_date"):
            CandidateAssessment(
                candidate_id="candidate-historical",
                group_id="technology-family-historical",
                canonical_name="Historical candidate",
                aliases=(),
                query="industrial AI",
                analysis_scope_id="scope-industrial-ai-v1",
                cutoff_date=date(2026, 9, 15),
                status=CandidateStatus.WATCHLIST,
                features=make_features(),
                prediction=None,
                explanation="Only pre-cutoff documents may affect the assessment.",
                documents=(future_document,),
            )

    def test_future_observation_is_rejected_at_historical_cutoff(self) -> None:
        future_document = SourceDocument(
            document_id="document-future-observation",
            connector_id="gdelt",
            external_id="article-001",
            snapshot_id="snapshot-historical",
            title="Later media report",
            url="https://example.org/future-report",
            canonical_url="https://example.org/future-report",
            source_type=SourceType.OTHER,
            language="en",
            trust_tier=TrustTier.UNKNOWN,
            origin_id="url:https://example.org/future-report",
            observed_at=datetime(2026, 10, 1, tzinfo=UTC),
        )
        with self.assertRaisesRegex(ValueError, "observed after cutoff_date"):
            CandidateAssessment(
                candidate_id="candidate-historical",
                group_id="technology-family-historical",
                canonical_name="Historical candidate",
                aliases=(),
                query="industrial AI",
                analysis_scope_id="scope-industrial-ai-v1",
                cutoff_date=date(2026, 9, 15),
                status=CandidateStatus.WATCHLIST,
                features=make_features(),
                prediction=None,
                explanation="Only pre-cutoff observations may affect the assessment.",
                documents=(future_document,),
            )

    def test_excluded_candidate_requires_reason(self) -> None:
        with self.assertRaisesRegex(ValueError, "exclusion_reason"):
            CandidateAssessment(
                candidate_id="candidate-3",
                group_id="technology-family-3",
                canonical_name="Established technology",
                aliases=(),
                query="industrial AI",
                analysis_scope_id="scope-industrial-ai-v1",
                cutoff_date=date(2026, 9, 15),
                status=CandidateStatus.EXCLUDED,
                features=make_features(),
                prediction=make_prediction(),
                explanation="The market is mature.",
            )

    def test_invalid_echo_share_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            CandidateFeatures(
                stage=DevelopmentStage.RESEARCH,
                temporal=make_temporal(),
                query_relevance=0.5,
                independent_origin_count=1,
                source_type_diversity=1,
                independent_actor_count=1,
                echo_share=1.2,
            )

    def test_invalid_document_url_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "absolute HTTP"):
            SourceDocument(
                document_id="document-2",
                connector_id="test",
                external_id="document-2",
                snapshot_id="snapshot-test",
                title="Invalid source",
                url="example.org/source",
                canonical_url="https://example.org/source",
                source_type=SourceType.OTHER,
                language="en",
                trust_tier=TrustTier.UNKNOWN,
                origin_id="url:example",
            )

    def test_only_main_candidate_may_have_rank(self) -> None:
        with self.assertRaisesRegex(ValueError, "only main"):
            CandidateAssessment(
                candidate_id="candidate-4",
                group_id="technology-family-4",
                canonical_name="Mature technology",
                aliases=(),
                query="industrial AI",
                analysis_scope_id="scope-industrial-ai-v1",
                cutoff_date=date(2026, 9, 15),
                status=CandidateStatus.EXCLUDED,
                features=make_features(),
                prediction=make_prediction(),
                explanation="Mass adoption is confirmed.",
                exclusion_reason=ExclusionReason.MASS_ADOPTION,
                rank=2,
            )


if __name__ == "__main__":
    unittest.main()
