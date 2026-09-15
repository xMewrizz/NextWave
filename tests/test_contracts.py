from __future__ import annotations

import unittest
from datetime import date

from nextwave.contracts import (
    CandidateAssessment,
    CandidateFeatures,
    CandidateStatus,
    DevelopmentStage,
    Evidence,
    ExclusionReason,
    ModelPrediction,
    SourceType,
    TrustLevel,
)


def make_features() -> CandidateFeatures:
    return CandidateFeatures(
        stage=DevelopmentStage.PILOT,
        independent_source_count=2,
        source_type_diversity=2,
        independent_actor_count=2,
        promotional_source_share=0.0,
    )


def make_prediction() -> ModelPrediction:
    return ModelPrediction(
        weak_signal_score=0.78,
        model_version="baseline-1",
        feature_version="candidate-features-1",
        calibrated=False,
    )


def make_evidence() -> Evidence:
    return Evidence(
        evidence_id="source-1",
        title="Prototype evaluation",
        url="https://example.org/prototype",
        source_type=SourceType.SCIENTIFIC_PUBLICATION,
        language="en",
        trust_level=TrustLevel.HIGH,
        published_at=date(2026, 8, 1),
    )


class ContractTests(unittest.TestCase):
    def test_main_candidate_requires_and_accepts_evidence(self) -> None:
        assessment = CandidateAssessment(
            candidate_id="candidate-1",
            canonical_name="Example technology",
            query="industrial AI",
            status=CandidateStatus.MAIN,
            features=make_features(),
            prediction=make_prediction(),
            explanation="Early pilot with independent evidence.",
            evidence=(make_evidence(),),
            priority_score=72.0,
        )

        self.assertEqual(assessment.status, CandidateStatus.MAIN)
        self.assertEqual(len(assessment.evidence), 1)

    def test_excluded_candidate_requires_reason(self) -> None:
        with self.assertRaisesRegex(ValueError, "exclusion_reason"):
            CandidateAssessment(
                candidate_id="candidate-2",
                canonical_name="Established technology",
                query="industrial AI",
                status=CandidateStatus.EXCLUDED,
                features=make_features(),
                prediction=make_prediction(),
                explanation="The market is already mature.",
            )

    def test_non_excluded_candidate_rejects_exclusion_reason(self) -> None:
        with self.assertRaisesRegex(ValueError, "only excluded"):
            CandidateAssessment(
                candidate_id="candidate-3",
                canonical_name="Uncertain technology",
                query="industrial AI",
                status=CandidateStatus.WATCHLIST,
                features=make_features(),
                prediction=make_prediction(),
                explanation="Evidence is incomplete.",
                exclusion_reason=ExclusionReason.INSUFFICIENT_TRUST,
            )

    def test_invalid_promotional_share_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            CandidateFeatures(
                stage=DevelopmentStage.RESEARCH,
                independent_source_count=1,
                source_type_diversity=1,
                independent_actor_count=1,
                promotional_source_share=1.2,
            )

    def test_invalid_evidence_url_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "absolute HTTP"):
            Evidence(
                evidence_id="source-2",
                title="Invalid source",
                url="example.org/source",
                source_type=SourceType.OTHER,
                language="en",
                trust_level=TrustLevel.UNKNOWN,
            )


if __name__ == "__main__":
    unittest.main()
