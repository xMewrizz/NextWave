from __future__ import annotations

import unittest
from datetime import date

from nextwave.labeling import (
    LABELING_CUTOFF_DATE,
    EvidenceDirection,
    EvidenceKind,
    GateLabelDecision,
    LabelEvidence,
    ModelLabelDecision,
    NegativeCandidateRecord,
    NegativeClass,
    NoiseType,
    ReviewRound,
    ReviewStatus,
    SearchCoverage,
    SearchSourceClass,
    SourceType,
    TrustLevel,
)


def make_evidence(
    number: int,
    *,
    kind: EvidenceKind,
    published_at: date = date(2026, 8, 1),
    trust_level: TrustLevel = TrustLevel.B,
    direction: EvidenceDirection = EvidenceDirection.SUPPORT,
    origin_id: str | None = None,
) -> LabelEvidence:
    return LabelEvidence(
        evidence_id=f"evidence-{number:03d}",
        direction=direction,
        kind=kind,
        source_type=SourceType.INDUSTRY_MEDIA,
        trust_level=trust_level,
        title=f"Материал {number}",
        url=f"https://example.org/{number}",
        published_at=published_at,
        organization="Example Organization",
        origin_id=origin_id or f"origin-{number:03d}",
        claim="Проверяемое утверждение",
        locator="раздел 1",
    )


def complete_coverage() -> SearchCoverage:
    return SearchCoverage(
        queries=("example technology", "example technology pilot"),
        source_classes=(
            SearchSourceClass.SCIENTIFIC,
            SearchSourceClass.INDUSTRY,
        ),
        searched_at=date(2026, 9, 19),
        notes="Проверены научный и отраслевой контуры.",
    )


def coverage_with(*classes: SearchSourceClass) -> SearchCoverage:
    return SearchCoverage(
        queries=("example technology", "example technology pilot"),
        source_classes=classes,
        searched_at=date(2026, 9, 19),
        notes="Проверочное покрытие.",
    )


def hype_evidence() -> tuple:
    return (
        make_evidence(1, kind=EvidenceKind.PUBLICITY_WAVE, origin_id="origin-a"),
        make_evidence(2, kind=EvidenceKind.PUBLICITY_WAVE, origin_id="origin-a"),
        make_evidence(3, kind=EvidenceKind.PUBLICITY_WAVE, origin_id="origin-b"),
    )


def hype_decision(coverage: SearchCoverage) -> ModelLabelDecision:
    return ModelLabelDecision(
        decision_id="decision-hype-001",
        candidate_id="team-negative-001",
        review_round=ReviewRound.PRIMARY,
        status=ReviewStatus.REVIEWED,
        label=NegativeClass.MARKETING_HYPE,
        rationale="Публичная волна без технической опоры.",
        reviewer_id="reviewer-1",
        annotated_at=date(2026, 9, 19),
        cutoff_date=LABELING_CUTOFF_DATE,
        evidence=hype_evidence(),
        search_coverage=coverage,
    )


class LabelingContractTests(unittest.TestCase):
    def test_negative_candidate_has_identity_without_final_label(self) -> None:
        candidate = NegativeCandidateRecord(
            candidate_id="team-negative-001",
            canonical_name="Зрелая технология",
            aliases=(),
            group_id="team-negative-001",
            source_query="технологии в ИИ",
            domain="Инфраструктура ИИ",
            analysis_scope_key="ai-infrastructure-v1",
            cutoff_date=LABELING_CUTOFF_DATE,
        )

        body = candidate.to_dict()
        self.assertNotIn("label", body)
        self.assertNotIn("rationale", body)
        self.assertEqual(body["cutoff_date"], "2026-09-15")

    def test_reviewed_mature_requires_ab_maturity_evidence(self) -> None:
        low_trust = make_evidence(
            1,
            kind=EvidenceKind.ESTABLISHED_MARKET,
            trust_level=TrustLevel.D,
        )

        with self.assertRaisesRegex(ValueError, "A/B maturity evidence"):
            ModelLabelDecision(
                decision_id="decision-mature-001",
                candidate_id="team-negative-001",
                review_round=ReviewRound.PRIMARY,
                status=ReviewStatus.REVIEWED,
                label=NegativeClass.MATURE,
                rationale="Рынок сформирован.",
                reviewer_id="reviewer-1",
                annotated_at=date(2026, 9, 19),
                cutoff_date=LABELING_CUTOFF_DATE,
                evidence=(low_trust,),
            )

    def test_reviewed_mature_accepts_qualified_evidence(self) -> None:
        decision = ModelLabelDecision(
            decision_id="decision-mature-001",
            candidate_id="team-negative-001",
            review_round=ReviewRound.PRIMARY,
            status=ReviewStatus.REVIEWED,
            label=NegativeClass.MATURE,
            rationale="Подтверждено серийное внедрение.",
            reviewer_id="reviewer-1",
            annotated_at=date(2026, 9, 19),
            cutoff_date=LABELING_CUTOFF_DATE,
            evidence=(make_evidence(1, kind=EvidenceKind.SERIAL_DEPLOYMENT),),
        )

        self.assertEqual(decision.to_dict()["label"], "mature")

    def test_reviewed_hype_requires_wave_and_search_coverage(self) -> None:
        one_wave_item = make_evidence(1, kind=EvidenceKind.PUBLICITY_WAVE)

        with self.assertRaisesRegex(ValueError, "publicity wave"):
            ModelLabelDecision(
                decision_id="decision-hype-001",
                candidate_id="team-negative-001",
                review_round=ReviewRound.PRIMARY,
                status=ReviewStatus.REVIEWED,
                label=NegativeClass.MARKETING_HYPE,
                rationale="Публичная волна без технической опоры.",
                reviewer_id="reviewer-1",
                annotated_at=date(2026, 9, 19),
                cutoff_date=LABELING_CUTOFF_DATE,
                evidence=(one_wave_item,),
                search_coverage=complete_coverage(),
            )

    def test_reviewed_hype_accepts_scientific_and_industry(self) -> None:
        decision = hype_decision(complete_coverage())

        self.assertEqual(decision.to_dict()["label"], "marketing_hype")

    def test_reviewed_hype_rejects_missing_scientific(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "scientific and industry"
        ):
            hype_decision(
                coverage_with(
                    SearchSourceClass.INDUSTRY,
                    SearchSourceClass.OFFICIAL,
                )
            )

    def test_reviewed_hype_rejects_missing_industry(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "scientific and industry"
        ):
            hype_decision(
                coverage_with(
                    SearchSourceClass.SCIENTIFIC,
                    SearchSourceClass.OFFICIAL,
                )
            )

    def test_reviewed_hype_accepts_extra_official_class(self) -> None:
        decision = hype_decision(
            coverage_with(
                SearchSourceClass.SCIENTIFIC,
                SearchSourceClass.INDUSTRY,
                SearchSourceClass.OFFICIAL,
            )
        )

        self.assertEqual(decision.to_dict()["label"], "marketing_hype")

    def test_evidence_after_cutoff_is_rejected(self) -> None:
        future = make_evidence(
            1,
            kind=EvidenceKind.SERIAL_DEPLOYMENT,
            published_at=date(2026, 9, 16),
        )

        with self.assertRaisesRegex(ValueError, "after cutoff"):
            ModelLabelDecision(
                decision_id="decision-mature-001",
                candidate_id="team-negative-001",
                review_round=ReviewRound.PRIMARY,
                status=ReviewStatus.DRAFT,
                label=NegativeClass.MATURE,
                rationale="Черновик.",
                reviewer_id="reviewer-1",
                annotated_at=date(2026, 9, 19),
                cutoff_date=LABELING_CUTOFF_DATE,
                evidence=(future,),
            )

    def test_adjudicated_status_requires_adjudication_round(self) -> None:
        with self.assertRaisesRegex(ValueError, "only adjudication"):
            GateLabelDecision(
                decision_id="decision-noise-001",
                noise_id="noise-001",
                review_round=ReviewRound.PRIMARY,
                status=ReviewStatus.ADJUDICATED,
                noise_type=NoiseType.BROAD_CONCEPT,
                rationale="Понятие не задаёт технический механизм.",
                reviewer_id="reviewer-1",
                annotated_at=date(2026, 9, 19),
            )


if __name__ == "__main__":
    unittest.main()
