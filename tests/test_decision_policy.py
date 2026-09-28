from __future__ import annotations

import unittest
from dataclasses import replace

from nextwave.contracts import CandidateStatus
from nextwave.evaluation.decision_policy import (
    DECISION_POLICY_VERSION,
    DecisionPolicyInput,
    PolicyReason,
    apply_decision_policy,
)


def eligible() -> DecisionPolicyInput:
    return DecisionPolicyInput(
        candidate_id="candidate-001",
        gate_decision="accept",
        duplicate=False,
        substantive=True,
        mature=False,
        marketing_hype=False,
        temporal_coverage_complete=True,
        independent_origin_count=2,
        independent_actor_count=2,
        grounded_ab_support=True,
        model_score=0.8,
        decision_threshold=0.6,
    )


class DecisionPolicyTests(unittest.TestCase):
    def test_eligible_candidate_is_main(self) -> None:
        result = apply_decision_policy(eligible())
        self.assertEqual(result.status, CandidateStatus.MAIN)
        self.assertEqual(result.reason, PolicyReason.PASSED)
        self.assertEqual(result.policy_version, DECISION_POLICY_VERSION)

    def test_gate_reject_has_highest_priority(self) -> None:
        value = replace(eligible(), gate_decision="reject", mature=True, marketing_hype=True)
        result = apply_decision_policy(value)
        self.assertEqual(result.status, CandidateStatus.EXCLUDED)
        self.assertEqual(result.reason, PolicyReason.GATE_REJECT)

    def test_duplicate_is_excluded_before_model(self) -> None:
        result = apply_decision_policy(replace(eligible(), duplicate=True))
        self.assertEqual(result.reason, PolicyReason.DUPLICATE)
        self.assertEqual(result.status, CandidateStatus.EXCLUDED)

    def test_gate_review_is_watchlist(self) -> None:
        result = apply_decision_policy(replace(eligible(), gate_decision="review"))
        self.assertEqual(result.reason, PolicyReason.GATE_REVIEW)
        self.assertEqual(result.status, CandidateStatus.WATCHLIST)

    def test_maturity_wins_over_growth_score(self) -> None:
        result = apply_decision_policy(replace(eligible(), mature=True, model_score=1.0))
        self.assertEqual(result.reason, PolicyReason.MATURE)
        self.assertEqual(result.status, CandidateStatus.EXCLUDED)

    def test_marketing_hype_is_excluded(self) -> None:
        result = apply_decision_policy(replace(eligible(), marketing_hype=True))
        self.assertEqual(result.reason, PolicyReason.MARKETING_HYPE)
        self.assertEqual(result.status, CandidateStatus.EXCLUDED)

    def test_incomplete_evidence_conditions_are_watchlist(self) -> None:
        cases = (
            (
                replace(eligible(), temporal_coverage_complete=False),
                PolicyReason.TEMPORAL_COVERAGE_INCOMPLETE,
            ),
            (replace(eligible(), independent_origin_count=1), PolicyReason.INSUFFICIENT_ORIGINS),
            (replace(eligible(), independent_actor_count=1), PolicyReason.INSUFFICIENT_ACTORS),
            (
                replace(eligible(), grounded_ab_support=False),
                PolicyReason.INSUFFICIENT_TRUSTED_EVIDENCE,
            ),
            (
                replace(eligible(), model_score=None, decision_threshold=None),
                PolicyReason.MODEL_UNAVAILABLE,
            ),
        )
        for value, reason in cases:
            with self.subTest(reason=reason):
                result = apply_decision_policy(value)
                self.assertEqual(result.status, CandidateStatus.WATCHLIST)
                self.assertEqual(result.reason, reason)

    def test_below_threshold_is_excluded(self) -> None:
        result = apply_decision_policy(replace(eligible(), model_score=0.59))
        self.assertEqual(result.status, CandidateStatus.EXCLUDED)
        self.assertEqual(result.reason, PolicyReason.MODEL_BELOW_THRESHOLD)

    def test_rejects_half_present_prediction(self) -> None:
        with self.assertRaisesRegex(ValueError, "present together"):
            replace(eligible(), model_score=None)

    def test_rejects_non_boolean_policy_flag(self) -> None:
        with self.assertRaisesRegex(ValueError, "mature must be boolean"):
            replace(eligible(), mature="false")


if __name__ == "__main__":
    unittest.main()
