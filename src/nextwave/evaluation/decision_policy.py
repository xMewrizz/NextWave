"""Explicit post-model policy for main, watchlist and excluded statuses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from nextwave.contracts import CandidateStatus

DECISION_POLICY_VERSION = "decision-policy-v2"


class PolicyReason(StrEnum):
    GATE_REJECT = "gate_reject"
    GATE_REVIEW = "gate_review"
    DUPLICATE = "duplicate"
    NOT_SUBSTANTIVE = "not_substantive"
    EVIDENCE_REVIEW_INCOMPLETE = "evidence_review_incomplete"
    MATURE = "mature"
    MARKETING_HYPE = "marketing_hype"
    TEMPORAL_COVERAGE_INCOMPLETE = "temporal_coverage_incomplete"
    INSUFFICIENT_ORIGINS = "insufficient_origins"
    INSUFFICIENT_ACTORS = "insufficient_actors"
    INSUFFICIENT_TRUSTED_EVIDENCE = "insufficient_trusted_evidence"
    MODEL_UNAVAILABLE = "model_unavailable"
    MODEL_BELOW_THRESHOLD = "model_below_threshold"
    PASSED = "passed"


@dataclass(frozen=True, slots=True)
class DecisionPolicyInput:
    candidate_id: str
    gate_decision: str
    duplicate: bool
    substantive: bool
    evidence_review_complete: bool
    mature: bool
    marketing_hype: bool
    temporal_coverage_complete: bool
    independent_origin_count: int
    independent_actor_count: int
    grounded_ab_support: bool
    model_score: float | None
    decision_threshold: float | None

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must not be blank")
        if self.gate_decision not in {"accept", "reject", "review"}:
            raise ValueError("gate_decision must be accept, reject or review")
        for name in (
            "duplicate",
            "substantive",
            "evidence_review_complete",
            "mature",
            "marketing_hype",
            "temporal_coverage_complete",
            "grounded_ab_support",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        for name in ("independent_origin_count", "independent_actor_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("model_score", "decision_threshold"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be null or a number between 0 and 1")
        if (self.model_score is None) != (self.decision_threshold is None):
            raise ValueError("model_score and decision_threshold must be present together")


@dataclass(frozen=True, slots=True)
class DecisionPolicyResult:
    candidate_id: str
    status: CandidateStatus
    reason: PolicyReason
    policy_version: str = DECISION_POLICY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "status": self.status.value,
            "reason": self.reason.value,
        }


def apply_decision_policy(value: DecisionPolicyInput) -> DecisionPolicyResult:
    """Apply the frozen precedence documented in docs/REQUIREMENTS.md."""

    if value.gate_decision == "reject":
        return _result(value, CandidateStatus.EXCLUDED, PolicyReason.GATE_REJECT)
    if value.duplicate:
        return _result(value, CandidateStatus.EXCLUDED, PolicyReason.DUPLICATE)
    if not value.substantive:
        return _result(value, CandidateStatus.EXCLUDED, PolicyReason.NOT_SUBSTANTIVE)
    if value.gate_decision == "review":
        return _result(value, CandidateStatus.WATCHLIST, PolicyReason.GATE_REVIEW)
    if not value.evidence_review_complete:
        return _result(
            value,
            CandidateStatus.WATCHLIST,
            PolicyReason.EVIDENCE_REVIEW_INCOMPLETE,
        )
    if value.mature:
        return _result(value, CandidateStatus.EXCLUDED, PolicyReason.MATURE)
    if value.marketing_hype:
        return _result(value, CandidateStatus.EXCLUDED, PolicyReason.MARKETING_HYPE)
    if not value.temporal_coverage_complete:
        return _result(
            value, CandidateStatus.WATCHLIST, PolicyReason.TEMPORAL_COVERAGE_INCOMPLETE
        )
    if value.independent_origin_count < 2:
        return _result(
            value, CandidateStatus.WATCHLIST, PolicyReason.INSUFFICIENT_ORIGINS
        )
    if value.independent_actor_count < 2:
        return _result(
            value, CandidateStatus.WATCHLIST, PolicyReason.INSUFFICIENT_ACTORS
        )
    if not value.grounded_ab_support:
        return _result(
            value,
            CandidateStatus.WATCHLIST,
            PolicyReason.INSUFFICIENT_TRUSTED_EVIDENCE,
        )
    if value.model_score is None:
        return _result(value, CandidateStatus.WATCHLIST, PolicyReason.MODEL_UNAVAILABLE)
    if value.model_score < value.decision_threshold:
        return _result(
            value, CandidateStatus.EXCLUDED, PolicyReason.MODEL_BELOW_THRESHOLD
        )
    return _result(value, CandidateStatus.MAIN, PolicyReason.PASSED)


def _result(
    value: DecisionPolicyInput, status: CandidateStatus, reason: PolicyReason
) -> DecisionPolicyResult:
    return DecisionPolicyResult(
        candidate_id=value.candidate_id,
        status=status,
        reason=reason,
    )
