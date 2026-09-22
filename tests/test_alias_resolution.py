from __future__ import annotations

import json
import unittest

from nextwave.discovery import (
    AliasSuggestionReason,
    CandidateGateDecision,
    CandidateGateResult,
    CandidateProposal,
    CandidateProposalBatch,
    GateDecision,
    GateReason,
    resolve_candidate_aliases,
)


def proposal(
    number: int,
    name: str,
    *,
    origin_id: str | None = None,
    aliases: tuple[str, ...] = (),
):
    return CandidateProposal(
        proposal_id=f"proposal-{number}",
        analysis_scope_id="scope-test",
        canonical_name=name,
        normalized_name=name.casefold(),
        aliases=aliases,
        mention_ids=(f"mention-{number}",),
        source_kinds=(),
        connector_ids=("openalex",),
        provider_term_ids=(),
        document_ids=(f"document-{number}",),
        origin_ids=(origin_id or f"origin-{number}",),
        max_provider_score=None,
        primary_provider_topic=False,
    )


def inputs(*items, rejected: tuple[str, ...] = ()):
    batch = CandidateProposalBatch("scope-test", tuple(items), ())
    decisions = tuple(
        CandidateGateDecision(
            proposal_id=item.proposal_id,
            decision=(
                GateDecision.REJECT if item.proposal_id in rejected else GateDecision.ACCEPT
            ),
            reason=(
                GateReason.IRRELEVANT
                if item.proposal_id in rejected
                else GateReason.CONCRETE_TECHNOLOGY
            ),
            basis_document_ids=(item.document_ids[0],),
            explanation="The source identifies a specific technology.",
        )
        for item in items
    )
    gate = CandidateGateResult(
        "scope-test", "test-gate", tuple(item.proposal_id for item in items), decisions, (), 1
    )
    return batch, gate


class AliasResolutionTests(unittest.TestCase):
    def test_preserves_pre_gate_group_and_its_aliases(self) -> None:
        batch, gate = inputs(proposal(
            1, "Speculative decoding", aliases=("Speculative-decoding",)
        ))
        result = resolve_candidate_aliases(batch, gate)

        self.assertEqual(len(result.groups), 1)
        group = result.groups[0]
        self.assertEqual(group.normalization_key, "speculative decoding")
        self.assertEqual(group.proposal_ids, ("proposal-1",))
        self.assertEqual(group.origin_ids, ("origin-1",))
        self.assertEqual({group.canonical_name, *group.aliases}, {
            "Speculative decoding", "Speculative-decoding"
        })
        self.assertEqual(result.review_suggestions, ())
        json.dumps(result.to_dict(), ensure_ascii=False)

    def test_shared_origin_acronym_is_suggested_not_merged(self) -> None:
        batch, gate = inputs(
            proposal(1, "Large Language Model", origin_id="doi:shared"),
            proposal(2, "LLM", origin_id="doi:shared"),
        )
        result = resolve_candidate_aliases(batch, gate)

        self.assertEqual(len(result.groups), 2)
        self.assertEqual(len(result.review_suggestions), 1)
        self.assertEqual(
            result.review_suggestions[0].reason,
            AliasSuggestionReason.SHARED_ORIGIN_ACRONYM,
        )
        self.assertEqual(
            result.review_suggestions[0].shared_origin_ids, ("doi:shared",)
        )

    def test_unrelated_origin_acronym_is_not_suggested(self) -> None:
        batch, gate = inputs(
            proposal(1, "Large Language Model"),
            proposal(2, "LLM"),
        )
        result = resolve_candidate_aliases(batch, gate)
        self.assertEqual(len(result.groups), 2)
        self.assertEqual(result.review_suggestions, ())

    def test_rejected_proposal_remains_outside_alias_groups(self) -> None:
        batch, gate = inputs(
            proposal(1, "Speculative-decoding"),
            proposal(2, "speculative decoding"),
            rejected=("proposal-2",),
        )
        result = resolve_candidate_aliases(batch, gate)
        self.assertEqual(result.input_proposal_ids, ("proposal-1",))
        self.assertEqual(result.groups[0].proposal_ids, ("proposal-1",))

    def test_duplicate_spellings_must_be_grouped_before_gate(self) -> None:
        batch, gate = inputs(
            proposal(1, "Speculative-decoding"),
            proposal(2, "speculative decoding"),
        )
        with self.assertRaisesRegex(ValueError, "before candidate gate"):
            resolve_candidate_aliases(batch, gate)

    def test_gate_for_different_proposals_is_rejected(self) -> None:
        batch, gate = inputs(proposal(1, "Speculative decoding"))
        other = CandidateGateResult(
            gate.analysis_scope_id, gate.gate_id, ("proposal-other",), gate.decisions, (), 1
        )
        with self.assertRaisesRegex(ValueError, "same order"):
            resolve_candidate_aliases(batch, other)


if __name__ == "__main__":
    unittest.main()
