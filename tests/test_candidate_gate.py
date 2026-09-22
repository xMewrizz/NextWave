from __future__ import annotations

import json
import unittest

from nextwave.contracts import SourceDocument, SourceType, TrustTier
from nextwave.discovery import (
    CandidateMentionKind,
    GateDecision,
    GateIssueCode,
    LlmProvider,
    LlmSelection,
    ScopeGranularity,
    StructuredCandidateGate,
    build_analysis_scope,
    build_candidate_mention,
    build_candidate_proposals,
)


def fixture(*names: str):
    scope = build_analysis_scope(
        raw_query="Технологии в ИИ",
        normalized_query="artificial intelligence",
        search_texts=("AI",),
        languages=("en", "ru"),
        granularity=ScopeGranularity.DIRECTION,
        subfield_ids=("1702",),
    )
    documents = tuple(
        SourceDocument(
            document_id=f"document-{index}",
            connector_id="openalex",
            external_id=f"W{index}",
            snapshot_id="snapshot-ai-001",
            title=f"{name} for AI systems",
            url=f"https://example.org/work-{index}",
            canonical_url=f"https://example.org/work-{index}",
            source_type=SourceType.SCIENTIFIC_PUBLICATION,
            language="en",
            trust_tier=TrustTier.A,
            origin_id=f"doi:10.1234/work-{index}",
        )
        for index, name in enumerate(names, 1)
    )
    mentions = tuple(
        build_candidate_mention(
            document,
            text=name,
            kind=CandidateMentionKind.TITLE,
            locator=f"title[0:{len(name)}]",
            extractor_id="test-extractor",
        )
        for document, name in zip(documents, names, strict=True)
    )
    return scope, documents, build_candidate_proposals(scope, documents, mentions)


def gate(generate):
    return StructuredCandidateGate(
        generate,
        selection=LlmSelection(LlmProvider.OPENAI, "gpt-4.1"),
    )


def decision(proposal, status, reason, *, basis=None, explanation="Grounded in the source title."):
    return {
        "proposal_id": proposal["proposal_id"],
        "decision": status,
        "reason": reason,
        "basis_document_ids": (
            [proposal["documents"][0]["document_id"]] if basis is None else basis
        ),
        "explanation": explanation,
    }


class CandidateGateTests(unittest.TestCase):
    def test_accepts_concrete_technology_and_rejects_generic_area(self) -> None:
        scope, documents, batch = fixture("Speculative decoding", "Data science")

        def generate(prompt):
            payload = json.loads(prompt.split("Input data as JSON:\n", 1)[1])
            self.assertEqual(len(payload["proposals"]), 2)
            self.assertEqual(
                {proposal["proposal_id"] for proposal in payload["proposals"]},
                {"p1", "p2"},
            )
            self.assertEqual(payload["scope"]["query"], "Технологии в ИИ")
            return json.dumps({
                "decisions": [
                    decision(
                        proposal,
                        "accept" if proposal["name"] == "Speculative decoding" else "reject",
                        "concrete_technology"
                        if proposal["name"] == "Speculative decoding"
                        else "generic_area",
                    )
                    for proposal in payload["proposals"]
                ]
            })

        result = gate(generate).evaluate(scope, batch, documents)
        self.assertEqual(len(result.decisions), 2)
        self.assertEqual(len(result.accepted_proposal_ids), 1)
        self.assertEqual(
            result.accepted_proposal_ids,
            (next(
                item.proposal_id
                for item in batch.proposals
                if item.canonical_name == "Speculative decoding"
            ),),
        )
        self.assertEqual(result.issues, ())
        self.assertEqual(
            {item.decision for item in result.decisions},
            {GateDecision.ACCEPT, GateDecision.REJECT},
        )
        json.dumps(result.to_dict(), ensure_ascii=False)

    def test_missing_and_duplicate_decisions_require_review(self) -> None:
        scope, documents, batch = fixture("Speculative decoding", "Photonic inference")

        def generate(prompt):
            proposals = json.loads(prompt.split("Input data as JSON:\n", 1)[1])["proposals"]
            repeated = decision(proposals[0], "accept", "concrete_technology")
            return json.dumps({"decisions": [repeated, repeated]})

        result = gate(generate).evaluate(scope, batch, documents)
        self.assertEqual(result.accepted_proposal_ids, ())
        self.assertTrue(all(item.decision is GateDecision.REVIEW for item in result.decisions))
        self.assertEqual(
            {issue.code for issue in result.issues},
            {GateIssueCode.DUPLICATE_PROPOSAL, GateIssueCode.MISSING_PROPOSAL},
        )

    def test_unknown_basis_and_mismatched_reason_cannot_pass(self) -> None:
        scope, documents, batch = fixture("Speculative decoding", "Photonic inference")

        def generate(prompt):
            proposals = json.loads(prompt.split("Input data as JSON:\n", 1)[1])["proposals"]
            return json.dumps({"decisions": [
                decision(proposals[0], "accept", "concrete_technology", basis=["invented"]),
                decision(proposals[1], "accept", "generic_area"),
            ]})

        result = gate(generate).evaluate(scope, batch, documents)
        self.assertEqual(result.accepted_proposal_ids, ())
        self.assertEqual(
            [issue.code for issue in result.issues],
            [GateIssueCode.INVALID_DECISION, GateIssueCode.INVALID_DECISION],
        )

    def test_model_failure_moves_proposals_to_review(self) -> None:
        scope, documents, batch = fixture("Speculative decoding")

        def generate(_prompt):
            raise RuntimeError("provider unavailable")

        result = gate(generate).evaluate(scope, batch, documents)
        self.assertEqual(result.decisions[0].decision, GateDecision.REVIEW)
        self.assertEqual(result.issues[0].code, GateIssueCode.MODEL_ERROR)
        self.assertEqual(result.accepted_proposal_ids, ())

    def test_malformed_response_cannot_accept_proposals(self) -> None:
        scope, documents, batch = fixture("Speculative decoding")
        result = gate(lambda _prompt: '{"decisions":[],"unapproved":true}').evaluate(
            scope, batch, documents
        )
        self.assertEqual(result.decisions[0].decision, GateDecision.REVIEW)
        self.assertEqual(result.issues[0].code, GateIssueCode.INVALID_RESPONSE)

    def test_seven_proposals_are_processed_in_two_bounded_batches(self) -> None:
        scope, documents, batch = fixture(*(f"Technology {index}" for index in range(7)))
        batch_sizes = []

        def generate(prompt):
            proposals = json.loads(prompt.split("Input data as JSON:\n", 1)[1])["proposals"]
            batch_sizes.append(len(proposals))
            return json.dumps({"decisions": [
                decision(proposal, "accept", "concrete_technology")
                for proposal in proposals
            ]})

        result = gate(generate).evaluate(scope, batch, documents)
        self.assertEqual(sorted(batch_sizes), [1, 6])
        self.assertEqual(result.batch_count, 2)
        self.assertEqual(len(result.accepted_proposal_ids), 7)

    def test_empty_proposal_batch_never_calls_model(self) -> None:
        scope, documents, _ = fixture("Speculative decoding")
        empty = build_candidate_proposals(scope, documents, ())

        def generate(_prompt):
            self.fail("model should not be called")

        result = gate(generate).evaluate(scope, empty, documents)
        self.assertEqual(result.batch_count, 0)
        self.assertEqual(result.decisions, ())


if __name__ == "__main__":
    unittest.main()
