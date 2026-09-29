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


def gate(generate, *, version="candidate-gate-v5"):
    return StructuredCandidateGate(
        generate,
        selection=LlmSelection(LlmProvider.YANDEX, "YandexGPT Lite 5"),
        version=version,
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
    def test_product_gate_audits_initial_accepts_and_preserves_rejects(self) -> None:
        scope, documents, batch = fixture("AI accelerator", "AMD", "Data science")
        calls: list[str] = []

        def generate(prompt):
            proposals = json.loads(prompt.split("Input data as JSON:\n", 1)[1])[
                "proposals"
            ]
            if "strict second-pass critic" in prompt:
                calls.append("audit")
                self.assertIn("commercial product, model number, hardware SKU", prompt)
                self.assertIn("phrase copied from one paper", prompt)
                self.assertIn("reason must agree with the decision exactly", prompt)
                self.assertEqual(
                    {item["name"] for item in proposals}, {"AI accelerator", "AMD"}
                )
                return json.dumps(
                    {
                        "decisions": [
                            decision(
                                proposal,
                                "accept" if proposal["name"] == "AI accelerator" else "reject",
                                "technical_mechanism"
                                if proposal["name"] == "AI accelerator"
                                else "organization",
                            )
                            for proposal in proposals
                        ]
                    }
                )
            calls.append("initial")
            return json.dumps(
                {
                    "decisions": [
                        decision(
                            proposal,
                            "reject" if proposal["name"] == "Data science" else "accept",
                            "generic_area"
                            if proposal["name"] == "Data science"
                            else "technical_application",
                        )
                        for proposal in proposals
                    ]
                }
            )

        result = gate(generate, version="candidate-gate-v6").evaluate(
            scope,
            batch,
            documents,
            progress=lambda phase, done, total: calls.append(
                f"{phase}:{done}/{total}"
            ),
        )

        self.assertEqual(
            calls,
            ["initial", "primary:3/3", "audit", "audit:2/2"],
        )
        self.assertEqual(
            {
                proposal.canonical_name: decision.decision
                for proposal, decision in zip(batch.proposals, result.decisions, strict=True)
            },
            {
                "AI accelerator": GateDecision.ACCEPT,
                "AMD": GateDecision.REJECT,
                "Data science": GateDecision.REJECT,
            },
        )
        self.assertEqual(result.batch_count, 2)

    def test_product_gate_audit_failure_cannot_leave_initial_accept(self) -> None:
        scope, documents, batch = fixture("AI accelerator")
        calls = 0

        def generate(prompt):
            nonlocal calls
            calls += 1
            if "strict second-pass critic" in prompt:
                return '{"decisions":[]}'
            proposal = json.loads(prompt.split("Input data as JSON:\n", 1)[1])[
                "proposals"
            ][0]
            return json.dumps(
                {"decisions": [decision(proposal, "accept", "technical_mechanism")]}
            )

        result = gate(generate, version="candidate-gate-v6").evaluate(
            scope, batch, documents
        )

        self.assertEqual(calls, 4)
        self.assertEqual(result.decisions[0].decision, GateDecision.REVIEW)
        self.assertEqual(result.accepted_proposal_ids, ())
        self.assertEqual(result.issues[0].code, GateIssueCode.MISSING_PROPOSAL)
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
            self.assertEqual(payload["scope"]["granularity"], "direction")
            self.assertEqual(
                payload["scope"]["search_texts"],
                ["artificial intelligence", "AI"],
            )
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

    def test_prompt_requires_specificity_and_direct_scope_relation(self) -> None:
        scope, documents, batch = fixture("Post-Training Quantization")

        def generate(prompt):
            self.assertIn("silently apply these checks", prompt)
            self.assertIn("direct functional role", prompt)
            self.assertIn("The same name may be accepted for one query", prompt)
            self.assertIn("tactile sensor", prompt)
            self.assertIn("fraud-detection method", prompt)
            self.assertIn("does not become an Edge technology", prompt)
            self.assertIn("Do not output analysis", prompt)
            proposal = json.loads(prompt.split("Input data as JSON:\n", 1)[1])["proposals"][0]
            return json.dumps({
                "decisions": [
                    decision(proposal, "accept", "technical_mechanism")
                ]
            })

        result = gate(generate).evaluate(scope, batch, documents)

        self.assertEqual(result.decisions[0].decision, GateDecision.ACCEPT)

    def test_context_excerpt_is_centered_on_late_candidate_mention(self) -> None:
        scope = build_analysis_scope(
            raw_query="Инфраструктура ИИ",
            normalized_query="AI infrastructure",
            search_texts=("AI infrastructure",),
            languages=("en",),
            granularity=ScopeGranularity.DIRECTION,
        )
        excerpt = "Unrelated introduction. " * 60 + (
            "AIOps-driven orchestration operates AI data-center workloads."
        )
        document = SourceDocument(
            document_id="document-aiops",
            connector_id="openalex",
            external_id="W-aiops",
            snapshot_id="snapshot-ai-001",
            title="Data centers in the age of AI",
            url="https://example.org/aiops",
            canonical_url="https://example.org/aiops",
            source_type=SourceType.SCIENTIFIC_PUBLICATION,
            language="en",
            trust_tier=TrustTier.A,
            origin_id="doi:10.1234/aiops",
            excerpt=excerpt,
        )
        mention = build_candidate_mention(
            document,
            text="AIOps-driven orchestration",
            kind=CandidateMentionKind.EXCERPT,
            locator="excerpt[1440:1466]",
            extractor_id="test-extractor",
        )
        batch = build_candidate_proposals(scope, (document,), (mention,))

        def generate(prompt):
            proposal = json.loads(prompt.split("Input data as JSON:\n", 1)[1])["proposals"][0]
            context = proposal["documents"][0]["excerpt"]
            self.assertLessEqual(len(context), 700)
            self.assertIn("AIOps-driven orchestration", context)
            return json.dumps({
                "decisions": [
                    decision(proposal, "accept", "technical_mechanism")
                ]
            })

        result = gate(generate).evaluate(scope, batch, (document,))

        self.assertEqual(result.decisions[0].decision, GateDecision.ACCEPT)

    def test_context_prefers_document_grounded_in_user_scope(self) -> None:
        scope = build_analysis_scope(
            raw_query="Технологии цифровых платежей",
            normalized_query="digital payment technologies",
            search_texts=("digital payments",),
            languages=("en",),
            granularity=ScopeGranularity.DIRECTION,
        )
        documents = tuple(
            SourceDocument(
                document_id=f"document-{index}",
                connector_id="openalex",
                external_id=f"W{index}",
                snapshot_id="snapshot-payments",
                title=title,
                url=f"https://example.org/{index}",
                canonical_url=f"https://example.org/{index}",
                source_type=SourceType.SCIENTIFIC_PUBLICATION,
                language="en",
                trust_tier=TrustTier.A,
                origin_id=f"doi:10.1234/{index}",
            )
            for index, title in enumerate(
                (
                    "Smart contracts in general software engineering",
                    "Smart contracts for secure digital payments",
                ),
                1,
            )
        )
        mentions = tuple(
            build_candidate_mention(
                document,
                text="Smart contracts",
                kind=CandidateMentionKind.TITLE,
                locator="title[0:15]",
                extractor_id="test-extractor",
            )
            for document in documents
        )
        batch = build_candidate_proposals(scope, documents, mentions)

        def generate(prompt):
            proposal = json.loads(prompt.split("Input data as JSON:\n", 1)[1])["proposals"][0]
            self.assertEqual(
                proposal["documents"][0]["title"],
                "Smart contracts for secure digital payments",
            )
            return json.dumps(
                {"decisions": [decision(proposal, "accept", "technical_mechanism")]}
            )

        result = gate(generate).evaluate(scope, batch, documents)
        self.assertEqual(result.decisions[0].decision, GateDecision.ACCEPT)

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
        self.assertEqual(
            result.issues[0].message, "candidate gate model request failed"
        )
        self.assertNotIn("provider unavailable", result.issues[0].message)
        self.assertEqual(result.accepted_proposal_ids, ())

    def test_malformed_response_cannot_accept_proposals(self) -> None:
        scope, documents, batch = fixture("Speculative decoding")
        result = gate(lambda _prompt: '{"decisions":[],"unapproved":true}').evaluate(
            scope, batch, documents
        )
        self.assertEqual(result.decisions[0].decision, GateDecision.REVIEW)
        self.assertEqual(result.issues[0].code, GateIssueCode.INVALID_RESPONSE)

    def test_invalid_first_response_is_retried_once(self) -> None:
        scope, documents, batch = fixture("Speculative decoding")
        calls = 0

        def generate(prompt):
            nonlocal calls
            calls += 1
            if calls == 1:
                return '{"decisions":[],"unapproved":true}'
            proposal = json.loads(prompt.split("Input data as JSON:\n", 1)[1])["proposals"][0]
            return json.dumps(
                {"decisions": [decision(proposal, "accept", "technical_mechanism")]}
            )

        result = gate(generate).evaluate(scope, batch, documents)
        self.assertEqual(calls, 2)
        self.assertEqual(result.issues, ())
        self.assertEqual(result.decisions[0].decision, GateDecision.ACCEPT)

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
