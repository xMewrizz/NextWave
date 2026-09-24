"""Tests for the labeling review-queue builder (hand-built fixtures, no network)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from nextwave.discovery.run_store import DiscoveryRun
from nextwave.labeling.contracts import NoiseType
from nextwave.labeling.queue import (
    CandidateSlot,
    NoiseSlot,
    build_labeling_queue,
    queue_to_jsonl,
)

CUTOFF = "2026-09-15"


def document(
    document_id: str,
    *,
    url: str = "https://example.org/paper",
    published_at: str | None = "2026-01-01",
    trust_tier: str = "A",
) -> dict:
    return {
        "document_id": document_id,
        "title": f"Study of {document_id}",
        "url": url,
        "published_at": published_at,
        "organizations": [],
        "source_type": "scientific_publication",
        "trust_tier": trust_tier,
        "origin_id": f"origin-{document_id}",
    }


def group(
    group_id: str,
    canonical_name: str,
    *,
    origins: int = 1,
    documents: int = 1,
) -> dict:
    return {
        "group_id": group_id,
        "canonical_name": canonical_name,
        "aliases": [],
        "origin_ids": [f"origin-{group_id}-{index}" for index in range(origins)],
        "document_ids": [f"document-{group_id}-{index}" for index in range(documents)],
        "normalization_key": canonical_name.casefold(),
        "proposal_ids": [f"proposal-{group_id}"],
    }


def make_run(
    run_id: str,
    *,
    groups: list | None = None,
    proposals: list | None = None,
    decisions: list | None = None,
    exclusions: list | None = None,
    issues: list | None = None,
    suggestions: list | None = None,
    evidence: list | None = None,
    documents: list | None = None,
    raw_query: str = "Технологии в ИИ",
    cutoff: str = CUTOFF,
    domain: str = "Финтех",
    scientific_status: str = "complete",
    scientific_usage: dict | None = None,
    media_status: str = "failed",
    media_provider: str | None = None,
    media_usage: list | dict | None = None,
) -> DiscoveryRun:
    plan = {
        "plan_id": f"plan-{run_id}",
        "analysis_id": f"analysis-{run_id}",
        "scope": {"scope_id": f"scope-{run_id}", "raw_query": raw_query},
    }
    result = {
        "alias_resolution": {
            "groups": groups or [],
            "review_suggestions": suggestions or [],
        },
        "candidate_proposals": {
            "proposals": proposals or [],
            "exclusions": exclusions or [],
        },
        "candidate_gate": {"gate_id": "gate-1", "decisions": decisions or []},
        "text_extraction": {"issues": issues or []},
        "evidence_extraction": {"proposals": evidence or []},
        "scientific": {"status": scientific_status, "documents": documents or []}
        | ({"usage": scientific_usage} if scientific_usage is not None else {}),
        "media": {
            "status": media_status,
            "provider_used": media_provider,
            "documents": [],
        }
        | ({"usage": media_usage} if media_usage is not None else {}),
        "verification": {"results": []},
    }
    manifest = {"run_id": run_id, "cutoff_date": cutoff, "domain": domain}
    return DiscoveryRun(
        run_id=run_id, run_dir=Path(run_id), plan=plan, result=result, manifest=manifest
    )


def evidence_proposal(
    group_id: str,
    number: int,
    *,
    direction: str = "support",
    published_at: str | None = "2026-01-01",
    trust_tier: str = "A",
) -> tuple[dict, dict]:
    document_id = f"document-{group_id}-{number}"
    return (
        {
            "alias_group_id": group_id,
            "origin_id": f"origin-{group_id}-{number}",
            "source_url": f"https://example.org/{document_id}",
            "review_status": "pending",
            "claim": {
                "claim_id": f"claim-{group_id}-{number}",
                "document_id": document_id,
                "claim_type": "research",
                "direction": direction,
                "text": f"Finding {number} supports the early signal.",
                "locator": "excerpt[0:10]",
                "organization": None,
                "extraction_confidence": None,
            },
        },
        document(
            document_id, published_at=published_at, trust_tier=trust_tier,
            url=f"https://example.org/{document_id}",
        ),
    )


class QueueCandidateTests(unittest.TestCase):
    def test_merges_dedups_orders_and_fills_slots(self) -> None:
        run1 = make_run(
            "run-1",
            groups=[
                group("group-b", "Beta Tech", origins=2),
                group("group-a", "Alpha Tech", origins=1),
            ],
        )
        run2 = make_run(
            "run-2",
            groups=[
                group("group-a2", "Alpha Tech", origins=5),
                group("group-c", "Gamma Tech", origins=1),
            ],
        )
        slots = (
            CandidateSlot("team-negative-001", "Финтех"),
            CandidateSlot("team-negative-002", "Финтех"),
        )

        queue = build_labeling_queue((run1, run2), candidate_slots=slots)

        # Beta first (more origins); Alpha from run-1 wins over run-2 duplicate.
        self.assertEqual(
            [item.canonical_name for item in queue.candidates],
            ["Beta Tech", "Alpha Tech"],
        )
        self.assertEqual(queue.candidates[1].run_id, "run-1")
        self.assertEqual(
            [item.candidate_id for item in queue.candidates],
            ["team-negative-001", "team-negative-002"],
        )
        self.assertTrue(all(item.analysis_scope_key == "fintech-v1" for item in queue.candidates))

    def test_shortage_becomes_deficit_and_surplus_becomes_overflow(self) -> None:
        run = make_run("run-1", groups=[group("group-a", "Alpha Tech")])
        slots = (
            CandidateSlot("team-negative-001", "Финтех"),
            CandidateSlot("team-negative-002", "Финтех"),
            CandidateSlot("team-negative-003", "Роботы"),
        )

        queue = build_labeling_queue((run,), candidate_slots=slots)

        self.assertEqual(len(queue.candidates), 1)
        self.assertEqual(
            [(item.area, item.need, item.missing) for item in queue.deficits],
            [("Роботы", "candidate_review", 1), ("Финтех", "candidate_review", 1)],
        )

    def test_wrong_cutoff_run_is_rejected(self) -> None:
        run = make_run("run-1", groups=[group("group-a", "Alpha Tech")], cutoff="2026-09-16")
        slots = (CandidateSlot("team-negative-001", "Финтех"),)

        with self.assertRaisesRegex(ValueError, "not eligible for labeling"):
            build_labeling_queue((run,), candidate_slots=slots)


class QueueNoiseTests(unittest.TestCase):
    def test_maps_every_provenance_to_noise_type(self) -> None:
        run = make_run(
            "run-1",
            proposals=[
                {"proposal_id": "proposal-1", "canonical_name": "Broad Area"},
                {"proposal_id": "proposal-2", "canonical_name": "Off Topic"},
            ],
            decisions=[
                {
                    "proposal_id": "proposal-1",
                    "decision": "reject",
                    "reason": "generic_area",
                    "basis_document_ids": ["document-1"],
                    "explanation": "Too broad.",
                },
                {
                    "proposal_id": "proposal-2",
                    "decision": "reject",
                    "reason": "irrelevant",
                    "basis_document_ids": ["document-1"],
                    "explanation": "Off topic.",
                },
            ],
            exclusions=[
                {
                    "display_name": "Some Company",
                    "normalized_name": "some company",
                    "document_id": "document-1",
                    "reason": "organization",
                    "connector_id": "openalex",
                }
            ],
            issues=[
                {
                    "code": "non_verbatim",
                    "document_id": "document-1",
                    "field": "title",
                    "message": "not exact",
                    "text": "made up fragment",
                }
            ],
            documents=[document("document-1")],
        )
        slots = (
            NoiseSlot("noise-001", NoiseType.BROAD_CONCEPT.value, "Финтех"),
            NoiseSlot("noise-002", NoiseType.IRRELEVANT.value, "Финтех"),
            NoiseSlot("noise-003", NoiseType.NOT_TECHNOLOGY.value, "Финтех"),
            NoiseSlot("noise-004", NoiseType.EXTRACTION_ERROR.value, "Финтех"),
        )

        queue = build_labeling_queue((run,), noise_slots=slots)

        by_id = {item.noise_id: item for item in queue.noise}
        self.assertEqual(by_id["noise-001"].extracted_text, "Broad Area")
        self.assertEqual(by_id["noise-002"].extracted_text, "Off Topic")
        self.assertEqual(by_id["noise-003"].extracted_text, "Some Company")
        self.assertEqual(by_id["noise-004"].extracted_text, "made up fragment")
        self.assertTrue(all(item.origin_kind != "gate_review" for item in queue.noise))

    def test_gate_reject_uses_proposal_document_when_basis_is_missing(self) -> None:
        run = make_run(
            "run-1",
            proposals=[
                {
                    "proposal_id": "proposal-1",
                    "canonical_name": "Broad Area",
                    "document_ids": ["document-1"],
                }
            ],
            decisions=[
                {
                    "proposal_id": "proposal-1",
                    "decision": "reject",
                    "reason": "generic_area",
                    "basis_document_ids": [],
                    "explanation": "Too broad.",
                }
            ],
            documents=[document("document-1", url="https://example.org/broad")],
        )
        slots = (NoiseSlot("noise-001", NoiseType.BROAD_CONCEPT.value, "Финтех"),)

        queue = build_labeling_queue((run,), noise_slots=slots)

        self.assertEqual(len(queue.noise), 1)
        self.assertEqual(queue.noise[0].source_document_url, "https://example.org/broad")

    def test_gate_review_stays_unresolved_instead_of_becoming_noise(self) -> None:
        run = make_run(
            "run-1",
            proposals=[{"proposal_id": "proposal-1", "canonical_name": "Fuzzy Thing"}],
            decisions=[
                {
                    "proposal_id": "proposal-1",
                    "decision": "review",
                    "reason": "insufficient_context",
                    "basis_document_ids": ["document-1"],
                    "explanation": "Unclear.",
                }
            ],
            documents=[document("document-1")],
        )
        slots = (NoiseSlot("noise-001", NoiseType.DUPLICATE.value, "Финтех"),)

        queue = build_labeling_queue((run,), noise_slots=slots)

        self.assertEqual(queue.noise, ())
        self.assertEqual(
            [(item.area, item.need, item.missing) for item in queue.deficits],
            [("duplicate", "noise", 1)],
        )
        self.assertEqual(queue.overflow[0].kind, "gate_review")

    def test_alias_suggestion_becomes_duplicate(self) -> None:
        run = make_run(
            "run-1",
            groups=[group("group-left", "Left Tech"), group("group-right", "Right Tech")],
            suggestions=[
                {
                    "left_group_id": "group-left",
                    "right_group_id": "group-right",
                    "reason": "shared_origin_acronym",
                    "shared_origin_ids": ["origin-1"],
                }
            ],
            documents=[
                document("document-group-left-0", url="https://example.org/left"),
                document("document-group-right-0", url="https://example.org/right"),
            ],
        )
        slots = (NoiseSlot("noise-001", NoiseType.DUPLICATE.value, "Финтех"),)

        queue = build_labeling_queue((run,), noise_slots=slots)

        self.assertEqual(len(queue.noise), 1)
        self.assertEqual(queue.noise[0].extracted_text, "Right Tech")
        self.assertEqual(queue.noise[0].source_document_url, "https://example.org/right")
        self.assertEqual(queue.noise[0].origin_kind, "alias_suggestion")


class QueueEvidenceTests(unittest.TestCase):
    def test_interleaves_directions_respects_cutoff_and_counts_unknowns(self) -> None:
        proposals_documents = []
        proposals = []
        for index, direction in enumerate(["support", "support", "counter"]):
            proposal, doc = evidence_proposal("group-a", index, direction=direction)
            proposals.append(proposal)
            proposals_documents.append(doc)
        future_proposal, future_doc = evidence_proposal(
            "group-a", 3, published_at="2026-10-01"
        )
        proposals.append(future_proposal)
        proposals_documents.append(future_doc)
        mystery_proposal, mystery_doc = evidence_proposal(
            "group-a", 4, published_at=None, trust_tier="unknown"
        )
        proposals.append(mystery_proposal)
        proposals_documents.append(mystery_doc)
        run = make_run(
            "run-1",
            groups=[group("group-a", "Alpha Tech")],
            evidence=proposals,
            documents=proposals_documents,
        )
        slots = (CandidateSlot("team-negative-001", "Финтех"),)

        queue = build_labeling_queue((run,), candidate_slots=slots)

        directions = [item.direction.value for item in queue.evidence]
        self.assertEqual(directions[:2], ["support", "counter"])
        self.assertEqual(len(queue.evidence), 4)
        self.assertEqual(queue.future_evidence_count, 1)
        self.assertEqual(queue.missing_date_count, 1)
        self.assertEqual(queue.unknown_trust_count, 1)
        self.assertTrue(all(item.candidate_id == "team-negative-001" for item in queue.evidence))

    def test_evidence_is_capped_at_400_rows(self) -> None:
        proposals = []
        documents = []
        for index in range(401):
            proposal, doc = evidence_proposal("group-a", index)
            proposals.append(proposal)
            documents.append(doc)
        run = make_run(
            "run-1",
            groups=[group("group-a", "Alpha Tech")],
            evidence=proposals,
            documents=documents,
        )
        slots = (CandidateSlot("team-negative-001", "Финтех"),)

        queue = build_labeling_queue((run,), candidate_slots=slots)

        self.assertEqual(len(queue.evidence), 400)
        self.assertEqual(queue.evidence[-1].evidence_id, "evidence-400")


class QueueJsonlTests(unittest.TestCase):
    def test_records_pass_labeling_contracts_and_rerun_is_identical(self) -> None:
        run = make_run(
            "run-1",
            groups=[group("group-a", "Alpha Tech")],
            decisions=[],
            documents=[document("document-group-a-0")],
        )
        candidate_slots = (CandidateSlot("team-negative-001", "Финтех"),)
        noise_slots = (NoiseSlot("noise-001", NoiseType.BROAD_CONCEPT.value, "Финтех"),)

        first = build_labeling_queue(
            (run,), candidate_slots=candidate_slots, noise_slots=noise_slots
        )
        second = build_labeling_queue(
            (run,), candidate_slots=candidate_slots, noise_slots=noise_slots
        )
        negative_first, noise_first = queue_to_jsonl(first)
        negative_second, noise_second = queue_to_jsonl(second)

        self.assertEqual(negative_first, negative_second)
        parsed = json.loads(negative_first.decode().strip())
        self.assertEqual(parsed["candidate_id"], "team-negative-001")
        # No noise pool in fixtures: empty side renders empty bytes, deficits recorded.
        self.assertEqual(noise_first, b"")
        self.assertTrue(any(item.area == "broad_concept" for item in first.deficits))

    def test_empty_queue_renders_empty_bytes(self) -> None:
        queue = build_labeling_queue(())
        negative, noise = queue_to_jsonl(queue)

        self.assertEqual((negative, noise), (b"", b""))


class QueueSearchCoverageTests(unittest.TestCase):
    def test_complete_zero_result_search_still_counts_as_coverage(self) -> None:
        run = make_run(
            "run-1",
            scientific_status="complete",
            scientific_usage={"requests_used": 1, "stop_reason": "channels_exhausted"},
            media_status="complete",
            media_provider="gdelt",
            media_usage=[{"requests_used": 1, "stop_reason": "channels_exhausted"}],
        )

        coverage = build_labeling_queue((run,)).search_coverage[0]

        self.assertEqual(coverage.source_classes, ("scientific", "industry"))

    def test_complete_without_usage_is_not_claimed_as_coverage(self) -> None:
        run = make_run(
            "run-1",
            scientific_status="complete",
            media_status="complete",
            media_provider="gdelt",
        )

        coverage = build_labeling_queue((run,)).search_coverage[0]

        self.assertEqual(coverage.source_classes, ())

    def test_failed_or_unknown_search_is_not_claimed_as_coverage(self) -> None:
        run = make_run(
            "run-1",
            scientific_status="failed",
            media_status="complete",
            media_provider="unexpected-provider",
        )

        coverage = build_labeling_queue((run,)).search_coverage[0]

        self.assertEqual(coverage.source_classes, ())


class QueueDomainTests(unittest.TestCase):
    def test_foreign_domain_slot_stays_deficit(self) -> None:
        run = make_run("run-1", groups=[group("group-a", "Alpha Tech")])
        slots = (
            CandidateSlot("team-negative-001", "Роботы"),
            CandidateSlot("team-negative-002", "Финтех"),
        )

        queue = build_labeling_queue((run,), candidate_slots=slots)

        self.assertEqual(len(queue.candidates), 1)
        self.assertEqual(queue.candidates[0].candidate_id, "team-negative-002")
        self.assertEqual(queue.candidates[0].domain, "Финтех")
        self.assertEqual(queue.candidates[0].analysis_scope_key, "fintech-v1")
        self.assertEqual(
            [(item.area, item.need, item.missing) for item in queue.deficits],
            [("Роботы", "candidate_review", 1)],
        )

    def test_run_domains_override_maps_free_text_vault(self) -> None:
        run = make_run(
            "run-1",
            groups=[group("group-a", "Alpha Tech")],
            domain="Перспективные решения в финтехе",
        )
        slots = (CandidateSlot("team-negative-001", "Финтех"),)

        queue = build_labeling_queue(
            (run,), candidate_slots=slots, run_domains={"run-1": "Финтех"}
        )

        self.assertEqual(len(queue.candidates), 1)
        self.assertEqual(queue.candidates[0].domain, "Финтех")

    def test_run_without_controlled_domain_is_rejected(self) -> None:
        run = make_run(
            "run-1",
            groups=[group("group-a", "Alpha Tech")],
            domain="Перспективные решения в финтехе",
        )
        slots = (CandidateSlot("team-negative-001", "Финтех"),)

        with self.assertRaisesRegex(ValueError, "no controlled domain"):
            build_labeling_queue((run,), candidate_slots=slots)


if __name__ == "__main__":
    unittest.main()
