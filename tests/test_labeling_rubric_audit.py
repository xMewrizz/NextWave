from __future__ import annotations

import unittest

from nextwave.labeling.rubric_audit import (
    audit_candidate,
    build_maturity_queue,
    maturity_cues,
)


def candidate() -> dict:
    return {
        "candidate_id": "team-negative-001",
        "canonical_name": "Example technology",
        "domain": "Edge",
    }


def document(
    number: int,
    *,
    trust: str = "A",
    source_class: str = "scientific",
    origin: str | None = None,
    published_at: str = "2026-01-01",
) -> dict:
    return {
        "document_id": f"document-{number}",
        "origin_id": origin or f"origin-{number}",
        "trust_tier": trust,
        "source_class": source_class,
        "published_at": published_at,
    }


def claim(number: int, kind: str, *, scope: str = "full_candidate") -> dict:
    return {
        "claim_id": f"claim-{number}",
        "document_id": f"document-{number}",
        "kind": kind,
        "scope": scope,
    }


class RubricAuditTests(unittest.TestCase):
    def test_mature_requires_full_candidate_ab_maturity_evidence(self) -> None:
        result = audit_candidate(
            candidate(),
            [document(1)],
            [claim(1, "adoption")],
            search_coverage_complete=True,
            recent_media_rows=[],
        )

        self.assertTrue(result["mature_eligible"])
        self.assertEqual(result["status"], "mature_ready")
        self.assertEqual(result["maturity_ab_origin_ids"], ["origin-1"])

    def test_core_only_or_low_trust_does_not_prove_maturity(self) -> None:
        result = audit_candidate(
            candidate(),
            [document(1), document(2, trust="C")],
            [claim(1, "adoption", scope="core_only"), claim(2, "market")],
            search_coverage_complete=True,
            recent_media_rows=[],
        )

        self.assertFalse(result["mature_eligible"])
        self.assertIn("missing_ab_standard_deployment_or_market", result["deficits"])

    def test_hype_requires_three_recent_publicity_claims_two_origins(self) -> None:
        documents = [
            document(1, source_class="industry", origin="origin-a"),
            document(2, source_class="industry", origin="origin-a"),
            document(3, source_class="industry", origin="origin-b"),
        ]
        claims = [
            claim(1, "promotional_claim"),
            claim(2, "growth"),
            claim(3, "investment"),
        ]
        result = audit_candidate(
            candidate(),
            documents,
            claims,
            search_coverage_complete=True,
            recent_media_rows=[],
        )

        self.assertTrue(result["marketing_hype_eligible"])
        self.assertEqual(result["status"], "marketing_hype_ready")

    def test_hype_is_blocked_by_two_ab_technical_origins(self) -> None:
        documents = [
            document(1, source_class="industry", origin="wave-a"),
            document(2, source_class="industry", origin="wave-a"),
            document(3, source_class="industry", origin="wave-b"),
            document(4, origin="technical-a"),
            document(5, origin="technical-b"),
        ]
        claims = [
            claim(1, "promotional_claim"),
            claim(2, "growth"),
            claim(3, "investment"),
            claim(4, "research"),
            claim(5, "prototype"),
        ]
        result = audit_candidate(
            candidate(),
            documents,
            claims,
            search_coverage_complete=True,
            recent_media_rows=[],
        )

        self.assertFalse(result["marketing_hype_eligible"])
        self.assertIn("two_or_more_ab_technical_origins", result["deficits"])

    def test_retrieval_wave_is_only_a_fetch_action_not_verified_hype(self) -> None:
        recent = [
            {
                "connector": "mediacloud",
                "relevance_class": "strong",
                "published_at": "2026-05-01",
                "origin_id": f"origin-{number}",
            }
            for number in range(1, 4)
        ]
        result = audit_candidate(
            candidate(),
            [],
            [],
            search_coverage_complete=True,
            recent_media_rows=recent,
        )

        self.assertFalse(result["marketing_hype_eligible"])
        self.assertEqual(result["status"], "needs_verified_publicity")
        self.assertEqual(result["next_action"], "fetch_and_verify_publicity_wave")

    def test_claim_for_unknown_document_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown document"):
            audit_candidate(
                candidate(),
                [],
                [claim(1, "research")],
                search_coverage_complete=True,
                recent_media_rows=[],
            )

    def test_maturity_cues_are_grouped_without_duplicate_matches(self) -> None:
        self.assertEqual(
            maturity_cues(
                "Commercial deployment standard",
                "The standard was adopted for production-scale infrastructure.",
            ),
            ("standard", "adoption", "deployment", "market", "infrastructure"),
        )

    def test_maturity_queue_keeps_diverse_top_four(self) -> None:
        documents = []
        ranked = []
        for number in range(1, 7):
            documents.append({
                "candidate_id": "team-negative-001",
                "document_id": f"document-{number}",
                "connector": "openalex",
                "origin_id": "origin-1" if number == 2 else f"origin-{number}",
                "published_at": f"2025-0{number}-01",
                "trust_tier": "A",
                "title": "Commercial deployment",
                "url": f"https://example.org/{number}",
                "excerpt": "The technology entered production use and market adoption.",
            })
            ranked.append({
                "candidate_id": "team-negative-001",
                "document_id": f"document-{number}",
                "connector": "openalex",
                "relevance_class": "strong",
                "score": 80 - number,
                "matched_term": "technology",
            })

        result = build_maturity_queue(documents, ranked)

        self.assertEqual(len(result), 4)
        self.assertEqual([row["selection_rank"] for row in result], [1, 2, 3, 4])
        self.assertEqual(len({row["origin_id"] for row in result}), 4)

    def test_maturity_queue_rejects_irrelevant_and_future_documents(self) -> None:
        documents = [
            {
                "candidate_id": "team-negative-001",
                "document_id": "document-1",
                "connector": "openalex",
                "origin_id": "origin-1",
                "published_at": "2027-01-01",
                "trust_tier": "A",
                "title": "Commercial deployment",
                "url": "https://example.org/1",
                "excerpt": "Production use.",
            }
        ]
        ranked = [{
            "candidate_id": "team-negative-001",
            "document_id": "document-1",
            "connector": "openalex",
            "relevance_class": "none",
            "score": 0,
            "matched_term": "technology",
        }]

        self.assertEqual(build_maturity_queue(documents, ranked), [])


if __name__ == "__main__":
    unittest.main()
