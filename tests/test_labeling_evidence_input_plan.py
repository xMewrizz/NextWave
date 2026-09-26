"""Evidence input plan tests: offline fixtures only, no network, no LLM."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from nextwave.__main__ import main
from nextwave.labeling.evidence_input_plan import (
    LABELING_EVIDENCE_INPUT_PLAN_VERSION,
    export_evidence_input_plan,
    media_capacity,
    score_media_document,
)

BUNDLE = "bundle-test-001"
EXCERPT = "Quantum error correction improves devices."


def digest(payload: bytes) -> dict:
    return {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def dump_json(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def dump_jsonl(rows: list[dict]) -> bytes:
    return (
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    ).encode("utf-8")


def plan_candidate(candidate_id: str, terms: list[str]) -> dict:
    return {
        "candidate_id": candidate_id,
        "search_terms": terms,
        "searches": [
            {"search_id": f"search-{candidate_id}-sci", "source_class": "scientific"},
            {"search_id": f"search-{candidate_id}-ind", "source_class": "industry"},
        ],
    }


def science_doc(candidate_id: str, number: int, **overrides) -> dict:
    payload = {
        "candidate_id": candidate_id,
        "document_id": f"document-{candidate_id}-sci-{number}",
        "connector": "openalex",
        "connector_id": "openalex",
        "search_id": f"search-{candidate_id}-sci",
        "request_id": f"request-{candidate_id}-sci",
        "title": f"Study of {candidate_id} mechanisms",
        "excerpt": f"Abstract on {candidate_id} with experimental details.",
        "published_at": "2025-04-01",
        "origin_id": f"doi:10.1/{candidate_id}-{number}",
        "publisher": "Test Journal",
        "trust_tier": "A",
        "url": f"https://example.org/{candidate_id}-sci-{number}",
        "canonical_url": f"https://example.org/{candidate_id}-sci-{number}",
        "doi": f"10.1/{candidate_id}-{number}",
        "external_id": f"W{number:07d}",
        "source_type": "scientific_publication",
        "snapshot_id": "snapshot-1",
    }
    payload.update(overrides)
    return payload


def media_doc(candidate_id: str, number: int, **overrides) -> dict:
    url = f"https://news.example/{candidate_id}-{number}"
    payload = {
        "candidate_id": candidate_id,
        "document_id": f"document-{candidate_id}-mc-{number}",
        "connector": "mediacloud",
        "connector_id": "mediacloud",
        "search_id": f"search-{candidate_id}-ind",
        "request_id": f"request-{candidate_id}-ind",
        "title": f"News on {candidate_id}",
        "excerpt": None,
        "published_at": "2025-06-01",
        "origin_id": f"url:{url}",
        "publisher": "news.example",
        "trust_tier": "unknown",
        "url": url,
        "canonical_url": url,
        "doi": None,
        "external_id": f"story-{candidate_id}-{number}",
        "source_type": "industry_media",
        "snapshot_id": "snapshot-1",
    }
    payload.update(overrides)
    return payload


def shortlist_row(document: dict, rank: int, **overrides) -> dict:
    payload = {
        "candidate_id": document["candidate_id"],
        "document_id": document["document_id"],
        "connector": "openalex",
        "search_id": document["search_id"],
        "selection_rank": rank,
        "score": 80,
        "relevance_class": "strong",
        "reasons": ["all_tokens_in_title"],
        "matched_term": "test term",
    }
    payload.update(overrides)
    return payload


def queue_row(document: dict, rank: int, **overrides) -> dict:
    payload = {
        "candidate_id": document["candidate_id"],
        "document_id": document["document_id"],
        "connector": "mediacloud",
        "search_id": document["search_id"],
        "request_id": document["request_id"],
        "selection_rank": rank,
        "score": 10,
        "relevance_class": "none",
        "matched_term": "test term",
        "url": document["url"],
        "document_identity": document["canonical_url"],
        "origin_id": document["origin_id"],
        "published_at": document["published_at"],
    }
    payload.update(overrides)
    return payload


def page_row(queue: dict, status: str = "success", content: str = "article_text",
             **overrides) -> dict:
    payload = {
        "candidate_id": queue["candidate_id"],
        "document_id": queue["document_id"],
        "url": queue["url"],
        "search_id": queue["search_id"],
        "request_id": queue["request_id"],
        "selection_rank": queue["selection_rank"],
        "status": status,
        "content_status": content,
        "issue_code": None,
        "http_status": 200,
        "content_sha256": "a" * 64,
        "excerpt_source": "html.article_paragraphs",
        "excerpt_truncated": False,
        "fetched_bytes": 5000,
    }
    payload.update(overrides)
    return payload


def enriched_row(queue: dict, page: dict, excerpt: str | None, evidence: bool,
                 **overrides) -> dict:
    payload = {
        "candidate_id": queue["candidate_id"],
        "document_id": queue["document_id"],
        "connector": "mediacloud",
        "search_id": queue["search_id"],
        "request_id": queue["request_id"],
        "selection_rank": queue["selection_rank"],
        "url": queue["url"],
        "canonical_url": queue["url"],
        "origin_id": queue["document_identity"],
        "origin_method": "canonical_url",
        "title": f"Full story on {queue['candidate_id']}",
        "publisher": "news.example",
        "published_at": "2025-06-01",
        "source_type": "industry_media",
        "trust_tier": "unknown",
        "snapshot_id": "snapshot-1",
        "excerpt": excerpt,
        "content_status": page["content_status"],
        "evidence_text_available": evidence,
        "content_sha256": "a" * 64,
        "extraction_method": "html.article_paragraphs",
        "excerpt_truncated": False,
        "http_status": 200,
        "fetched_bytes": 5000,
    }
    payload.update(overrides)
    return payload


def build_all(
    root: Path,
    candidates: list[dict],
    documents: list[dict],
    queue: list[dict],
    shortlist: list[dict],
    pages: list[dict],
    enriched: list[dict],
) -> tuple[Path, Path, Path, Path]:
    plan_bytes = dump_json({
        "schema_version": "labeling-enrichment-plan-v2",
        "bundle": {"bundle_id": BUNDLE, "candidate_count": len(candidates)},
        "cutoff_date": "2026-09-15",
        "candidates": candidates,
    })
    plan_dir = root / "plan"
    plan_dir.mkdir(parents=True)
    (plan_dir / "plan.json").write_bytes(plan_bytes)
    plan_manifest = {
        "schema_version": "labeling-enrichment-plan-v2",
        "bundle_id": BUNDLE,
        "candidate_count": len(candidates),
        "outputs": {"plan.json": digest(plan_bytes)},
    }
    plan_manifest_bytes = dump_json(plan_manifest)
    (plan_dir / "manifest.json").write_bytes(plan_manifest_bytes)

    result_dir = root / "result"
    result_dir.mkdir(parents=True)
    documents_bytes = dump_jsonl(documents)
    empty = b'{"unrelated": true}\n'
    (result_dir / "documents.jsonl").write_bytes(documents_bytes)
    (result_dir / "coverage.jsonl").write_bytes(empty)
    (result_dir / "request_results.jsonl").write_bytes(empty)
    result_manifest = {
        "schema_version": "labeling-enrichment-result-v2",
        "bundle_id": BUNDLE,
        "plan": digest(plan_bytes),
        "totals": {"returned_documents": len(documents)},
        "outputs": {
            "coverage.jsonl": digest(empty),
            "documents.jsonl": digest(documents_bytes),
            "request_results.jsonl": digest(empty),
        },
    }
    result_manifest_bytes = dump_json(result_manifest)
    (result_dir / "manifest.json").write_bytes(result_manifest_bytes)

    relevance_dir = root / "relevance"
    relevance_dir.mkdir(parents=True)
    ranked_bytes = b'{"unrelated": true}\n'
    shortlist_bytes = dump_jsonl(shortlist)
    queue_bytes = dump_jsonl(queue)
    rel_coverage_bytes = b'{"unrelated": true}\n'
    (relevance_dir / "ranked_documents.jsonl").write_bytes(ranked_bytes)
    (relevance_dir / "scientific_shortlist.jsonl").write_bytes(shortlist_bytes)
    (relevance_dir / "media_fetch_queue.jsonl").write_bytes(queue_bytes)
    (relevance_dir / "coverage.jsonl").write_bytes(rel_coverage_bytes)
    relevance_manifest = {
        "schema_version": "labeling-relevance-plan-v1",
        "bundle_id": BUNDLE,
        "inputs": {
            "plan_file": digest(plan_bytes),
            "plan_manifest": digest(plan_manifest_bytes),
            "result_manifest": digest(result_manifest_bytes),
            "result_files": {
                "coverage.jsonl": digest(empty),
                "documents.jsonl": digest(documents_bytes),
                "request_results.jsonl": digest(empty),
            },
        },
        "totals": {
            "media_fetch_queue_rows": len(queue),
            "scientific_shortlist_rows": len(shortlist),
        },
        "outputs": {
            "ranked_documents.jsonl": digest(ranked_bytes),
            "scientific_shortlist.jsonl": digest(shortlist_bytes),
            "media_fetch_queue.jsonl": digest(queue_bytes),
            "coverage.jsonl": digest(rel_coverage_bytes),
        },
    }
    relevance_manifest_bytes = dump_json(relevance_manifest)
    (relevance_dir / "manifest.json").write_bytes(relevance_manifest_bytes)

    media_dir = root / "media"
    media_dir.mkdir(parents=True)
    pages_bytes = dump_jsonl(pages)
    enriched_bytes = dump_jsonl(enriched)
    media_coverage_bytes = b'{"unrelated": true}\n'
    (media_dir / "page_results.jsonl").write_bytes(pages_bytes)
    (media_dir / "enriched_documents.jsonl").write_bytes(enriched_bytes)
    (media_dir / "coverage.jsonl").write_bytes(media_coverage_bytes)
    media_manifest = {
        "schema_version": "labeling-media-fetch-result-v1",
        "bundle_id": BUNDLE,
        "inputs": {
            "relevance_manifest": digest(relevance_manifest_bytes),
            "relevance_files": {
                "ranked_documents.jsonl": digest(ranked_bytes),
                "scientific_shortlist.jsonl": digest(shortlist_bytes),
                "media_fetch_queue.jsonl": digest(queue_bytes),
                "coverage.jsonl": digest(rel_coverage_bytes),
            },
            "result_manifest": digest(result_manifest_bytes),
            "result_files": {
                "coverage.jsonl": digest(empty),
                "documents.jsonl": digest(documents_bytes),
                "request_results.jsonl": digest(empty),
            },
        },
        "totals": {"planned_fetches": len(pages)},
        "outputs": {
            "page_results.jsonl": digest(pages_bytes),
            "enriched_documents.jsonl": digest(enriched_bytes),
            "coverage.jsonl": digest(media_coverage_bytes),
        },
    }
    (media_dir / "manifest.json").write_bytes(dump_json(media_manifest))
    return plan_dir, result_dir, relevance_dir, media_dir


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


LONG_TEXT = " ".join(f"filler{i}" for i in range(60))


class MediaScoreTests(unittest.TestCase):
    def test_exact_title_scores_100(self) -> None:
        score, final_class, term, reason, _, _, width = score_media_document(
            ("quantum error correction",),
            "Advances in Quantum Error Correction today",
            "Unrelated abstract about biology.",
        )

        self.assertEqual(
            (score, final_class, reason, width),
            (100, "strong", "exact_title_phrase", None),
        )
        self.assertEqual(term, "quantum error correction")

    def test_reordered_title_tokens_score_80(self) -> None:
        score, final_class, _, reason, _, _, _ = score_media_document(
            ("quantum error correction",),
            "Correction methods for error prone quantum devices",
            "No useful abstract.",
        )

        self.assertEqual((score, final_class, reason), (80, "strong", "all_tokens_in_title"))

    def test_exact_article_phrase_scores_70(self) -> None:
        score, _, _, reason, _, _, width = score_media_document(
            ("quantum error correction",),
            "Surface codes for noisy devices",
            "This paper studies quantum error correction with cats.",
        )

        self.assertEqual((score, reason, width), (70, "exact_article_phrase", None))

    def test_tokens_inside_32_window_score_60(self) -> None:
        article = (
            "quantum " + " ".join(f"filler{i}" for i in range(20))
            + " error correction methods improve devices"
        )
        score, final_class, _, reason, _, _, width = score_media_document(
            ("quantum error correction",), "Noisy devices overview", article
        )

        self.assertEqual((score, final_class, reason), (60, "strong", "all_tokens_in_window"))
        self.assertIsNotNone(width)
        self.assertLessEqual(width, 32)

    def test_tokens_beyond_32_apart_miss_60(self) -> None:
        article = "quantum " + " ".join(f"pad{i}" for i in range(40)) + " error correction"
        score, final_class, _, reason, _, _, width = score_media_document(
            ("quantum error correction",), "Noisy devices overview", article
        )

        self.assertEqual((score, final_class, reason), (40, "weak", "partial_token_overlap"))
        self.assertEqual(width, 43)

    def test_partial_overlap_is_weak(self) -> None:
        score, final_class, _, reason, matched, _, _ = score_media_document(
            ("quantum diagnostics",),
            "Quantum hardware benchmarks",
            "We benchmark processors and control lines.",
        )

        self.assertEqual(reason, "partial_token_overlap")
        self.assertEqual(score, 20)
        self.assertEqual(final_class, "weak")
        self.assertEqual(list(matched), ["quantum"])

    def test_no_overlap_is_none(self) -> None:
        score, final_class, _, reason, _, _, _ = score_media_document(
            ("quantum error correction",),
            "Baking sourdough at home",
            "Flour, water, salt and patience.",
        )

        self.assertEqual((score, final_class, reason), (0, "none", "no_token_overlap"))

    def test_short_terms_do_not_match_inside_words(self) -> None:
        score_rag, _, _, _, _, _, _ = score_media_document(
            ("RAG",), "Data storage systems", "A story about cats."
        )
        score_ai, _, _, _, _, _, _ = score_media_document(
            ("AI",), "Railway logistics overview", "A story about cats."
        )

        self.assertEqual(score_rag, 0)
        self.assertEqual(score_ai, 0)

    def test_hyphenated_term_matches_spaced_tokens(self) -> None:
        score, _, _, reason, _, _, _ = score_media_document(
            ("edge-ai",), "Edge AI accelerators for inference", "Short text."
        )

        self.assertEqual((score, reason), (100, "exact_title_phrase"))

    def test_best_term_wins_and_ties_keep_plan_order(self) -> None:
        best = score_media_document(
            ("cats and dogs", "quantum error correction"),
            "Quantum error correction overview",
            "Short text.",
        )
        tied = score_media_document(
            ("alpha beta", "beta alpha"), "Alpha and beta particles", "Short text."
        )

        self.assertEqual(best[2], "quantum error correction")
        self.assertEqual(best[0], 100)
        self.assertEqual(tied[2], "alpha beta")
        self.assertEqual(tied[0], 80)


class EvidenceSelectionTests(unittest.TestCase):
    def _two_candidates(self, root: Path):
        candidates = [
            plan_candidate("c1", ["quantum error correction"]),
            plan_candidate("c2", ["photonic interconnect"]),
        ]
        sci1 = science_doc("c1", 1)
        sci2 = science_doc("c1", 2)
        med1 = media_doc("c1", 1)
        med2 = media_doc("c1", 2)
        documents = [sci1, sci2, med1, med2]
        shortlist = [shortlist_row(sci1, 1), shortlist_row(sci2, 2)]
        queue = [queue_row(med1, 1), queue_row(med2, 2)]
        article = (
            "Engineers described quantum error correction with surface codes "
            "and reported the first prototype measurements from a laboratory."
        )
        pages = [
            page_row(queue[0]),
            page_row(queue[1], content="meta_description", issue_code="none",
                     http_status=200),
        ]
        enriched = [
            enriched_row(queue[0], pages[0], article, True),
            enriched_row(
                queue[1], pages[1],
                "A startup announced a limited pilot of a photonic device.",
                False, content_status="meta_description",
            ),
        ]
        return build_all(root, candidates, documents, queue, shortlist, pages, enriched)

    def test_meta_title_failed_stay_out_of_evidence_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result, relevance, media = self._two_candidates(root)
            paths = export_evidence_input_plan(
                plan_dir=plan, result_dir=result, relevance_dir=relevance,
                media_dir=media, output_dir=root / "out",
            )
            evidence = read_jsonl(paths.evidence_input_documents)
            reranked = read_jsonl(paths.media_reranked)

        by_doc = {row["document_id"]: row for row in evidence}
        self.assertIn("document-c1-sci-1", by_doc)
        self.assertIn("document-c1-mc-1", by_doc)
        self.assertNotIn("document-c1-mc-2", by_doc)
        meta_rows = [row for row in reranked if row["final_class"] == "not_evidence_text"]
        self.assertEqual(len(meta_rows), 1)
        self.assertEqual(meta_rows[0]["exclusion_reason"], "meta_description_not_evidence")

    def test_media_none_is_never_a_filler(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result, relevance, media = self._two_candidates(root)
            paths = export_evidence_input_plan(
                plan_dir=plan, result_dir=result, relevance_dir=relevance,
                media_dir=media, output_dir=root / "out",
            )
            evidence = read_jsonl(paths.evidence_input_documents)

        media_rows = [row for row in evidence if row["connector"] == "mediacloud"]
        self.assertTrue(media_rows)
        self.assertTrue(all(row["relevance_class"] in ("strong", "weak") for row in media_rows))

    def test_capacity_two_with_four_scientific(self) -> None:
        self.assertEqual(media_capacity(4), 2)

    def test_capacity_three_with_three_scientific(self) -> None:
        self.assertEqual(media_capacity(3), 3)

    def test_capacity_four_with_zero_to_two_scientific(self) -> None:
        self.assertEqual(media_capacity(0), 4)
        self.assertEqual(media_capacity(1), 4)
        self.assertEqual(media_capacity(2), 4)

    def test_final_input_capped_at_six(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = [plan_candidate("c1", ["quantum error correction"])]
            sci = [science_doc("c1", number) for number in range(1, 6)]
            meds = [media_doc("c1", number) for number in range(1, 6)]
            article = (
                "Quantum error correction with surface codes improves devices "
                "and reported prototype measurements from a laboratory team."
            )
            shortlist = [shortlist_row(doc, rank) for rank, doc in enumerate(sci[:4], 1)]
            queue = [queue_row(doc, rank) for rank, doc in enumerate(meds[:4], 1)]
            pages = [page_row(item) for item in queue]
            enriched = [
                enriched_row(item, page, article, True)
                for item, page in zip(queue, pages, strict=True)
            ]
            plan, result, relevance, media = build_all(
                root, candidates, sci + meds[:4], queue, shortlist, pages, enriched
            )
            paths = export_evidence_input_plan(
                plan_dir=plan, result_dir=result, relevance_dir=relevance,
                media_dir=media, output_dir=root / "out",
            )
            evidence = [
                row for row in read_jsonl(paths.evidence_input_documents)
                if row["candidate_id"] == "c1"
            ]

        self.assertEqual(len(evidence), 6)

    def test_round_robin_source_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result, relevance, media = self._two_candidates(root)
            paths = export_evidence_input_plan(
                plan_dir=plan, result_dir=result, relevance_dir=relevance,
                media_dir=media, output_dir=root / "out",
            )
            evidence = [
                row for row in read_jsonl(paths.evidence_input_documents)
                if row["candidate_id"] == "c1"
            ]

        self.assertEqual(
            [(row["final_rank"], row["source_class"]) for row in evidence],
            [(1, "scientific"), (2, "industry"), (3, "scientific")],
        )
        self.assertEqual([row["source_rank"] for row in evidence], [1, 1, 2])

    def test_diversity_prefers_unused_origin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = [plan_candidate("c1", ["quantum error correction"])]
            meds = [
                media_doc("c1", 1),
                media_doc("c1", 2),
                media_doc("c1", 3, url="https://other.example/x",
                          canonical_url="https://other.example/x",
                          origin_id="url:https://other.example/x",
                          publisher="other.example"),
            ]
            article = (
                "Quantum error correction with surface codes improves devices "
                "and reported prototype measurements from a laboratory team."
            )
            queue = [queue_row(doc, rank) for rank, doc in enumerate(meds, 1)]
            pages = [page_row(item) for item in queue]
            enriched = [
                enriched_row(item, page, article, True)
                for item, page in zip(queue, pages, strict=True)
            ]
            plan, result, relevance, media = build_all(
                root, candidates, meds, queue, [], pages, enriched
            )
            paths = export_evidence_input_plan(
                plan_dir=plan, result_dir=result, relevance_dir=relevance,
                media_dir=media, output_dir=root / "out",
            )
            evidence = read_jsonl(paths.evidence_input_documents)

        picked = [row["document_id"] for row in evidence]
        self.assertEqual(picked[0], "document-c1-mc-1")
        self.assertIn("document-c1-mc-3", picked)

    def test_candidate_without_media_stays_in_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = [plan_candidate("c1", ["quantum error correction"])]
            sci = [science_doc("c1", 1)]
            shortlist = [shortlist_row(sci[0], 1)]
            plan, result, relevance, media = build_all(
                root, candidates, sci, [], shortlist, [], []
            )
            paths = export_evidence_input_plan(
                plan_dir=plan, result_dir=result, relevance_dir=relevance,
                media_dir=media, output_dir=root / "out",
            )
            coverage = read_jsonl(paths.coverage)

        self.assertEqual(len(coverage), 1)
        self.assertEqual(coverage[0]["media_planned"], 0)
        self.assertTrue(coverage[0]["has_evidence_input"])

    def test_candidate_without_science_gets_media(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = [plan_candidate("c1", ["quantum error correction"])]
            med = [media_doc("c1", 1)]
            article = (
                "Quantum error correction with surface codes improves devices "
                "and reported prototype measurements from a laboratory team."
            )
            queue = [queue_row(med[0], 1)]
            pages = [page_row(queue[0])]
            enriched = [enriched_row(queue[0], pages[0], article, True)]
            plan, result, relevance, media = build_all(
                root, candidates, med, queue, [], pages, enriched
            )
            paths = export_evidence_input_plan(
                plan_dir=plan, result_dir=result, relevance_dir=relevance,
                media_dir=media, output_dir=root / "out",
            )
            evidence = read_jsonl(paths.evidence_input_documents)
            coverage = read_jsonl(paths.coverage)

        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["source_class"], "industry")
        self.assertEqual(coverage[0]["empty_reasons"], ["no_scientific_available"])

    def test_candidate_without_any_evidence_has_reasons(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = [plan_candidate("c1", ["quantum error correction"])]
            med = [media_doc("c1", 1)]
            queue = [queue_row(med[0], 1)]
            pages = [page_row(queue[0], status="failed", content="title_only",
                              issue_code="network_error", http_status=None)]
            plan, result, relevance, media = build_all(
                root, candidates, med, queue, [], pages, []
            )
            paths = export_evidence_input_plan(
                plan_dir=plan, result_dir=result, relevance_dir=relevance,
                media_dir=media, output_dir=root / "out",
            )
            coverage = read_jsonl(paths.coverage)

        self.assertFalse(coverage[0]["has_evidence_input"])
        self.assertIn("no_final_input", coverage[0]["empty_reasons"])

    def test_permuted_inputs_give_identical_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            plan, result, relevance, media = self._two_candidates(first)
            export_evidence_input_plan(
                plan_dir=plan, result_dir=result, relevance_dir=relevance,
                media_dir=media, output_dir=first / "out",
            )
            second = Path(directory) / "second"
            candidates = [
                plan_candidate("c2", ["photonic interconnect"]),
                plan_candidate("c1", ["quantum error correction"]),
            ]
            sci1 = science_doc("c1", 1)
            sci2 = science_doc("c1", 2)
            med1 = media_doc("c1", 1)
            med2 = media_doc("c1", 2)
            documents = [med2, med1, sci2, sci1]
            shortlist = [shortlist_row(sci2, 2), shortlist_row(sci1, 1)]
            queue = [queue_row(med2, 2), queue_row(med1, 1)]
            article = (
                "Engineers described quantum error correction with surface codes "
                "and reported the first prototype measurements from a laboratory."
            )
            pages = [
                page_row(queue[1]),
                page_row(queue[0], content="meta_description", issue_code="none",
                         http_status=200),
            ]
            enriched = [
                enriched_row(queue[1], pages[0], article, True),
                enriched_row(
                    queue[0], pages[1],
                    "A startup announced a limited pilot of a photonic device.",
                    False, content_status="meta_description",
                ),
            ]
            plan_b, result_b, relevance_b, media_b = build_all(
                second, candidates, documents, queue, shortlist, pages, enriched
            )
            export_evidence_input_plan(
                plan_dir=plan_b, result_dir=result_b, relevance_dir=relevance_b,
                media_dir=media_b, output_dir=second / "out",
            )
            for name in (
                "media_reranked.jsonl",
                "evidence_input_documents.jsonl",
                "coverage.jsonl",
            ):
                self.assertEqual(
                    (first / "out" / name).read_bytes(),
                    (second / "out" / name).read_bytes(),
                    name,
                )


class EvidenceChainTests(unittest.TestCase):
    def _minimal(self, root: Path):
        candidates = [plan_candidate("c1", ["quantum error correction"])]
        sci = [science_doc("c1", 1)]
        med = [media_doc("c1", 1)]
        article = (
            "Quantum error correction with surface codes improves devices "
            "and reported prototype measurements from a laboratory team."
        )
        queue = [queue_row(med[0], 1)]
        pages = [page_row(queue[0])]
        enriched = [enriched_row(queue[0], pages[0], article, True)]
        shortlist = [shortlist_row(sci[0], 1)]
        return build_all(root, candidates, sci + med, queue, shortlist, pages, enriched)

    def test_corrupt_checksum_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result, relevance, media = self._minimal(root)
            with (media / "page_results.jsonl").open("ab") as handle:
                handle.write(b" ")
            with self.assertRaisesRegex(ValueError, "mismatch with manifest"):
                export_evidence_input_plan(
                    plan_dir=plan, result_dir=result, relevance_dir=relevance,
                    media_dir=media, output_dir=root / "out",
                )
            self.assertFalse((root / "out").exists())

    def test_broken_manifest_chain_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result, relevance, media = self._minimal(root)
            manifest = json.loads((relevance / "manifest.json").read_text(encoding="utf-8"))
            manifest["inputs"]["result_manifest"]["sha256"] = "0" * 64
            (relevance / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not reference this exact"):
                export_evidence_input_plan(
                    plan_dir=plan, result_dir=result, relevance_dir=relevance,
                    media_dir=media, output_dir=root / "out",
                )
            self.assertFalse((root / "out").exists())

    def test_article_without_excerpt_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result, relevance, media = self._minimal(root)
            rows = read_jsonl(media / "enriched_documents.jsonl")
            rows[0]["excerpt"] = ""
            (media / "enriched_documents.jsonl").write_bytes(
                dump_jsonl(rows)
            )
            manifest = json.loads((media / "manifest.json").read_text(encoding="utf-8"))
            payload = (media / "enriched_documents.jsonl").read_bytes()
            manifest["outputs"]["enriched_documents.jsonl"] = {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            (media / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "has no excerpt"):
                export_evidence_input_plan(
                    plan_dir=plan, result_dir=result, relevance_dir=relevance,
                    media_dir=media, output_dir=root / "out",
                )
            self.assertFalse((root / "out").exists())

    def test_failed_page_in_enriched_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = [plan_candidate("c1", ["quantum error correction"])]
            med = [media_doc("c1", 1)]
            queue = [queue_row(med[0], 1)]
            pages = [page_row(queue[0], status="failed", content="title_only",
                              issue_code="network_error", http_status=None)]
            enriched = [enriched_row(
                queue[0], pages[0], None, False, content_status="title_only",
                extraction_method=None, http_status=None,
            )]
            plan, result, relevance, media = build_all(
                root, candidates, med, queue, [], pages, enriched
            )
            with self.assertRaisesRegex(ValueError, "failed page"):
                export_evidence_input_plan(
                    plan_dir=plan, result_dir=result, relevance_dir=relevance,
                    media_dir=media, output_dir=root / "out",
                )
            self.assertFalse((root / "out").exists())

    def test_future_dated_document_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = [plan_candidate("c1", ["quantum error correction"])]
            sci = [science_doc("c1", 1, published_at="2026-09-20")]
            shortlist = [shortlist_row(sci[0], 1)]
            plan, result, relevance, media = build_all(
                root, candidates, sci, [], shortlist, [], []
            )
            with self.assertRaisesRegex(ValueError, "past cutoff"):
                export_evidence_input_plan(
                    plan_dir=plan, result_dir=result, relevance_dir=relevance,
                    media_dir=media, output_dir=root / "out",
                )
            self.assertFalse((root / "out").exists())

    def test_target_and_labels_stay_out_and_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            plan, result, relevance, media = EvidenceSelectionTests()._two_candidates(first)
            export_evidence_input_plan(
                plan_dir=plan, result_dir=result, relevance_dir=relevance,
                media_dir=media, output_dir=first / "out",
            )
            second = Path(directory) / "second"
            candidates = [
                dict(plan_candidate("c1", ["quantum error correction"]),
                     target=1, label="weak_signal", planned_class="emerging"),
                dict(plan_candidate("c2", ["photonic interconnect"]),
                     target=0, label="mature", planned_class="mature"),
            ]
            sci1 = science_doc("c1", 1)
            sci2 = science_doc("c1", 2)
            med1 = media_doc("c1", 1)
            med2 = media_doc("c1", 2)
            tainted_docs = [
                dict(sci1, target=1, expert_score=7),
                dict(sci2, target=1, expert_score=7),
                dict(med1, target=1, expert_score=7),
                dict(med2, target=1, expert_score=7),
            ]
            article = (
                "Engineers described quantum error correction with surface codes "
                "and reported the first prototype measurements from a laboratory."
            )
            queue = [queue_row(med1, 1), queue_row(med2, 2)]
            pages = [
                page_row(queue[0]),
                page_row(queue[1], content="meta_description", issue_code="none",
                         http_status=200),
            ]
            enriched = [
                enriched_row(queue[0], pages[0], article, True),
                enriched_row(
                    queue[1], pages[1],
                    "A startup announced a limited pilot of a photonic device.",
                    False, content_status="meta_description",
                ),
            ]
            shortlist = [shortlist_row(sci1, 1), shortlist_row(sci2, 2)]
            plan_b, result_b, relevance_b, media_b = build_all(
                second, candidates, tainted_docs, queue, shortlist, pages, enriched
            )
            export_evidence_input_plan(
                plan_dir=plan_b, result_dir=result_b, relevance_dir=relevance_b,
                media_dir=media_b, output_dir=second / "out",
            )
            for name in (
                "media_reranked.jsonl",
                "evidence_input_documents.jsonl",
                "coverage.jsonl",
            ):
                self.assertEqual(
                    (first / "out" / name).read_bytes(),
                    (second / "out" / name).read_bytes(),
                    name,
                )
            blob = b"".join(
                (second / "out" / name).read_bytes()
                for name in (
                    "media_reranked.jsonl",
                    "evidence_input_documents.jsonl",
                    "coverage.jsonl",
                )
            )
            for token in (b"target", b"expert_score", b"planned_class", b"weak_signal"):
                self.assertNotIn(token, blob)

    def test_existing_output_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result, relevance, media = self._minimal(root)
            (root / "out").mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                export_evidence_input_plan(
                    plan_dir=plan, result_dir=result, relevance_dir=relevance,
                    media_dir=media, output_dir=root / "out",
                )

    def test_cli_error_returns_1_without_partial_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result, relevance, media = self._minimal(root)
            (media / "page_results.jsonl").write_bytes(b"corrupt{{")
            exit_code = main([
                "labeling-evidence-input-plan",
                "--plan", str(plan),
                "--result", str(result),
                "--relevance", str(relevance),
                "--media", str(media),
                "--output", str(root / "out"),
            ])

        self.assertEqual(exit_code, 1)
        self.assertFalse((root / "out").exists())

    def test_no_api_or_llm_usage(self) -> None:
        # Static check: the planner must not import network or LLM libraries.
        module_path = (
            Path(__file__).resolve().parent.parent
            / "src" / "nextwave" / "labeling" / "evidence_input_plan.py"
        )
        text = module_path.read_text(encoding="utf-8").casefold()
        for token in (
            "urllib", "requests", "socket", "yandex", "openai", "anthropic",
            "embedding", "sentence_transformers", "sklearn", "http.client",
        ):
            self.assertNotIn(token, text)
        self.assertEqual(
            LABELING_EVIDENCE_INPUT_PLAN_VERSION, "labeling-evidence-input-plan-v2"
        )


class EvidenceSetEqualityTests(unittest.TestCase):
    def test_missing_page_for_queue_pair_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = [plan_candidate("c1", ["quantum error correction"])]
            sci = [science_doc("c1", 1)]
            med = [media_doc("c1", 1), media_doc("c1", 2)]
            queue = [queue_row(med[0], 1), queue_row(med[1], 2)]
            pages = [page_row(queue[0])]
            enriched = [enriched_row(queue[0], pages[0], EXCERPT, True)]
            shortlist = [shortlist_row(sci[0], 1)]
            plan, result, relevance, media = build_all(
                root, candidates, sci + med, queue, shortlist, pages, enriched
            )
            with self.assertRaisesRegex(ValueError, "page result missing for queue pair"):
                export_evidence_input_plan(
                    plan_dir=plan, result_dir=result, relevance_dir=relevance,
                    media_dir=media, output_dir=root / "out",
                )
            self.assertFalse((root / "out").exists())

    def test_extra_page_pair_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = [plan_candidate("c1", ["quantum error correction"])]
            sci = [science_doc("c1", 1)]
            med = [media_doc("c1", 1)]
            queue = [queue_row(med[0], 1)]
            pages = [page_row(queue[0])]
            ghost_queue = queue_row(media_doc("c1", 9), 2)
            pages.append(page_row(ghost_queue))
            enriched = [enriched_row(queue[0], pages[0], EXCERPT, True)]
            shortlist = [shortlist_row(sci[0], 1)]
            plan, result, relevance, media = build_all(
                root, candidates, sci + med, queue, shortlist, pages, enriched
            )
            with self.assertRaisesRegex(ValueError, "missing in fetch queue"):
                export_evidence_input_plan(
                    plan_dir=plan, result_dir=result, relevance_dir=relevance,
                    media_dir=media, output_dir=root / "out",
                )
            self.assertFalse((root / "out").exists())

    def test_success_page_without_enriched_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = [plan_candidate("c1", ["quantum error correction"])]
            sci = [science_doc("c1", 1)]
            med = [media_doc("c1", 1)]
            queue = [queue_row(med[0], 1)]
            pages = [page_row(queue[0])]
            shortlist = [shortlist_row(sci[0], 1)]
            plan, result, relevance, media = build_all(
                root, candidates, sci + med, queue, shortlist, pages, []
            )
            with self.assertRaisesRegex(ValueError, "missing in enriched documents"):
                export_evidence_input_plan(
                    plan_dir=plan, result_dir=result, relevance_dir=relevance,
                    media_dir=media, output_dir=root / "out",
                )
            self.assertFalse((root / "out").exists())

    def test_extra_enriched_row_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = [plan_candidate("c1", ["quantum error correction"])]
            sci = [science_doc("c1", 1)]
            med = [media_doc("c1", 1)]
            queue = [queue_row(med[0], 1)]
            pages = [page_row(queue[0])]
            ghost_queue = queue_row(media_doc("c1", 9), 2)
            ghost_page = page_row(ghost_queue)
            enriched = [
                enriched_row(queue[0], pages[0], EXCERPT, True),
                enriched_row(ghost_queue, ghost_page, "Ghost text.", True),
            ]
            shortlist = [shortlist_row(sci[0], 1)]
            plan, result, relevance, media = build_all(
                root, candidates, sci + med, queue, shortlist, pages, enriched
            )
            with self.assertRaisesRegex(ValueError, "missing in page results"):
                export_evidence_input_plan(
                    plan_dir=plan, result_dir=result, relevance_dir=relevance,
                    media_dir=media, output_dir=root / "out",
                )
            self.assertFalse((root / "out").exists())

    def test_scientific_ranks_with_gap_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = [plan_candidate("c1", ["quantum error correction"])]
            sci = [science_doc("c1", 1), science_doc("c1", 2)]
            med = [media_doc("c1", 1)]
            queue = [queue_row(med[0], 1)]
            pages = [page_row(queue[0])]
            enriched = [enriched_row(queue[0], pages[0], EXCERPT, True)]
            shortlist = [shortlist_row(sci[0], 1), shortlist_row(sci[1], 3)]
            plan, result, relevance, media = build_all(
                root, candidates, sci + med, queue, shortlist, pages, enriched
            )
            with self.assertRaisesRegex(ValueError, "must form 1\\.\\.N"):
                export_evidence_input_plan(
                    plan_dir=plan, result_dir=result, relevance_dir=relevance,
                    media_dir=media, output_dir=root / "out",
                )
            self.assertFalse((root / "out").exists())

    def test_duplicate_scientific_rank_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = [plan_candidate("c1", ["quantum error correction"])]
            sci = [science_doc("c1", 1), science_doc("c1", 2)]
            med = [media_doc("c1", 1)]
            queue = [queue_row(med[0], 1)]
            pages = [page_row(queue[0])]
            enriched = [enriched_row(queue[0], pages[0], EXCERPT, True)]
            shortlist = [shortlist_row(sci[0], 1), shortlist_row(sci[1], 1)]
            plan, result, relevance, media = build_all(
                root, candidates, sci + med, queue, shortlist, pages, enriched
            )
            with self.assertRaisesRegex(ValueError, "must form 1\\.\\.N"):
                export_evidence_input_plan(
                    plan_dir=plan, result_dir=result, relevance_dir=relevance,
                    media_dir=media, output_dir=root / "out",
                )
            self.assertFalse((root / "out").exists())

    def test_media_ranks_with_gap_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = [plan_candidate("c1", ["quantum error correction"])]
            sci = [science_doc("c1", 1)]
            med = [media_doc("c1", 1), media_doc("c1", 2)]
            queue = [queue_row(med[0], 1), queue_row(med[1], 3)]
            pages = [page_row(item) for item in queue]
            enriched = [
                enriched_row(item, page, EXCERPT, True)
                for item, page in zip(queue, pages, strict=True)
            ]
            shortlist = [shortlist_row(sci[0], 1)]
            plan, result, relevance, media = build_all(
                root, candidates, sci + med, queue, shortlist, pages, enriched
            )
            with self.assertRaisesRegex(ValueError, "must form 1\\.\\.N"):
                export_evidence_input_plan(
                    plan_dir=plan, result_dir=result, relevance_dir=relevance,
                    media_dir=media, output_dir=root / "out",
                )
            self.assertFalse((root / "out").exists())

    def test_unknown_candidate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = [plan_candidate("c1", ["quantum error correction"])]
            sci = [science_doc("c1", 1)]
            med = [media_doc("c1", 1)]
            ghost = media_doc("ghost", 1)
            documents = sci + med + [ghost]
            queue = [queue_row(med[0], 1), queue_row(ghost, 1)]
            pages = [page_row(item) for item in queue]
            enriched = [
                enriched_row(item, page, EXCERPT, True)
                for item, page in zip(queue, pages, strict=True)
            ]
            shortlist = [shortlist_row(sci[0], 1)]
            plan, result, relevance, media = build_all(
                root, candidates, documents, queue, shortlist, pages, enriched
            )
            with self.assertRaisesRegex(ValueError, "unknown candidate"):
                export_evidence_input_plan(
                    plan_dir=plan, result_dir=result, relevance_dir=relevance,
                    media_dir=media, output_dir=root / "out",
                )
            self.assertFalse((root / "out").exists())

    def test_wrong_search_class_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = [plan_candidate("c1", ["quantum error correction"])]
            sci = [science_doc("c1", 1)]
            med = [media_doc("c1", 1)]
            queue = [queue_row(med[0], 1)]
            pages = [page_row(queue[0])]
            enriched = [enriched_row(queue[0], pages[0], EXCERPT, True)]
            bad_shortlist = [shortlist_row(sci[0], 1, search_id="search-c1-ind")]
            plan, result, relevance, media = build_all(
                root, candidates, sci + med, queue, bad_shortlist, pages, enriched
            )
            with self.assertRaisesRegex(ValueError, "non-scientific search"):
                export_evidence_input_plan(
                    plan_dir=plan, result_dir=result, relevance_dir=relevance,
                    media_dir=media, output_dir=root / "out",
                )
            self.assertFalse((root / "out").exists())

    def test_wrong_source_connector_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = [plan_candidate("c1", ["quantum error correction"])]
            sci = [science_doc("c1", 1, connector_id="mediacloud")]
            med = [media_doc("c1", 1)]
            queue = [queue_row(med[0], 1)]
            pages = [page_row(queue[0])]
            enriched = [enriched_row(queue[0], pages[0], EXCERPT, True)]
            shortlist = [shortlist_row(sci[0], 1)]
            plan, result, relevance, media = build_all(
                root, candidates, sci + med, queue, shortlist, pages, enriched
            )
            with self.assertRaisesRegex(ValueError, "source document is not openalex"):
                export_evidence_input_plan(
                    plan_dir=plan, result_dir=result, relevance_dir=relevance,
                    media_dir=media, output_dir=root / "out",
                )
            self.assertFalse((root / "out").exists())


            with self.assertRaisesRegex(ValueError, "source document is not openalex"):
                export_evidence_input_plan(
                    plan_dir=plan, result_dir=result, relevance_dir=relevance,
                    media_dir=media, output_dir=root / "out",
                )
            self.assertFalse((root / "out").exists())


def pads(count: int) -> str:
    return " ".join(f"pad{i}" for i in range(count))


def width_article(term_first: str, term_rest: str, fillers: int) -> str:
    return f"{term_first} {pads(fillers)} {term_rest}"


class EvidencePolicyCTests(unittest.TestCase):
    TERM = ("alpha", "beta", "gamma")
    NAMES = ["strong60", "w33", "w64", "w96", "w97", "w256", "wnull", "shallow"]

    def _boundary(self, root: Path):
        candidates = [
            plan_candidate("c1", ["alpha beta gamma"]),
            plan_candidate("c2", ["alpha beta gamma"]),
        ]
        articles = {
            "strong60": width_article("alpha", "beta gamma", 5),
            "w33": width_article("alpha", "beta gamma", 30),
            "w64": width_article("alpha", "beta gamma", 61),
            "w96": width_article("alpha", "beta gamma", 93),
            "w97": width_article("alpha", "beta gamma", 94),
            "w256": width_article("alpha", "beta gamma", 253),
            "wnull": "alpha processes and beta testing methods",
            "shallow": "alpha beta hardware benchmarks",
        }
        titles = {"wnull": "Gamma rays overview"}
        meds = {}
        queue = []
        pages = []
        enriched = []
        for cand, names in (("c1", self.NAMES[:4]), ("c2", self.NAMES[4:])):
            for rank, name in enumerate(names, start=1):
                title = titles.get(name, "Noisy devices overview")
                doc = media_doc(cand, len(meds) + 1, title=title)
                meds[name] = doc
                item = queue_row(doc, rank)
                queue.append(item)
                page = page_row(item)
                pages.append(page)
                enriched.append(enriched_row(item, page, articles[name], True, title=title))
        documents = list(meds.values())
        plan, result, relevance, media = build_all(
            root, candidates, documents, queue, [], pages, enriched
        )
        return meds, queue, pages, enriched, (plan, result, relevance, media)

    def test_strong_60_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meds, _, _, _, dirs = self._boundary(root)
            plan, result, relevance, media = dirs
            paths = export_evidence_input_plan(
                plan_dir=plan, result_dir=result, relevance_dir=relevance,
                media_dir=media, output_dir=root / "out",
            )
            picked = {
                row["document_id"] for row in read_jsonl(paths.evidence_input_documents)
            }

        self.assertIn(meds["strong60"]["document_id"], picked)

    def test_strong_70_80_100_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = [plan_candidate("c1", ["alpha beta gamma"])]
            meds = [media_doc("c1", 1), media_doc("c1", 2), media_doc("c1", 3)]
            articles = [
                "Alpha beta gamma devices improve recall.",
                "Gamma methods for beta testing alpha systems.",
                "Alpha processes with beta and gamma stages.",
            ]
            titles = [
                "Noisy devices overview",
                "Alpha beta gamma survey",
                "Beta alpha research",
            ]
            queue = [queue_row(doc, rank) for rank, doc in enumerate(meds, start=1)]
            pages = [page_row(item) for item in queue]
            enriched = [
                enriched_row(item, page, article, True, title=title)
                for item, page, article, title in zip(queue, pages, articles, titles, strict=True)
            ]
            documents = [
                dict(doc, title=title) for doc, title in zip(meds, titles, strict=True)
            ]
            plan, result, relevance, media = build_all(
                root, candidates, documents, queue, [], pages, enriched
            )
            paths = export_evidence_input_plan(
                plan_dir=plan, result_dir=result, relevance_dir=relevance,
                media_dir=media, output_dir=root / "out",
            )
            picked = {
                row["document_id"] for row in read_jsonl(paths.evidence_input_documents)
            }

        # 100 exact title, 80 scattered title, 70 exact article phrase.
        self.assertEqual(picked, {doc["document_id"] for doc in meds})

    def test_weak_width_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meds, _, _, _, dirs = self._boundary(root)
            plan, result, relevance, media = dirs
            paths = export_evidence_input_plan(
                plan_dir=plan, result_dir=result, relevance_dir=relevance,
                media_dir=media, output_dir=root / "out",
            )
            picked = {
                row["document_id"] for row in read_jsonl(paths.evidence_input_documents)
            }

        for name in ("w33", "w64", "w96"):
            self.assertIn(meds[name]["document_id"], picked, name)
        for name in ("w97", "w256"):
            self.assertNotIn(meds[name]["document_id"], picked, name)

    def test_weak_null_window_and_shallow_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meds, _, _, _, dirs = self._boundary(root)
            plan, result, relevance, media = dirs
            paths = export_evidence_input_plan(
                plan_dir=plan, result_dir=result, relevance_dir=relevance,
                media_dir=media, output_dir=root / "out",
            )
            evidence = read_jsonl(paths.evidence_input_documents)
            reranked = {
                row["document_id"]: row for row in read_jsonl(paths.media_reranked)
            }
            picked = {row["document_id"] for row in evidence}

            self.assertNotIn(meds["wnull"]["document_id"], picked)
            self.assertNotIn(meds["shallow"]["document_id"], picked)
            self.assertIsNone(reranked[meds["wnull"]["document_id"]]["min_window_width"])
            self.assertEqual(reranked[meds["w256"]["document_id"]]["min_window_width"], 256)
            rows = {row["candidate_id"]: row for row in read_jsonl(paths.coverage)}
            self.assertEqual(rows["c1"]["media_eligible"], 4)
            self.assertEqual(rows["c2"]["media_excluded_wide_window"], 2)
            self.assertEqual(rows["c2"]["media_excluded_no_article_window"], 1)
            self.assertEqual(rows["c2"]["media_excluded_low_overlap"], 1)

    def test_excluded_weak_stays_in_reranked_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meds, _, _, _, dirs = self._boundary(root)
            plan, result, relevance, media = dirs
            paths = export_evidence_input_plan(
                plan_dir=plan, result_dir=result, relevance_dir=relevance,
                media_dir=media, output_dir=root / "out",
            )
            reranked_ids = {
                row["document_id"] for row in read_jsonl(paths.media_reranked)
            }
            picked = {
                row["document_id"] for row in read_jsonl(paths.evidence_input_documents)
            }

        self.assertEqual(
            reranked_ids, {doc["document_id"] for doc in meds.values()}
        )
        self.assertEqual(len(picked), 4)

    def test_capacity_round_robin_diversity_after_filter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = [plan_candidate("c1", ["alpha beta gamma"])]
            sci = [science_doc("c1", number) for number in range(1, 5)]
            meds = [media_doc("c1", number) for number in range(1, 4)]
            article = "Alpha beta gamma devices improve recall in production."
            queue = [queue_row(doc, rank) for rank, doc in enumerate(meds, start=1)]
            pages = [page_row(item) for item in queue]
            enriched = [
                enriched_row(item, page, article, True)
                for item, page in zip(queue, pages, strict=True)
            ]
            shortlist = [shortlist_row(doc, rank) for rank, doc in enumerate(sci, start=1)]
            plan, result, relevance, media = build_all(
                root, candidates, sci + meds, queue, shortlist, pages, enriched
            )
            paths = export_evidence_input_plan(
                plan_dir=plan, result_dir=result, relevance_dir=relevance,
                media_dir=media, output_dir=root / "out",
            )
            evidence = [
                row for row in read_jsonl(paths.evidence_input_documents)
                if row["candidate_id"] == "c1"
            ]

        # 4 scientific -> capacity 2 -> media exhausts after 2, science continues.
        self.assertEqual(len(evidence), 6)
        self.assertEqual(
            [row["source_class"] for row in evidence],
            ["scientific", "industry"] * 2 + ["scientific", "scientific"],
        )
        self.assertEqual([row["final_rank"] for row in evidence], [1, 2, 3, 4, 5, 6])

    def test_permuted_policy_inputs_give_identical_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            _, _, _, _, dirs_a = self._boundary(first)
            export_evidence_input_plan(
                plan_dir=dirs_a[0], result_dir=dirs_a[1], relevance_dir=dirs_a[2],
                media_dir=dirs_a[3], output_dir=first / "out",
            )
            second = Path(directory) / "second"
            candidates = [
                plan_candidate("c1", ["alpha beta gamma"]),
                plan_candidate("c2", ["alpha beta gamma"]),
            ]
            meds_b, queue_a, pages_a, enriched_a, _ = self._boundary(second)
            queue_b = list(reversed(queue_a))
            pages_b = list(reversed(pages_a))
            enriched_b = list(reversed(enriched_a))
            docs_b = list(reversed(list(meds_b.values())))
            rev = second / "rev"
            plan, result, relevance, media = build_all(
                rev, candidates, docs_b, queue_b, [], pages_b, enriched_b
            )
            export_evidence_input_plan(
                plan_dir=plan, result_dir=result, relevance_dir=relevance,
                media_dir=media, output_dir=second / "out",
            )
            for name in (
                "media_reranked.jsonl",
                "evidence_input_documents.jsonl",
                "coverage.jsonl",
            ):
                first_bytes = (first / "out" / name).read_bytes()
                second_bytes = (second / "out" / name).read_bytes()
                self.assertEqual(first_bytes, second_bytes, name)

    def test_v1_schema_is_not_accepted_as_v2(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, _, _, dirs = self._boundary(root)
            plan, result, relevance, media = dirs
            paths = export_evidence_input_plan(
                plan_dir=plan, result_dir=result, relevance_dir=relevance,
                media_dir=media, output_dir=root / "out",
            )
            manifest = json.loads((paths.manifest).read_text(encoding="utf-8"))

        self.assertEqual(
            manifest["schema_version"], "labeling-evidence-input-plan-v2"
        )
        self.assertEqual(
            manifest["selection_policy"],
            {
                "strong_allowed": True,
                "weak_score": 40,
                "weak_min_window_tokens": 33,
                "weak_max_window_tokens": 96,
                "weak_without_article_window_allowed": False,
                "media_article_text_only": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
