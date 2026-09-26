"""Deterministic offline relevance ranking tests: no network, no LLM."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from nextwave.__main__ import main
from nextwave.labeling.relevance_plan import (
    LABELING_RELEVANCE_PLAN_VERSION,
    document_identity,
    export_relevance_plan,
    score_document,
)

BUNDLE = "bundle-test-001"
PLAN_REQUESTS = {
    "c1-oa-en": ("search-c1-oa", "openalex"),
    "c1-oa-ru": ("search-c1-oa", "openalex"),
    "c1-mc": ("search-c1-mc", "mediacloud"),
}


def plan_payload() -> dict:
    return {
        "schema_version": "labeling-enrichment-plan-v2",
        "bundle": {"bundle_id": BUNDLE, "candidate_count": 1},
        "cutoff_date": "2026-09-15",
        "candidates": [
            {
                "candidate_id": "organizer-001",
                "search_terms": ["quantum error correction"],
                "searches": [
                    {
                        "search_id": "search-c1-oa",
                        "connector": "openalex",
                        "requests": [
                            {"request_id": "c1-oa-en", "search_text": "q"},
                            {"request_id": "c1-oa-ru", "search_text": "q"},
                        ],
                    },
                    {
                        "search_id": "search-c1-mc",
                        "connector": "mediacloud",
                        "requests": [{"request_id": "c1-mc", "search_text": "q"}],
                    },
                ],
            }
        ],
    }


def doc(number: int, **overrides) -> dict:
    payload = {
        "candidate_id": "organizer-001",
        "document_id": f"document-test-{number:03d}",
        "connector": "openalex",
        "search_id": "search-c1-oa",
        "request_id": "c1-oa-en",
        "title": "Quantum error correction with surface codes",
        "excerpt": "We study quantum error correction in detail.",
        "published_at": "2025-03-01",
        "origin_id": f"doi:10.1/test-{number:03d}",
        "publisher": "Test Journal",
        "trust_tier": "A",
        "url": f"https://example.org/test-{number:03d}",
        "canonical_url": f"https://example.org/test-{number:03d}",
        "doi": f"10.1/test-{number:03d}",
        "external_id": f"W{number:07d}",
    }
    payload.update(overrides)
    return payload


def coverage_row(source_class: str, planned: int, **overrides) -> dict:
    payload = {
        "candidate_id": "organizer-001",
        "source_class": source_class,
        "planned_requests": planned,
        "successful_requests": planned,
        "failed_requests": 0,
        "failed_request_ids": [],
        "status": "complete",
    }
    payload.update(overrides)
    return payload


def write_directories(
    root: Path,
    documents: list[dict],
    *,
    plan_mutate=None,
    doc_mutate=None,
    coverage_rows: list[dict] | None = None,
    manifest_totals: dict | None = None,
    plan_manifest_mutate=None,
) -> tuple[Path, Path]:
    plan_data = plan_payload()
    if plan_mutate is not None:
        plan_mutate(plan_data)
    plan_bytes = (
        json.dumps(plan_data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    plan_dir = root / "plan"
    plan_dir.mkdir(parents=True)
    (plan_dir / "plan.json").write_bytes(plan_bytes)
    plan_manifest = {
        "schema_version": "labeling-enrichment-plan-v2",
        "bundle_id": BUNDLE,
        "candidate_count": 1,
        "outputs": {
            "plan.json": {
                "sha256": hashlib.sha256(plan_bytes).hexdigest(),
                "size_bytes": len(plan_bytes),
            }
        },
    }
    if plan_manifest_mutate is not None:
        plan_manifest_mutate(plan_manifest)
    (plan_dir / "manifest.json").write_text(
        json.dumps(plan_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = []
    for item in documents:
        item = dict(item)
        if doc_mutate is not None:
            doc_mutate(item)
        rows.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
    documents_bytes = ("\n".join(rows) + "\n").encode("utf-8")
    coverage_payloads = (
        coverage_rows
        if coverage_rows is not None
        else [coverage_row("scientific", 2), coverage_row("industry", 1)]
    )
    coverage_bytes = (
        "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True)
            for row in coverage_payloads
        )
        + "\n"
    ).encode("utf-8")
    requests_bytes = b'{"request_id": "c1-oa-en"}\n'
    result_dir = root / "result"
    result_dir.mkdir(parents=True)
    (result_dir / "documents.jsonl").write_bytes(documents_bytes)
    (result_dir / "coverage.jsonl").write_bytes(coverage_bytes)
    (result_dir / "request_results.jsonl").write_bytes(requests_bytes)
    totals = {"candidates": 1, "returned_documents": len(rows)}
    if manifest_totals is not None:
        totals.update(manifest_totals)
    (result_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "labeling-enrichment-result-v2",
                "bundle_id": BUNDLE,
                "totals": totals,
                "plan": {
                    "sha256": hashlib.sha256(plan_bytes).hexdigest(),
                    "size_bytes": len(plan_bytes),
                },
                "outputs": {
                    "coverage.jsonl": {
                        "sha256": hashlib.sha256(coverage_bytes).hexdigest(),
                        "size_bytes": len(coverage_bytes),
                    },
                    "documents.jsonl": {
                        "sha256": hashlib.sha256(documents_bytes).hexdigest(),
                        "size_bytes": len(documents_bytes),
                    },
                    "request_results.jsonl": {
                        "sha256": hashlib.sha256(requests_bytes).hexdigest(),
                        "size_bytes": len(requests_bytes),
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return plan_dir, result_dir


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class ScoreLadderTests(unittest.TestCase):
    def test_exact_title_phrase_scores_100_strong(self) -> None:
        score, term, reason, matched, tokens = score_document(
            ("quantum error correction",),
            "Advances in Quantum Error Correction today",
            "Unrelated abstract about biology.",
        )

        self.assertEqual((score, reason), (100, "exact_title_phrase"))
        self.assertEqual(term, "quantum error correction")
        self.assertEqual(list(matched), ["correction", "error", "quantum"])
        self.assertEqual(list(tokens), ["correction", "error", "quantum"])

    def test_reordered_title_tokens_score_80_strong(self) -> None:
        score, _, reason, _, _ = score_document(
            ("quantum error correction",),
            "Correction methods for error prone quantum devices",
            "No useful abstract.",
        )

        self.assertEqual((score, reason), (80, "all_tokens_in_title"))

    def test_excerpt_only_phrase_scores_70_strong(self) -> None:
        score, _, reason, _, _ = score_document(
            ("quantum error correction",),
            "Surface codes for noisy devices",
            "This paper studies quantum error correction with cats.",
        )

        self.assertEqual((score, reason), (70, "exact_excerpt_phrase"))

    def test_partial_overlap_scores_weak(self) -> None:
        score, _, reason, matched, _ = score_document(
            ("quantum diagnostics",),
            "Quantum hardware benchmarks",
            "We benchmark processors and control lines.",
        )

        self.assertEqual(reason, "partial_token_overlap")
        self.assertEqual(score, round(40 * 1 / 2))
        self.assertEqual(list(matched), ["quantum"])
        self.assertGreaterEqual(score, 20)
        self.assertLess(score, 60)

    def test_no_token_overlap_scores_none(self) -> None:
        score, _, reason, matched, _ = score_document(
            ("quantum error correction",),
            "Baking sourdough at home",
            "Flour, water, salt and patience.",
        )

        self.assertEqual((score, reason), (0, "no_token_overlap"))
        self.assertEqual(list(matched), [])

    def test_matching_is_token_based_not_substring(self) -> None:
        score, _, reason, _, _ = score_document(
            ("quantum error correction",),
            "Leap in errorless baking methods",
            "A story about cats.",
        )

        self.assertEqual((score, reason), (0, "no_token_overlap"))

    def test_best_of_several_terms_is_deterministic(self) -> None:
        first = score_document(
            ("cats and dogs", "quantum error correction"),
            "Quantum error correction overview",
            None,
        )
        swapped = score_document(
            ("quantum error correction", "cats and dogs"),
            "Quantum error correction overview",
            None,
        )

        self.assertEqual(first[0], 100)
        self.assertEqual(first[1], "quantum error correction")
        self.assertEqual(swapped[1], "quantum error correction")

    def test_tie_keeps_first_plan_term(self) -> None:
        score, term, _, _, _ = score_document(
            ("alpha beta", "beta alpha"),
            "Alpha and beta particles",
            None,
        )

        self.assertEqual(score, 80)
        self.assertEqual(term, "alpha beta")

    def test_short_term_does_not_match_inside_longer_token(self) -> None:
        score, _, reason, _, _ = score_document(
            ("RAG",),
            "Data storage systems at scale",
            "A story about cats.",
        )

        self.assertEqual((score, reason), (0, "no_token_overlap"))

    def test_two_letter_term_does_not_match_inside_word(self) -> None:
        score, _, reason, _, _ = score_document(
            ("AI",),
            "Railway logistics overview",
            "A story about cats.",
        )

        self.assertEqual((score, reason), (0, "no_token_overlap"))

    def test_hyphenated_term_matches_spaced_tokens(self) -> None:
        score, _, reason, _, _ = score_document(
            ("edge-ai",),
            "Edge AI accelerators for inference",
            None,
        )

        self.assertEqual((score, reason), (100, "exact_title_phrase"))

    def test_scattered_phrase_words_score_80_not_100(self) -> None:
        scattered = score_document(
            ("retrieval augmented generation",),
            "Generation methods for augmented retrieval pipelines",
            None,
        )
        contiguous = score_document(
            ("retrieval augmented generation",),
            "Retrieval augmented generation at scale",
            None,
        )

        self.assertEqual(scattered[0], 80)
        self.assertEqual(scattered[2], "all_tokens_in_title")
        self.assertEqual(contiguous[0], 100)
        self.assertEqual(contiguous[2], "exact_title_phrase")


class PublishedDateTests(unittest.TestCase):
    def test_null_published_at_is_unknown_date_and_shortlistable(self) -> None:
        documents = [doc(1, published_at=None)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result = write_directories(root, documents)
            paths = export_relevance_plan(
                plan_dir=plan, result_dir=result, output_dir=root / "out"
            )
            ranked = read_jsonl(paths.ranked_documents)
            shortlist = read_jsonl(paths.scientific_shortlist)
            coverage = read_jsonl(paths.coverage)

        self.assertEqual(ranked[0]["published_at"], None)
        self.assertEqual(ranked[0]["time_window"], "unknown_date")
        self.assertEqual(len(shortlist), 1)
        self.assertIn("unknown_date", shortlist[0]["selection_reason"])
        self.assertEqual(coverage[0]["window_unknown_date"], 1)

    def test_published_after_cutoff_is_rejected_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result = write_directories(root, [doc(1, published_at="2026-09-16")])
            with self.assertRaisesRegex(ValueError, "past cutoff"):
                export_relevance_plan(
                    plan_dir=plan, result_dir=result, output_dir=root / "out"
                )
            self.assertFalse((root / "out").exists())


class CoverageGateTests(unittest.TestCase):
    def _export(self, root: Path, **kwargs) -> Path:
        plan, result = write_directories(root, [doc(1)], **kwargs)
        out = root / "out"
        export_relevance_plan(plan_dir=plan, result_dir=result, output_dir=out)
        return out

    def test_complete_scientific_and_industry_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = self._export(Path(directory))
            self.assertTrue((out / "ranked_documents.jsonl").is_file())

    def test_partial_scientific_is_rejected_without_output(self) -> None:
        rows = [
            coverage_row(
                "scientific", 2, successful_requests=1, failed_requests=1,
                failed_request_ids=["c1-oa-en"], status="partial",
            ),
            coverage_row("industry", 1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result = write_directories(root, [doc(1)], coverage_rows=rows)
            with self.assertRaisesRegex(ValueError, "organizer-001.*scientific"):
                export_relevance_plan(
                    plan_dir=plan, result_dir=result, output_dir=root / "out"
                )
            self.assertFalse((root / "out").exists())

    def test_unknown_industry_is_rejected_without_output(self) -> None:
        rows = [
            coverage_row("scientific", 2),
            coverage_row(
                "industry", 1, successful_requests=0, failed_requests=1,
                failed_request_ids=["c1-mc"], status="unknown",
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result = write_directories(root, [doc(1)], coverage_rows=rows)
            with self.assertRaisesRegex(ValueError, "organizer-001.*industry"):
                export_relevance_plan(
                    plan_dir=plan, result_dir=result, output_dir=root / "out"
                )
            self.assertFalse((root / "out").exists())

    def test_missing_coverage_row_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result = write_directories(
                root, [doc(1)], coverage_rows=[coverage_row("scientific", 2)]
            )
            with self.assertRaisesRegex(ValueError, "missing.*industry"):
                export_relevance_plan(
                    plan_dir=plan, result_dir=result, output_dir=root / "out"
                )
            self.assertFalse((root / "out").exists())

    def test_duplicate_coverage_pair_is_rejected(self) -> None:
        rows = [
            coverage_row("scientific", 2),
            coverage_row("scientific", 2),
            coverage_row("industry", 1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result = write_directories(root, [doc(1)], coverage_rows=rows)
            with self.assertRaisesRegex(ValueError, "duplicate coverage"):
                export_relevance_plan(
                    plan_dir=plan, result_dir=result, output_dir=root / "out"
                )
            self.assertFalse((root / "out").exists())

    def test_inconsistent_counts_are_rejected(self) -> None:
        rows = [
            coverage_row("scientific", 2, successful_requests=1, failed_requests=0),
            coverage_row("industry", 1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result = write_directories(root, [doc(1)], coverage_rows=rows)
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                export_relevance_plan(
                    plan_dir=plan, result_dir=result, output_dir=root / "out"
                )
            self.assertFalse((root / "out").exists())


class ManifestConsistencyTests(unittest.TestCase):
    def test_wrong_plan_candidate_count_is_rejected(self) -> None:
        def inflate(manifest: dict) -> None:
            manifest["candidate_count"] = 2

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result = write_directories(
                root, [doc(1)], plan_manifest_mutate=inflate
            )
            with self.assertRaisesRegex(ValueError, "candidate_count"):
                export_relevance_plan(
                    plan_dir=plan, result_dir=result, output_dir=root / "out"
                )
            self.assertFalse((root / "out").exists())

    def test_wrong_returned_documents_total_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result = write_directories(
                root, [doc(1)], manifest_totals={"returned_documents": 99}
            )
            with self.assertRaisesRegex(ValueError, "returned_documents"):
                export_relevance_plan(
                    plan_dir=plan, result_dir=result, output_dir=root / "out"
                )
            self.assertFalse((root / "out").exists())


class ShortlistSelectionTests(unittest.TestCase):
    def test_openalex_none_never_enters_scientific_shortlist(self) -> None:
        documents = [
            doc(1),
            doc(2, title="Baking sourdough", excerpt="Flour and water."),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result = write_directories(root, documents)
            paths = export_relevance_plan(
                plan_dir=plan, result_dir=result, output_dir=root / "out"
            )
            shortlist = read_jsonl(paths.scientific_shortlist)

        self.assertEqual(len(shortlist), 1)
        self.assertEqual(shortlist[0]["document_id"], "document-test-001")
        self.assertEqual(shortlist[0]["selection_rank"], 1)
        self.assertIn("strong", shortlist[0]["selection_reason"])

    def test_media_none_fills_free_fetch_slots(self) -> None:
        documents = [
            doc(
                1,
                connector="mediacloud",
                search_id="search-c1-mc",
                request_id="c1-mc",
                title="Quantum error correction milestone",
                excerpt=None,
                doi=None,
                origin_id="url:https://example.org/strong",
            ),
            doc(
                2,
                connector="mediacloud",
                search_id="search-c1-mc",
                request_id="c1-mc",
                title="Baking sourdough",
                excerpt=None,
                doi=None,
                origin_id="url:https://example.org/none",
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result = write_directories(root, documents)
            paths = export_relevance_plan(
                plan_dir=plan, result_dir=result, output_dir=root / "out"
            )
            queue = read_jsonl(paths.media_fetch_queue)

        by_id = {row["document_id"]: row for row in queue}
        self.assertEqual(len(queue), 2)
        self.assertEqual(by_id["document-test-002"]["relevance_class"], "none")
        self.assertEqual(by_id["document-test-002"]["selection_rank"], 2)

    def test_four_per_candidate_limit_holds(self) -> None:
        documents = [doc(n) for n in range(1, 8)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result = write_directories(root, documents)
            paths = export_relevance_plan(
                plan_dir=plan, result_dir=result, output_dir=root / "out"
            )
            shortlist = read_jsonl(paths.scientific_shortlist)

        self.assertEqual(len(shortlist), 4)
        self.assertEqual(
            [row["selection_rank"] for row in shortlist], [1, 2, 3, 4]
        )

    def test_time_windows_round_robin(self) -> None:
        documents = [
            doc(1, published_at="2024-10-01"),
            doc(2, published_at="2025-10-01"),
            doc(3, published_at="2026-01-01"),
            doc(4, published_at="2024-11-01"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result = write_directories(root, documents)
            paths = export_relevance_plan(
                plan_dir=plan, result_dir=result, output_dir=root / "out"
            )
            shortlist = read_jsonl(paths.scientific_shortlist)

        windows = [row["selection_reason"].split("|")[1] for row in shortlist]
        self.assertEqual(
            windows, ["previous", "recent", "previous", "recent"]
        )

    def test_origin_diversity_prefers_unused_origins(self) -> None:
        documents = [
            doc(1, origin_id="doi:10.1/shared"),
            doc(2, origin_id="doi:10.1/shared"),
            doc(3, origin_id="doi:10.1/other"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result = write_directories(root, documents)
            paths = export_relevance_plan(
                plan_dir=plan, result_dir=result, output_dir=root / "out"
            )
            shortlist = read_jsonl(paths.scientific_shortlist)

        ids = [row["document_id"] for row in shortlist]
        self.assertEqual(ids[0], "document-test-001")
        self.assertEqual(ids[1], "document-test-003")
        self.assertIn("new_origin", shortlist[1]["selection_reason"])

    def test_deduplication_by_doi_url_and_external_id(self) -> None:
        self.assertEqual(
            document_identity({"doi": " 10.1/ABC ", "connector": "openalex"}),
            "10.1/abc",
        )
        self.assertEqual(
            document_identity({
                "doi": None,
                "canonical_url": "  https://example.org/x  ",
                "connector": "openalex",
            }),
            "https://example.org/x",
        )
        self.assertEqual(
            document_identity({
                "doi": "",
                "canonical_url": "",
                "url": "",
                "connector": "openalex",
                "external_id": "W1",
            }),
            "openalex:W1",
        )
        documents = [
            doc(1),
            doc(2, title="Quantum error correction survey"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result = write_directories(root, documents, doc_mutate=None)
            # Force the same DOI on both rows.
            rows = [
                json.loads(line)
                for line in (result / "documents.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            for row in rows:
                row["doi"] = "10.1/shared"
                row["canonical_url"] = "https://example.org/shared"
            (result / "documents.jsonl").write_text(
                "\n".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows
                )
                + "\n",
                encoding="utf-8",
            )
            # Rebuild the result manifest digest for the edited fixture file.
            manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
            payload = (result / "documents.jsonl").read_bytes()
            manifest["outputs"]["documents.jsonl"] = {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            (result / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            paths = export_relevance_plan(
                plan_dir=plan, result_dir=result, output_dir=root / "out"
            )
            ranked = read_jsonl(paths.ranked_documents)

        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["document_id"], "document-test-001")
        self.assertEqual(ranked[0]["duplicate_document_ids"], ["document-test-002"])


class DeterminismAndSafetyTests(unittest.TestCase):
    def _export(self, root: Path, documents: list[dict]) -> Path:
        plan, result = write_directories(root, documents)
        out = root / "out"
        export_relevance_plan(plan_dir=plan, result_dir=result, output_dir=out)
        return out

    def test_permuted_input_gives_identical_data_files(self) -> None:
        documents = [doc(n) for n in range(1, 6)]
        with tempfile.TemporaryDirectory() as first:
            out_a = self._export(Path(first), documents)
            with tempfile.TemporaryDirectory() as second:
                out_b = self._export(Path(second), list(reversed(documents)))
                for name in (
                    "ranked_documents.jsonl",
                    "scientific_shortlist.jsonl",
                    "media_fetch_queue.jsonl",
                    "coverage.jsonl",
                ):
                    self.assertEqual(
                        (out_a / name).read_bytes(),
                        (out_b / name).read_bytes(),
                        name,
                    )

    def test_tampered_plan_is_rejected_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result = write_directories(root, [doc(1)])
            (plan / "plan.json").write_text(
                (plan / "plan.json").read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                export_relevance_plan(
                    plan_dir=plan, result_dir=result, output_dir=root / "out"
                )
            self.assertFalse((root / "out").exists())

    def test_unknown_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result = write_directories(
                root, [doc(1, candidate_id="ghost-999")]
            )
            with self.assertRaisesRegex(ValueError, "unknown candidate_id"):
                export_relevance_plan(
                    plan_dir=plan, result_dir=result, output_dir=root / "out"
                )
            self.assertFalse((root / "out").exists())

    def test_output_has_no_labels_secrets_or_headers(self) -> None:
        allowed_ranked = {
            "candidate_id", "document_id", "connector", "search_id",
            "request_id", "matched_term", "score", "relevance_class",
            "reasons", "matched_tokens", "term_tokens", "published_at",
            "origin_id", "publisher", "trust_tier", "url",
            "document_identity", "duplicate_document_ids", "time_window",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = self._export(root, [doc(1)])
            for name in (
                "ranked_documents.jsonl",
                "scientific_shortlist.jsonl",
                "media_fetch_queue.jsonl",
            ):
                for row in read_jsonl(out / name):
                    self.assertTrue(set(row) <= allowed_ranked | {
                        "selection_rank", "selection_reason",
                    })
                    blob = json.dumps(row, ensure_ascii=False).casefold()
                    for forbidden in (
                        "target", "expert", "label", "annotation",
                        "authorization", "bearer", "api-key", "api_key",
                    ):
                        self.assertNotIn(forbidden, blob)

    def test_cli_returns_1_without_partial_files_on_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, result = write_directories(root, [doc(1)])
            (result / "documents.jsonl").write_bytes(b"corrupt{{")
            exit_code = main([
                "labeling-relevance-plan",
                "--plan", str(plan),
                "--result", str(result),
                "--output", str(root / "out"),
            ])

        self.assertEqual(exit_code, 1)
        self.assertFalse((root / "out").exists())

    def test_version_constant(self) -> None:
        self.assertEqual(LABELING_RELEVANCE_PLAN_VERSION, "labeling-relevance-plan-v1")


if __name__ == "__main__":
    unittest.main()
