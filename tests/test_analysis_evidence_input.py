"""Query-local Evidence input tests: offline fixtures only, no network."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from nextwave.evaluation.analysis_evidence_input import (
    ANALYSIS_EVIDENCE_INPUT_VERSION,
    build_analysis_evidence_input,
    export_analysis_evidence_input,
)
from nextwave.labeling.evidence_input_plan import (
    LABELING_EVIDENCE_INPUT_PLAN_VERSION,
)

PLAN_VERSION = "labeling-enrichment-plan-v2"
COMBINED_VERSION = "analysis-combined-enrichment-result-v1"
SHORTLIST_VERSION = "analysis-evidence-shortlist-v1"
BUNDLE = "bundle-analysis-001"
TERMS = ["quantum error correction"]


def digest(payload: bytes) -> dict[str, Any]:
    return {
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def dump_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def sci_doc(candidate: str, number: int, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "candidate_id": candidate,
        "document_id": f"sci-{candidate}-{number}",
        "connector": "openalex",
        "title": "Quantum error correction with surface codes",
        "excerpt": (
            "Quantum error correction with surface codes improves devices "
            "in laboratory measurements."
        ),
        "origin_id": f"doi:10.1/{candidate}-s{number}",
        "snapshot_id": f"snap-s{number}",
        "url": f"https://example.org/{candidate}/s{number}",
        "trust_tier": "A",
        "published_at": "2026-06-01",
    }
    payload.update(overrides)
    return payload


def ind_doc(candidate: str, number: int, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "candidate_id": candidate,
        "document_id": f"ind-{candidate}-{number}",
        "connector": "exa",
        "title": "Quantum error correction startup raises funding",
        "excerpt": (
            "A quantum error correction startup announced pilot deployments "
            "with industrial partners."
        ),
        "origin_id": f"url:https://press.example/{candidate}-{number}",
        "snapshot_id": f"snap-i{number}",
        "url": f"https://press.example/{candidate}-{number}",
        "trust_tier": "C",
        "published_at": "2026-05-01",
    }
    payload.update(overrides)
    return payload


def low_score_doc(candidate: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate,
        "document_id": f"weak-{candidate}",
        "connector": "exa",
        "title": "Market overview",
        "excerpt": "Quantum markets overview today.",
        "origin_id": f"url:https://press.example/{candidate}-weak",
        "snapshot_id": "snap-weak",
        "url": f"https://press.example/{candidate}-weak",
        "trust_tier": "D",
        "published_at": "2026-04-01",
    }


def build_dirs(
    root: Path,
    candidates: list[str],
    documents: list[dict[str, Any]],
    shortlist: list[str] | None = None,
    *,
    shortlist_ranks: list[int] | None = None,
    coverage_rows: list[dict[str, Any]] | None = None,
) -> tuple[Path, Path, Path]:
    plan_dir = root / "plan"
    combined_dir = root / "combined"
    shortlist_dir = root / "shortlist"
    plan_dir.mkdir(parents=True)
    combined_dir.mkdir(parents=True)
    shortlist_dir.mkdir(parents=True)

    plan_payload = (
        json.dumps(
            {
                "cutoff_date": "2026-09-15",
                "candidates": [
                    {
                        "candidate_id": cid,
                        "cutoff_date": "2026-09-15",
                        "search_terms": TERMS,
                    }
                    for cid in candidates
                ]
            },
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    (plan_dir / "plan.json").write_bytes(plan_payload)
    (plan_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": PLAN_VERSION,
                "bundle_id": BUNDLE,
                "outputs": {"plan.json": digest(plan_payload)},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    documents_payload = dump_jsonl(documents)
    (combined_dir / "documents.jsonl").write_bytes(documents_payload)
    if coverage_rows is None:
        coverage_rows = [
            {
                "candidate_id": cid,
                "source_class": cls,
                "status": "complete",
            }
            for cid in candidates
            for cls in ("scientific", "industry")
        ]
    coverage_payload = dump_jsonl(coverage_rows)
    (combined_dir / "coverage.jsonl").write_bytes(coverage_payload)
    (combined_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": COMBINED_VERSION,
                "bundle_id": BUNDLE,
                "plan": digest(plan_payload),
                "outputs": {
                    "documents.jsonl": digest(documents_payload),
                    "coverage.jsonl": digest(coverage_payload),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    listed = shortlist if shortlist is not None else list(candidates)
    ranks = (
        shortlist_ranks
        if shortlist_ranks is not None
        else list(range(1, len(listed) + 1))
    )
    shortlist_payload = dump_jsonl([
        {"candidate_id": cid, "evidence_rank": rank}
        for cid, rank in zip(listed, ranks, strict=True)
    ])
    (shortlist_dir / "shortlist.jsonl").write_bytes(shortlist_payload)
    (shortlist_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": SHORTLIST_VERSION,
                "outputs": {"shortlist.jsonl": digest(shortlist_payload)},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return plan_dir, combined_dir, shortlist_dir


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class AnalysisEvidenceInputTests(unittest.TestCase):
    def test_valid_openalex_and_exa_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, combined, shortlist = build_dirs(
                root,
                ["c1"],
                [sci_doc("c1", 1), sci_doc("c1", 2), ind_doc("c1", 1)],
            )
            files = build_analysis_evidence_input(
                analysis_plan_dir=plan,
                combined_result_dir=combined,
                shortlist_dir=shortlist,
            )
            documents = [
                json.loads(line)
                for line in files["evidence_input_documents.jsonl"].splitlines()
            ]
            coverage = [
                json.loads(line)
                for line in files["coverage.jsonl"].splitlines()
            ]
            manifest = json.loads(files["manifest.json"])

        by_id = {row["document_id"]: row for row in documents}
        self.assertEqual(len(documents), 3)
        self.assertEqual(by_id["sci-c1-1"]["source_class"], "scientific")
        self.assertEqual(by_id["ind-c1-1"]["source_class"], "industry")
        self.assertEqual(by_id["ind-c1-1"]["connector"], "exa")
        self.assertEqual(
            [row["final_rank"] for row in documents], [1, 2, 3]
        )
        for row in documents:
            self.assertEqual(row["relevance_score"], 100)
            self.assertEqual(row["relevance_class"], "strong")
            self.assertTrue(row["evidence_text_available"])
            self.assertNotIn("_published", row)
        self.assertEqual(len(coverage), 1)
        self.assertEqual(coverage[0]["candidate_id"], "c1")
        self.assertTrue(coverage[0]["has_evidence_input"])
        self.assertEqual(coverage[0]["final_evidence_input_count"], 3)
        self.assertEqual(coverage[0]["scientific_selected"], 2)
        self.assertEqual(coverage[0]["media_selected"], 1)
        self.assertEqual(
            manifest["schema_version"], LABELING_EVIDENCE_INPUT_PLAN_VERSION
        )
        self.assertEqual(
            manifest["artifact_role"], ANALYSIS_EVIDENCE_INPUT_VERSION
        )
        self.assertEqual(manifest["cutoff_date"], "2026-09-15")
        policy = manifest["selection_policy"]
        self.assertTrue(policy["shortlist_only"])
        self.assertEqual(policy["minimum_relevance_score"], 40)
        self.assertEqual(policy["max_per_source_class"], 3)
        self.assertEqual(manifest["totals"]["evidence_input_documents"], 3)
        self.assertEqual(
            manifest["outputs"]["evidence_input_documents.jsonl"],
            digest(files["evidence_input_documents.jsonl"]),
        )
        self.assertEqual(
            manifest["outputs"]["coverage.jsonl"],
            digest(files["coverage.jsonl"]),
        )

    def test_shortlist_only_excludes_unlisted_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, combined, shortlist = build_dirs(
                root,
                ["c1", "c2"],
                [sci_doc("c1", 1), ind_doc("c1", 1), sci_doc("c2", 1)],
                shortlist=["c1"],
            )
            files = build_analysis_evidence_input(
                analysis_plan_dir=plan,
                combined_result_dir=combined,
                shortlist_dir=shortlist,
            )
            documents = [
                json.loads(line)
                for line in files["evidence_input_documents.jsonl"].splitlines()
            ]
            coverage = [
                json.loads(line)
                for line in files["coverage.jsonl"].splitlines()
            ]
            manifest = json.loads(files["manifest.json"])

        self.assertEqual(
            {row["candidate_id"] for row in documents}, {"c1"}
        )
        self.assertEqual([row["candidate_id"] for row in coverage], ["c1"])
        self.assertEqual(manifest["totals"]["candidates"], 1)

    def test_caps_three_per_class_and_six_total(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, combined, shortlist = build_dirs(
                root,
                ["c1"],
                [sci_doc("c1", n) for n in range(1, 6)]
                + [ind_doc("c1", n) for n in range(1, 6)],
            )
            files = build_analysis_evidence_input(
                analysis_plan_dir=plan,
                combined_result_dir=combined,
                shortlist_dir=shortlist,
            )
            documents = [
                json.loads(line)
                for line in files["evidence_input_documents.jsonl"].splitlines()
            ]
            coverage = [
                json.loads(line)
                for line in files["coverage.jsonl"].splitlines()
            ]

        classes = [row["source_class"] for row in documents]
        self.assertEqual(len(documents), 6)
        self.assertEqual(classes.count("scientific"), 3)
        self.assertEqual(classes.count("industry"), 3)
        self.assertEqual(
            [row["final_rank"] for row in documents], [1, 2, 3, 4, 5, 6]
        )
        self.assertEqual(coverage[0]["scientific_selected"], 3)
        self.assertEqual(coverage[0]["media_selected"], 3)
        self.assertEqual(coverage[0]["final_evidence_input_count"], 6)

    def test_duplicate_origin_dropped_within_class(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, combined, shortlist = build_dirs(
                root,
                ["c1"],
                [
                    sci_doc("c1", 1),
                    sci_doc("c1", 2, origin_id="doi:10.1/c1-s1"),
                    sci_doc("c1", 3),
                    sci_doc("c1", 4),
                ],
            )
            files = build_analysis_evidence_input(
                analysis_plan_dir=plan,
                combined_result_dir=combined,
                shortlist_dir=shortlist,
            )
            documents = [
                json.loads(line)
                for line in files["evidence_input_documents.jsonl"].splitlines()
            ]

        origins = [row["origin_id"] for row in documents]
        self.assertEqual(len(documents), 3)
        self.assertEqual(len(set(origins)), 3)

    def test_after_cutoff_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, combined, shortlist = build_dirs(
                root,
                ["c1"],
                [sci_doc("c1", 1, published_at="2026-09-16")],
            )
            with self.assertRaisesRegex(ValueError, "after cutoff"):
                build_analysis_evidence_input(
                    analysis_plan_dir=plan,
                    combined_result_dir=combined,
                    shortlist_dir=shortlist,
                )

    def test_below_min_score_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, combined, shortlist = build_dirs(
                root, ["c1"], [low_score_doc("c1")]
            )
            files = build_analysis_evidence_input(
                analysis_plan_dir=plan,
                combined_result_dir=combined,
                shortlist_dir=shortlist,
            )
            coverage = [
                json.loads(line)
                for line in files["coverage.jsonl"].splitlines()
            ]

        self.assertEqual(files["evidence_input_documents.jsonl"], b"")
        self.assertFalse(coverage[0]["has_evidence_input"])
        self.assertEqual(coverage[0]["final_evidence_input_count"], 0)
        self.assertEqual(
            coverage[0]["empty_reasons"], ["no_relevant_source_text"]
        )

    def test_tampered_checksum_is_rejected_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, combined, shortlist = build_dirs(
                root, ["c1"], [sci_doc("c1", 1)]
            )
            with (combined / "documents.jsonl").open("ab") as handle:
                handle.write(b" ")
            with self.assertRaisesRegex(ValueError, "differs from manifest"):
                export_analysis_evidence_input(
                    analysis_plan_dir=plan,
                    combined_result_dir=combined,
                    shortlist_dir=shortlist,
                    output_dir=root / "out",
                )
            self.assertFalse((root / "out").exists())

    def test_incomplete_coverage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, combined, shortlist = build_dirs(
                root,
                ["c1"],
                [sci_doc("c1", 1)],
                coverage_rows=[
                    {
                        "candidate_id": "c1",
                        "source_class": "scientific",
                        "status": "complete",
                    }
                ],
            )
            with self.assertRaisesRegex(
                ValueError, "coverage is not exactly complete"
            ):
                build_analysis_evidence_input(
                    analysis_plan_dir=plan,
                    combined_result_dir=combined,
                    shortlist_dir=shortlist,
                )

    def test_broken_shortlist_ranks_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, combined, shortlist = build_dirs(
                root,
                ["c1", "c2"],
                [sci_doc("c1", 1), sci_doc("c2", 1)],
                shortlist_ranks=[1, 3],
            )
            with self.assertRaisesRegex(ValueError, "ranks must form 1\\.\\.N"):
                build_analysis_evidence_input(
                    analysis_plan_dir=plan,
                    combined_result_dir=combined,
                    shortlist_dir=shortlist,
                )

    def test_boolean_shortlist_rank_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, combined, shortlist = build_dirs(
                root,
                ["c1"],
                [sci_doc("c1", 1)],
                shortlist_ranks=[True],
            )
            with self.assertRaisesRegex(ValueError, "plain integers"):
                build_analysis_evidence_input(
                    analysis_plan_dir=plan,
                    combined_result_dir=combined,
                    shortlist_dir=shortlist,
                )

    def test_non_http_document_url_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, combined, shortlist = build_dirs(
                root,
                ["c1"],
                [sci_doc("c1", 1, url="file:///tmp/source")],
            )
            with self.assertRaisesRegex(ValueError, "invalid URL"):
                build_analysis_evidence_input(
                    analysis_plan_dir=plan,
                    combined_result_dir=combined,
                    shortlist_dir=shortlist,
                )

    def test_blank_document_title_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, combined, shortlist = build_dirs(
                root,
                ["c1"],
                [sci_doc("c1", 1, title=" ")],
            )
            with self.assertRaisesRegex(ValueError, "has no title"):
                build_analysis_evidence_input(
                    analysis_plan_dir=plan,
                    combined_result_dir=combined,
                    shortlist_dir=shortlist,
                )

    def test_unknown_connector_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, combined, shortlist = build_dirs(
                root,
                ["c1"],
                [sci_doc("c1", 1, connector="mediacloud")],
            )
            with self.assertRaisesRegex(
                ValueError, "invalid identity or connector"
            ):
                build_analysis_evidence_input(
                    analysis_plan_dir=plan,
                    combined_result_dir=combined,
                    shortlist_dir=shortlist,
                )

    def test_permuted_documents_give_identical_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            plan, combined, shortlist = build_dirs(
                first,
                ["c1"],
                [sci_doc("c1", n) for n in range(1, 4)]
                + [ind_doc("c1", n) for n in range(1, 4)],
            )
            before = build_analysis_evidence_input(
                analysis_plan_dir=plan,
                combined_result_dir=combined,
                shortlist_dir=shortlist,
            )
            raw = (combined / "documents.jsonl").read_bytes().splitlines(
                keepends=True
            )
            (combined / "documents.jsonl").write_bytes(b"".join(reversed(raw)))
            # Manifest still matches: permutation keeps the same byte multiset
            # only when line order changes length-neutrally, so rebuild digest.
            manifest = json.loads(
                (combined / "manifest.json").read_text(encoding="utf-8")
            )
            changed = (combined / "documents.jsonl").read_bytes()
            manifest["outputs"]["documents.jsonl"] = digest(changed)
            (combined / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            after = build_analysis_evidence_input(
                analysis_plan_dir=plan,
                combined_result_dir=combined,
                shortlist_dir=shortlist,
            )

        self.assertEqual(before.keys(), after.keys())
        for name in (
            "evidence_input_documents.jsonl",
            "coverage.jsonl",
        ):
            self.assertEqual(before[name], after[name])

    def test_export_publishes_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan, combined, shortlist = build_dirs(
                root, ["c1"], [sci_doc("c1", 1), ind_doc("c1", 1)]
            )
            paths = export_analysis_evidence_input(
                analysis_plan_dir=plan,
                combined_result_dir=combined,
                shortlist_dir=shortlist,
                output_dir=root / "out",
            )
            documents = read_jsonl(paths.documents)
            manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))

            self.assertEqual(len(documents), 2)
            self.assertEqual(
                manifest["outputs"]["evidence_input_documents.jsonl"],
                digest(paths.documents.read_bytes()),
            )
            with self.assertRaises(ValueError):
                export_analysis_evidence_input(
                    analysis_plan_dir=plan,
                    combined_result_dir=combined,
                    shortlist_dir=shortlist,
                    output_dir=root / "out",
                )


if __name__ == "__main__":
    unittest.main()
