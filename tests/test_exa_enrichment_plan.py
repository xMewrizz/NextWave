from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from nextwave.evaluation.exa_enrichment_plan import (
    EXA_ENRICHMENT_PLAN_VERSION,
    build_exa_enrichment_plan,
    export_exa_enrichment_plan,
)


def digest(data: bytes):
    return {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def fixture(root: Path):
    plan = {
        "schema_version": "labeling-enrichment-plan-v2",
        "cutoff_date": "2026-09-15",
        "windows": {
            "previous": {"from": "2024-09-15", "until": "2025-09-15"},
            "recent": {"from": "2025-09-15", "until": "2026-09-15"},
        },
        "bundle": {"bundle_id": "bundle-1", "candidate_count": 2},
        "candidates": [
            {"candidate_id": "candidate-b", "search_terms": ["Term B"]},
            {"candidate_id": "candidate-a", "search_terms": ["Term A", "Alias A"]},
        ],
    }
    raw = (json.dumps(plan, sort_keys=True) + "\n").encode()
    root.mkdir()
    (root / "plan.json").write_bytes(raw)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "labeling-enrichment-plan-v2",
                "plan_role": "analysis_candidates",
                "outputs": {"plan.json": digest(raw)},
            }
        ),
        encoding="utf-8",
    )


class ExaEnrichmentPlanTests(unittest.TestCase):
    def test_plans_every_term_in_both_windows_without_candidate_pruning(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            fixture(source)
            raw, provenance = build_exa_enrichment_plan(source)
        plan = json.loads(raw)
        self.assertEqual(plan["schema_version"], EXA_ENRICHMENT_PLAN_VERSION)
        self.assertEqual(plan["totals"], {"candidates": 2, "requests": 6})
        self.assertFalse(plan["policy"]["candidate_pruning"])
        self.assertEqual({item["window"] for item in plan["tasks"]}, {"previous", "recent"})
        self.assertEqual(provenance["candidate_count"], 2)

    def test_previous_window_excludes_recent_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            fixture(source)
            raw, _ = build_exa_enrichment_plan(source)
        task = next(item for item in json.loads(raw)["tasks"] if item["window"] == "previous")
        parameters = {item["name"]: item["value"] for item in task["request"]["parameters"]}
        self.assertEqual(parameters["endPublishedDate"], "2025-09-14T23:59:59.999Z")

    def test_permuted_candidates_produce_same_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            first = Path(temp) / "first"
            second = Path(temp) / "second"
            fixture(first)
            fixture(second)
            plan = json.loads((second / "plan.json").read_text())
            plan["candidates"].reverse()
            raw = (json.dumps(plan, sort_keys=True) + "\n").encode()
            (second / "plan.json").write_bytes(raw)
            manifest = json.loads((second / "manifest.json").read_text())
            manifest["outputs"]["plan.json"] = digest(raw)
            (second / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            left, _ = build_exa_enrichment_plan(first)
            right, _ = build_exa_enrichment_plan(second)
        self.assertEqual(left, right)

    def test_bad_checksum_creates_no_output(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            output = Path(temp) / "output"
            fixture(source)
            (source / "plan.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "version|checksum"):
                export_exa_enrichment_plan(analysis_plan_dir=source, output_dir=output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
