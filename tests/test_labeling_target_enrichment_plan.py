from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nextwave.datasets import OrganizerArtifactError
from nextwave.labeling.enrichment_plan import (
    LABELING_ENRICHMENT_PLAN_VERSION,
    LABELING_TARGET_ENRICHMENT_PLAN_VERSION,
    TARGET_CANDIDATES_VERSION,
    build_target_enrichment_plan,
    export_target_enrichment_plan,
)


def candidate(number: int, **overrides) -> dict:
    value = {
        "candidate_id": f"target-{number:03d}",
        "canonical_name": f"Candidate technology {number}",
        "aliases": [f"Candidate tech {number}"],
        "domain": "Финтех",
        "analysis_scope_key": "fintech-v1",
        "source_query": "Emerging financial technologies",
    }
    value.update(overrides)
    return value


def write_spec(path: Path, candidates: list[dict]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": TARGET_CANDIDATES_VERSION,
                "cutoff_date": "2026-09-15",
                "candidates": candidates,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


class TargetEnrichmentPlanTests(unittest.TestCase):
    def test_builds_ordinary_enrichment_v2_without_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "targets.json"
            write_spec(path, [candidate(1)])
            raw, provenance = build_target_enrichment_plan(path)

        plan = json.loads(raw)
        self.assertEqual(plan["schema_version"], LABELING_ENRICHMENT_PLAN_VERSION)
        self.assertEqual(plan["totals"]["candidates"], 1)
        self.assertEqual(plan["totals"]["openalex_primary_requests"], 4)
        self.assertEqual(plan["totals"]["mediacloud_primary_requests"], 2)
        self.assertEqual(
            {item["source_class"] for item in plan["candidates"][0]["searches"]},
            {"scientific", "industry"},
        )
        candidate_payload = plan["candidates"][0]
        self.assertTrue(
            {"label", "target", "review_status", "evidence"}.isdisjoint(
                candidate_payload
            )
        )
        self.assertEqual(set(provenance["inputs"]), {"target_candidates"})

    def test_input_order_does_not_change_plan_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            values = [candidate(2), candidate(1)]
            write_spec(first, values)
            write_spec(second, list(reversed(values)))
            first_raw, _ = build_target_enrichment_plan(first)
            second_raw, _ = build_target_enrichment_plan(second)
        self.assertEqual(first_raw, second_raw)

    def test_rejects_label_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "targets.json"
            write_spec(path, [candidate(1, label="marketing_hype")])
            with self.assertRaisesRegex(ValueError, "unexpected or missing"):
                build_target_enrichment_plan(path)

    def test_rejects_target_and_review_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "targets.json"
            write_spec(path, [candidate(1, target=0, review_status="reviewed")])
            with self.assertRaisesRegex(ValueError, "unexpected or missing"):
                build_target_enrichment_plan(path)

    def test_rejects_wrong_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "targets.json"
            write_spec(path, [candidate(1)])
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["cutoff_date"] = "2026-09-16"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cutoff"):
                build_target_enrichment_plan(path)

    def test_rejects_unknown_domain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "targets.json"
            write_spec(path, [candidate(1, domain="Other")])
            with self.assertRaisesRegex(ValueError, "unsupported domain"):
                build_target_enrichment_plan(path)

    def test_rejects_duplicate_candidate_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "targets.json"
            write_spec(
                path,
                [candidate(1), candidate(2, candidate_id="target-001")],
            )
            with self.assertRaisesRegex(ValueError, "duplicate candidate_id"):
                build_target_enrichment_plan(path)

    def test_rejects_duplicate_domain_name_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "targets.json"
            write_spec(
                path,
                [candidate(1), candidate(2, canonical_name=" candidate TECHNOLOGY 1 ")],
            )
            with self.assertRaisesRegex(ValueError, "duplicate domain/name"):
                build_target_enrichment_plan(path)

    def test_export_writes_planner_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "targets.json"
            output = root / "output"
            write_spec(path, [candidate(1)])
            paths = export_target_enrichment_plan(
                candidates_file=path, output_dir=output
            )
            manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["planner_version"], LABELING_TARGET_ENRICHMENT_PLAN_VERSION
        )
        self.assertEqual(manifest["candidate_count"], 1)
        self.assertEqual(set(manifest["inputs"]), {"target_candidates"})

    def test_existing_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "targets.json"
            output = root / "output"
            write_spec(path, [candidate(1)])
            output.mkdir()
            marker = output / "marker.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaises(OrganizerArtifactError):
                export_target_enrichment_plan(
                    candidates_file=path, output_dir=output
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
