from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from nextwave.discovery import PRODUCT_GATE_ID
from nextwave.discovery.pipeline import DISCOVERY_PIPELINE_VERSION
from nextwave.discovery.run_store import DISCOVERY_RUN_MANIFEST_SCHEMA_VERSION
from nextwave.labeling.enrichment_plan import (
    LABELING_ENRICHMENT_PLAN_VERSION,
    build_analysis_enrichment_plan,
    export_analysis_enrichment_plan,
)
from nextwave.labeling.enrichment_run import load_validated_plan


def _digest(payload: bytes) -> dict[str, object]:
    return {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _run(root: Path, *, cutoff_date: str = "2026-09-15") -> Path:
    run = root / "run-001"
    run.mkdir()
    result = {
        "pipeline_version": DISCOVERY_PIPELINE_VERSION,
        "gate_coverage": {
            "status": "complete",
            "total_proposals": 3,
            "checked_proposals": 3,
            "skipped_proposals": 0,
            "skipped_proposal_ids": [],
        },
        "candidate_gate": {
            "gate_id": PRODUCT_GATE_ID,
            "input_proposal_ids": ["proposal-1", "proposal-2", "proposal-3"],
            "decisions": [
                {"proposal_id": "proposal-1", "decision": "accept"},
                {"proposal_id": "proposal-2", "decision": "accept"},
                {"proposal_id": "proposal-3", "decision": "reject"},
            ],
            "accepted_proposal_ids": ["proposal-1", "proposal-2"],
        },
        "candidate_proposals": {
            "proposals": [
                {"proposal_id": "proposal-1"},
                {"proposal_id": "proposal-2"},
                {"proposal_id": "proposal-3"},
            ]
        },
        "alias_resolution": {
            "input_proposal_ids": ["proposal-1", "proposal-2"],
            "groups": [
                {
                    "group_id": "alias-group-001",
                    "canonical_name": "Speculative decoding",
                    "aliases": [
                        "Speculative decoding",
                        "speculative-decoding",
                        "speculative-decoding",
                    ],
                    "proposal_ids": ["proposal-1"],
                },
                {
                    "group_id": "alias-group-002",
                    "canonical_name": "Paged attention",
                    "aliases": [],
                    "proposal_ids": ["proposal-2"],
                },
            ],
        },
    }
    result_bytes = json.dumps(result, sort_keys=True).encode()
    (run / "pipeline_result.json").write_bytes(result_bytes)
    plan = {
        "query": {"cutoff_date": cutoff_date},
        "scope": {
            "raw_query": "Инфраструктурные технологии для обучения и инференса ИИ"
        },
    }
    plan_bytes = json.dumps(plan, sort_keys=True).encode()
    (run / "plan.json").write_bytes(plan_bytes)
    manifest = {
        "schema_version": DISCOVERY_RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": "run-001",
        "analysis_status": "complete",
        "pipeline_version": DISCOVERY_PIPELINE_VERSION,
        "gate_id": PRODUCT_GATE_ID,
        "cutoff_date": cutoff_date,
        "domain": "Инфраструктура ИИ",
        "analysis_scope_key": "ai-infrastructure-v1",
        "raw_query": "Инфраструктурные технологии для обучения и инференса ИИ",
        "outputs": [
            {"filename": "plan.json", **_digest(plan_bytes)},
            {"filename": "pipeline_result.json", **_digest(result_bytes)},
        ],
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run


class AnalysisEnrichmentPlanTests(unittest.TestCase):
    def test_historical_run_uses_its_own_cutoff_and_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = export_analysis_enrichment_plan(
                run_dir=_run(root, cutoff_date="2025-09-15"),
                output_dir=root / "output",
            )
            plan_bytes = paths.plan.read_bytes()
            validated, _, _ = load_validated_plan(root / "output")
        plan = json.loads(plan_bytes)
        self.assertEqual(validated, plan)
        self.assertEqual(plan["cutoff_date"], "2025-09-15")
        self.assertEqual(plan["history_from"], "2023-09-16")
        self.assertEqual(plan["recent_window_from"], "2024-09-15")
        for candidate in plan["candidates"]:
            self.assertEqual(candidate["cutoff_date"], "2025-09-15")
            for search in candidate["searches"]:
                self.assertEqual(
                    search["planned_window"],
                    {"from": "2023-09-16", "until": "2025-09-15"},
                )

    def test_labeling_cutoff_keeps_existing_plan_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, _ = build_analysis_enrichment_plan(_run(root))
            second, _ = build_analysis_enrichment_plan(root / "run-001")
        self.assertEqual(first, second)

    def test_builds_query_specific_plan_from_accepted_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan_bytes, provenance = build_analysis_enrichment_plan(
                _run(Path(temporary))
            )
        plan = json.loads(plan_bytes)
        self.assertEqual(plan["schema_version"], LABELING_ENRICHMENT_PLAN_VERSION)
        self.assertEqual(plan["totals"]["candidates"], 2)
        self.assertEqual(plan["totals"]["openalex_primary_requests"], 6)
        self.assertEqual(plan["totals"]["mediacloud_primary_requests"], 3)
        self.assertEqual(
            {row["analysis_scope_key"] for row in plan["candidates"]},
            {"ai-infrastructure-v1"},
        )
        self.assertEqual(provenance["candidate_count"], 2)
        self.assertEqual(
            plan["candidates"][0]["search_terms"],
            ["Speculative decoding", "speculative-decoding"],
        )

    def test_user_query_scope_does_not_require_an_organizer_domain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = _run(Path(temporary))
            manifest_path = run / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["domain"] = "Произвольная пользовательская область"
            manifest["analysis_scope_key"] = "scope-query-123"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            plan_bytes, _ = build_analysis_enrichment_plan(run)
        plan = json.loads(plan_bytes)
        self.assertEqual(
            {row["analysis_scope_key"] for row in plan["candidates"]},
            {"scope-query-123"},
        )

    def test_export_marks_analysis_candidate_role(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = export_analysis_enrichment_plan(
                run_dir=_run(root), output_dir=root / "output"
            )
            manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        self.assertEqual(manifest["plan_role"], "analysis_candidates")
        self.assertEqual(manifest["candidate_count"], 2)

    def test_product_analysis_accepts_bounded_gate_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _run(root)
            result_path = run / "pipeline_result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["gate_coverage"]["status"] = "partial"
            result["gate_coverage"]["total_proposals"] = 4
            result["gate_coverage"]["skipped_proposals"] = 1
            result["gate_coverage"]["skipped_proposal_ids"] = ["proposal-4"]
            result["candidate_proposals"]["proposals"].append(
                {"proposal_id": "proposal-4"}
            )
            payload = json.dumps(result, sort_keys=True).encode()
            result_path.write_bytes(payload)
            manifest_path = run / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["analysis_status"] = "partial"
            manifest["outputs"][1] = {
                "filename": "pipeline_result.json",
                **_digest(payload),
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output = root / "output"
            paths = export_analysis_enrichment_plan(run_dir=run, output_dir=output)
            exported = json.loads(paths.manifest.read_text(encoding="utf-8"))
            self.assertEqual(
                exported["inputs"]["discovery_run"]["gate_coverage"],
                {
                    "status": "partial",
                    "checked_proposals": 3,
                    "total_proposals": 4,
                    "skipped_proposals": 1,
                },
            )

    def test_rejects_alias_groups_that_do_not_partition_accepts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _run(root)
            result_path = run / "pipeline_result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["alias_resolution"]["groups"][1]["proposal_ids"] = ["proposal-1"]
            payload = json.dumps(result, sort_keys=True).encode()
            result_path.write_bytes(payload)
            manifest_path = run / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["outputs"][1] = {
                "filename": "pipeline_result.json",
                **_digest(payload),
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "partition accepted proposals"):
                build_analysis_enrichment_plan(run)

    def test_rejects_tampered_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = _run(Path(temporary))
            with (run / "pipeline_result.json").open("ab") as stream:
                stream.write(b" ")
            with self.assertRaisesRegex(ValueError, "artifact size mismatch"):
                build_analysis_enrichment_plan(run)

    def test_rejects_gate_accept_list_that_differs_from_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _run(root)
            result_path = run / "pipeline_result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["candidate_gate"]["accepted_proposal_ids"] = ["proposal-1"]
            payload = json.dumps(result, sort_keys=True).encode()
            result_path.write_bytes(payload)
            manifest_path = run / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["outputs"][1] = {
                "filename": "pipeline_result.json",
                **_digest(payload),
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differ from Gate decisions"):
                build_analysis_enrichment_plan(run)

    def test_rejects_raw_query_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = _run(root)
            manifest_path = run / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["raw_query"] = "Другой запрос"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "raw_query differs"):
                build_analysis_enrichment_plan(run)


if __name__ == "__main__":
    unittest.main()
