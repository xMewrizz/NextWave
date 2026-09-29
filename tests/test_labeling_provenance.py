"""Provenance control: only v10 / qualification Pro 5 Gate runs enter labeling."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from nextwave.__main__ import main
from nextwave.discovery.candidate_gate import (
    CANDIDATE_GATE_VERSION,
    QUALIFICATION_GATE_ID,
    QUALIFICATION_GATE_VERSION,
)
from nextwave.discovery.pipeline import DISCOVERY_PIPELINE_VERSION
from nextwave.discovery.run_store import DiscoveryRun, assert_run_labeling_eligible

CUTOFF = "2026-09-15"
CURRENT_GATE_ID = QUALIFICATION_GATE_ID
LITE_GATE_ID = "yandex-yandexgpt-lite-5-candidate-gate-v4"
PRO51_GATE_ID = "yandex-yandexgpt-pro-5-1-candidate-gate-v4"
OLD_GATE_ID = "yandex-yandexgpt-lite-5-candidate-gate-v1"


def provenance_run(
    run_id: str,
    *,
    pipeline_version: str | None = DISCOVERY_PIPELINE_VERSION,
    gate_id: str | None = CURRENT_GATE_ID,
    coverage_status: str | None = "complete",
    skipped_proposals: int = 0,
    skipped_ids: list[str] | None = None,
    cutoff: str = CUTOFF,
    plan_cutoff: str | None = None,
    analysis_status: str = "complete",
    manifest_eligible: bool = True,
    manifest_pipeline_version: str | None = "__default__",
    manifest_gate_id: str | None = "__default__",
) -> DiscoveryRun:
    plan_cutoff_value = CUTOFF if plan_cutoff is None else plan_cutoff
    plan: dict = {
        "plan_id": f"plan-{run_id}",
        "analysis_id": f"analysis-{run_id}",
        "scope": {"scope_id": f"scope-{run_id}", "raw_query": "Технологии в ИИ"},
    }
    if plan_cutoff_value is not None:
        plan["query"] = {"cutoff_date": plan_cutoff_value}
    proposal_id = f"proposal-{run_id}"
    proposals = [{"proposal_id": proposal_id}]
    input_ids = [proposal_id] if skipped_proposals == 0 else []
    total = len(proposals)
    checked = len(input_ids)
    coverage: dict | None = None
    if coverage_status is not None:
        ids = skipped_ids if skipped_ids is not None else (
            [] if skipped_proposals == 0 else [proposal_id]
        )
        coverage = {
            "status": coverage_status,
            "total_proposals": total,
            "checked_proposals": checked,
            "skipped_proposals": skipped_proposals,
            "skipped_proposal_ids": ids,
        }
    result: dict = {
        "candidate_proposals": {"proposals": proposals, "exclusions": []},
        "candidate_gate": {"gate_id": gate_id, "input_proposal_ids": input_ids, "decisions": []},
        "alias_resolution": {"groups": [], "review_suggestions": []},
        "evidence_extraction": {"proposals": []},
        "text_extraction": {"issues": []},
        "scientific": {"status": "complete", "documents": []},
        "media": {"status": "failed", "documents": []},
        "verification": {"results": []},
    }
    if pipeline_version is not None:
        result["pipeline_version"] = pipeline_version
    if coverage is not None:
        result["gate_coverage"] = coverage
    if manifest_pipeline_version == "__default__":
        manifest_pipeline = pipeline_version
    else:
        manifest_pipeline = manifest_pipeline_version
    manifest_gate = gate_id if manifest_gate_id == "__default__" else manifest_gate_id
    manifest: dict = {
        "run_id": run_id,
        "cutoff_date": cutoff,
        "domain": "Edge",
        "analysis_status": analysis_status,
        "labeling_eligible": manifest_eligible,
        "counts": {},
    }
    if manifest_pipeline is not None:
        manifest["pipeline_version"] = manifest_pipeline
    if manifest_gate is not None:
        manifest["gate_id"] = manifest_gate
    return DiscoveryRun(
        run_id=run_id, run_dir=Path(run_id), plan=plan, result=result, manifest=manifest
    )


def pretty(payload: dict) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def write_provenance_dir(
    root: Path,
    run_id: str,
    *,
    pipeline_version: str | None = DISCOVERY_PIPELINE_VERSION,
    gate_id: str | None = CURRENT_GATE_ID,
    coverage_status: str | None = "complete",
    skipped_proposals: int = 0,
    analysis_status: str = "complete",
    manifest_eligible: bool = True,
    cutoff: str = CUTOFF,
    plan_cutoff: str | None = None,
    manifest_pipeline_version: str | None = "__default__",
    manifest_gate_id: str | None = "__default__",
) -> Path:
    run = provenance_run(
        run_id,
        pipeline_version=pipeline_version,
        gate_id=gate_id,
        coverage_status=coverage_status,
        skipped_proposals=skipped_proposals,
        analysis_status=analysis_status,
        manifest_eligible=manifest_eligible,
        cutoff=cutoff,
        plan_cutoff=plan_cutoff,
        manifest_pipeline_version=manifest_pipeline_version,
        manifest_gate_id=manifest_gate_id,
    )
    plan_bytes = pretty(run.plan)
    result_bytes = pretty(run.result)

    def digest(data: bytes) -> dict:
        return {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    manifest = dict(run.manifest)
    manifest["outputs"] = [
        {"filename": "plan.json", **digest(plan_bytes)},
        {"filename": "pipeline_result.json", **digest(result_bytes)},
    ]
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "plan.json").write_bytes(plan_bytes)
    (run_dir / "pipeline_result.json").write_bytes(result_bytes)
    (run_dir / "manifest.json").write_bytes(pretty(manifest))
    return run_dir


class ProvenanceEligibilityTests(unittest.TestCase):
    def test_current_v10_gate_v4_is_accepted(self) -> None:
        run = provenance_run("run-new")
        assert_run_labeling_eligible(run)
        self.assertEqual(DISCOVERY_PIPELINE_VERSION, "discovery-pipeline-v10")
        self.assertEqual(QUALIFICATION_GATE_VERSION, "candidate-gate-v4")
        self.assertEqual(CANDIDATE_GATE_VERSION, "candidate-gate-v6")

    def test_old_gate_v1_is_rejected(self) -> None:
        run = provenance_run("run-old-gate", gate_id=OLD_GATE_ID)
        with self.assertRaisesRegex(ValueError, "qualification gate"):
            assert_run_labeling_eligible(run)

    def test_lite_gate_v4_is_rejected(self) -> None:
        run = provenance_run("run-lite-gate", gate_id=LITE_GATE_ID)
        self.assertEqual(run.manifest["gate_id"], LITE_GATE_ID)
        with self.assertRaisesRegex(ValueError, "qualification gate"):
            assert_run_labeling_eligible(run)

    def test_pro51_gate_v4_is_rejected(self) -> None:
        run = provenance_run("run-pro51-gate", gate_id=PRO51_GATE_ID)
        self.assertEqual(run.manifest["gate_id"], PRO51_GATE_ID)
        with self.assertRaisesRegex(ValueError, "qualification gate"):
            assert_run_labeling_eligible(run)

    def test_old_pipeline_version_is_rejected(self) -> None:
        for old in ("discovery-pipeline-v6", "discovery-pipeline-v8", "discovery-pipeline-v9"):
            with self.subTest(old=old):
                run = provenance_run("run-old-pipe", pipeline_version=old)
                with self.assertRaisesRegex(ValueError, "discovery-pipeline-v10"):
                    assert_run_labeling_eligible(run)

    def test_partial_gate_coverage_is_rejected(self) -> None:
        run = provenance_run(
            "run-partial",
            coverage_status="partial",
            skipped_proposals=1,
            analysis_status="partial",
        )
        with self.assertRaisesRegex(ValueError, "coverage is 'partial'"):
            assert_run_labeling_eligible(run)

    def test_saved_eligible_flag_does_not_bypass_result_check(self) -> None:
        run = provenance_run(
            "run-fake-eligible",
            pipeline_version="discovery-pipeline-v8",
            gate_id=OLD_GATE_ID,
            manifest_eligible=True,
            manifest_pipeline_version="discovery-pipeline-v8",
            manifest_gate_id=OLD_GATE_ID,
        )
        self.assertTrue(run.manifest["labeling_eligible"])
        with self.assertRaises(ValueError):
            assert_run_labeling_eligible(run)

    def test_missing_skipped_proposals_is_rejected(self) -> None:
        run = provenance_run("run-no-skipped")
        del run.result["gate_coverage"]["skipped_proposals"]
        with self.assertRaisesRegex(ValueError, "skipped_proposals.*missing"):
            assert_run_labeling_eligible(run)

    def test_wrong_type_skipped_ids_is_rejected(self) -> None:
        run = provenance_run("run-bad-ids")
        run.result["gate_coverage"]["skipped_proposal_ids"] = "p1"
        with self.assertRaisesRegex(ValueError, "skipped_proposal_ids.*list"):
            assert_run_labeling_eligible(run)

    def test_total_checked_mismatch_is_rejected(self) -> None:
        run = provenance_run("run-bad-math")
        run.result["gate_coverage"]["checked_proposals"] = 0
        with self.assertRaisesRegex(ValueError, "total.*checked.*skipped"):
            assert_run_labeling_eligible(run)

    def test_proposals_count_mismatch_is_rejected(self) -> None:
        run = provenance_run("run-bad-count")
        run.result["candidate_proposals"]["proposals"].append({"proposal_id": "p-extra"})
        with self.assertRaisesRegex(ValueError, "proposals"):
            assert_run_labeling_eligible(run)

    def test_manifest_result_pipeline_mismatch_is_rejected(self) -> None:
        run = provenance_run(
            "run-pipe-mismatch",
            manifest_pipeline_version="discovery-pipeline-v8",
        )
        with self.assertRaisesRegex(ValueError, "pipeline version mismatch"):
            assert_run_labeling_eligible(run)

    def test_manifest_result_gate_mismatch_is_rejected(self) -> None:
        run = provenance_run("run-gate-mismatch", manifest_gate_id=OLD_GATE_ID)
        with self.assertRaisesRegex(ValueError, "gate id mismatch"):
            assert_run_labeling_eligible(run)

    def test_manifest_plan_cutoff_mismatch_is_rejected(self) -> None:
        run = provenance_run("run-cutoff-mismatch", plan_cutoff="2026-09-16")
        with self.assertRaisesRegex(ValueError, "plan cutoff"):
            assert_run_labeling_eligible(run)

    def test_fake_gate_suffix_is_rejected(self) -> None:
        fake = "fakecandidate-gate-v4"
        self.assertTrue(fake.endswith(QUALIFICATION_GATE_VERSION))
        self.assertFalse(fake.endswith("-" + QUALIFICATION_GATE_VERSION))
        run = provenance_run(
            "run-fake-gate",
            gate_id=fake,
            manifest_gate_id=fake,
        )
        with self.assertRaisesRegex(ValueError, "candidate-gate-v4"):
            assert_run_labeling_eligible(run)

    def test_mixed_old_and_new_creates_no_export_files(self) -> None:
        template = Path("templates/labeling_workbook.xlsx")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            new_dir = write_provenance_dir(root, "run-new")
            old_dir = write_provenance_dir(
                root, "run-old", pipeline_version="discovery-pipeline-v8", gate_id=OLD_GATE_ID
            )
            output = root / "export-mixed"
            with redirect_stderr(StringIO()):
                exit_code = main(
                    [
                        "labeling-export",
                        "--runs",
                        str(new_dir),
                        str(old_dir),
                        "--template",
                        str(template),
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(exit_code, 1)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob("export-mixed*")), [])


if __name__ == "__main__":
    unittest.main()
