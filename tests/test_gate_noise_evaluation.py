from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from nextwave.discovery import QUALIFICATION_GATE_ID
from nextwave.evaluation.gate_noise import (
    GATE_NOISE_EVALUATION_VERSION,
    build_gate_noise_evaluation,
    export_gate_noise_evaluation,
)

TYPES = (
    "broad_concept",
    "irrelevant",
    "not_technology",
    "extraction_error",
    "duplicate",
)


def _digest(payload: bytes) -> dict[str, object]:
    return {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _fixture(root: Path, *, one_unobserved: bool = False) -> tuple[Path, Path]:
    discovery = root / "discovery"
    entries = []
    for number in range(1, 51):
        noise_type = TYPES[(number - 1) // 10]
        run_id = f"run-{number:03d}"
        if noise_type in {"broad_concept", "irrelevant", "not_technology"}:
            origin = "gate_reject"
            if one_unobserved and number == 1:
                origin = "exclusion"
        elif noise_type == "extraction_error":
            origin = "extraction_issue"
        else:
            origin = "candidate_duplicate"
        entries.append(
            {
                "noise_id": f"noise-{number:03d}",
                "planned_noise_type": noise_type,
                "origin_kind": origin,
                "run_id": run_id,
                "extracted_text": f"Example {number}",
                "source_document_url": f"https://example.org/{number}",
                "duplicate_of_candidate_id": None,
                "rationale": "reviewed control",
            }
        )
        if origin == "gate_reject":
            run = discovery / run_id
            run.mkdir(parents=True)
            result = {
                "candidate_proposals": {
                    "proposals": [
                        {
                            "proposal_id": f"proposal-{number:03d}",
                            "canonical_name": f"Example {number}",
                            "aliases": [],
                        }
                    ]
                },
                "candidate_gate": {
                    "decisions": [
                        {
                            "proposal_id": f"proposal-{number:03d}",
                            "decision": "reject",
                            "reason": "generic_area",
                        }
                    ]
                },
            }
            result_bytes = json.dumps(result, sort_keys=True).encode()
            (run / "pipeline_result.json").write_bytes(result_bytes)
            manifest = {
                "run_id": run_id,
                "pipeline_version": "discovery-pipeline-v10",
                "gate_id": QUALIFICATION_GATE_ID,
                "outputs": [{"filename": "pipeline_result.json", **_digest(result_bytes)}],
            }
            (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    selection = {
        "schema_version": "labeling-noise-selection-v1",
        "cutoff_date": "2026-09-15",
        "entries": entries,
    }
    selection_path = root / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    return selection_path, discovery


class GateNoiseEvaluationTests(unittest.TestCase):
    def test_reports_complete_gate_and_pipeline_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            selection, discovery = _fixture(Path(temporary))
            rows, report, manifest = build_gate_noise_evaluation(
                selection_path=selection, discovery_root=discovery
            )
            values = json.loads(report)
            self.assertEqual(values["status"], "complete")
            self.assertEqual(values["totals"]["gate_expected"], 30)
            self.assertEqual(values["totals"]["gate_reject"], 30)
            self.assertEqual(values["metrics"]["observed_false_accept_rate"], 0.0)
            self.assertEqual(values["metrics"]["end_to_end_containment_rate"], 1.0)
            self.assertEqual(len(rows.decode().splitlines()), 50)
            self.assertEqual(json.loads(manifest)["schema_version"], GATE_NOISE_EVALUATION_VERSION)

    def test_unobserved_gate_control_makes_report_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            selection, discovery = _fixture(Path(temporary), one_unobserved=True)
            _, report, _ = build_gate_noise_evaluation(
                selection_path=selection, discovery_root=discovery
            )
            values = json.loads(report)
            self.assertEqual(values["status"], "partial")
            self.assertEqual(values["totals"]["gate_observed"], 29)
            self.assertEqual(values["totals"]["gate_unobserved"], 1)
            self.assertEqual(values["metrics"]["gate_coverage"], 0.966667)

    def test_accept_is_counted_as_false_accept(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, discovery = _fixture(root)
            path = discovery / "run-001" / "pipeline_result.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            result["candidate_gate"]["decisions"][0]["decision"] = "accept"
            payload = json.dumps(result, sort_keys=True).encode()
            path.write_bytes(payload)
            manifest_path = path.parent / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["outputs"][0] = {"filename": "pipeline_result.json", **_digest(payload)}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            _, report, _ = build_gate_noise_evaluation(
                selection_path=selection, discovery_root=discovery
            )
            values = json.loads(report)
            self.assertEqual(values["totals"]["gate_accept"], 1)
            self.assertEqual(values["metrics"]["observed_false_accept_rate"], 0.033333)

    def test_rejects_tampered_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, discovery = _fixture(root)
            with (discovery / "run-001" / "pipeline_result.json").open("ab") as stream:
                stream.write(b" ")
            with self.assertRaisesRegex(ValueError, "diverges from manifest"):
                build_gate_noise_evaluation(selection_path=selection, discovery_root=discovery)

    def test_rejects_wrong_quota_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, discovery = _fixture(root)
            value = json.loads(selection.read_text(encoding="utf-8"))
            value["entries"][0]["planned_noise_type"] = "irrelevant"
            selection.write_text(json.dumps(value), encoding="utf-8")
            output = root / "output"
            with self.assertRaisesRegex(ValueError, "five exact 10-row quotas"):
                export_gate_noise_evaluation(
                    selection_path=selection,
                    discovery_root=discovery,
                    output_dir=output,
                )
            self.assertFalse(output.exists())

    def test_export_is_byte_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection, discovery = _fixture(root)
            first = build_gate_noise_evaluation(selection_path=selection, discovery_root=discovery)
            second = build_gate_noise_evaluation(selection_path=selection, discovery_root=discovery)
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
