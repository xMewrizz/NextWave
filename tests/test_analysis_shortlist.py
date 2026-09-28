from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from nextwave.evaluation.analysis_shortlist import (
    ANALYSIS_SHORTLIST_VERSION,
    build_analysis_shortlist,
    export_analysis_shortlist,
)


def _bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()


def _digest(payload: bytes) -> dict[str, object]:
    return {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


class AnalysisShortlistTests(unittest.TestCase):
    def _fixture(self, root: Path, reverse: bool = False) -> Path:
        directory = root / ("reverse" if reverse else "inference")
        directory.mkdir()
        rows = [
            {
                "candidate_id": candidate_id,
                "canonical_name": name,
                "domain": "Edge",
                "analysis_scope_key": "edge-v1",
                "source_query": "Edge AI",
                "model_score": score,
                "decision_threshold": 0.5,
                "prediction": int(score >= 0.5),
            }
            for candidate_id, name, score in (
                ("candidate-001", "Beta", 0.9),
                ("candidate-002", "Alpha", 0.9),
                ("candidate-003", "Gamma", 0.4),
            )
        ]
        if reverse:
            rows.reverse()
        payload = b"".join(_bytes(row) for row in rows)
        (directory / "predictions.jsonl").write_bytes(payload)
        manifest = {
            "schema_version": "analysis-inference-v1",
            "release_status": "development_only",
            "candidate_count": 3,
            "outputs": {"predictions.jsonl": _digest(payload)},
        }
        (directory / "manifest.json").write_bytes(_bytes(manifest))
        return directory

    def test_selects_query_local_evidence_queue_by_frozen_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shortlist, manifest = build_analysis_shortlist(
                inference_dir=self._fixture(Path(tmp)), limit=2
            )
        rows = [json.loads(line) for line in shortlist.decode().splitlines()]
        summary = json.loads(manifest)
        self.assertEqual([row["candidate_id"] for row in rows], ["candidate-002", "candidate-001"])
        self.assertEqual([row["evidence_rank"] for row in rows], [1, 2])
        self.assertEqual([row["model_rank"] for row in rows], [1, 2])
        self.assertEqual(summary["schema_version"], ANALYSIS_SHORTLIST_VERSION)
        self.assertEqual(summary["artifact_role"], "evidence_work_queue_not_final_top15")

    def test_input_order_does_not_change_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = build_analysis_shortlist(inference_dir=self._fixture(root), limit=3)
            second = build_analysis_shortlist(
                inference_dir=self._fixture(root, reverse=True), limit=3
            )
        self.assertEqual(first[0], second[0])

    def test_offset_builds_next_evidence_page_without_losing_global_rank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shortlist, manifest = build_analysis_shortlist(
                inference_dir=self._fixture(Path(tmp)), limit=2, offset=1
            )
        rows = [json.loads(line) for line in shortlist.decode().splitlines()]
        summary = json.loads(manifest)
        self.assertEqual(
            [row["candidate_id"] for row in rows],
            ["candidate-001", "candidate-003"],
        )
        self.assertEqual([row["evidence_rank"] for row in rows], [1, 2])
        self.assertEqual([row["model_rank"] for row in rows], [2, 3])
        self.assertEqual(summary["shortlist_offset"], 1)

    def test_rejects_mixed_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = self._fixture(Path(tmp))
            path = directory / "predictions.jsonl"
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            rows[1]["source_query"] = "Robotics"
            payload = b"".join(_bytes(row) for row in rows)
            path.write_bytes(payload)
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["outputs"]["predictions.jsonl"] = _digest(payload)
            manifest_path.write_bytes(_bytes(manifest))
            with self.assertRaisesRegex(ValueError, "mix queries"):
                build_analysis_shortlist(inference_dir=directory)

    def test_export_refuses_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = self._fixture(root)
            output = root / "output"
            paths = export_analysis_shortlist(
                inference_dir=directory, output_dir=output, limit=2
            )
            self.assertTrue(paths.shortlist.is_file())
            with self.assertRaisesRegex(ValueError, "already exists"):
                export_analysis_shortlist(
                    inference_dir=directory, output_dir=output, limit=2
                )


if __name__ == "__main__":
    unittest.main()
