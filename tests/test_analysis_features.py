from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from nextwave.evaluation.analysis_features import (
    ANALYSIS_FEATURE_TABLE_VERSION,
    build_analysis_feature_table,
    export_analysis_feature_table,
)


def _bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()


def _jsonl(rows: list[dict]) -> bytes:
    return b"".join(_bytes(row) for row in rows)


def _digest(payload: bytes) -> dict[str, object]:
    return {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _bundle(directory: Path, manifest: dict, files: dict[str, bytes]) -> None:
    directory.mkdir()
    for name, payload in files.items():
        (directory / name).write_bytes(payload)
    manifest["outputs"] = {name: _digest(payload) for name, payload in files.items()}
    (directory / "manifest.json").write_bytes(_bytes(manifest))


class AnalysisFeatureTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        candidate = {
            "candidate_id": "alias-group-001",
            "canonical_name": "Paged attention",
            "aliases": ["paged-attention"],
            "domain": "Инфраструктура ИИ",
            "analysis_scope_key": "ai-infrastructure-v1",
            "source_query": "Инфраструктура ИИ",
            "cutoff_date": "2026-09-15",
            "search_terms": ["Paged attention", "paged-attention"],
        }
        plan_value = {
            "schema_version": "labeling-enrichment-plan-v2",
            "cutoff_date": "2026-09-15",
            "candidates": [candidate],
        }
        plan_bytes = _bytes(plan_value)
        plan = root / "plan"
        _bundle(
            plan,
            {
                "schema_version": "labeling-enrichment-plan-v2",
                "plan_role": "analysis_candidates",
                "bundle_id": "bundle-analysis",
                "candidate_count": 1,
            },
            {"plan.json": plan_bytes},
        )
        result = root / "result"
        coverage = _jsonl(
            [
                {"candidate_id": "alias-group-001", "source_class": source, "status": "complete"}
                for source in ("scientific", "industry")
            ]
        )
        documents = _jsonl(
            [
                {
                    "candidate_id": "alias-group-001",
                    "document_id": "doc-1",
                    "connector": "openalex",
                    "published_at": "2026-01-01",
                    "trust_tier": "A",
                    "origin_id": "doi:one",
                    "organizations": ["Org A"],
                    "source_type": "scientific_publication",
                }
            ]
        )
        _bundle(
            result,
            {
                "schema_version": "labeling-enrichment-result-v2",
                "bundle_id": "bundle-analysis",
                "plan": _digest(plan_bytes),
                "totals": {"candidates": 1},
            },
            {"coverage.jsonl": coverage, "documents.jsonl": documents},
        )
        temporal = root / "temporal"
        temporal_row = {
            **candidate,
            "role": "analysis",
            "search_text": "Paged attention",
            "coverage": "complete",
            "counts": {
                "candidate_previous": 2,
                "candidate_recent": 5,
                "scope_previous": 100,
                "scope_recent": 200,
            },
            "candidate_log_growth": math.log1p(5) - math.log1p(2),
            "scope_share_previous": 0.02,
            "scope_share_recent": 0.025,
            "scope_share_delta": 0.005,
        }
        _bundle(
            temporal,
            {
                "schema_version": "openalex-temporal-count-result-v4",
                "status": "complete",
            },
            {"candidate_temporal_features.jsonl": _jsonl([temporal_row])},
        )
        return plan, result, temporal

    def test_builds_unlabeled_row_with_training_feature_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows, manifest = build_analysis_feature_table(
                analysis_plan_dir=self._fixture(Path(tmp))[0],
                enrichment_result_dir=Path(tmp) / "result",
                temporal_count_dir=Path(tmp) / "temporal",
            )
        row = json.loads(rows.decode())
        summary = json.loads(manifest)
        self.assertEqual(summary["schema_version"], ANALYSIS_FEATURE_TABLE_VERSION)
        self.assertEqual(summary["candidate_count"], 1)
        self.assertNotIn("target", row)
        self.assertNotIn("negative_class", row)
        self.assertTrue(row["features"]["temporal_count_coverage_complete"])
        self.assertAlmostEqual(row["features"]["scientific_recent_count_log1p"], math.log1p(5))

    def test_rejects_incomplete_enrichment_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, result, temporal = self._fixture(root)
            coverage_path = result / "coverage.jsonl"
            rows = [json.loads(line) for line in coverage_path.read_text().splitlines()]
            rows[1]["status"] = "unknown"
            payload = _jsonl(rows)
            coverage_path.write_bytes(payload)
            manifest = json.loads((result / "manifest.json").read_text())
            manifest["outputs"]["coverage.jsonl"] = _digest(payload)
            (result / "manifest.json").write_bytes(_bytes(manifest))
            with self.assertRaisesRegex(ValueError, "coverage must be complete"):
                build_analysis_feature_table(
                    analysis_plan_dir=plan,
                    enrichment_result_dir=result,
                    temporal_count_dir=temporal,
                )

    def test_rejects_temporal_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, result, temporal = self._fixture(root)
            path = temporal / "candidate_temporal_features.jsonl"
            row = json.loads(path.read_text(encoding="utf-8"))
            row["analysis_scope_key"] = "edge-v1"
            payload = _jsonl([row])
            path.write_bytes(payload)
            manifest = json.loads(
                (temporal / "manifest.json").read_text(encoding="utf-8")
            )
            manifest["outputs"]["candidate_temporal_features.jsonl"] = _digest(payload)
            (temporal / "manifest.json").write_bytes(_bytes(manifest))
            with self.assertRaisesRegex(ValueError, "temporal identity differs"):
                build_analysis_feature_table(
                    analysis_plan_dir=plan,
                    enrichment_result_dir=result,
                    temporal_count_dir=temporal,
                )

    def test_export_refuses_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, result, temporal = self._fixture(root)
            output = root / "output"
            paths = export_analysis_feature_table(
                analysis_plan_dir=plan,
                enrichment_result_dir=result,
                temporal_count_dir=temporal,
                output_dir=output,
            )
            self.assertTrue(paths.features.is_file())
            with self.assertRaisesRegex(ValueError, "already exists"):
                export_analysis_feature_table(
                    analysis_plan_dir=plan,
                    enrichment_result_dir=result,
                    temporal_count_dir=temporal,
                    output_dir=output,
                )


if __name__ == "__main__":
    unittest.main()
