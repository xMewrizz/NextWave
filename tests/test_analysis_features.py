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
    def _fixture(
        self,
        root: Path,
        *,
        cutoff_date: str = "2026-09-15",
        history_from: str = "2024-09-15",
        recent_from: str = "2025-09-15",
        published_at: str = "2026-01-01",
    ) -> tuple[Path, Path, Path]:
        candidate = {
            "candidate_id": "alias-group-001",
            "canonical_name": "Paged attention",
            "aliases": ["paged-attention"],
            "domain": "Инфраструктура ИИ",
            "analysis_scope_key": "ai-infrastructure-v1",
            "origin": {
                "bundle_id": "bundle-analysis",
                "group_id": "alias-group-001",
                "source_query": "Инфраструктура ИИ",
            },
            "cutoff_date": cutoff_date,
            "search_terms": ["Paged attention", "paged-attention"],
        }
        plan_value = {
            "schema_version": "labeling-enrichment-plan-v2",
            "cutoff_date": cutoff_date,
            "history_from": history_from,
            "recent_window_from": recent_from,
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
                    "published_at": published_at,
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
                "cutoff_date": cutoff_date,
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
        self.assertEqual(row["source_query"], "Инфраструктура ИИ")
        self.assertTrue(row["features"]["temporal_count_coverage_complete"])
        self.assertAlmostEqual(row["features"]["scientific_recent_count_log1p"], math.log1p(5))

    def test_historical_plan_uses_its_own_feature_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, result, temporal = self._fixture(
                root,
                cutoff_date="2025-09-15",
                history_from="2023-09-16",
                recent_from="2024-09-15",
                published_at="2024-05-24",
            )
            rows, _ = build_analysis_feature_table(
                analysis_plan_dir=plan,
                enrichment_result_dir=result,
                temporal_count_dir=temporal,
            )

        row = json.loads(rows.decode())
        self.assertTrue(row["features"]["scientific_previous_present"])
        self.assertFalse(row["features"]["scientific_recent_present"])

    def test_preserves_incomplete_enrichment_coverage_as_model_feature(self) -> None:
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
            output, _ = build_analysis_feature_table(
                analysis_plan_dir=plan,
                enrichment_result_dir=result,
                temporal_count_dir=temporal,
            )
            row = json.loads(output)
            self.assertTrue(row["features"]["scientific_coverage_complete"])
            self.assertFalse(row["features"]["industry_coverage_complete"])

    def test_partial_temporal_coverage_uses_missing_value_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, result, temporal = self._fixture(root)
            path = temporal / "candidate_temporal_features.jsonl"
            row = json.loads(path.read_text(encoding="utf-8"))
            row["coverage"] = "unknown"
            row["counts"] = {
                "candidate_previous": None,
                "candidate_recent": None,
                "scope_previous": None,
                "scope_recent": None,
            }
            for key in (
                "candidate_log_growth",
                "scope_share_previous",
                "scope_share_recent",
                "scope_share_delta",
            ):
                row[key] = None
            payload = _jsonl([row])
            path.write_bytes(payload)
            manifest = json.loads((temporal / "manifest.json").read_text())
            manifest["status"] = "partial"
            manifest["outputs"]["candidate_temporal_features.jsonl"] = _digest(
                payload
            )
            (temporal / "manifest.json").write_bytes(_bytes(manifest))

            output, _ = build_analysis_feature_table(
                analysis_plan_dir=plan,
                enrichment_result_dir=result,
                temporal_count_dir=temporal,
            )
            features = json.loads(output)["features"]
            self.assertFalse(features["temporal_count_coverage_complete"])
            self.assertIsNone(features["scientific_count_log_growth"])

    def test_temporal_identity_accepts_normalized_candidate_quotes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, result, temporal = self._fixture(root)
            plan_path = plan / "plan.json"
            plan_value = json.loads(plan_path.read_text(encoding="utf-8"))
            plan_value["candidates"][0]["search_terms"][0] = '"Quantum" platform'
            plan_payload = _bytes(plan_value)
            plan_path.write_bytes(plan_payload)
            plan_manifest = json.loads((plan / "manifest.json").read_text())
            plan_manifest["outputs"]["plan.json"] = _digest(plan_payload)
            (plan / "manifest.json").write_bytes(_bytes(plan_manifest))
            result_manifest = json.loads((result / "manifest.json").read_text())
            result_manifest["plan"] = _digest(plan_payload)
            (result / "manifest.json").write_bytes(_bytes(result_manifest))

            temporal_path = temporal / "candidate_temporal_features.jsonl"
            temporal_row = json.loads(temporal_path.read_text(encoding="utf-8"))
            temporal_row["search_text"] = "Quantum platform"
            temporal_payload = _jsonl([temporal_row])
            temporal_path.write_bytes(temporal_payload)
            temporal_manifest = json.loads((temporal / "manifest.json").read_text())
            temporal_manifest["outputs"]["candidate_temporal_features.jsonl"] = (
                _digest(temporal_payload)
            )
            (temporal / "manifest.json").write_bytes(_bytes(temporal_manifest))

            rows, _ = build_analysis_feature_table(
                analysis_plan_dir=plan,
                enrichment_result_dir=result,
                temporal_count_dir=temporal,
            )
            self.assertEqual(json.loads(rows)["candidate_id"], "alias-group-001")

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

    def test_rejects_missing_origin_source_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, result, temporal = self._fixture(root)
            path = plan / "plan.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            del value["candidates"][0]["origin"]["source_query"]
            payload = _bytes(value)
            path.write_bytes(payload)
            manifest = json.loads((plan / "manifest.json").read_text(encoding="utf-8"))
            manifest["outputs"]["plan.json"] = _digest(payload)
            (plan / "manifest.json").write_bytes(_bytes(manifest))

            with self.assertRaisesRegex(ValueError, "needs origin.source_query"):
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
