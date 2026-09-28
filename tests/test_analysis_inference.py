from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from nextwave.evaluation.analysis_inference import (
    ANALYSIS_INFERENCE_VERSION,
    EXPLANATION_VERSION,
    build_analysis_inference,
    export_analysis_inference,
)
from nextwave.evaluation.model import _NUMERIC_KEYS, _STRUCTURED_KEYS


def _bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()


def _digest(payload: bytes) -> dict[str, object]:
    return {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


class AnalysisInferenceTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        feature_dir = root / "features"
        feature_dir.mkdir()
        row = {
            "schema_version": "candidate-feature-table-v2",
            "candidate_id": "candidate-001",
            "canonical_name": "Paged attention",
            "aliases": [],
            "domain": "Edge",
            "analysis_scope_key": "edge-v1",
            "source_query": "Edge AI",
            "model_text": "Paged attention",
            "features": {
                **{key: False for key in _STRUCTURED_KEYS},
                **{key: 0.0 for key in _NUMERIC_KEYS},
            },
        }
        features = _bytes(row)
        (feature_dir / "features.jsonl").write_bytes(features)
        feature_manifest = {
            "schema_version": "analysis-feature-table-v1",
            "feature_version": "candidate-feature-table-v2",
            "candidate_count": 1,
            "outputs": {"features.jsonl": _digest(features)},
        }
        (feature_dir / "manifest.json").write_bytes(_bytes(feature_manifest))

        model_dir = root / "model"
        model_dir.mkdir()
        vector_size = 1 + len(_STRUCTURED_KEYS) + len(_NUMERIC_KEYS) + 1
        model = {
            "schema_version": "nextwave-logreg-report-v4",
            "feature_version": "candidate-feature-table-v2",
            "threshold": 0.5,
            "vectorizer": {
                "vocabulary": [],
                "idf": [],
                "domains": ["Edge"],
                "structured_keys": list(_STRUCTURED_KEYS),
                "numeric_keys": list(_NUMERIC_KEYS),
                "numeric_means": [0.0] * len(_NUMERIC_KEYS),
                "numeric_scales": [1.0] * len(_NUMERIC_KEYS),
                "include_text": True,
            },
            "weights": [0.0] * vector_size,
        }
        model_bytes = _bytes(model)
        (model_dir / "model.json").write_bytes(model_bytes)
        metrics = {
            "schema_version": "nextwave-logreg-report-v4",
            "qualification_eligible": False,
            "outputs": {"model.json": _digest(model_bytes)},
        }
        (model_dir / "metrics.json").write_bytes(_bytes(metrics))
        return feature_dir, model_dir

    def test_scores_with_frozen_model_and_preserves_development_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            feature_dir, model_dir = self._fixture(Path(tmp))
            rows, importance, manifest = build_analysis_inference(
                feature_dir=feature_dir, model_dir=model_dir
            )
        prediction = json.loads(rows.decode())
        importance_rows = [json.loads(line) for line in importance.decode().splitlines()]
        summary = json.loads(manifest)
        self.assertEqual(prediction["model_score"], 0.5)
        self.assertIsNone(prediction["confidence"])
        self.assertEqual(prediction["prediction"], 1)
        self.assertFalse(prediction["explanation"]["confidence_available"])
        self.assertEqual(
            prediction["explanation"]["explanation_version"], EXPLANATION_VERSION
        )
        self.assertEqual(prediction["explanation"]["feature_contributions"], [])
        self.assertTrue(importance_rows)
        self.assertEqual(summary["schema_version"], ANALYSIS_INFERENCE_VERSION)
        self.assertEqual(summary["release_status"], "development_only")
        self.assertFalse(summary["qualification_eligible"])
        self.assertFalse(summary["confidence_available"])

    def test_local_contributions_reconstruct_score_and_rank_factors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            feature_dir, model_dir = self._fixture(Path(tmp))
            feature_path = feature_dir / "features.jsonl"
            feature = json.loads(feature_path.read_text(encoding="utf-8"))
            feature["features"][_STRUCTURED_KEYS[0]] = True
            feature["features"][_NUMERIC_KEYS[0]] = 2.0
            feature_bytes = _bytes(feature)
            feature_path.write_bytes(feature_bytes)
            feature_manifest_path = feature_dir / "manifest.json"
            feature_manifest = json.loads(
                feature_manifest_path.read_text(encoding="utf-8")
            )
            feature_manifest["outputs"]["features.jsonl"] = _digest(feature_bytes)
            feature_manifest_path.write_bytes(_bytes(feature_manifest))

            model_path = model_dir / "model.json"
            model = json.loads(model_path.read_text(encoding="utf-8"))
            model["weights"][0] = -0.2
            model["weights"][1] = 0.8
            model["weights"][1 + len(_STRUCTURED_KEYS)] = 0.5
            model["weights"][-1] = -0.3
            model_bytes = _bytes(model)
            model_path.write_bytes(model_bytes)
            metrics_path = model_dir / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["outputs"]["model.json"] = _digest(model_bytes)
            metrics_path.write_bytes(_bytes(metrics))

            rows, _, _ = build_analysis_inference(
                feature_dir=feature_dir, model_dir=model_dir
            )

        prediction = json.loads(rows.decode())
        explanation = prediction["explanation"]
        reconstructed_logit = explanation["intercept"] + sum(
            item["contribution"] for item in explanation["feature_contributions"]
        )
        self.assertAlmostEqual(reconstructed_logit, explanation["logit"])
        reconstructed_score = 1.0 / (1.0 + math.exp(-reconstructed_logit))
        self.assertAlmostEqual(reconstructed_score, prediction["model_score"])
        self.assertEqual(
            explanation["top_positive_factors"][0]["feature_name"],
            _NUMERIC_KEYS[0],
        )
        self.assertEqual(
            explanation["top_negative_factors"][0]["feature_group"], "domain"
        )

    def test_rejects_tampered_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            feature_dir, model_dir = self._fixture(Path(tmp))
            with (model_dir / "model.json").open("ab") as stream:
                stream.write(b" ")
            with self.assertRaisesRegex(ValueError, "diverges"):
                build_analysis_inference(feature_dir=feature_dir, model_dir=model_dir)

    def test_rejects_weight_dimension_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            feature_dir, model_dir = self._fixture(Path(tmp))
            model_path = model_dir / "model.json"
            model = json.loads(model_path.read_text(encoding="utf-8"))
            model["weights"].pop()
            model_bytes = _bytes(model)
            model_path.write_bytes(model_bytes)
            metrics_path = model_dir / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["outputs"]["model.json"] = _digest(model_bytes)
            metrics_path.write_bytes(_bytes(metrics))
            with self.assertRaisesRegex(ValueError, "weight count"):
                build_analysis_inference(feature_dir=feature_dir, model_dir=model_dir)

    def test_export_refuses_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_dir, model_dir = self._fixture(root)
            output = root / "output"
            paths = export_analysis_inference(
                feature_dir=feature_dir, model_dir=model_dir, output_dir=output
            )
            self.assertTrue(paths.predictions.is_file())
            self.assertTrue(paths.feature_importance.is_file())
            with self.assertRaisesRegex(ValueError, "already exists"):
                export_analysis_inference(
                    feature_dir=feature_dir, model_dir=model_dir, output_dir=output
                )


if __name__ == "__main__":
    unittest.main()
