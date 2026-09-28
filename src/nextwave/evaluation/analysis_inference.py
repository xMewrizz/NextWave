"""Apply one frozen logistic-regression artifact to query-specific features."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nextwave.datasets.artifacts import publish_artifact_bundle

from .analysis_features import ANALYSIS_FEATURE_TABLE_VERSION
from .feature_table import FEATURE_TABLE_VERSION, FEATURES_FILENAME, MANIFEST_FILENAME, _digest
from .model import MODEL_REPORT_VERSION, _Fit, _score, _Vectorizer

ANALYSIS_INFERENCE_VERSION = "analysis-inference-v1"
PREDICTIONS_FILENAME = "predictions.jsonl"
MODEL_FILENAME = "model.json"
METRICS_FILENAME = "metrics.json"


@dataclass(frozen=True, slots=True)
class AnalysisInferencePaths:
    predictions: Path
    manifest: Path


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _load_fit(model: dict[str, Any]) -> tuple[_Fit, float]:
    if model.get("schema_version") != MODEL_REPORT_VERSION:
        raise ValueError("model schema version is not supported")
    if model.get("feature_version") != FEATURE_TABLE_VERSION:
        raise ValueError("model feature version differs from analysis features")
    vector = model.get("vectorizer")
    if not isinstance(vector, dict):
        raise ValueError("model vectorizer must be an object")
    tuple_fields = (
        "vocabulary",
        "idf",
        "domains",
        "structured_keys",
        "numeric_keys",
        "numeric_means",
        "numeric_scales",
    )
    if any(not isinstance(vector.get(field), list) for field in tuple_fields):
        raise ValueError("model vectorizer arrays are invalid")
    vocabulary = tuple(vector["vocabulary"])
    idf = tuple(vector["idf"])
    domains = tuple(vector["domains"])
    structured = tuple(vector["structured_keys"])
    numeric = tuple(vector["numeric_keys"])
    means = tuple(vector["numeric_means"])
    scales = tuple(vector["numeric_scales"])
    if len(vocabulary) != len(idf) or len(numeric) != len(means) or len(numeric) != len(scales):
        raise ValueError("model vectorizer dimensions do not match")
    names = (*vocabulary, *domains, *structured, *numeric)
    if any(not isinstance(value, str) or not value for value in names):
        raise ValueError("model vectorizer names must be non-blank strings")
    numeric_values = (*idf, *means, *scales)
    if any(type(value) not in {int, float} or not math.isfinite(value) for value in numeric_values):
        raise ValueError("model vectorizer numbers must be finite")
    if any(float(value) <= 0 for value in scales):
        raise ValueError("model numeric scales must be positive")
    if vector.get("include_text") is not True:
        raise ValueError("analysis inference requires the frozen text+structured model")
    weights = model.get("weights")
    if not isinstance(weights, list) or any(
        type(value) not in {int, float} or not math.isfinite(value) for value in weights
    ):
        raise ValueError("model weights must be finite numbers")
    expected_weights = 1 + len(vocabulary) + len(structured) + len(numeric) + len(domains)
    if len(weights) != expected_weights:
        raise ValueError("model weight count does not match vectorizer")
    threshold = model.get("threshold")
    if type(threshold) not in {int, float} or not 0 <= float(threshold) <= 1:
        raise ValueError("model threshold must be between zero and one")
    fit = _Fit(
        vectorizer=_Vectorizer(
            vocabulary=vocabulary,
            idf=tuple(float(value) for value in idf),
            domains=domains,
            structured_keys=structured,
            numeric_keys=numeric,
            numeric_means=tuple(float(value) for value in means),
            numeric_scales=tuple(float(value) for value in scales),
            include_text=True,
        ),
        weights=tuple(float(value) for value in weights),
    )
    return fit, float(threshold)


def build_analysis_inference(
    *, feature_dir: str | Path, model_dir: str | Path
) -> tuple[bytes, bytes]:
    features_path = Path(feature_dir)
    feature_manifest_bytes = (features_path / MANIFEST_FILENAME).read_bytes()
    feature_manifest = json.loads(feature_manifest_bytes)
    if (
        not isinstance(feature_manifest, dict)
        or feature_manifest.get("schema_version") != ANALYSIS_FEATURE_TABLE_VERSION
        or feature_manifest.get("feature_version") != FEATURE_TABLE_VERSION
    ):
        raise ValueError("analysis feature manifest is not supported")
    features_bytes = (features_path / FEATURES_FILENAME).read_bytes()
    if (feature_manifest.get("outputs") or {}).get(FEATURES_FILENAME) != _digest(features_bytes):
        raise ValueError("analysis features diverge from manifest")
    rows = [json.loads(line) for line in features_bytes.decode("utf-8").splitlines()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("analysis features must contain object rows")
    candidate_ids = [row.get("candidate_id") for row in rows]
    if any(not isinstance(value, str) or not value for value in candidate_ids):
        raise ValueError("analysis feature row needs candidate_id")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("analysis features duplicate candidate_id")
    if feature_manifest.get("candidate_count") != len(rows):
        raise ValueError("analysis feature candidate_count does not match rows")

    model_path = Path(model_dir)
    metrics_bytes = (model_path / METRICS_FILENAME).read_bytes()
    metrics = json.loads(metrics_bytes)
    if not isinstance(metrics, dict) or metrics.get("schema_version") != MODEL_REPORT_VERSION:
        raise ValueError("model metrics schema is not supported")
    model_bytes = (model_path / MODEL_FILENAME).read_bytes()
    if (metrics.get("outputs") or {}).get(MODEL_FILENAME) != _digest(model_bytes):
        raise ValueError("model.json diverges from metrics manifest")
    model = json.loads(model_bytes)
    if not isinstance(model, dict):
        raise ValueError("model.json must be an object")
    fit, threshold = _load_fit(model)

    predictions: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: item["candidate_id"]):
        score = _score(fit, row)
        predictions.append(
            {
                "schema_version": ANALYSIS_INFERENCE_VERSION,
                "candidate_id": row["candidate_id"],
                "canonical_name": row.get("canonical_name"),
                "domain": row.get("domain"),
                "analysis_scope_key": row.get("analysis_scope_key"),
                "source_query": row.get("source_query"),
                "model_score": score,
                "decision_threshold": threshold,
                "prediction": int(score >= threshold),
            }
        )
    prediction_bytes = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in predictions
    )
    qualification = metrics.get("qualification_eligible") is True
    manifest_value = {
        "schema_version": ANALYSIS_INFERENCE_VERSION,
        "release_status": "qualification" if qualification else "development_only",
        "qualification_eligible": qualification,
        "candidate_count": len(predictions),
        "threshold": threshold,
        "inputs": {
            "features": _digest(feature_manifest_bytes),
            "model": _digest(model_bytes),
            "model_metrics": _digest(metrics_bytes),
        },
        "limitations": [] if qualification else [
            "frozen model artifact is development_only and cannot support qualification claims"
        ],
        "outputs": {PREDICTIONS_FILENAME: _digest(prediction_bytes)},
    }
    manifest_bytes = (
        json.dumps(manifest_value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return prediction_bytes, manifest_bytes


def export_analysis_inference(
    *, feature_dir: str | Path, model_dir: str | Path, output_dir: str | Path
) -> AnalysisInferencePaths:
    predictions, manifest = build_analysis_inference(
        feature_dir=feature_dir, model_dir=model_dir
    )
    paths = publish_artifact_bundle(
        {PREDICTIONS_FILENAME: predictions, MANIFEST_FILENAME: manifest}, output_dir
    )
    return AnalysisInferencePaths(
        predictions=paths[PREDICTIONS_FILENAME], manifest=paths[MANIFEST_FILENAME]
    )
