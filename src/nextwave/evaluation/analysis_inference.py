"""Apply one frozen logistic-regression artifact to query-specific features."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nextwave.datasets.artifacts import publish_artifact_bundle

from .analysis_features import ANALYSIS_FEATURE_TABLE_VERSION
from .feature_table import FEATURE_TABLE_VERSION, FEATURES_FILENAME, MANIFEST_FILENAME, _digest
from .model import MODEL_REPORT_VERSION, _Fit, _score, _tokens, _vector, _Vectorizer

ANALYSIS_INFERENCE_VERSION = "analysis-inference-v2"
PREDICTIONS_FILENAME = "predictions.jsonl"
IMPORTANCE_FILENAME = "feature_importance.jsonl"
MODEL_FILENAME = "model.json"
METRICS_FILENAME = "metrics.json"
EXPLANATION_VERSION = "linear-logit-contributions-v1"
_TOP_FACTOR_COUNT = 5

_FEATURE_LABELS_RU = {
    "scientific_coverage_complete": "полное научное покрытие",
    "industry_coverage_complete": "полное отраслевое покрытие",
    "scientific_previous_present": "научные публикации в предыдущем окне",
    "scientific_recent_present": "научные публикации в недавнем окне",
    "industry_previous_present": "отраслевые публикации в предыдущем окне",
    "industry_recent_present": "отраслевые публикации в недавнем окне",
    "unknown_date_present": "источники с неизвестной датой",
    "trust_ab_present": "источник доверия A/B",
    "multiple_origins_present": "несколько независимых первоисточников",
    "multiple_organizations_present": "несколько независимых организаций",
    "multiple_source_types_present": "несколько типов источников",
    "temporal_count_coverage_complete": "полное временное покрытие",
    "scientific_previous_count_log1p": "число ранних научных публикаций",
    "scientific_recent_count_log1p": "число недавних научных публикаций",
    "scientific_count_log_growth": "рост числа научных публикаций",
    "scientific_scope_share_previous": "ранняя доля темы в направлении",
    "scientific_scope_share_recent": "недавняя доля темы в направлении",
    "scientific_scope_share_delta": "изменение доли темы в направлении",
}


@dataclass(frozen=True, slots=True)
class AnalysisInferencePaths:
    predictions: Path
    feature_importance: Path
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


def _feature_specs(
    fit: _Fit, row: dict[str, Any]
) -> list[dict[str, Any]]:
    vectorizer = fit.vectorizer
    values = _vector(row, vectorizer)
    features = row.get("features")
    if not isinstance(features, dict):
        raise ValueError("analysis feature row features must be an object")
    counts = Counter(_tokens(str(row["model_text"])))
    specs: list[dict[str, Any]] = []
    index = 1
    for token in vectorizer.vocabulary:
        specs.append(
            {
                "index": index,
                "feature_group": "text",
                "feature_name": token,
                "label_ru": f"текстовый признак: {token}",
                "raw_value": counts[token],
                "transformed_value": values[index],
            }
        )
        index += 1
    for name in vectorizer.structured_keys:
        specs.append(
            {
                "index": index,
                "feature_group": "structured",
                "feature_name": name,
                "label_ru": _FEATURE_LABELS_RU.get(name, name),
                "raw_value": features[name],
                "transformed_value": values[index],
            }
        )
        index += 1
    for name in vectorizer.numeric_keys:
        specs.append(
            {
                "index": index,
                "feature_group": "numeric",
                "feature_name": name,
                "label_ru": _FEATURE_LABELS_RU.get(name, name),
                "raw_value": features.get(name),
                "transformed_value": values[index],
            }
        )
        index += 1
    for domain in vectorizer.domains:
        specs.append(
            {
                "index": index,
                "feature_group": "domain",
                "feature_name": domain,
                "label_ru": f"область: {domain}",
                "raw_value": row.get("domain") == domain,
                "transformed_value": values[index],
            }
        )
        index += 1
    if index != len(values):
        raise ValueError("model explanation dimensions do not match vector")
    return specs


def _explain(fit: _Fit, row: dict[str, Any], score: float) -> dict[str, Any]:
    contributions: list[dict[str, Any]] = []
    for spec in _feature_specs(fit, row):
        coefficient = fit.weights[spec["index"]]
        contribution = coefficient * spec["transformed_value"]
        if contribution == 0.0:
            continue
        contributions.append(
            {
                key: value for key, value in spec.items() if key != "index"
            }
            | {
                "coefficient": coefficient,
                "contribution": contribution,
                "direction": (
                    "supports_weak_signal"
                    if contribution > 0
                    else "opposes_weak_signal"
                ),
            }
        )
    intercept = fit.weights[0]
    logit = intercept + sum(item["contribution"] for item in contributions)
    reconstructed = 1.0 / (1.0 + math.exp(-logit)) if logit >= 0 else (
        math.exp(logit) / (1.0 + math.exp(logit))
    )
    if not math.isclose(reconstructed, score, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("feature contributions do not reconstruct model score")
    positive = sorted(
        (item for item in contributions if item["contribution"] > 0),
        key=lambda item: (-item["contribution"], item["feature_group"], item["feature_name"]),
    )[:_TOP_FACTOR_COUNT]
    negative = sorted(
        (item for item in contributions if item["contribution"] < 0),
        key=lambda item: (item["contribution"], item["feature_group"], item["feature_name"]),
    )[:_TOP_FACTOR_COUNT]
    return {
        "explanation_version": EXPLANATION_VERSION,
        "score_semantics": "uncalibrated_sigmoid_score",
        "confidence_available": False,
        "intercept": intercept,
        "logit": logit,
        "feature_contributions": contributions,
        "top_positive_factors": positive,
        "top_negative_factors": negative,
    }


def _global_importance(fit: _Fit, row: dict[str, Any]) -> bytes:
    items = []
    for spec in _feature_specs(fit, row):
        coefficient = fit.weights[spec["index"]]
        items.append(
            {
                "schema_version": ANALYSIS_INFERENCE_VERSION,
                "explanation_version": EXPLANATION_VERSION,
                "feature_group": spec["feature_group"],
                "feature_name": spec["feature_name"],
                "label_ru": spec["label_ru"],
                "coefficient": coefficient,
                "absolute_coefficient": abs(coefficient),
            }
        )
    items.sort(
        key=lambda item: (
            -item["absolute_coefficient"],
            item["feature_group"],
            item["feature_name"],
        )
    )
    for rank, item in enumerate(items, 1):
        item["importance_rank"] = rank
    return b"".join(
        (json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for item in items
    )


def build_analysis_inference(
    *, feature_dir: str | Path, model_dir: str | Path
) -> tuple[bytes, bytes, bytes]:
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
                "confidence": None,
                "decision_threshold": threshold,
                "prediction": int(score >= threshold),
                "explanation": _explain(fit, row, score),
            }
        )
    prediction_bytes = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in predictions
    )
    importance_bytes = _global_importance(fit, rows[0])
    qualification = metrics.get("qualification_eligible") is True
    manifest_value = {
        "schema_version": ANALYSIS_INFERENCE_VERSION,
        "release_status": "qualification" if qualification else "development_only",
        "qualification_eligible": qualification,
        "candidate_count": len(predictions),
        "threshold": threshold,
        "score_semantics": "uncalibrated_sigmoid_score",
        "confidence_available": False,
        "explanation_version": EXPLANATION_VERSION,
        "inputs": {
            "features": _digest(feature_manifest_bytes),
            "model": _digest(model_bytes),
            "model_metrics": _digest(metrics_bytes),
        },
        "limitations": (
            []
            if qualification
            else [
                "frozen model artifact is development_only and cannot support qualification claims"
            ]
        )
        + [
            "model score is not calibrated and must not be presented as probability or confidence"
        ],
        "outputs": {
            PREDICTIONS_FILENAME: _digest(prediction_bytes),
            IMPORTANCE_FILENAME: _digest(importance_bytes),
        },
    }
    manifest_bytes = (
        json.dumps(manifest_value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return prediction_bytes, importance_bytes, manifest_bytes


def export_analysis_inference(
    *, feature_dir: str | Path, model_dir: str | Path, output_dir: str | Path
) -> AnalysisInferencePaths:
    predictions, importance, manifest = build_analysis_inference(
        feature_dir=feature_dir, model_dir=model_dir
    )
    paths = publish_artifact_bundle(
        {
            PREDICTIONS_FILENAME: predictions,
            IMPORTANCE_FILENAME: importance,
            MANIFEST_FILENAME: manifest,
        },
        output_dir,
    )
    return AnalysisInferencePaths(
        predictions=paths[PREDICTIONS_FILENAME],
        feature_importance=paths[IMPORTANCE_FILENAME],
        manifest=paths[MANIFEST_FILENAME],
    )
