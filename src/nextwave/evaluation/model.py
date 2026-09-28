"""Deterministic grouped evaluation for the frozen candidate feature table."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nextwave.datasets.artifacts import publish_artifact_bundle

from .feature_table import FEATURE_TABLE_VERSION, FEATURES_FILENAME, MANIFEST_FILENAME

MODEL_REPORT_VERSION = "nextwave-logreg-report-v4"
PREDICTIONS_FILENAME = "oof_predictions.jsonl"
ERRORS_FILENAME = "errors.jsonl"
REPORT_FILENAME = "metrics.json"
MODEL_FILENAME = "model.json"
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_STRUCTURED_KEYS = (
    "scientific_coverage_complete",
    "industry_coverage_complete",
    "scientific_previous_present",
    "scientific_recent_present",
    "industry_previous_present",
    "industry_recent_present",
    "unknown_date_present",
    "trust_ab_present",
    "multiple_origins_present",
    "multiple_organizations_present",
    "multiple_source_types_present",
    "temporal_count_coverage_complete",
)
_MEDIA_KEYS = frozenset(
    {
        "industry_coverage_complete",
        "industry_previous_present",
        "industry_recent_present",
    }
)
_NUMERIC_KEYS = (
    "scientific_previous_count_log1p",
    "scientific_recent_count_log1p",
    "scientific_count_log_growth",
    "scientific_scope_share_previous",
    "scientific_scope_share_recent",
    "scientific_scope_share_delta",
)


@dataclass(frozen=True, slots=True)
class ModelReportPaths:
    predictions: Path
    report: Path
    model: Path
    errors: Path | None = None


@dataclass(frozen=True, slots=True)
class _Vectorizer:
    vocabulary: tuple[str, ...]
    idf: tuple[float, ...]
    domains: tuple[str, ...]
    structured_keys: tuple[str, ...]
    numeric_keys: tuple[str, ...]
    numeric_means: tuple[float, ...]
    numeric_scales: tuple[float, ...]
    include_text: bool


@dataclass(frozen=True, slots=True)
class _Fit:
    vectorizer: _Vectorizer
    weights: tuple[float, ...]


def _digest(payload: bytes) -> dict[str, Any]:
    return {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    words = _TOKEN.findall(normalized)
    return [*words, *(f"{left}::{right}" for left, right in zip(words, words[1:], strict=False))]


def _fit_vectorizer(
    rows: Sequence[Mapping[str, Any]], *, include_text: bool, include_media: bool
) -> _Vectorizer:
    document_frequency: Counter[str] = Counter()
    if include_text:
        for row in rows:
            text = row.get("model_text")
            if not isinstance(text, str):
                raise ValueError("feature row model_text must be a string")
            document_frequency.update(set(_tokens(text)))
    eligible = [token for token, count in document_frequency.items() if count >= 2]
    eligible.sort(key=lambda token: (-document_frequency[token], token))
    vocabulary = tuple(eligible[:2000])
    size = len(rows)
    idf = tuple(
        math.log((1 + size) / (1 + document_frequency[token])) + 1.0 for token in vocabulary
    )
    domains = tuple(sorted({str(row.get("domain")) for row in rows}))
    structured_keys = tuple(
        key for key in _STRUCTURED_KEYS if include_media or key not in _MEDIA_KEYS
    )
    numeric_values: list[list[float]] = [[] for _ in _NUMERIC_KEYS]
    for row in rows:
        features = row.get("features")
        if not isinstance(features, dict):
            raise ValueError("feature row features must be an object")
        for index, key in enumerate(_NUMERIC_KEYS):
            value = features.get(key)
            if value is None:
                continue
            if type(value) not in {int, float} or not math.isfinite(value):
                raise ValueError(f"numeric feature {key} must be finite or null")
            numeric_values[index].append(float(value))
    numeric_means = tuple(sum(values) / len(values) if values else 0.0 for values in numeric_values)
    numeric_scales = tuple(
        (math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)) if values else 1.0)
        for values, mean in zip(numeric_values, numeric_means, strict=True)
    )
    numeric_scales = tuple(scale if scale > 0 else 1.0 for scale in numeric_scales)
    return _Vectorizer(
        vocabulary=vocabulary,
        idf=idf,
        domains=domains,
        structured_keys=structured_keys,
        numeric_keys=_NUMERIC_KEYS,
        numeric_means=numeric_means,
        numeric_scales=numeric_scales,
        include_text=include_text,
    )


def _vector(row: Mapping[str, Any], vectorizer: _Vectorizer) -> list[float]:
    result = [1.0]
    if vectorizer.include_text:
        counts = Counter(_tokens(str(row["model_text"])))
        denominator = sum(counts.values()) or 1
        result.extend(
            counts[token] / denominator * idf
            for token, idf in zip(vectorizer.vocabulary, vectorizer.idf, strict=False)
        )
    features = row.get("features")
    if not isinstance(features, dict):
        raise ValueError("feature row features must be an object")
    for key in vectorizer.structured_keys:
        value = features.get(key)
        if type(value) is not bool:
            raise ValueError(f"feature {key} must be boolean")
        result.append(float(value))
    for key, mean, scale in zip(
        vectorizer.numeric_keys,
        vectorizer.numeric_means,
        vectorizer.numeric_scales,
        strict=True,
    ):
        value = features.get(key)
        numeric = mean if value is None else float(value)
        result.append((numeric - mean) / scale)
    result.extend(float(row.get("domain") == domain) for domain in vectorizer.domains)
    return result


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    forward = math.exp(value)
    return forward / (1.0 + forward)


def _train(rows: Sequence[Mapping[str, Any]], *, include_text: bool, include_media: bool) -> _Fit:
    labels = [int(row["target"]) for row in rows]
    if set(labels) != {0, 1}:
        raise ValueError("logistic regression training requires both targets")
    vectorizer = _fit_vectorizer(rows, include_text=include_text, include_media=include_media)
    matrix = [_vector(row, vectorizer) for row in rows]
    weights = [0.0] * len(matrix[0])
    counts = Counter(labels)
    sample_weights = [len(labels) / (2 * counts[label]) for label in labels]
    learning_rate = 0.35
    l2 = 0.2
    total_weight = sum(sample_weights)
    for epoch in range(600):
        gradient = [0.0] * len(weights)
        for values, label, sample_weight in zip(matrix, labels, sample_weights, strict=False):
            error = _sigmoid(sum(w * x for w, x in zip(weights, values, strict=False))) - label
            for index, value in enumerate(values):
                gradient[index] += sample_weight * error * value
        step = learning_rate / math.sqrt(1.0 + epoch / 40.0)
        for index in range(len(weights)):
            regularization = 0.0 if index == 0 else l2 * weights[index]
            weights[index] -= step * (gradient[index] / total_weight + regularization)
    return _Fit(vectorizer=vectorizer, weights=tuple(weights))


def _score(fit: _Fit, row: Mapping[str, Any]) -> float:
    values = _vector(row, fit.vectorizer)
    return _sigmoid(sum(weight * value for weight, value in zip(fit.weights, values, strict=False)))


def _assign_folds(rows: Sequence[Mapping[str, Any]], fold_count: int = 5) -> dict[str, int]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        group_id = row.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("feature row needs group_id")
        grouped[group_id].append(row)
    if len(grouped) < fold_count:
        raise ValueError("not enough identity groups for grouped folds")
    label_totals = Counter(int(row["target"]) for row in rows)
    if not label_totals[0] or not label_totals[1]:
        raise ValueError("grouped folds require both targets")
    targets = {label: label_totals[label] / fold_count for label in (0, 1)}
    totals = [Counter() for _ in range(fold_count)]
    assignments: dict[str, int] = {}
    ordered = sorted(
        grouped.items(),
        key=lambda item: (
            -len(item[1]),
            -abs(sum(int(row["target"]) for row in item[1]) * 2 - len(item[1])),
            item[0],
        ),
    )
    for group_id, members in ordered:
        labels = Counter(int(row["target"]) for row in members)
        fold = min(
            range(fold_count),
            key=lambda index: (
                sum(
                    ((totals[index][label] + labels[label]) / targets[label]) ** 2
                    for label in (0, 1)
                ),
                sum(totals[index].values()),
                index,
            ),
        )
        assignments[group_id] = fold
        totals[fold].update(labels)
    for index, counts in enumerate(totals):
        if not counts[0] or not counts[1]:
            raise ValueError(f"fold {index} does not contain both targets")
    return assignments


def _metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, Any]:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("metrics need equally sized non-empty inputs")
    pairs = tuple(zip(labels, predictions, strict=True))
    tp = sum(label == 1 and prediction == 1 for label, prediction in pairs)
    tn = sum(label == 0 and prediction == 0 for label, prediction in pairs)
    fp = sum(label == 0 and prediction == 1 for label, prediction in pairs)
    fn = sum(label == 1 and prediction == 0 for label, prediction in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    false_positive_rate = fp / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "f1": f1,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _predict_rule(row: Mapping[str, Any]) -> int:
    features = row["features"]
    return int(
        features["scientific_recent_present"]
        and features["industry_recent_present"]
        and features["multiple_origins_present"]
        and not features["unknown_date_present"]
    )


def _evaluate_variant(
    rows: list[dict[str, Any]],
    assignments: Mapping[str, int],
    *,
    include_text: bool,
    include_media: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []
    for fold in range(5):
        train = [row for row in rows if assignments[row["group_id"]] != fold]
        test = [row for row in rows if assignments[row["group_id"]] == fold]
        fit = _train(train, include_text=include_text, include_media=include_media)
        fold_labels: list[int] = []
        fold_predictions: list[int] = []
        for row in test:
            score = _score(fit, row)
            prediction = int(score >= 0.5)
            fold_labels.append(int(row["target"]))
            fold_predictions.append(prediction)
            predictions.append(
                {
                    "candidate_id": row["candidate_id"],
                    "group_id": row["group_id"],
                    "domain": row["domain"],
                    "negative_class": row["negative_class"],
                    "target": row["target"],
                    "fold": fold,
                    "score": score,
                    "prediction": prediction,
                    "threshold": 0.5,
                }
            )
        fold_metrics.append({"fold": fold, **_metrics(fold_labels, fold_predictions)})
    predictions.sort(key=lambda row: row["candidate_id"])
    labels = [int(row["target"]) for row in predictions]
    predicted = [int(row["prediction"]) for row in predictions]
    return predictions, {"folds": fold_metrics, "oof": _metrics(labels, predicted)}


def _subgroup_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in ("domain", "negative_class"):
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            value = row.get(field)
            groups["positive" if value is None else str(value)].append(row)
        result[field] = {
            name: _metrics(
                [int(row["target"]) for row in members],
                [int(row["prediction"]) for row in members],
            )
            for name, members in sorted(groups.items())
        }
    return result


def build_model_report(feature_dir: str | Path) -> tuple[bytes, bytes, bytes, bytes]:
    feature_path = Path(feature_dir)
    manifest_bytes = (feature_path / MANIFEST_FILENAME).read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != FEATURE_TABLE_VERSION:
        raise ValueError("feature table version does not match model evaluator")
    payload = (feature_path / FEATURES_FILENAME).read_bytes()
    if (manifest.get("outputs") or {}).get(FEATURES_FILENAME) != _digest(payload):
        raise ValueError("feature table diverges from its manifest")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("feature table must contain object rows")
    rows.sort(key=lambda row: str(row.get("candidate_id")))
    candidate_ids = [row.get("candidate_id") for row in rows]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("feature table duplicates candidate_id")
    assignments = _assign_folds(rows)

    chosen_predictions, chosen_metrics = _evaluate_variant(
        rows, assignments, include_text=True, include_media=True
    )
    _, structured_metrics = _evaluate_variant(
        rows, assignments, include_text=False, include_media=True
    )
    _, no_media_metrics = _evaluate_variant(
        rows, assignments, include_text=True, include_media=False
    )
    labels = [int(row["target"]) for row in rows]
    majority = int(sum(labels) * 2 >= len(labels))
    majority_metrics = _metrics(labels, [majority] * len(labels))
    rule_metrics = _metrics(labels, [_predict_rule(row) for row in rows])
    subgroup = _subgroup_metrics(chosen_predictions)

    stress: dict[str, Any] = {}
    for domain in sorted({str(row["domain"]) for row in rows}):
        train = [row for row in rows if row["domain"] != domain]
        test = [row for row in rows if row["domain"] == domain]
        if set(int(row["target"]) for row in train) != {0, 1}:
            stress[domain] = {"status": "not_evaluable", "reason": "training_class_missing"}
            continue
        fit = _train(train, include_text=True, include_media=True)
        scores = [_score(fit, row) for row in test]
        stress[domain] = {
            "status": "complete",
            **_metrics(
                [int(row["target"]) for row in test],
                [int(score >= 0.5) for score in scores],
            ),
        }

    final_fit = _train(rows, include_text=True, include_media=True)
    model_value = {
        "schema_version": MODEL_REPORT_VERSION,
        "feature_version": FEATURE_TABLE_VERSION,
        "feature_manifest": _digest(manifest_bytes),
        "threshold": 0.5,
        "regularization_l2": 0.2,
        "epochs": 600,
        "vectorizer": {
            "vocabulary": list(final_fit.vectorizer.vocabulary),
            "idf": list(final_fit.vectorizer.idf),
            "domains": list(final_fit.vectorizer.domains),
            "structured_keys": list(final_fit.vectorizer.structured_keys),
            "numeric_keys": list(final_fit.vectorizer.numeric_keys),
            "numeric_means": list(final_fit.vectorizer.numeric_means),
            "numeric_scales": list(final_fit.vectorizer.numeric_scales),
            "include_text": final_fit.vectorizer.include_text,
        },
        "weights": list(final_fit.weights),
    }
    model_bytes = (json.dumps(model_value, ensure_ascii=False, sort_keys=True) + "\n").encode()

    prediction_bytes = (
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in chosen_predictions
        )
    ).encode()
    errors = [
        {
            "candidate_id": row["candidate_id"],
            "group_id": row["group_id"],
            "domain": row["domain"],
            "negative_class": row["negative_class"],
            "target": row["target"],
            "fold": row["fold"],
            "score": row["score"],
            "prediction": row["prediction"],
            "threshold": row["threshold"],
            "error_type": "false_positive" if row["target"] == 0 else "false_negative",
        }
        for row in chosen_predictions
        if row["target"] != row["prediction"]
    ]
    error_bytes = (
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in errors)
    ).encode()
    negative_false_positive_rate = {
        negative_class: subgroup["negative_class"].get(negative_class, {}).get(
            "false_positive_rate"
        )
        for negative_class in ("mature", "marketing_hype")
    }
    feature_policy = manifest.get("feature_policy")
    counts = manifest.get("counts")
    limitations: list[str] = []
    if not isinstance(counts, dict) or (
        counts.get("accepted_mature") != 50
        or counts.get("accepted_marketing_hype") != 50
    ):
        limitations.append(
            "negative corpus has not reached reviewed 50 mature / 50 marketing_hype"
        )
    if not isinstance(feature_policy, dict) or not feature_policy.get(
        "reviewed_identity_groups_used"
    ):
        limitations.append(
            "cross-corpus identities are exact-normalized only and remain unreviewed"
        )
    elif isinstance(counts, dict) and counts.get("identity_conflicts", 0) != 0:
        limitations.append("reviewed cross-corpus identity conflicts remain unresolved")
    limitations.extend(
        [
            "text style may encode organizer-versus-discovery provenance",
            "metrics are diagnostic and must not be reported as qualification accuracy",
        ]
    )
    if not isinstance(feature_policy, dict) or not feature_policy.get(
        "uncapped_openalex_temporal_counts_used"
    ):
        limitations.insert(
            3,
            "temporal count features are absent until dedicated OpenAlex count coverage exists",
        )
    report_value = {
        "schema_version": MODEL_REPORT_VERSION,
        "feature_table_status": manifest.get("release_status"),
        "feature_manifest": _digest(manifest_bytes),
        "qualification_eligible": manifest.get("qualification_eligible") is True,
        "evaluation_status": (
            "qualification"
            if manifest.get("qualification_eligible") is True
            else "development_only"
        ),
        "counts": manifest.get("counts"),
        "feature_policy": feature_policy,
        "fold_policy": "deterministic-stratified-grouped-5-fold-v2",
        "threshold_policy": "fixed-0.5-no-test-fold-tuning",
        "organizer_recall": chosen_metrics["oof"]["recall"],
        "negative_false_positive_rate": negative_false_positive_rate,
        "error_count": len(errors),
        "variants": {
            "majority": majority_metrics,
            "rule": rule_metrics,
            "logreg_structured_only": structured_metrics,
            "logreg_text_structured": chosen_metrics,
            "logreg_without_media": no_media_metrics,
        },
        "subgroups": subgroup,
        "leave_one_domain_out": stress,
        "limitations": limitations,
        "outputs": {
            PREDICTIONS_FILENAME: _digest(prediction_bytes),
            ERRORS_FILENAME: _digest(error_bytes),
            MODEL_FILENAME: _digest(model_bytes),
        },
    }
    report_bytes = (
        json.dumps(report_value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    return prediction_bytes, error_bytes, report_bytes, model_bytes


def export_model_report(*, feature_dir: str | Path, output_dir: str | Path) -> ModelReportPaths:
    predictions, errors, report, model = build_model_report(feature_dir)
    paths = publish_artifact_bundle(
        {
            PREDICTIONS_FILENAME: predictions,
            ERRORS_FILENAME: errors,
            REPORT_FILENAME: report,
            MODEL_FILENAME: model,
        },
        output_dir,
    )
    return ModelReportPaths(
        predictions=paths[PREDICTIONS_FILENAME],
        report=paths[REPORT_FILENAME],
        model=paths[MODEL_FILENAME],
        errors=paths[ERRORS_FILENAME],
    )
