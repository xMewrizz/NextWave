"""Select a query-local evidence review shortlist from frozen-model scores."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nextwave.datasets.artifacts import publish_artifact_bundle

from .analysis_inference import ANALYSIS_INFERENCE_VERSION, PREDICTIONS_FILENAME
from .feature_table import MANIFEST_FILENAME, _digest

ANALYSIS_SHORTLIST_VERSION = "analysis-evidence-shortlist-v1"
SHORTLIST_FILENAME = "shortlist.jsonl"
DEFAULT_SHORTLIST_SIZE = 30


@dataclass(frozen=True, slots=True)
class AnalysisShortlistPaths:
    shortlist: Path
    manifest: Path


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def build_analysis_shortlist(
    *,
    inference_dir: str | Path,
    limit: int = DEFAULT_SHORTLIST_SIZE,
    offset: int = 0,
) -> tuple[bytes, bytes]:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("shortlist limit must be a positive integer")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("shortlist offset must be a non-negative integer")
    directory = Path(inference_dir)
    manifest_bytes = (directory / MANIFEST_FILENAME).read_bytes()
    manifest = _read_object(directory / MANIFEST_FILENAME, "analysis inference manifest")
    if manifest.get("schema_version") != ANALYSIS_INFERENCE_VERSION:
        raise ValueError("analysis shortlist requires analysis-inference-v1")
    predictions_bytes = (directory / PREDICTIONS_FILENAME).read_bytes()
    if (manifest.get("outputs") or {}).get(PREDICTIONS_FILENAME) != _digest(
        predictions_bytes
    ):
        raise ValueError("analysis predictions diverge from manifest")
    predictions: list[dict[str, Any]] = []
    seen: set[str] = set()
    scope: str | None = None
    query: str | None = None
    for lineno, line in enumerate(predictions_bytes.decode("utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"prediction line {lineno} must be an object")
        candidate_id = value.get("candidate_id")
        score = value.get("model_score")
        threshold = value.get("decision_threshold")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in seen:
            raise ValueError(f"prediction line {lineno} has invalid candidate_id")
        if type(score) not in {int, float} or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"prediction {candidate_id} has invalid model_score")
        if type(threshold) not in {int, float} or not 0 <= threshold <= 1:
            raise ValueError(f"prediction {candidate_id} has invalid threshold")
        if value.get("prediction") != int(score >= threshold):
            raise ValueError(f"prediction {candidate_id} differs from score and threshold")
        current_scope = value.get("analysis_scope_key")
        current_query = value.get("source_query")
        if not isinstance(current_scope, str) or not isinstance(current_query, str):
            raise ValueError(f"prediction {candidate_id} misses query identity")
        scope = current_scope if scope is None else scope
        query = current_query if query is None else query
        if current_scope != scope or current_query != query:
            raise ValueError("analysis predictions mix queries or scopes")
        seen.add(candidate_id)
        predictions.append(value)
    if not predictions or manifest.get("candidate_count") != len(predictions):
        raise ValueError("analysis inference candidate_count does not match predictions")

    ordered = sorted(
        predictions,
        key=lambda row: (
            -float(row["model_score"]),
            str(row.get("canonical_name", "")).casefold(),
            row["candidate_id"],
        ),
    )
    selected = [
        {
            "schema_version": ANALYSIS_SHORTLIST_VERSION,
            "evidence_rank": rank,
            "model_rank": offset + rank,
            "candidate_id": row["candidate_id"],
            "canonical_name": row.get("canonical_name"),
            "domain": row.get("domain"),
            "analysis_scope_key": row["analysis_scope_key"],
            "source_query": row["source_query"],
            "model_score": row["model_score"],
            "decision_threshold": row["decision_threshold"],
            "preliminary_prediction": row["prediction"],
        }
        for rank, row in enumerate(ordered[offset : offset + limit], 1)
    ]
    shortlist_bytes = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in selected
    )
    output_manifest = {
        "schema_version": ANALYSIS_SHORTLIST_VERSION,
        "artifact_role": "evidence_work_queue_not_final_top15",
        "release_status": manifest.get("release_status"),
        "analysis_scope_key": scope,
        "source_query": query,
        "candidate_count": len(predictions),
        "shortlist_limit": limit,
        "shortlist_offset": offset,
        "selected_count": len(selected),
        "ranking": "model_score_desc_then_name_then_candidate_id",
        "inputs": {"analysis_inference": _digest(manifest_bytes)},
        "outputs": {SHORTLIST_FILENAME: _digest(shortlist_bytes)},
    }
    output_manifest_bytes = (
        json.dumps(output_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return shortlist_bytes, output_manifest_bytes


def export_analysis_shortlist(
    *,
    inference_dir: str | Path,
    output_dir: str | Path,
    limit: int = DEFAULT_SHORTLIST_SIZE,
    offset: int = 0,
) -> AnalysisShortlistPaths:
    shortlist, manifest = build_analysis_shortlist(
        inference_dir=inference_dir, limit=limit, offset=offset
    )
    paths = publish_artifact_bundle(
        {SHORTLIST_FILENAME: shortlist, MANIFEST_FILENAME: manifest}, output_dir
    )
    return AnalysisShortlistPaths(
        shortlist=paths[SHORTLIST_FILENAME], manifest=paths[MANIFEST_FILENAME]
    )
