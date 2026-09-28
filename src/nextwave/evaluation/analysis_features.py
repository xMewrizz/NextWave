"""Build unlabeled model features for one query-specific analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nextwave.datasets.artifacts import publish_artifact_bundle

from .feature_table import (
    FEATURE_TABLE_VERSION,
    FEATURES_FILENAME,
    MANIFEST_FILENAME,
    _checked_file,
    _coverage_index,
    _digest,
    _document_features,
    _index,
    _manifest_digest,
    _read_json,
    _rows_from_bytes,
    _temporal_feature_index,
)

ANALYSIS_FEATURE_TABLE_VERSION = "analysis-feature-table-v1"
_ENRICHMENT_PLAN_VERSION = "labeling-enrichment-plan-v2"
_ENRICHMENT_RESULT_VERSION = "labeling-enrichment-result-v2"
_COMBINED_ENRICHMENT_RESULT_VERSION = "analysis-combined-enrichment-result-v1"


@dataclass(frozen=True, slots=True)
class AnalysisFeatureTablePaths:
    features: Path
    manifest: Path


def build_analysis_feature_table(
    *,
    analysis_plan_dir: str | Path,
    enrichment_result_dir: str | Path,
    temporal_count_dir: str | Path,
) -> tuple[bytes, bytes]:
    plan_dir = Path(analysis_plan_dir)
    plan_manifest = _read_json(plan_dir / MANIFEST_FILENAME, "analysis plan manifest")
    if (
        plan_manifest.get("schema_version") != _ENRICHMENT_PLAN_VERSION
        or plan_manifest.get("plan_role") != "analysis_candidates"
    ):
        raise ValueError("analysis features require an analysis_candidates enrichment plan")
    plan_payload = _checked_file(plan_dir, plan_manifest, "plan.json")
    plan = json.loads(plan_payload)
    if not isinstance(plan, dict) or plan.get("schema_version") != _ENRICHMENT_PLAN_VERSION:
        raise ValueError("analysis enrichment plan schema is invalid")
    raw_candidates = plan.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("analysis enrichment plan needs candidates")
    candidates = _index(raw_candidates, "candidate_id", "analysis candidate")
    if plan_manifest.get("candidate_count") != len(candidates):
        raise ValueError("analysis plan candidate_count does not match candidates")
    scopes = {row.get("analysis_scope_key") for row in candidates.values()}
    queries: set[str] = set()
    for candidate_id, candidate in candidates.items():
        origin = candidate.get("origin")
        source_query = origin.get("source_query") if isinstance(origin, dict) else None
        if not isinstance(source_query, str) or not source_query.strip():
            raise ValueError(f"analysis candidate {candidate_id} needs origin.source_query")
        queries.add(source_query.strip())
    domains = {row.get("domain") for row in candidates.values()}
    if len(scopes) != 1 or len(queries) != 1 or len(domains) != 1:
        raise ValueError("analysis candidates must share one query, domain, and scope")

    result_dir = Path(enrichment_result_dir)
    result_manifest = _read_json(
        result_dir / MANIFEST_FILENAME, "analysis enrichment result manifest"
    )
    if result_manifest.get("schema_version") not in {
        _ENRICHMENT_RESULT_VERSION,
        _COMBINED_ENRICHMENT_RESULT_VERSION,
    }:
        raise ValueError("analysis enrichment result version is not supported")
    if result_manifest.get("bundle_id") != plan_manifest.get("bundle_id"):
        raise ValueError("analysis enrichment result bundle_id differs from plan")
    if result_manifest.get("plan") != _digest(plan_payload):
        raise ValueError("analysis enrichment result was built from another plan")
    totals = result_manifest.get("totals")
    if not isinstance(totals, dict) or totals.get("candidates") != len(candidates):
        raise ValueError("analysis enrichment result candidate total is invalid")

    candidate_ids = set(candidates)
    coverage = _coverage_index(result_dir, candidate_ids)
    documents = _document_features(result_dir, candidate_ids)
    if any(not all(classes.values()) for classes in coverage.values()):
        raise ValueError("analysis enrichment coverage must be complete for every candidate")

    temporal_dir = Path(temporal_count_dir)
    temporal_manifest = _read_json(
        temporal_dir / MANIFEST_FILENAME, "analysis temporal count manifest"
    )
    temporal_rows = _rows_from_bytes(
        _checked_file(
            temporal_dir, temporal_manifest, "candidate_temporal_features.jsonl"
        ),
        "analysis temporal features",
    )
    temporal_raw = _index(temporal_rows, "candidate_id", "analysis temporal feature")
    if set(temporal_raw) != candidate_ids:
        raise ValueError("analysis temporal result candidate roster differs from plan")
    for candidate_id, candidate in candidates.items():
        temporal = temporal_raw[candidate_id]
        if (
            temporal.get("analysis_scope_key") != candidate.get("analysis_scope_key")
            or temporal.get("domain") != candidate.get("domain")
            or temporal.get("search_text") != candidate.get("search_terms", [None])[0]
            or temporal.get("role") != "analysis"
        ):
            raise ValueError(f"analysis temporal identity differs for {candidate_id}")
    temporal = _temporal_feature_index(temporal_dir, candidate_ids)

    rows: list[dict[str, Any]] = []
    for candidate_id in sorted(candidate_ids):
        candidate = candidates[candidate_id]
        canonical = candidate.get("canonical_name")
        aliases = candidate.get("aliases")
        if not isinstance(canonical, str) or not canonical.strip():
            raise ValueError(f"analysis candidate {candidate_id} needs canonical_name")
        if not isinstance(aliases, list) or any(not isinstance(alias, str) for alias in aliases):
            raise ValueError(f"analysis candidate {candidate_id} aliases are invalid")
        rows.append(
            {
                "schema_version": FEATURE_TABLE_VERSION,
                "candidate_id": candidate_id,
                "canonical_name": canonical.strip(),
                "aliases": aliases,
                "domain": candidate["domain"],
                "analysis_scope_key": candidate["analysis_scope_key"],
                "source_query": candidate["origin"]["source_query"].strip(),
                "cutoff_date": candidate["cutoff_date"],
                "model_text": " ; ".join([canonical.strip(), *aliases]),
                "features": {
                    "scientific_coverage_complete": coverage[candidate_id]["scientific"],
                    "industry_coverage_complete": coverage[candidate_id]["industry"],
                    **documents[candidate_id],
                    **temporal[candidate_id],
                },
            }
        )
    features_bytes = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )
    manifest_value = {
        "schema_version": ANALYSIS_FEATURE_TABLE_VERSION,
        "feature_version": FEATURE_TABLE_VERSION,
        "analysis_scope_key": next(iter(scopes)),
        "source_query": next(iter(queries)),
        "domain": next(iter(domains)),
        "candidate_count": len(rows),
        "inputs": {
            "analysis_plan": _manifest_digest(plan_dir),
            "enrichment_result": _manifest_digest(result_dir),
            "temporal_counts": _manifest_digest(temporal_dir),
        },
        "outputs": {FEATURES_FILENAME: _digest(features_bytes)},
    }
    manifest_bytes = (
        json.dumps(manifest_value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return features_bytes, manifest_bytes


def export_analysis_feature_table(
    *,
    analysis_plan_dir: str | Path,
    enrichment_result_dir: str | Path,
    temporal_count_dir: str | Path,
    output_dir: str | Path,
) -> AnalysisFeatureTablePaths:
    features, manifest = build_analysis_feature_table(
        analysis_plan_dir=analysis_plan_dir,
        enrichment_result_dir=enrichment_result_dir,
        temporal_count_dir=temporal_count_dir,
    )
    paths = publish_artifact_bundle(
        {FEATURES_FILENAME: features, MANIFEST_FILENAME: manifest}, output_dir
    )
    return AnalysisFeatureTablePaths(
        features=paths[FEATURES_FILENAME], manifest=paths[MANIFEST_FILENAME]
    )
