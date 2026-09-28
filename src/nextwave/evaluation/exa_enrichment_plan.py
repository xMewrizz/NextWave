"""Build a fast Exa industry-enrichment plan from an analysis candidate plan."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from nextwave.datasets.artifacts import publish_artifact_bundle
from nextwave.sources import QueryPurpose, SourceQuery, build_exa_news_request

EXA_ENRICHMENT_PLAN_VERSION = "analysis-exa-enrichment-plan-v1"
EXA_ENRICHMENT_PLAN_FILENAME = "plan.json"
EXA_ENRICHMENT_MANIFEST_FILENAME = "manifest.json"
_BASE_PLAN_VERSION = "labeling-enrichment-plan-v2"
_BASE_MANIFEST_VERSION = "labeling-enrichment-plan-v2"
_NUM_RESULTS = 10


@dataclass(frozen=True, slots=True)
class ExaEnrichmentPlanPaths:
    plan: Path
    manifest: Path


def _digest(payload: bytes) -> dict[str, Any]:
    return {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    return payload, raw


def _require_date(value: object, label: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a date")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{label} must be a date") from error


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:16]


def build_exa_enrichment_plan(
    analysis_plan_dir: str | Path,
) -> tuple[bytes, dict[str, Any]]:
    """Plan every candidate and both fixed time windows; no ranking or pruning."""

    root = Path(analysis_plan_dir)
    manifest, manifest_bytes = _read_json(root / "manifest.json", "analysis manifest")
    plan, plan_bytes = _read_json(root / "plan.json", "analysis plan")
    if manifest.get("schema_version") != _BASE_MANIFEST_VERSION:
        raise ValueError("analysis manifest version is not supported")
    if manifest.get("plan_role") != "analysis_candidates":
        raise ValueError("Exa enrichment requires an analysis candidate plan")
    if plan.get("schema_version") != _BASE_PLAN_VERSION:
        raise ValueError("analysis plan version is not supported")
    expected = (manifest.get("outputs") or {}).get("plan.json")
    if expected != _digest(plan_bytes):
        raise ValueError("analysis plan checksum differs from manifest")
    bundle = plan.get("bundle")
    candidates = plan.get("candidates")
    windows = plan.get("windows")
    if not isinstance(bundle, dict) or not isinstance(candidates, list):
        raise ValueError("analysis plan misses bundle or candidates")
    if not isinstance(windows, dict):
        raise ValueError("analysis plan misses windows")
    if bundle.get("candidate_count") != len(candidates) or not candidates:
        raise ValueError("analysis candidate_count does not match candidates")
    cutoff = _require_date(plan.get("cutoff_date"), "cutoff_date")
    tasks: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("analysis candidate must be an object")
        candidate_id = candidate.get("candidate_id")
        terms = candidate.get("search_terms")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("analysis candidate_id must not be blank")
        if candidate_id in seen_candidates:
            raise ValueError(f"duplicate analysis candidate {candidate_id!r}")
        seen_candidates.add(candidate_id)
        if not isinstance(terms, list) or not terms or any(
            not isinstance(term, str) or not term.strip() for term in terms
        ):
            raise ValueError(f"candidate {candidate_id!r} search_terms are invalid")
        for window_name in ("previous", "recent"):
            window = windows.get(window_name)
            if not isinstance(window, dict):
                raise ValueError(f"analysis window {window_name!r} is missing")
            published_from = _require_date(window.get("from"), f"{window_name}.from")
            published_until = _require_date(window.get("until"), f"{window_name}.until")
            if window_name == "previous":
                published_until = date.fromordinal(published_until.toordinal() - 1)
            for term_index, term in enumerate(terms, 1):
                query_id = f"query-{_stable_id(candidate_id, window_name, str(term_index), term)}"
                query = SourceQuery(
                    query_id=query_id,
                    analysis_scope_id=f"scope-{_stable_id(candidate_id)}",
                    purpose=QueryPurpose.HISTORICAL_ENRICHMENT,
                    raw_query=term,
                    normalized_query=term.casefold(),
                    search_texts=(term,),
                    published_from=published_from,
                    published_until=published_until,
                    cutoff_date=cutoff,
                    languages=("en", "ru"),
                )
                request = build_exa_news_request(
                    query, search_text=term, num_results=_NUM_RESULTS
                )
                tasks.append(
                    {
                        "candidate_id": candidate_id,
                        "request": request.to_dict(),
                        "search_text": term,
                        "term_rank": term_index,
                        "window": window_name,
                    }
                )
    tasks.sort(
        key=lambda item: (
            item["candidate_id"],
            0 if item["window"] == "previous" else 1,
            item["term_rank"],
        )
    )
    output = {
        "schema_version": EXA_ENRICHMENT_PLAN_VERSION,
        "bundle_id": bundle.get("bundle_id"),
        "cutoff_date": cutoff.isoformat(),
        "policy": {
            "candidate_pruning": False,
            "category": "news",
            "contents": "highlights",
            "num_results_per_term_window": _NUM_RESULTS,
            "windows": ["previous", "recent"],
        },
        "totals": {"candidates": len(candidates), "requests": len(tasks)},
        "tasks": tasks,
    }
    output_bytes = (
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    provenance = {
        "bundle_id": bundle.get("bundle_id"),
        "candidate_count": len(candidates),
        "request_count": len(tasks),
        "inputs": {
            "analysis_manifest": _digest(manifest_bytes),
            "analysis_plan": _digest(plan_bytes),
        },
        "output": _digest(output_bytes),
    }
    return output_bytes, provenance


def export_exa_enrichment_plan(
    *, analysis_plan_dir: str | Path, output_dir: str | Path
) -> ExaEnrichmentPlanPaths:
    plan_bytes, provenance = build_exa_enrichment_plan(analysis_plan_dir)
    manifest_bytes = (
        json.dumps(
            {
                "schema_version": EXA_ENRICHMENT_PLAN_VERSION,
                **provenance,
                "outputs": {EXA_ENRICHMENT_PLAN_FILENAME: provenance["output"]},
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    paths = publish_artifact_bundle(
        {
            EXA_ENRICHMENT_PLAN_FILENAME: plan_bytes,
            EXA_ENRICHMENT_MANIFEST_FILENAME: manifest_bytes,
        },
        output_dir,
    )
    return ExaEnrichmentPlanPaths(
        plan=paths[EXA_ENRICHMENT_PLAN_FILENAME],
        manifest=paths[EXA_ENRICHMENT_MANIFEST_FILENAME],
    )
