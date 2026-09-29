"""Build deterministic OpenAlex count tasks without calling the network."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from nextwave.datasets.artifacts import publish_artifact_bundle

TEMPORAL_COUNT_PLAN_VERSION = "openalex-temporal-count-plan-v4"
PLAN_FILENAME = "plan.json"
MANIFEST_FILENAME = "manifest.json"
_ENRICHMENT_PLAN_VERSION = "labeling-enrichment-plan-v2"
_WINDOWS = (
    ("previous", "2024-09-15", "2025-09-14"),
    ("recent", "2025-09-15", "2026-09-15"),
)


def _windows_for_cutoff(cutoff: date) -> tuple[tuple[str, str, str], ...]:
    recent_from = cutoff - timedelta(days=365)
    previous_from = cutoff - timedelta(days=730)
    return (
        ("previous", previous_from.isoformat(), (recent_from - timedelta(days=1)).isoformat()),
        ("recent", recent_from.isoformat(), cutoff.isoformat()),
    )
_SCOPE_QUERIES = {
    "edge-v1": '"edge computing" OR "edge AI"',
    "ai-security-v1": '"AI security" OR "machine learning security" OR "model security"',
    "industrial-ai-v1": ('"industrial AI" OR "intelligent manufacturing" OR "smart manufacturing"'),
    "ai-infrastructure-v1": (
        '"AI infrastructure" OR "machine learning infrastructure" OR "AI training infrastructure"'
    ),
    "robotics-v1": 'robotics OR "autonomous robots" OR "human robot interaction"',
    "fintech-v1": '"financial AI" OR fintech OR "digital payments" OR "risk analytics"',
}


def _dynamic_scope_query(search_texts: object, label: str) -> str:
    if not isinstance(search_texts, list) or not search_texts:
        raise ValueError(f"{label} search_texts must be a non-empty list")
    clauses: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(search_texts, 1):
        normalized = _normalized_term(value, f"{label} search_text {index}")
        normalized = " ".join(normalized.replace('"', " ").split())
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        clauses.append(f'"{normalized}"')
    if not clauses:
        raise ValueError(f"{label} search_texts do not contain usable text")
    return " OR ".join(clauses)


@dataclass(frozen=True, slots=True)
class TemporalCountPlanPaths:
    plan: Path
    manifest: Path


def _digest(payload: bytes) -> dict[str, Any]:
    return {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _load_enrichment_plan(directory: Path) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    manifest = _read_json(directory / MANIFEST_FILENAME, f"{directory.name} manifest")
    if manifest.get("schema_version") != _ENRICHMENT_PLAN_VERSION:
        raise ValueError(f"{directory.name} is not an enrichment plan v2")
    payload = (directory / PLAN_FILENAME).read_bytes()
    if (manifest.get("outputs") or {}).get(PLAN_FILENAME) != _digest(payload):
        raise ValueError(f"{directory.name} plan diverges from manifest")
    try:
        plan = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError(f"{directory.name} plan is invalid JSON") from error
    if not isinstance(plan, dict) or not isinstance(plan.get("candidates"), list):
        raise ValueError(f"{directory.name} plan candidates must be a list")
    if manifest.get("candidate_count") != len(plan["candidates"]):
        raise ValueError(f"{directory.name} candidate_count does not match plan")
    return manifest, plan, payload


def _normalized_term(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = " ".join(unicodedata.normalize("NFKC", value).split())
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    if len(normalized) > 160:
        raise ValueError(f"{field_name} is too long")
    return normalized


def _task_id(spec: dict[str, Any]) -> str:
    identity = json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256((TEMPORAL_COUNT_PLAN_VERSION + "|" + identity).encode()).hexdigest()[
        :20
    ]
    return f"count-{digest}"


def _task(
    *,
    entity_type: str,
    entity_id: str,
    analysis_scope_key: str,
    search_text: str,
    scope_search_text: str,
    window: str,
    published_from: str,
    published_until: str,
) -> dict[str, Any]:
    spec = {
        "entity_type": entity_type,
        "entity_id": entity_id,
        "analysis_scope_key": analysis_scope_key,
        "search_text": search_text,
        "scope_search_text": scope_search_text,
        "search_mode": (
            "scoped_proximity_5" if entity_type == "candidate" else "boolean_scope"
        ),
        "window": window,
        "published_from": published_from,
        "published_until": published_until,
        "languages": ["en", "ru"],
        "channel": "text",
        "per_page": 1,
        "count_field": "meta.count",
        "population": "works_with_abstract_in_openalex",
    }
    return {"count_id": _task_id(spec), **spec}


def _collect_candidates(
    *,
    directory: Path,
    role: str,
    candidates: dict[str, dict[str, Any]],
    required_cutoff: str | None = "2026-09-15",
    dynamic_scope_queries: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], date]:
    manifest, plan, payload = _load_enrichment_plan(directory)
    cutoff_raw = plan.get("cutoff_date")
    try:
        cutoff = date.fromisoformat(cutoff_raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{directory.name} cutoff is not an ISO date") from error
    if required_cutoff is not None and cutoff_raw != required_cutoff:
        raise ValueError(f"{directory.name} cutoff is not frozen")
    bundle_id = manifest.get("bundle_id")
    if not isinstance(bundle_id, str) or not bundle_id:
        raise ValueError(f"{directory.name} misses bundle_id")
    logical_input = {
        "role": role,
        "bundle_id": bundle_id,
        "candidate_count": len(plan["candidates"]),
    }
    if dynamic_scope_queries is not None:
        analysis_scope = plan.get("analysis_scope")
        if analysis_scope is not None:
            if not isinstance(analysis_scope, dict):
                raise ValueError(f"{directory.name} analysis_scope must be an object")
            scope_id = analysis_scope.get("scope_id")
            if not isinstance(scope_id, str) or not scope_id:
                raise ValueError(f"{directory.name} analysis_scope has invalid scope_id")
            dynamic_scope_queries[scope_id] = _dynamic_scope_query(
                analysis_scope.get("search_texts"), f"{directory.name} analysis_scope"
            )
    for raw in plan["candidates"]:
        if not isinstance(raw, dict):
            raise ValueError(f"{directory.name} candidate must be an object")
        candidate_id = raw.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(f"{directory.name} candidate misses candidate_id")
        if candidate_id in candidates:
            raise ValueError(f"candidate_id {candidate_id!r} occurs in both plans")
        scope = raw.get("analysis_scope_key")
        if not isinstance(scope, str) or not scope:
            raise ValueError(f"candidate {candidate_id} has invalid analysis scope")
        if scope not in _SCOPE_QUERIES:
            if dynamic_scope_queries is None:
                raise ValueError(f"candidate {candidate_id} has unknown analysis scope")
            if scope not in dynamic_scope_queries:
                raise ValueError(
                    f"candidate {candidate_id} differs from analysis_scope"
                )
        terms = raw.get("search_terms")
        if not isinstance(terms, list) or not terms:
            raise ValueError(f"candidate {candidate_id} needs reviewed search terms")
        search_text = _normalized_term(terms[0], f"candidate {candidate_id} search term")
        if raw.get("cutoff_date") != cutoff_raw:
            raise ValueError(f"candidate {candidate_id} cutoff differs from plan")
        candidates[candidate_id] = {
            "candidate_id": candidate_id,
            "role": role,
            "domain": raw.get("domain"),
            "analysis_scope_key": scope,
            "search_text": search_text,
        }
    return logical_input, {**logical_input, "plan": _digest(payload)}, cutoff


def _render_temporal_count_plan(
    *,
    plan_inputs: list[dict[str, Any]],
    manifest_inputs: list[dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    scopes: tuple[str, ...],
    scope_queries: dict[str, str],
    cutoff: date,
) -> tuple[bytes, bytes]:
    if not candidates:
        raise ValueError("temporal count plan requires candidates")
    if not scopes or any(scope not in scope_queries for scope in scopes):
        raise ValueError("temporal count plan requires known scopes")
    if len(scopes) != len(set(scopes)):
        raise ValueError("temporal count scopes must be unique")

    windows = _windows_for_cutoff(cutoff)
    tasks: list[dict[str, Any]] = []
    for candidate in sorted(candidates.values(), key=lambda row: row["candidate_id"]):
        for window, start, end in windows:
            tasks.append(
                _task(
                    entity_type="candidate",
                    entity_id=candidate["candidate_id"],
                    analysis_scope_key=candidate["analysis_scope_key"],
                    search_text=candidate["search_text"],
                    scope_search_text=scope_queries[candidate["analysis_scope_key"]],
                    window=window,
                    published_from=start,
                    published_until=end,
                )
            )
    for scope in sorted(scopes):
        search_text = scope_queries[scope]
        for window, start, end in windows:
            tasks.append(
                _task(
                    entity_type="scope",
                    entity_id=scope,
                    analysis_scope_key=scope,
                    search_text=search_text,
                    scope_search_text=search_text,
                    window=window,
                    published_from=start,
                    published_until=end,
                )
            )
    tasks.sort(key=lambda row: (row["entity_type"], row["entity_id"], row["window"]))
    count_ids = [task["count_id"] for task in tasks]
    if len(set(count_ids)) != len(count_ids):
        raise ValueError("temporal count task IDs collide")
    plan_value = {
        "schema_version": TEMPORAL_COUNT_PLAN_VERSION,
        "cutoff_date": cutoff.isoformat(),
        "windows": {
            name: {"from": start, "until": end, "inclusive": True}
            for name, start, end in windows
        },
        "scope_queries": {scope: scope_queries[scope] for scope in sorted(scopes)},
        "retrieval_policy": {
            "connector": "openalex",
            "channel": "text",
            "per_page": 1,
            "languages": ["en", "ru"],
            "count_field": "meta.count",
            "population": "works_with_abstract_in_openalex",
            "candidate_search_mode": "quoted_phrase_proximity_5_within_frozen_scope",
            "scope_search_mode": "frozen_boolean_query",
            "successful_zero": "covered_zero",
            "failure": "unknown",
        },
        "inputs": plan_inputs,
        "candidates": [candidates[key] for key in sorted(candidates)],
        "tasks": tasks,
    }
    plan_bytes = (
        json.dumps(plan_value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    manifest_value = {
        "schema_version": TEMPORAL_COUNT_PLAN_VERSION,
        "cutoff_date": cutoff.isoformat(),
        "counts": {
            "candidates": len(candidates),
            "scopes": len(scopes),
            "candidate_tasks": len(candidates) * len(windows),
            "scope_tasks": len(scopes) * len(windows),
            "tasks": len(tasks),
        },
        "inputs": manifest_inputs,
        "outputs": {PLAN_FILENAME: _digest(plan_bytes)},
    }
    manifest_bytes = (
        json.dumps(manifest_value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    return plan_bytes, manifest_bytes


def build_temporal_count_plan(
    *, positive_plan_dir: str | Path, negative_plan_dir: str | Path
) -> tuple[bytes, bytes]:
    candidates: dict[str, dict[str, Any]] = {}
    plan_inputs: list[dict[str, Any]] = []
    manifest_inputs: list[dict[str, Any]] = []
    for role, raw_directory in (
        ("positive", positive_plan_dir),
        ("negative", negative_plan_dir),
    ):
        logical, manifest_input, cutoff = _collect_candidates(
            directory=Path(raw_directory), role=role, candidates=candidates
        )
        plan_inputs.append(logical)
        manifest_inputs.append(manifest_input)
    return _render_temporal_count_plan(
        plan_inputs=plan_inputs,
        manifest_inputs=manifest_inputs,
        candidates=candidates,
        scopes=tuple(sorted(_SCOPE_QUERIES)),
        scope_queries=_SCOPE_QUERIES,
        cutoff=cutoff,
    )


def build_analysis_temporal_count_plan(
    *, analysis_plan_dir: str | Path
) -> tuple[bytes, bytes]:
    directory = Path(analysis_plan_dir)
    manifest = _read_json(directory / MANIFEST_FILENAME, f"{directory.name} manifest")
    if manifest.get("plan_role") != "analysis_candidates":
        raise ValueError("analysis temporal counts require an analysis_candidates plan")
    candidates: dict[str, dict[str, Any]] = {}
    dynamic_scope_queries: dict[str, str] = {}
    logical, manifest_input, cutoff = _collect_candidates(
        directory=directory,
        role="analysis",
        candidates=candidates,
        required_cutoff=None,
        dynamic_scope_queries=dynamic_scope_queries,
    )
    scopes = tuple(
        sorted({str(candidate["analysis_scope_key"]) for candidate in candidates.values()})
    )
    if len(scopes) != 1:
        raise ValueError("analysis enrichment plan must contain exactly one scope")
    scope_queries = {
        scope: _SCOPE_QUERIES.get(scope, dynamic_scope_queries.get(scope, ""))
        for scope in scopes
    }
    return _render_temporal_count_plan(
        plan_inputs=[logical],
        manifest_inputs=[manifest_input],
        candidates=candidates,
        scopes=scopes,
        scope_queries=scope_queries,
        cutoff=cutoff,
    )


def export_temporal_count_plan(
    *,
    positive_plan_dir: str | Path,
    negative_plan_dir: str | Path,
    output_dir: str | Path,
) -> TemporalCountPlanPaths:
    plan, manifest = build_temporal_count_plan(
        positive_plan_dir=positive_plan_dir,
        negative_plan_dir=negative_plan_dir,
    )
    paths = publish_artifact_bundle({PLAN_FILENAME: plan, MANIFEST_FILENAME: manifest}, output_dir)
    return TemporalCountPlanPaths(plan=paths[PLAN_FILENAME], manifest=paths[MANIFEST_FILENAME])


def export_analysis_temporal_count_plan(
    *, analysis_plan_dir: str | Path, output_dir: str | Path
) -> TemporalCountPlanPaths:
    plan, manifest = build_analysis_temporal_count_plan(
        analysis_plan_dir=analysis_plan_dir
    )
    paths = publish_artifact_bundle(
        {PLAN_FILENAME: plan, MANIFEST_FILENAME: manifest}, output_dir
    )
    return TemporalCountPlanPaths(
        plan=paths[PLAN_FILENAME], manifest=paths[MANIFEST_FILENAME]
    )
