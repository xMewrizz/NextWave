"""Deterministically merge a full Evidence run with one targeted retry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nextwave.datasets.artifacts import publish_artifact_bundle

from .evidence_llm_run import (
    CLAIMS_FILENAME,
    COVERAGE_FILENAME,
    DOCUMENT_RESULTS_FILENAME,
    ISSUES_FILENAME,
    LABELING_EVIDENCE_LLM_RESULT_VERSION,
    QUALIFICATION_EVIDENCE_MODEL,
    QUALIFICATION_EVIDENCE_PROVIDER,
    REQUEST_RESULTS_FILENAME,
    RESULT_MANIFEST_FILENAME,
)

LABELING_EVIDENCE_LLM_MERGE_VERSION = "labeling-evidence-llm-merge-v1"
LABELING_EVIDENCE_LLM_MERGED_RESULT_VERSION = (
    "labeling-evidence-llm-merged-result-v1"
)

_OUTPUT_FILES = (
    REQUEST_RESULTS_FILENAME,
    CLAIMS_FILENAME,
    DOCUMENT_RESULTS_FILENAME,
    ISSUES_FILENAME,
    COVERAGE_FILENAME,
)
_SHARED_MANIFEST_FIELDS = (
    "bundle_id",
    "cutoff_date",
    "plan_manifest",
    "plan_files",
    "provider",
    "model",
    "extractor_id",
    "max_output_tokens",
    "concurrency",
)


@dataclass(frozen=True)
class LabelingEvidenceLlmMergePaths:
    manifest: Path
    request_results: Path
    claims: Path
    document_results: Path
    issues: Path
    coverage: Path


def _digest(payload: bytes) -> dict[str, Any]:
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {error}") from None
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_rows(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read {label}: {error}") from None
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label}:{line_number}: invalid JSON: {error}") from None
        if not isinstance(row, dict):
            raise ValueError(f"{label}:{line_number}: row must be an object")
        rows.append(row)
    return rows


def _manifest_entry(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"manifest output {label!r} must be an object")
    size = value.get("size_bytes")
    sha = value.get("sha256")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError(f"manifest output {label!r} has invalid size")
    if not isinstance(sha, str) or len(sha) != 64:
        raise ValueError(f"manifest output {label!r} has invalid checksum")
    return value


def _load_result(path: Path, label: str) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    manifest_path = path / RESULT_MANIFEST_FILENAME
    manifest = _read_json(manifest_path, f"{label} manifest")
    if manifest.get("schema_version") != LABELING_EVIDENCE_LLM_RESULT_VERSION:
        raise ValueError(f"{label} has incompatible result version")
    if manifest.get("provider") != QUALIFICATION_EVIDENCE_PROVIDER:
        raise ValueError(f"{label} was not produced by qualification provider")
    if manifest.get("model") != QUALIFICATION_EVIDENCE_MODEL:
        raise ValueError(f"{label} was not produced by qualification Evidence model")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(_OUTPUT_FILES):
        raise ValueError(f"{label} manifest outputs diverge")
    rows: dict[str, list[dict[str, Any]]] = {}
    for name in _OUTPUT_FILES:
        payload_path = path / name
        try:
            payload = payload_path.read_bytes()
        except OSError as error:
            raise ValueError(f"cannot read {label} {name}: {error}") from None
        expected = _manifest_entry(outputs[name], name)
        if _digest(payload) != expected:
            raise ValueError(f"{label} {name} checksum or size mismatch")
        rows[name] = _read_rows(payload_path, f"{label} {name}")
    return manifest, rows


def _unique_by(
    rows: list[dict[str, Any]], key: str, label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} row has invalid {key}")
        if value in result:
            raise ValueError(f"{label} has duplicate {key} {value!r}")
        result[value] = row
    return result


def _group_by_candidate(
    rows: list[dict[str, Any]], label: str
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        candidate = row.get("candidate_id")
        if not isinstance(candidate, str) or not candidate:
            raise ValueError(f"{label} row has invalid candidate_id")
        grouped.setdefault(candidate, []).append(row)
    return grouped


def _render(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def merge_evidence_llm_results(
    *, primary_dir: str | Path, retry_dir: str | Path, output_dir: str | Path
) -> LabelingEvidenceLlmMergePaths:
    """Publish one result, using retry only when it materially improves a candidate."""

    primary_path = Path(primary_dir)
    retry_path = Path(retry_dir)
    primary_manifest, primary_rows = _load_result(primary_path, "primary")
    retry_manifest, retry_rows = _load_result(retry_path, "retry")
    for field in _SHARED_MANIFEST_FIELDS:
        if primary_manifest.get(field) != retry_manifest.get(field):
            raise ValueError(f"primary and retry {field} diverge")
    if primary_manifest.get("applied_candidate_filter") is not None:
        raise ValueError("primary result must be an unfiltered full run")
    retry_filter = retry_manifest.get("applied_candidate_filter")
    if (
        not isinstance(retry_filter, list)
        or not retry_filter
        or any(not isinstance(item, str) or not item for item in retry_filter)
        or len(set(retry_filter)) != len(retry_filter)
    ):
        raise ValueError("retry result must have a unique non-empty candidate filter")
    retry_candidates = set(retry_filter)

    primary_coverage = _unique_by(
        primary_rows[COVERAGE_FILENAME], "candidate_id", "primary coverage"
    )
    retry_coverage = _unique_by(
        retry_rows[COVERAGE_FILENAME], "candidate_id", "retry coverage"
    )
    if set(primary_coverage) != set(retry_coverage):
        raise ValueError("primary and retry candidate coverage diverge")
    if not retry_candidates <= set(primary_coverage):
        raise ValueError("retry filter contains an unknown candidate")

    primary_requests = _unique_by(
        primary_rows[REQUEST_RESULTS_FILENAME], "candidate_id", "primary requests"
    )
    retry_requests = _unique_by(
        retry_rows[REQUEST_RESULTS_FILENAME], "candidate_id", "retry requests"
    )
    if set(primary_requests) != set(retry_requests):
        raise ValueError("primary and retry planned request sets diverge")
    if any(row.get("status") == "not_run" for row in primary_requests.values()):
        raise ValueError("primary result is not a full run")

    primary_groups = {
        name: _group_by_candidate(primary_rows[name], f"primary {name}")
        for name in (CLAIMS_FILENAME, DOCUMENT_RESULTS_FILENAME, ISSUES_FILENAME)
    }
    retry_groups = {
        name: _group_by_candidate(retry_rows[name], f"retry {name}")
        for name in (CLAIMS_FILENAME, DOCUMENT_RESULTS_FILENAME, ISSUES_FILENAME)
    }

    selected_retry: set[str] = set()
    recovered_failures: set[str] = set()
    for candidate in sorted(retry_candidates):
        retry_status = retry_coverage[candidate].get("status")
        primary_status = primary_coverage[candidate].get("status")
        retry_claims = retry_groups[CLAIMS_FILENAME].get(candidate, [])
        if retry_status == "complete" and retry_claims:
            selected_retry.add(candidate)
        elif primary_status == "failed" and retry_status == "complete":
            selected_retry.add(candidate)
            recovered_failures.add(candidate)

    def source_for(candidate: str) -> str:
        return "retry" if candidate in selected_retry else "primary"

    request_rows: list[dict[str, Any]] = []
    for candidate in sorted(primary_requests):
        source = source_for(candidate)
        request_rows.append(
            dict((retry_requests if source == "retry" else primary_requests)[candidate])
        )
    coverage_rows: list[dict[str, Any]] = []
    for candidate in sorted(primary_coverage):
        source = source_for(candidate)
        coverage_rows.append(
            dict((retry_coverage if source == "retry" else primary_coverage)[candidate])
        )

    selected_rows: dict[str, list[dict[str, Any]]] = {}
    for name in (CLAIMS_FILENAME, DOCUMENT_RESULTS_FILENAME, ISSUES_FILENAME):
        combined: list[dict[str, Any]] = []
        candidates = set(primary_groups[name]) | set(retry_groups[name])
        for candidate in sorted(candidates):
            groups = retry_groups if candidate in selected_retry else primary_groups
            combined.extend(dict(row) for row in groups[name].get(candidate, []))
        selected_rows[name] = combined

    claim_rows = sorted(
        selected_rows[CLAIMS_FILENAME],
        key=lambda row: (
            row.get("candidate_id", ""),
            row.get("document_id", ""),
            row.get("claim_id", ""),
        ),
    )
    document_rows = sorted(
        selected_rows[DOCUMENT_RESULTS_FILENAME],
        key=lambda row: (row.get("candidate_id", ""), row.get("document_id", "")),
    )
    issue_rows = sorted(
        selected_rows[ISSUES_FILENAME],
        key=lambda row: (
            row.get("candidate_id", ""),
            row.get("document_id") or "",
            row.get("code", ""),
            row.get("message", ""),
        ),
    )
    success = sum(row.get("status") == "success" for row in request_rows)
    failed = sum(row.get("status") == "failed" for row in request_rows)
    not_run = sum(row.get("status") == "not_run" for row in request_rows)
    no_input = sum(row.get("status") == "no_input" for row in coverage_rows)
    totals = {
        "candidates": len(coverage_rows),
        "planned_tasks": len(request_rows),
        "effective_calls": sum(
            isinstance(row.get("attempts"), int) and row.get("attempts", 0) > 0
            for row in request_rows
        ),
        "source_calls": (
            primary_manifest.get("totals", {}).get("called", 0)
            + retry_manifest.get("totals", {}).get("called", 0)
        ),
        "success": success,
        "failed": failed,
        "not_run": not_run,
        "no_input": no_input,
        "claims": len(claim_rows),
        "issues": len(issue_rows),
        "candidates_with_claims": len({row["candidate_id"] for row in claim_rows}),
        "retry_candidates": len(retry_candidates),
        "retry_candidates_selected": len(selected_retry),
    }
    files = {
        REQUEST_RESULTS_FILENAME: _render(request_rows),
        CLAIMS_FILENAME: _render(claim_rows),
        DOCUMENT_RESULTS_FILENAME: _render(document_rows),
        ISSUES_FILENAME: _render(issue_rows),
        COVERAGE_FILENAME: _render(coverage_rows),
    }
    manifest = {
        "schema_version": LABELING_EVIDENCE_LLM_MERGED_RESULT_VERSION,
        "merge_version": LABELING_EVIDENCE_LLM_MERGE_VERSION,
        "analysis_status": "complete" if failed == 0 and not_run == 0 else "partial",
        **{field: primary_manifest[field] for field in _SHARED_MANIFEST_FIELDS},
        "selection_policy": (
            "retry replaces primary only when it adds at least one accepted claim "
            "or converts a failed primary task to complete"
        ),
        "selected_retry_candidates": sorted(selected_retry),
        "recovered_failed_candidates": sorted(recovered_failures),
        "inputs": {
            "primary_manifest": _digest(
                (primary_path / RESULT_MANIFEST_FILENAME).read_bytes()
            ),
            "retry_manifest": _digest(
                (retry_path / RESULT_MANIFEST_FILENAME).read_bytes()
            ),
        },
        "totals": totals,
        "outputs": {name: _digest(payload) for name, payload in files.items()},
    }
    files[RESULT_MANIFEST_FILENAME] = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    paths = publish_artifact_bundle(files, output_dir)
    return LabelingEvidenceLlmMergePaths(
        manifest=paths[RESULT_MANIFEST_FILENAME],
        request_results=paths[REQUEST_RESULTS_FILENAME],
        claims=paths[CLAIMS_FILENAME],
        document_results=paths[DOCUMENT_RESULTS_FILENAME],
        issues=paths[ISSUES_FILENAME],
        coverage=paths[COVERAGE_FILENAME],
    )
