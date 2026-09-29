"""Merge frozen OpenAlex coverage with source-typed Exa industry enrichment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nextwave.datasets.artifacts import publish_artifact_bundle

from .exa_enrichment_plan import EXA_ENRICHMENT_PLAN_VERSION
from .exa_enrichment_run import EXA_ENRICHMENT_RESULT_VERSION
from .exa_source_policy import EXA_SOURCE_POLICY_VERSION, classify_exa_source

ANALYSIS_COMBINED_ENRICHMENT_VERSION = "analysis-combined-enrichment-result-v1"
_ANALYSIS_PLAN_VERSION = "labeling-enrichment-plan-v2"
_SCIENTIFIC_RESULT_VERSION = "labeling-enrichment-result-v2"


@dataclass(frozen=True, slots=True)
class CombinedEnrichmentPaths:
    documents: Path
    coverage: Path
    excluded_documents: Path
    manifest: Path


def _digest(data: bytes) -> dict[str, Any]:
    return {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    return payload


def _checked(root: Path, manifest: dict[str, Any], filename: str) -> bytes:
    try:
        raw = (root / filename).read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {filename}") from error
    expected = (manifest.get("outputs") or {}).get(filename)
    if expected != _digest(raw):
        raise ValueError(f"{filename} differs from manifest")
    return raw


def _rows(raw: bytes, label: str) -> list[dict[str, Any]]:
    result = []
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8") from error
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label} line {number} is invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{label} line {number} must be an object")
        result.append(row)
    return result


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
        for row in rows
    )


def build_combined_enrichment(
    *,
    analysis_plan_dir: str | Path,
    scientific_result_dir: str | Path,
    exa_plan_dir: str | Path,
    exa_result_dir: str | Path,
) -> dict[str, bytes]:
    analysis_root = Path(analysis_plan_dir)
    analysis_manifest = _read_json(analysis_root / "manifest.json", "analysis manifest")
    if analysis_manifest.get("schema_version") != _ANALYSIS_PLAN_VERSION:
        raise ValueError("analysis plan version is not supported")
    analysis_plan_bytes = _checked(analysis_root, analysis_manifest, "plan.json")
    analysis_plan = json.loads(analysis_plan_bytes)
    candidates_raw = analysis_plan.get("candidates")
    if not isinstance(candidates_raw, list) or not candidates_raw:
        raise ValueError("analysis plan has no candidates")
    candidate_ids: set[str] = set()
    for row in candidates_raw:
        if not isinstance(row, dict):
            raise ValueError("analysis candidates must be objects")
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("analysis candidate IDs are invalid or duplicate")
        if candidate_id in candidate_ids:
            raise ValueError("analysis candidate IDs are invalid or duplicate")
        candidate_ids.add(candidate_id)
    bundle_id = analysis_manifest.get("bundle_id")

    science_root = Path(scientific_result_dir)
    science_manifest = _read_json(science_root / "manifest.json", "scientific manifest")
    if science_manifest.get("schema_version") != _SCIENTIFIC_RESULT_VERSION:
        raise ValueError("scientific result version is not supported")
    if science_manifest.get("bundle_id") != bundle_id:
        raise ValueError("scientific result bundle differs from analysis plan")
    if science_manifest.get("plan") != _digest(analysis_plan_bytes):
        raise ValueError("scientific result was built from another analysis plan")
    science_documents = _rows(
        _checked(science_root, science_manifest, "documents.jsonl"),
        "scientific documents",
    )
    science_coverage = _rows(
        _checked(science_root, science_manifest, "coverage.jsonl"),
        "scientific coverage",
    )

    exa_plan_root = Path(exa_plan_dir)
    exa_plan_manifest = _read_json(exa_plan_root / "manifest.json", "Exa plan manifest")
    if exa_plan_manifest.get("schema_version") != EXA_ENRICHMENT_PLAN_VERSION:
        raise ValueError("Exa plan version is not supported")
    exa_plan_bytes = _checked(exa_plan_root, exa_plan_manifest, "plan.json")
    if exa_plan_manifest.get("bundle_id") != bundle_id:
        raise ValueError("Exa plan bundle differs from analysis plan")
    if (exa_plan_manifest.get("inputs") or {}).get("analysis_plan") != _digest(
        analysis_plan_bytes
    ):
        raise ValueError("Exa plan was built from another analysis plan")

    exa_root = Path(exa_result_dir)
    exa_manifest = _read_json(exa_root / "manifest.json", "Exa result manifest")
    if exa_manifest.get("schema_version") != EXA_ENRICHMENT_RESULT_VERSION:
        raise ValueError("Exa result version is not supported")
    if exa_manifest.get("bundle_id") != bundle_id:
        raise ValueError("Exa result bundle differs from analysis plan")
    if exa_manifest.get("plan") != _digest(exa_plan_bytes):
        raise ValueError("Exa result was built from another Exa plan")
    exa_documents = _rows(
        _checked(exa_root, exa_manifest, "documents.jsonl"), "Exa documents"
    )
    exa_coverage = _rows(
        _checked(exa_root, exa_manifest, "coverage.jsonl"), "Exa coverage"
    )

    documents: list[dict[str, Any]] = []
    science_seen: set[tuple[str, str]] = set()
    for row in science_documents:
        candidate_id = row.get("candidate_id")
        if candidate_id not in candidate_ids:
            raise ValueError("scientific document references an unknown candidate")
        if row.get("connector") != "openalex":
            continue
        pair = (candidate_id, row.get("document_id"))
        if not isinstance(pair[1], str) or pair in science_seen:
            raise ValueError("scientific documents contain invalid duplicates")
        science_seen.add(pair)
        documents.append(row)

    excluded: list[dict[str, Any]] = []
    best_by_url: dict[tuple[str, str], dict[str, Any]] = {}
    for row in exa_documents:
        candidate_id = row.get("candidate_id")
        document = row.get("document")
        if candidate_id not in candidate_ids or not isinstance(document, dict):
            raise ValueError("Exa document references an unknown candidate")
        url = document.get("canonical_url")
        if not isinstance(url, str):
            raise ValueError("Exa document misses canonical_url")
        decision = classify_exa_source(url)
        if not decision.eligible_for_industry:
            excluded.append(
                {
                    "candidate_id": candidate_id,
                    "document_id": document.get("document_id"),
                    "canonical_url": url,
                    "reason": decision.reason,
                    "classified_source_type": decision.source_type,
                }
            )
            continue
        flattened = {
            **document,
            "candidate_id": candidate_id,
            "connector": "exa",
            "request_id": row.get("request_id"),
            "source_type": decision.source_type,
        }
        key = (candidate_id, url.casefold())
        previous = best_by_url.get(key)
        excerpt_length = len(flattened.get("excerpt") or "")
        previous_length = len(previous.get("excerpt") or "") if previous else -1
        document_id = flattened.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("Exa document misses document_id")
        if previous is None or excerpt_length > previous_length or (
            excerpt_length == previous_length
            and document_id < previous["document_id"]
        ):
            best_by_url[key] = flattened
    documents.extend(best_by_url.values())
    documents.sort(key=lambda row: (row["candidate_id"], row["connector"], row["document_id"]))
    excluded.sort(key=lambda row: (row["candidate_id"], row["document_id"] or ""))

    science_complete: dict[str, bool] = {
        candidate_id: False for candidate_id in candidate_ids
    }
    science_coverage_seen: set[str] = set()
    for row in science_coverage:
        candidate_id = row.get("candidate_id")
        if candidate_id not in candidate_ids:
            raise ValueError("scientific coverage references an unknown candidate")
        if row.get("source_class") != "scientific":
            continue
        if candidate_id in science_coverage_seen:
            raise ValueError("scientific coverage duplicates a candidate")
        science_coverage_seen.add(candidate_id)
        science_complete[candidate_id] = row.get("status") == "complete"
    if science_coverage_seen != candidate_ids:
        raise ValueError("scientific coverage misses candidates")
    exa_complete: dict[str, bool] = {
        candidate_id: False for candidate_id in candidate_ids
    }
    exa_seen: set[str] = set()
    for row in exa_coverage:
        candidate_id = row.get("candidate_id")
        if candidate_id not in candidate_ids:
            raise ValueError("Exa coverage references an unknown candidate")
        if candidate_id in exa_seen:
            raise ValueError("Exa coverage duplicates a candidate")
        exa_seen.add(candidate_id)
        exa_complete[candidate_id] = row.get("status") == "complete"
    if exa_seen != candidate_ids:
        raise ValueError("Exa coverage misses candidates")
    if not all(science_complete.values()) or not all(exa_complete.values()):
        raise ValueError("combined enrichment requires complete coverage for every candidate")
    coverage = []
    for candidate_id in sorted(candidate_ids):
        coverage.extend(
            (
                {"candidate_id": candidate_id, "source_class": "scientific", "status": "complete"},
                {"candidate_id": candidate_id, "source_class": "industry", "status": "complete"},
            )
        )
    files = {
        "documents.jsonl": _jsonl(documents),
        "coverage.jsonl": _jsonl(coverage),
        "excluded_documents.jsonl": _jsonl(excluded),
    }
    manifest = {
        "schema_version": ANALYSIS_COMBINED_ENRICHMENT_VERSION,
        "bundle_id": bundle_id,
        "plan": _digest(analysis_plan_bytes),
        "candidate_count": len(candidate_ids),
        "source_policy_version": EXA_SOURCE_POLICY_VERSION,
        "totals": {
            "candidates": len(candidate_ids),
            "scientific_documents": sum(row["connector"] == "openalex" for row in documents),
            "industry_documents": sum(row["connector"] == "exa" for row in documents),
            "excluded_exa_documents": len(excluded),
        },
        "inputs": {
            "analysis_plan": _digest(analysis_plan_bytes),
            "scientific_result_manifest": _digest((science_root / "manifest.json").read_bytes()),
            "exa_plan_manifest": _digest((exa_plan_root / "manifest.json").read_bytes()),
            "exa_result_manifest": _digest((exa_root / "manifest.json").read_bytes()),
        },
        "outputs": {name: _digest(data) for name, data in files.items()},
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    return {**files, "manifest.json": manifest_bytes}


def export_combined_enrichment(
    *,
    analysis_plan_dir: str | Path,
    scientific_result_dir: str | Path,
    exa_plan_dir: str | Path,
    exa_result_dir: str | Path,
    output_dir: str | Path,
) -> CombinedEnrichmentPaths:
    root = Path(output_dir)
    publish_artifact_bundle(
        build_combined_enrichment(
            analysis_plan_dir=analysis_plan_dir,
            scientific_result_dir=scientific_result_dir,
            exa_plan_dir=exa_plan_dir,
            exa_result_dir=exa_result_dir,
        ),
        root,
    )
    return CombinedEnrichmentPaths(
        documents=root / "documents.jsonl",
        coverage=root / "coverage.jsonl",
        excluded_documents=root / "excluded_documents.jsonl",
        manifest=root / "manifest.json",
    )
