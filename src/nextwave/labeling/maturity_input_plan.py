"""Build an Evidence-LLM-compatible input focused on maturity facts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nextwave.datasets.artifacts import publish_artifact_bundle

from .contracts import LABELING_CUTOFF_DATE
from .enrichment_plan import LABELING_ENRICHMENT_PLAN_VERSION
from .evidence_input_plan import LABELING_EVIDENCE_INPUT_PLAN_VERSION
from .rubric_audit import (
    LABELING_RUBRIC_AUDIT_VERSION,
    MANIFEST_FILENAME,
    MATURITY_QUEUE_FILENAME,
)

MATURITY_INPUT_POLICY_VERSION = "maturity-rubric-input-v1"
DOCUMENTS_FILENAME = "evidence_input_documents.jsonl"
COVERAGE_FILENAME = "coverage.jsonl"


@dataclass(frozen=True, slots=True)
class MaturityEvidenceInputPaths:
    manifest: Path
    documents: Path
    coverage: Path


def _digest(payload: bytes) -> dict[str, Any]:
    return {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return (
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    ).encode("utf-8")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _checked_payload(directory: Path, filename: str, manifest: dict[str, Any]) -> bytes:
    outputs = manifest.get("outputs")
    entry = outputs.get(filename) if isinstance(outputs, dict) else None
    if not isinstance(entry, dict):
        raise ValueError(f"manifest misses {filename}")
    payload = (directory / filename).read_bytes()
    if entry.get("size_bytes") != len(payload):
        raise ValueError(f"{filename} size mismatch")
    if entry.get("sha256") != hashlib.sha256(payload).hexdigest():
        raise ValueError(f"{filename} checksum mismatch")
    return payload


def _rows(payload: bytes, label: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for lineno, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label} line {lineno} is invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{label} line {lineno} must be an object")
        result.append(row)
    return result


def build_maturity_evidence_input(
    *, plan_dir: str | Path, audit_dir: str | Path
) -> tuple[bytes, bytes, bytes]:
    plan_path = Path(plan_dir)
    audit_path = Path(audit_dir)
    plan_manifest = _read_json(plan_path / MANIFEST_FILENAME, "enrichment plan manifest")
    audit_manifest = _read_json(audit_path / MANIFEST_FILENAME, "rubric audit manifest")
    if plan_manifest.get("schema_version") != LABELING_ENRICHMENT_PLAN_VERSION:
        raise ValueError("enrichment plan version does not match current version")
    if audit_manifest.get("schema_version") != LABELING_RUBRIC_AUDIT_VERSION:
        raise ValueError("rubric audit version does not match current version")
    if plan_manifest.get("bundle_id") != audit_manifest.get("bundle_id"):
        raise ValueError("maturity input sources have different bundle_id")

    plan_payload = _checked_payload(plan_path, "plan.json", plan_manifest)
    plan = json.loads(plan_payload)
    candidates = plan.get("candidates") if isinstance(plan, dict) else None
    if not isinstance(candidates, list):
        raise ValueError("enrichment plan candidates must be a list")
    candidate_ids: list[str] = []
    for candidate in candidates:
        candidate_id = candidate.get("candidate_id") if isinstance(candidate, dict) else None
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("enrichment candidate needs candidate_id")
        candidate_ids.append(candidate_id)
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("enrichment plan duplicates candidate_id")

    queue = _rows(
        _checked_payload(audit_path, MATURITY_QUEUE_FILENAME, audit_manifest),
        MATURITY_QUEUE_FILENAME,
    )
    by_candidate: dict[str, list[dict[str, Any]]] = {key: [] for key in candidate_ids}
    for row in queue:
        candidate_id = row.get("candidate_id")
        if candidate_id not in by_candidate:
            raise ValueError(f"maturity queue references unknown candidate {candidate_id!r}")
        rank = row.get("selection_rank")
        if not isinstance(rank, int) or isinstance(rank, bool) or not 1 <= rank <= 4:
            raise ValueError("maturity queue selection_rank must be 1..4")
        for field in ("document_id", "snapshot_id", "origin_id", "matched_term", "title"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"maturity queue row needs {field}")
        if not isinstance(row.get("excerpt"), str) or not row["excerpt"].strip():
            raise ValueError("maturity queue row needs excerpt")
        if not isinstance(row.get("url"), str) or not row["url"].startswith(("http://", "https://")):
            raise ValueError("maturity queue row needs an absolute URL")
        by_candidate[candidate_id].append(row)

    documents: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for candidate_id in sorted(candidate_ids):
        rows = sorted(by_candidate[candidate_id], key=lambda row: row["selection_rank"])
        if [row["selection_rank"] for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError(f"candidate {candidate_id} maturity ranks must form 1..N")
        for row in rows:
            documents.append({
                "candidate_id": candidate_id,
                "document_id": row["document_id"],
                "connector": "openalex",
                "source_class": "scientific",
                "final_rank": row["selection_rank"],
                "title": row["title"],
                "url": row["url"],
                "origin_id": row["origin_id"],
                "snapshot_id": row["snapshot_id"],
                "published_at": row.get("published_at"),
                "trust_tier": row["trust_tier"],
                "matched_term": row["matched_term"],
                "relevance_score": row["relevance_score"],
                "relevance_class": "strong",
                "evidence_text_available": True,
                "excerpt": row["excerpt"],
                "maturity_cues": row["maturity_cues"],
                "maturity_score": row["maturity_score"],
            })
        coverage.append({
            "candidate_id": candidate_id,
            "has_evidence_input": bool(rows),
            "final_evidence_input_count": len(rows),
            "scientific_selected": len(rows),
            "media_selected": 0,
            "source_classes_present": ["scientific"] if rows else [],
            "empty_reasons": [] if rows else ["no_maturity_context"],
        })

    documents_bytes = _jsonl_bytes(documents)
    coverage_bytes = _jsonl_bytes(coverage)
    manifest = {
        "schema_version": LABELING_EVIDENCE_INPUT_PLAN_VERSION,
        "bundle_id": plan_manifest["bundle_id"],
        "cutoff_date": LABELING_CUTOFF_DATE.isoformat(),
        "selection_policy": {
            "mode": MATURITY_INPUT_POLICY_VERSION,
            "max_documents_per_candidate": 4,
            "requires_maturity_cue": True,
            "labels_read": False,
        },
        "totals": {
            "candidates": len(coverage),
            "evidence_input_documents": len(documents),
            "scientific_selected": len(documents),
            "media_selected": 0,
            "candidates_without_final_input": sum(
                not row["has_evidence_input"] for row in coverage
            ),
        },
        "inputs": {
            "enrichment_plan_manifest": _digest((plan_path / MANIFEST_FILENAME).read_bytes()),
            "rubric_audit_manifest": _digest((audit_path / MANIFEST_FILENAME).read_bytes()),
        },
        "outputs": {
            DOCUMENTS_FILENAME: _digest(documents_bytes),
            COVERAGE_FILENAME: _digest(coverage_bytes),
        },
    }
    return documents_bytes, coverage_bytes, _json_bytes(manifest)


def export_maturity_evidence_input(
    *, plan_dir: str | Path, audit_dir: str | Path, output_dir: str | Path
) -> MaturityEvidenceInputPaths:
    documents, coverage, manifest = build_maturity_evidence_input(
        plan_dir=plan_dir, audit_dir=audit_dir
    )
    paths = publish_artifact_bundle(
        {
            DOCUMENTS_FILENAME: documents,
            COVERAGE_FILENAME: coverage,
            MANIFEST_FILENAME: manifest,
        },
        output_dir,
    )
    return MaturityEvidenceInputPaths(
        manifest=paths[MANIFEST_FILENAME],
        documents=paths[DOCUMENTS_FILENAME],
        coverage=paths[COVERAGE_FILENAME],
    )
