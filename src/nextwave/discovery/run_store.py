"""Persist one discovery run as plan + result + manifest.

One run directory under ``data/development/discovery/<run_id>/`` contains:
- ``plan.json`` — what we intended to search (DiscoveryPlan.to_dict());
- ``pipeline_result.json`` — what the pipeline found (DiscoveryPipelineResult.to_dict());
- ``manifest.json`` — inventory with SHA-256 digests and counts.

Re-running into an existing directory is an error, never a silent overwrite.
Files render deterministically (indent=2, sort_keys, trailing LF) so the same
inputs always produce identical bytes. No wall-clock goes inside the files.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nextwave.sources import publish_staging

from .contracts import DiscoveryPlan
from .pipeline import DISCOVERY_PIPELINE_VERSION, DiscoveryPipelineResult

DISCOVERY_RUN_MANIFEST_SCHEMA_VERSION = "discovery-run-manifest-v1"

# Must stay in sync with LABELING.md and labeling/contracts.py.
# Runs with any other cutoff are fine for demo, but only this date is
# eligible for the labeling (training) export.
LABELING_CUTOFF_DATE_ISO = "2026-09-15"

PLAN_FILENAME = "plan.json"
RESULT_FILENAME = "pipeline_result.json"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True, slots=True)
class DiscoveryRun:
    """A loaded run: raw dicts plus the verified manifest."""

    run_id: str
    run_dir: Path
    plan: dict[str, Any]
    result: dict[str, Any]
    manifest: dict[str, Any]


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _pretty_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_run_id(analysis_id: str, plan_dict: dict[str, Any]) -> str:
    """Build ``{analysis_id}-{12 hex chars of plan digest}``.

    Twelve hex chars (not eight) keep collisions away once we run
    dozens of searches across the six organizer domains.
    """

    if not analysis_id or not analysis_id.strip():
        raise ValueError("analysis_id must not be blank")
    digest = _sha256_hex(_canonical_bytes(plan_dict))[:12]
    return f"{analysis_id.strip()}-{digest}"


def _counts_from_result(result_dict: dict[str, Any]) -> dict[str, int]:
    proposals = result_dict.get("candidate_proposals") or {}
    gate = result_dict.get("candidate_gate") or {}
    aliases = result_dict.get("alias_resolution") or {}
    evidence = result_dict.get("evidence_extraction") or {}
    text_extraction = result_dict.get("text_extraction") or {}
    decisions = gate.get("decisions") or []

    accepted = sum(1 for item in decisions if item.get("decision") == "accept")
    rejected = sum(1 for item in decisions if item.get("decision") == "reject")
    review = sum(1 for item in decisions if item.get("decision") == "review")
    text_issues = text_extraction.get("issues") or []
    evidence_issues = evidence.get("issues") or []
    judged = accepted + rejected + review

    return {
        "documents": len(result_dict.get("documents") or []),
        "proposals": len(proposals.get("proposals") or []),
        "exclusions": len(proposals.get("exclusions") or []),
        "accepted": accepted,
        "rejected": rejected,
        "review": review,
        "gate_skipped": len(proposals.get("proposals") or []) - judged,
        "alias_suggestions": len(aliases.get("review_suggestions") or []),
        "evidence_proposals": len(evidence.get("proposals") or []),
        "issues": len(text_issues) + len(evidence_issues),
    }


def _snapshot_ids_from_result(result_dict: dict[str, Any]) -> list[str]:
    found: list[str] = []
    scientific = result_dict.get("scientific") or {}
    media = result_dict.get("media") or {}
    for section in (scientific, media):
        snapshot_path = section.get("snapshot_path")
        if snapshot_path:
            found.append(str(snapshot_path))
    verification = result_dict.get("verification") or {}
    for item in verification.get("results") or []:
        snapshot_path = item.get("snapshot_path")
        if snapshot_path:
            found.append(str(snapshot_path))
    return sorted(set(found))


def is_labeling_eligible(cutoff_iso: str) -> bool:
    """Only the fixed labeling cutoff may train the model; anything else is demo-only."""

    return cutoff_iso == LABELING_CUTOFF_DATE_ISO


def assert_labeling_cutoff(cutoff_iso: str) -> None:
    if not is_labeling_eligible(cutoff_iso):
        raise ValueError(
            "run cutoff "
            f"{cutoff_iso!r} is not eligible for labeling; "
            f"expected {LABELING_CUTOFF_DATE_ISO!r}"
        )


def iter_nested_documents(result_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect full document payloads nested inside the result dict.

    The top-level ``documents`` entry holds IDs only, so the queue builder
    must read payloads from here to resolve metadata (title, url, dates,
    source_type, trust_tier, origin_id).
    """

    collected: list[dict[str, Any]] = []
    scientific = result_dict.get("scientific") or {}
    media = result_dict.get("media") or {}
    collected.extend(scientific.get("documents") or [])
    collected.extend(media.get("documents") or [])
    verification = result_dict.get("verification") or {}
    for item in verification.get("results") or []:
        collected.extend(item.get("matching_documents") or [])
    return collected


def save_discovery_run(
    plan: DiscoveryPlan,
    result: DiscoveryPipelineResult,
    *,
    analysis_scope_key: str,
    domain: str,
    output_root: str | Path = Path("data") / "development" / "discovery",
) -> Path:
    """Save one pipeline run into a new run directory and return its path."""

    if plan.plan_id != result.plan_id:
        raise ValueError("plan and result plan_id must match")
    if not analysis_scope_key or not analysis_scope_key.strip():
        raise ValueError("analysis_scope_key must not be blank")
    if not domain or not domain.strip():
        raise ValueError("domain must not be blank")

    plan_dict = plan.to_dict()
    result_dict = result.to_dict()
    run_id = build_run_id(plan.analysis_id, plan_dict)

    plan_bytes = _pretty_bytes(plan_dict)
    result_bytes = _pretty_bytes(result_dict)
    counts = _counts_from_result(result_dict)
    cutoff_iso = str(plan.query.cutoff_date)

    manifest_dict: dict[str, Any] = {
        "schema_version": DISCOVERY_RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "analysis_id": plan.analysis_id,
        "analysis_scope_key": analysis_scope_key.strip(),
        "domain": domain.strip(),
        "raw_query": plan.scope.raw_query,
        "cutoff_date": cutoff_iso,
        "labeling_eligible": is_labeling_eligible(cutoff_iso),
        "pipeline_version": result.pipeline_version or DISCOVERY_PIPELINE_VERSION,
        "gate_id": (result_dict.get("candidate_gate") or {}).get("gate_id"),
        "snapshot_ids": _snapshot_ids_from_result(result_dict),
        "counts": counts,
        "outputs": [
            {
                "filename": PLAN_FILENAME,
                "size_bytes": len(plan_bytes),
                "sha256": _sha256_hex(plan_bytes),
            },
            {
                "filename": RESULT_FILENAME,
                "size_bytes": len(result_bytes),
                "sha256": _sha256_hex(result_bytes),
            },
        ],
    }
    manifest_bytes = _pretty_bytes(manifest_dict)

    target = Path(output_root) / run_id
    if target.exists():
        raise FileExistsError(f"discovery run already exists: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        (staging / PLAN_FILENAME).write_bytes(plan_bytes)
        (staging / RESULT_FILENAME).write_bytes(result_bytes)
        (staging / MANIFEST_FILENAME).write_bytes(manifest_bytes)
        publish_staging(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def load_discovery_run(run_dir: str | Path) -> DiscoveryRun:
    """Load a run directory and verify both payloads against the manifest."""

    target = Path(run_dir)
    manifest_path = target / MANIFEST_FILENAME
    plan_path = target / PLAN_FILENAME
    result_path = target / RESULT_FILENAME
    if not manifest_path.is_file() or not plan_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(f"discovery run is incomplete: {target}")

    manifest_dict = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan_bytes = plan_path.read_bytes()
    result_bytes = result_path.read_bytes()

    expected = {item["filename"]: item for item in manifest_dict.get("outputs", [])}
    for filename, payload in ((PLAN_FILENAME, plan_bytes), (RESULT_FILENAME, result_bytes)):
        item = expected.get(filename)
        if item is None:
            raise ValueError(f"manifest is missing output entry: {filename}")
        if len(payload) != item["size_bytes"]:
            raise ValueError(f"artifact size mismatch: {filename}")
        if _sha256_hex(payload) != item["sha256"]:
            raise ValueError(f"artifact checksum mismatch: {filename}")

    return DiscoveryRun(
        run_id=str(manifest_dict["run_id"]),
        run_dir=target,
        plan=json.loads(plan_bytes.decode("utf-8")),
        result=json.loads(result_bytes.decode("utf-8")),
        manifest=manifest_dict,
    )
