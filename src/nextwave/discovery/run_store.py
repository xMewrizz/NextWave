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

from .candidate_gate import CANDIDATE_GATE_VERSION
from .contracts import DiscoveryPlan
from .pipeline import DISCOVERY_PIPELINE_VERSION, DiscoveryPipelineResult

DISCOVERY_RUN_MANIFEST_SCHEMA_VERSION = "discovery-run-manifest-v2"

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
    input_proposal_ids = gate.get("input_proposal_ids")
    checked = (
        len(input_proposal_ids)
        if isinstance(input_proposal_ids, list)
        else accepted + rejected + review
    )
    coverage = result_dict.get("gate_coverage") or {}
    skipped = coverage.get("skipped_proposals")
    if not isinstance(skipped, int) or isinstance(skipped, bool):
        skipped = len(proposals.get("proposals") or []) - checked

    return {
        "documents": len(result_dict.get("documents") or []),
        "proposals": len(proposals.get("proposals") or []),
        "exclusions": len(proposals.get("exclusions") or []),
        "accepted": accepted,
        "rejected": rejected,
        "review": review,
        "gate_skipped": skipped,
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


def is_labeling_eligible(
    cutoff_iso: str, analysis_status: str = "complete"
) -> bool:
    """Require both the fixed cutoff and complete candidate coverage."""

    return cutoff_iso == LABELING_CUTOFF_DATE_ISO and analysis_status == "complete"


def assert_labeling_cutoff(cutoff_iso: str) -> None:
    if not is_labeling_eligible(cutoff_iso):
        raise ValueError(
            "run cutoff "
            f"{cutoff_iso!r} is not eligible for labeling; "
            f"expected {LABELING_CUTOFF_DATE_ISO!r}"
        )


def analysis_status_from_result(result_dict: dict[str, Any]) -> str:
    """Read new coverage metadata or infer legacy partial Gate runs."""

    coverage = result_dict.get("gate_coverage") or {}
    status = coverage.get("status")
    if status in {"complete", "partial"}:
        return str(status)
    if status is not None:
        raise ValueError(f"invalid candidate gate coverage status: {status!r}")
    proposals = (result_dict.get("candidate_proposals") or {}).get("proposals") or []
    input_ids = (result_dict.get("candidate_gate") or {}).get("input_proposal_ids")
    if isinstance(input_ids, list) and len(input_ids) < len(proposals):
        return "partial"
    return "complete"


def _require_strict_count(run_id: str, field: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(
            f"run {run_id!r} is not eligible for labeling; "
            f"gate_coverage.{field} must be a non-negative int, got {value!r}"
        )
    return value


def assert_run_labeling_eligible(run: DiscoveryRun) -> None:
    """Reject anything but a current complete run before queue construction.

    Every provenance field is cross-checked between ``plan.json``,
    ``pipeline_result.json`` and ``manifest.json``; the saved
    ``manifest.labeling_eligible`` flag is never trusted on its own, so old
    v6/v8 safes and Gate v1 results cannot slip into a new export.
    """

    manifest_cutoff = run.manifest.get("cutoff_date")
    plan_cutoff = (run.plan.get("query") or {}).get("cutoff_date")
    if manifest_cutoff != LABELING_CUTOFF_DATE_ISO:
        raise ValueError(
            f"run {run.run_id!r} is not eligible for labeling; "
            f"run cutoff {manifest_cutoff!r} is not eligible for labeling; "
            f"expected {LABELING_CUTOFF_DATE_ISO!r}"
        )
    if plan_cutoff != LABELING_CUTOFF_DATE_ISO:
        raise ValueError(
            f"run {run.run_id!r} is not eligible for labeling; "
            f"plan cutoff {plan_cutoff!r} does not match expected {LABELING_CUTOFF_DATE_ISO!r}"
        )
    if manifest_cutoff != plan_cutoff:
        raise ValueError(
            f"run {run.run_id!r} is not eligible for labeling; "
            f"cutoff mismatch manifest {manifest_cutoff!r} vs plan {plan_cutoff!r}"
        )
    manifest_status = run.manifest.get("analysis_status")
    if manifest_status is None:
        raise ValueError(
            f"run {run.run_id!r} is not eligible for labeling; "
            "manifest.analysis_status is missing"
        )
    derived_status = analysis_status_from_result(run.result)
    if manifest_status != derived_status:
        raise ValueError(
            f"run {run.run_id!r} is not eligible for labeling; "
            f"analysis_status mismatch manifest {manifest_status!r} "
            f"vs result {derived_status!r}"
        )
    if manifest_status != "complete":
        raise ValueError(
            f"run {run.run_id!r} is not eligible for labeling; "
            f"candidate gate coverage is {manifest_status!r}"
        )
    manifest_pipeline = run.manifest.get("pipeline_version")
    result_pipeline = run.result.get("pipeline_version")
    if manifest_pipeline is None or result_pipeline is None:
        raise ValueError(
            f"run {run.run_id!r} is not eligible for labeling; "
            f"pipeline version is missing (manifest {manifest_pipeline!r}, "
            f"result {result_pipeline!r})"
        )
    if manifest_pipeline != result_pipeline:
        raise ValueError(
            f"run {run.run_id!r} is not eligible for labeling; "
            f"pipeline version mismatch manifest {manifest_pipeline!r} "
            f"vs result {result_pipeline!r}"
        )
    if result_pipeline != DISCOVERY_PIPELINE_VERSION:
        raise ValueError(
            f"run {run.run_id!r} is not eligible for labeling; "
            f"pipeline version {result_pipeline!r} does not match "
            f"current {DISCOVERY_PIPELINE_VERSION!r}"
        )
    coverage = run.result.get("gate_coverage")
    if not isinstance(coverage, dict):
        raise ValueError(
            f"run {run.run_id!r} is not eligible for labeling; "
            "gate_coverage is missing or not an object"
        )
    coverage_status = coverage.get("status")
    if coverage_status != "complete":
        raise ValueError(
            f"run {run.run_id!r} is not eligible for labeling; "
            f"candidate gate coverage is {coverage_status!r}"
        )
    if "skipped_proposals" not in coverage:
        raise ValueError(
            f"run {run.run_id!r} is not eligible for labeling; "
            "gate_coverage.skipped_proposals is missing"
        )
    skipped = coverage.get("skipped_proposals")
    if not isinstance(skipped, int) or isinstance(skipped, bool):
        raise ValueError(
            f"run {run.run_id!r} is not eligible for labeling; "
            f"gate_coverage.skipped_proposals must be int, got {skipped!r}"
        )
    if skipped != 0:
        raise ValueError(
            f"run {run.run_id!r} is not eligible for labeling; "
            f"gate has {skipped!r} skipped proposals"
        )
    if "skipped_proposal_ids" not in coverage:
        raise ValueError(
            f"run {run.run_id!r} is not eligible for labeling; "
            "gate_coverage.skipped_proposal_ids is missing"
        )
    skipped_ids = coverage.get("skipped_proposal_ids")
    if not isinstance(skipped_ids, list):
        raise ValueError(
            f"run {run.run_id!r} is not eligible for labeling; "
            f"gate_coverage.skipped_proposal_ids must be list, got {skipped_ids!r}"
        )
    if len(skipped_ids) != 0:
        raise ValueError(
            f"run {run.run_id!r} is not eligible for labeling; "
            f"gate has {len(skipped_ids)} skipped proposal ids"
        )
    if "total_proposals" not in coverage or "checked_proposals" not in coverage:
        raise ValueError(
            f"run {run.run_id!r} is not eligible for labeling; "
            "gate_coverage.total_proposals/checked_proposals are missing"
        )
    total = _require_strict_count(
        run.run_id, "total_proposals", coverage.get("total_proposals")
    )
    checked = _require_strict_count(
        run.run_id, "checked_proposals", coverage.get("checked_proposals")
    )
    if total != checked + skipped:
        raise ValueError(
            f"run {run.run_id!r} is not eligible for labeling; "
            f"coverage mismatch total {total!r} != checked {checked!r} "
            f"+ skipped {skipped!r}"
        )
    proposals = (run.result.get("candidate_proposals") or {}).get("proposals")
    if not isinstance(proposals, list):
        raise ValueError(
            f"run {run.run_id!r} is not eligible for labeling; "
            "candidate_proposals.proposals is missing or not a list"
        )
    if total != len(proposals):
        raise ValueError(
            f"run {run.run_id!r} is not eligible for labeling; "
            f"coverage total {total!r} != proposals {len(proposals)!r}"
        )
    input_ids = (run.result.get("candidate_gate") or {}).get("input_proposal_ids")
    if not isinstance(input_ids, list):
        raise ValueError(
            f"run {run.run_id!r} is not eligible for labeling; "
            "candidate_gate.input_proposal_ids is missing or not a list"
        )
    if checked != len(input_ids):
        raise ValueError(
            f"run {run.run_id!r} is not eligible for labeling; "
            f"coverage checked {checked!r} != gate inputs {len(input_ids)!r}"
        )
    manifest_gate = run.manifest.get("gate_id")
    result_gate = (run.result.get("candidate_gate") or {}).get("gate_id")
    if not isinstance(manifest_gate, str) or not isinstance(result_gate, str):
        raise ValueError(
            f"run {run.run_id!r} is not eligible for labeling; "
            f"gate id is missing (manifest {manifest_gate!r}, result {result_gate!r})"
        )
    if manifest_gate != result_gate:
        raise ValueError(
            f"run {run.run_id!r} is not eligible for labeling; "
            f"gate id mismatch manifest {manifest_gate!r} vs result {result_gate!r}"
        )
    expected_suffix = "-" + CANDIDATE_GATE_VERSION
    if not result_gate.endswith(expected_suffix):
        raise ValueError(
            f"run {run.run_id!r} is not eligible for labeling; "
            f"candidate gate {result_gate!r} does not match "
            f"current {CANDIDATE_GATE_VERSION!r}"
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
    analysis_status = analysis_status_from_result(result_dict)

    manifest_dict: dict[str, Any] = {
        "schema_version": DISCOVERY_RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "analysis_id": plan.analysis_id,
        "analysis_scope_key": analysis_scope_key.strip(),
        "domain": domain.strip(),
        "raw_query": plan.scope.raw_query,
        "cutoff_date": cutoff_iso,
        "analysis_status": analysis_status,
        "labeling_eligible": is_labeling_eligible(cutoff_iso, analysis_status),
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
