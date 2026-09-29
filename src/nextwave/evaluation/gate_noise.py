"""Audit how the discovery pipeline contains reviewed noise controls."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nextwave.datasets.artifacts import publish_artifact_bundle
from nextwave.discovery import QUALIFICATION_GATE_ID

GATE_NOISE_EVALUATION_VERSION = "gate-noise-evaluation-v1"
_CUTOFF = "2026-09-15"
_PIPELINE_VERSION = "discovery-pipeline-v10"
_GATE_TYPES = {"broad_concept", "irrelevant", "not_technology"}
_NOISE_QUOTAS = {
    "broad_concept": 10,
    "duplicate": 10,
    "extraction_error": 10,
    "irrelevant": 10,
    "not_technology": 10,
}
_ORIGIN_STAGE = {
    "gate_reject": "candidate_gate",
    "exclusion": "pre_gate_exclusion",
    "extraction_issue": "extraction_validation",
    "alias_suggestion": "alias_resolution",
    "candidate_alias": "alias_resolution",
    "candidate_duplicate": "alias_resolution",
}


@dataclass(frozen=True, slots=True)
class GateNoiseEvaluationPaths:
    rows: Path
    report: Path
    manifest: Path


def _digest(payload: bytes) -> dict[str, Any]:
    return {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


def _verify_run(run_dir: Path, expected_run_id: str) -> tuple[dict[str, Any], bytes]:
    manifest_path = run_dir / "manifest.json"
    result_path = run_dir / "pipeline_result.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = _read_object(manifest_path, f"run {expected_run_id} manifest")
    if manifest.get("run_id") != expected_run_id:
        raise ValueError(f"run directory {expected_run_id} has a mismatched run_id")
    if manifest.get("pipeline_version") != _PIPELINE_VERSION:
        raise ValueError(f"run {expected_run_id} must use {_PIPELINE_VERSION}")
    if manifest.get("gate_id") != QUALIFICATION_GATE_ID:
        raise ValueError(f"run {expected_run_id} must use the qualification gate")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError(f"run {expected_run_id} manifest outputs must be a list")
    expected = next(
        (
            item
            for item in outputs
            if isinstance(item, dict) and item.get("filename") == "pipeline_result.json"
        ),
        None,
    )
    result_bytes = result_path.read_bytes()
    if expected != {"filename": "pipeline_result.json", **_digest(result_bytes)}:
        raise ValueError(f"run {expected_run_id} pipeline_result.json diverges from manifest")
    result = json.loads(result_bytes)
    if not isinstance(result, dict):
        raise ValueError(f"run {expected_run_id} result must be an object")
    return result, manifest_bytes


def _gate_observation(result: dict[str, Any], entry: dict[str, Any]) -> dict[str, str]:
    proposals_value = result.get("candidate_proposals")
    gate_value = result.get("candidate_gate")
    if not isinstance(proposals_value, dict) or not isinstance(gate_value, dict):
        raise ValueError(f"run {entry['run_id']} misses candidate proposals or gate")
    proposals = proposals_value.get("proposals")
    decisions = gate_value.get("decisions")
    if not isinstance(proposals, list) or not isinstance(decisions, list):
        raise ValueError(f"run {entry['run_id']} has invalid proposal or gate lists")
    needle = _normalize(entry["extracted_text"])
    matches: list[dict[str, Any]] = []
    for proposal in proposals:
        if not isinstance(proposal, dict):
            continue
        names = [proposal.get("canonical_name"), *(proposal.get("aliases") or [])]
        if needle in {_normalize(value) for value in names if isinstance(value, str)}:
            matches.append(proposal)
    if len(matches) != 1:
        raise ValueError(
            f"noise {entry['noise_id']} must match exactly one proposal, found {len(matches)}"
        )
    proposal_id = matches[0].get("proposal_id")
    matched = [
        item
        for item in decisions
        if isinstance(item, dict) and item.get("proposal_id") == proposal_id
    ]
    if len(matched) != 1:
        raise ValueError(f"noise {entry['noise_id']} must have exactly one gate decision")
    decision = matched[0]
    value, reason = decision.get("decision"), decision.get("reason")
    if value not in {"accept", "reject", "review"} or not isinstance(reason, str):
        raise ValueError(f"noise {entry['noise_id']} has an invalid gate decision")
    return {"proposal_id": str(proposal_id), "decision": value, "reason": reason}


def build_gate_noise_evaluation(
    *, selection_path: str | Path, discovery_root: str | Path
) -> tuple[bytes, bytes, bytes]:
    selection_file = Path(selection_path)
    selection_bytes = selection_file.read_bytes()
    selection = _read_object(selection_file, "noise selection")
    if (
        selection.get("schema_version") != "labeling-noise-selection-v1"
        or selection.get("cutoff_date") != _CUTOFF
    ):
        raise ValueError(
            "noise selection must use labeling-noise-selection-v1 and the frozen cutoff"
        )
    entries = selection.get("entries")
    if not isinstance(entries, list) or len(entries) != 50:
        raise ValueError("noise selection must contain exactly 50 entries")
    ids: set[str] = set()
    counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    run_cache: dict[str, dict[str, Any]] = {}
    run_inputs: dict[str, dict[str, Any]] = {}
    root = Path(discovery_root)
    for index, raw in enumerate(entries, 1):
        if not isinstance(raw, dict):
            raise ValueError(f"noise entry {index} must be an object")
        noise_id = raw.get("noise_id")
        noise_type = raw.get("planned_noise_type")
        origin = raw.get("origin_kind")
        run_id = raw.get("run_id")
        text = raw.get("extracted_text")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (noise_id, noise_type, origin, run_id, text)
        ):
            raise ValueError(f"noise entry {index} has blank required fields")
        if noise_id in ids:
            raise ValueError(f"duplicate noise_id {noise_id}")
        if noise_type not in _NOISE_QUOTAS or origin not in _ORIGIN_STAGE:
            raise ValueError(f"noise {noise_id} has an unsupported type or origin")
        ids.add(noise_id)
        counts[noise_type] += 1
        observation: dict[str, str] | None = None
        if origin == "gate_reject":
            if run_id not in run_cache:
                result, manifest_bytes = _verify_run(root / run_id, run_id)
                run_cache[run_id] = result
                run_inputs[run_id] = _digest(manifest_bytes)
            observation = _gate_observation(run_cache[run_id], raw)
        gate_expected = noise_type in _GATE_TYPES
        rows.append(
            {
                "noise_id": noise_id,
                "noise_type": noise_type,
                "extracted_text": text,
                "run_id": run_id,
                "origin_kind": origin,
                "containment_stage": _ORIGIN_STAGE[origin],
                "gate_expected": gate_expected,
                "gate_observed": observation is not None,
                "gate_decision": observation["decision"] if observation else None,
                "gate_reason": observation["reason"] if observation else None,
                "proposal_id": observation["proposal_id"] if observation else None,
                "contained_before_ranking": True,
                "schema_version": GATE_NOISE_EVALUATION_VERSION,
            }
        )
    if dict(counts) != _NOISE_QUOTAS:
        raise ValueError("noise selection must contain five exact 10-row quotas")
    rows.sort(key=lambda row: row["noise_id"])
    gate_rows = [row for row in rows if row["gate_expected"]]
    observed = [row for row in gate_rows if row["gate_observed"]]
    decisions = Counter(row["gate_decision"] for row in observed)
    stage_counts = Counter(row["containment_stage"] for row in rows)
    report_value = {
        "schema_version": GATE_NOISE_EVALUATION_VERSION,
        "cutoff_date": _CUTOFF,
        "status": "complete" if len(observed) == len(gate_rows) else "partial",
        "interpretation": (
            "Gate metrics use only controls with a preserved Gate decision; "
            "end-to-end containment covers all reviewed controls."
        ),
        "totals": {
            "noise_controls": len(rows),
            "gate_expected": len(gate_rows),
            "gate_observed": len(observed),
            "gate_unobserved": len(gate_rows) - len(observed),
            "gate_reject": decisions["reject"],
            "gate_review": decisions["review"],
            "gate_accept": decisions["accept"],
            "contained_before_ranking": sum(row["contained_before_ranking"] for row in rows),
        },
        "metrics": {
            "gate_coverage": round(len(observed) / len(gate_rows), 6),
            "observed_rejection_rate": round(decisions["reject"] / len(observed), 6),
            "observed_false_accept_rate": round(decisions["accept"] / len(observed), 6),
            "end_to_end_containment_rate": round(
                sum(row["contained_before_ranking"] for row in rows) / len(rows), 6
            ),
        },
        "by_noise_type": dict(sorted(counts.items())),
        "by_containment_stage": dict(sorted(stage_counts.items())),
    }
    rows_bytes = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode()
    report_bytes = (
        json.dumps(report_value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode()
    manifest_value = {
        "schema_version": GATE_NOISE_EVALUATION_VERSION,
        "cutoff_date": _CUTOFF,
        "inputs": {
            "noise_selection": _digest(selection_bytes),
            "run_manifests": dict(sorted(run_inputs.items())),
        },
        "outputs": {
            "noise_results.jsonl": _digest(rows_bytes),
            "report.json": _digest(report_bytes),
        },
    }
    manifest_bytes = (
        json.dumps(manifest_value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode()
    return rows_bytes, report_bytes, manifest_bytes


def export_gate_noise_evaluation(
    *, selection_path: str | Path, discovery_root: str | Path, output_dir: str | Path
) -> GateNoiseEvaluationPaths:
    rows, report, manifest = build_gate_noise_evaluation(
        selection_path=selection_path, discovery_root=discovery_root
    )
    output = Path(output_dir)
    publish_artifact_bundle(
        {"noise_results.jsonl": rows, "report.json": report, "manifest.json": manifest},
        output,
    )
    return GateNoiseEvaluationPaths(
        rows=output / "noise_results.jsonl",
        report=output / "report.json",
        manifest=output / "manifest.json",
    )
