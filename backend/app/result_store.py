"""Read and validate the immutable query result produced by the main CLI."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

RESULT_SCHEMA_VERSION = "analysis-result-v2"
RESPONSE_SCHEMA_VERSION = "analysis-response-v1"
DEFAULT_RESULT_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "development"
    / "analysis-result-aiinfra-004-unified-v2"
)


def configured_result_dir() -> Path:
    value = os.getenv("NEXTWAVE_RESULT_DIR")
    return Path(value) if value else DEFAULT_RESULT_DIR


def _read_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value, raw


def _read_jsonl(raw: bytes, label: str) -> list[dict[str, Any]]:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be UTF-8") from error
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label} line {line_number} is invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{label} line {line_number} must be an object")
        result.append(row)
    return result


def _checked(root: Path, manifest: dict[str, Any], filename: str) -> bytes:
    try:
        raw = (root / filename).read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {filename}") from error
    expected = (manifest.get("outputs") or {}).get(filename)
    actual = {"size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    if expected != actual:
        raise ValueError(f"{filename} differs from manifest")
    return raw


def load_result_bundle(root: str | Path | None = None) -> dict[str, Any]:
    result_root = Path(root) if root is not None else configured_result_dir()
    manifest, _ = _read_object(result_root / "manifest.json", "result manifest")
    if manifest.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("result schema version is not supported")

    candidates = _read_jsonl(
        _checked(result_root, manifest, "candidates.jsonl"), "candidates"
    )
    top15_raw = _checked(result_root, manifest, "top15.json")
    summary_raw = _checked(result_root, manifest, "summary.json")
    result_raw = _checked(result_root, manifest, "result.json")
    try:
        top15 = json.loads(top15_raw.decode("utf-8"))
        summary = json.loads(summary_raw.decode("utf-8"))
        result = json.loads(result_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("result JSON is invalid") from error
    if not isinstance(top15, list) or not all(isinstance(row, dict) for row in top15):
        raise ValueError("top15 must be a list of objects")
    if not isinstance(summary, dict):
        raise ValueError("summary must be an object")
    if not isinstance(result, dict) or result.get("schema_version") != RESPONSE_SCHEMA_VERSION:
        raise ValueError("result.json schema version is not supported")

    candidate_ids: set[str] = set()
    for line_number, candidate in enumerate(candidates, 1):
        candidate_id = candidate.get("candidate_id")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id in candidate_ids
        ):
            raise ValueError(f"candidates line {line_number} has invalid identity")
        if candidate.get("schema_version") != RESULT_SCHEMA_VERSION:
            raise ValueError(f"candidate {candidate_id!r} has unsupported schema")
        candidate_ids.add(candidate_id)

    top15_ids = [row.get("candidate_id") for row in top15]
    if (
        len(top15_ids) > 15
        or len(set(top15_ids)) != len(top15_ids)
        or any(candidate_id not in candidate_ids for candidate_id in top15_ids)
        or [row.get("top15_rank") for row in top15] != list(range(1, len(top15) + 1))
        or any(row.get("status") != "main" for row in top15)
    ):
        raise ValueError("top15 is inconsistent with candidates")
    if (
        manifest.get("candidate_count") != len(candidates)
        or manifest.get("top15_count") != len(top15)
        or summary.get("candidate_count") != len(candidates)
        or summary.get("top15_count") != len(top15)
    ):
        raise ValueError("result totals are inconsistent")
    if (
        result.get("summary") != summary
        or result.get("top15") != top15
        or result.get("candidates") != candidates
    ):
        raise ValueError("result.json differs from checked component files")
    return result
