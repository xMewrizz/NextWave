"""Resumable Evidence LLM executor (offline-first design, no model calls here).

Reads deterministic tasks from an evidence LLM plan, calls the configured
Yandex model once per candidate task, validates every claim against the
verbatim passage, and publishes claims with resumable work storage. Secrets,
prompts duplication and silent zeros are all rejected loudly.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from nextwave.contracts import ClaimType, EvidenceDirection
from nextwave.datasets.artifacts import publish_artifact_bundle
from nextwave.discovery.llm import (
    LlmProvider,
    YandexContentFilterError,
    YandexTruncationError,
    build_json_generator,
    load_evidence_llm_settings,
)
from nextwave.sources import publish_staging

from . import evidence_llm_plan as plan_module
from .contracts import LABELING_CUTOFF_DATE
from .evidence_llm_plan import (
    EVIDENCE_CLAIM_SCOPES,
    EVIDENCE_PURPOSE_GENERAL,
    EVIDENCE_PURPOSE_MATURITY,
    LABELING_EVIDENCE_LLM_PLAN_VERSION,
    MAX_CLAIMS_PER_DOCUMENT,
    MAX_DOCUMENTS_PER_TASK,
    build_evidence_prompt,
)
from .evidence_term_policy import text_supports_matched_term

LABELING_EVIDENCE_LLM_EXECUTOR_VERSION = "labeling-evidence-llm-executor-v12"
LABELING_EVIDENCE_LLM_WORK_VERSION = "labeling-evidence-llm-work-v5"
LABELING_EVIDENCE_LLM_CACHE_VERSION = "labeling-evidence-llm-cache-v5"
LABELING_EVIDENCE_LLM_RESULT_VERSION = "labeling-evidence-llm-result-v5"

EVIDENCE_MAX_OUTPUT_TOKENS = 2000
EVIDENCE_CONCURRENCY = 3
EVIDENCE_EXTRACTOR_VERSION = "evidence-llm-v11-general-maturity-relation"
QUALIFICATION_EVIDENCE_PROVIDER = "yandex"
QUALIFICATION_EVIDENCE_MODEL = "YandexGPT Pro 5.1"
EVIDENCE_RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["candidate_id", "documents"],
    "properties": {
        "candidate_id": {"type": "string"},
        "documents": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_DOCUMENTS_PER_TASK,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["document_id", "claims"],
                "properties": {
                    "document_id": {"type": "string"},
                    "claims": {
                        "type": "array",
                        "minItems": 0,
                        "maxItems": MAX_CLAIMS_PER_DOCUMENT,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "quote",
                                "kind",
                                "direction",
                                "scope",
                                "explanation_ru",
                                "missing_components",
                            ],
                            "properties": {
                                "quote": {
                                    "type": "string",
                                    "minLength": 20,
                                    "maxLength": 500,
                                },
                                "kind": {
                                    "type": "string",
                                    "enum": [item.value for item in ClaimType],
                                },
                                "direction": {
                                    "type": "string",
                                    "enum": [item.value for item in EvidenceDirection],
                                },
                                "scope": {
                                    "type": "string",
                                    "enum": list(EVIDENCE_CLAIM_SCOPES),
                                },
                                "explanation_ru": {
                                    "type": "string",
                                    "minLength": 10,
                                    "maxLength": 500,
                                },
                                "missing_components": {
                                    "type": "string",
                                    "maxLength": 300,
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}

WORK_MANIFEST_FILENAME = "work_manifest.json"
CACHE_MANIFEST_FILENAME = "cache_manifest.json"
REQUEST_RESULTS_FILENAME = "request_results.jsonl"
CLAIMS_FILENAME = "claims.jsonl"
DOCUMENT_RESULTS_FILENAME = "document_results.jsonl"
ISSUES_FILENAME = "issues.jsonl"
COVERAGE_FILENAME = "coverage.jsonl"
RESULT_MANIFEST_FILENAME = "manifest.json"

_CONNECTORS_BY_CLASS = {
    "scientific": {"openalex"},
    "industry": {"mediacloud", "exa"},
}
_VALID_CLASSES = ("scientific", "industry")


@dataclass(frozen=True, slots=True)
class LabelingEvidenceLlmRunPaths:
    """Paths of one published Evidence LLM result."""

    manifest: Path
    request_results: Path
    claims: Path
    document_results: Path
    issues: Path
    coverage: Path


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {error}") from error


def _check_digest(path: Path, entry: Any, label: str) -> bytes:
    if not isinstance(entry, dict):
        raise ValueError(f"manifest entry for {label} must be an object")
    size = entry.get("size_bytes")
    sha = entry.get("sha256")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError(f"manifest size_bytes for {label} must be a non-negative int")
    if not isinstance(sha, str) or len(sha) != 64:
        raise ValueError(f"manifest sha256 for {label} must be 64 hex characters")
    try:
        int(sha, 16)
    except ValueError as error:
        raise ValueError(f"manifest sha256 for {label} must be hex") from error
    if sha != sha.lower():
        raise ValueError(f"manifest sha256 for {label} must be lowercase hex")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if len(payload) != size:
        raise ValueError(f"{label} size mismatch with manifest")
    if hashlib.sha256(payload).hexdigest() != sha:
        raise ValueError(f"{label} checksum mismatch with manifest")
    return payload


def _read_rows(path: Path, payload: bytes, label: str) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not valid UTF-8: {error}") from error
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label} line {lineno} is not valid JSON") from error
        if not isinstance(item, dict):
            raise ValueError(f"{label} line {lineno} must be an object")
        rows.append(item)
    return rows


def _digest(payload: bytes) -> dict[str, Any]:
    return {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def _render_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return (
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ).encode("utf-8")
    )


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_plain_count(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative plain int")
    return value


def _require_url(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string URL")
    parsed = value.strip()
    if not parsed.startswith(("http://", "https://")) or " " in parsed:
        raise ValueError(f"{label} must be an absolute HTTP(S) URL")
    return value


def _check_date(value: Any, label: str) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise ValueError(f"{label} published_at must be an ISO date string or null")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{label} published_at is not an ISO date") from error
    if parsed > LABELING_CUTOFF_DATE:
        raise ValueError(f"{label} published_at {value} is past cutoff")


def _extractor_id(provider: str, model: str) -> str:
    slug = "-".join(model.lower().split())
    return f"{provider}-{slug}-{EVIDENCE_EXTRACTOR_VERSION}"


def _load_plan(
    plan_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest_path = plan_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"evidence LLM plan is incomplete: {plan_dir}")
    manifest = _read_json(manifest_path, "evidence LLM plan manifest")
    if not isinstance(manifest, dict):
        raise ValueError("evidence LLM plan manifest must be a JSON object")
    if manifest.get("schema_version") != LABELING_EVIDENCE_LLM_PLAN_VERSION:
        raise ValueError(
            f"evidence LLM plan manifest {manifest.get('schema_version')!r} "
            f"does not match {LABELING_EVIDENCE_LLM_PLAN_VERSION!r}"
        )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("evidence LLM plan manifest outputs must be an object")
    tasks_payload = _check_digest(
        plan_dir / "tasks.jsonl", outputs.get("tasks.jsonl"), "tasks.jsonl"
    )
    coverage_payload = _check_digest(
        plan_dir / "coverage.jsonl", outputs.get("coverage.jsonl"), "coverage.jsonl"
    )
    tasks = _read_rows(plan_dir / "tasks.jsonl", tasks_payload, "tasks.jsonl")
    coverage = _read_rows(plan_dir / "coverage.jsonl", coverage_payload, "coverage.jsonl")
    totals = manifest.get("totals")
    if not isinstance(totals, dict):
        raise ValueError("evidence LLM plan manifest totals must be an object")
    if totals.get("candidates") != len(coverage):
        raise ValueError("plan totals.candidates does not match coverage rows")
    if totals.get("planned_tasks") != len(tasks):
        raise ValueError("plan totals.planned_tasks does not match task rows")
    input_manifest = manifest.get("input_manifest")
    if not isinstance(input_manifest, dict):
        raise ValueError("evidence LLM plan manifest input_manifest must be an object")
    input_sha = input_manifest.get("sha256")
    if (
        not isinstance(input_sha, str)
        or len(input_sha) != 64
        or input_sha != input_sha.lower()
    ):
        raise ValueError("evidence LLM plan input manifest SHA is malformed")
    try:
        int(input_sha, 16)
    except ValueError as error:
        raise ValueError("evidence LLM plan input manifest SHA must be hex") from error
    input_size = input_manifest.get("size_bytes")
    if not isinstance(input_size, int) or isinstance(input_size, bool) or input_size < 0:
        raise ValueError("evidence LLM plan input manifest size is malformed")
    manifest_bytes = manifest_path.read_bytes()
    digests = {
        "manifest.json": _digest(manifest_bytes),
        "tasks.jsonl": _digest(tasks_payload),
        "coverage.jsonl": _digest(coverage_payload),
        "input_manifest": {"sha256": input_sha, "size_bytes": input_size},
    }
    return manifest, tasks, coverage, digests


_TASK_KEYS = frozenset({
    "task_id",
    "candidate_id",
    "documents",
    "document_count",
    "prompt_chars",
    "estimated_input_tokens",
    "input_manifest_sha256",
    "prompt_sha256",
})
_TASK_KEYS_WITH_PURPOSE = _TASK_KEYS | {"purpose"}

_TASK_DOCUMENT_KEYS = frozenset({
    "document_id",
    "final_rank",
    "source_class",
    "connector",
    "title",
    "url",
    "origin_id",
    "published_at",
    "trust_tier",
    "matched_term",
    "relevance_score",
    "relevance_class",
    "passage",
    "passage_start",
    "passage_end",
    "passage_truncated",
    "source_excerpt_sha256",
    "passage_sha256",
})


def _validate_task(
    task: Mapping[str, Any], input_manifest_sha: str, lineno: int
) -> dict[str, Any]:
    """Validate one planned task, including a rebuilt prompt and task_id."""

    label = f"tasks.jsonl line {lineno}"
    if not isinstance(task, dict):
        raise ValueError(f"{label} must be an object")
    if set(task) not in (set(_TASK_KEYS), set(_TASK_KEYS_WITH_PURPOSE)):
        raise ValueError(f"{label} holds unexpected task keys")
    purpose = task.get("purpose", EVIDENCE_PURPOSE_GENERAL)
    if purpose not in (EVIDENCE_PURPOSE_GENERAL, EVIDENCE_PURPOSE_MATURITY):
        raise ValueError(f"{label} has an unsupported evidence purpose")
    task_id = _require_text(task.get("task_id"), f"{label} task_id")
    candidate_id = _require_text(task.get("candidate_id"), f"{label} candidate_id")
    documents = task.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError(f"{label} documents must be a non-empty list")
    if len(documents) > MAX_DOCUMENTS_PER_TASK:
        raise ValueError(f"{label} has more than {MAX_DOCUMENTS_PER_TASK} documents")
    if task.get("document_count") != len(documents):
        raise ValueError(f"{label} document_count diverges from documents")
    ordered = []
    for index, doc in enumerate(documents):
        if not isinstance(doc, dict):
            raise ValueError(f"{label} document {index} must be an object")
        if set(doc) != set(_TASK_DOCUMENT_KEYS):
            raise ValueError(f"{label} document {index} holds unexpected keys")
        rank = doc.get("final_rank")
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise ValueError(f"{label} final_rank must be plain ints")
        ordered.append(rank)
        document_id = _require_text(doc.get("document_id"), f"{label} document_id")
        source_class = doc.get("source_class")
        if source_class not in _VALID_CLASSES:
            raise ValueError(f"{label} has an unknown source_class")
        if doc.get("connector") not in _CONNECTORS_BY_CLASS[source_class]:
            raise ValueError(f"{label} connector does not match source_class")
        for name in ("title", "origin_id", "matched_term"):
            _require_text(doc.get(name), f"{label} {name}")
        _require_url(doc.get("url"), f"{label} url")
        score = doc.get("relevance_score")
        if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 100:
            raise ValueError(f"{label} relevance_score must be a plain int 0..100")
        if doc.get("relevance_class") not in ("strong", "weak"):
            raise ValueError(f"{label} relevance_class must be strong or weak")
        if doc.get("trust_tier") not in ("A", "B", "C", "D", "unknown"):
            raise ValueError(f"{label} trust_tier must be A/B/C/D/unknown")
        passage = doc.get("passage")
        if not isinstance(passage, str) or not passage:
            raise ValueError(f"{label} passage must be a non-empty string")
        if len(passage) > 3000:
            raise ValueError(f"{label} passage exceeds 3000 characters")
        term_supported = (
            plan_module.maturity_text_supports_term(passage, doc["matched_term"])
            if purpose == EVIDENCE_PURPOSE_MATURITY
            else text_supports_matched_term(passage, doc["matched_term"])
        )
        if not term_supported:
            raise ValueError(
                f"{label} passage cannot satisfy the claim lexical gate "
                f"for {document_id!r}"
            )
        if hashlib.sha256(passage.encode("utf-8")).hexdigest() != doc.get("passage_sha256"):
            raise ValueError(f"{label} passage SHA mismatch for {document_id!r}")
        source_sha = doc.get("source_excerpt_sha256")
        if (
            not isinstance(source_sha, str)
            or len(source_sha) != 64
            or source_sha != source_sha.lower()
        ):
            raise ValueError(f"{label} source excerpt SHA has a bad format")
        try:
            int(source_sha, 16)
        except ValueError as error:
            raise ValueError(f"{label} source excerpt SHA must be hex") from error
        for name in ("passage_start", "passage_end"):
            value = doc.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{label} {name} must be a non-negative plain int")
        if doc["passage_end"] != doc["passage_start"] + len(passage):
            raise ValueError(f"{label} passage span diverges from passage length")
        _check_date(doc.get("published_at"), label)
    if ordered != list(range(1, len(documents) + 1)):
        raise ValueError(f"{label} final_rank must form 1..N without gaps")
    prompt = build_evidence_prompt({
        "candidate_id": candidate_id,
        "documents": documents,
        "purpose": purpose,
    })
    if len(prompt) != task.get("prompt_chars"):
        raise ValueError(f"{label} prompt_chars diverges from the rebuilt prompt")
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if prompt_sha != task.get("prompt_sha256"):
        raise ValueError(f"{label} prompt_sha256 diverges from the rebuilt prompt")
    if task.get("estimated_input_tokens") != math.ceil(len(prompt) / 3):
        raise ValueError(f"{label} token estimate diverges from ceil(chars/3)")
    if task.get("input_manifest_sha256") != input_manifest_sha:
        raise ValueError(f"{label} input manifest SHA diverges")
    expected_id = plan_module._task_id(
        input_manifest_sha,
        candidate_id,
        [doc["document_id"] for doc in documents],
        [doc["passage_sha256"] for doc in documents],
        [(doc["passage_start"], doc["passage_end"]) for doc in documents],
        prompt_sha,
    )
    if expected_id != task_id:
        raise ValueError(f"{label} task_id does not recompute")
    validated = {
        "task_id": task_id,
        "candidate_id": candidate_id,
        "documents": documents,
    }
    if purpose == EVIDENCE_PURPOSE_MATURITY:
        validated["purpose"] = purpose
    return validated


def _validate(
    tasks: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    input_manifest_sha: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Validate tasks against coverage before any work, output or model call."""

    seen_tasks: set[str] = set()
    seen_candidates: set[str] = set()
    validated: dict[str, dict[str, Any]] = {}
    for lineno, task in enumerate(tasks, start=1):
        record = _validate_task(task, input_manifest_sha, lineno)
        if record["task_id"] in seen_tasks:
            raise ValueError(f"duplicate task_id {record['task_id']!r}")
        seen_tasks.add(record["task_id"])
        if record["candidate_id"] in seen_candidates:
            raise ValueError(f"duplicate task candidate {record['candidate_id']!r}")
        seen_candidates.add(record["candidate_id"])
        if not isinstance(task, dict):
            raise ValueError(f"tasks.jsonl line {lineno} must be an object")
        validated[record["candidate_id"]] = task
    seen_coverage: set[str] = set()
    for row in coverage:
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("coverage rows need a candidate_id")
        if candidate_id in seen_coverage:
            raise ValueError(f"coverage duplicates candidate {candidate_id!r}")
        seen_coverage.add(candidate_id)
        status = row.get("status")
        if status == "planned":
            task = validated.get(candidate_id)
            if task is None:
                raise ValueError(f"planned coverage {candidate_id!r} has no task")
            if row.get("task_id") != task["task_id"]:
                raise ValueError(
                    f"planned coverage {candidate_id!r} references a wrong task_id"
                )
            for name in (
                "input_documents",
                "scientific_documents",
                "industry_documents",
                "prompt_chars",
                "estimated_input_tokens",
            ):
                value = row.get(name)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError(
                        f"planned coverage {candidate_id!r} {name} must be a plain int"
                    )
            if row["input_documents"] != len(task["documents"]):
                raise ValueError(
                    f"planned coverage {candidate_id!r} document count diverges"
                )
            if row["input_documents"] != (
                row["scientific_documents"] + row["industry_documents"]
            ):
                raise ValueError(
                    f"planned coverage {candidate_id!r} class counts do not add up"
                )
            if row.get("llm_call_planned") is not True:
                raise ValueError(
                    f"planned coverage {candidate_id!r} must set llm_call_planned"
                )
        elif status == "no_input":
            if candidate_id in validated:
                raise ValueError(f"no_input coverage {candidate_id!r} has a task")
            if row.get("task_id") is not None:
                raise ValueError(f"no_input coverage {candidate_id!r} carries a task_id")
            for name in (
                "input_documents",
                "scientific_documents",
                "industry_documents",
                "prompt_chars",
                "estimated_input_tokens",
            ):
                if row.get(name) != 0:
                    raise ValueError(f"no_input coverage {candidate_id!r} {name} must be 0")
            if row.get("llm_call_planned") is not False:
                raise ValueError(f"no_input coverage {candidate_id!r} must not plan a call")
            reasons = row.get("empty_reasons")
            if not isinstance(reasons, list) or any(
                not isinstance(item, str) for item in reasons
            ):
                raise ValueError(
                    f"no_input coverage {candidate_id!r} empty_reasons must be strings"
                )
        else:
            raise ValueError(f"coverage {candidate_id!r} has an unknown status")
    planned_ids = {
        row["candidate_id"] for row in coverage if row.get("status") == "planned"
    }
    unknown = sorted(set(validated) - planned_ids)
    if unknown:
        raise ValueError(
            f"unknown candidate(s) missing in planned coverage: {', '.join(unknown)}"
        )
    if set(validated) != planned_ids:
        raise ValueError("planned coverage rows do not match task rows")
    return validated, {row["candidate_id"]: row for row in coverage}


def _task_spec(
    task: Mapping[str, Any],
    prompt_sha: str,
    provider: str,
    model: str,
    extractor_id: str,
) -> dict[str, Any]:
    return {
        "executor_version": LABELING_EVIDENCE_LLM_EXECUTOR_VERSION,
        "task_id": task["task_id"],
        "candidate_id": task["candidate_id"],
        "document_ids": sorted(doc["document_id"] for doc in task["documents"]),
        "prompt_sha256": prompt_sha,
        "provider": provider,
        "model": model,
        "extractor_id": extractor_id,
        "max_output_tokens": EVIDENCE_MAX_OUTPUT_TOKENS,
    }


def _spec_digest(spec: Mapping[str, Any]) -> str:
    canonical = json.dumps(spec, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _work_fingerprint(
    plan_digests: Mapping[str, Any],
    bundle_id: str,
    cutoff: str,
    provider: str,
    model: str,
    extractor_id: str,
    planned: int,
) -> dict[str, Any]:
    return {
        "schema_version": LABELING_EVIDENCE_LLM_WORK_VERSION,
        "plan_manifest": dict(plan_digests["manifest.json"]),
        "plan_tasks": dict(plan_digests["tasks.jsonl"]),
        "plan_coverage": dict(plan_digests["coverage.jsonl"]),
        "bundle_id": bundle_id,
        "cutoff_date": cutoff,
        "provider": provider,
        "model": model,
        "extractor_id": extractor_id,
        "max_output_tokens": EVIDENCE_MAX_OUTPUT_TOKENS,
        "concurrency": EVIDENCE_CONCURRENCY,
        "planned_tasks": planned,
    }


def _init_or_check_work(work_dir: str | Path, fingerprint: Mapping[str, Any]) -> Path:
    """Create the work store atomically or verify it belongs to these inputs."""

    work = Path(work_dir)
    manifest_bytes = (
        json.dumps(dict(fingerprint), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    if not work.exists():
        work.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{work.name}-", dir=work.parent))
        try:
            (staging / WORK_MANIFEST_FILENAME).write_bytes(manifest_bytes)
            (staging / "completed").mkdir()
            (staging / "failures").mkdir()
            publish_staging(staging, work)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return work
    manifest_path = work / WORK_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ValueError(f"evidence LLM work store is corrupt, manifest missing: {work}")
    stored = _read_json(manifest_path, "evidence LLM work manifest")
    if not isinstance(stored, dict):
        raise ValueError(f"evidence LLM work manifest must be an object: {work}")
    if stored.get("schema_version") != LABELING_EVIDENCE_LLM_WORK_VERSION:
        raise ValueError(
            f"evidence LLM work schema {stored.get('schema_version')!r} "
            f"does not match {LABELING_EVIDENCE_LLM_WORK_VERSION!r}"
        )
    if stored != dict(fingerprint):
        raise ValueError(
            "evidence LLM work store belongs to different inputs; "
            "use a fresh work directory instead of reusing it silently"
        )
    for service in ("completed", "failures"):
        if not (work / service).is_dir():
            raise ValueError(f"evidence LLM work service directory is missing: {service}")
    return work


def _claim_id(
    candidate_id: str,
    document_id: str,
    kind: str,
    direction: str,
    scope: str,
    quote: str,
) -> str:
    identity = "|".join((candidate_id, document_id, kind, direction, scope, quote))
    return "claim-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def _valid_russian_explanation(value: Any) -> bool:
    return (
        isinstance(value, str)
        and 10 <= len(value) <= 500
        and any("а" <= char.casefold() <= "я" or char.casefold() == "ё" for char in value)
    )


def _component_tokens(value: str) -> list[str]:
    return "".join(
        char if char.isalnum() else " " for char in value.casefold()
    ).split()


def _contains_component(term: str, component: str) -> bool:
    term_tokens = _component_tokens(term)
    component_tokens = _component_tokens(component)
    if not component_tokens or len(component_tokens) > len(term_tokens):
        return False
    width = len(component_tokens)
    return any(
        term_tokens[index:index + width] == component_tokens
        for index in range(len(term_tokens) - width + 1)
    )


def _validated_missing_components(
    value: Any, *, scope: str, matched_term: str
) -> list[str] | None:
    if not isinstance(value, str) or len(value) > 300:
        return None
    if scope == "full_candidate":
        return [] if not value.strip() else None
    if scope != "core_only":
        return None
    if not value.strip():
        return []
    components = [component.strip() for component in value.split(";")]
    if not 1 <= len(components) <= 6 or any(not component for component in components):
        return None
    seen: set[str] = set()
    for component in components:
        if len(component) > 100:
            return None
        folded = component.casefold()
        if folded in seen or not _contains_component(matched_term, component):
            return None
        seen.add(folded)
    return components


_MATURITY_QUOTE_CUES = {
    "standard": (
        "adopted standard",
        "approved standard",
        "published standard",
        "ratified standard",
        "standardized",
        "standardised",
        "standardization",
        "standardisation",
        "iso standard",
        "iec standard",
        "ieee standard",
        "3gpp standard",
        "rfc standard",
    ),
    "adoption": (
        "adopted",
        "adoption",
        "deployed",
        "deployment",
        "in production",
        "production deployment",
        "industrial deployment",
        "commercial deployment",
        "commercially available",
        "widely used",
        "used in practice",
        "currently used",
        "operational use",
        "implemented across",
        "used across",
        "employed by",
        "successful implementation",
        "successful implementations",
        "primary driver of uptake",
        "go to accelerator",
        "integral component",
        "integral components",
        "relies heavily on",
        "rely heavily on",
        "penetration",
        "uptake",
        "critical role",
        "essential role",
    ),
    "market": (
        "established market",
        "commercial market",
        "market share",
        "multiple vendors",
        "multiple suppliers",
        "vendors offer",
        "commercial products",
        "commercially available",
    ),
}

_ADOPTION_RELATION_PREFIXES = (
    "adopt",
    "commercial",
    "deploy",
    "employ",
    "implement",
    "integrat",
    "operat",
    "standardiz",
    "standardis",
    "transform",
    "utiliz",
    "utilis",
)
_ADOPTION_RELATION_FORMS = frozenset({
    "applied",
    "applies",
    "apply",
    "applying",
    "based",
    "embedded",
    "revolutionized",
    "revolutionised",
    "underpin",
    "underpins",
    "use",
    "used",
    "uses",
    "using",
})
_PROSPECTIVE_ADOPTION_PHRASES = (
    "can be adopted",
    "can be applied",
    "can be deployed",
    "can be implemented",
    "can be integrated",
    "can be used",
    "could be adopted",
    "could be applied",
    "could be deployed",
    "could be implemented",
    "could be integrated",
    "could be used",
    "future adoption",
    "future deployment",
    "may be adopted",
    "may be applied",
    "may be deployed",
    "may be implemented",
    "may be integrated",
    "may be used",
    "potential adoption",
    "potential deployment",
    "proposed adoption",
    "proposed deployment",
    "will be adopted",
    "will be applied",
    "will be deployed",
    "will be implemented",
    "will be integrated",
    "will be used",
)


def _term_alias_token_sequences(matched_term: str) -> list[list[str]]:
    sequences: list[list[str]] = []
    full_without_parentheses = re.sub(r"\([^)]*\)", " ", matched_term)
    full_tokens = _component_tokens(full_without_parentheses)
    if full_tokens:
        sequences.append(full_tokens)
    for alias in re.findall(
        r"\(([A-Za-zА-Яа-я0-9][A-Za-zА-Яа-я0-9-]{1,14})\)", matched_term
    ):
        tokens = _component_tokens(alias)
        if tokens and tokens not in sequences:
            sequences.append(tokens)
    for sequence in list(sequences):
        last = sequence[-1]
        if (
            len(last) > 3
            and last.endswith("s")
            and (
                not last.endswith(("ss", "us", "is"))
                or last in {"apus", "cpus", "gpus", "npus", "tpus"}
            )
        ):
            singular = [*sequence[:-1], last[:-1]]
            if singular not in sequences:
                sequences.append(singular)
    return sequences


def _sequence_positions(tokens: list[str], sequence: list[str]) -> list[tuple[int, int]]:
    width = len(sequence)
    return [
        (index, index + width)
        for index in range(len(tokens) - width + 1)
        if tokens[index:index + width] == sequence
    ]


def _adoption_relation_is_explicit(quote: str, matched_term: str) -> bool:
    tokens = _component_tokens(quote)
    aliases = _term_alias_token_sequences(matched_term)
    alias_spans = [
        span for alias in aliases for span in _sequence_positions(tokens, alias)
    ]
    if not alias_spans:
        return False
    cue_sequences = [
        _component_tokens(cue) for cue in _MATURITY_QUOTE_CUES["adoption"]
    ]
    cue_spans = [
        span for cue in cue_sequences for span in _sequence_positions(tokens, cue)
    ]
    cue_spans.extend(
        (index, index + 1)
        for index, token in enumerate(tokens)
        if token in _ADOPTION_RELATION_FORMS
        or any(token.startswith(prefix) for prefix in _ADOPTION_RELATION_PREFIXES)
    )
    blockers = {
        "application", "applications", "component", "components", "device",
        "devices", "method", "methods", "network", "networks", "sensor",
        "sensors", "system", "systems", "tool", "tools",
    }
    for alias_start, alias_end in alias_spans:
        if (
            alias_start > 0
            and tokens[alias_start - 1] == "successful"
            and alias_end < len(tokens)
            and tokens[alias_end] in {"implementation", "implementations"}
        ):
            return True
        for cue_start, cue_end in cue_spans:
            if cue_end <= alias_start:
                between = tokens[cue_end:alias_start]
            elif alias_end <= cue_start:
                between = tokens[alias_end:cue_start]
            else:
                between = []
            if len(between) <= 6 and not blockers.intersection(between):
                return True
        following = tokens[alias_end:alias_end + 8]
        if following[:4] == ["has", "emerged", "as", "the"] and (
            "accelerator" in following or "infrastructure" in following
        ):
            return True
        sentence_tail = tokens[alias_end:alias_end + 25]
        if "its" in sentence_tail and (
            "adoption" in sentence_tail
            or "penetration" in sentence_tail
            or "uptake" in sentence_tail
        ):
            return True
        joined_tail = " ".join(sentence_tail)
        if (
            "critical role" in joined_tail or "essential role" in joined_tail
        ) and ("modern" in sentence_tail or "infrastructure" in sentence_tail):
            return True
    return False


def _maturity_quote_supports_kind(
    kind: str, quote: str, matched_term: str | None = None
) -> bool:
    normalized = " ".join(_component_tokens(quote))
    negative = {
        "standard": (
            "proposed standard", "proposal for a standard", "lack of standard",
            "need for standard", "standardization remain", "standardisation remain",
        ),
        "adoption": (
            "potential for adoption", "adoption still needs", "adoption is not up",
            "not widely adopted", "not widely deployed", "not widely used",
            "adoption remains limited", "deployment remains limited",
            *_PROSPECTIVE_ADOPTION_PHRASES,
        ),
        "market": (
            "market forecast", "market opportunity", "future market",
        ),
    }
    if any(cue in normalized for cue in negative.get(kind, ())):
        return False
    has_cue = any(cue in normalized for cue in _MATURITY_QUOTE_CUES.get(kind, ()))
    if kind == "adoption" and matched_term is not None:
        return _adoption_relation_is_explicit(quote, matched_term)
    return has_cue


_claim_match_is_sufficient = text_supports_matched_term
_CORE_IGNORED_TOKENS = frozenset({
    "a", "an", "and", "by", "for", "from", "in", "of", "on", "the", "to", "with"
})


def _claim_scope_normalization(
    quote: str,
    matched_term: str,
    *,
    scope: str,
    missing_components: list[str],
    allow_explicit_alias: bool = False,
) -> tuple[str, list[str]] | None:
    term_tokens = _component_tokens(matched_term)
    quote_tokens = _component_tokens(quote)
    unique_term = frozenset(term_tokens)
    unique_quote = frozenset(quote_tokens)
    full_supported = (
        bool(unique_term)
        and unique_term <= unique_quote
        and text_supports_matched_term(quote, matched_term)
    )
    if (
        allow_explicit_alias
        and not full_supported
        and plan_module.maturity_text_supports_term(quote, matched_term)
    ):
        full_supported = True
    if full_supported:
        if scope == "full_candidate" or not missing_components:
            return "full_candidate", []
        return "core_only", missing_components

    meaningful_term_tokens = [
        token for token in term_tokens if token not in _CORE_IGNORED_TOKENS
    ]
    meaningful_quote_tokens = [
        token for token in quote_tokens if token not in _CORE_IGNORED_TOKENS
    ]
    meaningful_term = frozenset(meaningful_term_tokens)
    meaningful_quote = frozenset(meaningful_quote_tokens)
    matched = meaningful_term & meaningful_quote
    minimum_core_tokens = max(2, math.ceil(len(meaningful_term) / 2))
    quote_pairs = set(
        zip(meaningful_quote_tokens, meaningful_quote_tokens[1:], strict=False)
    )
    term_positions = {
        token: index for index, token in enumerate(meaningful_term_tokens)
    }
    ordered_quote_pair = any(
        first in term_positions
        and second in term_positions
        and term_positions[first] < term_positions[second]
        for first, second in quote_pairs
    )
    declared_missing_tokens = {
        token
        for component in missing_components
        for token in _component_tokens(component)
    }
    declared_core = meaningful_term - declared_missing_tokens
    declared_core_supported = (
        len(declared_core) >= minimum_core_tokens and declared_core <= meaningful_quote
    )
    core_supported = declared_core_supported or (
        len(matched) >= minimum_core_tokens
        and (len(matched) >= 3 or ordered_quote_pair)
    )
    if not core_supported:
        return None
    derived_missing = [
        token
        for token in dict.fromkeys(meaningful_term_tokens)
        if token not in meaningful_quote
    ]
    normalized_missing = missing_components or derived_missing
    if not normalized_missing:
        return None
    return "core_only", normalized_missing


def _validate_response(
    task: Mapping[str, Any], text: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split a model answer into valid claims, per-claim issues and doc rows.

    Returns ``(claims, issues, document_rows)``. Raises ValueError when the
    whole task answer is structurally unusable.
    """

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("model answer is not valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"candidate_id", "documents"}:
        raise ValueError("model answer root must hold only candidate_id/documents")
    if payload.get("candidate_id") != task["candidate_id"]:
        raise ValueError("model answer candidate_id diverges from the task")
    records = payload.get("documents")
    if not isinstance(records, list):
        raise ValueError("model answer documents must be a list")
    expected_ids = [doc["document_id"] for doc in task["documents"]]
    passages = {doc["document_id"]: doc["passage"] for doc in task["documents"]}
    actual_ids = []
    records_by_id: dict[str, dict[str, Any]] = {}
    unresolved_records: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("model answer document record must be an object")
        document_id = record.get("document_id")
        actual_ids.append(document_id)
        if not isinstance(document_id, str) or actual_ids.count(document_id) > 1:
            raise ValueError("model answer documents diverge from the task roster")
        if document_id in passages:
            records_by_id[document_id] = record
        else:
            unresolved_records.append(record)
    for record in unresolved_records:
        raw_claims = record.get("claims")
        quotes = [
            claim.get("quote")
            for claim in raw_claims
            if isinstance(claim, dict) and isinstance(claim.get("quote"), str)
        ] if isinstance(raw_claims, list) else []
        remaining = [
            document_id
            for document_id in expected_ids
            if document_id not in records_by_id
            and quotes
            and len(quotes) == len(raw_claims)
            and all(quote in passages[document_id] for quote in quotes)
        ]
        if len(remaining) != 1:
            raise ValueError("model answer documents diverge from the task roster")
        resolved_id = remaining[0]
        records_by_id[resolved_id] = {**record, "document_id": resolved_id}
    if set(records_by_id) != set(expected_ids) or len(actual_ids) != len(expected_ids):
        raise ValueError("model answer documents diverge from the task roster")
    records = [records_by_id[document_id] for document_id in expected_ids]
    starts = {doc["document_id"]: doc["passage_start"] for doc in task["documents"]}
    matched_terms = {doc["document_id"]: doc["matched_term"] for doc in task["documents"]}
    claims: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    document_rows: list[dict[str, Any]] = []
    purpose = task.get("purpose", EVIDENCE_PURPOSE_GENERAL)
    if purpose == EVIDENCE_PURPOSE_MATURITY:
        valid_kinds = {"standard", "adoption", "market"}
        valid_directions = {"counter"}
    else:
        valid_kinds = {item.value for item in ClaimType}
        valid_directions = {item.value for item in EvidenceDirection}
    for record in records:
        document_id = record["document_id"]
        if set(record) != {"document_id", "claims"}:
            raise ValueError(f"document record {document_id!r} holds unexpected keys")
        raw_claims = record.get("claims")
        if not isinstance(raw_claims, list):
            raise ValueError(f"document {document_id!r} claims must be a list")
        if len(raw_claims) > MAX_CLAIMS_PER_DOCUMENT:
            raise ValueError(f"document {document_id!r} carries too many claims")
        passage = passages[document_id]
        kept = 0
        doc_issues = 0
        seen: set[tuple[str, str, str, str]] = set()
        for raw in raw_claims:
            if (
                not isinstance(raw, dict)
                or set(raw) != {
                    "quote",
                    "kind",
                    "direction",
                    "scope",
                    "explanation_ru",
                    "missing_components",
                }
            ):
                issues.append(_issue(task, document_id, "invalid_claim"))
                doc_issues += 1
                continue
            quote = raw["quote"]
            kind = raw["kind"]
            direction = raw["direction"]
            scope = raw["scope"]
            explanation_ru = raw["explanation_ru"]
            missing_components = raw["missing_components"]
            if (
                not isinstance(quote, str)
                or not 20 <= len(quote) <= 500
            ):
                issues.append(_issue(task, document_id, "invalid_claim"))
                doc_issues += 1
                continue
            validated_missing = _validated_missing_components(
                missing_components,
                scope=scope,
                matched_term=matched_terms[document_id],
            )
            if (
                kind not in valid_kinds
                or direction not in valid_directions
                or scope not in EVIDENCE_CLAIM_SCOPES
                or not _valid_russian_explanation(explanation_ru)
                or validated_missing is None
            ):
                issues.append(_issue(task, document_id, "invalid_claim"))
                doc_issues += 1
                continue
            if (
                purpose == EVIDENCE_PURPOSE_MATURITY
                and not _maturity_quote_supports_kind(
                    kind, quote, matched_terms[document_id]
                )
            ):
                issues.append(
                    _issue(task, document_id, "insufficient_maturity_evidence")
                )
                doc_issues += 1
                continue
            local = passage.find(quote)
            if local < 0:
                issues.append(_issue(task, document_id, "non_verbatim"))
                doc_issues += 1
                continue
            normalized_scope = _claim_scope_normalization(
                quote,
                matched_terms[document_id],
                scope=scope,
                missing_components=validated_missing,
                allow_explicit_alias=purpose == EVIDENCE_PURPOSE_MATURITY,
            )
            if normalized_scope is None:
                issues.append(_issue(task, document_id, "insufficient_term_match"))
                doc_issues += 1
                continue
            scope, validated_missing = normalized_scope
            key = (quote, kind, direction, scope)
            if key in seen:
                issues.append(_issue(task, document_id, "duplicate_claim"))
                doc_issues += 1
                continue
            seen.add(key)
            start = starts[document_id] + local
            end = start + len(quote)
            claims.append({
                "claim_id": _claim_id(
                    task["candidate_id"], document_id, kind, direction, scope, quote
                ),
                "candidate_id": task["candidate_id"],
                "task_id": task["task_id"],
                "document_id": document_id,
                "quote": quote,
                "kind": kind,
                "direction": direction,
                "scope": scope,
                "explanation_ru": explanation_ru,
                "missing_components": validated_missing,
                "locator_start": start,
                "locator_end": end,
                "locator": f"excerpt[{start}:{end}]",
            })
            kept += 1
        document_rows.append({
            "candidate_id": task["candidate_id"],
            "task_id": task["task_id"],
            "document_id": document_id,
            "status": "processed",
            "claim_count": kept,
            "issue_count": doc_issues,
        })
    return claims, issues, document_rows


def _issue(task: Mapping[str, Any], document_id: str | None, code: str) -> dict[str, Any]:
    messages = {
        "invalid_claim": "claim dropped: malformed quote, kind or direction",
        "non_verbatim": "claim dropped: quote is not verbatim in the passage",
        "duplicate_claim": "claim dropped: repeated claim kept once",
        "insufficient_term_match": (
            "claim dropped: quote does not support the declared matched_term scope"
        ),
        "insufficient_maturity_evidence": (
            "claim dropped: quote lacks an explicit maturity fact"
        ),
        "model_error": "model call failed: transport or provider error",
        "invalid_response": "model answer failed structural validation",
        "truncation": "model answer was truncated before completion",
        "content_filter": "model answer was refused by content filter",
    }
    return {
        "candidate_id": task["candidate_id"],
        "task_id": task["task_id"],
        "document_id": document_id,
        "code": code,
        "message": messages[code],
    }


def _publish_completed(
    work: Path,
    task_id: str,
    result: Mapping[str, Any],
    spec_digest: str,
    raw_text: str,
    manifest_sha: str,
    provider: str,
    model: str,
    extractor_id: str,
) -> None:
    completed_dir = work / "completed" / task_id
    if completed_dir.exists():
        raise FileExistsError(
            f"completed task already exists and must never be replaced: {completed_dir}"
        )
    result_bytes = (
        json.dumps(dict(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    raw_bytes = raw_text.encode("utf-8")
    cache_bytes = (
        json.dumps(
            {
                "schema_version": LABELING_EVIDENCE_LLM_CACHE_VERSION,
                "task_id": task_id,
                "candidate_id": result["candidate_id"],
                "spec_digest": spec_digest,
                "result": _digest(result_bytes),
                "raw_response": _digest(raw_bytes),
                "plan_manifest_sha256": manifest_sha,
                "provider": provider,
                "model": model,
                "extractor_id": extractor_id,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    staging = Path(tempfile.mkdtemp(prefix=f".{task_id}-", dir=work))
    try:
        (staging / "result.json").write_bytes(result_bytes)
        (staging / "raw_response.txt").write_bytes(raw_bytes)
        (staging / CACHE_MANIFEST_FILENAME).write_bytes(cache_bytes)
        publish_staging(staging, completed_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _next_cycle(work: Path, task_id: str) -> tuple[Path, str]:
    failures_dir = work / "failures" / task_id
    existing = sorted(
        path.name for path in failures_dir.glob("cycle-*") if path.is_dir()
    ) if failures_dir.is_dir() else []
    cycle = f"cycle-{len(existing) + 1:03d}"
    return failures_dir / cycle, cycle


def _publish_failure(
    work: Path, task_id: str, result: Mapping[str, Any], raw_text: str | None
) -> str:
    cycle_dir, cycle = _next_cycle(work, task_id)
    if cycle_dir.exists():
        raise FileExistsError(f"failure cycle already exists: {cycle_dir}")
    result_bytes = (
        json.dumps(dict(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    staging = Path(tempfile.mkdtemp(prefix=f".{task_id}-", dir=work))
    try:
        (staging / "result.json").write_bytes(result_bytes)
        if raw_text is not None:
            (staging / "raw_response.txt").write_bytes(raw_text.encode("utf-8"))
        cycle_dir.parent.mkdir(parents=True, exist_ok=True)
        publish_staging(staging, cycle_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return cycle


def _check_file_digest(path: Path, entry: Any, task_id: str) -> bytes:
    if not isinstance(entry, dict):
        raise ValueError(f"completed cache digest entry is invalid for {task_id!r}")
    size = entry.get("size_bytes")
    sha = entry.get("sha256")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError(f"completed cache size is invalid for {task_id!r}")
    if not isinstance(sha, str) or len(sha) != 64:
        raise ValueError(f"completed cache sha256 is invalid for {task_id!r}")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(
            f"completed cache file is missing for {task_id!r}: {error}"
        ) from error
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != sha:
        raise ValueError(f"completed cache checksum mismatch for {task_id!r}")
    return payload


def _check_completed_cache(
    completed_dir: Path,
    task: Mapping[str, Any],
    spec: Mapping[str, Any],
    prompt_sha: str,
    expect: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one cached task by fully reconstructing it from raw text."""

    task_id = task["task_id"]
    cache_path = completed_dir / CACHE_MANIFEST_FILENAME
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"completed cache for {task_id!r} has no readable cache manifest: {error}"
        ) from error
    if not isinstance(cache, dict):
        raise ValueError(f"completed cache manifest for {task_id!r} must be an object")
    if set(cache) != {
        "schema_version",
        "task_id",
        "candidate_id",
        "spec_digest",
        "result",
        "raw_response",
        "plan_manifest_sha256",
        "provider",
        "model",
        "extractor_id",
    }:
        raise ValueError(f"completed cache manifest keys diverge for {task_id!r}")
    if cache.get("schema_version") != LABELING_EVIDENCE_LLM_CACHE_VERSION:
        raise ValueError(
            f"completed cache schema {cache.get('schema_version')!r} "
            f"does not match {LABELING_EVIDENCE_LLM_CACHE_VERSION!r}"
        )
    if cache.get("task_id") != task_id:
        raise ValueError(f"completed cache task_id mismatch for {task_id!r}")
    if cache.get("candidate_id") != task["candidate_id"]:
        raise ValueError(f"completed cache candidate mismatch for {task_id!r}")
    if cache.get("spec_digest") != _spec_digest(spec):
        raise ValueError(f"completed cache spec_digest mismatch for {task_id!r}")
    for name in ("plan_manifest_sha256", "provider", "model", "extractor_id"):
        if cache.get(name) != expect.get(name):
            raise ValueError(f"completed cache {name} diverges for {task_id!r}")
    result_path = completed_dir / "result.json"
    _check_file_digest(result_path, cache.get("result") or {}, task_id)
    raw_path = completed_dir / "raw_response.txt"
    _check_file_digest(raw_path, cache.get("raw_response") or {}, task_id)
    try:
        raw_text = raw_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(
            f"completed cache raw text is missing for {task_id!r}: {error}"
        ) from error
    except UnicodeDecodeError as error:
        raise ValueError(
            f"completed cache raw text is not UTF-8 for {task_id!r}: {error}"
        ) from error
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"completed cache result for {task_id!r} is not valid JSON: {error}"
        ) from error
    if not isinstance(result, dict):
        raise ValueError(f"completed cache result for {task_id!r} must be an object")
    if set(result) != {
        "schema_version",
        "task_id",
        "candidate_id",
        "spec_digest",
        "plan_manifest_sha256",
        "provider",
        "model",
        "extractor_id",
        "status",
        "attempts",
        "prompt_sha256",
        "claims",
        "issues",
        "document_results",
        "raw_response_sha256",
        "note",
    }:
        raise ValueError(f"completed cache result keys diverge for {task_id!r}")
    try:
        claims, issues, document_rows = _validate_response(task, raw_text)
    except ValueError as error:
        raise ValueError(
            f"completed cache raw text does not reconstruct for {task_id!r}: {error}"
        ) from error
    expected_result = {
        "schema_version": LABELING_EVIDENCE_LLM_RESULT_VERSION,
        "task_id": task_id,
        "candidate_id": task["candidate_id"],
        "spec_digest": _spec_digest(spec),
        "plan_manifest_sha256": expect["plan_manifest_sha256"],
        "provider": expect["provider"],
        "model": expect["model"],
        "extractor_id": expect["extractor_id"],
        "status": "success",
        "attempts": 1,
        "prompt_sha256": prompt_sha,
        "claims": claims,
        "issues": issues,
        "document_results": document_rows,
        "raw_response_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "note": f"model call succeeded with {len(claims)} claims",
    }
    if result != expected_result:
        raise ValueError(f"completed cache result diverges for {task_id!r}")
    _check_no_secret_keys(result, task_id)
    return result


def _check_no_secret_keys(value: Any, task_id: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            folded = str(key).casefold()
            if (
                "header" in folded
                or "cookie" in folded
                or "authorization" in folded
                or "api_key" in folded
                or "api-key" in folded
            ):
                raise ValueError(
                    f"completed cache for {task_id!r} stores forbidden secret data"
                )
            _check_no_secret_keys(item, task_id)
    elif isinstance(value, list):
        for item in value:
            _check_no_secret_keys(item, task_id)


def _call_model(
    generator: Callable[[str], str], prompt: str
) -> tuple[str | None, str | None]:
    """Call once; return ``(text, failure_code)`` with fixed failure codes."""

    try:
        return generator(prompt), None
    except YandexTruncationError:
        return None, "truncation"
    except YandexContentFilterError:
        return None, "content_filter"
    except (RuntimeError, OSError):
        return None, "model_error"
    except ValueError:
        return None, "invalid_response"
    except Exception:
        return None, "model_error"


def _run_single_task(
    work: Path,
    generator: Callable[[str], str],
    provider: str,
    model: str,
    extractor_id: str,
    manifest_sha: str,
    task: Mapping[str, Any],
    spec: Mapping[str, Any],
    spec_digest: str,
    prompt: str,
    prompt_sha: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Call the model once, publish the outcome, return output rows."""

    task_id = task["task_id"]
    candidate_id = task["candidate_id"]
    text, failure = _call_model(generator, prompt)
    if failure is not None:
        result = {
            "schema_version": LABELING_EVIDENCE_LLM_RESULT_VERSION,
            "task_id": task_id,
            "candidate_id": candidate_id,
            "spec_digest": spec_digest,
            "status": "failed",
            "error": failure,
            "attempts": 1,
            "note": f"model call failed: {failure}",
        }
        _publish_failure(work, task_id, result, text)
        record = {
            "candidate_id": candidate_id,
            "task_id": task_id,
            "status": "failed",
            "reused": False,
            "attempts": 1,
            "error": failure,
        }
        issue = {
            "candidate_id": candidate_id,
            "task_id": task_id,
            "document_id": None,
            "code": failure,
            "message": f"model call failed: {failure}",
        }
        return record, [], [], [issue]
    if text is None:
        raise RuntimeError("model call returned neither text nor failure")
    try:
        claims, issues, document_rows = _validate_response(task, text)
    except ValueError:
        result = {
            "schema_version": LABELING_EVIDENCE_LLM_RESULT_VERSION,
            "task_id": task_id,
            "candidate_id": candidate_id,
            "spec_digest": spec_digest,
            "status": "failed",
            "error": "invalid_response",
            "attempts": 1,
            "note": "model call failed: invalid_response",
        }
        _publish_failure(work, task_id, result, text)
        record = {
            "candidate_id": candidate_id,
            "task_id": task_id,
            "status": "failed",
            "reused": False,
            "attempts": 1,
            "error": "invalid_response",
        }
        issue = {
            "candidate_id": candidate_id,
            "task_id": task_id,
            "document_id": None,
            "code": "invalid_response",
            "message": "model call failed: invalid_response",
        }
        return record, [], [], [issue]
    result = {
        "schema_version": LABELING_EVIDENCE_LLM_RESULT_VERSION,
        "task_id": task_id,
        "candidate_id": candidate_id,
        "spec_digest": spec_digest,
        "plan_manifest_sha256": manifest_sha,
        "provider": provider,
        "model": model,
        "extractor_id": extractor_id,
        "status": "success",
        "attempts": 1,
        "prompt_sha256": prompt_sha,
        "claims": claims,
        "issues": issues,
        "document_results": document_rows,
        "raw_response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "note": f"model call succeeded with {len(claims)} claims",
    }
    _publish_completed(
        work, task_id, result, spec_digest, text,
        manifest_sha, provider, model, extractor_id,
    )
    record = {
        "candidate_id": candidate_id,
        "task_id": task_id,
        "status": "success",
        "reused": False,
        "attempts": 1,
        "error": None,
    }
    return record, claims, document_rows, issues


def _reused_rows(
    task: Mapping[str, Any], result: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    claims = result.get("claims") or []
    issues = result.get("issues") or []
    document_rows = result.get("document_results") or []
    record = {
        "candidate_id": task["candidate_id"],
        "task_id": task["task_id"],
        "status": "success",
        "reused": True,
        "attempts": 0,
        "error": None,
    }
    return record, claims, document_rows, issues


def run_evidence_llm(
    *,
    plan_dir: str | Path,
    work_dir: str | Path,
    output_dir: str | Path,
    environment: Mapping[str, str],
    max_new_tasks: int | None = None,
    candidate_ids: list[str] | tuple[str, ...] | None = None,
    generator: Callable[[str], str] | None = None,
    generator_transport: Any | None = None,
) -> LabelingEvidenceLlmRunPaths:
    """Execute planned Evidence tasks with resume, then publish the result."""

    if max_new_tasks is not None and (
        not isinstance(max_new_tasks, int)
        or isinstance(max_new_tasks, bool)
        or max_new_tasks < 1
    ):
        raise ValueError("max_new_tasks must be a positive int")
    wanted: set[str] | None = None
    if candidate_ids is not None:
        wanted = set(candidate_ids)
        if not wanted or any(not isinstance(item, str) for item in wanted):
            raise ValueError("candidate_ids must be a non-empty string collection")
    settings = load_evidence_llm_settings(environment)
    if settings.selection.provider is not LlmProvider.YANDEX:
        raise ValueError(
            "evidence executor supports a Yandex evidence model, "
            f"not {settings.selection.provider.value!r}"
        )
    provider = settings.selection.provider.value
    model = settings.selection.model
    if (
        provider != QUALIFICATION_EVIDENCE_PROVIDER
        or model != QUALIFICATION_EVIDENCE_MODEL
    ):
        raise ValueError(
            "qualification Evidence requires "
            f"{QUALIFICATION_EVIDENCE_PROVIDER}/{QUALIFICATION_EVIDENCE_MODEL}; "
            f"resolved {provider}/{model}. Use config/hackathon.env as --env-file "
            "and keep credentials in the automatically loaded .env"
        )
    extractor_id = _extractor_id(provider, model)

    plan_path = Path(plan_dir)
    manifest, tasks, coverage, digests = _load_plan(plan_path)
    bundle_id = manifest.get("bundle_id")
    cutoff = manifest.get("cutoff_date")
    validated, coverage_by_id = _validate(
        tasks, coverage, digests["input_manifest"]["sha256"]
    )
    if wanted is not None:
        unknown = sorted(wanted - set(coverage_by_id))
        if unknown:
            raise ValueError(f"unknown candidate_id filter: {', '.join(unknown)}")

    output = Path(output_dir)
    if output.exists():
        raise ValueError(f"output directory already exists: {output}")
    fingerprint = _work_fingerprint(
        digests, bundle_id, cutoff, provider, model, extractor_id, len(validated)
    )
    work = _init_or_check_work(work_dir, fingerprint)

    active = (
        generator
        or build_json_generator(
            settings,
            transport=generator_transport,
            json_schema=EVIDENCE_RESPONSE_JSON_SCHEMA,
            schema_name="evidence_response",
            max_output_tokens=EVIDENCE_MAX_OUTPUT_TOKENS,
            server_side_json_schema=True,
        )
    )
    prompts: dict[str, str] = {}
    prompt_shas: dict[str, str] = {}
    specs: dict[str, dict[str, Any]] = {}
    for task in validated.values():
        prompt = build_evidence_prompt(task)
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        prompts[task["task_id"]] = prompt
        prompt_shas[task["task_id"]] = prompt_sha
        specs[task["task_id"]] = _task_spec(task, prompt_sha, provider, model, extractor_id)

    selected = [
        task for candidate_id, task in validated.items()
        if wanted is None or candidate_id in wanted
    ]
    pending: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []
    document_rows: list[dict[str, Any]] = []
    issue_rows: list[dict[str, Any]] = []
    called = 0
    reused = 0
    expect = {
        "plan_manifest_sha256": digests["manifest.json"]["sha256"],
        "provider": provider,
        "model": model,
        "extractor_id": extractor_id,
    }
    for task in selected:
        task_id = task["task_id"]
        completed_dir = work / "completed" / task_id
        if completed_dir.exists():
            result = _check_completed_cache(
                completed_dir, task, specs[task_id], prompt_shas[task_id], expect
            )
            record, claims, documents, issues = _reused_rows(task, result)
            request_rows.append(record)
            claim_rows.extend(claims)
            document_rows.extend(documents)
            issue_rows.extend(issues)
            reused += 1
        else:
            pending.append(task)
    if max_new_tasks is not None:
        pending = pending[:max_new_tasks]

    if pending:
        with ThreadPoolExecutor(max_workers=EVIDENCE_CONCURRENCY) as pool:
            futures = {
                pool.submit(
                    _run_single_task,
                    work,
                    active,
                    provider,
                    model,
                    extractor_id,
                    digests["manifest.json"]["sha256"],
                    task,
                    specs[task["task_id"]],
                    _spec_digest(specs[task["task_id"]]),
                    prompts[task["task_id"]],
                    prompt_shas[task["task_id"]],
                ): task
                for task in pending
            }
            finished: dict[str, tuple] = {}
            for future in as_completed(futures):
                task = futures[future]
                finished[task["task_id"]] = future.result()
            for task in pending:
                record, claims, documents, issues = finished[task["task_id"]]
                request_rows.append(record)
                claim_rows.extend(claims)
                document_rows.extend(documents)
                issue_rows.extend(issues)
                called += 1

    ran_ids = {row["task_id"] for row in request_rows}
    for task in validated.values():
        if task["task_id"] in ran_ids:
            continue
        request_rows.append({
            "candidate_id": task["candidate_id"],
            "task_id": task["task_id"],
            "status": "not_run",
            "reused": False,
            "attempts": 0,
            "error": None,
        })
    request_rows.sort(key=lambda row: (row["candidate_id"], row["task_id"]))
    claim_rows.sort(
        key=lambda row: (row["candidate_id"], row["task_id"], row["document_id"], row["claim_id"])
    )
    document_rows.sort(
        key=lambda row: (row["candidate_id"], row["task_id"], row["document_id"])
    )
    issue_rows.sort(key=lambda row: (
        row["candidate_id"],
        row["task_id"],
        row["document_id"] or "",
        row["code"],
    ))

    coverage_rows: list[dict[str, Any]] = []
    by_request = {row["task_id"]: row for row in request_rows}
    for candidate_id in sorted(coverage_by_id):
        plan_row = coverage_by_id[candidate_id]
        if plan_row.get("status") == "no_input":
            coverage_rows.append({
                "candidate_id": candidate_id,
                "status": "no_input",
                "task_id": None,
                "input_documents": 0,
                "scientific_documents": 0,
                "industry_documents": 0,
                "prompt_chars": 0,
                "estimated_input_tokens": 0,
                "llm_call_planned": False,
                "empty_reasons": list(plan_row.get("empty_reasons") or []),
            })
            continue
        task = validated[candidate_id]
        request = by_request[task["task_id"]]
        scientific = sum(
            1 for doc in task["documents"] if doc["source_class"] == "scientific"
        )
        coverage_rows.append({
            "candidate_id": candidate_id,
            "status": (
                "complete"
                if request["status"] == "success"
                else request["status"]
            ),
            "task_id": task["task_id"],
            "input_documents": len(task["documents"]),
            "scientific_documents": scientific,
            "industry_documents": len(task["documents"]) - scientific,
            "prompt_chars": task["prompt_chars"],
            "estimated_input_tokens": task["estimated_input_tokens"],
            "llm_call_planned": True,
            "reused": request["reused"],
            "error": request["error"],
        })

    success = sum(1 for row in request_rows if row["status"] == "success")
    failed = sum(1 for row in request_rows if row["status"] == "failed")
    not_run = sum(1 for row in request_rows if row["status"] == "not_run")
    no_input = sum(1 for row in coverage_rows if row["status"] == "no_input")
    totals = {
        "candidates": len(coverage_rows),
        "planned_tasks": len(request_rows),
        "called": called,
        "reused": reused,
        "success": success,
        "failed": failed,
        "not_run": not_run,
        "no_input": no_input,
        "input_documents_processed": sum(
            row["input_documents"]
            for row in coverage_rows
            if row["status"] == "complete"
        ),
        "input_documents_attempted": sum(
            row["input_documents"]
            for row in coverage_rows
            if row["status"] in ("complete", "failed")
        ),
        "claims": len(claim_rows),
        "issues": len(issue_rows),
    }
    analysis_status = (
        "complete"
        if failed == 0
        and not_run == 0
        and success + no_input == len(coverage_rows)
        and success == len(validated)
        else "partial"
    )
    files = {
        REQUEST_RESULTS_FILENAME: _render_jsonl(request_rows),
        CLAIMS_FILENAME: _render_jsonl(claim_rows),
        DOCUMENT_RESULTS_FILENAME: _render_jsonl(document_rows),
        ISSUES_FILENAME: _render_jsonl(issue_rows),
        COVERAGE_FILENAME: _render_jsonl(coverage_rows),
    }
    applied_filter = sorted(wanted) if wanted is not None else None
    manifest_out = {
        "schema_version": LABELING_EVIDENCE_LLM_RESULT_VERSION,
        "analysis_status": analysis_status,
        "bundle_id": bundle_id,
        "cutoff_date": cutoff,
        "plan_manifest": digests["manifest.json"],
        "plan_files": {
            "tasks.jsonl": digests["tasks.jsonl"],
            "coverage.jsonl": digests["coverage.jsonl"],
        },
        "provider": provider,
        "model": model,
        "extractor_id": extractor_id,
        "max_output_tokens": EVIDENCE_MAX_OUTPUT_TOKENS,
        "concurrency": EVIDENCE_CONCURRENCY,
        "applied_candidate_filter": applied_filter,
        "applied_max_new_tasks": max_new_tasks,
        "totals": totals,
        "outputs": {
            name: {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for name, payload in files.items()
        },
    }
    files[RESULT_MANIFEST_FILENAME] = (
        json.dumps(manifest_out, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    paths = publish_artifact_bundle(files, output)
    return LabelingEvidenceLlmRunPaths(
        manifest=paths[RESULT_MANIFEST_FILENAME],
        request_results=paths[REQUEST_RESULTS_FILENAME],
        claims=paths[CLAIMS_FILENAME],
        document_results=paths[DOCUMENT_RESULTS_FILENAME],
        issues=paths[ISSUES_FILENAME],
        coverage=paths[COVERAGE_FILENAME],
    )
