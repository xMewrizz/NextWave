"""Resumable executor for query-wide Exa industry enrichment."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from nextwave.datasets.artifacts import publish_artifact_bundle
from nextwave.sources import (
    ConnectorStatus,
    ExaConnector,
    ExaTransport,
    QueryPurpose,
    SnapshotManifest,
    SnapshotWriter,
    SourceQuery,
    build_exa_news_request,
    parse_exa_response,
    publish_staging,
)

from .exa_enrichment_plan import EXA_ENRICHMENT_PLAN_VERSION

EXA_ENRICHMENT_EXECUTOR_VERSION = "analysis-exa-enrichment-executor-v1"
EXA_ENRICHMENT_WORK_VERSION = "analysis-exa-enrichment-work-v1"
EXA_ENRICHMENT_RESULT_VERSION = "analysis-exa-enrichment-result-v1"
_BACKOFF = (1.0, 3.0)
_MAX_RETRY_AFTER = 120.0


@dataclass(frozen=True, slots=True)
class ExaEnrichmentRunPaths:
    manifest: Path
    request_results: Path
    documents: Path
    coverage: Path


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


def _digest(data: bytes) -> dict[str, Any]:
    return {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _canonical(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode()


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:16]


def _load_plan(root: Path) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    try:
        plan_bytes = (root / "plan.json").read_bytes()
        plan = json.loads(plan_bytes.decode("utf-8"))
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("cannot read Exa enrichment plan") from error
    if not isinstance(plan, dict) or not isinstance(manifest, dict):
        raise ValueError("Exa plan and manifest must be objects")
    if plan.get("schema_version") != EXA_ENRICHMENT_PLAN_VERSION:
        raise ValueError("Exa enrichment plan version is not supported")
    if manifest.get("schema_version") != EXA_ENRICHMENT_PLAN_VERSION:
        raise ValueError("Exa enrichment manifest version is not supported")
    if (manifest.get("outputs") or {}).get("plan.json") != _digest(plan_bytes):
        raise ValueError("Exa enrichment plan checksum differs from manifest")
    tasks = plan.get("tasks")
    totals = plan.get("totals")
    if not isinstance(tasks, list) or not isinstance(totals, dict):
        raise ValueError("Exa enrichment plan misses tasks or totals")
    if totals.get("requests") != len(tasks):
        raise ValueError("Exa request total does not match tasks")
    cutoff = plan.get("cutoff_date")
    if not isinstance(cutoff, str):
        raise ValueError("Exa cutoff_date must be a date")
    date.fromisoformat(cutoff)
    seen: set[str] = set()
    candidates: set[str] = set()
    for index, task in enumerate(tasks, 1):
        if not isinstance(task, dict):
            raise ValueError(f"Exa task {index} must be an object")
        candidate_id = task.get("candidate_id")
        window = task.get("window")
        term = task.get("search_text")
        request = task.get("request")
        if not isinstance(candidate_id, str) or window not in {"previous", "recent"}:
            raise ValueError(f"Exa task {index} identity is invalid")
        if not isinstance(term, str) or not term.strip() or not isinstance(request, dict):
            raise ValueError(f"Exa task {index} request is invalid")
        request_id = request.get("request_id")
        if not isinstance(request_id, str) or request_id in seen:
            raise ValueError(f"Exa task {index} request_id is invalid or duplicate")
        seen.add(request_id)
        candidates.add(candidate_id)
        query = _source_query(task, date.fromisoformat(cutoff))
        expected = build_exa_news_request(query, search_text=term, num_results=10)
        if expected.to_dict() != request:
            raise ValueError(f"Exa task {request_id!r} does not match its request")
    if totals.get("candidates") != len(candidates):
        raise ValueError("Exa candidate total does not match tasks")
    return plan, plan_bytes, manifest


def _source_query(task: Mapping[str, Any], cutoff: date) -> SourceQuery:
    request = task["request"]
    parameters = {item["name"]: item["value"] for item in request["parameters"]}
    candidate_id = task["candidate_id"]
    term = task["search_text"]
    return SourceQuery(
        query_id=request["query_id"],
        analysis_scope_id=f"scope-{_stable_id(candidate_id)}",
        purpose=QueryPurpose.HISTORICAL_ENRICHMENT,
        raw_query=term,
        normalized_query=term.casefold(),
        search_texts=(term,),
        published_from=date.fromisoformat(parameters["startPublishedDate"][:10]),
        published_until=date.fromisoformat(parameters["endPublishedDate"][:10]),
        cutoff_date=cutoff,
        languages=("en", "ru"),
    )


def _prepare_work(work: Path, plan_bytes: bytes, plan: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": EXA_ENRICHMENT_WORK_VERSION,
        "executor_version": EXA_ENRICHMENT_EXECUTOR_VERSION,
        "plan": _digest(plan_bytes),
        "bundle_id": plan.get("bundle_id"),
    }
    manifest_path = work / "work_manifest.json"
    if work.exists():
        try:
            stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("existing Exa work is invalid") from error
        if (
            stored != expected
            or not (work / "completed").is_dir()
            or not (work / "failures").is_dir()
        ):
            raise ValueError("existing Exa work belongs to another plan or is incomplete")
        return
    work.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{work.name}-", suffix=".staging", dir=work.parent))
    (staging / "completed").mkdir()
    (staging / "failures").mkdir()
    (staging / "work_manifest.json").write_bytes(_canonical(expected))
    publish_staging(staging, work)


def _spec_digest(task: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(task)).hexdigest()


def _load_cached(work: Path, task: Mapping[str, Any]) -> dict[str, Any] | None:
    request_id = task["request"]["request_id"]
    root = work / "completed" / request_id
    if not root.exists():
        return None
    try:
        raw = (root / "result.json").read_bytes()
        cache = json.loads((root / "cache_manifest.json").read_text(encoding="utf-8"))
        result = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"completed Exa cache {request_id!r} is invalid") from error
    snapshot_entries = cache.get("snapshots")
    expected_cache = {
        "request_id": request_id,
        "spec_digest": _spec_digest(task),
        "result": _digest(raw),
        "snapshots": snapshot_entries,
    }
    if cache != expected_cache or not isinstance(snapshot_entries, list):
        raise ValueError(f"completed Exa cache {request_id!r} failed integrity checks")
    actual_entries = _snapshot_digests(root / "snapshots")
    if snapshot_entries != actual_entries:
        raise ValueError(f"completed Exa cache {request_id!r} snapshots are corrupted")
    if result.get("request_id") != request_id or result.get("status") != "success":
        raise ValueError(f"completed Exa cache {request_id!r} has invalid result")
    result["reused"] = True
    return result


def _snapshot_digests(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise ValueError("completed Exa cache misses snapshots")
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        raw = path.read_bytes()
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                **_digest(raw),
            }
        )
    if not entries:
        raise ValueError("completed Exa cache snapshots are empty")
    return entries


def _execute_one(
    task: Mapping[str, Any],
    *,
    cutoff: date,
    work: Path,
    connector: ExaConnector,
    sleeper,
) -> dict[str, Any]:
    request_id = task["request"]["request_id"]
    staging = Path(tempfile.mkdtemp(prefix=f".{request_id}-", suffix=".staging", dir=work))
    query = _source_query(task, cutoff)
    attempts: list[dict[str, Any]] = []
    final_run = None
    final_writer = None
    for attempt in range(1, 4):
        snapshot_id = f"snapshot-{request_id.removeprefix('request-')}-{attempt}"
        writer = SnapshotWriter(staging / "snapshots", snapshot_id)
        run = connector.run_search(
            query,
            writer,
            search_text=task["search_text"],
            num_results=10,
            attempt=attempt,
        )
        writer.finalize(
            SnapshotManifest(
                snapshot_id=snapshot_id,
                snapshot_version="exa-search-v1",
                analysis_id=f"analysis-{_stable_id(task['candidate_id'])}",
                created_at=run.finished_at,
                query=query,
                runs=(run,),
            )
        )
        attempts.append(
            {
                "attempt": attempt,
                "status": run.status.value,
                "http_status": run.http_status,
                "error_code": run.error.code if run.error else None,
            }
        )
        final_run, final_writer = run, writer
        if run.status is ConnectorStatus.SUCCESS or not run.error or not run.error.retryable:
            break
        delay = _BACKOFF[attempt - 1] if attempt <= len(_BACKOFF) else 0
        retry_after = run.error.retry_after_seconds
        if retry_after is not None and retry_after > _MAX_RETRY_AFTER:
            break
        if retry_after is not None:
            delay = max(delay, retry_after)
        if attempt < 3 and delay:
            sleeper(delay)
    if final_run is None or final_writer is None:
        raise RuntimeError("Exa execution produced no attempt")
    documents: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    status = "failed"
    if final_run.status is ConnectorStatus.SUCCESS and final_run.artifact is not None:
        snapshot_path = final_writer.final_path
        payload = snapshot_path.joinpath(*final_run.artifact.uri.split("/")).read_bytes()
        parsed = parse_exa_response(
            payload,
            snapshot_id=final_writer.snapshot_id,
            retrieved_at=final_run.finished_at,
            cutoff_date=cutoff,
        )
        accepted_documents = []
        issues = [_json_value(asdict(issue)) for issue in parsed.issues]
        for document in parsed.documents:
            if document.published_at is not None and not (
                query.published_from <= document.published_at <= query.published_until
            ):
                issues.append(
                    {
                        "record_index": -1,
                        "external_id": document.external_id,
                        "code": "outside_planned_window",
                        "message": "publishedDate is outside the planned request window",
                    }
                )
                continue
            accepted_documents.append(document)
        documents = [_json_value(asdict(document)) for document in accepted_documents]
        status = "success"
    result = {
        "request_id": request_id,
        "candidate_id": task["candidate_id"],
        "window": task["window"],
        "term_rank": task["term_rank"],
        "search_text": task["search_text"],
        "status": status,
        "reused": False,
        "attempts": attempts,
        "returned_records": final_run.returned_records,
        "documents": documents,
        "parse_issues": issues,
        "error_code": final_run.error.code if final_run.error else None,
    }
    raw = _canonical(result)
    (staging / "result.json").write_bytes(raw)
    if status == "success":
        (staging / "cache_manifest.json").write_bytes(
            _canonical(
                {
                    "request_id": request_id,
                    "spec_digest": _spec_digest(task),
                    "result": _digest(raw),
                    "snapshots": _snapshot_digests(staging / "snapshots"),
                }
            )
        )
        publish_staging(staging, work / "completed" / request_id)
    else:
        cycles = list((work / "failures" / request_id).glob("cycle-*"))
        target = work / "failures" / request_id / f"cycle-{len(cycles) + 1:03d}"
        target.parent.mkdir(parents=True, exist_ok=True)
        publish_staging(staging, target)
    return result


def run_exa_enrichment(
    *,
    plan_dir: str | Path,
    work_dir: str | Path,
    output_dir: str | Path,
    environment: Mapping[str, str],
    max_new_requests: int | None = None,
    concurrency: int = 5,
    transport: ExaTransport | None = None,
    sleeper=time.sleep,
) -> ExaEnrichmentRunPaths:
    plan, plan_bytes, _ = _load_plan(Path(plan_dir))
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if max_new_requests is not None and max_new_requests < 0:
        raise ValueError("max_new_requests must be non-negative")
    if not 1 <= concurrency <= 10:
        raise ValueError("concurrency must be between 1 and 10")
    work = Path(work_dir)
    _prepare_work(work, plan_bytes, plan)
    key = environment.get("NEXTWAVE_EXA_API_KEY", "")
    connector = ExaConnector(api_key=key, transport=transport)
    tasks = plan["tasks"]
    results: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for task in tasks:
        cached = _load_cached(work, task)
        if cached is not None:
            results.append(cached)
        elif max_new_requests is None or len(pending) < max_new_requests:
            pending.append(task)
        else:
            results.append(
                {
                    "request_id": task["request"]["request_id"],
                    "candidate_id": task["candidate_id"],
                    "window": task["window"],
                    "term_rank": task["term_rank"],
                    "search_text": task["search_text"],
                    "status": "not_run",
                    "reused": False,
                    "attempts": [],
                    "returned_records": None,
                    "documents": [],
                    "parse_issues": [],
                    "error_code": None,
                }
            )
    cutoff = date.fromisoformat(plan["cutoff_date"])
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                _execute_one,
                task,
                cutoff=cutoff,
                work=work,
                connector=connector,
                sleeper=sleeper,
            ): task
            for task in pending
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item["request_id"])
    result_rows = b"".join(_canonical(item) for item in results)
    document_rows: list[dict[str, Any]] = []
    for result in results:
        for document in result["documents"]:
            document_rows.append(
                {
                    "candidate_id": result["candidate_id"],
                    "window": result["window"],
                    "request_id": result["request_id"],
                    "document": document,
                }
            )
    document_rows.sort(
        key=lambda item: (
            item["candidate_id"], item["window"], item["document"]["document_id"]
        )
    )
    documents_bytes = b"".join(_canonical(item) for item in document_rows)
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_candidate.setdefault(result["candidate_id"], []).append(result)
    coverage_rows: list[dict[str, Any]] = []
    for candidate_id in sorted(by_candidate):
        rows = by_candidate[candidate_id]
        successful = sum(row["status"] == "success" for row in rows)
        status = "complete" if successful == len(rows) else ("partial" if successful else "unknown")
        coverage_rows.append(
            {
                "candidate_id": candidate_id,
                "status": status,
                "planned_requests": len(rows),
                "successful_requests": successful,
                "failed_requests": sum(row["status"] == "failed" for row in rows),
                "not_run_requests": sum(row["status"] == "not_run" for row in rows),
                "returned_documents": sum(len(row["documents"]) for row in rows),
                "windows_present": sorted(
                    {row["window"] for row in rows if row["documents"]}
                ),
            }
        )
    coverage_bytes = b"".join(_canonical(item) for item in coverage_rows)
    files = {
        "request_results.jsonl": result_rows,
        "documents.jsonl": documents_bytes,
        "coverage.jsonl": coverage_bytes,
    }
    totals = {
        "candidates": len(coverage_rows),
        "planned_requests": len(results),
        "successful_requests": sum(row["status"] == "success" for row in results),
        "failed_requests": sum(row["status"] == "failed" for row in results),
        "not_run_requests": sum(row["status"] == "not_run" for row in results),
        "reused_requests": sum(bool(row["reused"]) for row in results),
        "documents": len(document_rows),
    }
    manifest_bytes = _canonical(
        {
            "schema_version": EXA_ENRICHMENT_RESULT_VERSION,
            "executor_version": EXA_ENRICHMENT_EXECUTOR_VERSION,
            "plan": _digest(plan_bytes),
            "bundle_id": plan.get("bundle_id"),
            "concurrency": concurrency,
            "totals": totals,
            "outputs": {name: _digest(payload) for name, payload in files.items()},
        }
    )
    paths = publish_artifact_bundle({**files, "manifest.json": manifest_bytes}, output)
    return ExaEnrichmentRunPaths(
        manifest=paths["manifest.json"],
        request_results=paths["request_results.jsonl"],
        documents=paths["documents.jsonl"],
        coverage=paths["coverage.jsonl"],
    )
