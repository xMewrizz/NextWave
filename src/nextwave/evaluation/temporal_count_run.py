"""Resumable execution of frozen OpenAlex temporal count tasks."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from nextwave.datasets.artifacts import publish_artifact_bundle
from nextwave.sources import (
    ConnectorStatus,
    OpenAlexConnector,
    QueryPurpose,
    RetrievalChannel,
    SnapshotManifest,
    SnapshotWriter,
    SourceQuery,
    publish_staging,
)

from .temporal_count_plan import (
    MANIFEST_FILENAME,
    PLAN_FILENAME,
    TEMPORAL_COUNT_PLAN_VERSION,
    _task_id,
)

TEMPORAL_COUNT_EXECUTOR_VERSION = "openalex-temporal-count-executor-v4"
TEMPORAL_COUNT_WORK_VERSION = "openalex-temporal-count-work-v4"
TEMPORAL_COUNT_RESULT_VERSION = "openalex-temporal-count-result-v4"
TEMPORAL_COUNT_CACHE_VERSION = "openalex-temporal-count-cache-v4"
COUNT_RESULTS_FILENAME = "count_results.jsonl"
CANDIDATE_FEATURES_FILENAME = "candidate_temporal_features.jsonl"
_TASK_KEYS = {
    "count_id",
    "entity_type",
    "entity_id",
    "analysis_scope_key",
    "search_text",
    "scope_search_text",
    "search_mode",
    "window",
    "published_from",
    "published_until",
    "languages",
    "channel",
    "per_page",
    "count_field",
    "population",
}


def _digest(payload: bytes) -> dict[str, Any]:
    return {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _canonical_bytes(value: object, *, indent: int | None = None) -> bytes:
    suffix = "\n"
    return (json.dumps(value, ensure_ascii=False, indent=indent, sort_keys=True) + suffix).encode()


def _spec_digest(task: Mapping[str, Any]) -> str:
    payload = json.dumps(task, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_plan(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    manifest_bytes = (directory / MANIFEST_FILENAME).read_bytes()
    manifest = json.loads(manifest_bytes)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != TEMPORAL_COUNT_PLAN_VERSION
    ):
        raise ValueError("temporal count plan manifest version does not match executor")
    plan_bytes = (directory / PLAN_FILENAME).read_bytes()
    if (manifest.get("outputs") or {}).get(PLAN_FILENAME) != _digest(plan_bytes):
        raise ValueError("temporal count plan diverges from manifest")
    plan = json.loads(plan_bytes)
    if not isinstance(plan, dict) or plan.get("schema_version") != TEMPORAL_COUNT_PLAN_VERSION:
        raise ValueError("temporal count plan version does not match executor")
    cutoff_raw = plan.get("cutoff_date")
    try:
        cutoff = date.fromisoformat(cutoff_raw)
    except (TypeError, ValueError) as error:
        raise ValueError("temporal count plan cutoff is not an ISO date") from error
    if manifest.get("cutoff_date") != cutoff.isoformat():
        raise ValueError("temporal count cutoff differs between plan and manifest")
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or any(not isinstance(task, dict) for task in tasks):
        raise ValueError("temporal count tasks must be a list of objects")
    candidates = plan.get("candidates")
    scope_queries = plan.get("scope_queries")
    if not isinstance(candidates, list) or any(
        not isinstance(candidate, dict) for candidate in candidates
    ):
        raise ValueError("temporal count candidates must be a list of objects")
    if not isinstance(scope_queries, dict) or not scope_queries:
        raise ValueError("temporal count scope_queries must be a non-empty object")
    candidate_index: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        candidate_id = candidate.get("candidate_id")
        scope = candidate.get("analysis_scope_key")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in candidate_index:
            raise ValueError("temporal count candidates need unique candidate_id")
        if scope not in scope_queries:
            raise ValueError(f"temporal count candidate {candidate_id} has unknown scope")
        candidate_index[candidate_id] = candidate
    seen: set[str] = set()
    seen_pairs: set[tuple[str, str, str]] = set()
    for task in tasks:
        if set(task) != _TASK_KEYS:
            raise ValueError("temporal count task fields do not match contract")
        count_id = task.get("count_id")
        if not isinstance(count_id, str) or count_id in seen:
            raise ValueError("temporal count IDs must be unique strings")
        seen.add(count_id)
        spec = {key: value for key, value in task.items() if key != "count_id"}
        if _task_id(spec) != count_id:
            raise ValueError(f"temporal count ID does not match task {count_id!r}")
        if task.get("entity_type") not in {"candidate", "scope"}:
            raise ValueError(f"temporal count task {count_id} has invalid entity_type")
        if task.get("window") not in {"previous", "recent"}:
            raise ValueError(f"temporal count task {count_id} has invalid window")
        if task.get("languages") != ["en", "ru"]:
            raise ValueError(f"temporal count task {count_id} has invalid languages")
        if (
            task.get("channel") != "text"
            or task.get("per_page") != 1
            or task.get("count_field") != "meta.count"
            or task.get("population") != "works_with_abstract_in_openalex"
        ):
            raise ValueError(f"temporal count task {count_id} changes retrieval policy")
        entity_type = str(task["entity_type"])
        entity_id = str(task["entity_id"])
        pair = (entity_type, entity_id, str(task["window"]))
        if pair in seen_pairs:
            raise ValueError("temporal count tasks duplicate entity/window")
        seen_pairs.add(pair)
        if entity_type == "candidate":
            candidate = candidate_index.get(entity_id)
            if candidate is None or candidate.get("analysis_scope_key") != task.get(
                "analysis_scope_key"
            ):
                raise ValueError(f"temporal count task {count_id} mismatches candidate")
        elif entity_id not in scope_queries or entity_id != task.get("analysis_scope_key"):
            raise ValueError(f"temporal count task {count_id} mismatches scope")
        scope = str(task["analysis_scope_key"])
        if task.get("scope_search_text") != scope_queries[scope]:
            raise ValueError(f"temporal count task {count_id} mismatches scope query")
        if entity_type == "scope" and task.get("search_text") != scope_queries[scope]:
            raise ValueError(f"temporal count task {count_id} changes scope search text")
        _query_from_task(task, cutoff=cutoff)
    expected_pairs = {
        ("candidate", candidate_id, window)
        for candidate_id in candidate_index
        for window in ("previous", "recent")
    } | {("scope", scope, window) for scope in scope_queries for window in ("previous", "recent")}
    if seen_pairs != expected_pairs:
        raise ValueError("temporal count tasks do not cover every candidate and scope window")
    counts = manifest.get("counts") or {}
    expected_counts = {
        "candidates": len(candidate_index),
        "scopes": len(scope_queries),
        "candidate_tasks": 2 * len(candidate_index),
        "scope_tasks": 2 * len(scope_queries),
        "tasks": len(expected_pairs),
    }
    if counts != expected_counts:
        raise ValueError("temporal count manifest task count does not match plan")
    return plan, tasks, hashlib.sha256(plan_bytes).hexdigest()


def _query_from_task(task: Mapping[str, Any], *, cutoff: date) -> SourceQuery:
    count_id = str(task["count_id"])
    search_text = task.get("search_text")
    scope_search_text = task.get("scope_search_text")
    scope = task.get("analysis_scope_key")
    if not isinstance(search_text, str) or not search_text.strip():
        raise ValueError(f"temporal count task {count_id} has blank search_text")
    if not isinstance(scope, str) or not scope:
        raise ValueError(f"temporal count task {count_id} has invalid scope")
    if not isinstance(scope_search_text, str) or not scope_search_text.strip():
        raise ValueError(f"temporal count task {count_id} has blank scope_search_text")
    mode = task.get("search_mode")
    if mode == "candidate_proximity_5":
        if '"' in search_text:
            raise ValueError(f"temporal count task {count_id} cannot quote search_text")
        query_expression = f'("{search_text}"~5)'
    elif mode == "boolean_scope":
        query_expression = search_text
    else:
        raise ValueError(f"temporal count task {count_id} has invalid search_mode")
    raw_query = search_text if len(search_text) <= 200 else scope
    return SourceQuery(
        query_id=f"query-{count_id.removeprefix('count-')}",
        analysis_scope_id=scope,
        purpose=QueryPurpose.HISTORICAL_ENRICHMENT,
        raw_query=raw_query,
        normalized_query=raw_query.casefold(),
        search_texts=(query_expression,),
        published_from=date.fromisoformat(str(task["published_from"])),
        published_until=date.fromisoformat(str(task["published_until"])),
        cutoff_date=cutoff,
        languages=("en", "ru"),
    )


def _prepare_work(
    work_dir: Path, plan_digest: str, task_count: int, cutoff: date
) -> None:
    expected = {
        "schema_version": TEMPORAL_COUNT_WORK_VERSION,
        "executor_version": TEMPORAL_COUNT_EXECUTOR_VERSION,
        "plan_sha256": plan_digest,
        "task_count": task_count,
        "cutoff_date": cutoff.isoformat(),
    }
    manifest_path = work_dir / "work_manifest.json"
    if work_dir.exists():
        if not manifest_path.is_file():
            raise ValueError("temporal count work exists without work_manifest.json")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != expected:
            raise ValueError("temporal count work belongs to another plan")
        for name in ("completed", "failures"):
            if not (work_dir / name).is_dir():
                raise ValueError(f"temporal count work misses {name}/")
        return
    parent = work_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{work_dir.name}-", dir=parent))
    try:
        (staging / "completed").mkdir()
        (staging / "failures").mkdir()
        (staging / "work_manifest.json").write_bytes(_canonical_bytes(expected, indent=2))
        publish_staging(staging, work_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _verify_snapshot(directory: Path, expected_snapshot_id: str) -> None:
    manifest_path = directory / "snapshot" / "manifest.json"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("snapshot_id") != expected_snapshot_id:
        raise ValueError("temporal count cache has invalid snapshot manifest")
    runs = value.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("temporal count cache snapshot has no runs")
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("temporal count cache snapshot run must be an object")
        artifact = run.get("artifact")
        if artifact is None:
            continue
        if not isinstance(artifact, dict):
            raise ValueError("temporal count cache artifact must be an object")
        uri = artifact.get("uri")
        if (
            not isinstance(uri, str)
            or Path(uri).is_absolute()
            or ".." in Path(uri).parts
            or "\\" in uri
        ):
            raise ValueError("temporal count cache artifact path is unsafe")
        payload = (directory / "snapshot" / Path(uri)).read_bytes()
        if _digest(payload) != {
            "size_bytes": artifact.get("size_bytes"),
            "sha256": artifact.get("sha256"),
        }:
            raise ValueError("temporal count cache raw artifact diverges")


def _load_completed(directory: Path, task: Mapping[str, Any]) -> dict[str, Any] | None:
    if not directory.exists():
        return None
    result_bytes = (directory / "result.json").read_bytes()
    snapshot_bytes = (directory / "snapshot" / "manifest.json").read_bytes()
    cache = json.loads((directory / "cache_manifest.json").read_text(encoding="utf-8"))
    expected_cache = {
        "schema_version": TEMPORAL_COUNT_CACHE_VERSION,
        "count_id": task["count_id"],
        "spec_digest": _spec_digest(task),
        "result": _digest(result_bytes),
        "snapshot_manifest": _digest(snapshot_bytes),
    }
    if cache != expected_cache:
        raise ValueError(f"temporal count cache {task['count_id']} diverges")
    result = json.loads(result_bytes)
    if (
        not isinstance(result, dict)
        or result.get("task") != task
        or result.get("spec_digest") != _spec_digest(task)
        or result.get("status") != "complete"
        or type(result.get("count")) is not int
        or result["count"] < 0
    ):
        raise ValueError(f"temporal count cache {task['count_id']} is invalid")
    _verify_snapshot(directory, str(result.get("snapshot_id")))
    return result


def _retry_delay(run: Any, attempt: int) -> float | None:
    if run.error is None or not run.error.retryable or attempt >= 3:
        return None
    backoff = (5.0, 15.0)[attempt - 1]
    retry_after = run.error.retry_after_seconds or 0.0
    delay = max(backoff, retry_after)
    return delay if delay <= 120 else None


def _execute_task(
    task: dict[str, Any],
    *,
    work_dir: Path,
    connector: OpenAlexConnector,
    sleeper: Callable[[float], None],
    cutoff: date,
) -> dict[str, Any]:
    count_id = task["count_id"]
    final = work_dir / "completed" / count_id
    cached = _load_completed(final, task)
    if cached is not None:
        return {**cached, "reused": True}
    staging = Path(tempfile.mkdtemp(prefix=f".{count_id}-", dir=work_dir / "completed"))
    snapshot_id = "snapshot"
    writer = SnapshotWriter(staging, snapshot_id)
    query = _query_from_task(task, cutoff=cutoff)
    runs = []
    count: int | None = None
    note: str | None = None
    try:
        for attempt in range(1, 4):
            run = connector.run_page(
                query,
                writer,
                channel=RetrievalChannel.TEXT,
                search_text=query.search_texts[0],
                per_page=1,
                attempt=attempt,
            )
            runs.append(run)
            if run.status is ConnectorStatus.SUCCESS:
                if run.artifact is None:
                    raise ValueError("successful OpenAlex count run has no artifact")
                raw = json.loads(writer.read_response(run.artifact))
                meta = raw.get("meta") if isinstance(raw, dict) else None
                value = meta.get("count") if isinstance(meta, dict) else None
                if type(value) is not int or value < 0:
                    note = "invalid meta.count in successful OpenAlex response"
                else:
                    count = value
                break
            delay = _retry_delay(run, attempt)
            if delay is None:
                note = run.error.code if run.error is not None else "connector_failed"
                break
            sleeper(delay)
        created_at = runs[-1].finished_at if runs else datetime.now(UTC)
        writer.finalize(
            SnapshotManifest(
                snapshot_id=snapshot_id,
                snapshot_version=TEMPORAL_COUNT_EXECUTOR_VERSION,
                analysis_id=count_id,
                created_at=created_at,
                query=query,
                runs=tuple(runs),
            )
        )
        result = {
            "schema_version": TEMPORAL_COUNT_RESULT_VERSION,
            "task": task,
            "spec_digest": _spec_digest(task),
            "status": "complete" if count is not None else "unknown",
            "count": count,
            "attempts": len(runs),
            "snapshot_id": snapshot_id,
            "note": note,
            "reused": False,
        }
        result_bytes = _canonical_bytes(result, indent=2)
        (staging / "result.json").write_bytes(result_bytes)
        snapshot_bytes = (staging / "snapshot" / "manifest.json").read_bytes()
        cache = {
            "schema_version": TEMPORAL_COUNT_CACHE_VERSION,
            "count_id": count_id,
            "spec_digest": _spec_digest(task),
            "result": _digest(result_bytes),
            "snapshot_manifest": _digest(snapshot_bytes),
        }
        (staging / "cache_manifest.json").write_bytes(_canonical_bytes(cache, indent=2))
        if count is not None:
            publish_staging(staging, final)
        else:
            failure_root = work_dir / "failures" / count_id
            failure_root.mkdir(parents=True, exist_ok=True)
            cycles = [path for path in failure_root.iterdir() if path.name.startswith("cycle-")]
            failure = failure_root / f"cycle-{len(cycles) + 1:03d}"
            publish_staging(staging, failure)
        return result
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _candidate_features(
    plan: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    task_index = {
        (task["entity_type"], task["entity_id"], task["window"]): task for task in plan["tasks"]
    }
    rows: list[dict[str, Any]] = []
    for candidate in plan["candidates"]:
        candidate_id = candidate["candidate_id"]
        scope = candidate["analysis_scope_key"]
        counts: dict[str, int | None] = {}
        count_ids: dict[str, str] = {}
        for entity_type, entity_id, prefix in (
            ("candidate", candidate_id, "candidate"),
            ("scope", scope, "scope"),
        ):
            for window in ("previous", "recent"):
                task = task_index[(entity_type, entity_id, window)]
                result = results.get(task["count_id"])
                key = f"{prefix}_{window}"
                counts[key] = (
                    result.get("count")
                    if isinstance(result, Mapping) and result.get("status") == "complete"
                    else None
                )
                count_ids[key] = task["count_id"]
        complete = all(type(value) is int for value in counts.values())
        # Candidate counts intentionally use the candidate phrase without the
        # domain expression. They are therefore not a subset of scope counts,
        # so a ratio between the two populations would be misleading.
        shares: dict[str, float | None] = {"previous": None, "recent": None}
        growth = None
        share_delta = None
        if complete:
            previous = counts["candidate_previous"]
            recent = counts["candidate_recent"]
            if isinstance(previous, int) and isinstance(recent, int):
                growth = math.log1p(recent) - math.log1p(previous)
            if shares["previous"] is not None and shares["recent"] is not None:
                share_delta = shares["recent"] - shares["previous"]
        rows.append(
            {
                "schema_version": TEMPORAL_COUNT_RESULT_VERSION,
                **candidate,
                "coverage": "complete" if complete else "unknown",
                "counts": counts,
                "count_ids": count_ids,
                "candidate_log_growth": growth,
                "scope_share_previous": shares["previous"],
                "scope_share_recent": shares["recent"],
                "scope_share_delta": share_delta,
            }
        )
    return rows


def run_temporal_counts(
    *,
    plan_dir: str | Path,
    work_dir: str | Path,
    output_dir: str | Path,
    environment: Mapping[str, str] | None = None,
    max_new_tasks: int | None = None,
    concurrency: int = 1,
    connector: OpenAlexConnector | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Path]:
    if max_new_tasks is not None and max_new_tasks < 0:
        raise ValueError("max_new_tasks must be non-negative")
    if not 1 <= concurrency <= 16:
        raise ValueError("concurrency must be between 1 and 16")
    plan, tasks, plan_digest = _load_plan(Path(plan_dir))
    cutoff = date.fromisoformat(plan["cutoff_date"])
    output = Path(output_dir)
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    work = Path(work_dir)
    _prepare_work(work, plan_digest, len(tasks), cutoff)
    env = environment or {}
    active_connector = connector or OpenAlexConnector(
        contact_email=env.get("NEXTWAVE_OPENALEX_MAILTO"),
        api_key=env.get("NEXTWAVE_OPENALEX_API_KEY"),
    )
    results: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    new_tasks = 0
    for task in tasks:
        completed = work / "completed" / task["count_id"]
        if completed.exists():
            results.append(
                _execute_task(
                    task,
                    work_dir=work,
                    connector=active_connector,
                    sleeper=sleeper,
                    cutoff=cutoff,
                )
            )
            continue
        if max_new_tasks is not None and new_tasks >= max_new_tasks:
            results.append(
                {
                    "schema_version": TEMPORAL_COUNT_RESULT_VERSION,
                    "task": task,
                    "spec_digest": _spec_digest(task),
                    "status": "not_run",
                    "count": None,
                    "attempts": 0,
                    "snapshot_id": None,
                    "note": "max_new_tasks",
                    "reused": False,
                }
            )
            continue
        pending.append(task)
        new_tasks += 1
    with ThreadPoolExecutor(max_workers=min(concurrency, max(1, len(pending)))) as pool:
        futures = {
            pool.submit(
                _execute_task,
                task,
                work_dir=work,
                connector=active_connector,
                sleeper=sleeper,
                cutoff=cutoff,
            ): task
            for task in pending
        }
        for future in as_completed(futures):
            results.append(future.result())
    result_index = {row["task"]["count_id"]: row for row in results}
    results = [result_index[task["count_id"]] for task in tasks]
    features = _candidate_features(plan, result_index)
    output_results = [
        {key: value for key, value in row.items() if key != "reused"} for row in results
    ]
    result_bytes = b"".join(_canonical_bytes(row) for row in output_results)
    feature_bytes = b"".join(_canonical_bytes(row) for row in features)
    statuses: dict[str, int] = {}
    for row in results:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    manifest_value = {
        "schema_version": TEMPORAL_COUNT_RESULT_VERSION,
        "executor_version": TEMPORAL_COUNT_EXECUTOR_VERSION,
        "plan_sha256": plan_digest,
        "cutoff_date": cutoff.isoformat(),
        "status": "complete" if statuses.get("complete") == len(tasks) else "partial",
        "counts": {
            "planned_tasks": len(tasks),
            "new_tasks": new_tasks,
            "reused": sum(bool(row.get("reused")) for row in results),
            "candidate_features": len(features),
            **statuses,
        },
        "outputs": {
            COUNT_RESULTS_FILENAME: _digest(result_bytes),
            CANDIDATE_FEATURES_FILENAME: _digest(feature_bytes),
        },
    }
    manifest_bytes = _canonical_bytes(manifest_value, indent=2)
    return publish_artifact_bundle(
        {
            COUNT_RESULTS_FILENAME: result_bytes,
            CANDIDATE_FEATURES_FILENAME: feature_bytes,
            MANIFEST_FILENAME: manifest_bytes,
        },
        output,
    )
