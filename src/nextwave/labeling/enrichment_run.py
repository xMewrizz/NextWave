"""Resumable offline executor for labeling-enrichment plans.

Reads an immutable ``labeling-enrichment-plan`` directory, executes every
planned OpenAlex/Media Cloud request exactly once, and publishes an
immutable result. Successful requests are cached in a permanent ``work``
directory verified against the plan, so a rerun only repeats what is still
open. No secrets ever enter results, manifests, snapshots or messages.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any

from nextwave.contracts import SourceDocument
from nextwave.sources import (
    ConnectorRun,
    ConnectorStatus,
    HttpTransport,
    MediaCloudConnector,
    OpenAlexConnector,
    QueryPurpose,
    RetrievalChannel,
    SnapshotManifest,
    SnapshotWriter,
    SourceQuery,
    build_mediacloud_story_request,
    build_openalex_request,
    normalize_openalex_api_key,
    parse_mediacloud_response,
    parse_openalex_response,
    publish_staging,
)

from ..datasets.artifacts import publish_artifact_bundle
from .contracts import LABELING_CUTOFF_DATE
from .enrichment_plan import (
    COMPLETION_RULE,
    LABELING_ENRICHMENT_PLAN_VERSION,
    LANGUAGES,
    MEDIACLOUD_COLLECTION_IDS,
    MEDIACLOUD_RETRIEVAL_POLICY,
    OPENALEX_RETRIEVAL_POLICY,
    _bundle_id,
    _candidate_searches,
    _clean_term,
    _EnrichmentCandidateRecord,
    _windows,
)

ENRICHMENT_EXECUTOR_VERSION = "labeling-enrichment-executor-v2"
ENRICHMENT_RESULT_VERSION = "labeling-enrichment-result-v2"
ENRICHMENT_WORK_VERSION = "labeling-enrichment-work-v2"
ENRICHMENT_REQUEST_VERSION = "labeling-enrichment-request-v2"
ENRICHMENT_CACHE_VERSION = "labeling-enrichment-cache-v2"

PLAN_FILENAME = "plan.json"
RUN_MANIFEST_FILENAME = "manifest.json"
WORK_MANIFEST_FILENAME = "work_manifest.json"
REQUEST_RESULTS_FILENAME = "request_results.jsonl"
DOCUMENTS_FILENAME = "documents.jsonl"
COVERAGE_FILENAME = "coverage.jsonl"
CACHE_MANIFEST_FILENAME = "cache_manifest.json"

_RETRY_BACKOFF_SECONDS = (5.0, 15.0)
_SAFE_ID = re.compile(r"(request|search)-[0-9a-f]{16}\Z")
_EXPECTED_COVERAGE_POLICY = {
    "successful_response_including_zero": "covered",
    "failed_timeout_429_or_skipped": "unknown",
}
_FORBIDDEN_PLAN_KEYS = {"status", "searched", "search_performed", "results", "returned_records"}


@dataclass(frozen=True, slots=True)
class LabelingEnrichmentRunPaths:
    """Paths of one published enrichment result."""

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


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _walk_keys(payload: Any) -> Any:
    if isinstance(payload, dict):
        for key, value in payload.items():
            yield key
            yield from _walk_keys(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _walk_keys(value)


def _spec_digest(spec: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _request_spec(
    candidate: Mapping[str, Any], search: Mapping[str, Any], request: Mapping[str, Any]
) -> dict[str, Any]:
    window = search.get("planned_window") or {}
    return {
        "request_id": request.get("request_id"),
        "candidate_id": candidate.get("candidate_id"),
        "search_id": search.get("search_id"),
        "search_text": request.get("search_text"),
        "languages": request.get("languages"),
        "connector": search.get("connector"),
        "role": search.get("role"),
        "planned_window": {"from": window.get("from"), "until": window.get("until")},
        "retrieval_policy": search.get("retrieval_policy"),
        "collection_ids": search.get("collection_ids"),
    }


def _expected_coverage_policy(connector: str) -> dict[str, Any]:
    if connector == "mediacloud":
        return {**_EXPECTED_COVERAGE_POLICY, "fallback": None}
    return dict(_EXPECTED_COVERAGE_POLICY)


def _validate_search_task(
    entry: Mapping[str, Any],
    search: Mapping[str, Any],
    record_candidate_id: str,
    terms: list[str],
    full_window: tuple[str, str],
    recent_window: tuple[str, str],
) -> None:
    """Validate one planned search against the scheduler's exact shape."""

    candidate_id = entry.get("candidate_id")
    if not isinstance(search, dict):
        raise ValueError(f"candidate {candidate_id!r} holds a non-object search")
    connector = search.get("connector")
    if connector == "openalex":
        expected_class = "scientific"
        expected_policy = dict(OPENALEX_RETRIEVAL_POLICY)
        expected_collections: list[int] | None = None
        expected_publicity: dict[str, str] | None = None
    elif connector == "mediacloud":
        expected_class, expected_policy = "industry", dict(MEDIACLOUD_RETRIEVAL_POLICY)
        expected_collections = list(MEDIACLOUD_COLLECTION_IDS)
        expected_publicity = {"from": recent_window[0], "until": recent_window[1]}
    else:
        raise ValueError(
            f"unsupported enrichment connector {connector!r}; "
            "only openalex and mediacloud are planned"
        )
    if search.get("source_class") != expected_class or search.get("role") != "primary":
        raise ValueError(
            f"candidate {candidate_id!r} search mixes source_class, connector and role: "
            f"{search.get('source_class')!r}/{connector!r}/{search.get('role')!r}"
        )
    if search.get("purpose") != "candidate_search":
        raise ValueError(
            f"candidate {candidate_id!r} search purpose is not candidate_search"
        )
    if search.get("completion_rule") != COMPLETION_RULE:
        raise ValueError(
            f"candidate {candidate_id!r} completion rule is not {COMPLETION_RULE!r}"
        )
    if search.get("languages") != list(LANGUAGES):
        raise ValueError(
            f"candidate {candidate_id!r} search languages differ "
            f"from the planned {list(LANGUAGES)!r}"
        )
    exact_window = {"from": full_window[0], "until": full_window[1]}
    if search.get("required_window") != exact_window:
        raise ValueError(
            f"candidate {candidate_id!r} required window is not the planned {exact_window!r}"
        )
    if search.get("planned_window") != exact_window:
        raise ValueError(
            f"candidate {candidate_id!r} planned window is not the planned {exact_window!r}"
        )
    if expected_publicity is None:
        if "publicity_window" in search:
            raise ValueError(
                f"candidate {candidate_id!r} scientific search must not carry a publicity window"
            )
    elif search.get("publicity_window") != expected_publicity:
        raise ValueError(
            f"candidate {candidate_id!r} publicity window is not the planned {expected_publicity!r}"
        )
    if search.get("retrieval_policy") != expected_policy:
        raise ValueError(
            f"candidate {candidate_id!r} {connector} retrieval policy "
            "does not match the planned policy"
        )
    if expected_collections is None:
        if "collection_ids" in search:
            raise ValueError(
                f"candidate {candidate_id!r} scientific search must not carry collections"
            )
    elif search.get("collection_ids") != expected_collections:
        raise ValueError(
            f"candidate {candidate_id!r} mediacloud collections "
            f"do not match {expected_collections!r}"
        )
    if search.get("coverage_policy") != _expected_coverage_policy(connector):
        raise ValueError(
            f"candidate {candidate_id!r} {connector} coverage policy "
            "does not match the planned policy"
        )
    requests = search.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError(f"candidate {candidate_id!r} {connector} search holds no requests")
    for request in requests:
        if not isinstance(request, dict):
            raise ValueError(f"candidate {candidate_id!r} holds a non-object request")
        for field in ("request_id", "search_text"):
            value = request.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"candidate {candidate_id!r} request field {field!r} "
                    "must be a non-blank string"
                )
        if "language" in request:
            raise ValueError(
                f"candidate {candidate_id!r} request {request['request_id']!r} "
                "uses the retired singular field 'language'; "
                "v2 requires 'languages'"
            )
        languages = request.get("languages")
        if (
            not isinstance(languages, list)
            or not languages
            or any(not isinstance(item, str) or not item.strip() for item in languages)
        ):
            raise ValueError(
                f"candidate {candidate_id!r} request languages "
                "must be a non-empty list of non-blank strings"
            )
        if len(set(languages)) != len(languages):
            raise ValueError(
                f"candidate {candidate_id!r} request languages must be unique"
            )
        if any(item not in LANGUAGES for item in languages):
            raise ValueError(
                f"candidate {candidate_id!r} request languages "
                f"{languages!r} use an unsupported language"
            )
        if connector == "openalex":
            if languages != ["en"] and languages != ["ru"]:
                raise ValueError(
                    f"candidate {candidate_id!r} openalex request must carry "
                    'exactly ["en"] or ["ru"]'
                )
        elif languages != ["en", "ru"]:
            raise ValueError(
                f"candidate {candidate_id!r} mediacloud request must carry "
                'exactly ["en", "ru"] in order'
            )
        if _SAFE_ID.fullmatch(request["request_id"]) is None:
            raise ValueError(
                f"candidate {candidate_id!r} request id {request['request_id']!r} "
                "has an unsafe format"
            )
    if _SAFE_ID.fullmatch(search.get("search_id", "")) is None:
        raise ValueError(
            f"candidate {candidate_id!r} search id {search.get('search_id')!r} "
            "has an unsafe format"
        )
    expected_searches = _candidate_searches(
        record_candidate_id, terms, full_window, recent_window
    )
    expected = next(
        item for item in expected_searches if item["connector"] == connector
    )
    if search["search_id"] != expected["search_id"]:
        raise ValueError(
            f"candidate {candidate_id!r} {connector} search id "
            "does not match the scheduler recomputation"
        )
    actual_ids = [
        (item["request_id"], item["search_text"], tuple(item["languages"]))
        for item in requests
    ]
    recomputed_ids = [
        (item["request_id"], item["search_text"], tuple(item["languages"]))
        for item in expected["requests"]
    ]
    if actual_ids != recomputed_ids:
        raise ValueError(
            f"candidate {candidate_id!r} {connector} requests "
            "do not match the scheduler recomputation"
        )
    window = search["planned_window"]
    for request in requests:
        try:
            query = _build_query(entry, request, window)
            if connector == "openalex":
                build_openalex_request(
                    query,
                    channel=RetrievalChannel.TEXT,
                    search_text=request["search_text"],
                    page_index=1,
                    per_page=search["retrieval_policy"]["per_page"],
                    attempt=1,
                )
            else:
                build_mediacloud_story_request(
                    query,
                    search_text=request["search_text"],
                    collection_ids=tuple(search["collection_ids"]),
                    page_size=search["retrieval_policy"]["page_size"],
                    page_index=1,
                    attempt=1,
                    languages=tuple(request["languages"]),
                )
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"candidate {candidate_id!r} request {request['request_id']!r} "
                f"cannot be built with connector contracts: {error}"
            ) from error


def load_validated_plan(plan_dir: str | Path) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    """Load and strictly validate an enrichment plan before any request.

    Returns the plan, its raw bytes and digest. Raises before anything is
    created elsewhere, so failures never leave partial state behind.
    """

    target = Path(plan_dir)
    plan_path = target / PLAN_FILENAME
    manifest_path = target / RUN_MANIFEST_FILENAME
    if not plan_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"enrichment plan is incomplete: {target}")
    plan_bytes = plan_path.read_bytes()
    try:
        plan = json.loads(plan_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"enrichment plan is not valid JSON: {error}") from error
    if not isinstance(plan, dict):
        raise ValueError("enrichment plan must be a JSON object")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"enrichment plan manifest is not valid JSON: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError("enrichment plan manifest must be a JSON object")
    if manifest.get("schema_version") != LABELING_ENRICHMENT_PLAN_VERSION:
        raise ValueError(
            "enrichment plan manifest "
            f"{manifest.get('schema_version')!r} does not match "
            f"{LABELING_ENRICHMENT_PLAN_VERSION!r}"
        )
    expected = (manifest.get("outputs") or {}).get(PLAN_FILENAME) or {}
    if len(plan_bytes) != expected.get("size_bytes"):
        raise ValueError("enrichment plan size mismatch with manifest")
    if hashlib.sha256(plan_bytes).hexdigest() != expected.get("sha256"):
        raise ValueError("enrichment plan checksum mismatch with manifest")
    if plan.get("schema_version") != LABELING_ENRICHMENT_PLAN_VERSION:
        raise ValueError(
            "enrichment plan "
            f"{plan.get('schema_version')!r} does not match "
            f"{LABELING_ENRICHMENT_PLAN_VERSION!r}"
        )
    bundle_section = plan.get("bundle")
    if not isinstance(bundle_section, dict):
        raise ValueError("enrichment plan bundle must be an object")
    if manifest.get("bundle_id") != bundle_section.get("bundle_id"):
        raise ValueError("enrichment plan bundle_id mismatch with manifest")
    if plan.get("cutoff_date") != LABELING_CUTOFF_DATE.isoformat():
        raise ValueError(
            f"enrichment plan cutoff {plan.get('cutoff_date')!r} "
            f"does not match {LABELING_CUTOFF_DATE.isoformat()!r}"
        )
    if _FORBIDDEN_PLAN_KEYS & set(_walk_keys(plan)):
        raise ValueError("enrichment plan claims an executed search")

    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("enrichment plan holds no candidates")
    if any(not isinstance(entry, dict) for entry in candidates):
        raise ValueError("enrichment plan holds a non-object candidate")
    if bundle_section.get("candidate_count") != len(candidates):
        raise ValueError(
            "enrichment plan bundle candidate count "
            f"{bundle_section.get('candidate_count')!r} "
            f"does not match {len(candidates)} candidates"
        )
    candidate_ids = [entry.get("candidate_id") for entry in candidates]
    if any(not isinstance(value, str) or not value for value in candidate_ids):
        raise ValueError("enrichment plan candidate_id values must be non-blank strings")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("duplicate candidate_id in enrichment plan")

    records = []
    for entry in candidates:
        try:
            cutoff_date = date.fromisoformat(str(entry.get("cutoff_date", "")))
        except ValueError as error:
            raise ValueError(
                f"candidate {entry.get('candidate_id')!r} cutoff_date is not a date"
            ) from error
        origin = entry.get("origin")
        if not isinstance(origin, dict):
            origin = {}
        try:
            records.append(
                _EnrichmentCandidateRecord(
                    candidate_id=entry.get("candidate_id", ""),
                    canonical_name=entry.get("canonical_name", ""),
                    aliases=tuple(entry.get("aliases") or ()),
                    group_id=origin.get("group_id", ""),
                    source_query=origin.get("source_query", ""),
                    domain=entry.get("domain", ""),
                    analysis_scope_key=entry.get("analysis_scope_key", ""),
                    cutoff_date=cutoff_date,
                )
            )
        except (ValueError, TypeError) as error:
            raise ValueError(
                f"candidate {entry.get('candidate_id')!r} is invalid: {error}"
            ) from error
    bundle_id = bundle_section.get("bundle_id")
    recomputed = _bundle_id(records)
    if recomputed != bundle_id:
        raise ValueError("enrichment plan bundle_id does not match its candidates")

    full_window = (_windows()["history_from"], _windows()["cutoff_date"])
    recent_window = (_windows()["recent_window_from"], _windows()["cutoff_date"])
    search_ids: list[str] = []
    request_ids: list[str] = []
    totals = {"openalex": 0, "mediacloud": 0}
    for entry, record in zip(candidates, records, strict=True):
        origin = entry.get("origin")
        if not isinstance(origin, dict) or origin.get("bundle_id") != bundle_id:
            raise ValueError(
                f"candidate {entry.get('candidate_id')!r} origin bundle_id "
                "does not match the plan bundle"
            )
        searches = entry.get("searches")
        if (
            not isinstance(searches, list)
            or len(searches) != 2
            or any(not isinstance(search, dict) for search in searches)
        ):
            raise ValueError(
                f"candidate {entry.get('candidate_id')!r} must hold exactly two "
                "search objects"
            )
        if {search.get("connector") for search in searches} != {
            "openalex",
            "mediacloud",
        }:
            raise ValueError(
                f"candidate {entry.get('candidate_id')!r} must hold exactly one "
                "openalex and one mediacloud search"
            )
        raw_terms = entry.get("search_terms")
        if (
            not isinstance(raw_terms, list)
            or not raw_terms
            or any(not isinstance(term, str) for term in raw_terms)
        ):
            raise ValueError(
                f"candidate {record.candidate_id!r} search_terms must be "
                "a non-empty list of strings"
            )
        terms = [_clean_term(term) for term in raw_terms]
        if (
            any(not term for term in terms)
            or terms != raw_terms
            or len({term.casefold() for term in terms}) != len(terms)
        ):
            raise ValueError(
                f"candidate {record.candidate_id!r} search_terms are not normalized"
            )
        for search in searches:
            _validate_search_task(
                entry, search, record.candidate_id, terms, full_window, recent_window
            )
            search_ids.append(search["search_id"])
            request_ids.extend(request["request_id"] for request in search["requests"])
            totals[search["connector"]] += len(search["requests"])
    if len(set(search_ids)) != len(search_ids):
        raise ValueError("duplicate search_id in enrichment plan")
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("duplicate request_id in enrichment plan")
    declared = plan.get("totals") or {}
    if declared.get("candidates") != len(candidates):
        raise ValueError("enrichment plan totals do not match its candidates")
    if declared.get("openalex_primary_requests") != totals["openalex"]:
        raise ValueError("enrichment plan openalex totals do not match its requests")
    if declared.get("mediacloud_primary_requests") != totals["mediacloud"]:
        raise ValueError("enrichment plan mediacloud totals do not match its requests")
    return plan, plan_bytes, _digest(plan_bytes)


def _init_or_check_work(
    work_dir: str | Path, plan_digest: Mapping[str, Any], bundle_id: str
) -> Path:
    """Create the work store atomically or verify it belongs to this plan.

    A new store is assembled in a sibling staging directory and published
    whole, so a crash cannot leave a half-written work area. An existing
    store with a valid manifest but missing empty service directories is
    safely restored; an invalid manifest is rejected loudly, never fixed
    silently. Orphan staging directories are ignored by every later step.
    """

    work = Path(work_dir)
    manifest_bytes = (
        json.dumps(
            {
                "schema_version": ENRICHMENT_WORK_VERSION,
                "plan": dict(plan_digest),
                "bundle_id": bundle_id,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if not work.exists():
        work.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{work.name}-", dir=work.parent)
        )
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
        raise ValueError(f"enrichment work store is corrupt, manifest missing: {work}")
    try:
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"enrichment work manifest is not valid JSON: {error}") from error
    if not isinstance(stored, dict):
        raise ValueError(f"enrichment work manifest must be an object: {work}")
    if stored.get("schema_version") != ENRICHMENT_WORK_VERSION:
        raise ValueError(
            f"enrichment work schema {stored.get('schema_version')!r} "
            f"does not match {ENRICHMENT_WORK_VERSION!r}"
        )
    if stored.get("plan") != dict(plan_digest) or stored.get("bundle_id") != bundle_id:
        raise ValueError(
            "enrichment work store belongs to a different plan; "
            "use a fresh work directory instead of reusing it silently"
        )
    for service in ("completed", "failures"):
        service_dir = work / service
        if not service_dir.is_dir():
            raise ValueError(
                f"enrichment work service directory is missing: {service_dir}"
            )
    return work


_DOCUMENT_FIELDS = (
    "document_id",
    "title",
    "url",
    "candidate_id",
    "search_id",
    "request_id",
    "connector",
)


def _check_completed_cache(
    completed_dir: Path, spec: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate one cached request without trusting it blindly.

    Order: cache manifest, result.json checksum, snapshot manifest
    checksum, every raw artifact, then the result schema, field agreement
    and stored documents. Any mismatch raises loudly instead of being
    overwritten or silently re-requested.
    """

    request_id = spec["request_id"]
    cache_path = completed_dir / CACHE_MANIFEST_FILENAME
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"completed cache for {request_id!r} has no readable cache manifest: {error}"
        ) from error
    if not isinstance(cache, dict):
        raise ValueError(
            f"completed cache manifest for {request_id!r} must be an object"
        )
    if cache.get("schema_version") != ENRICHMENT_CACHE_VERSION:
        raise ValueError(
            f"completed cache schema {cache.get('schema_version')!r} "
            f"does not match {ENRICHMENT_CACHE_VERSION!r}"
        )
    for field in ("request_id", "candidate_id", "search_id"):
        if cache.get(field) != spec.get(field):
            raise ValueError(
                f"completed cache {field} mismatch for {request_id!r}"
            )
    if cache.get("spec_digest") != _spec_digest(spec):
        raise ValueError(
            f"completed cache spec_digest mismatch for {request_id!r}"
        )
    result_path = completed_dir / "result.json"
    _check_file_digest(
        result_path, cache.get("result") or {}, request_id, "result.json"
    )
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"completed cache result for {request_id!r} is not valid JSON: {error}"
        ) from error
    snapshot_root = completed_dir / "snapshot"
    snapshot_manifest_path = snapshot_root / "manifest.json"
    _check_file_digest(
        snapshot_manifest_path,
        cache.get("snapshot_manifest") or {},
        request_id,
        "snapshot manifest",
    )
    try:
        snapshot_manifest = json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"completed cache snapshot manifest is not valid JSON: {error}"
        ) from error
    if not isinstance(snapshot_manifest, dict):
        raise ValueError(
            f"completed cache snapshot manifest for {request_id!r} must be an object"
        )
    if not isinstance(result, dict):
        raise ValueError(f"completed cache result for {request_id!r} must be an object")
    snapshot_id = result.get("snapshot_id")
    if snapshot_manifest.get("snapshot_id") != snapshot_id:
        raise ValueError(
            f"completed cache snapshot mismatch for {request_id!r}"
        )
    for run in snapshot_manifest.get("runs") or []:
        if not isinstance(run, dict):
            raise ValueError(
                f"completed cache holds a non-object run for {request_id!r}"
            )
        artifact = run.get("artifact")
        if artifact is None:
            continue
        if not isinstance(artifact, dict):
            raise ValueError(
                f"completed cache holds a non-object artifact for {request_id!r}"
            )
        artifact_path = snapshot_root / artifact["uri"]
        try:
            payload = artifact_path.read_bytes()
        except OSError as error:
            raise ValueError(
                f"completed cache artifact is missing for {request_id!r}: "
                f"{artifact['uri']}"
            ) from error
        if len(payload) != artifact["size_bytes"]:
            raise ValueError(
                f"completed cache artifact size mismatch for {request_id!r}"
            )
        if hashlib.sha256(payload).hexdigest() != artifact["sha256"]:
            raise ValueError(
                f"completed cache artifact checksum mismatch for {request_id!r}"
            )
    if result.get("spec_digest") != _spec_digest(spec):
        raise ValueError(
            f"completed cache spec_digest mismatch for {request_id!r}"
        )
    if result.get("schema_version") != ENRICHMENT_REQUEST_VERSION:
        raise ValueError(
            f"completed cache result schema {result.get('schema_version')!r} "
            f"does not match {ENRICHMENT_REQUEST_VERSION!r}"
        )
    for field in (
        "request_id",
        "candidate_id",
        "search_id",
        "connector",
        "role",
        "search_text",
        "languages",
    ):
        if result.get(field) != spec.get(field):
            raise ValueError(
                f"completed cache {field} mismatch for {request_id!r}"
            )
    if result.get("status") != "success":
        raise ValueError(
            f"completed cache for {request_id!r} is not successful"
        )
    if not isinstance(result.get("attempts"), list) or not result.get("attempts"):
        raise ValueError(
            f"completed cache attempts for {request_id!r} must be a non-empty list"
        )
    documents = result.get("documents")
    if not isinstance(documents, list):
        raise ValueError(
            f"completed cache documents for {request_id!r} must be a list"
        )
    parse_issues = result.get("parse_issues")
    if not isinstance(parse_issues, list):
        raise ValueError(
            f"completed cache parse_issues for {request_id!r} must be a list"
        )
    returned_records = result.get("returned_records")
    if not _is_plain_int(returned_records) or returned_records < 0:
        raise ValueError(
            f"completed cache returned_records for {request_id!r} "
            "must be a non-negative int"
        )
    parsed_documents = result.get("parsed_documents")
    if (
        not _is_plain_int(parsed_documents)
        or parsed_documents < 0
        or parsed_documents != len(documents)
    ):
        raise ValueError(
            f"completed cache parsed_documents for {request_id!r} "
            "must match its documents"
        )
    parse_issue_count = result.get("parse_issue_count")
    if (
        not _is_plain_int(parse_issue_count)
        or parse_issue_count < 0
        or parse_issue_count != len(parse_issues)
    ):
        raise ValueError(
            f"completed cache parse_issue_count for {request_id!r} "
            "must match its parse_issues"
        )
    http_status = result.get("http_status")
    if (
        not _is_plain_int(http_status)
        or not 200 <= http_status <= 299
    ):
        raise ValueError(
            f"completed cache http_status for {request_id!r} must be 2xx"
        )
    for issue in parse_issues:
        if not isinstance(issue, dict):
            raise ValueError(
                f"completed cache holds a non-object parse issue for {request_id!r}"
            )
        record_index = issue.get("record_index")
        if not _is_plain_int(record_index) or record_index < 0:
            raise ValueError(
                f"completed cache parse issue record_index for {request_id!r} "
                "must be a non-negative int"
            )
        for field in ("code", "message"):
            value = issue.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"completed cache parse issue {field!r} for {request_id!r} "
                    "must be a non-blank string"
                )
        external_id = issue.get("external_id")
        if external_id is not None and not isinstance(external_id, str):
            raise ValueError(
                f"completed cache parse issue external_id for {request_id!r} "
                "must be a string or null"
            )
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError(
                f"completed cache holds a non-object document for {request_id!r}"
            )
        for field in _DOCUMENT_FIELDS:
            value = document.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"completed cache document field {field!r} "
                    f"is not usable for {request_id!r}"
                )
        if (
            document.get("candidate_id") != spec.get("candidate_id")
            or document.get("search_id") != spec.get("search_id")
            or document.get("request_id") != request_id
            or document.get("connector") != spec.get("connector")
        ):
            raise ValueError(
                f"completed cache document provenance mismatch for {request_id!r}"
            )
        if (
            document.get("snapshot_id") != snapshot_id
            or snapshot_id != snapshot_manifest.get("snapshot_id")
        ):
            raise ValueError(
                f"completed cache document snapshot mismatch for {request_id!r}"
            )
    return result


def _check_file_digest(
    path: Path, expected: Mapping[str, Any], request_id: str, label: str
) -> None:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(
            f"completed cache {label} is missing for {request_id!r}: {error}"
        ) from error
    if len(payload) != expected.get("size_bytes"):
        raise ValueError(
            f"completed cache {label} size mismatch for {request_id!r}"
        )
    if hashlib.sha256(payload).hexdigest() != expected.get("sha256"):
        raise ValueError(
            f"completed cache {label} checksum mismatch for {request_id!r}"
        )


def _document_to_json(document: SourceDocument) -> dict[str, Any]:
    return _json_value(asdict(document))


def _issue_to_json(issue: Any) -> dict[str, Any]:
    """Serialize one parser rejection without raw payloads or secrets."""

    return {
        "record_index": issue.record_index,
        "external_id": issue.external_id,
        "code": issue.code,
        "message": issue.message,
    }


MAX_RETRY_DELAY_SECONDS = 120.0


def _valid_retry_after(error: Any) -> float | None:
    """Provider pause worth honoring, or None for garbage values."""

    retry_after = getattr(error, "retry_after_seconds", None)
    if (
        isinstance(retry_after, (int, float))
        and not isinstance(retry_after, bool)
        and isfinite(retry_after)
        and retry_after >= 0
    ):
        return float(retry_after)
    return None


def _build_query(
    candidate: Mapping[str, Any], request: Mapping[str, Any], window: Mapping[str, Any]
) -> SourceQuery:
    return SourceQuery(
        query_id=f"q-{request['request_id']}",
        analysis_scope_id=str(candidate.get("analysis_scope_key") or "scope-unknown"),
        purpose=QueryPurpose.VERIFICATION,
        raw_query=str(candidate.get("canonical_name") or request["search_text"]),
        normalized_query=str(candidate.get("canonical_name") or request["search_text"]),
        search_texts=(request["search_text"],),
        published_from=date.fromisoformat(str(window.get("from"))),
        published_until=date.fromisoformat(str(window.get("until"))),
        cutoff_date=date.fromisoformat("2026-09-15"),
        languages=tuple(request["languages"]),
    )


def _execute_request(
    *,
    candidate: Mapping[str, Any],
    search: Mapping[str, Any],
    request: Mapping[str, Any],
    window: Mapping[str, Any],
    openalex_connector: OpenAlexConnector,
    mediacloud_connector: MediaCloudConnector,
    staging_parent: Path,
    snapshot_version: str,
    bundle_id: str,
    clock: Callable[[], datetime],
    sleeper: Callable[[float], None],
) -> tuple[str, dict[str, Any], Path]:
    """Run one planned request with retries from a private staging directory.

    Always returns ``(outcome, record, staging)`` where staging holds the
    finalized ``snapshot/`` tree; the caller writes ``result.json`` into
    staging and publishes it atomically. Transport-level and contract-level
    problems of a single request become a normal failed record with its own
    failure cycle — never an assertion; only staging failures raise.
    """

    connector_id = search["connector"]
    policy = search["retrieval_policy"]
    max_attempts = policy.get("max_attempts")
    if not _is_plain_int(max_attempts) or max_attempts < 1:
        raise ValueError("enrichment retrieval policy max_attempts must be positive")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{request['request_id']}-", dir=staging_parent)
    )
    try:
        try:
            query = _build_query(candidate, request, window)
        except (ValueError, TypeError) as error:
            return "failed", {
                "runs": [],
                "documents": [],
                "error": {
                    "code": "invalid_request",
                    "retryable": False,
                    "http_status": None,
                    "retry_after_seconds": None,
                    "retry_deferred": False,
                },
                "note": f"request cannot be built: {error}",
            }, staging
        writer = SnapshotWriter(staging, "snapshot")
        runs: list[ConnectorRun] = []
        deferred: dict[str, Any] = {}
        media_safe_syntax = False
        for attempt in range(1, max_attempts + 1):
            if connector_id == "openalex":
                run = openalex_connector.run_page(
                    query,
                    writer,
                    channel=RetrievalChannel.TEXT,
                    search_text=request["search_text"],
                    page_index=1,
                    per_page=policy.get("per_page", 20),
                    attempt=attempt,
                )
            else:
                run = mediacloud_connector.run_page(
                    query,
                    writer,
                    search_text=request["search_text"],
                    collection_ids=tuple(search.get("collection_ids") or ()),
                    page_size=policy.get("page_size", 20),
                    page_index=1,
                    attempt=attempt,
                    languages=tuple(request["languages"]),
                    safe_syntax=media_safe_syntax,
                )
            runs.append(run)
            if run.status is ConnectorStatus.SUCCESS:
                break
            if (
                connector_id == "mediacloud"
                and not media_safe_syntax
                and run.http_status == 400
                and attempt < max_attempts
            ):
                media_safe_syntax = True
                continue
            if run.error is None or not run.error.retryable:
                break
            if attempt >= max_attempts:
                break
            retry_after = _valid_retry_after(run.error)
            if retry_after is not None and retry_after > MAX_RETRY_DELAY_SECONDS:
                deferred = {
                    "retry_after_seconds": retry_after,
                    "retry_deferred": True,
                }
                break
            sleeper(
                max(
                    _RETRY_BACKOFF_SECONDS[
                        min(attempt - 1, len(_RETRY_BACKOFF_SECONDS) - 1)
                    ],
                    retry_after if retry_after is not None else 0.0,
                )
            )
        manifest = SnapshotManifest(
            snapshot_id="snapshot",
            snapshot_version=snapshot_version,
            analysis_id=bundle_id,
            created_at=clock(),
            query=query,
            runs=tuple(runs),
        )
        last = runs[-1]
        if last.status is not ConnectorStatus.SUCCESS:
            writer.finalize(manifest)
            error = last.error
            return "failed", {
                "runs": [run.to_dict() for run in runs],
                "documents": [],
                "error": {
                    "code": error.code if error else "unknown_error",
                    "retryable": bool(error.retryable) if error else False,
                    "http_status": last.http_status,
                    "retry_after_seconds": (
                        _valid_retry_after(error) if error is not None else None
                    ),
                    "retry_deferred": bool(deferred),
                },
            }, staging
        try:
            payload = writer.read_response(last.artifact)
            if connector_id == "openalex":
                parsed = parse_openalex_response(
                    payload,
                    snapshot_id="snapshot",
                    retrieved_at=last.artifact.retrieved_at,
                    cutoff_date=date.fromisoformat("2026-09-15"),
                )
            else:
                parsed = parse_mediacloud_response(
                    payload,
                    snapshot_id="snapshot",
                    retrieved_at=last.artifact.retrieved_at,
                    cutoff_date=date.fromisoformat("2026-09-15"),
                )
            documents = [_document_to_json(document) for document in parsed.documents]
            issues = [_issue_to_json(issue) for issue in parsed.issues]
        except (ValueError, TypeError, OSError) as error:
            writer.finalize(manifest)
            return "failed", {
                "runs": [run.to_dict() for run in runs],
                "documents": [],
                "error": {
                    "code": "invalid_response",
                    "retryable": False,
                    "http_status": last.http_status,
                    "retry_after_seconds": None,
                    "retry_deferred": False,
                },
                "note": f"response cannot be parsed: {error}",
            }, staging
        writer.finalize(manifest)
        return "success", {
            "runs": [run.to_dict() for run in runs],
            "documents": documents,
            "http_status": last.http_status,
            "returned_records": last.returned_records,
            "parsed_documents": len(documents),
            "parse_issue_count": len(issues),
            "parse_issues": issues,
        }, staging
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run_enrichment(
    *,
    plan_dir: str | Path,
    work_dir: str | Path,
    output_dir: str | Path,
    environment: Mapping[str, str],
    openalex_transport: HttpTransport | None = None,
    mediacloud_transport: HttpTransport | None = None,
    clock: Callable[[], datetime] | None = None,
    monotonic_clock: Callable[[], float] | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> LabelingEnrichmentRunPaths:
    """Execute every planned request once, reusing verified work, and publish."""

    now = clock or (lambda: datetime.now(UTC))
    sleep = sleeper or time.sleep
    plan, _plan_bytes, plan_digest = load_validated_plan(plan_dir)
    bundle_id = plan["bundle"]["bundle_id"]
    output = Path(output_dir)
    if output.exists():
        raise ValueError(f"output directory already exists: {output}")
    work = _init_or_check_work(work_dir, plan_digest, bundle_id)

    api_key = (environment.get("NEXTWAVE_MEDIACLOUD_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("NEXTWAVE_MEDIACLOUD_API_KEY is required to run enrichment")
    contact_email = (environment.get("NEXTWAVE_OPENALEX_MAILTO") or "").strip() or None
    openalex_api_key = normalize_openalex_api_key(
        environment.get("NEXTWAVE_OPENALEX_API_KEY")
    )

    scientific_tasks = [
        search
        for entry in plan["candidates"]
        for search in entry["searches"]
        if search["connector"] == "openalex"
    ]
    industry_tasks = [
        search
        for entry in plan["candidates"]
        for search in entry["searches"]
        if search["connector"] == "mediacloud"
    ]
    openalex_policy = (scientific_tasks[0]["retrieval_policy"] if scientific_tasks else {})
    mediacloud_policy = (industry_tasks[0]["retrieval_policy"] if industry_tasks else {})
    openalex_connector = OpenAlexConnector(
        transport=openalex_transport,
        contact_email=contact_email,
        api_key=openalex_api_key,
        timeout_seconds=float(openalex_policy.get("timeout_seconds", 20)),
        clock=now,
    )
    mediacloud_connector = MediaCloudConnector(
        api_key=api_key,
        transport=mediacloud_transport,
        timeout_seconds=float(mediacloud_policy.get("timeout_seconds", 60)),
        min_interval_seconds=float(mediacloud_policy.get("min_interval_seconds", 30)),
        clock=now,
        monotonic_clock=monotonic_clock,
        sleeper=sleep,
    )

    request_rows: list[dict[str, Any]] = []
    document_rows: list[dict[str, Any]] = []
    for entry in sorted(plan["candidates"], key=lambda item: item["candidate_id"]):
        for search in entry["searches"]:
            window = search["planned_window"]
            snapshot_version = (
                "openalex-discovery-v1"
                if search["connector"] == "openalex"
                else "media-discovery-v1"
            )
            for request in search["requests"]:
                spec = _request_spec(entry, search, request)
                spec_digest = _spec_digest(spec)
                completed_dir = work / "completed" / request["request_id"]
                if completed_dir.exists():
                    cached = _check_completed_cache(completed_dir, spec)
                    reused = True
                    status = "success"
                    attempts = cached.get("attempts") or []
                    documents = cached.get("documents") or []
                    http_status = cached.get("http_status")
                    returned_records = cached.get("returned_records")
                    parsed_documents = cached.get(
                        "parsed_documents", len(documents)
                    )
                    parse_issue_count = cached.get("parse_issue_count", 0)
                    error = None
                    snapshot_id = cached.get("snapshot_id")
                else:
                    outcome, record, staging = _execute_request(
                        candidate=entry,
                        search=search,
                        request=request,
                        window=window,
                        openalex_connector=openalex_connector,
                        mediacloud_connector=mediacloud_connector,
                        staging_parent=work,
                        snapshot_version=snapshot_version,
                        bundle_id=bundle_id,
                        clock=now,
                        sleeper=sleep,
                    )
                    if outcome == "success":
                        provenance_rows = [
                            {
                                **document,
                                "candidate_id": entry["candidate_id"],
                                "search_id": search["search_id"],
                                "request_id": request["request_id"],
                                "connector": search["connector"],
                                "snapshot_id": "snapshot",
                            }
                            for document in record.get("documents") or []
                        ]
                        stored = {
                            "schema_version": ENRICHMENT_REQUEST_VERSION,
                            "request_id": request["request_id"],
                            "candidate_id": entry["candidate_id"],
                            "search_id": search["search_id"],
                            "connector": search["connector"],
                            "role": search.get("role"),
                            "search_text": request["search_text"],
                            "languages": list(request["languages"]),
                            "spec_digest": spec_digest,
                            "snapshot_id": "snapshot",
                            "status": "success",
                            "http_status": record.get("http_status"),
                            "returned_records": record.get("returned_records"),
                            "parsed_documents": record.get("parsed_documents"),
                            "parse_issue_count": record.get("parse_issue_count"),
                            "parse_issues": record.get("parse_issues"),
                            "attempts": record.get("runs"),
                            "documents": provenance_rows,
                        }
                        if completed_dir.exists():
                            shutil.rmtree(staging, ignore_errors=True)
                            raise FileExistsError(
                                "completed request already exists and must "
                                f"never be replaced: {completed_dir}"
                            )
                        result_bytes = (
                            json.dumps(
                                stored,
                                ensure_ascii=False,
                                indent=2,
                                sort_keys=True,
                            )
                            + "\n"
                        ).encode("utf-8")
                        (staging / "result.json").write_bytes(result_bytes)
                        snapshot_manifest_bytes = (
                            staging / "snapshot" / "manifest.json"
                        ).read_bytes()
                        (staging / CACHE_MANIFEST_FILENAME).write_bytes(
                            (
                                json.dumps(
                                    {
                                        "schema_version": ENRICHMENT_CACHE_VERSION,
                                        "request_id": request["request_id"],
                                        "candidate_id": entry["candidate_id"],
                                        "search_id": search["search_id"],
                                        "spec_digest": spec_digest,
                                        "result": _digest(result_bytes),
                                        "snapshot_manifest": _digest(
                                            snapshot_manifest_bytes
                                        ),
                                    },
                                    ensure_ascii=False,
                                    indent=2,
                                    sort_keys=True,
                                )
                                + "\n"
                            ).encode("utf-8")
                        )
                        publish_staging(staging, completed_dir)
                        reused = False
                        status = "success"
                        attempts = record.get("runs")
                        documents = provenance_rows
                        http_status = record.get("http_status")
                        returned_records = record.get("returned_records")
                        parsed_documents = record.get("parsed_documents")
                        parse_issue_count = record.get("parse_issue_count")
                        error = None
                        snapshot_id = "snapshot"
                    else:
                        _publish_failure(work, request["request_id"], record, staging)
                        reused = False
                        status = "failed"
                        attempts = record.get("runs")
                        documents = []
                        http_status = (record.get("error") or {}).get("http_status")
                        returned_records = None
                        parsed_documents = 0
                        parse_issue_count = 0
                        error = record.get("error")
                        snapshot_id = None
                request_rows.append(
                    {
                        "candidate_id": entry["candidate_id"],
                        "search_id": search["search_id"],
                        "request_id": request["request_id"],
                        "connector": search["connector"],
                        "role": search.get("role"),
                        "search_text": request["search_text"],
                        "languages": list(request["languages"]),
                        "status": status,
                        "reused": reused,
                        "attempts": len(attempts or []),
                        "http_status": http_status,
                        "returned_records": returned_records,
                        "returned_documents": parsed_documents
                        if status == "success"
                        else 0,
                        "parse_issue_count": parse_issue_count
                        if status == "success"
                        else 0,
                        "snapshot_id": snapshot_id,
                        "error": error,
                    }
                )
                for document in documents or []:
                    document_rows.append(document)

    coverage_rows: list[dict[str, Any]] = []
    for entry in sorted(plan["candidates"], key=lambda item: item["candidate_id"]):
        for search in entry["searches"]:
            planned = [r["request_id"] for r in search["requests"]]
            rows = [
                row
                for row in request_rows
                if row["search_id"] == search["search_id"]
            ]
            successful = [row for row in rows if row["status"] == "success"]
            failed = [row for row in rows if row["status"] != "success"]
            reasons: list[str] = []
            for row in failed:
                code = (row["error"] or {}).get("code", "unknown_error")
                reasons.append(f"request {row['request_id']} failed ({code})")
            attempted = {row["request_id"] for row in rows}
            for request_id in planned:
                if request_id not in attempted:
                    reasons.append(f"request {request_id} was not attempted")
            if len(successful) == len(planned):
                status = "complete"
            elif successful:
                status = "partial"
            else:
                status = "unknown"
            coverage_rows.append(
                {
                    "candidate_id": entry["candidate_id"],
                    "source_class": search["source_class"],
                    "connector": search["connector"],
                    "search_id": search["search_id"],
                    "planned_requests": len(planned),
                    "successful_requests": len(successful),
                    "failed_requests": len(failed),
                    "reused_requests": sum(1 for row in successful if row["reused"]),
                    "returned_records": sum(
                        row["returned_records"] or 0 for row in successful
                    ),
                    "returned_documents": sum(
                        1
                        for document in document_rows
                        if document["search_id"] == search["search_id"]
                    ),
                    "parse_issue_count": sum(
                        row["parse_issue_count"] or 0 for row in successful
                    ),
                    "failed_request_ids": sorted(row["request_id"] for row in failed),
                    "status": status,
                    "incompleteness_reasons": reasons,
                }
            )

    request_rows.sort(
        key=lambda row: (row["candidate_id"], row["search_id"], row["request_id"])
    )
    document_rows.sort(
        key=lambda row: (
            row["candidate_id"],
            row["search_id"],
            row["request_id"],
            row["document_id"],
        )
    )
    coverage_rows.sort(key=lambda row: (row["candidate_id"], row["source_class"]))

    def render_jsonl(rows: list[dict[str, Any]]) -> bytes:
        return (
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
                for row in rows
            )
        ).encode("utf-8")

    request_bytes = render_jsonl(request_rows)
    documents_bytes = render_jsonl(document_rows)
    coverage_bytes = render_jsonl(coverage_rows)
    totals = {
        "candidates": len(plan["candidates"]),
        "planned_requests": len(request_rows),
        "successful_requests": sum(1 for row in request_rows if row["status"] == "success"),
        "failed_requests": sum(1 for row in request_rows if row["status"] != "success"),
        "reused_requests": sum(1 for row in request_rows if row["reused"]),
        "returned_documents": len(document_rows),
        "coverage_complete": sum(1 for row in coverage_rows if row["status"] == "complete"),
        "coverage_partial": sum(1 for row in coverage_rows if row["status"] == "partial"),
        "coverage_unknown": sum(1 for row in coverage_rows if row["status"] == "unknown"),
    }
    manifest_bytes = (
        json.dumps(
            {
                "schema_version": ENRICHMENT_RESULT_VERSION,
                "executor_version": ENRICHMENT_EXECUTOR_VERSION,
                "plan": _digest((Path(plan_dir) / PLAN_FILENAME).read_bytes()),
                "bundle_id": bundle_id,
                "totals": totals,
                "outputs": {
                    REQUEST_RESULTS_FILENAME: _digest(request_bytes),
                    DOCUMENTS_FILENAME: _digest(documents_bytes),
                    COVERAGE_FILENAME: _digest(coverage_bytes),
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    paths = publish_artifact_bundle(
        {
            REQUEST_RESULTS_FILENAME: request_bytes,
            DOCUMENTS_FILENAME: documents_bytes,
            COVERAGE_FILENAME: coverage_bytes,
            RUN_MANIFEST_FILENAME: manifest_bytes,
        },
        output,
    )
    return LabelingEnrichmentRunPaths(
        manifest=paths[RUN_MANIFEST_FILENAME],
        request_results=paths[REQUEST_RESULTS_FILENAME],
        documents=paths[DOCUMENTS_FILENAME],
        coverage=paths[COVERAGE_FILENAME],
    )


def _publish_failure(
    work: Path, request_id: str, record: Mapping[str, Any], staging: Path
) -> None:
    """Persist one failed request cycle without touching completed results."""

    failures_root = work / "failures" / request_id
    failures_root.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        int(name.removeprefix("cycle-"))
        for name in [path.name for path in failures_root.iterdir() if path.is_dir()]
        if name.startswith("cycle-") and name.removeprefix("cycle-").isdigit()
    )
    cycle = f"cycle-{(existing[-1] + 1) if existing else 1:03d}"
    try:
        (staging / "result.json").write_bytes(
            (
                json.dumps(
                    {
                        "schema_version": ENRICHMENT_REQUEST_VERSION,
                        "request_id": request_id,
                        "status": "failed",
                        "attempts": record.get("runs"),
                        "error": record.get("error"),
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
        )
        publish_staging(staging, failures_root / cycle)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
