"""Resumable news page fetcher for the media shortlist (offline-first design).

Reads page targets from a relevance ``media_fetch_queue.jsonl``, resolves the
original ``SourceDocument`` rows from the enrichment result, and fetches each
page with the shared ``NewsDocumentEnricher``. No Media Cloud API, no OpenAlex,
no LLM: only the publisher page URLs from the queue are ever requested.

A fetched title or meta description is retrieval data, never evidence: only
``article_text`` sets ``evidence_text_available``. A failed page stays
``failed/unknown`` and never becomes a silent zero.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

from nextwave.contracts import SourceDocument, SourceType, TrustTier
from nextwave.datasets.artifacts import publish_artifact_bundle
from nextwave.sources import (
    NEWS_CONTENT_ENRICHER_VERSION,
    HttpTransport,
    NewsContentStatus,
    NewsDocumentEnricher,
    NewsDocumentEnrichment,
    NewsEnrichmentIssueCode,
    publish_staging,
)

from .enrichment_run import ENRICHMENT_RESULT_VERSION
from .relevance_plan import LABELING_RELEVANCE_PLAN_VERSION, document_identity

LABELING_MEDIA_FETCH_EXECUTOR_VERSION = "labeling-media-fetch-executor-v1"
LABELING_MEDIA_FETCH_WORK_VERSION = "labeling-media-fetch-work-v1"
LABELING_MEDIA_FETCH_RESULT_VERSION = "labeling-media-fetch-result-v1"
LABELING_MEDIA_FETCH_CACHE_VERSION = "labeling-media-fetch-cache-v1"

MEDIA_FETCH_MAX_ATTEMPTS = 3
MEDIA_FETCH_BACKOFF_SECONDS = (2.0, 5.0)
MEDIA_FETCH_CONCURRENCY = 6
MEDIA_FETCH_TIMEOUT_SECONDS = 10.0

WORK_MANIFEST_FILENAME = "work_manifest.json"
CACHE_MANIFEST_FILENAME = "cache_manifest.json"
PAGE_RESULTS_FILENAME = "page_results.jsonl"
ENRICHED_FILENAME = "enriched_documents.jsonl"
COVERAGE_FILENAME = "coverage.jsonl"
RESULT_MANIFEST_FILENAME = "manifest.json"

_QUEUE_FILES = (
    "media_fetch_queue.jsonl",
    "ranked_documents.jsonl",
    "scientific_shortlist.jsonl",
    "coverage.jsonl",
)
_RESULT_FILES = ("coverage.jsonl", "documents.jsonl", "request_results.jsonl")

_TERMINAL_ISSUES = frozenset({
    NewsEnrichmentIssueCode.BLOCKED_URL,
    NewsEnrichmentIssueCode.RESPONSE_TOO_LARGE,
    NewsEnrichmentIssueCode.UNSUPPORTED_CONTENT_TYPE,
    NewsEnrichmentIssueCode.NO_USABLE_TEXT,
})


@dataclass(frozen=True, slots=True)
class LabelingMediaFetchRunPaths:
    """Paths of one published media fetch result."""

    manifest: Path
    page_results: Path
    enriched_documents: Path
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


def _require_plain_count(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative int")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {label}: {error}") from error
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


def _load_relevance(
    relevance_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    manifest_path = relevance_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"relevance artifact is incomplete: {relevance_dir}")
    manifest = _read_json(manifest_path, "relevance manifest")
    if not isinstance(manifest, dict):
        raise ValueError("relevance manifest must be a JSON object")
    if manifest.get("schema_version") != LABELING_RELEVANCE_PLAN_VERSION:
        raise ValueError(
            f"relevance manifest {manifest.get('schema_version')!r} "
            f"does not match {LABELING_RELEVANCE_PLAN_VERSION!r}"
        )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("relevance manifest outputs must be an object")
    digests: dict[str, Any] = {}
    queue_rows: list[dict[str, Any]] = []
    for filename in _QUEUE_FILES:
        payload = _check_digest(relevance_dir / filename, outputs.get(filename), filename)
        digests[filename] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        if filename == "media_fetch_queue.jsonl":
            for lineno, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"media_fetch_queue.jsonl line {lineno} is not valid JSON"
                    ) from error
                if not isinstance(item, dict):
                    raise ValueError(
                        f"media_fetch_queue.jsonl line {lineno} must be an object"
                    )
                queue_rows.append(item)
    manifest_bytes = manifest_path.read_bytes()
    digests["manifest.json"] = {
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "size_bytes": len(manifest_bytes),
    }
    return manifest, queue_rows, digests


def _load_result(
    result_dir: Path,
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    manifest_path = result_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"enrichment result is incomplete: {result_dir}")
    manifest = _read_json(manifest_path, "enrichment result manifest")
    if not isinstance(manifest, dict):
        raise ValueError("enrichment result manifest must be a JSON object")
    if manifest.get("schema_version") != ENRICHMENT_RESULT_VERSION:
        raise ValueError(
            f"enrichment result manifest {manifest.get('schema_version')!r} "
            f"does not match {ENRICHMENT_RESULT_VERSION!r}"
        )
    bundle_id = manifest.get("bundle_id")
    if not isinstance(bundle_id, str) or not bundle_id:
        raise ValueError("enrichment result bundle_id must be a non-empty string")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("enrichment result manifest outputs must be an object")
    digests: dict[str, Any] = {}
    documents: dict[tuple[str, str], dict[str, Any]] = {}
    first_seen: dict[tuple[str, str], int] = {}
    for filename in _RESULT_FILES:
        payload = _check_digest(result_dir / filename, outputs.get(filename), filename)
        digests[filename] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        if filename == "documents.jsonl":
            for lineno, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"documents.jsonl line {lineno} is not valid JSON"
                    ) from error
                if not isinstance(item, dict):
                    raise ValueError(
                        f"documents.jsonl line {lineno} must be an object"
                    )
                key = (item.get("candidate_id"), item.get("document_id"))
                if not isinstance(key[0], str) or not isinstance(key[1], str):
                    raise ValueError(
                        f"documents.jsonl line {lineno} needs candidate_id/document_id"
                    )
                if key in documents:
                    raise ValueError(
                        f"documents.jsonl line {lineno} duplicates "
                        f"candidate/document {key!r} from line {first_seen[key]}"
                    )
                first_seen[key] = lineno
                documents[key] = item
    if not documents:
        raise ValueError("enrichment documents.jsonl holds no documents")
    manifest_bytes = manifest_path.read_bytes()
    digests["manifest.json"] = {
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "size_bytes": len(manifest_bytes),
    }
    return manifest, documents, digests


def _source_document_from_row(row: Mapping[str, Any], label: str) -> SourceDocument:
    """Rebuild the original SourceDocument through one explicit validator."""

    def text(name: str, *, required: bool = True) -> str | None:
        value = row.get(name)
        if value is None and not required:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label}: {name} must be a non-blank string")
        return value

    def optional_text(name: str) -> str | None:
        value = row.get(name)
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label}: {name} must be a non-blank string or null")
        return value

    def flag(name: str) -> bool:
        value = row.get(name)
        if not isinstance(value, bool):
            raise ValueError(f"{label}: {name} must be a boolean")
        return value

    def names(name: str) -> tuple[str, ...]:
        value = row.get(name, ())
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"{label}: {name} must be a list of strings")
        return tuple(value)

    def iso_date(name: str) -> date | None:
        value = row.get(name)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"{label}: {name} must be an ISO date string or null")
        try:
            return date.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"{label}: {name} is not an ISO date") from error

    def iso_datetime(name: str) -> datetime | None:
        value = row.get(name)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"{label}: {name} must be an ISO datetime string or null")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"{label}: {name} is not an ISO datetime") from error
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError(f"{label}: {name} must include a timezone")
        return parsed

    try:
        source_type = SourceType(row.get("source_type"))
    except ValueError as error:
        raise ValueError(f"{label}: unknown source_type {row.get('source_type')!r}") from error
    try:
        trust_tier = TrustTier(row.get("trust_tier"))
    except ValueError as error:
        raise ValueError(f"{label}: unknown trust_tier {row.get('trust_tier')!r}") from error
    confidence = row.get("origin_confidence")
    if confidence is not None and (
        not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
    ):
        raise ValueError(f"{label}: origin_confidence must be a number or null")
    return SourceDocument(
        document_id=text("document_id"),
        connector_id=text("connector_id"),
        external_id=text("external_id"),
        snapshot_id=text("snapshot_id"),
        title=text("title"),
        url=text("url"),
        canonical_url=text("canonical_url"),
        source_type=source_type,
        language=text("language"),
        trust_tier=trust_tier,
        origin_id=text("origin_id"),
        doi=optional_text("doi"),
        authors=names("authors"),
        organizations=names("organizations"),
        published_at=iso_date("published_at"),
        observed_at=iso_datetime("observed_at"),
        retrieved_at=iso_datetime("retrieved_at"),
        publisher=optional_text("publisher"),
        excerpt=optional_text("excerpt"),
        automatic_translation=flag("automatic_translation"),
        generated_summary=flag("generated_summary"),
        origin_method=optional_text("origin_method"),
        origin_confidence=confidence,
    )


def _validate_queue(
    queue_rows: list[dict[str, Any]],
    documents: dict[tuple[str, str], dict[str, Any]],
    expected_count: Any,
) -> list[dict[str, Any]]:
    """Check every queue row against its enrichment source document."""

    if _require_plain_count(expected_count, "relevance manifest media_fetch_queue_rows") != len(
        queue_rows
    ):
        raise ValueError("media fetch queue row count does not match manifest totals")
    seen: set[tuple[str, str]] = set()
    ranks_by_candidate: dict[str, list[int]] = {}
    for lineno, row in enumerate(queue_rows, start=1):
        label = f"media_fetch_queue.jsonl line {lineno}"
        candidate_id = row.get("candidate_id")
        document_id = row.get("document_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(f"{label}: candidate_id must be a non-empty string")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError(f"{label}: document_id must be a non-empty string")
        pair = (candidate_id, document_id)
        if pair in seen:
            raise ValueError(f"{label}: duplicate queue pair {pair!r}")
        seen.add(pair)
        rank = row.get("selection_rank")
        if not isinstance(rank, int) or isinstance(rank, bool) or not 1 <= rank <= 4:
            raise ValueError(f"{label}: selection_rank must be an int from 1 to 4")
        ranks_by_candidate.setdefault(candidate_id, []).append(rank)
        if row.get("connector") != "mediacloud":
            raise ValueError(f"{label}: queue connector must be mediacloud")
        url = row.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ValueError(f"{label}: queue URL must be a non-empty string")
        if row.get("relevance_class") not in ("strong", "weak", "none"):
            raise ValueError(f"{label}: unknown relevance_class")
        source = documents.get(pair)
        if source is None:
            raise ValueError(f"{label}: unknown candidate/document {pair!r}")
        for name in ("search_id", "request_id"):
            value = row.get(name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{label}: {name} must be a non-empty string")
            if value != source.get(name):
                raise ValueError(f"{label}: {name} does not match the source document")
        if source.get("connector") != "mediacloud":
            raise ValueError(f"{label}: source document is not a mediacloud row")
        if source.get("url") != url:
            raise ValueError(f"{label}: queue URL does not match the source document")
        if document_identity(source) != row.get("document_identity"):
            raise ValueError(f"{label}: document identity does not match the source")
    for candidate_id in sorted(ranks_by_candidate):
        ranks = sorted(ranks_by_candidate[candidate_id])
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError(
                f"selection_rank for {candidate_id!r} must form 1..N "
                "without repeats or gaps"
            )
    return sorted(queue_rows, key=lambda row: (row["candidate_id"], row["selection_rank"]))


def _fetch_id(
    relevance_manifest_digest: str,
    candidate_id: str,
    document_id: str,
    url: str,
    enricher_version: str,
    timeout_seconds: float,
) -> str:
    identity = "|".join((
        LABELING_MEDIA_FETCH_EXECUTOR_VERSION,
        relevance_manifest_digest,
        candidate_id,
        document_id,
        url,
        enricher_version,
        repr(timeout_seconds),
    ))
    return "fetch-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _fetch_spec(
    queue_row: Mapping[str, Any],
    source: Mapping[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    """Fixed-field spec: unknown queue fields never affect the digest."""

    return {
        "executor_version": LABELING_MEDIA_FETCH_EXECUTOR_VERSION,
        "enricher_version": NEWS_CONTENT_ENRICHER_VERSION,
        "timeout_seconds": timeout_seconds,
        "max_attempts": MEDIA_FETCH_MAX_ATTEMPTS,
        "candidate_id": queue_row["candidate_id"],
        "document_id": queue_row["document_id"],
        "search_id": queue_row["search_id"],
        "request_id": queue_row["request_id"],
        "url": queue_row["url"],
        "origin_id": source["origin_id"],
        "document_identity": queue_row["document_identity"],
        "selection_rank": queue_row["selection_rank"],
    }


def _spec_digest(spec: Mapping[str, Any]) -> str:
    canonical = json.dumps(spec, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _work_fingerprint(
    relevance_digests: Mapping[str, Any],
    result_digests: Mapping[str, Any],
    bundle_id: str,
    timeout_seconds: float,
    planned_fetch_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": LABELING_MEDIA_FETCH_WORK_VERSION,
        "relevance_manifest": dict(relevance_digests["manifest.json"]),
        "result_manifest": dict(result_digests["manifest.json"]),
        "bundle_id": bundle_id,
        "executor_version": LABELING_MEDIA_FETCH_EXECUTOR_VERSION,
        "enricher_version": NEWS_CONTENT_ENRICHER_VERSION,
        "timeout_seconds": timeout_seconds,
        "max_attempts": MEDIA_FETCH_MAX_ATTEMPTS,
        "backoff_seconds": list(MEDIA_FETCH_BACKOFF_SECONDS),
        "planned_fetch_count": planned_fetch_count,
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
        raise ValueError(f"media fetch work store is corrupt, manifest missing: {work}")
    stored = _read_json(manifest_path, "media fetch work manifest")
    if not isinstance(stored, dict):
        raise ValueError(f"media fetch work manifest must be an object: {work}")
    if stored.get("schema_version") != LABELING_MEDIA_FETCH_WORK_VERSION:
        raise ValueError(
            f"media fetch work schema {stored.get('schema_version')!r} "
            f"does not match {LABELING_MEDIA_FETCH_WORK_VERSION!r}"
        )
    if stored != dict(fingerprint):
        raise ValueError(
            "media fetch work store belongs to different inputs; "
            "use a fresh work directory instead of reusing it silently"
        )
    for service in ("completed", "failures"):
        if not (work / service).is_dir():
            raise ValueError(f"media fetch work service directory is missing: {service}")
    return work


def _is_retryable(enrichment: NewsDocumentEnrichment) -> bool:
    if enrichment.issue_code is None:
        return False
    if enrichment.issue_code is NewsEnrichmentIssueCode.NETWORK_ERROR:
        return True
    if enrichment.issue_code is NewsEnrichmentIssueCode.HTTP_ERROR:
        status = enrichment.http_status
        return status == 429 or (status is not None and status >= 500)
    return False


def _attempt_record(attempt: int, enrichment: NewsDocumentEnrichment) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "content_status": enrichment.status.value,
        "issue_code": enrichment.issue_code.value if enrichment.issue_code else None,
        "http_status": enrichment.http_status,
        "fetched_bytes": enrichment.fetched_bytes,
    }


def _safe_fetch(
    enricher: NewsDocumentEnricher, document: SourceDocument
) -> NewsDocumentEnrichment:
    """Call the enricher without letting transport exception text leak out."""

    try:
        return enricher.enrich(document)
    except Exception as error:
        return NewsDocumentEnrichment(
            document=document,
            status=NewsContentStatus.TITLE_ONLY,
            issue_code=NewsEnrichmentIssueCode.NETWORK_ERROR,
            message=f"{type(error).__name__}: news page fetch failed",
            content_sha256=None,
            fetched_bytes=0,
            excerpt_source=None,
            excerpt_truncated=False,
            http_status=None,
        )


def _fetch_with_retries(
    enricher: NewsDocumentEnricher,
    document: SourceDocument,
    sleeper: Callable[[float], None],
) -> tuple[NewsDocumentEnrichment, list[dict[str, Any]], bool]:
    """Fetch one page; return ``(enrichment, attempts, terminal)``."""

    attempts: list[dict[str, Any]] = []
    enrichment = _safe_fetch(enricher, document)
    for attempt in range(1, MEDIA_FETCH_MAX_ATTEMPTS + 1):
        if attempt > 1:
            enrichment = _safe_fetch(enricher, document)
        attempts.append(_attempt_record(attempt, enrichment))
        if not _is_retryable(enrichment):
            return enrichment, attempts, True
        if attempt < MEDIA_FETCH_MAX_ATTEMPTS:
            sleeper(MEDIA_FETCH_BACKOFF_SECONDS[attempt - 1])
    return enrichment, attempts, False


def _render_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return (
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ).encode("utf-8")
    )


def _digest(payload: bytes) -> dict[str, Any]:
    return {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _publish_completed(
    work: Path,
    fetch_id: str,
    result: Mapping[str, Any],
    spec_digest: str,
) -> None:
    completed_dir = work / "completed" / fetch_id
    if completed_dir.exists():
        raise FileExistsError(
            f"completed fetch already exists and must never be replaced: {completed_dir}"
        )
    result_bytes = (
        json.dumps(dict(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    cache_bytes = (
        json.dumps(
            {
                "schema_version": LABELING_MEDIA_FETCH_CACHE_VERSION,
                "fetch_id": fetch_id,
                "candidate_id": result["candidate_id"],
                "document_id": result["document_id"],
                "spec_digest": spec_digest,
                "result": _digest(result_bytes),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    staging = Path(tempfile.mkdtemp(prefix=f".{fetch_id}-", dir=work))
    try:
        (staging / "result.json").write_bytes(result_bytes)
        (staging / CACHE_MANIFEST_FILENAME).write_bytes(cache_bytes)
        publish_staging(staging, completed_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _next_cycle(work: Path, fetch_id: str) -> tuple[Path, str]:
    failures_dir = work / "failures" / fetch_id
    existing = sorted(
        path.name for path in failures_dir.glob("cycle-*") if path.is_dir()
    ) if failures_dir.is_dir() else []
    cycle = f"cycle-{len(existing) + 1:03d}"
    return failures_dir / cycle, cycle


def _publish_failure(
    work: Path, fetch_id: str, result: Mapping[str, Any]
) -> str:
    cycle_dir, cycle = _next_cycle(work, fetch_id)
    if cycle_dir.exists():
        raise FileExistsError(f"failure cycle already exists: {cycle_dir}")
    result_bytes = (
        json.dumps(dict(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    staging = Path(tempfile.mkdtemp(prefix=f".{fetch_id}-", dir=work))
    try:
        (staging / "result.json").write_bytes(result_bytes)
        cycle_dir.parent.mkdir(parents=True, exist_ok=True)
        publish_staging(staging, cycle_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return cycle


def _completed_record(
    *,
    fetch_id: str,
    queue_row: Mapping[str, Any],
    spec_digest: str,
    enrichment: NewsDocumentEnrichment,
    attempts: list[dict[str, Any]],
    reused: bool,
) -> dict[str, Any]:
    return {
        "fetch_id": fetch_id,
        "candidate_id": queue_row["candidate_id"],
        "document_id": queue_row["document_id"],
        "selection_rank": queue_row["selection_rank"],
        "url": queue_row["url"],
        "status": "success",
        "content_status": enrichment.status.value,
        "issue_code": enrichment.issue_code.value if enrichment.issue_code else None,
        "http_status": enrichment.http_status,
        "attempts_count": len(attempts),
        "reused": reused,
        "content_sha256": enrichment.content_sha256,
        "fetched_bytes": enrichment.fetched_bytes,
        "excerpt_source": enrichment.excerpt_source,
        "excerpt_truncated": enrichment.excerpt_truncated,
        "retryable_exhausted": False,
        "note": (
            "reused completed cache"
            if reused
            else f"{enrichment.status.value} terminal after {len(attempts)} attempt(s)"
        ),
    }


def _failed_record(
    *,
    fetch_id: str,
    queue_row: Mapping[str, Any],
    enrichment: NewsDocumentEnrichment,
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    code = enrichment.issue_code.value if enrichment.issue_code else "unknown_error"
    return {
        "fetch_id": fetch_id,
        "candidate_id": queue_row["candidate_id"],
        "document_id": queue_row["document_id"],
        "selection_rank": queue_row["selection_rank"],
        "url": queue_row["url"],
        "status": "failed",
        "content_status": enrichment.status.value,
        "issue_code": code,
        "http_status": enrichment.http_status,
        "attempts_count": len(attempts),
        "reused": False,
        "content_sha256": enrichment.content_sha256,
        "fetched_bytes": enrichment.fetched_bytes,
        "excerpt_source": enrichment.excerpt_source,
        "excerpt_truncated": enrichment.excerpt_truncated,
        "retryable_exhausted": True,
        "note": f"{code} retryable after {len(attempts)} attempts",
    }


def _check_no_secret_keys(value: Any, fetch_id: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            folded = str(key).casefold()
            if "header" in folded or "cookie" in folded or "authorization" in folded:
                raise ValueError(
                    f"completed cache for {fetch_id!r} stores forbidden HTTP data"
                )
            _check_no_secret_keys(item, fetch_id)
    elif isinstance(value, list):
        for item in value:
            _check_no_secret_keys(item, fetch_id)


def _require_cache_http_status(value: Any, label: str, fetch_id: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or not 100 <= value <= 599:
        raise ValueError(f"completed cache {label} is invalid for {fetch_id!r}")
    return value


def _require_cache_bool(value: Any, label: str, fetch_id: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"completed cache {label} must be a boolean for {fetch_id!r}")
    return value


def _require_cache_optional_str(value: Any, label: str, fetch_id: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"completed cache {label} must be a string or null for {fetch_id!r}")
    return value


def _require_cache_bytes(value: Any, label: str, fetch_id: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"completed cache {label} is invalid for {fetch_id!r}")
    return value


def _check_completed_cache(
    completed_dir: Path,
    fetch_id: str,
    spec: Mapping[str, Any],
    document: SourceDocument,
) -> dict[str, Any]:
    """Validate one cached page without trusting it blindly."""

    cache_path = completed_dir / CACHE_MANIFEST_FILENAME
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"completed cache for {fetch_id!r} has no readable cache manifest: {error}"
        ) from error
    if not isinstance(cache, dict):
        raise ValueError(f"completed cache manifest for {fetch_id!r} must be an object")
    if cache.get("schema_version") != LABELING_MEDIA_FETCH_CACHE_VERSION:
        raise ValueError(
            f"completed cache schema {cache.get('schema_version')!r} "
            f"does not match {LABELING_MEDIA_FETCH_CACHE_VERSION!r}"
        )
    for field in ("candidate_id", "document_id"):
        if cache.get(field) != spec.get(field):
            raise ValueError(f"completed cache {field} mismatch for {fetch_id!r}")
    if cache.get("fetch_id") != fetch_id:
        raise ValueError(f"completed cache fetch_id mismatch for {fetch_id!r}")
    if cache.get("spec_digest") != _spec_digest(spec):
        raise ValueError(f"completed cache spec_digest mismatch for {fetch_id!r}")
    result_path = completed_dir / "result.json"
    _check_file_digest(result_path, cache.get("result") or {}, fetch_id)
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"completed cache result for {fetch_id!r} is not valid JSON: {error}"
        ) from error
    if not isinstance(result, dict):
        raise ValueError(f"completed cache result for {fetch_id!r} must be an object")
    if result.get("schema_version") != LABELING_MEDIA_FETCH_RESULT_VERSION:
        raise ValueError(f"completed cache result schema mismatch for {fetch_id!r}")
    for field in ("candidate_id", "document_id", "url", "origin_id"):
        if result.get(field) != spec.get(field):
            raise ValueError(
                f"completed cache {field} does not match the planned fetch {fetch_id!r}"
            )
    if result.get("spec_digest") != _spec_digest(spec):
        raise ValueError(f"completed cache result spec mismatch for {fetch_id!r}")
    if result.get("status") != "success":
        raise ValueError(
            f"completed cache status must be success for {fetch_id!r}"
        )
    if result.get("retryable_exhausted", False) is not False:
        raise ValueError(
            f"completed cache retryable_exhausted must be false for {fetch_id!r}"
        )
    try:
        content_status = NewsContentStatus(result.get("content_status"))
    except ValueError as error:
        raise ValueError(
            f"completed cache content status is unknown for {fetch_id!r}"
        ) from error
    issue_code = result.get("issue_code")
    if issue_code is not None:
        try:
            issue = NewsEnrichmentIssueCode(issue_code)
        except ValueError as error:
            raise ValueError(
                f"completed cache issue code is unknown for {fetch_id!r}"
            ) from error
    else:
        issue = None
    http_status = _require_cache_http_status(
        result.get("http_status"), "http_status", fetch_id
    )
    _require_cache_bytes(result.get("fetched_bytes"), "fetched_bytes", fetch_id)
    _require_cache_bool(result.get("excerpt_truncated"), "excerpt_truncated", fetch_id)
    evidence_available = _require_cache_bool(
        result.get("evidence_text_available"), "evidence_text_available", fetch_id
    )
    excerpt_source = _require_cache_optional_str(
        result.get("excerpt_source"), "excerpt_source", fetch_id
    )
    _require_cache_optional_str(result.get("message"), "message", fetch_id)
    if evidence_available != (content_status is NewsContentStatus.ARTICLE_TEXT):
        raise ValueError(
            f"completed cache evidence flag disagrees with content status for {fetch_id!r}"
        )
    enrichment = result.get("enrichment")
    if not isinstance(enrichment, dict):
        raise ValueError(f"completed cache enrichment must be an object for {fetch_id!r}")
    for top_name, nested_name in (
        ("content_status", "status"),
        ("issue_code", "issue_code"),
        ("http_status", "http_status"),
        ("content_sha256", "content_sha256"),
        ("fetched_bytes", "fetched_bytes"),
        ("excerpt_source", "excerpt_source"),
        ("excerpt_truncated", "excerpt_truncated"),
    ):
        if result.get(top_name) != enrichment.get(nested_name):
            raise ValueError(
                f"completed cache {top_name} disagrees with enrichment for {fetch_id!r}"
            )
    nested_document = enrichment.get("document")
    if not isinstance(nested_document, dict):
        raise ValueError(
            f"completed cache nested document must be an object for {fetch_id!r}"
        )
    if "excerpt" not in nested_document:
        raise ValueError(
            f"completed cache nested document has no excerpt field for {fetch_id!r}"
        )
    for field, expected in (
        ("document_id", document.document_id),
        ("connector_id", document.connector_id),
        ("external_id", document.external_id),
        ("snapshot_id", document.snapshot_id),
        ("url", document.url),
        ("canonical_url", document.canonical_url),
        ("origin_id", document.origin_id),
        ("title", document.title),
    ):
        if nested_document.get(field) != expected:
            raise ValueError(
                f"completed cache nested document {field} mismatch for {fetch_id!r}"
            )
    excerpt = nested_document.get("excerpt")
    if content_status is NewsContentStatus.ARTICLE_TEXT:
        if http_status != 200 or issue is not None:
            raise ValueError(
                f"completed cache article_text must be a clean HTTP 200 for {fetch_id!r}"
            )
        if not isinstance(excerpt, str) or not excerpt.strip():
            raise ValueError(
                f"completed cache article_text has no excerpt for {fetch_id!r}"
            )
        if not excerpt_source:
            raise ValueError(
                f"completed cache article_text has no excerpt source for {fetch_id!r}"
            )
    elif content_status is NewsContentStatus.META_DESCRIPTION:
        if http_status != 200 or issue is not None:
            raise ValueError(
                f"completed cache meta_description must be a clean HTTP 200 for {fetch_id!r}"
            )
        if not isinstance(excerpt, str) or not excerpt.strip():
            raise ValueError(
                f"completed cache meta_description has no excerpt for {fetch_id!r}"
            )
    elif content_status is NewsContentStatus.TITLE_ONLY:
        if excerpt is not None:
            raise ValueError(
                f"completed cache title_only must not carry an excerpt for {fetch_id!r}"
            )
        if issue is NewsEnrichmentIssueCode.HTTP_ERROR and http_status is None:
            raise ValueError(
                f"completed cache terminal HTTP error has no status for {fetch_id!r}"
            )
    elif content_status is NewsContentStatus.EXISTING_EXCERPT:
        if http_status is not None or issue is not None:
            raise ValueError(
                f"completed cache existing_excerpt must not carry HTTP data for {fetch_id!r}"
            )
        if excerpt != document.excerpt:
            raise ValueError(
                f"completed cache existing_excerpt differs from the source for {fetch_id!r}"
            )
    if issue is NewsEnrichmentIssueCode.NETWORK_ERROR:
        raise ValueError(
            f"completed cache holds a retryable network error for {fetch_id!r}"
        )
    if (
        issue is NewsEnrichmentIssueCode.HTTP_ERROR
        and http_status is not None
        and (http_status == 429 or http_status >= 500)
    ):
        raise ValueError(
            f"completed cache holds a retryable HTTP error for {fetch_id!r}"
        )
    sha = enrichment.get("content_sha256")
    if sha is not None and (
        not isinstance(sha, str) or len(sha) != 64 or sha != sha.lower()
    ):
        raise ValueError(f"completed cache content_sha256 is invalid for {fetch_id!r}")
    try:
        int(sha, 16) if sha is not None else None
    except ValueError as error:
        raise ValueError(f"completed cache content_sha256 is invalid for {fetch_id!r}") from error
    attempts = result.get("attempts")
    if (
        not isinstance(attempts, list)
        or not 1 <= len(attempts) <= MEDIA_FETCH_MAX_ATTEMPTS
    ):
        raise ValueError(
            f"completed cache attempts must list 1..{MEDIA_FETCH_MAX_ATTEMPTS} "
            f"entries for {fetch_id!r}"
        )
    for position, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, dict):
            raise ValueError(
                f"completed cache attempt must be an object for {fetch_id!r}"
            )
        number = attempt.get("attempt")
        if not isinstance(number, int) or isinstance(number, bool) or number != position:
            raise ValueError(
                f"completed cache attempts must run 1..N in order for {fetch_id!r}"
            )
        try:
            NewsContentStatus(attempt.get("content_status"))
        except ValueError as error:
            raise ValueError(
                f"completed cache attempt status is unknown for {fetch_id!r}"
            ) from error
        attempt_issue = attempt.get("issue_code")
        if attempt_issue is not None:
            try:
                NewsEnrichmentIssueCode(attempt_issue)
            except ValueError as error:
                raise ValueError(
                    f"completed cache attempt issue is unknown for {fetch_id!r}"
                ) from error
        _require_cache_http_status(
            attempt.get("http_status"), "attempt http_status", fetch_id
        )
        _require_cache_bytes(attempt.get("fetched_bytes"), "attempt fetched_bytes", fetch_id)
    _check_no_secret_keys(result, fetch_id)
    return result


def _check_file_digest(path: Path, entry: Any, fetch_id: str) -> bytes:
    if not isinstance(entry, dict):
        raise ValueError(f"completed cache digest entry is invalid for {fetch_id!r}")
    size = entry.get("size_bytes")
    sha = entry.get("sha256")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError(f"completed cache size is invalid for {fetch_id!r}")
    if not isinstance(sha, str) or len(sha) != 64:
        raise ValueError(f"completed cache sha256 is invalid for {fetch_id!r}")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(
            f"completed cache file is missing for {fetch_id!r}: {error}"
        ) from error
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != sha:
        raise ValueError(f"completed cache checksum mismatch for {fetch_id!r}")
    return payload


def _store_completed_result(
    work: Path,
    fetch_id: str,
    queue_row: Mapping[str, Any],
    spec: Mapping[str, Any],
    spec_digest: str,
    enrichment: NewsDocumentEnrichment,
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {
        "schema_version": LABELING_MEDIA_FETCH_RESULT_VERSION,
        "fetch_id": fetch_id,
        "candidate_id": queue_row["candidate_id"],
        "document_id": queue_row["document_id"],
        "search_id": queue_row["search_id"],
        "request_id": queue_row["request_id"],
        "selection_rank": queue_row["selection_rank"],
        "url": queue_row["url"],
        "origin_id": spec["origin_id"],
        "spec_digest": spec_digest,
        "status": "success",
        "content_status": enrichment.status.value,
        "issue_code": enrichment.issue_code.value if enrichment.issue_code else None,
        "http_status": enrichment.http_status,
        "attempts": attempts,
        "fetched_bytes": enrichment.fetched_bytes,
        "content_sha256": enrichment.content_sha256,
        "excerpt_source": enrichment.excerpt_source,
        "excerpt_truncated": enrichment.excerpt_truncated,
        "evidence_text_available": enrichment.evidence_text_available,
        "enrichment": enrichment.to_dict(),
        "note": f"{enrichment.status.value} terminal after {len(attempts)} attempt(s)",
    }
    _publish_completed(work, fetch_id, result, spec_digest)
    return result


def _store_failure_result(
    work: Path,
    fetch_id: str,
    queue_row: Mapping[str, Any],
    spec_digest: str,
    enrichment: NewsDocumentEnrichment,
    attempts: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    code = enrichment.issue_code.value if enrichment.issue_code else "unknown_error"
    result = {
        "schema_version": LABELING_MEDIA_FETCH_RESULT_VERSION,
        "fetch_id": fetch_id,
        "candidate_id": queue_row["candidate_id"],
        "document_id": queue_row["document_id"],
        "selection_rank": queue_row["selection_rank"],
        "url": queue_row["url"],
        "spec_digest": spec_digest,
        "status": "failed",
        "content_status": enrichment.status.value,
        "issue_code": code,
        "http_status": enrichment.http_status,
        "attempts": attempts,
        "retryable_exhausted": True,
        "note": f"{code} retryable after {len(attempts)} attempts",
    }
    cycle = _publish_failure(work, fetch_id, result)
    return result, cycle


def _fetch_and_publish(
    work: Path,
    enricher: NewsDocumentEnricher,
    sleeper: Callable[[float], None],
    fetch_id: str,
    queue_row: Mapping[str, Any],
    document: SourceDocument,
    spec: Mapping[str, Any],
    spec_digest: str,
) -> dict[str, Any]:
    """Fetch one page with retries and publish the outcome; return a page record."""

    enrichment, attempts, terminal = _fetch_with_retries(enricher, document, sleeper)
    if terminal:
        _store_completed_result(
            work, fetch_id, queue_row, spec, spec_digest, enrichment, attempts
        )
        return _completed_record(
            fetch_id=fetch_id,
            queue_row=queue_row,
            spec_digest=spec_digest,
            enrichment=enrichment,
            attempts=attempts,
            reused=False,
        )
    _store_failure_result(work, fetch_id, queue_row, spec_digest, enrichment, attempts)
    return _failed_record(
        fetch_id=fetch_id,
        queue_row=queue_row,
        enrichment=enrichment,
        attempts=attempts,
    )


def _rebuild_enrichment(
    document: SourceDocument, stored: Mapping[str, Any]
) -> NewsDocumentEnrichment:
    """Rebuild an enrichment from its validated serialized form.

    Every value below was already type-checked by ``_check_completed_cache``;
    nothing is coerced or defaulted here.
    """

    nested = stored["document"]
    excerpt = nested["excerpt"] if isinstance(nested, dict) else None
    issue = stored["issue_code"]
    return NewsDocumentEnrichment(
        document=(
            document if excerpt is None else replace(document, excerpt=excerpt)
        ),
        status=NewsContentStatus(stored["status"]),
        issue_code=NewsEnrichmentIssueCode(issue) if issue else None,
        message=stored["message"],
        content_sha256=stored["content_sha256"],
        fetched_bytes=stored["fetched_bytes"],
        excerpt_source=stored["excerpt_source"],
        excerpt_truncated=stored["excerpt_truncated"],
        http_status=stored["http_status"],
    )


def _enriched_row(
    queue_row: Mapping[str, Any],
    source: Mapping[str, Any],
    enrichment: NewsDocumentEnrichment,
) -> dict[str, Any]:
    published = source.get("published_at")
    return {
        "candidate_id": queue_row["candidate_id"],
        "document_id": queue_row["document_id"],
        "connector": queue_row["connector"],
        "search_id": queue_row["search_id"],
        "request_id": queue_row["request_id"],
        "selection_rank": queue_row["selection_rank"],
        "url": queue_row["url"],
        "canonical_url": source.get("canonical_url"),
        "origin_id": source.get("origin_id"),
        "origin_method": source.get("origin_method"),
        "title": source.get("title"),
        "publisher": source.get("publisher"),
        "published_at": published,
        "source_type": source.get("source_type"),
        "trust_tier": source.get("trust_tier"),
        "snapshot_id": source.get("snapshot_id"),
        "excerpt": enrichment.document.excerpt,
        "content_status": enrichment.status.value,
        "evidence_text_available": enrichment.evidence_text_available,
        "content_sha256": enrichment.content_sha256,
        "extraction_method": enrichment.excerpt_source,
        "excerpt_truncated": enrichment.excerpt_truncated,
        "http_status": enrichment.http_status,
        "fetched_bytes": enrichment.fetched_bytes,
    }


def run_media_fetch(
    *,
    relevance_dir: str | Path,
    result_dir: str | Path,
    work_dir: str | Path,
    output_dir: str | Path,
    page_transport: HttpTransport | None = None,
    page_enricher: NewsDocumentEnricher | None = None,
    timeout_seconds: float = MEDIA_FETCH_TIMEOUT_SECONDS,
    sleeper: Callable[[float], None] | None = None,
) -> LabelingMediaFetchRunPaths:
    """Fetch every queued news page once, reusing verified work, and publish."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    sleep = sleeper or time.sleep
    relevance_path = Path(relevance_dir)
    enrichment_path = Path(result_dir)
    relevance_manifest, queue_rows, relevance_digests = _load_relevance(relevance_path)
    result_manifest, documents, result_digests = _load_result(enrichment_path)
    if relevance_manifest.get("bundle_id") != result_manifest.get("bundle_id"):
        raise ValueError("relevance bundle_id does not match enrichment result bundle_id")
    bundle_id = result_manifest["bundle_id"]
    expected_result_manifest = relevance_manifest.get("inputs", {}).get("result_manifest")
    if (
        not isinstance(expected_result_manifest, dict)
        or expected_result_manifest.get("sha256")
        != result_digests["manifest.json"]["sha256"]
        or expected_result_manifest.get("size_bytes")
        != result_digests["manifest.json"]["size_bytes"]
    ):
        raise ValueError("relevance artifact does not reference this exact result manifest")
    expected_result_files = relevance_manifest.get("inputs", {}).get("result_files")
    if not isinstance(expected_result_files, dict):
        raise ValueError("relevance manifest inputs.result_files must be an object")
    for filename in _RESULT_FILES:
        expected = expected_result_files.get(filename)
        actual = result_digests[filename]
        if (
            not isinstance(expected, dict)
            or expected.get("sha256") != actual["sha256"]
            or expected.get("size_bytes") != actual["size_bytes"]
        ):
            raise ValueError(
                f"relevance artifact does not reference this exact {filename}"
            )
    totals = relevance_manifest.get("totals")
    if not isinstance(totals, dict):
        raise ValueError("relevance manifest totals must be an object")
    planned = _validate_queue(queue_rows, documents, totals.get("media_fetch_queue_rows"))

    # Full preflight before work, output, or network: rebuild every source
    # document, derive every spec and fetch ID, and prove IDs are unique.
    relevance_digest = relevance_digests["manifest.json"]["sha256"]
    tasks: list[dict[str, Any]] = []
    for row in planned:
        source = documents[(row["candidate_id"], row["document_id"])]
        document = _source_document_from_row(
            source, f"enrichment document {row['candidate_id']}/{row['document_id']}"
        )
        if document.connector_id != "mediacloud":
            raise ValueError(
                f"source document connector must be mediacloud for {document.document_id!r}"
            )
        spec = _fetch_spec(row, source, timeout_seconds)
        spec_digest = _spec_digest(spec)
        fetch_id = _fetch_id(
            relevance_digest,
            row["candidate_id"],
            row["document_id"],
            row["url"],
            NEWS_CONTENT_ENRICHER_VERSION,
            timeout_seconds,
        )
        tasks.append({
            "fetch_id": fetch_id,
            "queue_row": row,
            "document": document,
            "source": source,
            "spec": spec,
            "spec_digest": spec_digest,
        })
    seen_ids: set[str] = set()
    for task in tasks:
        if task["fetch_id"] in seen_ids:
            raise ValueError(
                f"duplicate fetch_id {task['fetch_id']!r} "
                f"for {task['queue_row']['candidate_id']!r}/"
                f"{task['queue_row']['document_id']!r}"
            )
        seen_ids.add(task["fetch_id"])

    output = Path(output_dir)
    if output.exists():
        raise ValueError(f"output directory already exists: {output}")
    fingerprint = _work_fingerprint(
        relevance_digests, result_digests, bundle_id, timeout_seconds, len(planned)
    )
    work = _init_or_check_work(work_dir, fingerprint)

    enricher = (
        page_enricher
        or NewsDocumentEnricher(transport=page_transport, timeout_seconds=timeout_seconds)
    )

    page_records: list[dict[str, Any]] = []
    enriched_rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for task in tasks:
        completed_dir = work / "completed" / task["fetch_id"]
        if completed_dir.exists():
            result = _check_completed_cache(
                completed_dir, task["fetch_id"], task["spec"], task["document"]
            )
            enrichment = _rebuild_enrichment(task["document"], result["enrichment"])
            page_records.append(_completed_record(
                fetch_id=task["fetch_id"],
                queue_row=task["queue_row"],
                spec_digest=task["spec_digest"],
                enrichment=enrichment,
                attempts=result.get("attempts") or [],
                reused=True,
            ))
            enriched_rows.append(_enriched_row(task["queue_row"], task["source"], enrichment))
        else:
            pending.append(task)

    if pending:
        with ThreadPoolExecutor(max_workers=MEDIA_FETCH_CONCURRENCY) as pool:
            futures = {
                pool.submit(
                    _fetch_and_publish,
                    work,
                    enricher,
                    sleep,
                    task["fetch_id"],
                    task["queue_row"],
                    task["document"],
                    task["spec"],
                    task["spec_digest"],
                ): task
                for task in pending
            }
            for future in as_completed(futures):
                task = futures[future]
                record = future.result()
                page_records.append(record)
                if record["status"] == "success":
                    completed_dir = work / "completed" / task["fetch_id"]
                    stored = _check_completed_cache(
                        completed_dir, task["fetch_id"], task["spec"], task["document"]
                    )
                    enrichment = _rebuild_enrichment(
                        task["document"], stored["enrichment"]
                    )
                    enriched_rows.append(
                        _enriched_row(task["queue_row"], task["source"], enrichment)
                    )

    page_records.sort(
        key=lambda row: (row["candidate_id"], row["selection_rank"], row["document_id"])
    )
    enriched_rows.sort(
        key=lambda row: (row["candidate_id"], row["selection_rank"], row["document_id"])
    )
    coverage_rows: list[dict[str, Any]] = []
    for candidate_id in sorted({row["candidate_id"] for row in planned}):
        candidate_records = [
            row for row in page_records if row["candidate_id"] == candidate_id
        ]
        completed = [row for row in candidate_records if row["status"] == "success"]
        failed = [row for row in candidate_records if row["status"] != "success"]
        enriched = [
            row for row in enriched_rows if row["candidate_id"] == candidate_id
        ]
        by_status = {
            status: sum(1 for row in enriched if row["content_status"] == status)
            for status in (
                "article_text",
                "meta_description",
                "title_only",
                "existing_excerpt",
            )
        }
        if len(completed) == len(candidate_records):
            status = "complete"
        elif completed:
            status = "partial"
        else:
            status = "unknown"
        coverage_rows.append({
            "candidate_id": candidate_id,
            "planned": len(candidate_records),
            "completed": len(completed),
            "failed": len(failed),
            "reused": sum(1 for row in completed if row["reused"]),
            "article_text": by_status["article_text"],
            "meta_description": by_status["meta_description"],
            "title_only": by_status["title_only"],
            "existing_excerpt": by_status["existing_excerpt"],
            "evidence_text_available": sum(
                1 for row in enriched if row["content_status"] == "article_text"
            ),
            "failed_fetch_ids": sorted(row["fetch_id"] for row in failed),
            "status": status,
        })

    totals_out = {
        "planned_fetches": len(page_records),
        "completed": sum(1 for row in page_records if row["status"] == "success"),
        "failed": sum(1 for row in page_records if row["status"] != "success"),
        "reused": sum(1 for row in page_records if row["reused"]),
        "article_text": sum(
            1 for row in enriched_rows if row["content_status"] == "article_text"
        ),
        "meta_description": sum(
            1 for row in enriched_rows if row["content_status"] == "meta_description"
        ),
        "title_only": sum(
            1 for row in enriched_rows if row["content_status"] == "title_only"
        ),
        "existing_excerpt": sum(
            1 for row in enriched_rows if row["content_status"] == "existing_excerpt"
        ),
        "evidence_text_available": sum(
            1 for row in enriched_rows if row["content_status"] == "article_text"
        ),
        "candidates": len(coverage_rows),
    }
    files = {
        PAGE_RESULTS_FILENAME: _render_jsonl(page_records),
        ENRICHED_FILENAME: _render_jsonl(enriched_rows),
        COVERAGE_FILENAME: _render_jsonl(coverage_rows),
    }
    manifest = {
        "schema_version": LABELING_MEDIA_FETCH_RESULT_VERSION,
        "bundle_id": bundle_id,
        "executor_version": LABELING_MEDIA_FETCH_EXECUTOR_VERSION,
        "enricher_version": NEWS_CONTENT_ENRICHER_VERSION,
        "work_version": LABELING_MEDIA_FETCH_WORK_VERSION,
        "cache_version": LABELING_MEDIA_FETCH_CACHE_VERSION,
        "inputs": {
            "relevance_manifest": dict(relevance_digests["manifest.json"]),
            "relevance_files": {
                name: relevance_digests[name] for name in _QUEUE_FILES
            },
            "result_manifest": dict(result_digests["manifest.json"]),
            "result_files": {name: result_digests[name] for name in _RESULT_FILES},
        },
        "policy": {
            "timeout_seconds": timeout_seconds,
            "max_concurrency": MEDIA_FETCH_CONCURRENCY,
            "max_attempts": MEDIA_FETCH_MAX_ATTEMPTS,
            "backoff_seconds": list(MEDIA_FETCH_BACKOFF_SECONDS),
        },
        "totals": totals_out,
        "outputs": {
            name: {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for name, payload in files.items()
        },
    }
    files[RESULT_MANIFEST_FILENAME] = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    paths = publish_artifact_bundle(files, output)
    return LabelingMediaFetchRunPaths(
        manifest=paths[RESULT_MANIFEST_FILENAME],
        page_results=paths[PAGE_RESULTS_FILENAME],
        enriched_documents=paths[ENRICHED_FILENAME],
        coverage=paths[COVERAGE_FILENAME],
    )
