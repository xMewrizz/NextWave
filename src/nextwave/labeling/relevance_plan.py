"""Deterministic offline retrieval ranking of enriched labeling documents.

Reads only verified ``search_terms`` from an enrichment plan and document
links from an enrichment result. No API, no LLM, no embeddings. Scores never
use ``target``, expert labels, ``canonical_name``, ``source_query`` or
``domain``; those fields are not even read. A found document is not evidence:
this stage only ranks retrieval so later stages can load a small top-K.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from nextwave.datasets.artifacts import publish_artifact_bundle

from .contracts import LABELING_CUTOFF_DATE
from .enrichment_plan import LABELING_ENRICHMENT_PLAN_VERSION
from .enrichment_run import ENRICHMENT_RESULT_VERSION

LABELING_RELEVANCE_PLAN_VERSION = "labeling-relevance-plan-v1"

RANKED_FILENAME = "ranked_documents.jsonl"
SCIENTIFIC_FILENAME = "scientific_shortlist.jsonl"
MEDIA_QUEUE_FILENAME = "media_fetch_queue.jsonl"
COVERAGE_FILENAME = "coverage.jsonl"
RELEVANCE_MANIFEST_FILENAME = "manifest.json"

CUTOFF_ISO = LABELING_CUTOFF_DATE.isoformat()
PREVIOUS_FROM = "2024-09-15"
PREVIOUS_UNTIL = "2025-09-15"
RECENT_FROM = "2025-09-15"
RECENT_UNTIL = "2026-09-15"

SCIENTIFIC_PER_CANDIDATE = 4
MEDIA_PER_CANDIDATE = 4

CLASS_STRONG = "strong"
CLASS_WEAK = "weak"
CLASS_NONE = "none"

REASON_EXACT_TITLE = "exact_title_phrase"
REASON_TITLE_TOKENS = "all_tokens_in_title"
REASON_EXACT_EXCERPT = "exact_excerpt_phrase"
REASON_COMBINED_TOKENS = "all_tokens_in_combined"
REASON_PARTIAL = "partial_token_overlap"
REASON_NO_OVERLAP = "no_token_overlap"

_TRUST_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3}
_WINDOW_CYCLE = ("previous", "recent", "unknown_date")


@dataclass(frozen=True, slots=True)
class LabelingRelevancePlanPaths:
    """Paths of one published relevance ranking."""

    manifest: Path
    ranked_documents: Path
    scientific_shortlist: Path
    media_fetch_queue: Path
    coverage: Path


def _normalize_text(value: str) -> str:
    """NFKC + casefold + non-alphanumeric runs to one space."""

    folded = unicodedata.normalize("NFKC", value).casefold()
    return " ".join("".join(char if char.isalnum() else " " for char in folded).split())


def _classify(score: int) -> str:
    if score >= 60:
        return CLASS_STRONG
    if score >= 20:
        return CLASS_WEAK
    return CLASS_NONE


def _contains_sequence(field_tokens: list[str], term_tokens: list[str]) -> bool:
    """Contiguous whole-token subsequence match preserving term order."""

    if not term_tokens or len(term_tokens) > len(field_tokens):
        return False
    width = len(term_tokens)
    return any(
        field_tokens[start:start + width] == term_tokens
        for start in range(len(field_tokens) - width + 1)
    )


def _score_term(
    term_sequence: list[str],
    term_set: frozenset[str],
    title_sequence: list[str],
    title_set: frozenset[str],
    excerpt_sequence: list[str] | None,
    excerpt_set: frozenset[str],
    combined_set: frozenset[str],
) -> tuple[int, str]:
    """Score one verified term against title/excerpt following the fixed ladder."""

    if _contains_sequence(title_sequence, term_sequence):
        return 100, REASON_EXACT_TITLE
    if term_set <= title_set:
        return 80, REASON_TITLE_TOKENS
    if excerpt_sequence is not None and _contains_sequence(excerpt_sequence, term_sequence):
        return 70, REASON_EXACT_EXCERPT
    if term_set <= combined_set:
        return 60, REASON_COMBINED_TOKENS
    matched = len(term_set & combined_set)
    if matched == 0:
        return 0, REASON_NO_OVERLAP
    return round(40 * matched / len(term_set)), REASON_PARTIAL


def score_document(
    search_terms: tuple[str, ...],
    title: str,
    excerpt: str | None,
) -> tuple[int, str, str, tuple[str, ...], tuple[str, ...]]:
    """Return ``(score, matched_term, reason, matched_tokens, term_tokens)``.

    The best term wins; ties keep the first term in plan order.
    """

    title_norm = _normalize_text(title)
    title_sequence = title_norm.split()
    title_set = frozenset(title_sequence)
    excerpt_norm = _normalize_text(excerpt) if excerpt else None
    excerpt_sequence = excerpt_norm.split() if excerpt_norm else None
    excerpt_set = frozenset(excerpt_sequence) if excerpt_sequence else frozenset()
    combined_set = title_set | excerpt_set
    best: tuple[int, str, str, tuple[str, ...], tuple[str, ...]] | None = None
    for term in search_terms:
        term_sequence = _normalize_text(term).split()
        if not term_sequence:
            raise ValueError("verified search term is blank after normalization")
        term_set = frozenset(term_sequence)
        term_tokens = tuple(sorted(term_set))
        score, reason = _score_term(
            term_sequence,
            term_set,
            title_sequence,
            title_set,
            excerpt_sequence,
            excerpt_set,
            combined_set,
        )
        matched = tuple(sorted(term_set & combined_set))
        candidate = (score, term, reason, matched, term_tokens)
        if best is None or score > best[0]:
            best = candidate
    if best is None:
        raise ValueError("search_terms must not be empty")
    return best


def document_identity(document: dict[str, Any]) -> str:
    """Stable identity: DOI casefolded, else canonical URL, else connector id."""

    doi = document.get("doi")
    if isinstance(doi, str) and doi.strip():
        return doi.strip().casefold()
    canonical = document.get("canonical_url")
    if isinstance(canonical, str) and canonical.strip():
        return canonical.strip()
    url = document.get("url")
    if isinstance(url, str) and url.strip():
        return url.strip()
    return f"{document['connector']}:{document['external_id']}"


def time_window(published_at: str | None) -> str:
    """Bucket a validated ISO date into previous / recent / unknown_date."""

    if published_at is None:
        return "unknown_date"
    if PREVIOUS_FROM <= published_at < PREVIOUS_UNTIL:
        return "previous"
    if RECENT_FROM <= published_at <= RECENT_UNTIL:
        return "recent"
    return "unknown_date"


def _trust_rank(tier: str) -> int:
    return _TRUST_ORDER.get(tier, 4)


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


def _load_plan(plan_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    manifest_path = plan_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"enrichment plan is incomplete: {plan_dir}")
    manifest = _read_json(manifest_path, "enrichment plan manifest")
    if not isinstance(manifest, dict):
        raise ValueError("enrichment plan manifest must be a JSON object")
    if manifest.get("schema_version") != LABELING_ENRICHMENT_PLAN_VERSION:
        raise ValueError(
            f"enrichment plan manifest {manifest.get('schema_version')!r} "
            f"does not match {LABELING_ENRICHMENT_PLAN_VERSION!r}"
        )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("enrichment plan manifest outputs must be an object")
    plan_bytes = _check_digest(plan_dir / "plan.json", outputs.get("plan.json"), "plan.json")
    try:
        plan = json.loads(plan_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"plan.json is not valid JSON: {error}") from error
    if not isinstance(plan, dict):
        raise ValueError("plan.json must be a JSON object")
    if plan.get("schema_version") != LABELING_ENRICHMENT_PLAN_VERSION:
        raise ValueError("plan.json schema_version does not match plan v2")
    bundle = plan.get("bundle")
    if not isinstance(bundle, dict) or not bundle.get("bundle_id"):
        raise ValueError("plan.json bundle.bundle_id is required")
    bundle_id = bundle["bundle_id"]
    if plan.get("cutoff_date") != CUTOFF_ISO:
        raise ValueError(f"plan cutoff {plan.get('cutoff_date')!r} does not match {CUTOFF_ISO!r}")
    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("plan.json candidates must be a non-empty list")
    if manifest.get("bundle_id") != bundle_id:
        raise ValueError("plan manifest bundle_id does not match plan bundle_id")
    candidate_count = _require_plain_count(
        manifest.get("candidate_count"), "plan manifest candidate_count"
    )
    bundle_count = bundle.get("candidate_count")
    if bundle_count != candidate_count or candidate_count != len(candidates):
        raise ValueError(
            "plan manifest candidate_count does not match "
            "plan bundle candidate_count and candidate list length"
        )
    manifest_bytes = manifest_path.read_bytes()
    return plan, {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_size": len(manifest_bytes),
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "plan_size": len(plan_bytes),
    }


def _load_result(
    result_dir: Path,
    bundle_id: str,
    plan_sha256: str,
    plan_size: int,
    candidate_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
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
    if manifest.get("bundle_id") != bundle_id:
        raise ValueError("enrichment result bundle_id does not match plan bundle_id")
    totals = manifest.get("totals")
    if not isinstance(totals, dict) or not totals:
        raise ValueError("enrichment result manifest totals must be a non-empty object")
    result_totals = {
        name: _require_plain_count(value, f"result manifest totals.{name}")
        for name, value in totals.items()
    }
    plan_ref = manifest.get("plan")
    if (
        not isinstance(plan_ref, dict)
        or plan_ref.get("sha256") != plan_sha256
        or plan_ref.get("size_bytes") != plan_size
    ):
        raise ValueError("enrichment result does not reference this exact plan.json")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("enrichment result manifest outputs must be an object")
    digests: dict[str, Any] = {}
    payloads: dict[str, bytes] = {}
    for filename in ("coverage.jsonl", "documents.jsonl", "request_results.jsonl"):
        payload = _check_digest(result_dir / filename, outputs.get(filename), filename)
        digests[filename] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        payloads[filename] = payload
    documents_payload = payloads["documents.jsonl"]
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(documents_payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"documents.jsonl line {lineno} is not valid JSON") from error
        if not isinstance(payload, dict):
            raise ValueError(f"documents.jsonl line {lineno} must be an object")
        rows.append(payload)
    if not rows:
        raise ValueError("documents.jsonl holds no document links")
    if result_totals.get("candidates") != candidate_count:
        raise ValueError(
            "result manifest totals.candidates does not match plan candidate count"
        )
    if result_totals.get("returned_documents") != len(rows):
        raise ValueError(
            "result manifest totals.returned_documents does not match documents.jsonl rows"
        )
    coverage_rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(
        payloads["coverage.jsonl"].decode("utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"coverage.jsonl line {lineno} is not valid JSON") from error
        if not isinstance(payload, dict):
            raise ValueError(f"coverage.jsonl line {lineno} must be an object")
        coverage_rows.append(payload)
    manifest_bytes = manifest_path.read_bytes()
    digests["manifest.json"] = {
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "size_bytes": len(manifest_bytes),
    }
    return rows, coverage_rows, digests


def _require_plain_count(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative int")
    return value


def _validate_coverage(
    coverage_rows: list[dict[str, Any]], candidate_ids: list[str]
) -> None:
    """Require complete scientific+industry coverage for every planned candidate.

    A partial or unknown result is a coverage fact, never a silent zero, so it
    stops the ranking before any output exists.
    """

    seen: set[tuple[str, str]] = set()
    for row in coverage_rows:
        candidate_id = row.get("candidate_id")
        source_class = row.get("source_class")
        if candidate_id not in candidate_ids:
            raise ValueError(
                f"coverage row has unknown candidate_id {candidate_id!r}"
            )
        if source_class not in ("scientific", "industry"):
            raise ValueError(
                f"coverage row for {candidate_id!r} "
                f"has unknown source_class {source_class!r}"
            )
        pair = (candidate_id, source_class)
        if pair in seen:
            raise ValueError(
                f"duplicate coverage row for {candidate_id!r} {source_class!r}"
            )
        seen.add(pair)
        status = row.get("status")
        if status != "complete":
            raise ValueError(
                f"coverage for {candidate_id!r} {source_class!r} "
                f"is {status!r}, not complete"
            )
        planned = _require_plain_count(
            row.get("planned_requests"),
            f"coverage planned_requests for {candidate_id!r} {source_class!r}",
        )
        successful = _require_plain_count(
            row.get("successful_requests"),
            f"coverage successful_requests for {candidate_id!r} {source_class!r}",
        )
        failed = _require_plain_count(
            row.get("failed_requests"),
            f"coverage failed_requests for {candidate_id!r} {source_class!r}",
        )
        if successful != planned or failed != 0:
            raise ValueError(
                f"coverage counts for {candidate_id!r} {source_class!r} "
                f"are inconsistent with complete status"
            )
        failed_ids = row.get("failed_request_ids")
        if not isinstance(failed_ids, list) or failed_ids:
            raise ValueError(
                f"coverage failed_request_ids for {candidate_id!r} {source_class!r} "
                f"must be an empty list"
            )
    for candidate_id in candidate_ids:
        for source_class in ("scientific", "industry"):
            if (candidate_id, source_class) not in seen:
                raise ValueError(
                    f"coverage row missing for {candidate_id!r} {source_class!r}"
                )


def _index_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Map candidate/search/request IDs to connectors and verified terms."""

    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("plan.json candidates must be a non-empty list")
    index: dict[str, Any] = {}
    search_ids: set[str] = set()
    request_ids: set[str] = set()
    for entry in candidates:
        if not isinstance(entry, dict):
            raise ValueError("plan candidate must be an object")
        candidate_id = entry.get("candidate_id")
        terms = entry.get("search_terms")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("plan candidate_id must be a non-empty string")
        if candidate_id in index:
            raise ValueError(f"duplicate candidate_id in plan: {candidate_id}")
        if not isinstance(terms, list) or not terms:
            raise ValueError(f"verified search_terms missing for {candidate_id}")
        for term in terms:
            if not isinstance(term, str) or not term.strip():
                raise ValueError(f"verified search term is blank for {candidate_id}")
            if not _normalize_text(term):
                raise ValueError(
                    f"verified search term is blank after normalization for {candidate_id}"
                )
        searches = entry.get("searches")
        if not isinstance(searches, list) or not searches:
            raise ValueError(f"plan searches missing for {candidate_id}")
        search_map: dict[str, Any] = {}
        for search in searches:
            if not isinstance(search, dict):
                raise ValueError("plan search must be an object")
            search_id = search.get("search_id")
            connector = search.get("connector")
            if not isinstance(search_id, str) or not search_id:
                raise ValueError("plan search_id must be a non-empty string")
            if connector not in ("openalex", "mediacloud"):
                raise ValueError(f"unknown plan connector: {connector!r}")
            if search_id in search_ids:
                raise ValueError(f"duplicate search_id in plan: {search_id}")
            search_ids.add(search_id)
            requests = search.get("requests")
            if not isinstance(requests, list) or not requests:
                raise ValueError(f"plan requests missing for {search_id}")
            request_map: dict[str, str] = {}
            for request in requests:
                if not isinstance(request, dict):
                    raise ValueError("plan request must be an object")
                request_id = request.get("request_id")
                if not isinstance(request_id, str) or not request_id:
                    raise ValueError("plan request_id must be a non-empty string")
                if request_id in request_ids:
                    raise ValueError(f"duplicate request_id in plan: {request_id}")
                request_ids.add(request_id)
                request_map[request_id] = connector
            search_map[search_id] = {"connector": connector, "requests": request_map}
        index[candidate_id] = {"search_terms": tuple(terms), "searches": search_map}
    return index


def _require_str(document: dict[str, Any], name: str, lineno: int) -> str:
    value = document.get(name)
    if not isinstance(value, str):
        raise ValueError(f"documents.jsonl line {lineno}: {name} must be a string")
    return value


def _validate_document(
    document: dict[str, Any], index: dict[str, Any], lineno: int
) -> str | None:
    """Validate one document link; return its normalized published_at (or None)."""
    candidate_id = _require_str(document, "candidate_id", lineno)
    search_id = _require_str(document, "search_id", lineno)
    request_id = _require_str(document, "request_id", lineno)
    connector = _require_str(document, "connector", lineno)
    candidate = index.get(candidate_id)
    if candidate is None:
        raise ValueError(f"documents.jsonl line {lineno}: unknown candidate_id {candidate_id!r}")
    search = candidate["searches"].get(search_id)
    if search is None:
        raise ValueError(f"documents.jsonl line {lineno}: unknown search_id {search_id!r}")
    if search["requests"].get(request_id) is None:
        raise ValueError(f"documents.jsonl line {lineno}: unknown request_id {request_id!r}")
    if connector != search["connector"]:
        raise ValueError(
            f"documents.jsonl line {lineno}: connector {connector!r} "
            f"does not match planned request {search['connector']!r}"
        )
    published_at = document.get("published_at")
    if published_at is None:
        window_date: str | None = None
    elif isinstance(published_at, str):
        try:
            parsed = date.fromisoformat(published_at)
        except ValueError as error:
            raise ValueError(
                f"documents.jsonl line {lineno}: published_at is not an ISO date"
            ) from error
        if parsed > LABELING_CUTOFF_DATE:
            raise ValueError(
                f"documents.jsonl line {lineno}: published_at {published_at} "
                f"is past cutoff {CUTOFF_ISO}"
            )
        window_date = published_at
    else:
        raise ValueError(
            f"documents.jsonl line {lineno}: published_at must be an ISO date string or null"
        )
    for name in ("document_id", "origin_id", "trust_tier", "url", "canonical_url", "external_id"):
        _require_str(document, name, lineno)
    title = document.get("title")
    if not isinstance(title, str):
        raise ValueError(f"documents.jsonl line {lineno}: title must be a string")
    excerpt = document.get("excerpt")
    if excerpt is not None and not isinstance(excerpt, str):
        raise ValueError(f"documents.jsonl line {lineno}: excerpt must be a string or null")
    doi = document.get("doi")
    if doi is not None and not isinstance(doi, str):
        raise ValueError(f"documents.jsonl line {lineno}: doi must be a string or null")
    publisher = document.get("publisher")
    if publisher is not None and not isinstance(publisher, str):
        raise ValueError(f"documents.jsonl line {lineno}: publisher must be a string or null")
    return window_date


def _render_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return (
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ).encode("utf-8")
    )


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the best row per document_identity inside one candidate."""

    best: dict[str, dict[str, Any]] = {}
    dropped: dict[str, list[str]] = {}

    def sort_key(row: dict[str, Any]) -> tuple[int, int, int, str]:
        excerpt = row["excerpt"] or ""
        empty = 1 if not excerpt.strip() else 0
        return (-row["score"], empty, -len(excerpt), row["document_id"])

    for row in sorted(rows, key=lambda item: item["document_id"]):
        identity = row["document_identity"]
        current = best.get(identity)
        if current is None:
            best[identity] = row
            continue
        if sort_key(row) < sort_key(current):
            dropped.setdefault(identity, []).append(current["document_id"])
            best[identity] = row
        else:
            dropped.setdefault(identity, []).append(row["document_id"])
    survivors = []
    for row in best.values():
        row = dict(row)
        row["duplicate_document_ids"] = sorted(dropped.get(row["document_identity"], ()))
        survivors.append(row)
    return survivors


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            row["candidate_id"],
            row["connector"],
            -row["score"],
            row["document_identity"],
            row["document_id"],
        ),
    )


def _bucket_key(row: dict[str, Any]) -> tuple[int, int, int, str, str, str]:
    published_at = row["published_at"]
    return (
        -row["score"],
        _trust_rank(row["trust_tier"]),
        1 if published_at is None else 0,
        _invert_date(published_at) if published_at is not None else "",
        row["document_identity"],
        row["document_id"],
    )


def _invert_date(published_at: str) -> str:
    return "".join(chr(0x10FFFF - ord(char)) for char in published_at)


def _select_scientific(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Round-robin strong then weak across time windows, max 4, origin-diverse."""

    eligible = [
        row
        for row in rows
        if row["connector"] == "openalex"
        and (row["excerpt"] or "").strip()
        and row["relevance_class"] in (CLASS_STRONG, CLASS_WEAK)
    ]
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in eligible:
        buckets.setdefault((row["relevance_class"], row["time_window"]), []).append(row)
    for bucket in buckets.values():
        bucket.sort(key=_bucket_key)
    used_origins: set[str] = set()
    selected: list[dict[str, Any]] = []
    for relevance_class in (CLASS_STRONG, CLASS_WEAK):
        while len(selected) < SCIENTIFIC_PER_CANDIDATE:
            progressed = False
            for window in _WINDOW_CYCLE:
                if len(selected) >= SCIENTIFIC_PER_CANDIDATE:
                    break
                bucket = buckets.get((relevance_class, window))
                if not bucket:
                    continue
                pick_index = next(
                    (
                        position
                        for position, row in enumerate(bucket)
                        if row["origin_id"] not in used_origins
                    ),
                    0,
                )
                row = bucket.pop(pick_index)
                fresh = row["origin_id"] not in used_origins
                used_origins.add(row["origin_id"])
                selected.append({
                    **row,
                    "selection_rank": len(selected) + 1,
                    "selection_reason": (
                        f"{relevance_class}|{window}|"
                        f"{'new_origin' if fresh else 'repeat_origin'}"
                    ),
                })
                progressed = True
            if not progressed:
                break
    return selected


def _select_media(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Round-robin strong, weak, then none across windows, publishers, origins."""

    eligible = [row for row in rows if row["connector"] == "mediacloud" and row["url"].strip()]
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in eligible:
        buckets.setdefault((row["relevance_class"], row["time_window"]), []).append(row)
    for bucket in buckets.values():
        bucket.sort(key=_bucket_key)
    used_origins: set[str] = set()
    used_publishers: set[str] = set()
    selected: list[dict[str, Any]] = []
    for relevance_class in (CLASS_STRONG, CLASS_WEAK, CLASS_NONE):
        while len(selected) < MEDIA_PER_CANDIDATE:
            progressed = False
            for window in _WINDOW_CYCLE:
                if len(selected) >= MEDIA_PER_CANDIDATE:
                    break
                bucket = buckets.get((relevance_class, window))
                if not bucket:
                    continue
                pick_index = _diverse_pick(bucket, used_origins, used_publishers)
                row = bucket.pop(pick_index)
                fresh_origin = row["origin_id"] not in used_origins
                publisher = row["publisher"] or ""
                fresh_publisher = publisher not in used_publishers
                used_origins.add(row["origin_id"])
                used_publishers.add(publisher)
                selected.append({
                    **row,
                    "selection_rank": len(selected) + 1,
                    "selection_reason": (
                        f"{relevance_class}|{window}|"
                        f"{'new_origin' if fresh_origin else 'repeat_origin'}|"
                        f"{'new_publisher' if fresh_publisher else 'repeat_publisher'}"
                    ),
                })
                progressed = True
            if not progressed:
                break
    return selected


def _diverse_pick(
    bucket: list[dict[str, Any]],
    used_origins: set[str],
    used_publishers: set[str],
) -> int:
    """Prefer unused origin and publisher pairs, keeping bucket order otherwise."""

    for rank in range(4):
        for position, row in enumerate(bucket):
            fresh_origin = row["origin_id"] not in used_origins
            fresh_publisher = (row["publisher"] or "") not in used_publishers
            if rank == 0 and fresh_origin and fresh_publisher:
                return position
            if rank == 1 and fresh_origin:
                return position
            if rank == 2 and fresh_publisher:
                return position
            if rank == 3:
                return position
    return 0


def build_relevance_plan(
    plan_dir: str | Path,
    result_dir: str | Path,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Validate inputs and render deterministic ranking bytes.

    Raises before touching the output directory, so failures never leave
    partial files behind.
    """

    plan_path = Path(plan_dir)
    result_path = Path(result_dir)
    plan, plan_digests = _load_plan(plan_path)
    bundle_id = plan["bundle"]["bundle_id"]
    index = _index_plan(plan)
    documents, coverage_rows, result_digests = _load_result(
        result_path,
        bundle_id,
        plan_digests["plan_sha256"],
        plan_digests["plan_size"],
        len(index),
    )
    _validate_coverage(coverage_rows, sorted(index))

    rows_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for lineno, document in enumerate(documents, start=1):
        window_date = _validate_document(document, index, lineno)
        candidate_id = document["candidate_id"]
        excerpt = document.get("excerpt")
        excerpt = excerpt if isinstance(excerpt, str) and excerpt.strip() else None
        score, term, reason, matched, term_tokens = score_document(
            index[candidate_id]["search_terms"],
            document.get("title") or "",
            excerpt,
        )
        rows_by_candidate.setdefault(candidate_id, []).append({
            "candidate_id": candidate_id,
            "document_id": document["document_id"],
            "connector": document["connector"],
            "search_id": document["search_id"],
            "request_id": document["request_id"],
            "matched_term": term,
            "score": score,
            "relevance_class": _classify(score),
            "reasons": [reason],
            "matched_tokens": list(matched),
            "term_tokens": list(term_tokens),
            "published_at": window_date,
            "origin_id": document["origin_id"],
            "publisher": document.get("publisher"),
            "trust_tier": document["trust_tier"],
            "url": document["url"],
            "excerpt": excerpt,
            "time_window": time_window(window_date),
            "document_identity": document_identity(document),
        })

    ranked: list[dict[str, Any]] = []
    scientific: list[dict[str, Any]] = []
    media_queue: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    totals = {
        "candidates": 0,
        "links_found": 0,
        "links_kept": 0,
        "openalex": {CLASS_STRONG: 0, CLASS_WEAK: 0, CLASS_NONE: 0},
        "mediacloud": {CLASS_STRONG: 0, CLASS_WEAK: 0, CLASS_NONE: 0},
        "scientific_shortlist_rows": 0,
        "media_fetch_queue_rows": 0,
        "candidates_without_scientific_shortlist": 0,
        "candidates_without_media_targets": 0,
    }
    for candidate_id in sorted(index):
        found = rows_by_candidate.get(candidate_id, [])
        kept = _dedupe(found)
        kept_ranked = _rank_rows(kept)
        for row in kept_ranked:
            totals[row["connector"]][row["relevance_class"]] += 1
        shortlist = _select_scientific(kept_ranked)
        fetch = _select_media(kept_ranked)
        window_counts = {"previous": 0, "recent": 0, "unknown_date": 0}
        for row in kept_ranked:
            window_counts[row["time_window"]] += 1
        reasons: list[str] = []
        if not any(row["connector"] == "openalex" for row in kept_ranked):
            reasons.append("scientific:no_openalex_links")
        elif not any(
            row["connector"] == "openalex" and (row["excerpt"] or "").strip()
            for row in kept_ranked
        ):
            reasons.append("scientific:no_openalex_with_excerpt")
        elif not shortlist:
            reasons.append("scientific:no_strong_or_weak_openalex")
        if not any(row["connector"] == "mediacloud" for row in kept_ranked):
            reasons.append("media:no_media_links")
        elif not fetch:
            reasons.append("media:no_media_with_url")
        coverage.append({
            "candidate_id": candidate_id,
            "links_found": len(found),
            "links_kept": len(kept_ranked),
            "openalex_strong": sum(
                1 for row in kept_ranked
                if row["connector"] == "openalex" and row["relevance_class"] == CLASS_STRONG
            ),
            "openalex_weak": sum(
                1 for row in kept_ranked
                if row["connector"] == "openalex" and row["relevance_class"] == CLASS_WEAK
            ),
            "openalex_none": sum(
                1 for row in kept_ranked
                if row["connector"] == "openalex" and row["relevance_class"] == CLASS_NONE
            ),
            "mediacloud_strong": sum(
                1 for row in kept_ranked
                if row["connector"] == "mediacloud" and row["relevance_class"] == CLASS_STRONG
            ),
            "mediacloud_weak": sum(
                1 for row in kept_ranked
                if row["connector"] == "mediacloud" and row["relevance_class"] == CLASS_WEAK
            ),
            "mediacloud_none": sum(
                1 for row in kept_ranked
                if row["connector"] == "mediacloud" and row["relevance_class"] == CLASS_NONE
            ),
            "scientific_shortlist_count": len(shortlist),
            "media_fetch_queue_count": len(fetch),
            "window_previous": window_counts["previous"],
            "window_recent": window_counts["recent"],
            "window_unknown_date": window_counts["unknown_date"],
            "has_scientific_shortlist": bool(shortlist),
            "has_media_fetch_targets": bool(fetch),
            "empty_shortlist_reasons": reasons,
        })
        ranked.extend(
            {key: value for key, value in row.items() if key != "excerpt"}
            for row in kept_ranked
        )
        scientific.extend(
            {key: value for key, value in row.items() if key != "excerpt"}
            for row in shortlist
        )
        media_queue.extend(
            {key: value for key, value in row.items() if key != "excerpt"}
            for row in fetch
        )
        totals["candidates"] += 1
        totals["links_found"] += len(found)
        totals["links_kept"] += len(kept_ranked)
        totals["scientific_shortlist_rows"] += len(shortlist)
        totals["media_fetch_queue_rows"] += len(fetch)
        if not shortlist:
            totals["candidates_without_scientific_shortlist"] += 1
        if not fetch:
            totals["candidates_without_media_targets"] += 1

    files = {
        RANKED_FILENAME: _render_jsonl(ranked),
        SCIENTIFIC_FILENAME: _render_jsonl(scientific),
        MEDIA_QUEUE_FILENAME: _render_jsonl(media_queue),
        COVERAGE_FILENAME: _render_jsonl(coverage),
    }
    manifest = {
        "schema_version": LABELING_RELEVANCE_PLAN_VERSION,
        "bundle_id": bundle_id,
        "cutoff_date": CUTOFF_ISO,
        "inputs": {
            "plan_manifest": {
                "sha256": plan_digests["manifest_sha256"],
                "size_bytes": plan_digests["manifest_size"],
            },
            "plan_file": {
                "sha256": plan_digests["plan_sha256"],
                "size_bytes": plan_digests["plan_size"],
            },
            "result_manifest": {
                "sha256": result_digests["manifest.json"]["sha256"],
                "size_bytes": result_digests["manifest.json"]["size_bytes"],
            },
            "result_files": {
                name: result_digests[name]
                for name in ("coverage.jsonl", "documents.jsonl", "request_results.jsonl")
            },
        },
        "limits": {
            "scientific_shortlist_per_candidate": SCIENTIFIC_PER_CANDIDATE,
            "media_fetch_queue_per_candidate": MEDIA_PER_CANDIDATE,
            "shortlist_connector": "openalex",
            "shortlist_requires_excerpt": True,
            "shortlist_classes": [CLASS_STRONG, CLASS_WEAK],
            "fetch_connector": "mediacloud",
            "fetch_fill_class": CLASS_NONE,
            "fetch_requires_url": True,
        },
        "score_policy": {
            "description": (
                "Whole-token retrieval ranking of verified search_terms against "
                "title and excerpt. Normalization: Unicode NFKC, casefold, "
                "non-alphanumeric runs to space, collapsed spaces. Not evidence, "
                "not a candidate model score; target and expert labels are unused."
            ),
            "hierarchy": [
                {"score": 100, "reason": REASON_EXACT_TITLE},
                {"score": 80, "reason": REASON_TITLE_TOKENS},
                {"score": 70, "reason": REASON_EXACT_EXCERPT},
                {"score": 60, "reason": REASON_COMBINED_TOKENS},
                {"score": "round(40 * matched_unique / term_unique)", "reason": REASON_PARTIAL},
                {"score": 0, "reason": REASON_NO_OVERLAP},
            ],
            "classes": {"strong": "score >= 60", "weak": "20..59", "none": "score < 20"},
            "tie_break": "maximum over terms; ties keep the first term in plan order",
        },
        "totals": totals,
        "outputs": {
            name: {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for name, payload in files.items()
        },
    }
    files[RELEVANCE_MANIFEST_FILENAME] = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    provenance = {
        "bundle_id": bundle_id,
        "candidate_count": totals["candidates"],
        "plan_manifest": {
            "sha256": plan_digests["manifest_sha256"],
            "size_bytes": plan_digests["manifest_size"],
        },
    }
    return files, provenance


def export_relevance_plan(
    *,
    plan_dir: str | Path,
    result_dir: str | Path,
    output_dir: str | Path,
) -> LabelingRelevancePlanPaths:
    """Validate inputs and atomically publish the relevance ranking."""

    files, _provenance = build_relevance_plan(plan_dir, result_dir)
    paths = publish_artifact_bundle(files, output_dir)
    return LabelingRelevancePlanPaths(
        manifest=paths[RELEVANCE_MANIFEST_FILENAME],
        ranked_documents=paths[RANKED_FILENAME],
        scientific_shortlist=paths[SCIENTIFIC_FILENAME],
        media_fetch_queue=paths[MEDIA_QUEUE_FILENAME],
        coverage=paths[COVERAGE_FILENAME],
    )
