"""Deterministic evidence-input planner: reranked media texts plus science.

Reads the enrichment plan (reviewed search_terms only), the enrichment
result, the relevance plan and the media fetch result. No API, no network,
no LLM, no EvidenceClaim. Target, expert labels, planned classes, canonical
names, source queries and domains are never read and never scored.

Only ``article_text`` pages re-enter scoring with their full text; meta
descriptions, title-only pages and failed fetches are recorded as
``not_evidence_text`` with an explicit exclusion reason and never selected.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from nextwave.datasets.artifacts import publish_artifact_bundle

from .contracts import LABELING_CUTOFF_DATE
from .enrichment_plan import LABELING_ENRICHMENT_PLAN_VERSION
from .enrichment_run import ENRICHMENT_RESULT_VERSION
from .media_fetch_run import LABELING_MEDIA_FETCH_RESULT_VERSION
from .relevance_plan import (
    LABELING_RELEVANCE_PLAN_VERSION,
    document_identity,
    time_window,
)

LABELING_EVIDENCE_INPUT_PLAN_VERSION = "labeling-evidence-input-plan-v4"

CUTOFF_ISO = LABELING_CUTOFF_DATE.isoformat()
MEDIA_WINDOW_TOKENS = 32
WEAK_MIN_WINDOW_TOKENS = 33
WEAK_MAX_WINDOW_TOKENS = 96
WEAK_ADMITTED_SCORE = 40
SCIENTIFIC_WEAK_MIN_SCORE = 30
SCIENTIFIC_WEAK_MIN_MATCHED_TOKENS = 2
SCIENTIFIC_WEAK_MAX_EXCERPT_WINDOW_TOKENS = 32
SCIENTIFIC_PER_CANDIDATE = 4
MEDIA_CAPACITY_MAX = 4
MEDIA_CAPACITY_MIN = 2
FINAL_PER_CANDIDATE = 6

RERANKED_FILENAME = "media_reranked.jsonl"
EVIDENCE_INPUT_FILENAME = "evidence_input_documents.jsonl"
COVERAGE_FILENAME = "coverage.jsonl"
EVIDENCE_INPUT_MANIFEST_FILENAME = "manifest.json"

REASON_EXACT_TITLE = "exact_title_phrase"
REASON_TITLE_TOKENS = "all_tokens_in_title"
REASON_EXACT_ARTICLE = "exact_article_phrase"
REASON_WINDOW = "all_tokens_in_window"
REASON_PARTIAL = "partial_token_overlap"
REASON_NO_OVERLAP = "no_token_overlap"

NOT_EVIDENCE_CLASS = "not_evidence_text"

_TRUST_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3}
_WINDOW_CYCLE = ("recent", "previous", "unknown_date")
_VALID_CLASSES = ("strong", "weak", "none")


@dataclass(frozen=True, slots=True)
class LabelingEvidenceInputPlanPaths:
    """Paths of one published evidence input plan."""

    manifest: Path
    media_reranked: Path
    evidence_input_documents: Path
    coverage: Path


def _normalize_text(value: str) -> str:
    """NFKC + casefold + non-alphanumeric runs to one space."""

    folded = unicodedata.normalize("NFKC", value).casefold()
    return " ".join("".join(char if char.isalnum() else " " for char in folded).split())


def _classify(score: int) -> str:
    if score >= 60:
        return "strong"
    if score >= 20:
        return "weak"
    return "none"


def media_capacity(scientific_count: int) -> int:
    """Upper media bound from the scientific shortlist size."""

    return min(MEDIA_CAPACITY_MAX, max(MEDIA_CAPACITY_MIN, 6 - scientific_count))


def _contains_sequence(field_tokens: list[str], term_tokens: list[str]) -> bool:
    """Contiguous whole-token subsequence match preserving term order."""

    if not term_tokens or len(term_tokens) > len(field_tokens):
        return False
    width = len(term_tokens)
    return any(
        field_tokens[start:start + width] == term_tokens
        for start in range(len(field_tokens) - width + 1)
    )


def _min_window_width(article_tokens: list[str], term_set: frozenset[str]) -> int | None:
    """Narrowest token span covering every unique term token, or None."""

    positions = [
        (index, token) for index, token in enumerate(article_tokens) if token in term_set
    ]
    if {token for _, token in positions} != set(term_set):
        return None
    best: int | None = None
    counts: dict[str, int] = {}
    left = 0
    for right in range(len(positions)):
        token = positions[right][1]
        counts[token] = counts.get(token, 0) + 1
        while len(counts) == len(term_set):
            span = positions[right][0] - positions[left][0] + 1
            if best is None or span < best:
                best = span
            dropped = positions[left][1]
            counts[dropped] -= 1
            if counts[dropped] == 0:
                del counts[dropped]
            left += 1
    return best


def _score_term(
    term_sequence: list[str],
    term_set: frozenset[str],
    title_sequence: list[str],
    title_set: frozenset[str],
    article_sequence: list[str],
    article_set: frozenset[str],
    combined_set: frozenset[str],
) -> tuple[int, str, int | None]:
    """Score one reviewed term against title and full article text."""

    if _contains_sequence(title_sequence, term_sequence):
        return 100, REASON_EXACT_TITLE, None
    if term_set <= title_set:
        return 80, REASON_TITLE_TOKENS, None
    if _contains_sequence(article_sequence, term_sequence):
        return 70, REASON_EXACT_ARTICLE, None
    width = _min_window_width(article_sequence, term_set)
    if width is not None and width <= MEDIA_WINDOW_TOKENS:
        return 60, REASON_WINDOW, width
    matched = len(term_set & combined_set)
    if matched == 0:
        return 0, REASON_NO_OVERLAP, None
    if matched == len(term_set):
        return 40, REASON_PARTIAL, width
    return round(40 * matched / len(term_set)), REASON_PARTIAL, None


def score_media_document(
    search_terms: tuple[str, ...],
    title: str,
    article_text: str,
) -> tuple[int, str, str, str, tuple[str, ...], tuple[str, ...], int | None]:
    """Return ``(score, class, term, reason, matched, term_tokens, width)``.

    The best reviewed term wins; ties keep the first term in plan order.
    """

    title_sequence = _normalize_text(title).split()
    title_set = frozenset(title_sequence)
    article_sequence = _normalize_text(article_text).split()
    article_set = frozenset(article_sequence)
    combined_set = title_set | article_set
    best: tuple[int, str, str, str, tuple[str, ...], tuple[str, ...], int | None] | None = None
    for term in search_terms:
        term_sequence = _normalize_text(term).split()
        if not term_sequence:
            raise ValueError("reviewed search term is blank after normalization")
        term_set = frozenset(term_sequence)
        term_tokens = tuple(sorted(term_set))
        score, reason, width = _score_term(
            term_sequence,
            term_set,
            title_sequence,
            title_set,
            article_sequence,
            article_set,
            combined_set,
        )
        matched = tuple(sorted(term_set & combined_set))
        candidate = (score, _classify(score), term, reason, matched, term_tokens, width)
        if best is None or score > best[0]:
            best = candidate
    if best is None:
        raise ValueError("search_terms must not be empty")
    return best


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


def _check_date(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label} published_at must be an ISO date string or null")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{label} published_at is not an ISO date") from error
    if parsed > LABELING_CUTOFF_DATE:
        raise ValueError(f"{label} published_at {value} is past cutoff {CUTOFF_ISO}")
    return value


def _load_plan(plan_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load the enrichment plan: reviewed terms and search classes only."""

    manifest_path = plan_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"enrichment plan is incomplete: {plan_dir}")
    manifest = _read_json(manifest_path, "enrichment plan manifest")
    if not isinstance(manifest, dict):
        raise ValueError("enrichment plan manifest must be a JSON object")
    if manifest.get("schema_version") != LABELING_ENRICHMENT_PLAN_VERSION:
        raise ValueError("enrichment plan manifest version mismatch")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("enrichment plan manifest outputs must be an object")
    plan_bytes = _check_digest(plan_dir / "plan.json", outputs.get("plan.json"), "plan.json")
    try:
        plan = json.loads(plan_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"plan.json is not valid JSON: {error}") from error
    if not isinstance(plan, dict) or plan.get("schema_version") != LABELING_ENRICHMENT_PLAN_VERSION:
        raise ValueError("plan.json schema version mismatch")
    bundle = plan.get("bundle")
    if not isinstance(bundle, dict) or not bundle.get("bundle_id"):
        raise ValueError("plan.json bundle.bundle_id is required")
    if manifest.get("bundle_id") != bundle["bundle_id"]:
        raise ValueError("plan manifest bundle_id does not match plan bundle_id")
    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("plan.json candidates must be a non-empty list")
    terms: dict[str, tuple[str, ...]] = {}
    search_classes: dict[str, str] = {}
    for entry in candidates:
        if not isinstance(entry, dict):
            raise ValueError("plan candidate must be an object")
        candidate_id = entry.get("candidate_id")
        raw_terms = entry.get("search_terms")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("plan candidate_id must be a non-empty string")
        if candidate_id in terms:
            raise ValueError(f"duplicate plan candidate_id: {candidate_id}")
        if not isinstance(raw_terms, list) or not raw_terms:
            raise ValueError(f"reviewed search_terms missing for {candidate_id}")
        clean_terms = []
        for term in raw_terms:
            if not isinstance(term, str) or not term.strip():
                raise ValueError(f"reviewed search term is blank for {candidate_id}")
            if not _normalize_text(term):
                raise ValueError(f"reviewed search term normalizes blank for {candidate_id}")
            clean_terms.append(term)
        terms[candidate_id] = tuple(clean_terms)
        searches = entry.get("searches")
        if not isinstance(searches, list) or not searches:
            raise ValueError(f"plan searches missing for {candidate_id}")
        for search in searches:
            if not isinstance(search, dict):
                raise ValueError("plan search must be an object")
            search_id = search.get("search_id")
            source_class = search.get("source_class")
            if not isinstance(search_id, str) or not search_id:
                raise ValueError("plan search_id must be a non-empty string")
            if source_class not in ("scientific", "industry"):
                raise ValueError(f"unknown plan source_class: {source_class!r}")
            if search_id in search_classes and search_classes[search_id] != source_class:
                raise ValueError(f"conflicting source_class for {search_id}")
            search_classes[search_id] = source_class
    manifest_bytes = manifest_path.read_bytes()
    digests = {
        "manifest.json": _digest(manifest_bytes),
        "plan.json": _digest(plan_bytes),
    }
    return manifest, {"terms": terms, "search_classes": search_classes}, digests


def _load_result(
    result_dir: Path, bundle_id: str, plan_digest: dict[str, Any]
) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    manifest_path = result_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"enrichment result is incomplete: {result_dir}")
    manifest = _read_json(manifest_path, "enrichment result manifest")
    if not isinstance(manifest, dict):
        raise ValueError("enrichment result manifest must be a JSON object")
    if manifest.get("schema_version") != ENRICHMENT_RESULT_VERSION:
        raise ValueError("enrichment result manifest version mismatch")
    if manifest.get("bundle_id") != bundle_id:
        raise ValueError("enrichment result bundle_id mismatch")
    plan_ref = manifest.get("plan")
    if (
        not isinstance(plan_ref, dict)
        or plan_ref.get("sha256") != plan_digest["sha256"]
        or plan_ref.get("size_bytes") != plan_digest["size_bytes"]
    ):
        raise ValueError("enrichment result does not reference this exact plan.json")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("enrichment result manifest outputs must be an object")
    digests: dict[str, Any] = {}
    documents: dict[tuple[str, str], dict[str, Any]] = {}
    for filename in ("coverage.jsonl", "documents.jsonl", "request_results.jsonl"):
        payload = _check_digest(result_dir / filename, outputs.get(filename), filename)
        digests[filename] = _digest(payload)
        if filename != "documents.jsonl":
            continue
        for lineno, row in enumerate(_read_rows(result_dir / filename, payload, filename), start=1):
            pair = (row.get("candidate_id"), row.get("document_id"))
            if not isinstance(pair[0], str) or not isinstance(pair[1], str):
                raise ValueError(f"documents.jsonl line {lineno} needs candidate/document IDs")
            if pair in documents:
                raise ValueError(f"documents.jsonl line {lineno} duplicates pair {pair!r}")
            documents[pair] = row
    totals = manifest.get("totals")
    if not isinstance(totals, dict) or totals.get("returned_documents") != len(documents):
        raise ValueError("enrichment result totals do not match documents.jsonl rows")
    manifest_bytes = manifest_path.read_bytes()
    digests["manifest.json"] = _digest(manifest_bytes)
    return manifest, documents, digests


def _load_relevance(
    relevance_dir: Path,
    bundle_id: str,
    plan_digests: dict[str, Any],
    result_digests: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest_path = relevance_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"relevance plan is incomplete: {relevance_dir}")
    manifest = _read_json(manifest_path, "relevance manifest")
    if not isinstance(manifest, dict):
        raise ValueError("relevance manifest must be a JSON object")
    if manifest.get("schema_version") != LABELING_RELEVANCE_PLAN_VERSION:
        raise ValueError("relevance manifest version mismatch")
    if manifest.get("bundle_id") != bundle_id:
        raise ValueError("relevance bundle_id mismatch")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("relevance manifest inputs must be an object")
    for name, expected in (
        ("plan_file", plan_digests["plan.json"]),
        ("plan_manifest", plan_digests["manifest.json"]),
        ("result_manifest", result_digests["manifest.json"]),
    ):
        actual = inputs.get(name)
        if (
            not isinstance(actual, dict)
            or actual.get("sha256") != expected["sha256"]
            or actual.get("size_bytes") != expected["size_bytes"]
        ):
            raise ValueError(f"relevance artifact does not reference this exact {name}")
    expected_files = inputs.get("result_files")
    if not isinstance(expected_files, dict):
        raise ValueError("relevance manifest inputs.result_files must be an object")
    for filename in ("coverage.jsonl", "documents.jsonl", "request_results.jsonl"):
        expected = expected_files.get(filename)
        actual = result_digests[filename]
        if (
            not isinstance(expected, dict)
            or expected.get("sha256") != actual["sha256"]
            or expected.get("size_bytes") != actual["size_bytes"]
        ):
            raise ValueError(f"relevance artifact does not reference this exact {filename}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("relevance manifest outputs must be an object")
    digests: dict[str, Any] = {}
    ranked: list[dict[str, Any]] = []
    queue: list[dict[str, Any]] = []
    shortlist: list[dict[str, Any]] = []
    for filename in (
        "ranked_documents.jsonl",
        "scientific_shortlist.jsonl",
        "media_fetch_queue.jsonl",
        "coverage.jsonl",
    ):
        payload = _check_digest(relevance_dir / filename, outputs.get(filename), filename)
        digests[filename] = _digest(payload)
        rows = _read_rows(relevance_dir / filename, payload, filename)
        if filename == "ranked_documents.jsonl":
            ranked = rows
        elif filename == "media_fetch_queue.jsonl":
            queue = rows
        elif filename == "scientific_shortlist.jsonl":
            shortlist = rows
    totals = manifest.get("totals")
    if not isinstance(totals, dict):
        raise ValueError("relevance manifest totals must be an object")
    if totals.get("media_fetch_queue_rows") != len(queue):
        raise ValueError("relevance totals do not match media_fetch_queue.jsonl rows")
    if totals.get("scientific_shortlist_rows") != len(shortlist):
        raise ValueError("relevance totals do not match scientific_shortlist.jsonl rows")
    if totals.get("links_kept") != len(ranked):
        raise ValueError("relevance totals do not match ranked_documents.jsonl rows")
    manifest_bytes = manifest_path.read_bytes()
    digests["manifest.json"] = _digest(manifest_bytes)
    return manifest, ranked, queue, shortlist, digests


def _load_media(
    media_dir: Path,
    bundle_id: str,
    relevance_digests: dict[str, Any],
    result_digests: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest_path = media_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"media fetch result is incomplete: {media_dir}")
    manifest = _read_json(manifest_path, "media fetch manifest")
    if not isinstance(manifest, dict):
        raise ValueError("media fetch manifest must be a JSON object")
    if manifest.get("schema_version") != LABELING_MEDIA_FETCH_RESULT_VERSION:
        raise ValueError("media fetch manifest version mismatch")
    if manifest.get("bundle_id") != bundle_id:
        raise ValueError("media fetch bundle_id mismatch")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("media fetch manifest inputs must be an object")
    actual_rel_manifest = inputs.get("relevance_manifest")
    if (
        not isinstance(actual_rel_manifest, dict)
        or actual_rel_manifest.get("sha256") != relevance_digests["manifest.json"]["sha256"]
        or actual_rel_manifest.get("size_bytes")
        != relevance_digests["manifest.json"]["size_bytes"]
    ):
        raise ValueError("media result does not reference this exact relevance manifest")
    for group, expected_digests in (
        ("relevance_files", {
            name: relevance_digests[name]
            for name in (
                "ranked_documents.jsonl",
                "scientific_shortlist.jsonl",
                "media_fetch_queue.jsonl",
                "coverage.jsonl",
            )
        }),
        ("result_files", {
            name: result_digests[name]
            for name in ("coverage.jsonl", "documents.jsonl", "request_results.jsonl")
        }),
    ):
        expected_group = inputs.get(group)
        if not isinstance(expected_group, dict):
            raise ValueError(f"media fetch manifest inputs.{group} must be an object")
        for filename, expected in expected_digests.items():
            actual = expected_group.get(filename)
            if (
                not isinstance(actual, dict)
                or actual.get("sha256") != expected["sha256"]
                or actual.get("size_bytes") != expected["size_bytes"]
            ):
                raise ValueError(
                    f"media result does not reference this exact {group}/{filename}"
                )
    expected_result_manifest = inputs.get("result_manifest")
    if (
        not isinstance(expected_result_manifest, dict)
        or expected_result_manifest.get("sha256")
        != result_digests["manifest.json"]["sha256"]
        or expected_result_manifest.get("size_bytes")
        != result_digests["manifest.json"]["size_bytes"]
    ):
        raise ValueError("media result does not reference this exact result manifest")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("media fetch manifest outputs must be an object")
    digests: dict[str, Any] = {}
    pages: list[dict[str, Any]] = []
    enriched: list[dict[str, Any]] = []
    for filename in ("page_results.jsonl", "enriched_documents.jsonl", "coverage.jsonl"):
        payload = _check_digest(media_dir / filename, outputs.get(filename), filename)
        digests[filename] = _digest(payload)
        rows = _read_rows(media_dir / filename, payload, filename)
        if filename == "page_results.jsonl":
            pages = rows
        elif filename == "enriched_documents.jsonl":
            enriched = rows
    totals = manifest.get("totals")
    if not isinstance(totals, dict):
        raise ValueError("media fetch manifest totals must be an object")
    if totals.get("planned_fetches") != len(pages):
        raise ValueError("media totals do not match page_results.jsonl rows")
    manifest_bytes = manifest_path.read_bytes()
    digests["manifest.json"] = _digest(manifest_bytes)
    return manifest, pages, enriched, digests


def _require_pair_uniqueness(
    rows: list[dict[str, Any]], label: str
) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for lineno, row in enumerate(rows, start=1):
        pair = (row.get("candidate_id"), row.get("document_id"))
        if not isinstance(pair[0], str) or not isinstance(pair[1], str):
            raise ValueError(f"{label} line {lineno} needs candidate/document IDs")
        if pair in index:
            raise ValueError(f"{label} line {lineno} duplicates pair {pair!r}")
        index[pair] = row
    return index


def _require_rank_sequence(
    index: dict[tuple[str, str], dict[str, Any]], label: str, limit: int
) -> None:
    ranks_by_candidate: dict[str, list[int]] = {}
    for (candidate_id, _), row in index.items():
        ranks_by_candidate.setdefault(candidate_id, []).append(row["selection_rank"])
    for candidate_id in sorted(ranks_by_candidate):
        ordered = sorted(ranks_by_candidate[candidate_id])
        if ordered != list(range(1, len(ordered) + 1)):
            raise ValueError(
                f"{label} ranks for {candidate_id!r} must form 1..N "
                "without repeats or gaps"
            )
        if len(ordered) > limit:
            raise ValueError(
                f"{label} ranks for {candidate_id!r} exceed the limit of {limit}"
            )


def _require_plain_rank(row: dict[str, Any], label: str, pair: tuple[str, str]) -> None:
    rank = row.get("selection_rank")
    if not isinstance(rank, int) or isinstance(rank, bool):
        raise ValueError(f"{label} pair {pair!r} selection_rank must be a plain int")


def _cross_check(
    terms: dict[str, tuple[str, ...]],
    search_classes: dict[str, str],
    documents: dict[tuple[str, str], dict[str, Any]],
    ranked: list[dict[str, Any]],
    queue: list[dict[str, Any]],
    shortlist: list[dict[str, Any]],
    pages: list[dict[str, Any]],
    enriched: list[dict[str, Any]],
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
]:
    """Verify every cross-artifact reference before any scoring."""

    ranked_index = _require_pair_uniqueness(ranked, "ranked_documents.jsonl")
    queue_index = _require_pair_uniqueness(queue, "media_fetch_queue.jsonl")
    short_index = _require_pair_uniqueness(shortlist, "scientific_shortlist.jsonl")
    page_index = _require_pair_uniqueness(pages, "page_results.jsonl")
    enriched_index = _require_pair_uniqueness(enriched, "enriched_documents.jsonl")

    for pair, row in ranked_index.items():
        if pair[0] not in terms:
            raise ValueError(f"ranked pair {pair!r} has an unknown candidate")
        connector = row.get("connector")
        if connector not in ("openalex", "mediacloud"):
            raise ValueError(f"ranked pair {pair!r} has an unknown connector")
        source = documents.get(pair)
        if source is None:
            raise ValueError(f"ranked pair {pair!r} missing in result")
        if source.get("connector_id") != connector:
            raise ValueError(f"ranked pair {pair!r} connector diverges from result")
        score = row.get("score")
        if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 100:
            raise ValueError(f"ranked pair {pair!r} has an invalid score")
        if row.get("relevance_class") not in ("strong", "weak", "none"):
            raise ValueError(f"ranked pair {pair!r} has an unknown class")
        if not isinstance(row.get("matched_term"), str):
            raise ValueError(f"ranked pair {pair!r} has no matched term")
        matched = row.get("matched_tokens")
        if not isinstance(matched, list) or any(not isinstance(item, str) for item in matched):
            raise ValueError(f"ranked pair {pair!r} has invalid matched tokens")
        search_id = row.get("search_id")
        expected_class = "scientific" if connector == "openalex" else "industry"
        if search_classes.get(search_id) != expected_class:
            raise ValueError(f"ranked pair {pair!r} has a mismatched search class")

    for pair, row in queue_index.items():
        if pair[0] not in terms:
            raise ValueError(f"fetch queue pair {pair!r} has an unknown candidate")
        if row.get("connector") != "mediacloud":
            raise ValueError(f"fetch queue pair {pair!r} is not mediacloud")
        score = row.get("score")
        if not isinstance(score, int) or isinstance(score, bool):
            raise ValueError(f"fetch queue pair {pair!r} has no int score")
        if row.get("relevance_class") not in ("strong", "weak", "none"):
            raise ValueError(f"fetch queue pair {pair!r} has unknown class")
        if not isinstance(row.get("matched_term"), str):
            raise ValueError(f"fetch queue pair {pair!r} has no matched term")
        _require_plain_rank(row, "fetch queue", pair)
        rank = row["selection_rank"]
        if not 1 <= rank <= 4:
            raise ValueError(f"fetch queue pair {pair!r} has a bad selection rank")
        search_id = row.get("search_id")
        if search_classes.get(search_id) != "industry":
            raise ValueError(f"fetch queue pair {pair!r} has a non-industry search")
        source = documents.get(pair)
        if source is None:
            raise ValueError(f"fetch queue pair {pair!r} missing in result")
        if source.get("connector_id") != "mediacloud":
            raise ValueError(
                f"fetch queue pair {pair!r} source document is not mediacloud"
            )
    _require_rank_sequence(queue_index, "media_fetch_queue.jsonl", 4)

    for pair, row in short_index.items():
        if pair[0] not in terms:
            raise ValueError(f"scientific shortlist pair {pair!r} has an unknown candidate")
        if row.get("connector") != "openalex":
            raise ValueError(f"scientific shortlist pair {pair!r} is not openalex")
        score = row.get("score")
        if not isinstance(score, int) or isinstance(score, bool):
            raise ValueError(f"scientific shortlist pair {pair!r} has no int score")
        if row.get("relevance_class") not in ("strong", "weak"):
            raise ValueError(f"scientific shortlist pair {pair!r} is not strong/weak")
        reasons = row.get("reasons")
        if not isinstance(reasons, list) or not reasons or not isinstance(reasons[0], str):
            raise ValueError(f"scientific shortlist pair {pair!r} has no reason")
        if not isinstance(row.get("matched_term"), str):
            raise ValueError(f"scientific shortlist pair {pair!r} has no matched term")
        _require_plain_rank(row, "scientific shortlist", pair)
        source = documents.get(pair)
        if source is None:
            raise ValueError(f"scientific shortlist pair {pair!r} missing in result")
        if source.get("connector_id") != "openalex":
            raise ValueError(
                f"scientific shortlist pair {pair!r} source document is not openalex"
            )
        if not isinstance(source.get("excerpt"), str) or not source["excerpt"].strip():
            raise ValueError(f"scientific pair {pair!r} has no excerpt in result")
        _check_date(source.get("published_at"), f"scientific pair {pair!r}")
        search_id = row.get("search_id")
        if search_classes.get(search_id) != "scientific":
            raise ValueError(f"scientific pair {pair!r} has a non-scientific search")
        if pair[0] not in terms:
            raise ValueError(f"scientific pair {pair!r} has no reviewed terms")
    _require_rank_sequence(short_index, "scientific_shortlist.jsonl", SCIENTIFIC_PER_CANDIDATE)

    for pair in queue_index:
        if pair not in page_index:
            raise ValueError(f"page result missing for queue pair {pair!r}")
    for pair, row in page_index.items():
        queue_row = queue_index.get(pair)
        if queue_row is None:
            raise ValueError(f"page result pair {pair!r} missing in fetch queue")
        for name in ("url", "selection_rank"):
            if row.get(name) != queue_row.get(name):
                raise ValueError(f"page result pair {pair!r} {name} diverges from queue")
        if row.get("status") not in ("success", "failed"):
            raise ValueError(f"page result pair {pair!r} has unknown status")

    for pair, row in enriched_index.items():
        page = page_index.get(pair)
        if page is None:
            raise ValueError(f"enriched pair {pair!r} missing in page results")
        if page.get("status") != "success":
            raise ValueError(f"enriched pair {pair!r} comes from a failed page")
        queue_row = queue_index.get(pair)
        if queue_row is None:
            raise ValueError(f"enriched pair {pair!r} missing in fetch queue")
        for name in ("url", "search_id", "request_id", "selection_rank"):
            if row.get(name) != queue_row.get(name):
                raise ValueError(f"enriched pair {pair!r} {name} diverges from queue")
        if row.get("content_status") != page.get("content_status"):
            raise ValueError(f"enriched pair {pair!r} content status diverges from page")
        if row.get("candidate_id") not in terms:
            raise ValueError(f"enriched pair {pair!r} has no reviewed terms")
        excerpt = row.get("excerpt")
        evidence = row.get("evidence_text_available")
        if page.get("content_status") == "article_text":
            if not isinstance(excerpt, str) or not excerpt.strip():
                raise ValueError(f"article_text pair {pair!r} has no excerpt")
            if evidence is not True:
                raise ValueError(f"article_text pair {pair!r} must carry evidence true")
        else:
            if evidence is not False:
                raise ValueError(f"non-article pair {pair!r} must carry evidence false")
        _check_date(row.get("published_at"), f"enriched pair {pair!r}")
    for pair, page in page_index.items():
        if page.get("status") == "failed" and pair in enriched_index:
            raise ValueError(f"failed page pair {pair!r} present in enriched documents")
        if page.get("status") == "success" and pair not in enriched_index:
            raise ValueError(f"successful page pair {pair!r} missing in enriched documents")
    return ranked_index, queue_index, short_index, page_index


def _media_eligible(row: Mapping[str, Any]) -> bool:
    """Policy C admission: every strong plus narrow score-40 weak rows.

    Strong article_text is always admitted. A weak row is admitted only with
    final_score exactly 40 and a plain-int article-only window inside
    [33, 96]. Shallow partials, title-only completions and wide windows stay
    in media_reranked but never enter the evidence input.
    """

    if row.get("final_class") == "strong":
        return True
    if row.get("final_class") != "weak" or row.get("final_score") != WEAK_ADMITTED_SCORE:
        return False
    width = row.get("min_window_width")
    if not isinstance(width, int) or isinstance(width, bool):
        return False
    return WEAK_MIN_WINDOW_TOKENS <= width <= WEAK_MAX_WINDOW_TOKENS


def _scientific_eligible(
    row: Mapping[str, Any], source: Mapping[str, Any]
) -> tuple[bool, int, int | None]:
    """Admit strong science and weak matches focused in title or excerpt.

    Weak hits must match at least two reviewed-term tokens. Two title matches
    are enough despite a long query; otherwise score must be at least 30 and
    all matched tokens must fit in a 32-token excerpt window.
    """

    matched = row.get("matched_tokens")
    if not isinstance(matched, list) or any(not isinstance(item, str) for item in matched):
        return False, 0, None
    matched_set = frozenset(_normalize_text(item) for item in matched if _normalize_text(item))
    title_tokens = _normalize_text(str(source.get("title") or "")).split()
    excerpt_tokens = _normalize_text(str(source.get("excerpt") or "")).split()
    title_matches = len(matched_set & frozenset(title_tokens))
    excerpt_window = _min_window_width(excerpt_tokens, matched_set) if matched_set else None
    if row.get("relevance_class") == "strong":
        return True, title_matches, excerpt_window
    score = row.get("score")
    eligible = (
        row.get("relevance_class") == "weak"
        and isinstance(score, int)
        and not isinstance(score, bool)
        and len(matched_set) >= SCIENTIFIC_WEAK_MIN_MATCHED_TOKENS
        and (
            title_matches >= SCIENTIFIC_WEAK_MIN_MATCHED_TOKENS
            or (
                score >= SCIENTIFIC_WEAK_MIN_SCORE
                and excerpt_window is not None
                and excerpt_window <= SCIENTIFIC_WEAK_MAX_EXCERPT_WINDOW_TOKENS
            )
        )
    )
    return eligible, title_matches, excerpt_window


def _exclusion_bucket(row: Mapping[str, Any]) -> str:
    """Classify a non-eligible article row for coverage accounting."""

    if row.get("final_class") == "weak" and row.get("final_score") == WEAK_ADMITTED_SCORE:
        width = row.get("min_window_width")
        if isinstance(width, int) and not isinstance(width, bool):
            return "wide_window"
        return "no_article_window"
    return "low_overlap"


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


def _invert_date(published: str) -> str:
    return "".join(chr(0x10FFFF - ord(char)) for char in published)


def _media_order_key(row: dict[str, Any]) -> tuple[int, int, int, str, str, str]:
    published = row.get("published_at")
    return (
        -row["final_score"],
        _trust_rank(row.get("trust_tier") or "unknown"),
        1 if published is None else 0,
        _invert_date(published) if published is not None else "",
        row["identity"],
        row["document_id"],
    )


def _scientific_order_key(row: dict[str, Any]) -> tuple[int, int, int, str, str, str]:
    published = row.get("published_at")
    return (
        -row["score"],
        _trust_rank(row.get("trust_tier") or "unknown"),
        1 if published is None else 0,
        _invert_date(published) if published is not None else "",
        row["document_identity"],
        row["document_id"],
    )


def _select_scientific(
    candidate_id: str,
    ranked_index: Mapping[tuple[str, str], dict[str, Any]],
    documents: Mapping[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
    """Filter all ranked OpenAlex rows, then choose up to four diversely."""

    eligible: list[dict[str, Any]] = []
    exclusions = {"low_score": 0, "weak_context": 0}
    available = 0
    for pair in sorted(ranked_index):
        if pair[0] != candidate_id:
            continue
        row = ranked_index[pair]
        if row.get("connector") != "openalex":
            continue
        source = documents[pair]
        if not isinstance(source.get("excerpt"), str) or not source["excerpt"].strip():
            continue
        if row.get("relevance_class") not in ("strong", "weak"):
            continue
        available += 1
        admitted, title_matches, excerpt_window = _scientific_eligible(row, source)
        if not admitted:
            if (
                row.get("relevance_class") == "weak"
                and row.get("score", 0) < SCIENTIFIC_WEAK_MIN_SCORE
            ):
                exclusions["low_score"] += 1
            else:
                exclusions["weak_context"] += 1
            continue
        eligible.append({
            **row,
            "reason": (row.get("reasons") or [None])[0],
            "scientific_title_matches": title_matches,
            "scientific_excerpt_window": excerpt_window,
        })
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in eligible:
        buckets.setdefault((row["relevance_class"], row["time_window"]), []).append(row)
    for bucket in buckets.values():
        bucket.sort(key=_scientific_order_key)
    selected: list[dict[str, Any]] = []
    used_origins: set[str] = set()
    for relevance_class in ("strong", "weak"):
        while len(selected) < SCIENTIFIC_PER_CANDIDATE:
            progressed = False
            for window in _WINDOW_CYCLE:
                if len(selected) >= SCIENTIFIC_PER_CANDIDATE:
                    break
                bucket = buckets.get((relevance_class, window))
                if not bucket:
                    continue
                pick = next(
                    (
                        index for index, row in enumerate(bucket)
                        if row["origin_id"] not in used_origins
                    ),
                    0,
                )
                row = bucket.pop(pick)
                used_origins.add(row["origin_id"])
                selected.append(row)
                progressed = True
            if not progressed:
                break
    return selected, available, exclusions


def build_evidence_input_plan(
    plan_dir: str | Path,
    result_dir: str | Path,
    relevance_dir: str | Path,
    media_dir: str | Path,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Validate the full chain and render deterministic evidence input bytes.

    Raises before touching the output directory, so failures never leave
    partial files behind.
    """

    plan_manifest, plan_data, plan_digests = _load_plan(Path(plan_dir))
    del plan_manifest
    terms = plan_data["terms"]
    search_classes = plan_data["search_classes"]
    bundle_id = _bundle_of_plan(plan_dir)
    _result_manifest, documents, result_digests = _load_result(
        Path(result_dir), bundle_id, plan_digests["plan.json"]
    )
    return _assemble(
        bundle_id,
        terms, search_classes, documents, result_digests, plan_digests,
        Path(relevance_dir), Path(media_dir),
    )


def _bundle_of_plan(plan_dir: str | Path) -> str:
    manifest = _read_json(Path(plan_dir) / "manifest.json", "enrichment plan manifest")
    if not isinstance(manifest, dict) or not manifest.get("bundle_id"):
        raise ValueError("enrichment plan manifest bundle_id is required")
    return manifest["bundle_id"]


def _assemble(
    bundle_id: str,
    terms: dict[str, tuple[str, ...]],
    search_classes: dict[str, str],
    documents: dict[tuple[str, str], dict[str, Any]],
    result_digests: dict[str, Any],
    plan_digests: dict[str, Any],
    relevance_path: Path,
    media_path: Path,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    _rel_manifest, ranked, queue, shortlist, relevance_digests = _load_relevance(
        relevance_path, bundle_id, plan_digests, result_digests
    )
    _media_manifest, pages, enriched, media_digests = _load_media(
        media_path, bundle_id, relevance_digests, result_digests
    )
    ranked_index, queue_index, short_index, page_index = _cross_check(
        terms, search_classes, documents, ranked, queue, shortlist, pages, enriched
    )
    enriched_index = _require_pair_uniqueness(enriched, "enriched_documents.jsonl")

    reranked: list[dict[str, Any]] = []
    articles_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for pair in sorted(page_index):
        page = page_index[pair]
        queue_row = queue_index[pair]
        if page.get("content_status") == "article_text":
            enriched_row = enriched_index.get(pair)
            if enriched_row is None:
                raise ValueError(f"article_text pair {pair!r} missing in enriched documents")
            excerpt = enriched_row.get("excerpt")
            if not isinstance(excerpt, str) or not excerpt.strip():
                raise ValueError(f"article_text pair {pair!r} has no excerpt")
            title = enriched_row.get("title")
            if not isinstance(title, str):
                raise ValueError(f"article_text pair {pair!r} has no title")
            score, final_class, term, reason, matched, term_tokens, width = score_media_document(
                terms[pair[0]], title, excerpt
            )
            reranked.append({
                "candidate_id": pair[0],
                "document_id": pair[1],
                "connector": "mediacloud",
                "search_id": page.get("search_id"),
                "request_id": page.get("request_id"),
                "selection_rank": page.get("selection_rank"),
                "preliminary_score": queue_row.get("score"),
                "preliminary_class": queue_row.get("relevance_class"),
                "final_score": score,
                "final_class": final_class,
                "matched_term": term,
                "reason": reason,
                "matched_tokens": list(matched),
                "term_tokens": list(term_tokens),
                "min_window_width": width,
                "content_sha256": enriched_row.get("content_sha256"),
                "excerpt_source": enriched_row.get("extraction_method"),
                "excerpt_truncated": enriched_row.get("excerpt_truncated"),
                "article_text_length": len(excerpt),
                "url": page.get("url"),
                "origin_id": enriched_row.get("origin_id"),
                "published_at": enriched_row.get("published_at"),
                "time_window": time_window(enriched_row.get("published_at")),
            })
            articles_by_candidate.setdefault(pair[0], []).append(reranked[-1])
        else:
            status = page.get("content_status")
            if status == "meta_description":
                exclusion = "meta_description_not_evidence"
            elif status == "title_only":
                exclusion = "title_only_not_evidence"
            else:
                exclusion = "fetch_failed_not_evidence"
            reranked.append({
                "candidate_id": pair[0],
                "document_id": pair[1],
                "connector": "mediacloud",
                "search_id": page.get("search_id"),
                "request_id": page.get("request_id"),
                "selection_rank": page.get("selection_rank"),
                "preliminary_score": queue_row.get("score"),
                "preliminary_class": queue_row.get("relevance_class"),
                "final_score": None,
                "final_class": NOT_EVIDENCE_CLASS,
                "matched_term": None,
                "reason": None,
                "matched_tokens": [],
                "term_tokens": [],
                "min_window_width": None,
                "content_sha256": page.get("content_sha256"),
                "excerpt_source": page.get("excerpt_source"),
                "excerpt_truncated": page.get("excerpt_truncated"),
                "article_text_length": 0,
                "url": page.get("url"),
                "origin_id": queue_row.get("origin_id"),
                "published_at": queue_row.get("published_at"),
                "time_window": time_window(queue_row.get("published_at")),
                "exclusion_reason": exclusion,
            })
    reranked.sort(key=lambda row: (row["candidate_id"], row["selection_rank"], row["document_id"]))

    for candidate_id, rows in articles_by_candidate.items():
        for row in rows:
            source = documents[(candidate_id, row["document_id"])]
            row["identity"] = document_identity(source)
            row["trust_tier"] = source.get("trust_tier") or "unknown"
            row["publisher"] = source.get("publisher")

    evidence_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    totals = {
        "candidates": 0,
        "scientific_available": 0,
        "scientific_eligible": 0,
        "scientific_excluded_low_score": 0,
        "scientific_excluded_weak_context": 0,
        "scientific_selected": 0,
        "media_article_text": 0,
        "media_strong": 0,
        "media_weak": 0,
        "media_none": 0,
        "media_not_evidence_text": 0,
        "media_selected": 0,
        "evidence_input_documents": 0,
        "candidates_without_final_input": 0,
    }
    for candidate_id in sorted(terms):
        science, scientific_available, scientific_exclusions = _select_scientific(
            candidate_id, ranked_index, documents
        )
        capacity = media_capacity(len(science))
        media_pool = articles_by_candidate.get(candidate_id, [])
        strong = [row for row in media_pool if row["final_class"] == "strong"]
        weak = [
            row for row in media_pool
            if row["final_class"] == "weak" and _media_eligible(row)
        ]
        buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in strong + weak:
            buckets.setdefault((row["final_class"], row["time_window"]), []).append(row)
        for bucket in buckets.values():
            bucket.sort(key=_media_order_key)
        used_origins: set[str] = set()
        used_publishers: set[str] = set()
        selected_media: list[dict[str, Any]] = []
        for final_class in ("strong", "weak"):
            while len(selected_media) < capacity:
                progressed = False
                for window in _WINDOW_CYCLE:
                    if len(selected_media) >= capacity:
                        break
                    bucket = buckets.get((final_class, window))
                    if not bucket:
                        continue
                    row = bucket.pop(_diverse_pick(bucket, used_origins, used_publishers))
                    used_origins.add(row["origin_id"])
                    used_publishers.add(row["publisher"] or "")
                    selected_media.append(row)
                    progressed = True
                if not progressed:
                    break
        ordered: list[dict[str, Any]] = []
        for index in range(max(len(science), len(selected_media))):
            if index < len(science):
                ordered.append(("scientific", science[index]))
            if index < len(selected_media):
                ordered.append(("industry", selected_media[index]))
        ordered = ordered[:FINAL_PER_CANDIDATE]
        source_counters = {"scientific": 0, "industry": 0}
        for position, (source_class, row) in enumerate(ordered, start=1):
            source_counters[source_class] += 1
            if source_class == "scientific":
                source = documents[(candidate_id, row["document_id"])]
                evidence_rows.append({
                    "candidate_id": candidate_id,
                    "document_id": row["document_id"],
                    "connector": "openalex",
                    "source_class": "scientific",
                    "final_rank": position,
                    "source_rank": source_counters[source_class],
                    "title": source.get("title"),
                    "excerpt": source.get("excerpt"),
                    "url": source.get("url"),
                    "canonical_url": source.get("canonical_url"),
                    "origin_id": source.get("origin_id"),
                    "origin_method": source.get("origin_method"),
                    "publisher": source.get("publisher"),
                    "published_at": row["published_at"],
                    "trust_tier": source.get("trust_tier"),
                    "search_id": row.get("search_id"),
                    "request_id": source.get("request_id"),
                    "snapshot_id": source.get("snapshot_id"),
                    "relevance_score": row["score"],
                    "relevance_class": row["relevance_class"],
                    "relevance_reason": row["reason"],
                    "matched_term": row["matched_term"],
                    "matched_tokens": row["matched_tokens"],
                    "content_sha256": None,
                    "excerpt_truncated": False,
                    "evidence_text_available": True,
                })
            else:
                enriched_row = enriched_index[(candidate_id, row["document_id"])]
                evidence_rows.append({
                    "candidate_id": candidate_id,
                    "document_id": row["document_id"],
                    "connector": "mediacloud",
                    "source_class": "industry",
                    "final_rank": position,
                    "source_rank": source_counters[source_class],
                    "title": enriched_row.get("title"),
                    "excerpt": enriched_row.get("excerpt"),
                    "url": row["url"],
                    "canonical_url": enriched_row.get("canonical_url"),
                    "origin_id": row["origin_id"],
                    "origin_method": enriched_row.get("origin_method"),
                    "publisher": enriched_row.get("publisher"),
                    "published_at": row["published_at"],
                    "trust_tier": enriched_row.get("trust_tier"),
                    "search_id": row["search_id"],
                    "request_id": row["request_id"],
                    "snapshot_id": enriched_row.get("snapshot_id"),
                    "relevance_score": row["final_score"],
                    "relevance_class": row["final_class"],
                    "relevance_reason": row["reason"],
                    "matched_term": row["matched_term"],
                    "matched_tokens": row["matched_tokens"],
                    "content_sha256": row["content_sha256"],
                    "excerpt_truncated": enriched_row.get("excerpt_truncated"),
                    "evidence_text_available": True,
                })
        queue_rows = [row for row in queue if row["candidate_id"] == candidate_id]
        page_rows = [page_index[pair] for pair in sorted(page_index) if pair[0] == candidate_id]
        windows = {"previous": 0, "recent": 0, "unknown_date": 0}
        for row in ordered:
            windows[row[1]["time_window"]] += 1
        classes = sorted({source for source, _ in ordered})
        candidate_articles = [
            row for row in reranked
            if row["candidate_id"] == candidate_id and row["final_class"] in _VALID_CLASSES
        ]
        excluded = [
            _exclusion_bucket(row) for row in candidate_articles if not _media_eligible(row)
        ]
        reasons: list[str] = []
        if not science:
            reasons.append("no_scientific_available")
        if not [row for row in candidate_articles if row["final_class"] in ("strong", "weak")]:
            reasons.append("no_media_article_text")
        if not ordered:
            reasons.append("no_final_input")
        coverage_rows.append({
            "candidate_id": candidate_id,
            "scientific_available": scientific_available,
            "scientific_eligible": len(science),
            "scientific_excluded_low_score": scientific_exclusions["low_score"],
            "scientific_excluded_weak_context": scientific_exclusions["weak_context"],
            "scientific_selected": len(science),
            "media_planned": len(queue_rows),
            "media_completed": sum(1 for row in page_rows if row.get("status") == "success"),
            "media_failed": sum(1 for row in page_rows if row.get("status") != "success"),
            "media_article_text": len(candidate_articles),
            "media_strong": sum(1 for row in candidate_articles if row["final_class"] == "strong"),
            "media_weak": sum(1 for row in candidate_articles if row["final_class"] == "weak"),
            "media_none": sum(1 for row in candidate_articles if row["final_class"] == "none"),
            "media_eligible": sum(1 for row in candidate_articles if _media_eligible(row)),
            "media_excluded_low_overlap": sum(
                1 for bucket in excluded if bucket == "low_overlap"
            ),
            "media_excluded_wide_window": sum(
                1 for bucket in excluded if bucket == "wide_window"
            ),
            "media_excluded_no_article_window": sum(
                1 for bucket in excluded if bucket == "no_article_window"
            ),
            "media_selected": len(selected_media),
            "final_evidence_input_count": len(ordered),
            "window_previous": windows["previous"],
            "window_recent": windows["recent"],
            "window_unknown_date": windows["unknown_date"],
            "source_classes_present": classes,
            "has_evidence_input": bool(ordered),
            "empty_reasons": reasons,
        })
        totals["candidates"] += 1
        totals["scientific_available"] += scientific_available
        totals["scientific_eligible"] += len(science)
        totals["scientific_excluded_low_score"] += scientific_exclusions["low_score"]
        totals["scientific_excluded_weak_context"] += scientific_exclusions["weak_context"]
        totals["scientific_selected"] += len(science)
        totals["media_selected"] += len(selected_media)
        totals["evidence_input_documents"] += len(ordered)
        if not ordered:
            totals["candidates_without_final_input"] += 1
    for row in reranked:
        if row["final_class"] == "strong":
            totals["media_strong"] += 1
        elif row["final_class"] == "weak":
            totals["media_weak"] += 1
        elif row["final_class"] == "none":
            totals["media_none"] += 1
        else:
            totals["media_not_evidence_text"] += 1
    totals["media_article_text"] = (
        totals["media_strong"] + totals["media_weak"] + totals["media_none"]
    )

    files = {
        RERANKED_FILENAME: _render_jsonl(reranked),
        EVIDENCE_INPUT_FILENAME: _render_jsonl(evidence_rows),
        COVERAGE_FILENAME: _render_jsonl(coverage_rows),
    }
    manifest = {
        "schema_version": LABELING_EVIDENCE_INPUT_PLAN_VERSION,
        "cutoff_date": CUTOFF_ISO,
        "bundle_id": bundle_id,
        "inputs": {
            "plan_manifest": plan_digests["manifest.json"],
            "plan_file": plan_digests["plan.json"],
            "result_manifest": result_digests["manifest.json"],
            "result_files": {
                name: result_digests[name]
                for name in ("coverage.jsonl", "documents.jsonl", "request_results.jsonl")
            },
            "relevance_manifest": relevance_digests["manifest.json"],
            "relevance_files": {
                name: relevance_digests[name]
                for name in (
                    "ranked_documents.jsonl",
                    "scientific_shortlist.jsonl",
                    "media_fetch_queue.jsonl",
                    "coverage.jsonl",
                )
            },
            "media_manifest": media_digests["manifest.json"],
            "media_files": {
                name: media_digests[name]
                for name in ("page_results.jsonl", "enriched_documents.jsonl", "coverage.jsonl")
            },
        },
        "score_policy": {
            "description": (
                "Full-text retrieval rerank of article_text pages against reviewed "
                "search_terms. Normalization: Unicode NFKC, casefold, "
                "non-alphanumeric runs to space, collapsed spaces, whole-token "
                "matches only. Not evidence, not a candidate model score."
            ),
            "hierarchy": [
                {"score": 100, "reason": REASON_EXACT_TITLE},
                {"score": 80, "reason": REASON_TITLE_TOKENS},
                {"score": 70, "reason": REASON_EXACT_ARTICLE},
                {"score": 60, "reason": REASON_WINDOW},
                {"score": "round(40 * matched_unique / term_unique)", "reason": REASON_PARTIAL},
                {"score": 0, "reason": REASON_NO_OVERLAP},
            ],
            "classes": {"strong": "score >= 60", "weak": "20..59", "none": "score < 20"},
            "sliding_window_tokens": MEDIA_WINDOW_TOKENS,
            "tie_break": "maximum over terms; ties keep the first term in plan order",
            "partial_window_note": (
                "score-40 partial rows keep the measured article-only window "
                "even above 32 tokens; null when a token lives only in the title"
            ),
        },
        "selection_policy": {
            "strong_allowed": True,
            "scientific_source": "all ranked OpenAlex documents, before top-4",
            "scientific_weak_excerpt_min_score": SCIENTIFIC_WEAK_MIN_SCORE,
            "scientific_weak_min_matched_tokens": SCIENTIFIC_WEAK_MIN_MATCHED_TOKENS,
            "scientific_weak_max_excerpt_window_tokens": (
                SCIENTIFIC_WEAK_MAX_EXCERPT_WINDOW_TOKENS
            ),
            "scientific_weak_title_matches": SCIENTIFIC_WEAK_MIN_MATCHED_TOKENS,
            "weak_score": WEAK_ADMITTED_SCORE,
            "weak_min_window_tokens": WEAK_MIN_WINDOW_TOKENS,
            "weak_max_window_tokens": WEAK_MAX_WINDOW_TOKENS,
            "weak_without_article_window_allowed": False,
            "media_article_text_only": True,
        },
        "limits": {
            "scientific_per_candidate": SCIENTIFIC_PER_CANDIDATE,
            "media_capacity_formula": "min(4, max(2, 6 - scientific_count))",
            "final_per_candidate": FINAL_PER_CANDIDATE,
        },
        "totals": totals,
        "outputs": {
            name: _digest(payload) for name, payload in files.items()
        },
    }
    files[EVIDENCE_INPUT_MANIFEST_FILENAME] = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return files, {"bundle_id": bundle_id, "candidate_count": totals["candidates"]}


def export_evidence_input_plan(
    *,
    plan_dir: str | Path,
    result_dir: str | Path,
    relevance_dir: str | Path,
    media_dir: str | Path,
    output_dir: str | Path,
) -> LabelingEvidenceInputPlanPaths:
    """Validate the full chain and atomically publish the evidence input plan."""

    files, _provenance = build_evidence_input_plan(
        plan_dir, result_dir, relevance_dir, media_dir
    )
    paths = publish_artifact_bundle(files, output_dir)
    return LabelingEvidenceInputPlanPaths(
        manifest=paths[EVIDENCE_INPUT_MANIFEST_FILENAME],
        media_reranked=paths[RERANKED_FILENAME],
        evidence_input_documents=paths[EVIDENCE_INPUT_FILENAME],
        coverage=paths[COVERAGE_FILENAME],
    )
