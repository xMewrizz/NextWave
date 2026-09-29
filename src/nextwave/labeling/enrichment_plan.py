"""Deterministic offline plan for candidate evidence enrichment.

The command validates one supported candidate bundle and emits an
immutable per-candidate search plan with executable per-term requests for
the OpenAlex and Media Cloud connectors. No network calls happen here and
no search is executed: a future executor will report a successful response
(including zero) as covered and any failure, timeout, 429 or skip as
unknown. GDELT does not take part in labeling enrichment.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from ..datasets.artifacts import publish_artifact_bundle
from ..datasets.contracts import (
    MANIFEST_SCHEMA_VERSION as ORGANIZER_MANIFEST_VERSION,
)
from ..datasets.contracts import (
    ORGANIZER_SCOPE_KEYS,
    POSITIVE_SCHEMA_VERSION,
    IdentityStatus,
    PositiveCandidateRecord,
)
from ..discovery.run_store import (
    DISCOVERY_RUN_MANIFEST_SCHEMA_VERSION,
    assert_run_analysis_eligible,
    load_discovery_run,
)
from .contracts import (
    LABELING_CUTOFF_DATE,
    NEGATIVE_CANDIDATE_SCHEMA_VERSION,
    NegativeCandidateRecord,
)
from .export import LABELING_EXPORT_MANIFEST_VERSION

LABELING_ENRICHMENT_PLAN_VERSION = "labeling-enrichment-plan-v2"
LABELING_TARGET_ENRICHMENT_PLAN_VERSION = "labeling-target-enrichment-plan-v1"
TARGET_CANDIDATES_VERSION = "labeling-target-candidates-v1"
ENRICHMENT_PLAN_FILENAME = "plan.json"
ENRICHMENT_MANIFEST_FILENAME = "manifest.json"
NEGATIVE_CANDIDATES_FILENAME = "negative_candidates.jsonl"
POSITIVE_CANDIDATES_FILENAME = "positive_candidates.jsonl"
ENRICHMENT_SEARCH_TERMS_FILENAME = "search_terms.json"
ENRICHMENT_SEARCH_TERMS_VERSION = "labeling-enrichment-search-terms-v1"

_HISTORY_DAYS = 730
_RECENT_DAYS = 365
_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")
_CANDIDATE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")

LANGUAGES: tuple[str, ...] = ("en", "ru")
OPENALEX_RETRIEVAL_POLICY: dict[str, Any] = {
    "channel": "text",
    "per_page": 20,
    "max_pages_per_request": 1,
    "max_attempts": 3,
    "timeout_seconds": 20,
}
MEDIACLOUD_COLLECTION_IDS: tuple[int, ...] = (34412234, 34412118)
MEDIACLOUD_RETRIEVAL_POLICY: dict[str, Any] = {
    "page_size": 40,
    "max_pages_per_request": 1,
    "max_attempts": 3,
    "timeout_seconds": 60,
    "min_interval_seconds": 30,
}
COMPLETION_RULE = "all_planned_requests_successful"

_TARGET_DOMAINS = {
    "Edge",
    "Защита ИИ",
    "Индустриальный ИИ",
    "Инфраструктура ИИ",
    "Роботы",
    "Финтех",
}
_TARGET_CANDIDATE_KEYS = {
    "candidate_id",
    "canonical_name",
    "aliases",
    "domain",
    "analysis_scope_key",
    "source_query",
}


@dataclass(frozen=True, slots=True)
class LabelingEnrichmentPlanPaths:
    """Paths of one successfully published enrichment plan."""

    plan: Path
    manifest: Path


@dataclass(frozen=True, slots=True)
class _EnrichmentCandidateRecord:
    """Leak-safe candidate identity shared by positive and negative bundles."""

    candidate_id: str
    canonical_name: str
    aliases: tuple[str, ...]
    group_id: str
    source_query: str
    domain: str
    analysis_scope_key: str
    cutoff_date: date

    def __post_init__(self) -> None:
        if _CANDIDATE_ID.fullmatch(self.candidate_id) is None:
            raise ValueError("candidate_id must be a safe stable identifier")
        if _CANDIDATE_ID.fullmatch(self.group_id) is None:
            raise ValueError("group_id must be a safe stable identifier")
        for field, value in (
            ("canonical_name", self.canonical_name),
            ("source_query", self.source_query),
            ("domain", self.domain),
            ("analysis_scope_key", self.analysis_scope_key),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-blank string")
        normalized_aliases = [alias.strip().casefold() for alias in self.aliases]
        if any(not alias for alias in normalized_aliases):
            raise ValueError("aliases must not contain blank values")
        if len(set(normalized_aliases)) != len(normalized_aliases):
            raise ValueError("aliases must be unique")
        if self.canonical_name.strip().casefold() in normalized_aliases:
            raise ValueError("aliases must not repeat canonical_name")


def _clean_term(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip()


def _search_terms(canonical_name: str, aliases: tuple[str, ...]) -> list[str]:
    """Canonical first, then aliases in order; NFKC-cleaned, casefold-deduped.

    The first spelling wins; the canonical name is never dropped, no new
    aliases are invented, and identity stays bound to ``candidate_id``.
    """

    terms: list[str] = []
    seen: set[str] = set()
    for raw in (canonical_name, *aliases):
        cleaned = _clean_term(raw) if isinstance(raw, str) else ""
        if not cleaned or cleaned.casefold() in seen:
            continue
        seen.add(cleaned.casefold())
        terms.append(cleaned)
    if not terms:
        raise ValueError("candidate has no usable search terms")
    return terms


def _load_positive_search_terms(
    path: Path,
    *,
    positive_candidates_sha256: str,
) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    """Load a reviewed retrieval overlay without changing candidate identity."""

    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read positive search terms: {path}") from error
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("positive search terms must be valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("positive search terms must be a JSON object")
    if payload.get("schema_version") != ENRICHMENT_SEARCH_TERMS_VERSION:
        raise ValueError(
            "positive search terms schema "
            f"{payload.get('schema_version')!r} does not match "
            f"{ENRICHMENT_SEARCH_TERMS_VERSION!r}"
        )
    if payload.get("positive_candidates_sha256") != positive_candidates_sha256:
        raise ValueError("positive search terms do not match positive_candidates.jsonl")
    entries = payload.get("candidates")
    if not isinstance(entries, list) or not entries:
        raise ValueError("positive search terms candidates must be a non-empty list")

    result: dict[str, tuple[str, ...]] = {}
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"positive search terms entry {index} must be an object")
        candidate_id = entry.get("candidate_id")
        if not isinstance(candidate_id, str) or _CANDIDATE_ID.fullmatch(candidate_id) is None:
            raise ValueError(
                f"positive search terms entry {index} candidate_id is invalid"
            )
        if candidate_id in result:
            raise ValueError(f"duplicate positive search terms for {candidate_id}")
        terms = entry.get("terms")
        if not isinstance(terms, list) or not 1 <= len(terms) <= 3:
            raise ValueError(
                f"positive search terms for {candidate_id} must contain 1 to 3 terms"
            )
        cleaned: list[str] = []
        seen: set[str] = set()
        for term in terms:
            value = _clean_term(term) if isinstance(term, str) else ""
            content_words = re.findall(r"[^\W_]+(?:-[^\W_]+)*", value, re.UNICODE)
            if not value or len(value) > 80 or not 1 <= len(content_words) <= 10:
                raise ValueError(
                    f"positive search term for {candidate_id} must be 1 to 10 words "
                    "and at most 80 characters"
                )
            key = value.casefold()
            if key in seen:
                raise ValueError(
                    f"positive search terms for {candidate_id} must be unique"
                )
            seen.add(key)
            cleaned.append(value)
        result[candidate_id] = tuple(cleaned)
    return result, _digest(raw)


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def _bundle_id(
    records: list[NegativeCandidateRecord | _EnrichmentCandidateRecord],
) -> str:
    """Semantic bundle identity: version, cutoff and canonical candidates."""

    lines = [
        json.dumps(
            {
                "analysis_scope_key": record.analysis_scope_key,
                "aliases": list(record.aliases),
                "candidate_id": record.candidate_id,
                "canonical_name": record.canonical_name,
                "cutoff_date": record.cutoff_date.isoformat(),
                "domain": record.domain,
                "group_id": record.group_id,
                "source_query": record.source_query,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for record in sorted(records, key=lambda item: item.candidate_id)
    ]
    cutoff_dates = {record.cutoff_date for record in records}
    if len(cutoff_dates) != 1:
        raise ValueError("enrichment candidates must share one cutoff date")
    cutoff_date = next(iter(cutoff_dates))
    digest = hashlib.sha256(
        "\x1f".join(
            (LABELING_ENRICHMENT_PLAN_VERSION, cutoff_date.isoformat(), *lines)
        ).encode("utf-8")
    ).hexdigest()
    return f"bundle-{digest[:16]}"


def _canonical_policy(policy: Mapping[str, Any]) -> str:
    return json.dumps(policy, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _request_id(
    candidate_id: str,
    connector: str,
    role: str,
    search_text: str,
    languages: list[str],
    window: tuple[str, str],
    retrieval_policy: Mapping[str, Any],
    collection_ids: tuple[int, ...] = (),
) -> str:
    return "request-" + _stable_id(
        LABELING_ENRICHMENT_PLAN_VERSION,
        candidate_id,
        connector,
        role,
        search_text,
        ",".join(languages),
        window[0],
        window[1],
        _canonical_policy(retrieval_policy),
        ",".join(str(item) for item in collection_ids),
    )


def _search_id(
    candidate_id: str,
    connector: str,
    role: str,
    request_ids: list[str],
    required_window: tuple[str, str],
    planned_window: tuple[str, str],
    publicity_window: tuple[str, str] | None,
    retrieval_policy: Mapping[str, Any],
    collection_ids: tuple[int, ...],
    completion_rule: str,
    coverage_policy: Mapping[str, Any],
) -> str:
    return "search-" + _stable_id(
        LABELING_ENRICHMENT_PLAN_VERSION,
        candidate_id,
        connector,
        role,
        "\x1f".join(request_ids),
        required_window[0],
        required_window[1],
        planned_window[0],
        planned_window[1],
        f"{publicity_window[0]}..{publicity_window[1]}" if publicity_window else "",
        _canonical_policy(retrieval_policy),
        ",".join(str(item) for item in collection_ids),
        completion_rule,
        _canonical_policy(coverage_policy),
    )


def _digest(data: bytes) -> dict[str, Any]:
    return {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _windows(cutoff_date: date = LABELING_CUTOFF_DATE) -> dict[str, Any]:
    history_from = (cutoff_date - timedelta(days=_HISTORY_DAYS)).isoformat()
    recent_from = (cutoff_date - timedelta(days=_RECENT_DAYS)).isoformat()
    cutoff = cutoff_date.isoformat()
    return {
        "cutoff_date": cutoff,
        "history_from": history_from,
        "recent_window_from": recent_from,
        "previous": {
            "from": history_from,
            "until": recent_from,
            "start_inclusive": True,
            "end_inclusive": False,
        },
        "recent": {
            "from": recent_from,
            "until": cutoff,
            "start_inclusive": True,
            "end_inclusive": True,
        },
    }


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _candidate_searches(
    candidate_id: str,
    terms: list[str],
    full_window: tuple[str, str],
    recent_window: tuple[str, str],
) -> list[dict[str, Any]]:
    window = {"from": full_window[0], "until": full_window[1]}
    policy = {
        "successful_response_including_zero": "covered",
        "failed_timeout_429_or_skipped": "unknown",
    }

    def requests(
        connector: str,
        role: str,
        retrieval_policy: Mapping[str, Any],
        language_groups: tuple[tuple[str, ...], ...],
        collection_ids: tuple[int, ...] = (),
    ) -> list[dict[str, Any]]:
        return [
            {
                "request_id": _request_id(
                    candidate_id,
                    connector,
                    role,
                    term,
                    list(languages),
                    full_window,
                    retrieval_policy,
                    collection_ids,
                ),
                "search_text": term,
                "languages": list(languages),
            }
            for term in terms
            for languages in language_groups
        ]

    scientific_policy = dict(OPENALEX_RETRIEVAL_POLICY)
    scientific_coverage = dict(policy)
    scientific_requests = requests(
        "openalex", "primary", OPENALEX_RETRIEVAL_POLICY, (("en",), ("ru",))
    )
    industry_policy = dict(MEDIACLOUD_RETRIEVAL_POLICY)
    industry_coverage = {**policy, "fallback": None}
    industry_requests = requests(
        "mediacloud",
        "primary",
        MEDIACLOUD_RETRIEVAL_POLICY,
        (("en", "ru"),),
        MEDIACLOUD_COLLECTION_IDS,
    )
    return [
        {
            "search_id": _search_id(
                candidate_id,
                "openalex",
                "primary",
                [item["request_id"] for item in scientific_requests],
                full_window,
                full_window,
                None,
                OPENALEX_RETRIEVAL_POLICY,
                (),
                COMPLETION_RULE,
                scientific_coverage,
            ),
            "source_class": "scientific",
            "connector": "openalex",
            "role": "primary",
            "purpose": "candidate_search",
            "languages": list(LANGUAGES),
            "retrieval_policy": scientific_policy,
            "completion_rule": COMPLETION_RULE,
            "required_window": dict(window),
            "planned_window": dict(window),
            "requests": scientific_requests,
            "coverage_policy": scientific_coverage,
        },
        {
            "search_id": _search_id(
                candidate_id,
                "mediacloud",
                "primary",
                [item["request_id"] for item in industry_requests],
                full_window,
                full_window,
                recent_window,
                MEDIACLOUD_RETRIEVAL_POLICY,
                MEDIACLOUD_COLLECTION_IDS,
                COMPLETION_RULE,
                industry_coverage,
            ),
            "source_class": "industry",
            "connector": "mediacloud",
            "role": "primary",
            "purpose": "candidate_search",
            "languages": list(LANGUAGES),
            "collection_ids": list(MEDIACLOUD_COLLECTION_IDS),
            "retrieval_policy": industry_policy,
            "completion_rule": COMPLETION_RULE,
            "required_window": dict(window),
            "planned_window": dict(window),
            "publicity_window": {"from": recent_window[0], "until": recent_window[1]},
            "requests": industry_requests,
            "coverage_policy": industry_coverage,
        },
    ]


def _require_text_field(payload: dict, field: str, lineno: int) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise ValueError(f"negative candidate line {lineno} field {field!r} must be a string")
    return value


def _parse_candidate(payload: Any, lineno: int) -> NegativeCandidateRecord:
    if not isinstance(payload, dict):
        raise ValueError(f"negative candidate line {lineno} must be a JSON object")
    if payload.get("schema_version") != NEGATIVE_CANDIDATE_SCHEMA_VERSION:
        raise ValueError(
            f"negative candidate line {lineno} schema "
            f"{payload.get('schema_version')!r} does not match "
            f"{NEGATIVE_CANDIDATE_SCHEMA_VERSION!r}"
        )
    aliases = payload.get("aliases")
    if not isinstance(aliases, list) or any(not isinstance(item, str) for item in aliases):
        raise ValueError(f"negative candidate line {lineno} aliases must be a list of strings")
    candidate_id = _require_text_field(payload, "candidate_id", lineno)
    canonical_name = _require_text_field(payload, "canonical_name", lineno)
    group_id = _require_text_field(payload, "group_id", lineno)
    source_query = _require_text_field(payload, "source_query", lineno)
    domain = _require_text_field(payload, "domain", lineno)
    analysis_scope_key = _require_text_field(payload, "analysis_scope_key", lineno)
    cutoff_raw = _require_text_field(payload, "cutoff_date", lineno)
    try:
        cutoff_date = date.fromisoformat(cutoff_raw)
    except ValueError as error:
        raise ValueError(
            f"negative candidate line {lineno} cutoff_date is not a date"
        ) from error
    try:
        return NegativeCandidateRecord(
            candidate_id=candidate_id,
            canonical_name=canonical_name,
            aliases=tuple(aliases),
            group_id=group_id,
            source_query=source_query,
            domain=domain,
            analysis_scope_key=analysis_scope_key,
            cutoff_date=cutoff_date,
        )
    except (ValueError, TypeError) as error:
        raise ValueError(f"negative candidate line {lineno} is invalid: {error}") from error


def _parse_positive_candidate(payload: Any, lineno: int) -> _EnrichmentCandidateRecord:
    """Map the leak-safe organizer candidate projection to the shared search shape."""

    if not isinstance(payload, dict):
        raise ValueError(f"positive candidate line {lineno} must be a JSON object")
    if payload.get("schema_version") != POSITIVE_SCHEMA_VERSION:
        raise ValueError(
            f"positive candidate line {lineno} schema "
            f"{payload.get('schema_version')!r} does not match "
            f"{POSITIVE_SCHEMA_VERSION!r}"
        )
    aliases = payload.get("aliases")
    if not isinstance(aliases, list) or any(not isinstance(item, str) for item in aliases):
        raise ValueError(f"positive candidate line {lineno} aliases must be a list of strings")
    candidate_id = payload.get("record_id")
    canonical_name = payload.get("canonical_name")
    group_id = payload.get("group_id")
    domain = payload.get("domain")
    analysis_scope_key = payload.get("analysis_scope_key")
    cutoff_raw = payload.get("cutoff_date")
    for field, value in (
        ("record_id", candidate_id),
        ("canonical_name", canonical_name),
        ("group_id", group_id),
        ("domain", domain),
        ("analysis_scope_key", analysis_scope_key),
        ("cutoff_date", cutoff_raw),
    ):
        if not isinstance(value, str):
            raise ValueError(
                f"positive candidate line {lineno} field {field!r} must be a string"
            )
    if (
        payload.get("label") != "weak_signal"
        or payload.get("target") != 1
        or payload.get("label_origin") != "organizer_confirmed"
    ):
        raise ValueError(
            f"positive candidate line {lineno} does not hold the organizer positive label"
        )
    try:
        cutoff_date = date.fromisoformat(cutoff_raw)
    except ValueError as error:
        raise ValueError(
            f"positive candidate line {lineno} cutoff_date is not a date"
        ) from error
    try:
        positive = PositiveCandidateRecord(
            record_id=candidate_id,
            source_row=payload.get("source_row"),
            canonical_name=canonical_name,
            domain=domain,
            analysis_scope_key=analysis_scope_key,
            aliases=tuple(aliases),
            group_id=group_id,
            identity_status=IdentityStatus(payload.get("identity_status")),
            cutoff_date=cutoff_date,
        )
        return _EnrichmentCandidateRecord(
            candidate_id=positive.record_id,
            canonical_name=positive.canonical_name,
            aliases=positive.aliases,
            group_id=positive.group_id,
            source_query=positive.canonical_name,
            domain=positive.domain,
            analysis_scope_key=positive.analysis_scope_key,
            cutoff_date=positive.cutoff_date,
        )
    except (ValueError, TypeError) as error:
        raise ValueError(f"positive candidate line {lineno} is invalid: {error}") from error


def _render_plan(
    records: list[NegativeCandidateRecord | _EnrichmentCandidateRecord],
    *,
    manifest_path: Path,
    candidates_bytes: bytes,
    candidates_input_key: str,
    search_term_overrides: Mapping[str, tuple[str, ...]] | None = None,
    extra_inputs: Mapping[str, dict[str, Any]] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    bundle_id = _bundle_id(records)
    cutoff_dates = {record.cutoff_date for record in records}
    if len(cutoff_dates) != 1:
        raise ValueError("enrichment candidates must share one cutoff date")
    windows = _windows(next(iter(cutoff_dates)))
    full_window = (windows["history_from"], windows["cutoff_date"])
    recent_window = (windows["recent_window_from"], windows["cutoff_date"])
    entries: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: item.candidate_id):
        terms = (
            list(search_term_overrides[record.candidate_id])
            if search_term_overrides is not None
            else _search_terms(record.canonical_name, record.aliases)
        )
        searches = _candidate_searches(record.candidate_id, terms, full_window, recent_window)
        entries.append(
            {
                "candidate_id": record.candidate_id,
                "canonical_name": record.canonical_name,
                "aliases": list(record.aliases),
                "domain": record.domain,
                "analysis_scope_key": record.analysis_scope_key,
                "cutoff_date": record.cutoff_date.isoformat(),
                "origin": {
                    "bundle_id": bundle_id,
                    "group_id": record.group_id,
                    "source_query": record.source_query,
                },
                "search_terms": terms,
                "searches": searches,
            }
        )

    totals = {
        "candidates": len(entries),
        "openalex_primary_requests": sum(
            len(search["requests"])
            for entry in entries
            for search in entry["searches"]
            if search["connector"] == "openalex"
        ),
        "mediacloud_primary_requests": sum(
            len(search["requests"])
            for entry in entries
            for search in entry["searches"]
            if search["connector"] == "mediacloud"
        ),
    }
    plan = {
        "schema_version": LABELING_ENRICHMENT_PLAN_VERSION,
        "cutoff_date": windows["cutoff_date"],
        "history_from": windows["history_from"],
        "recent_window_from": windows["recent_window_from"],
        "windows": {"previous": windows["previous"], "recent": windows["recent"]},
        "bundle": {"bundle_id": bundle_id, "candidate_count": len(entries)},
        "totals": totals,
        "candidates": entries,
    }
    plan_bytes = (
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return plan_bytes, {
        "bundle_id": bundle_id,
        "candidate_count": len(entries),
        "inputs": {
            "manifest": _digest(manifest_path.read_bytes()),
            candidates_input_key: _digest(candidates_bytes),
            **dict(extra_inputs or {}),
        },
        "plan_digest": _digest(plan_bytes),
    }


def _build_positive_enrichment_plan(
    bundle: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    search_terms_file: str | Path | None,
) -> tuple[bytes, dict[str, Any]]:
    if manifest.get("cutoff_date") != LABELING_CUTOFF_DATE.isoformat():
        raise ValueError(
            f"organizer bundle cutoff {manifest.get('cutoff_date')!r} "
            f"does not match {LABELING_CUTOFF_DATE.isoformat()!r}"
        )
    if manifest.get("validation_errors") != []:
        raise ValueError("organizer bundle must not contain validation errors")
    input_count = manifest.get("input_record_count")
    accepted = manifest.get("accepted_record_count")
    rejected = manifest.get("rejected_record_count")
    if (
        not _is_plain_int(input_count)
        or not _is_plain_int(accepted)
        or not _is_plain_int(rejected)
        or accepted <= 0
        or rejected != 0
        or input_count != accepted + rejected
    ):
        raise ValueError("organizer bundle must contain accepted records and no rejections")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError("organizer bundle manifest outputs must be a list")
    digest = next(
        (
            item
            for item in outputs
            if isinstance(item, dict)
            and item.get("filename") == POSITIVE_CANDIDATES_FILENAME
        ),
        None,
    )
    if not isinstance(digest, dict):
        raise ValueError("organizer bundle has no positive_candidates.jsonl digest")
    size_bytes = digest.get("size_bytes")
    sha256 = digest.get("sha256")
    if not _is_plain_int(size_bytes) or size_bytes < 0:
        raise ValueError("organizer bundle size_bytes must be a non-negative int")
    if not isinstance(sha256, str) or _SHA256_HEX.fullmatch(sha256) is None:
        raise ValueError("organizer bundle sha256 must be 64 lowercase hex characters")
    candidates_path = bundle / POSITIVE_CANDIDATES_FILENAME
    if not candidates_path.is_file():
        raise FileNotFoundError(f"organizer bundle is incomplete: {bundle}")
    candidates_bytes = candidates_path.read_bytes()
    if len(candidates_bytes) != size_bytes:
        raise ValueError("positive_candidates.jsonl size mismatch with manifest")
    if hashlib.sha256(candidates_bytes).hexdigest() != sha256:
        raise ValueError("positive_candidates.jsonl checksum mismatch with manifest")

    records: list[_EnrichmentCandidateRecord] = []
    for lineno, line in enumerate(candidates_bytes.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"positive candidate line {lineno} is not valid JSON") from error
        records.append(_parse_positive_candidate(payload, lineno))
    if len(records) != accepted:
        raise ValueError(
            f"positive candidate rows {len(records)} do not match accepted total {accepted}"
        )
    candidate_ids = [record.candidate_id for record in records]
    group_ids = [record.group_id for record in records]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("duplicate candidate_id in positive candidates")
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("duplicate group_id in positive candidates")

    terms_path = (
        Path(search_terms_file)
        if search_terms_file is not None
        else bundle / ENRICHMENT_SEARCH_TERMS_FILENAME
    )
    if not terms_path.is_file():
        raise ValueError(
            "organizer positive enrichment requires a reviewed --search-terms file"
        )
    term_overrides, terms_digest = _load_positive_search_terms(
        terms_path,
        positive_candidates_sha256=hashlib.sha256(candidates_bytes).hexdigest(),
    )
    records_by_id = {record.candidate_id: record for record in records}
    unknown_ids = sorted(set(term_overrides) - set(records_by_id))
    if unknown_ids:
        raise ValueError(
            f"positive search terms contain unknown candidate_id {unknown_ids[0]!r}"
        )
    selected_records = [
        records_by_id[candidate_id] for candidate_id in sorted(term_overrides)
    ]
    return _render_plan(
        selected_records,
        manifest_path=manifest_path,
        candidates_bytes=candidates_bytes,
        candidates_input_key="positive_candidates",
        search_term_overrides=term_overrides,
        extra_inputs={"search_terms": terms_digest},
    )


def build_enrichment_plan(
    bundle_dir: str | Path,
    *,
    search_terms_file: str | Path | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Validate the bundle and render deterministic ``plan.json`` bytes.

    Raises before touching the output directory, so failures never leave
    partial files behind.
    """

    bundle = Path(bundle_dir)
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"candidate bundle is incomplete: {bundle}")
    try:
        initial_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"candidate bundle manifest is not valid JSON: {error}") from error
    if not isinstance(initial_manifest, dict):
        raise ValueError("candidate bundle manifest must be a JSON object")
    if initial_manifest.get("schema_version") == ORGANIZER_MANIFEST_VERSION:
        return _build_positive_enrichment_plan(
            bundle, manifest_path, initial_manifest, search_terms_file
        )

    if search_terms_file is not None:
        raise ValueError("--search-terms is supported only for organizer positives")
    candidates_path = bundle / NEGATIVE_CANDIDATES_FILENAME
    if not candidates_path.is_file():
        raise FileNotFoundError(f"labeling bundle is incomplete: {bundle}")
    manifest = initial_manifest
    if manifest.get("schema_version") != LABELING_EXPORT_MANIFEST_VERSION:
        raise ValueError(
            "labeling bundle manifest "
            f"{manifest.get('schema_version')!r} does not match "
            f"{LABELING_EXPORT_MANIFEST_VERSION!r}"
        )
    if manifest.get("cutoff_date") != LABELING_CUTOFF_DATE.isoformat():
        raise ValueError(
            f"labeling bundle cutoff {manifest.get('cutoff_date')!r} "
            f"does not match {LABELING_CUTOFF_DATE.isoformat()!r}"
        )
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("labeling bundle manifest outputs must be an object")
    entry = outputs.get(NEGATIVE_CANDIDATES_FILENAME)
    if not isinstance(entry, dict):
        raise ValueError("labeling bundle manifest outputs entry must be an object")
    size_bytes = entry.get("size_bytes")
    if not _is_plain_int(size_bytes) or size_bytes < 0:
        raise ValueError("labeling bundle size_bytes must be a non-negative int")
    sha256 = entry.get("sha256")
    if not isinstance(sha256, str) or _SHA256_HEX.fullmatch(sha256) is None:
        raise ValueError("labeling bundle sha256 must be 64 lowercase hex characters")
    candidates_bytes = candidates_path.read_bytes()
    if len(candidates_bytes) != size_bytes:
        raise ValueError("negative_candidates.jsonl size mismatch with manifest")
    if hashlib.sha256(candidates_bytes).hexdigest() != sha256:
        raise ValueError("negative_candidates.jsonl checksum mismatch with manifest")
    filled = manifest.get("filled_candidates")
    if not isinstance(filled, dict) or not filled:
        raise ValueError("labeling bundle filled_candidates must be a non-empty object")
    for area, count in filled.items():
        if not _is_plain_int(count) or count < 0:
            raise ValueError(
                f"labeling bundle filled_candidates[{area!r}] must be a non-negative int"
            )

    records: list[NegativeCandidateRecord] = []
    for lineno, line in enumerate(candidates_bytes.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"negative candidate line {lineno} is not valid JSON") from error
        records.append(_parse_candidate(payload, lineno))
    if not records:
        raise ValueError("negative candidates file holds no records")

    candidate_ids = [record.candidate_id for record in records]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("duplicate candidate_id in negative candidates")
    group_ids = [record.group_id for record in records]
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("duplicate group_id in negative candidates")
    expected_count = sum(filled.values())
    if len(records) != expected_count:
        raise ValueError(
            f"negative candidate rows {len(records)} "
            f"do not match filled_candidates total {expected_count}"
        )
    actual_domains: dict[str, int] = {}
    for record in records:
        actual_domains[record.domain] = actual_domains.get(record.domain, 0) + 1
    if actual_domains != dict(filled):
        raise ValueError(
            f"negative candidate domains {actual_domains!r} "
            f"do not match filled_candidates {dict(filled)!r}"
        )

    return _render_plan(
        records,
        manifest_path=manifest_path,
        candidates_bytes=candidates_bytes,
        candidates_input_key="negative_candidates",
    )


def export_enrichment_plan(
    *,
    bundle_dir: str | Path,
    output_dir: str | Path,
    search_terms_file: str | Path | None = None,
) -> LabelingEnrichmentPlanPaths:
    """Validate the bundle and atomically publish plan.json with its manifest."""

    plan_bytes, provenance = build_enrichment_plan(
        bundle_dir, search_terms_file=search_terms_file
    )
    manifest_bytes = (
        json.dumps(
            {
                "schema_version": LABELING_ENRICHMENT_PLAN_VERSION,
                "bundle_id": provenance["bundle_id"],
                "candidate_count": provenance["candidate_count"],
                "inputs": provenance["inputs"],
                "outputs": {ENRICHMENT_PLAN_FILENAME: provenance["plan_digest"]},
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    paths = publish_artifact_bundle(
        {
            ENRICHMENT_PLAN_FILENAME: plan_bytes,
            ENRICHMENT_MANIFEST_FILENAME: manifest_bytes,
        },
        output_dir,
    )
    return LabelingEnrichmentPlanPaths(
        plan=paths[ENRICHMENT_PLAN_FILENAME],
        manifest=paths[ENRICHMENT_MANIFEST_FILENAME],
    )


def build_analysis_enrichment_plan(
    run_dir: str | Path,
) -> tuple[bytes, dict[str, Any]]:
    """Build the shared candidate enrichment plan for one complete discovery run."""

    run = Path(run_dir)
    manifest_path = run / "manifest.json"
    try:
        loaded = load_discovery_run(run)
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load discovery run: {error}") from error
    manifest = loaded.manifest
    result = loaded.result
    if manifest.get("schema_version") != DISCOVERY_RUN_MANIFEST_SCHEMA_VERSION:
        raise ValueError("discovery run manifest version is not supported")
    assert_run_analysis_eligible(loaded)
    if not isinstance(loaded.run_id, str) or not loaded.run_id.strip():
        raise ValueError("discovery run_id must not be blank")
    domain = manifest.get("domain")
    if not isinstance(domain, str) or not domain.strip():
        raise ValueError("analysis domain must not be blank")
    analysis_scope_key = manifest.get("analysis_scope_key")
    if not isinstance(analysis_scope_key, str) or not analysis_scope_key.strip():
        analysis_scope_key = ORGANIZER_SCOPE_KEYS.get(domain)
    if not isinstance(analysis_scope_key, str) or not analysis_scope_key.strip():
        raise ValueError("analysis run has no feature scope key")
    source_query = (loaded.plan.get("scope") or {}).get("raw_query")
    if not isinstance(source_query, str) or not source_query.strip():
        raise ValueError("discovery plan raw_query must not be blank")
    if manifest.get("raw_query") != source_query:
        raise ValueError("discovery raw_query differs between plan and manifest")
    result_bytes = (run / "pipeline_result.json").read_bytes()
    gate = result.get("candidate_gate")
    aliases = result.get("alias_resolution")
    coverage = result.get("gate_coverage")
    if not isinstance(gate, dict) or not isinstance(aliases, dict):
        raise ValueError("discovery result misses gate or alias resolution")
    if not isinstance(coverage, dict) or coverage.get("status") not in {
        "complete",
        "partial",
    }:
        raise ValueError("analysis enrichment requires valid Gate coverage")
    accepted = gate.get("accepted_proposal_ids")
    gate_inputs = gate.get("input_proposal_ids")
    decisions = gate.get("decisions")
    input_ids = aliases.get("input_proposal_ids")
    groups = aliases.get("groups")
    if not isinstance(accepted, list) or not isinstance(input_ids, list):
        raise ValueError("gate and alias proposal IDs must be lists")
    if not isinstance(gate_inputs, list) or not isinstance(decisions, list):
        raise ValueError("gate inputs and decisions must be lists")
    decision_ids: list[str] = []
    accepted_from_decisions: list[str] = []
    for index, decision in enumerate(decisions, 1):
        if not isinstance(decision, dict):
            raise ValueError(f"gate decision {index} must be an object")
        proposal_id = decision.get("proposal_id")
        value = decision.get("decision")
        if not isinstance(proposal_id, str) or value not in {"accept", "reject", "review"}:
            raise ValueError(f"gate decision {index} is invalid")
        decision_ids.append(proposal_id)
        if value == "accept":
            accepted_from_decisions.append(proposal_id)
    if len(decision_ids) != len(set(decision_ids)) or set(decision_ids) != set(gate_inputs):
        raise ValueError("gate decisions must cover every gate input exactly once")
    if accepted != accepted_from_decisions:
        raise ValueError("accepted proposal IDs differ from Gate decisions")
    if set(accepted) != set(input_ids) or len(accepted) != len(set(accepted)):
        raise ValueError("alias resolution must cover every accepted proposal exactly once")
    if not isinstance(groups, list) or not groups:
        raise ValueError("analysis enrichment requires accepted alias groups")
    cutoff_raw = manifest.get("cutoff_date")
    try:
        cutoff_date = date.fromisoformat(cutoff_raw)
    except (TypeError, ValueError) as error:
        raise ValueError("discovery cutoff_date is not an ISO date") from error
    records: list[_EnrichmentCandidateRecord] = []
    grouped_proposals: list[str] = []
    for index, group in enumerate(groups, 1):
        if not isinstance(group, dict):
            raise ValueError(f"alias group {index} must be an object")
        group_id = group.get("group_id")
        canonical = group.get("canonical_name")
        raw_aliases = group.get("aliases")
        proposal_ids = group.get("proposal_ids")
        if not isinstance(group_id, str) or not isinstance(canonical, str):
            raise ValueError(f"alias group {index} misses identity")
        if not isinstance(raw_aliases, list) or any(
            not isinstance(value, str) for value in raw_aliases
        ):
            raise ValueError(f"alias group {group_id} aliases must be strings")
        if not isinstance(proposal_ids, list) or any(
            not isinstance(value, str) for value in proposal_ids
        ):
            raise ValueError(f"alias group {group_id} proposal_ids must be strings")
        if not proposal_ids:
            raise ValueError(f"alias group {group_id} proposal_ids must not be empty")
        canonical_key = _clean_term(canonical).casefold()
        aliases_for_search: list[str] = []
        seen_aliases = {canonical_key}
        for raw_alias in raw_aliases:
            cleaned_alias = _clean_term(raw_alias)
            alias_key = cleaned_alias.casefold()
            if not cleaned_alias or alias_key in seen_aliases:
                continue
            seen_aliases.add(alias_key)
            aliases_for_search.append(cleaned_alias)
        grouped_proposals.extend(proposal_ids)
        records.append(
            _EnrichmentCandidateRecord(
                candidate_id=group_id,
                canonical_name=canonical,
                aliases=tuple(aliases_for_search),
                group_id=group_id,
                source_query=source_query,
                domain=domain,
                analysis_scope_key=analysis_scope_key.strip(),
                cutoff_date=cutoff_date,
            )
        )
    if len({record.group_id for record in records}) != len(records):
        raise ValueError("discovery alias group IDs must be unique")
    if len(grouped_proposals) != len(set(grouped_proposals)) or set(
        grouped_proposals
    ) != set(accepted):
        raise ValueError("alias groups must partition accepted proposals")
    return _render_plan(
        records,
        manifest_path=manifest_path,
        candidates_bytes=result_bytes,
        candidates_input_key="pipeline_result",
        extra_inputs={
            "discovery_run": {
                "run_id": loaded.run_id,
                "gate_coverage": {
                    "status": coverage["status"],
                    "checked_proposals": coverage["checked_proposals"],
                    "total_proposals": coverage["total_proposals"],
                    "skipped_proposals": coverage["skipped_proposals"],
                },
            }
        },
    )


def export_analysis_enrichment_plan(
    *, run_dir: str | Path, output_dir: str | Path
) -> LabelingEnrichmentPlanPaths:
    plan_bytes, provenance = build_analysis_enrichment_plan(run_dir)
    manifest_bytes = (
        json.dumps(
            {
                "schema_version": LABELING_ENRICHMENT_PLAN_VERSION,
                "plan_role": "analysis_candidates",
                "bundle_id": provenance["bundle_id"],
                "candidate_count": provenance["candidate_count"],
                "inputs": provenance["inputs"],
                "outputs": {ENRICHMENT_PLAN_FILENAME: provenance["plan_digest"]},
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    paths = publish_artifact_bundle(
        {
            ENRICHMENT_PLAN_FILENAME: plan_bytes,
            ENRICHMENT_MANIFEST_FILENAME: manifest_bytes,
        },
        output_dir,
    )
    return LabelingEnrichmentPlanPaths(
        plan=paths[ENRICHMENT_PLAN_FILENAME],
        manifest=paths[ENRICHMENT_MANIFEST_FILENAME],
    )


def build_target_enrichment_plan(
    candidates_file: str | Path,
) -> tuple[bytes, dict[str, Any]]:
    """Build retrieval for neutral targets used to repair a class deficit.

    This is not a labeling shortcut. The input cannot carry a class, target,
    review decision, or evidence verdict. Its output still has to pass a
    grounded Candidate Gate before a target can enter the labeling bundle.
    """

    path = Path(candidates_file)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read target candidates: {path}") from error
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("target candidates must be valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("target candidates must be a JSON object")
    if set(payload) != {"schema_version", "cutoff_date", "candidates"}:
        raise ValueError("target candidates contain unexpected or missing fields")
    if payload.get("schema_version") != TARGET_CANDIDATES_VERSION:
        raise ValueError(
            f"target candidates schema {payload.get('schema_version')!r} does not "
            f"match {TARGET_CANDIDATES_VERSION!r}"
        )
    if payload.get("cutoff_date") != LABELING_CUTOFF_DATE.isoformat():
        raise ValueError(
            f"target candidates cutoff {payload.get('cutoff_date')!r} does not "
            f"match {LABELING_CUTOFF_DATE.isoformat()!r}"
        )
    entries = payload.get("candidates")
    if not isinstance(entries, list) or not 1 <= len(entries) <= 20:
        raise ValueError("target candidates must contain 1 to 20 entries")

    records: list[_EnrichmentCandidateRecord] = []
    identities: set[tuple[str, str]] = set()
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict) or set(entry) != _TARGET_CANDIDATE_KEYS:
            raise ValueError(
                f"target candidate {index} contains unexpected or missing fields"
            )
        aliases = entry.get("aliases")
        if not isinstance(aliases, list) or any(
            not isinstance(alias, str) for alias in aliases
        ):
            raise ValueError(f"target candidate {index} aliases must be strings")
        for field in _TARGET_CANDIDATE_KEYS - {"aliases"}:
            if not isinstance(entry.get(field), str):
                raise ValueError(
                    f"target candidate {index} field {field!r} must be a string"
                )
        domain = entry["domain"].strip()
        if domain not in _TARGET_DOMAINS:
            raise ValueError(f"target candidate {index} has unsupported domain")
        canonical_name = _clean_term(entry["canonical_name"])
        identity = (domain.casefold(), canonical_name.casefold())
        if identity in identities:
            raise ValueError("target candidates contain duplicate domain/name identity")
        identities.add(identity)
        try:
            records.append(
                _EnrichmentCandidateRecord(
                    candidate_id=entry["candidate_id"].strip(),
                    canonical_name=canonical_name,
                    aliases=tuple(_clean_term(alias) for alias in aliases),
                    group_id="target-group-" + _stable_id(domain, identity[1]),
                    source_query=entry["source_query"].strip(),
                    domain=domain,
                    analysis_scope_key=entry["analysis_scope_key"].strip(),
                    cutoff_date=LABELING_CUTOFF_DATE,
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"target candidate {index} is invalid: {error}") from error

    candidate_ids = [record.candidate_id for record in records]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("target candidates contain duplicate candidate_id")
    plan_bytes, provenance = _render_plan(
        records,
        manifest_path=path,
        candidates_bytes=raw,
        candidates_input_key="target_candidates",
    )
    provenance["inputs"] = {"target_candidates": _digest(raw)}
    return plan_bytes, provenance


def export_target_enrichment_plan(
    *, candidates_file: str | Path, output_dir: str | Path
) -> LabelingEnrichmentPlanPaths:
    """Publish an ordinary enrichment-plan-v2 for neutral targets."""

    plan_bytes, provenance = build_target_enrichment_plan(candidates_file)
    manifest_bytes = (
        json.dumps(
            {
                "schema_version": LABELING_ENRICHMENT_PLAN_VERSION,
                "planner_version": LABELING_TARGET_ENRICHMENT_PLAN_VERSION,
                "bundle_id": provenance["bundle_id"],
                "candidate_count": provenance["candidate_count"],
                "inputs": provenance["inputs"],
                "outputs": {ENRICHMENT_PLAN_FILENAME: provenance["plan_digest"]},
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    paths = publish_artifact_bundle(
        {
            ENRICHMENT_PLAN_FILENAME: plan_bytes,
            ENRICHMENT_MANIFEST_FILENAME: manifest_bytes,
        },
        output_dir,
    )
    return LabelingEnrichmentPlanPaths(
        plan=paths[ENRICHMENT_PLAN_FILENAME],
        manifest=paths[ENRICHMENT_MANIFEST_FILENAME],
    )
