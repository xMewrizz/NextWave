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
    POSITIVE_SCHEMA_VERSION,
    IdentityStatus,
    PositiveCandidateRecord,
)
from .contracts import (
    LABELING_CUTOFF_DATE,
    NEGATIVE_CANDIDATE_SCHEMA_VERSION,
    NegativeCandidateRecord,
)
from .export import LABELING_EXPORT_MANIFEST_VERSION

LABELING_ENRICHMENT_PLAN_VERSION = "labeling-enrichment-plan-v2"
ENRICHMENT_PLAN_FILENAME = "plan.json"
ENRICHMENT_MANIFEST_FILENAME = "manifest.json"
NEGATIVE_CANDIDATES_FILENAME = "negative_candidates.jsonl"
POSITIVE_CANDIDATES_FILENAME = "positive_candidates.jsonl"

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
    digest = hashlib.sha256(
        "\x1f".join(
            (LABELING_ENRICHMENT_PLAN_VERSION, LABELING_CUTOFF_DATE.isoformat(), *lines)
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


def _windows() -> dict[str, Any]:
    history_from = (LABELING_CUTOFF_DATE - timedelta(days=_HISTORY_DAYS)).isoformat()
    recent_from = (LABELING_CUTOFF_DATE - timedelta(days=_RECENT_DAYS)).isoformat()
    cutoff = LABELING_CUTOFF_DATE.isoformat()
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
) -> tuple[bytes, dict[str, Any]]:
    bundle_id = _bundle_id(records)
    windows = _windows()
    full_window = (windows["history_from"], windows["cutoff_date"])
    recent_window = (windows["recent_window_from"], windows["cutoff_date"])
    entries: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: item.candidate_id):
        terms = _search_terms(record.canonical_name, record.aliases)
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
        },
        "plan_digest": _digest(plan_bytes),
    }


def _build_positive_enrichment_plan(
    bundle: Path, manifest_path: Path, manifest: dict[str, Any]
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
    return _render_plan(
        records,
        manifest_path=manifest_path,
        candidates_bytes=candidates_bytes,
        candidates_input_key="positive_candidates",
    )


def build_enrichment_plan(bundle_dir: str | Path) -> tuple[bytes, dict[str, Any]]:
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
        return _build_positive_enrichment_plan(bundle, manifest_path, initial_manifest)

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
) -> LabelingEnrichmentPlanPaths:
    """Validate the bundle and atomically publish plan.json with its manifest."""

    plan_bytes, provenance = build_enrichment_plan(bundle_dir)
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
