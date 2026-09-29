"""Audit whether negative candidates satisfy the labeling rubric.

This stage is deliberately read-only and offline.  It never assigns a label:
it reports which reviewed evidence requirements are already satisfied and what
must be collected or replaced before the 50/50 corpus can be finalized.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nextwave.datasets.artifacts import publish_artifact_bundle

from .contracts import LABELING_CUTOFF_DATE
from .enrichment_plan import LABELING_ENRICHMENT_PLAN_VERSION
from .enrichment_run import ENRICHMENT_RESULT_VERSION
from .evidence_input_plan import LABELING_EVIDENCE_INPUT_PLAN_VERSION
from .evidence_llm_run import LABELING_EVIDENCE_LLM_RESULT_VERSION
from .relevance_plan import LABELING_RELEVANCE_PLAN_VERSION

LABELING_RUBRIC_AUDIT_VERSION = "labeling-rubric-audit-v1"

AUDIT_FILENAME = "candidate_rubric_audit.jsonl"
MATURITY_QUEUE_FILENAME = "maturity_evidence_queue.jsonl"
MANIFEST_FILENAME = "manifest.json"
RECENT_FROM = "2025-09-15"
RECENT_UNTIL = LABELING_CUTOFF_DATE.isoformat()

_MATURE_KINDS = frozenset({"standard", "adoption", "market"})
_PUBLICITY_KINDS = frozenset({"investment", "growth", "promotional_claim"})
_TECHNICAL_KINDS = frozenset({
    "standard", "adoption", "market", "pilot", "prototype", "research"
})
_MATURITY_CUES = {
    "standard": re.compile(r"\bstandard(?:s|ized|isation|ization)?\b", re.I),
    "adoption": re.compile(r"\b(?:adopted|adoption|widely used|widespread)\b", re.I),
    "deployment": re.compile(
        r"\b(?:deployed|deployment|operational|production[- ]scale|production use)\b",
        re.I,
    ),
    "market": re.compile(
        r"\b(?:commercial|commercialization|market|vendors?|industry adoption)\b",
        re.I,
    ),
    "infrastructure": re.compile(r"\binfrastructure\b", re.I),
}


@dataclass(frozen=True, slots=True)
class LabelingRubricAuditPaths:
    manifest: Path
    candidates: Path
    maturity_queue: Path


def _digest(payload: bytes) -> dict[str, Any]:
    return {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return (
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        )
    ).encode("utf-8")


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {error}") from error


def _checked_output(directory: Path, filename: str) -> bytes:
    manifest = _read_json(directory / MANIFEST_FILENAME, f"{directory.name} manifest")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("outputs"), dict):
        raise ValueError(f"{directory.name} manifest outputs must be an object")
    entry = manifest["outputs"].get(filename)
    if not isinstance(entry, dict):
        raise ValueError(f"{directory.name} manifest misses {filename}")
    size = entry.get("size_bytes")
    sha = entry.get("sha256")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError(f"{filename} size must be a non-negative int")
    if not isinstance(sha, str) or len(sha) != 64 or sha != sha.lower():
        raise ValueError(f"{filename} sha256 must be lowercase hex")
    try:
        int(sha, 16)
    except ValueError as error:
        raise ValueError(f"{filename} sha256 must be hexadecimal") from error
    payload = (directory / filename).read_bytes()
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != sha:
        raise ValueError(f"{filename} diverges from its manifest")
    return payload


def _rows(payload: bytes, label: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for lineno, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label} line {lineno} is not valid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{label} line {lineno} must be an object")
        result.append(row)
    return result


def _manifest(directory: Path, version: str, label: str) -> dict[str, Any]:
    value = _read_json(directory / MANIFEST_FILENAME, f"{label} manifest")
    if not isinstance(value, dict):
        raise ValueError(f"{label} manifest must be an object")
    if value.get("schema_version") != version:
        raise ValueError(
            f"{label} schema {value.get('schema_version')!r} does not match {version!r}"
        )
    return value


def _unique_id_rows(
    rows: Iterable[dict[str, Any]], key: str, label: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} row needs {key}")
        if value in indexed:
            raise ValueError(f"{label} duplicates {key} {value!r}")
        indexed[value] = row
    return indexed


def audit_candidate(
    candidate: Mapping[str, Any],
    documents: Iterable[Mapping[str, Any]],
    claims: Iterable[Mapping[str, Any]],
    *,
    search_coverage_complete: bool,
    recent_media_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return rubric facts for one candidate without assigning a label."""

    candidate_id = candidate.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("candidate needs candidate_id")
    document_index: dict[str, Mapping[str, Any]] = {}
    for document in documents:
        document_id = document.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError(f"candidate {candidate_id} document needs document_id")
        if document_id in document_index:
            raise ValueError(f"candidate {candidate_id} duplicates document {document_id}")
        document_index[document_id] = document

    maturity_origins: set[str] = set()
    publicity_claim_ids: list[str] = []
    publicity_origins: set[str] = set()
    technical_origins: set[str] = set()
    pilot_origins: set[str] = set()
    full_claims = 0
    claim_count = 0
    for claim in claims:
        claim_count += 1
        document_id = claim.get("document_id")
        document = document_index.get(document_id) if isinstance(document_id, str) else None
        if document is None:
            raise ValueError(
                f"candidate {candidate_id} claim references unknown document {document_id!r}"
            )
        if claim.get("scope") != "full_candidate":
            continue
        full_claims += 1
        kind = claim.get("kind")
        origin_id = document.get("origin_id")
        published_at = document.get("published_at")
        trust = document.get("trust_tier")
        if not isinstance(origin_id, str) or not origin_id:
            raise ValueError(f"candidate {candidate_id} document needs origin_id")
        if trust in ("A", "B") and kind in _MATURE_KINDS:
            maturity_origins.add(origin_id)
        if trust in ("A", "B") and kind in _TECHNICAL_KINDS:
            technical_origins.add(origin_id)
        if trust in ("A", "B") and kind == "pilot":
            pilot_origins.add(origin_id)
        if (
            kind in _PUBLICITY_KINDS
            and document.get("source_class") == "industry"
            and isinstance(published_at, str)
            and RECENT_FROM <= published_at <= RECENT_UNTIL
        ):
            claim_id = claim.get("claim_id")
            if not isinstance(claim_id, str) or not claim_id:
                raise ValueError(f"candidate {candidate_id} publicity claim needs claim_id")
            publicity_claim_ids.append(claim_id)
            publicity_origins.add(origin_id)

    recent_media_origins: set[str] = set()
    for row in recent_media_rows:
        if (
            row.get("connector") == "mediacloud"
            and row.get("relevance_class") in ("strong", "weak")
            and isinstance(row.get("published_at"), str)
            and RECENT_FROM <= row["published_at"] <= RECENT_UNTIL
        ):
            origin_id = row.get("origin_id")
            if isinstance(origin_id, str) and origin_id:
                recent_media_origins.add(origin_id)

    mature_eligible = bool(maturity_origins)
    wave_complete = len(publicity_claim_ids) >= 3 and len(publicity_origins) >= 2
    hype_eligible = (
        search_coverage_complete
        and wave_complete
        and len(technical_origins) < 2
        and not pilot_origins
    )
    deficits: list[str] = []
    if not maturity_origins:
        deficits.append("missing_ab_standard_deployment_or_market")
    if not search_coverage_complete:
        deficits.append("incomplete_candidate_search_coverage")
    if len(publicity_claim_ids) < 3:
        deficits.append("fewer_than_three_verified_recent_publicity_items")
    if len(publicity_origins) < 2:
        deficits.append("fewer_than_two_verified_publicity_origins")
    if len(technical_origins) >= 2:
        deficits.append("two_or_more_ab_technical_origins")
    if pilot_origins:
        deficits.append("confirmed_ab_pilot")

    if mature_eligible and hype_eligible:
        status = "conflict"
        next_action = "adjudicate_conflicting_rubric_evidence"
    elif mature_eligible:
        status = "mature_ready"
        next_action = "review_mature_decision"
    elif hype_eligible:
        status = "marketing_hype_ready"
        next_action = "review_marketing_hype_decision"
    elif len(recent_media_origins) >= 3:
        status = "needs_verified_publicity"
        next_action = "fetch_and_verify_publicity_wave"
    else:
        status = "insufficient_evidence"
        next_action = "targeted_rubric_search_or_replace"

    return {
        "candidate_id": candidate_id,
        "canonical_name": candidate.get("canonical_name"),
        "domain": candidate.get("domain"),
        "search_coverage_complete": search_coverage_complete,
        "claim_count": claim_count,
        "full_candidate_claims": full_claims,
        "maturity_ab_origin_ids": sorted(maturity_origins),
        "verified_publicity_claim_ids": sorted(publicity_claim_ids),
        "verified_publicity_origin_ids": sorted(publicity_origins),
        "recent_relevant_media_origin_ids": sorted(recent_media_origins),
        "technical_ab_origin_ids": sorted(technical_origins),
        "pilot_ab_origin_ids": sorted(pilot_origins),
        "mature_eligible": mature_eligible,
        "marketing_hype_eligible": hype_eligible,
        "status": status,
        "deficits": deficits,
        "next_action": next_action,
    }


def maturity_cues(title: str, excerpt: str) -> tuple[str, ...]:
    """Return deterministic maturity cue groups found in a document."""

    text = f"{title}\n{excerpt}"
    return tuple(name for name, pattern in _MATURITY_CUES.items() if pattern.search(text))


def build_maturity_queue(
    documents: Iterable[Mapping[str, Any]],
    ranked_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select up to four maturity-oriented OpenAlex documents per candidate."""

    ranked = {
        (row.get("candidate_id"), row.get("document_id")): row
        for row in ranked_rows
        if row.get("connector") == "openalex"
    }
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for document in documents:
        candidate_id = document.get("candidate_id")
        document_id = document.get("document_id")
        if not isinstance(candidate_id, str) or not isinstance(document_id, str):
            raise ValueError("enrichment document needs candidate_id/document_id")
        rank = ranked.get((candidate_id, document_id))
        if rank is None or rank.get("relevance_class") not in ("strong", "weak"):
            continue
        excerpt = document.get("excerpt")
        title = document.get("title")
        if not isinstance(excerpt, str) or not excerpt.strip():
            continue
        if not isinstance(title, str):
            raise ValueError(f"document {document_id} title must be a string")
        cues = maturity_cues(title, excerpt)
        if not cues:
            continue
        published_at = document.get("published_at")
        if isinstance(published_at, str) and published_at > RECENT_UNTIL:
            continue
        relevance_score = rank.get("score")
        if not isinstance(relevance_score, int) or isinstance(relevance_score, bool):
            raise ValueError(f"ranked document {document_id} score must be an int")
        by_candidate.setdefault(candidate_id, []).append({
            "candidate_id": candidate_id,
            "document_id": document_id,
            "connector": "openalex",
            "source_class": "scientific",
            "origin_id": document.get("origin_id"),
            "snapshot_id": document.get("snapshot_id"),
            "published_at": published_at,
            "trust_tier": document.get("trust_tier"),
            "matched_term": rank.get("matched_term"),
            "relevance_score": relevance_score,
            "maturity_cues": list(cues),
            "maturity_score": relevance_score + 15 * len(cues),
            "title": title,
            "url": document.get("url"),
            "excerpt": excerpt,
        })

    selected: list[dict[str, Any]] = []
    for candidate_id in sorted(by_candidate):
        ordered = sorted(
            by_candidate[candidate_id],
            key=lambda row: (
                -row["maturity_score"],
                0 if row["trust_tier"] == "A" else 1,
                row["published_at"] is None,
                row["published_at"] or "",
                row["document_id"],
            ),
        )
        seen_origins: set[str] = set()
        rank_number = 0
        for row in ordered:
            origin_id = row["origin_id"]
            if not isinstance(origin_id, str) or not origin_id or origin_id in seen_origins:
                continue
            seen_origins.add(origin_id)
            rank_number += 1
            selected.append({**row, "selection_rank": rank_number})
            if rank_number == 4:
                break
    return selected


def build_rubric_audit(
    *,
    plan_dir: str | Path,
    enrichment_result_dir: str | Path,
    evidence_input_dir: str | Path,
    evidence_result_dir: str | Path,
    relevance_dir: str | Path,
) -> tuple[bytes, bytes, bytes]:
    plan_path = Path(plan_dir)
    enrichment_path = Path(enrichment_result_dir)
    evidence_input_path = Path(evidence_input_dir)
    evidence_result_path = Path(evidence_result_dir)
    relevance_path = Path(relevance_dir)

    manifests = (
        _manifest(plan_path, LABELING_ENRICHMENT_PLAN_VERSION, "enrichment plan"),
        _manifest(enrichment_path, ENRICHMENT_RESULT_VERSION, "enrichment result"),
        _manifest(
            evidence_input_path,
            LABELING_EVIDENCE_INPUT_PLAN_VERSION,
            "evidence input",
        ),
        _manifest(
            evidence_result_path,
            LABELING_EVIDENCE_LLM_RESULT_VERSION,
            "evidence result",
        ),
        _manifest(relevance_path, LABELING_RELEVANCE_PLAN_VERSION, "relevance plan"),
    )
    bundle_ids = {item.get("bundle_id") for item in manifests}
    if len(bundle_ids) != 1 or not isinstance(next(iter(bundle_ids)), str):
        raise ValueError("rubric audit inputs do not share one bundle_id")
    bundle_id = next(iter(bundle_ids))
    if manifests[3].get("analysis_status") != "complete":
        raise ValueError("evidence result must be complete before rubric audit")

    plan = json.loads(_checked_output(plan_path, "plan.json"))
    if not isinstance(plan, dict) or not isinstance(plan.get("candidates"), list):
        raise ValueError("plan.json candidates must be a list")
    candidates = _unique_id_rows(plan["candidates"], "candidate_id", "candidate")
    candidate_ids = set(candidates)

    enrichment_coverage = _rows(
        _checked_output(enrichment_path, "coverage.jsonl"), "enrichment coverage"
    )
    enrichment_documents = _rows(
        _checked_output(enrichment_path, "documents.jsonl"), "enrichment documents"
    )
    coverage_classes: dict[str, set[str]] = {key: set() for key in candidate_ids}
    for row in enrichment_coverage:
        candidate_id = row.get("candidate_id")
        source_class = row.get("source_class")
        if candidate_id not in coverage_classes or source_class not in (
            "scientific", "industry"
        ):
            raise ValueError("enrichment coverage contains an unknown candidate/class")
        if row.get("status") != "complete":
            continue
        if source_class in coverage_classes[candidate_id]:
            raise ValueError("enrichment coverage duplicates candidate/class")
        coverage_classes[candidate_id].add(source_class)

    documents = _rows(
        _checked_output(evidence_input_path, "evidence_input_documents.jsonl"),
        "evidence input documents",
    )
    claims = _rows(
        _checked_output(evidence_result_path, "claims.jsonl"), "evidence claims"
    )
    evidence_coverage = _rows(
        _checked_output(evidence_result_path, "coverage.jsonl"), "evidence coverage"
    )
    coverage_by_candidate = _unique_id_rows(
        evidence_coverage, "candidate_id", "evidence coverage"
    )
    if set(coverage_by_candidate) != candidate_ids:
        raise ValueError("evidence coverage candidate roster diverges from plan")
    if any(
        row.get("status") not in ("complete", "no_input")
        for row in coverage_by_candidate.values()
    ):
        raise ValueError("evidence coverage must contain only complete/no_input")

    ranked = _rows(
        _checked_output(relevance_path, "ranked_documents.jsonl"),
        "ranked documents",
    )
    docs_by_candidate: dict[str, list[dict[str, Any]]] = {
        key: [] for key in candidate_ids
    }
    claims_by_candidate: dict[str, list[dict[str, Any]]] = {
        key: [] for key in candidate_ids
    }
    ranked_by_candidate: dict[str, list[dict[str, Any]]] = {
        key: [] for key in candidate_ids
    }
    for collection, target, label in (
        (documents, docs_by_candidate, "evidence document"),
        (claims, claims_by_candidate, "claim"),
        (ranked, ranked_by_candidate, "ranked document"),
    ):
        for row in collection:
            candidate_id = row.get("candidate_id")
            if candidate_id not in target:
                raise ValueError(f"{label} references unknown candidate {candidate_id!r}")
            target[candidate_id].append(row)

    audited = [
        audit_candidate(
            candidates[candidate_id],
            docs_by_candidate[candidate_id],
            claims_by_candidate[candidate_id],
            search_coverage_complete=(
                coverage_classes[candidate_id] == {"scientific", "industry"}
            ),
            recent_media_rows=ranked_by_candidate[candidate_id],
        )
        for candidate_id in sorted(candidate_ids)
    ]
    audit_bytes = _jsonl_bytes(audited)
    maturity_queue = build_maturity_queue(enrichment_documents, ranked)
    maturity_bytes = _jsonl_bytes(maturity_queue)
    statuses = Counter(row["status"] for row in audited)
    maturity_candidates = {row["candidate_id"] for row in maturity_queue}
    manifest = {
        "schema_version": LABELING_RUBRIC_AUDIT_VERSION,
        "bundle_id": bundle_id,
        "cutoff_date": LABELING_CUTOFF_DATE.isoformat(),
        "inputs": {
            "enrichment_plan": str(plan_path),
            "enrichment_result": str(enrichment_path),
            "evidence_input": str(evidence_input_path),
            "evidence_result": str(evidence_result_path),
            "relevance": str(relevance_path),
        },
        "totals": {
            "candidates": len(audited),
            "mature_ready": statuses["mature_ready"],
            "marketing_hype_ready": statuses["marketing_hype_ready"],
            "conflict": statuses["conflict"],
            "needs_verified_publicity": statuses["needs_verified_publicity"],
            "insufficient_evidence": statuses["insufficient_evidence"],
            "maturity_queue_documents": len(maturity_queue),
            "candidates_with_maturity_queue": len(maturity_candidates),
        },
        "outputs": {
            AUDIT_FILENAME: _digest(audit_bytes),
            MATURITY_QUEUE_FILENAME: _digest(maturity_bytes),
        },
    }
    return audit_bytes, maturity_bytes, _json_bytes(manifest)


def export_rubric_audit(
    *,
    plan_dir: str | Path,
    enrichment_result_dir: str | Path,
    evidence_input_dir: str | Path,
    evidence_result_dir: str | Path,
    relevance_dir: str | Path,
    output_dir: str | Path,
) -> LabelingRubricAuditPaths:
    audit_bytes, maturity_bytes, manifest_bytes = build_rubric_audit(
        plan_dir=plan_dir,
        enrichment_result_dir=enrichment_result_dir,
        evidence_input_dir=evidence_input_dir,
        evidence_result_dir=evidence_result_dir,
        relevance_dir=relevance_dir,
    )
    paths = publish_artifact_bundle(
        {
            AUDIT_FILENAME: audit_bytes,
            MATURITY_QUEUE_FILENAME: maturity_bytes,
            MANIFEST_FILENAME: manifest_bytes,
        },
        output_dir,
    )
    return LabelingRubricAuditPaths(
        manifest=paths[MANIFEST_FILENAME],
        candidates=paths[AUDIT_FILENAME],
        maturity_queue=paths[MATURITY_QUEUE_FILENAME],
    )
