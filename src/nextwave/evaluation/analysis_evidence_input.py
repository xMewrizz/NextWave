"""Build query-local Evidence input from the frozen shortlist and enrichment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from nextwave.datasets.artifacts import publish_artifact_bundle
from nextwave.labeling.evidence_input_plan import (
    LABELING_EVIDENCE_INPUT_PLAN_VERSION,
)
from nextwave.labeling.relevance_plan import score_document

from .analysis_shortlist import ANALYSIS_SHORTLIST_VERSION
from .exa_enrichment_merge import ANALYSIS_COMBINED_ENRICHMENT_VERSION

ANALYSIS_EVIDENCE_INPUT_VERSION = "analysis-evidence-input-v2"
_ANALYSIS_PLAN_VERSION = "labeling-enrichment-plan-v2"
_MAX_PER_CLASS = 3
_PRIMARY_MIN_SCORE = 40
_FALLBACK_MIN_SCORE = 30
_FALLBACK_TARGET_PER_CLASS = 2


@dataclass(frozen=True, slots=True)
class AnalysisEvidenceInputPaths:
    documents: Path
    coverage: Path
    manifest: Path


def _digest(raw: bytes) -> dict[str, Any]:
    return {"size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value, raw


def _checked(root: Path, manifest: dict[str, Any], filename: str) -> bytes:
    try:
        raw = (root / filename).read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {filename}") from error
    if (manifest.get("outputs") or {}).get(filename) != _digest(raw):
        raise ValueError(f"{filename} differs from manifest")
    return raw


def _rows(raw: bytes, label: str) -> list[dict[str, Any]]:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8") from error
    rows = []
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label} line {number} is invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{label} line {number} must be an object")
        rows.append(row)
    return rows


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )


def _published(value: object, document_id: object, cutoff: date) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"document {document_id!r} has invalid published_at")
    try:
        result = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"document {document_id!r} has invalid published_at") from error
    if result > cutoff:
        raise ValueError(f"document {document_id!r} is after cutoff")
    return result


def _http_url(value: object, document_id: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"document {document_id!r} misses provenance")
    normalized = value.strip()
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"document {document_id!r} has invalid URL")
    return normalized


def _rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    trust = {"A": 0, "B": 1, "C": 2, "D": 3, "unknown": 4}
    published = row["_published"]
    return (
        -row["relevance_score"],
        trust.get(row["trust_tier"], 5),
        -(published.toordinal() if published else 0),
        row["document_id"],
    )


def _select(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    origins: set[str] = set()
    ranked = sorted(rows, key=_rank_key)
    for row in ranked:
        if row["relevance_score"] < _PRIMARY_MIN_SCORE:
            continue
        origin = row["origin_id"]
        if origin in origins:
            continue
        origins.add(origin)
        selected.append(row)
        if len(selected) == _MAX_PER_CLASS:
            break
    if len(selected) >= _FALLBACK_TARGET_PER_CLASS:
        return selected
    for row in ranked:
        if row["relevance_score"] >= _PRIMARY_MIN_SCORE:
            continue
        origin = row["origin_id"]
        if origin in origins:
            continue
        origins.add(origin)
        selected.append(row)
        if len(selected) == _FALLBACK_TARGET_PER_CLASS:
            break
    return selected


def build_analysis_evidence_input(
    *,
    analysis_plan_dir: str | Path,
    combined_result_dir: str | Path,
    shortlist_dir: str | Path,
) -> dict[str, bytes]:
    plan_root = Path(analysis_plan_dir)
    plan_manifest, plan_manifest_raw = _read_json(
        plan_root / "manifest.json", "analysis plan manifest"
    )
    if plan_manifest.get("schema_version") != _ANALYSIS_PLAN_VERSION:
        raise ValueError("analysis plan version is not supported")
    plan_raw = _checked(plan_root, plan_manifest, "plan.json")
    plan = json.loads(plan_raw)
    cutoff_raw = plan.get("cutoff_date")
    if not isinstance(cutoff_raw, str):
        raise ValueError("analysis plan has no cutoff_date")
    try:
        cutoff = date.fromisoformat(cutoff_raw)
    except ValueError as error:
        raise ValueError("analysis plan cutoff_date is invalid") from error
    raw_candidates = plan.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("analysis plan has no candidates")
    candidates: dict[str, dict[str, Any]] = {}
    for row in raw_candidates:
        if not isinstance(row, dict):
            raise ValueError("analysis candidates must be objects")
        candidate_id = row.get("candidate_id")
        terms = row.get("search_terms")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id in candidates
            or not isinstance(terms, list)
            or not terms
            or any(not isinstance(term, str) or not term.strip() for term in terms)
        ):
            raise ValueError("analysis candidate identity or search terms are invalid")
        candidates[candidate_id] = row

    combined_root = Path(combined_result_dir)
    combined_manifest, combined_manifest_raw = _read_json(
        combined_root / "manifest.json", "combined enrichment manifest"
    )
    if combined_manifest.get("schema_version") != ANALYSIS_COMBINED_ENRICHMENT_VERSION:
        raise ValueError("combined enrichment version is not supported")
    if combined_manifest.get("bundle_id") != plan_manifest.get("bundle_id"):
        raise ValueError("combined enrichment bundle differs from analysis plan")
    if combined_manifest.get("plan") != _digest(plan_raw):
        raise ValueError("combined enrichment was built from another analysis plan")
    documents = _rows(
        _checked(combined_root, combined_manifest, "documents.jsonl"), "documents"
    )
    combined_coverage = _rows(
        _checked(combined_root, combined_manifest, "coverage.jsonl"), "coverage"
    )
    coverage_pairs = {
        (row.get("candidate_id"), row.get("source_class"))
        for row in combined_coverage
        if row.get("status") in {"complete", "partial", "unknown"}
    }
    expected_pairs = {
        (candidate_id, source_class)
        for candidate_id in candidates
        for source_class in ("scientific", "industry")
    }
    if coverage_pairs != expected_pairs or len(combined_coverage) != len(expected_pairs):
        raise ValueError("combined enrichment coverage is not exactly defined")

    shortlist_root = Path(shortlist_dir)
    shortlist_manifest, shortlist_manifest_raw = _read_json(
        shortlist_root / "manifest.json", "shortlist manifest"
    )
    if shortlist_manifest.get("schema_version") != ANALYSIS_SHORTLIST_VERSION:
        raise ValueError("shortlist version is not supported")
    shortlist = _rows(
        _checked(shortlist_root, shortlist_manifest, "shortlist.jsonl"), "shortlist"
    )
    raw_ranks = [row.get("evidence_rank") for row in shortlist]
    if any(isinstance(rank, bool) or not isinstance(rank, int) for rank in raw_ranks):
        raise ValueError("shortlist ranks must be plain integers")
    ranks = sorted(raw_ranks)
    if ranks != list(range(1, len(shortlist) + 1)):
        raise ValueError("shortlist ranks must form 1..N")
    selected_ids: list[str] = []
    for row in shortlist:
        candidate_id = row.get("candidate_id")
        if candidate_id not in candidates or candidate_id in selected_ids:
            raise ValueError("shortlist candidate is unknown or duplicate")
        selected_ids.append(candidate_id)

    by_candidate: dict[str, dict[str, list[dict[str, Any]]]] = {
        candidate_id: {"scientific": [], "industry": []}
        for candidate_id in selected_ids
    }
    seen_pairs: set[tuple[str, str]] = set()
    for row in documents:
        candidate_id = row.get("candidate_id")
        if candidate_id not in by_candidate:
            continue
        document_id = row.get("document_id")
        connector = row.get("connector")
        if not isinstance(document_id, str) or connector not in {"openalex", "exa"}:
            raise ValueError("combined document has invalid identity or connector")
        pair = (candidate_id, document_id)
        if pair in seen_pairs:
            raise ValueError("combined documents duplicate candidate/document")
        seen_pairs.add(pair)
        title = row.get("title")
        excerpt = row.get("excerpt")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"document {document_id!r} has no title")
        if not isinstance(excerpt, str) or not excerpt.strip():
            continue
        score, term, reason, matched, _ = score_document(
            tuple(candidates[candidate_id]["search_terms"]), title, excerpt
        )
        if score < _FALLBACK_MIN_SCORE:
            continue
        source_class = "scientific" if connector == "openalex" else "industry"
        trust_tier = row.get("trust_tier")
        if trust_tier not in {"A", "B", "C", "D", "unknown"}:
            trust_tier = "unknown"
        origin_id = row.get("origin_id")
        snapshot_id = row.get("snapshot_id")
        url = row.get("url") or row.get("canonical_url")
        if not all(isinstance(value, str) and value for value in (origin_id, snapshot_id)):
            raise ValueError(f"document {document_id!r} misses provenance")
        url = _http_url(url, document_id)
        by_candidate[candidate_id][source_class].append(
            {
                **row,
                "source_class": source_class,
                "matched_term": term,
                "matched_tokens": list(matched),
                "relevance_score": score,
                "relevance_class": "strong" if score >= 60 else "weak",
                "relevance_reason": reason,
                "trust_tier": trust_tier,
                "url": url,
                "_published": _published(
                    row.get("published_at"), document_id, cutoff
                ),
            }
        )

    output_documents: list[dict[str, Any]] = []
    output_coverage: list[dict[str, Any]] = []
    for candidate_id in selected_ids:
        science = _select(by_candidate[candidate_id]["scientific"])
        industry = _select(by_candidate[candidate_id]["industry"])
        chosen: list[dict[str, Any]] = []
        for index in range(_MAX_PER_CLASS):
            if index < len(science):
                chosen.append(science[index])
            if index < len(industry):
                chosen.append(industry[index])
        classes: list[str] = []
        for final_rank, row in enumerate(chosen, 1):
            clean = {key: value for key, value in row.items() if key != "_published"}
            clean["final_rank"] = final_rank
            clean["evidence_text_available"] = True
            output_documents.append(clean)
            if clean["source_class"] not in classes:
                classes.append(clean["source_class"])
        output_coverage.append(
            {
                "candidate_id": candidate_id,
                "has_evidence_input": bool(chosen),
                "final_evidence_input_count": len(chosen),
                "scientific_selected": len(science),
                "media_selected": len(industry),
                "source_classes_present": sorted(classes),
                "empty_reasons": [] if chosen else ["no_relevant_source_text"],
            }
        )

    files = {
        "evidence_input_documents.jsonl": _jsonl(output_documents),
        "coverage.jsonl": _jsonl(output_coverage),
    }
    manifest = {
        "schema_version": LABELING_EVIDENCE_INPUT_PLAN_VERSION,
        "artifact_role": ANALYSIS_EVIDENCE_INPUT_VERSION,
        "bundle_id": plan_manifest.get("bundle_id"),
        "cutoff_date": cutoff.isoformat(),
        "selection_policy": {
            "mode": ANALYSIS_EVIDENCE_INPUT_VERSION,
            "shortlist_only": True,
            "primary_minimum_relevance_score": _PRIMARY_MIN_SCORE,
            "fallback_minimum_relevance_score": _FALLBACK_MIN_SCORE,
            "fallback_target_per_source_class": _FALLBACK_TARGET_PER_CLASS,
            "max_per_source_class": _MAX_PER_CLASS,
            "deduplicate_origin_within_source_class": True,
        },
        "totals": {
            "candidates": len(output_coverage),
            "evidence_input_documents": len(output_documents),
            "scientific_documents": sum(
                row["source_class"] == "scientific" for row in output_documents
            ),
            "industry_documents": sum(
                row["source_class"] == "industry" for row in output_documents
            ),
        },
        "inputs": {
            "analysis_plan_manifest": _digest(plan_manifest_raw),
            "combined_result_manifest": _digest(combined_manifest_raw),
            "shortlist_manifest": _digest(shortlist_manifest_raw),
        },
        "outputs": {name: _digest(raw) for name, raw in files.items()},
    }
    files["manifest.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return files


def export_analysis_evidence_input(
    *,
    analysis_plan_dir: str | Path,
    combined_result_dir: str | Path,
    shortlist_dir: str | Path,
    output_dir: str | Path,
) -> AnalysisEvidenceInputPaths:
    root = Path(output_dir)
    publish_artifact_bundle(
        build_analysis_evidence_input(
            analysis_plan_dir=analysis_plan_dir,
            combined_result_dir=combined_result_dir,
            shortlist_dir=shortlist_dir,
        ),
        root,
    )
    return AnalysisEvidenceInputPaths(
        documents=root / "evidence_input_documents.jsonl",
        coverage=root / "coverage.jsonl",
        manifest=root / "manifest.json",
    )
