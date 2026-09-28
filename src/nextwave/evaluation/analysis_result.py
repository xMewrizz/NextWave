"""Build one auditable query-local result from model and Evidence outputs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nextwave.contracts import CandidateStatus
from nextwave.datasets.artifacts import publish_artifact_bundle
from nextwave.labeling.evidence_llm_run import (
    LABELING_EVIDENCE_LLM_RESULT_VERSION,
)

from .analysis_features import ANALYSIS_FEATURE_TABLE_VERSION
from .analysis_inference import ANALYSIS_INFERENCE_VERSION, PREDICTIONS_FILENAME
from .decision_policy import (
    DECISION_POLICY_VERSION,
    DecisionPolicyInput,
    apply_decision_policy,
)
from .exa_enrichment_merge import ANALYSIS_COMBINED_ENRICHMENT_VERSION
from .feature_table import FEATURES_FILENAME, MANIFEST_FILENAME, _digest

ANALYSIS_RESULT_VERSION = "analysis-result-v1"
CANDIDATES_FILENAME = "candidates.jsonl"
TOP15_FILENAME = "top15.json"
SUMMARY_FILENAME = "summary.json"
_PLAN_VERSION = "labeling-enrichment-plan-v2"
_CUTOFF = "2026-09-15"
_SUPPORT_KINDS = {
    "novelty",
    "growth",
    "research",
    "patent",
    "prototype",
    "pilot",
    "investment",
}
_MATURITY_KINDS = {"adoption", "standard", "market"}
_CASE_KINDS = ("pilot", "prototype", "adoption", "investment", "research")
_REASON_RU = {
    "evidence_review_incomplete": "Evidence-проверка не завершена",
    "mature": "найдены признаки зрелости, внедрения, стандарта или рынка",
    "marketing_hype": "маркетинговое утверждение не подтверждено независимыми основаниями",
    "temporal_coverage_incomplete": "неполное временное покрытие",
    "insufficient_origins": "меньше двух независимых подтверждающих первоисточников",
    "insufficient_actors": "меньше двух независимых организаций или издателей",
    "insufficient_trusted_evidence": "нет подтверждающего источника доверия A/B",
    "model_below_threshold": "оценка модели ниже зафиксированного порога",
    "passed": "пройдены модельный порог и обязательные проверки evidence",
}


@dataclass(frozen=True, slots=True)
class AnalysisResultPaths:
    candidates: Path
    top15: Path
    summary: Path
    manifest: Path


def _read_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
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
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label} line {line_number} is invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{label} line {line_number} must be an object")
        result.append(row)
    return result


def _indexed(
    rows: list[dict[str, Any]], key: str, label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(rows, 1):
        value = row.get(key)
        if not isinstance(value, str) or not value or value in result:
            raise ValueError(f"{label} line {line_number} has invalid or duplicate {key}")
        result[value] = row
    return result


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )


def _source(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_id": document["document_id"],
        "name": document.get("title"),
        "url": document.get("url") or document.get("canonical_url"),
        "published_at": document.get("published_at"),
        "source_type": document.get("source_type"),
        "language": document.get("language"),
        "trust_tier": document.get("trust_tier", "unknown"),
        "publisher": document.get("publisher"),
        "automatic_translation": document.get("automatic_translation") is True,
        "generated_summary": document.get("generated_summary") is True,
    }


def _claim_view(claim: dict[str, Any], document: dict[str, Any]) -> dict[str, Any]:
    return {
        "claim_id": claim["claim_id"],
        "kind": claim["kind"],
        "direction": claim["direction"],
        "scope": claim["scope"],
        "quote": claim["quote"],
        "explanation_ru": claim["explanation_ru"],
        "source": _source(document),
    }


def _best_claim(
    claims: list[dict[str, Any]], preferred_kinds: tuple[str, ...]
) -> dict[str, Any] | None:
    for kind in preferred_kinds:
        for claim in claims:
            if claim["kind"] == kind:
                return claim
    return claims[0] if claims else None


def _load_evidence(
    roots: tuple[Path, ...],
    *,
    bundle_id: str,
    candidate_ids: set[str],
    documents: dict[tuple[str, str], dict[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    coverage: dict[str, dict[str, Any]] = {}
    claims: dict[str, list[dict[str, Any]]] = defaultdict(list)
    inputs: list[dict[str, Any]] = []
    seen_claims: set[str] = set()
    for root in roots:
        manifest, manifest_raw = _read_object(root / MANIFEST_FILENAME, "evidence manifest")
        if manifest.get("schema_version") != LABELING_EVIDENCE_LLM_RESULT_VERSION:
            raise ValueError("evidence result version is not supported")
        if manifest.get("bundle_id") != bundle_id or manifest.get("cutoff_date") != _CUTOFF:
            raise ValueError("evidence result belongs to another bundle or cutoff")
        coverage_rows = _rows(_checked(root, manifest, "coverage.jsonl"), "evidence coverage")
        claim_rows = _rows(_checked(root, manifest, "claims.jsonl"), "evidence claims")
        for row in coverage_rows:
            candidate_id = row.get("candidate_id")
            status = row.get("status")
            if (
                candidate_id not in candidate_ids
                or candidate_id in coverage
                or status not in {"complete", "failed", "no_input", "not_run"}
            ):
                raise ValueError("evidence coverage has invalid candidate or status")
            coverage[candidate_id] = row
        for claim in claim_rows:
            candidate_id = claim.get("candidate_id")
            document_id = claim.get("document_id")
            claim_id = claim.get("claim_id")
            quote = claim.get("quote")
            if (
                candidate_id not in candidate_ids
                or not isinstance(document_id, str)
                or not isinstance(claim_id, str)
                or not claim_id
                or claim_id in seen_claims
                or not isinstance(quote, str)
                or not quote
            ):
                raise ValueError("evidence claim has invalid identity")
            document = documents.get((candidate_id, document_id))
            if document is None or quote not in str(document.get("excerpt") or ""):
                raise ValueError("evidence claim is not verbatim in its source document")
            if claim.get("scope") not in {"full_candidate", "core_only"}:
                raise ValueError("evidence claim has invalid scope")
            if claim.get("direction") not in {"support", "counter"}:
                raise ValueError("evidence claim has invalid direction")
            if not isinstance(claim.get("kind"), str):
                raise ValueError("evidence claim has invalid kind")
            if not isinstance(claim.get("explanation_ru"), str):
                raise ValueError("evidence claim has no Russian explanation")
            seen_claims.add(claim_id)
            claims[candidate_id].append(claim)
        inputs.append(
            {
                "directory": root.name,
                "manifest": _digest(manifest_raw),
                "candidate_count": len(coverage_rows),
                "claim_count": len(claim_rows),
            }
        )
    if set(coverage) != candidate_ids:
        missing = sorted(candidate_ids - set(coverage))
        raise ValueError(f"evidence coverage does not cover all candidates: {missing[:3]}")
    for value in claims.values():
        value.sort(key=lambda row: row["claim_id"])
    inputs.sort(key=lambda row: row["directory"])
    return coverage, claims, inputs


def build_analysis_result(
    *,
    analysis_plan_dir: str | Path,
    combined_result_dir: str | Path,
    feature_dir: str | Path,
    inference_dir: str | Path,
    evidence_result_dirs: tuple[str | Path, ...],
) -> dict[str, bytes]:
    if not evidence_result_dirs:
        raise ValueError("at least one evidence result is required")
    plan_root = Path(analysis_plan_dir)
    plan_manifest, plan_manifest_raw = _read_object(
        plan_root / MANIFEST_FILENAME, "analysis plan manifest"
    )
    if plan_manifest.get("schema_version") != _PLAN_VERSION:
        raise ValueError("analysis plan version is not supported")
    plan_raw = _checked(plan_root, plan_manifest, "plan.json")
    plan = json.loads(plan_raw)
    raw_candidates = plan.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("analysis plan has no candidates")
    candidates = _indexed(raw_candidates, "candidate_id", "analysis candidates")
    candidate_ids = set(candidates)
    bundle_id = plan_manifest.get("bundle_id") or (plan.get("bundle") or {}).get("bundle_id")
    if not isinstance(bundle_id, str) or not bundle_id:
        raise ValueError("analysis plan has no bundle_id")

    combined_root = Path(combined_result_dir)
    combined_manifest, combined_manifest_raw = _read_object(
        combined_root / MANIFEST_FILENAME, "combined manifest"
    )
    if (
        combined_manifest.get("schema_version") != ANALYSIS_COMBINED_ENRICHMENT_VERSION
        or combined_manifest.get("bundle_id") != bundle_id
        or combined_manifest.get("plan") != _digest(plan_raw)
    ):
        raise ValueError("combined result does not match analysis plan")
    document_rows = _rows(
        _checked(combined_root, combined_manifest, "documents.jsonl"), "documents"
    )
    documents: dict[tuple[str, str], dict[str, Any]] = {}
    for line_number, row in enumerate(document_rows, 1):
        key = (row.get("candidate_id"), row.get("document_id"))
        if (
            key[0] not in candidate_ids
            or not isinstance(key[1], str)
            or not key[1]
            or key in documents
        ):
            raise ValueError(f"documents line {line_number} has invalid identity")
        documents[key] = row

    feature_root = Path(feature_dir)
    feature_manifest, feature_manifest_raw = _read_object(
        feature_root / MANIFEST_FILENAME, "feature manifest"
    )
    if feature_manifest.get("schema_version") != ANALYSIS_FEATURE_TABLE_VERSION:
        raise ValueError("feature table version is not supported")
    feature_rows = _rows(_checked(feature_root, feature_manifest, FEATURES_FILENAME), "features")
    features = _indexed(feature_rows, "candidate_id", "features")
    if set(features) != candidate_ids:
        raise ValueError("feature candidates differ from analysis plan")

    inference_root = Path(inference_dir)
    inference_manifest, inference_manifest_raw = _read_object(
        inference_root / MANIFEST_FILENAME, "inference manifest"
    )
    if (
        inference_manifest.get("schema_version") != ANALYSIS_INFERENCE_VERSION
        or (inference_manifest.get("inputs") or {}).get("features")
        != _digest(feature_manifest_raw)
    ):
        raise ValueError("inference does not match feature table")
    prediction_rows = _rows(
        _checked(inference_root, inference_manifest, PREDICTIONS_FILENAME), "predictions"
    )
    predictions = _indexed(prediction_rows, "candidate_id", "predictions")
    if set(predictions) != candidate_ids:
        raise ValueError("prediction candidates differ from analysis plan")

    evidence_coverage, claims, evidence_inputs = _load_evidence(
        tuple(Path(value) for value in evidence_result_dirs),
        bundle_id=bundle_id,
        candidate_ids=candidate_ids,
        documents=documents,
    )

    results: list[dict[str, Any]] = []
    for candidate_id in sorted(candidate_ids):
        prediction = predictions[candidate_id]
        score = prediction.get("model_score")
        threshold = prediction.get("decision_threshold")
        if (
            type(score) not in {int, float}
            or type(threshold) not in {int, float}
            or prediction.get("prediction") != int(float(score) >= float(threshold))
        ):
            raise ValueError(f"prediction {candidate_id!r} is inconsistent")
        full_claims = [
            claim for claim in claims.get(candidate_id, []) if claim["scope"] == "full_candidate"
        ]
        support = [
            claim
            for claim in full_claims
            if claim["direction"] == "support" and claim["kind"] in _SUPPORT_KINDS
        ]
        maturity = [claim for claim in full_claims if claim["kind"] in _MATURITY_KINDS]
        promotion = [claim for claim in full_claims if claim["kind"] == "promotional_claim"]
        origins: set[str] = set()
        actors: set[str] = set()
        grounded_ab = False
        for claim in support:
            document = documents[(candidate_id, claim["document_id"])]
            origin_id = document.get("origin_id")
            if isinstance(origin_id, str) and origin_id:
                origins.add(origin_id)
            for organization in document.get("organizations") or []:
                if isinstance(organization, str) and organization.strip():
                    actors.add(organization.strip().casefold())
            publisher = document.get("publisher")
            if isinstance(publisher, str) and publisher.strip():
                actors.add(publisher.strip().casefold())
            grounded_ab = grounded_ab or document.get("trust_tier") in {"A", "B"}
        marketing_hype = bool(promotion) and (len(origins) < 2 or not grounded_ab)
        feature_values = features[candidate_id].get("features")
        if not isinstance(feature_values, dict):
            raise ValueError(f"features {candidate_id!r} has no feature object")
        coverage_status = evidence_coverage[candidate_id]["status"]
        policy = apply_decision_policy(
            DecisionPolicyInput(
                candidate_id=candidate_id,
                gate_decision="accept",
                duplicate=False,
                substantive=True,
                evidence_review_complete=coverage_status == "complete",
                mature=bool(maturity),
                marketing_hype=marketing_hype,
                temporal_coverage_complete=feature_values.get(
                    "temporal_count_coverage_complete"
                )
                is True,
                independent_origin_count=len(origins),
                independent_actor_count=len(actors),
                grounded_ab_support=grounded_ab,
                model_score=float(score),
                decision_threshold=float(threshold),
            )
        )
        signal_views = [
            _claim_view(claim, documents[(candidate_id, claim["document_id"])])
            for claim in support
        ]
        skeptic_claims = [
            claim
            for claim in full_claims
            if claim["direction"] == "counter"
            or claim["kind"] in _MATURITY_KINDS
            or claim["kind"] == "promotional_claim"
        ]
        skeptic_views = [
            _claim_view(claim, documents[(candidate_id, claim["document_id"])])
            for claim in skeptic_claims
        ]
        description_claim = _best_claim(support, ("novelty", "research", "growth"))
        advantage_claim = _best_claim(support, ("novelty", "growth", "prototype", "pilot"))
        case_claim = _best_claim(full_claims, _CASE_KINDS)
        explanation = prediction.get("explanation")
        if not isinstance(explanation, dict):
            raise ValueError(f"prediction {candidate_id!r} has no model explanation")
        result = {
            "schema_version": ANALYSIS_RESULT_VERSION,
            "candidate_id": candidate_id,
            "canonical_name": candidates[candidate_id].get("canonical_name"),
            "aliases": candidates[candidate_id].get("aliases") or [],
            "domain": candidates[candidate_id].get("domain"),
            "source_query": (candidates[candidate_id].get("origin") or {}).get(
                "source_query"
            ),
            "status": policy.status.value,
            "reason": policy.reason.value,
            "reason_ru": _REASON_RU.get(policy.reason.value, policy.reason.value),
            "model": {
                "score": score,
                "threshold": threshold,
                "prediction": prediction["prediction"],
                "score_semantics": "uncalibrated_sigmoid_score",
                "confidence": None,
                "top_positive_factors": explanation.get("top_positive_factors", []),
                "top_negative_factors": explanation.get("top_negative_factors", []),
            },
            "evidence_review": {
                "status": coverage_status,
                "full_candidate_claims": len(full_claims),
                "support_claims": len(support),
                "counter_claims": len(skeptic_claims),
                "independent_origins": len(origins),
                "independent_actors": len(actors),
                "grounded_ab_support": grounded_ab,
            },
            "description_ru": (
                description_claim["explanation_ru"] if description_claim else None
            ),
            "potential_advantage_ru": (
                advantage_claim["explanation_ru"] if advantage_claim else None
            ),
            "case_example": (
                _claim_view(
                    case_claim,
                    documents[(candidate_id, case_claim["document_id"])],
                )
                if case_claim
                else None
            ),
            "signal_case": signal_views,
            "skeptic_case": skeptic_views,
            "limitations": (
                []
                if coverage_status == "complete"
                else ["Evidence-проверка не завершена; кандидат не допускается в TOP-15"]
            ),
        }
        results.append(result)

    main = [row for row in results if row["status"] == CandidateStatus.MAIN.value]
    main.sort(
        key=lambda row: (
            -float(row["model"]["score"]),
            -int(row["evidence_review"]["independent_origins"]),
            str(row["canonical_name"] or "").casefold(),
            row["candidate_id"],
        )
    )
    top15 = []
    for rank, row in enumerate(main[:15], 1):
        row["top15_rank"] = rank
        top15.append(row)
    by_id = {row["candidate_id"]: row for row in results}
    for row in top15:
        by_id[row["candidate_id"]]["top15_rank"] = row["top15_rank"]
    results.sort(key=lambda row: row["candidate_id"])

    status_counts = Counter(row["status"] for row in results)
    reason_counts = Counter(row["reason"] for row in results)
    unique_documents = {row["document_id"] for row in document_rows}
    unique_origins = {
        row.get("origin_id")
        for row in document_rows
        if isinstance(row.get("origin_id"), str) and row.get("origin_id")
    }
    summary = {
        "schema_version": ANALYSIS_RESULT_VERSION,
        "source_query": next(iter(results))["source_query"],
        "release_status": inference_manifest.get("release_status"),
        "candidate_count": len(results),
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "top15_count": len(top15),
        "processed_document_relations": len(document_rows),
        "processed_unique_documents": len(unique_documents),
        "processed_unique_origins": len(unique_origins),
        "candidates_model_score_gt_075": sum(
            float(row["model"]["score"]) > 0.75 for row in results
        ),
        "main_model_score_gt_075": sum(
            float(row["model"]["score"]) > 0.75 for row in main
        ),
        "high_confidence_weak_signals": None,
        "confidence_available": False,
        "confidence_note": (
            "model_score is not calibrated; counts above 0.75 are score counters"
        ),
    }
    candidate_bytes = _jsonl(results)
    top15_bytes = (
        json.dumps(top15, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    summary_bytes = (
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    files = {
        CANDIDATES_FILENAME: candidate_bytes,
        TOP15_FILENAME: top15_bytes,
        SUMMARY_FILENAME: summary_bytes,
    }
    manifest = {
        "schema_version": ANALYSIS_RESULT_VERSION,
        "bundle_id": bundle_id,
        "cutoff_date": _CUTOFF,
        "release_status": inference_manifest.get("release_status"),
        "decision_policy_version": DECISION_POLICY_VERSION,
        "candidate_count": len(results),
        "top15_count": len(top15),
        "assumptions": {
            "all_analysis_plan_candidates_passed_candidate_gate": True,
            "analysis_plan_alias_groups_are_not_duplicates": True,
            "semantic_maturity_kinds_override_llm_direction": True,
        },
        "inputs": {
            "analysis_plan_manifest": _digest(plan_manifest_raw),
            "combined_result_manifest": _digest(combined_manifest_raw),
            "feature_manifest": _digest(feature_manifest_raw),
            "inference_manifest": _digest(inference_manifest_raw),
            "evidence_results": evidence_inputs,
        },
        "outputs": {name: _digest(raw) for name, raw in files.items()},
    }
    files[MANIFEST_FILENAME] = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return files


def export_analysis_result(
    *,
    analysis_plan_dir: str | Path,
    combined_result_dir: str | Path,
    feature_dir: str | Path,
    inference_dir: str | Path,
    evidence_result_dirs: tuple[str | Path, ...],
    output_dir: str | Path,
) -> AnalysisResultPaths:
    root = Path(output_dir)
    publish_artifact_bundle(
        build_analysis_result(
            analysis_plan_dir=analysis_plan_dir,
            combined_result_dir=combined_result_dir,
            feature_dir=feature_dir,
            inference_dir=inference_dir,
            evidence_result_dirs=evidence_result_dirs,
        ),
        root,
    )
    return AnalysisResultPaths(
        candidates=root / CANDIDATES_FILENAME,
        top15=root / TOP15_FILENAME,
        summary=root / SUMMARY_FILENAME,
        manifest=root / MANIFEST_FILENAME,
    )
