"""Build a Gate-aware Evidence input for marketing-hype adjudication.

The ordinary Evidence input balances scientific and industry documents, which
can leave fewer than the three media items required by the hype rubric.  This
stage keeps only target candidates accepted by the qualification Gate, admits
up to four recent strong full-text media documents, and uses the remaining
task capacity for scientific documents that may disprove hype.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from nextwave.datasets.artifacts import publish_artifact_bundle
from nextwave.discovery.candidate_gate import QUALIFICATION_GATE_ID

from .contracts import LABELING_CUTOFF_DATE
from .enrichment_plan import LABELING_ENRICHMENT_PLAN_VERSION
from .evidence_input_plan import LABELING_EVIDENCE_INPUT_PLAN_VERSION
from .media_fetch_run import LABELING_MEDIA_FETCH_RESULT_VERSION
from .target_gate import LABELING_TARGET_GATE_VERSION

HYPE_INPUT_POLICY_VERSION = "hype-rubric-input-v1"
DOCUMENTS_FILENAME = "evidence_input_documents.jsonl"
COVERAGE_FILENAME = "coverage.jsonl"
MANIFEST_FILENAME = "manifest.json"
MEDIA_RERANKED_FILENAME = "media_reranked.jsonl"
ENRICHED_MEDIA_FILENAME = "enriched_documents.jsonl"
GATE_RESULTS_FILENAME = "gate_results.jsonl"

RECENT_FROM = "2025-09-15"
RECENT_UNTIL = LABELING_CUTOFF_DATE.isoformat()
MIN_PUBLICITY_DOCUMENTS = 3
MIN_PUBLICITY_PUBLISHERS = 2
MAX_MEDIA_DOCUMENTS = 4
MAX_DOCUMENTS_PER_TASK = 6


@dataclass(frozen=True, slots=True)
class HypeEvidenceInputPaths:
    manifest: Path
    documents: Path
    coverage: Path


def _digest(payload: bytes) -> dict[str, Any]:
    return {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _checked_payload(directory: Path, filename: str, manifest: dict[str, Any]) -> bytes:
    outputs = manifest.get("outputs")
    entry = outputs.get(filename) if isinstance(outputs, dict) else None
    if not isinstance(entry, dict):
        raise ValueError(f"manifest misses {filename}")
    payload = (directory / filename).read_bytes()
    if entry.get("size_bytes") != len(payload):
        raise ValueError(f"{filename} size mismatch")
    if entry.get("sha256") != hashlib.sha256(payload).hexdigest():
        raise ValueError(f"{filename} checksum mismatch")
    return payload


def _rows(payload: bytes, label: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for lineno, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label} line {lineno} is invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{label} line {lineno} must be an object")
        result.append(row)
    return result


def _manifest(directory: Path, version: str, label: str) -> dict[str, Any]:
    value = _read_json(directory / MANIFEST_FILENAME, f"{label} manifest")
    if value.get("schema_version") != version:
        raise ValueError(
            f"{label} schema {value.get('schema_version')!r} does not match {version!r}"
        )
    return value


def _unique(rows: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for lineno, row in enumerate(rows, start=1):
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} line {lineno} needs {key}")
        if value in indexed:
            raise ValueError(f"{label} duplicates {key} {value!r}")
        indexed[value] = row
    return indexed


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_url(value: Any, label: str) -> str:
    value = _require_text(value, label)
    if not value.startswith(("http://", "https://")) or " " in value:
        raise ValueError(f"{label} must be an absolute HTTP(S) URL")
    return value


def _require_date(value: Any, label: str) -> str:
    value = _require_text(value, label)
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO date") from error
    if parsed > LABELING_CUTOFF_DATE:
        raise ValueError(f"{label} is after the labeling cutoff")
    return value


def _require_digest_match(actual: bytes, expected: Any, label: str) -> None:
    if expected != _digest(actual):
        raise ValueError(f"{label} digest mismatch")


def _pair_index(
    rows: list[dict[str, Any]], label: str
) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for lineno, row in enumerate(rows, start=1):
        candidate_id = row.get("candidate_id")
        document_id = row.get("document_id")
        if not isinstance(candidate_id, str) or not isinstance(document_id, str):
            raise ValueError(f"{label} line {lineno} needs candidate_id/document_id")
        pair = (candidate_id, document_id)
        if pair in indexed:
            raise ValueError(f"{label} duplicates pair {pair!r}")
        indexed[pair] = row
    return indexed


def _document_from_media(
    ranked: dict[str, Any], source: dict[str, Any], final_rank: int
) -> dict[str, Any]:
    excerpt = source.get("excerpt")
    if (
        source.get("evidence_text_available") is not True
        or source.get("content_status") != "article_text"
        or not isinstance(excerpt, str)
        or not excerpt.strip()
    ):
        raise ValueError("selected hype media document lacks article text")
    _require_text(source.get("extraction_method"), "selected media extraction_method")
    return {
        "candidate_id": ranked["candidate_id"],
        "document_id": ranked["document_id"],
        "connector": "mediacloud",
        "source_class": "industry",
        "final_rank": final_rank,
        "title": source["title"],
        "url": source["url"],
        "origin_id": source["origin_id"],
        "snapshot_id": source["snapshot_id"],
        "published_at": source.get("published_at"),
        "trust_tier": source["trust_tier"],
        "matched_term": ranked["matched_term"],
        "relevance_score": ranked["final_score"],
        "relevance_class": "strong",
        "evidence_text_available": True,
        "excerpt": excerpt,
    }


def build_hype_evidence_input(
    *,
    plan_dir: str | Path,
    gate_dir: str | Path,
    evidence_input_dir: str | Path,
    media_result_dir: str | Path,
) -> tuple[bytes, bytes, bytes]:
    plan_path = Path(plan_dir)
    gate_path = Path(gate_dir)
    input_path = Path(evidence_input_dir)
    media_path = Path(media_result_dir)

    plan_manifest = _manifest(plan_path, LABELING_ENRICHMENT_PLAN_VERSION, "plan")
    gate_manifest = _manifest(gate_path, LABELING_TARGET_GATE_VERSION, "target gate")
    input_manifest = _manifest(
        input_path, LABELING_EVIDENCE_INPUT_PLAN_VERSION, "evidence input"
    )
    media_manifest = _manifest(
        media_path, LABELING_MEDIA_FETCH_RESULT_VERSION, "media result"
    )
    for label, manifest in (
        ("target Gate", gate_manifest),
        ("evidence input", input_manifest),
    ):
        if manifest.get("cutoff_date") != LABELING_CUTOFF_DATE.isoformat():
            raise ValueError(f"{label} cutoff does not match the labeling cutoff")
    if gate_manifest.get("gate_id") != QUALIFICATION_GATE_ID:
        raise ValueError("target Gate does not use the qualification Gate")
    bundle_ids = {
        plan_manifest.get("bundle_id"),
        input_manifest.get("bundle_id"),
        media_manifest.get("bundle_id"),
    }
    if len(bundle_ids) != 1 or not isinstance(next(iter(bundle_ids)), str):
        raise ValueError("hype input sources do not share one bundle_id")
    bundle_id = next(iter(bundle_ids))

    plan_payload = _checked_payload(plan_path, "plan.json", plan_manifest)
    gate_inputs = gate_manifest.get("inputs")
    if not isinstance(gate_inputs, dict):
        raise ValueError("target Gate manifest inputs must be an object")
    if gate_inputs.get("plan.json") != _digest(plan_payload):
        raise ValueError("target Gate is not bound to the supplied plan.json")
    media_inputs = media_manifest.get("inputs")
    result_manifest = (
        media_inputs.get("result_manifest") if isinstance(media_inputs, dict) else None
    )
    if gate_inputs.get("enrichment_manifest.json") != result_manifest:
        raise ValueError("target Gate and media result use different enrichment results")
    input_links = input_manifest.get("inputs")
    if not isinstance(input_links, dict):
        raise ValueError("evidence input manifest inputs must be an object")
    _require_digest_match(
        plan_payload, input_links.get("plan_file"), "evidence input plan_file"
    )
    _require_digest_match(
        (plan_path / MANIFEST_FILENAME).read_bytes(),
        input_links.get("plan_manifest"),
        "evidence input plan_manifest",
    )
    _require_digest_match(
        (media_path / MANIFEST_FILENAME).read_bytes(),
        input_links.get("media_manifest"),
        "evidence input media_manifest",
    )
    plan = json.loads(plan_payload)
    if plan.get("cutoff_date") != LABELING_CUTOFF_DATE.isoformat():
        raise ValueError("plan cutoff does not match the labeling cutoff")
    candidates = plan.get("candidates") if isinstance(plan, dict) else None
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("plan candidates must be a non-empty list")
    candidate_index = _unique(candidates, "candidate_id", "candidate")
    for candidate_id, candidate in candidate_index.items():
        if candidate.get("cutoff_date") != LABELING_CUTOFF_DATE.isoformat():
            raise ValueError(f"candidate {candidate_id} cutoff mismatch")

    gate_rows = _rows(
        _checked_payload(gate_path, GATE_RESULTS_FILENAME, gate_manifest),
        GATE_RESULTS_FILENAME,
    )
    gate_index = _unique(gate_rows, "candidate_id", "gate result")
    if set(gate_index) != set(candidate_index):
        raise ValueError("target Gate roster diverges from plan candidates")
    for candidate_id, gate in gate_index.items():
        candidate = candidate_index[candidate_id]
        if gate.get("schema_version") != LABELING_TARGET_GATE_VERSION:
            raise ValueError(f"candidate {candidate_id} has a wrong Gate row version")
        if gate.get("canonical_name") != candidate.get("canonical_name"):
            raise ValueError(f"candidate {candidate_id} Gate canonical_name mismatch")
        if gate.get("domain") != candidate.get("domain"):
            raise ValueError(f"candidate {candidate_id} Gate domain mismatch")
        promoted = gate.get("promotion_status") == "evidence_review_required"
        accepted = (
            gate.get("canonical_grounded") is True
            and gate.get("gate_called") is True
            and gate.get("gate_id") == QUALIFICATION_GATE_ID
            and gate.get("gate_decision") == "accept"
        )
        if promoted != accepted:
            raise ValueError(f"candidate {candidate_id} has contradictory Gate fields")

    base_documents = _rows(
        _checked_payload(input_path, DOCUMENTS_FILENAME, input_manifest),
        DOCUMENTS_FILENAME,
    )
    reranked = _rows(
        _checked_payload(input_path, MEDIA_RERANKED_FILENAME, input_manifest),
        MEDIA_RERANKED_FILENAME,
    )
    media_documents = _rows(
        _checked_payload(media_path, ENRICHED_MEDIA_FILENAME, media_manifest),
        ENRICHED_MEDIA_FILENAME,
    )
    media_index = _pair_index(media_documents, "enriched media")

    science_by_candidate: dict[str, list[dict[str, Any]]] = {
        candidate_id: [] for candidate_id in candidate_index
    }
    science_pairs: set[tuple[str, str]] = set()
    science_ranks: dict[str, set[int]] = {key: set() for key in candidate_index}
    for lineno, row in enumerate(base_documents, start=1):
        candidate_id = row.get("candidate_id")
        if candidate_id not in science_by_candidate:
            raise ValueError("evidence input references an unknown candidate")
        if row.get("source_class") == "scientific":
            document_id = _require_text(
                row.get("document_id"), f"scientific line {lineno} document_id"
            )
            pair = (candidate_id, document_id)
            if pair in science_pairs:
                raise ValueError(f"scientific input duplicates pair {pair!r}")
            science_pairs.add(pair)
            rank = row.get("final_rank")
            if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
                raise ValueError("scientific final_rank must be a positive plain int")
            if rank in science_ranks[candidate_id]:
                raise ValueError(f"candidate {candidate_id} duplicates scientific rank")
            science_ranks[candidate_id].add(rank)
            if row.get("connector") != "openalex":
                raise ValueError("scientific input connector must be openalex")
            if row.get("evidence_text_available") is not True:
                raise ValueError("scientific input must have evidence text")
            _require_text(row.get("excerpt"), "scientific excerpt")
            _require_text(row.get("title"), "scientific title")
            _require_text(row.get("origin_id"), "scientific origin_id")
            _require_text(row.get("snapshot_id"), "scientific snapshot_id")
            _require_text(row.get("matched_term"), "scientific matched_term")
            _require_url(row.get("url"), "scientific url")
            if row.get("trust_tier") not in ("A", "B", "C", "D", "unknown"):
                raise ValueError("scientific trust_tier is invalid")
            score = row.get("relevance_score")
            if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 100:
                raise ValueError("scientific relevance_score must be a plain int 0..100")
            if row.get("relevance_class") not in ("strong", "weak"):
                raise ValueError("scientific relevance_class must be strong or weak")
            published_at = row.get("published_at")
            if published_at is not None:
                _require_date(published_at, "scientific published_at")
            science_by_candidate[candidate_id].append(row)
    for rows in science_by_candidate.values():
        rows.sort(key=lambda row: (row["final_rank"], row["document_id"]))

    media_by_candidate: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {
        candidate_id: [] for candidate_id in candidate_index
    }
    seen_pairs: set[tuple[str, str]] = set()
    for row in reranked:
        candidate_id = row.get("candidate_id")
        document_id = row.get("document_id")
        pair = (candidate_id, document_id)
        if pair in seen_pairs:
            raise ValueError(f"media reranking duplicates pair {pair!r}")
        seen_pairs.add(pair)
        if candidate_id not in media_by_candidate or not isinstance(document_id, str):
            raise ValueError("media reranking references an unknown candidate/document")
        if row.get("final_class") != "strong":
            continue
        score = row.get("final_score")
        if not isinstance(score, int) or isinstance(score, bool) or score < 60:
            raise ValueError(f"strong media row {pair!r} has an invalid final_score")
        published_at = _require_date(row.get("published_at"), "media ranked published_at")
        if not RECENT_FROM <= published_at <= RECENT_UNTIL:
            continue
        source = media_index.get((candidate_id, document_id))
        if source is None:
            raise ValueError(f"strong media row {pair!r} lacks enriched document")
        if source.get("published_at") != published_at:
            raise ValueError(f"strong media row {pair!r} date diverges from source")
        if source.get("url") != row.get("url"):
            raise ValueError(f"strong media row {pair!r} URL diverges from source")
        if source.get("origin_id") != row.get("origin_id"):
            raise ValueError(f"strong media row {pair!r} origin diverges from source")
        source_identity = source.get("canonical_url") or source.get("url")
        if row.get("identity") != source_identity:
            raise ValueError(f"strong media row {pair!r} identity diverges from source")
        if (
            source.get("evidence_text_available") is not True
            or source.get("content_status") != "article_text"
        ):
            continue
        excerpt = source.get("excerpt")
        if not isinstance(excerpt, str) or not excerpt.strip():
            continue
        media_by_candidate[candidate_id].append((row, source))
    for rows in media_by_candidate.values():
        rows.sort(
            key=lambda item: (
                item[0].get("selection_rank", 0),
                item[0].get("document_id", ""),
            )
        )

    output_documents: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for candidate_id in sorted(candidate_index):
        gate = gate_index[candidate_id]
        promotion_status = gate.get("promotion_status")
        media_rows = media_by_candidate[candidate_id][:MAX_MEDIA_DOCUMENTS]
        reasons: list[str] = []
        selected: list[dict[str, Any]] = []
        if promotion_status != "evidence_review_required":
            reasons.append(f"target_gate:{promotion_status}")
        elif len(media_rows) < MIN_PUBLICITY_DOCUMENTS:
            reasons.append("fewer_than_three_strong_recent_media_documents")
        elif len({source.get("publisher") for _, source in media_rows}) < MIN_PUBLICITY_PUBLISHERS:
            reasons.append("fewer_than_two_recent_media_publishers")
        else:
            for ranked, source in media_rows:
                selected.append(_document_from_media(ranked, source, len(selected) + 1))
            science_limit = MAX_DOCUMENTS_PER_TASK - len(selected)
            for source in science_by_candidate[candidate_id][:science_limit]:
                selected.append({**source, "final_rank": len(selected) + 1})

        output_documents.extend(selected)
        scientific = sum(row["source_class"] == "scientific" for row in selected)
        industry = len(selected) - scientific
        coverage.append({
            "candidate_id": candidate_id,
            "has_evidence_input": bool(selected),
            "final_evidence_input_count": len(selected),
            "scientific_selected": scientific,
            "media_selected": industry,
            "source_classes_present": [
                source_class
                for source_class, count in (("industry", industry), ("scientific", scientific))
                if count
            ],
            "empty_reasons": reasons,
            "gate_promotion_status": promotion_status,
            "strong_recent_media_available": len(media_by_candidate[candidate_id]),
        })

    documents_bytes = _jsonl_bytes(output_documents)
    coverage_bytes = _jsonl_bytes(coverage)
    manifest = {
        "schema_version": LABELING_EVIDENCE_INPUT_PLAN_VERSION,
        "bundle_id": bundle_id,
        "cutoff_date": LABELING_CUTOFF_DATE.isoformat(),
        "selection_policy": {
            "mode": HYPE_INPUT_POLICY_VERSION,
            "requires_gate_promotion_status": "evidence_review_required",
            "minimum_strong_recent_media_documents": MIN_PUBLICITY_DOCUMENTS,
            "minimum_recent_media_publishers": MIN_PUBLICITY_PUBLISHERS,
            "max_media_documents": MAX_MEDIA_DOCUMENTS,
            "max_scientific_documents": MAX_DOCUMENTS_PER_TASK - MIN_PUBLICITY_DOCUMENTS,
            "max_documents_per_candidate": MAX_DOCUMENTS_PER_TASK,
            "recent_window": [RECENT_FROM, RECENT_UNTIL],
            "labels_read": False,
        },
        "totals": {
            "candidates": len(coverage),
            "evidence_input_documents": len(output_documents),
            "scientific_selected": sum(
                row["source_class"] == "scientific" for row in output_documents
            ),
            "media_selected": sum(
                row["source_class"] == "industry" for row in output_documents
            ),
            "candidates_without_final_input": sum(
                not row["has_evidence_input"] for row in coverage
            ),
        },
        "inputs": {
            "plan_manifest": _digest((plan_path / MANIFEST_FILENAME).read_bytes()),
            "target_gate_manifest": _digest((gate_path / MANIFEST_FILENAME).read_bytes()),
            "evidence_input_manifest": _digest((input_path / MANIFEST_FILENAME).read_bytes()),
            "media_result_manifest": _digest((media_path / MANIFEST_FILENAME).read_bytes()),
        },
        "outputs": {
            DOCUMENTS_FILENAME: _digest(documents_bytes),
            COVERAGE_FILENAME: _digest(coverage_bytes),
        },
    }
    return documents_bytes, coverage_bytes, _json_bytes(manifest)


def export_hype_evidence_input(
    *,
    plan_dir: str | Path,
    gate_dir: str | Path,
    evidence_input_dir: str | Path,
    media_result_dir: str | Path,
    output_dir: str | Path,
) -> HypeEvidenceInputPaths:
    documents, coverage, manifest = build_hype_evidence_input(
        plan_dir=plan_dir,
        gate_dir=gate_dir,
        evidence_input_dir=evidence_input_dir,
        media_result_dir=media_result_dir,
    )
    paths = publish_artifact_bundle(
        {
            DOCUMENTS_FILENAME: documents,
            COVERAGE_FILENAME: coverage,
            MANIFEST_FILENAME: manifest,
        },
        output_dir,
    )
    return HypeEvidenceInputPaths(
        manifest=paths[MANIFEST_FILENAME],
        documents=paths[DOCUMENTS_FILENAME],
        coverage=paths[COVERAGE_FILENAME],
    )
