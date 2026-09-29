"""Build a deterministic, offline readiness audit for the labeling corpus."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlparse

from nextwave.datasets.artifacts import publish_artifact_bundle

from .finalize import _MODEL_QUOTAS

LABELING_CORPUS_READINESS_VERSION: Final = "labeling-corpus-readiness-v1"
CANDIDATE_STATUS_FILENAME: Final = "candidate_status.jsonl"
DEFICITS_FILENAME: Final = "deficits.json"
MANIFEST_FILENAME: Final = "manifest.json"
ADJUDICATION_MANIFEST_VERSION: Final = "negative-corpus-adjudication-manifest-v6"
ADJUDICATION_VERSION: Final = "negative-corpus-adjudication-v6"
HYPE_EVIDENCE_VERSION: Final = "negative-corpus-hype-evidence-v1"
REVIEW_QUEUE_MANIFEST_VERSION: Final = "negative-hype-review-queue-manifest-v1"
REVIEW_QUEUE_ITEM_VERSION: Final = "negative-hype-review-item-v1"
REVIEW_QUEUE_FILENAME: Final = "review_queue.jsonl"
_START_DATE: Final = date(2025, 9, 15)
_CUTOFF_DATE: Final = date(2026, 9, 15)
_DIGEST_LENGTH: Final = 64
_EVIDENCE_REASON_ORDER: Final = (
    "insufficient_rows",
    "foreign_candidate",
    "bad_date",
    "bad_url",
    "empty_text",
    "name_absent",
    "single_origin",
    "single_publisher",
    "no_substantive_evidence",
)


@dataclass(frozen=True, slots=True)
class LabelingCorpusReadinessPaths:
    """Paths of one successfully published readiness bundle."""

    manifest: Path
    candidate_status: Path
    deficits: Path


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


def _valid_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _DIGEST_LENGTH
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_manifest_files(
    directory: Path, manifest: dict[str, Any], label: str
) -> dict[str, dict[str, dict[str, Any]]]:
    verified: dict[str, dict[str, dict[str, Any]]] = {"inputs": {}, "outputs": {}}
    for section_name in ("inputs", "outputs"):
        section = manifest.get(section_name)
        if not isinstance(section, dict):
            raise ValueError(f"{label} manifest {section_name} must be an object")
        for filename, entry in section.items():
            if (
                not isinstance(filename, str)
                or not filename
                or Path(filename).name != filename
                or not isinstance(entry, dict)
            ):
                raise ValueError(f"{label} manifest has an invalid file entry")
            size = entry.get("size_bytes")
            checksum = entry.get("sha256")
            if set(entry) != {"size_bytes", "sha256"}:
                raise ValueError(f"{label} {filename} has unexpected digest fields")
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise ValueError(f"{label} {filename} has an invalid size")
            if not _valid_digest(checksum):
                raise ValueError(f"{label} {filename} has an invalid sha256")
            expected = {"size_bytes": size, "sha256": checksum}
            verified[section_name][filename] = expected
            if section_name == "inputs":
                continue
            try:
                payload = (directory / filename).read_bytes()
            except OSError as error:
                raise ValueError(f"cannot read {label} file {filename}: {error}") from error
            actual = _digest(payload)
            if actual != expected:
                raise ValueError(f"{label} {filename} digest mismatch")
    return verified


def _load_manifest(
    directory: Path, label: str
) -> tuple[dict[str, Any], bytes, dict[str, dict[str, dict[str, Any]]]]:
    if not directory.is_dir():
        raise ValueError(f"{label} directory does not exist")
    manifest_path = directory / MANIFEST_FILENAME
    manifest, raw = _read_json_with_bytes(manifest_path, f"{label} manifest")
    verified = _verify_manifest_files(directory, manifest, label)
    return manifest, raw, verified


def _read_json_with_bytes(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value, raw


def _require_string(row: dict[str, Any], key: str, label: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} needs non-empty {key}")
    return value


def _require_manifest_counts(
    manifest: dict[str, Any], expected: dict[str, int], label: str
) -> None:
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise ValueError(f"{label} manifest counts must be an object")
    for key, expected_value in expected.items():
        value = counts.get(key)
        if type(value) is not int or value != expected_value:
            raise ValueError(f"{label} manifest count {key} does not match data")


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _evidence_contract(
    candidate: dict[str, Any], evidence: list[dict[str, Any]]
) -> dict[str, Any]:
    canonical_name = _require_string(candidate, "canonical_name", "candidate")
    aliases = candidate.get("aliases", [])
    if not isinstance(aliases, list) or not all(
        isinstance(alias, str) and alias.strip() for alias in aliases
    ):
        raise ValueError("candidate aliases must be a list of non-empty strings")
    names = [_normalize(canonical_name), *(_normalize(alias) for alias in aliases)]

    def headline_only(row: dict[str, Any]) -> bool:
        return (
            row.get("content_status") in ("title_only", "not_fetched")
            or row.get("quote_kind") == "verbatim_headline"
            or row.get("locator") == "headline"
        )

    headline_only_rows = sum(headline_only(row) for row in evidence)
    substantive_rows = len(evidence) - headline_only_rows
    contract: dict[str, Any] = {
        "checked": True,
        "rows_total": len(evidence),
        "substantive_rows": substantive_rows,
        "headline_only_rows": headline_only_rows,
        "origins": len({_normalize(str(row.get("origin_id", ""))) for row in evidence}),
        "publishers": len({_normalize(str(row.get("publisher", ""))) for row in evidence}),
        "reason": None,
    }
    detected_reasons: set[str] = set()
    if len(evidence) < 3:
        detected_reasons.add("insufficient_rows")
    for row in evidence:
        if row.get("candidate_id") != candidate.get("candidate_id"):
            detected_reasons.add("foreign_candidate")
        if row.get("canonical_name") != canonical_name:
            detected_reasons.add("foreign_candidate")
        raw_date = row.get("published_at")
        try:
            published = date.fromisoformat(raw_date) if isinstance(raw_date, str) else None
        except ValueError:
            published = None
        if published is None or not _START_DATE <= published <= _CUTOFF_DATE:
            detected_reasons.add("bad_date")
        parsed_url = urlparse(row.get("url", ""))
        if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
            detected_reasons.add("bad_url")
        quote = row.get("quote")
        headline = row.get("headline")
        quote_ok = isinstance(quote, str) and bool(quote.strip())
        headline_ok = isinstance(headline, str) and bool(headline.strip())
        if not quote_ok and not headline_ok:
            detected_reasons.add("empty_text")
        normalized_quote = _normalize(quote) if isinstance(quote, str) else ""
        normalized_headline = _normalize(headline) if isinstance(headline, str) else ""
        if not any(
            name and (name in normalized_quote or name in normalized_headline)
            for name in names
        ):
            detected_reasons.add("name_absent")
    if contract["origins"] < 2:
        detected_reasons.add("single_origin")
    if contract["publishers"] < 2:
        detected_reasons.add("single_publisher")
    if substantive_rows < 1:
        detected_reasons.add("no_substantive_evidence")
    reasons = [reason for reason in _EVIDENCE_REASON_ORDER if reason in detected_reasons]
    contract["reason"] = reasons[0] if reasons else None
    contract["reasons"] = reasons
    return contract


def _candidate_status(
    candidate: dict[str, Any], evidence_by_candidate: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    candidate_id = _require_string(candidate, "candidate_id", "adjudication row")
    decision_status = candidate.get("decision_status")
    proposed_label = candidate.get("proposed_label")
    if decision_status not in {"accepted", "proposed"}:
        raise ValueError(f"candidate {candidate_id} has an invalid decision_status")
    if proposed_label not in {"mature", "hype"}:
        raise ValueError(f"candidate {candidate_id} has an invalid proposed_label")
    if decision_status == "proposed":
        contract = {
            "checked": False,
            "rows_total": 0,
            "substantive_rows": 0,
            "headline_only_rows": 0,
            "origins": 0,
            "publishers": 0,
            "reason": "proposed_not_accepted",
            "reasons": ["proposed_not_accepted"],
        }
        label_ready = False
    elif proposed_label == "mature":
        contract = {
            "checked": False,
            "rows_total": 0,
            "substantive_rows": 0,
            "headline_only_rows": 0,
            "origins": 0,
            "publishers": 0,
            "reason": None,
            "reasons": [],
        }
        label_ready = True
    else:
        contract = _evidence_contract(candidate, evidence_by_candidate.get(candidate_id, []))
        label_ready = contract["reason"] is None
    return {
        "candidate_id": candidate_id,
        "canonical_name": _require_string(candidate, "canonical_name", "candidate"),
        "domain": _require_string(candidate, "domain", "candidate"),
        "decision_status": decision_status,
        "proposed_label": proposed_label,
        "audit_status": _require_string(candidate, "audit_status", "candidate"),
        "label_ready": label_ready,
        "evidence_contract": contract,
    }


def build_corpus_readiness(
    *, adjudication_dir: Path, review_queue_dir: Path
) -> tuple[bytes, bytes, bytes]:
    """Audit adjudication and review-queue artifacts without external I/O."""
    adjudication_manifest, adjudication_manifest_bytes, adjudication_files = _load_manifest(
        adjudication_dir, "adjudication"
    )
    queue_manifest, queue_manifest_bytes, queue_files = _load_manifest(
        review_queue_dir, "review queue"
    )
    if adjudication_manifest.get("schema_version") != ADJUDICATION_MANIFEST_VERSION:
        raise ValueError("adjudication manifest has the wrong schema_version")
    if adjudication_manifest.get("cutoff_date") != _CUTOFF_DATE.isoformat():
        raise ValueError("adjudication manifest has the wrong cutoff_date")
    queue_schema = queue_manifest.get("schema_version")
    if queue_schema != REVIEW_QUEUE_MANIFEST_VERSION:
        raise ValueError("review queue manifest has the wrong schema_version")
    if queue_manifest.get("cutoff_date") != _CUTOFF_DATE.isoformat():
        raise ValueError("review queue manifest has the wrong cutoff_date")
    if queue_manifest.get("recent_window_from") != _START_DATE.isoformat():
        raise ValueError("review queue manifest has the wrong recent_window_from")
    adjudication_payload = _checked_payload(
        adjudication_dir, "adjudication.jsonl", adjudication_manifest
    )
    evidence_payload = _checked_payload(
        adjudication_dir, "hype_evidence.jsonl", adjudication_manifest
    )
    queue_payload = _checked_payload(review_queue_dir, REVIEW_QUEUE_FILENAME, queue_manifest)
    adjudication_rows = _rows(adjudication_payload, "adjudication.jsonl")
    evidence_rows = _rows(evidence_payload, "hype_evidence.jsonl")
    queue_rows = _rows(queue_payload, REVIEW_QUEUE_FILENAME)
    if len(adjudication_rows) != 100:
        raise ValueError("adjudication.jsonl must contain exactly 100 rows")
    candidates: dict[str, dict[str, Any]] = {}
    for row in adjudication_rows:
        if row.get("schema_version") != ADJUDICATION_VERSION:
            raise ValueError("adjudication row has the wrong schema_version")
        candidate_id = _require_string(row, "candidate_id", "adjudication row")
        if candidate_id in candidates:
            raise ValueError(f"duplicate candidate_id {candidate_id!r}")
        for key in (
            "canonical_name", "domain", "decision_status",
            "proposed_label", "audit_status", "evidence_status",
        ):
            _require_string(row, key, "adjudication row")
        candidates[candidate_id] = row
    evidence_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    evidence_fields = (
        "evidence_id", "candidate_id", "canonical_name", "content_status", "document_id",
        "fact_kind", "headline", "locator", "origin_id", "published_at", "publisher",
        "quote", "quote_kind", "source_class", "trust_tier", "url",
    )
    evidence_ids: set[str] = set()
    for row in evidence_rows:
        if row.get("schema_version") != HYPE_EVIDENCE_VERSION:
            raise ValueError("hype evidence row has the wrong schema_version")
        for key in evidence_fields:
            _require_string(row, key, "hype evidence row")
        evidence_id = row["evidence_id"]
        if evidence_id in evidence_ids:
            raise ValueError(f"duplicate hype evidence_id {evidence_id!r}")
        evidence_ids.add(evidence_id)
        evidence_by_candidate[row["candidate_id"]].append(row)
    if len(queue_rows) != 50:
        raise ValueError("review_queue.jsonl must contain exactly 50 rows")
    queue_by_candidate: dict[str, dict[str, Any]] = {}
    for row in queue_rows:
        if row.get("schema_version") != REVIEW_QUEUE_ITEM_VERSION:
            raise ValueError("review queue row has the wrong schema_version")
        candidate_id = _require_string(row, "candidate_id", "review queue row")
        if candidate_id in queue_by_candidate:
            raise ValueError(f"duplicate review queue candidate_id {candidate_id!r}")
        for key in ("canonical_name", "domain", "review_status"):
            _require_string(row, key, "review queue row")
        if row["review_status"] not in {
            "needs_manual_publicity_review",
            "replace_or_targeted_search",
        }:
            raise ValueError(f"review queue candidate {candidate_id} has invalid review_status")
        if type(row.get("search_coverage_complete")) is not bool:
            raise ValueError(
                f"review queue candidate {candidate_id} needs boolean search coverage"
            )
        for key in ("technical_ab_origin_ids", "pilot_ab_origin_ids", "media"):
            value = row.get(key)
            if not isinstance(value, list):
                raise ValueError(f"review queue candidate {candidate_id} needs list {key}")
            if key != "media" and not all(isinstance(item, str) and item for item in value):
                raise ValueError(f"review queue candidate {candidate_id} has invalid {key}")
            if key == "media" and not all(isinstance(item, dict) for item in value):
                raise ValueError(f"review queue candidate {candidate_id} has invalid media")
        for key in ("exact_title_groups", "publisher_hosts", "retrieved_recent_media"):
            value = row.get(key)
            if type(value) is not int or value < 0:
                raise ValueError(f"review queue candidate {candidate_id} has invalid {key}")
        if type(row.get("has_three_title_groups_two_hosts")) is not bool:
            raise ValueError(f"review queue candidate {candidate_id} has invalid title flag")
        queue_by_candidate[candidate_id] = row
    hype_candidate_ids = {
        candidate_id
        for candidate_id, candidate in candidates.items()
        if candidate.get("proposed_label") == "hype"
    }
    missing_hype = sorted(hype_candidate_ids - queue_by_candidate.keys())
    if missing_hype:
        raise ValueError(f"review queue misses hype candidates: {', '.join(missing_hype)}")
    for candidate_id in sorted(hype_candidate_ids):
        candidate = candidates[candidate_id]
        queue_row = queue_by_candidate[candidate_id]
        if (
            queue_row["canonical_name"] != candidate["canonical_name"]
            or queue_row["domain"] != candidate["domain"]
        ):
            raise ValueError(f"review queue candidate {candidate_id} identity mismatch")
    queue_status_counts = Counter(row["review_status"] for row in queue_rows)
    _require_manifest_counts(
        queue_manifest,
        {
            "candidates": len(queue_rows),
            "needs_manual_publicity_review": queue_status_counts[
                "needs_manual_publicity_review"
            ],
            "replace_or_targeted_search": queue_status_counts["replace_or_targeted_search"],
            "with_retrieved_recent_media": sum(
                row["retrieved_recent_media"] > 0 for row in queue_rows
            ),
            "with_three_title_groups_two_hosts": sum(
                row["has_three_title_groups_two_hosts"] for row in queue_rows
            ),
        },
        "review queue",
    )
    statuses = sorted(
        (_candidate_status(candidate, evidence_by_candidate) for candidate in candidates.values()),
        key=lambda row: row["candidate_id"],
    )
    counts = Counter(row["decision_status"] for row in statuses)

    def is_accepted_mature(row: dict[str, Any]) -> bool:
        return (
            row["decision_status"] == "accepted"
            and row["proposed_label"] == "mature"
        )

    def is_accepted_hype(row: dict[str, Any]) -> bool:
        return (
            row["decision_status"] == "accepted"
            and row["proposed_label"] == "hype"
        )

    def is_proposed_hype(row: dict[str, Any]) -> bool:
        return (
            row["decision_status"] == "proposed"
            and row["proposed_label"] == "hype"
        )

    accepted_mature = sum(is_accepted_mature(row) for row in statuses)
    accepted_hype = sum(is_accepted_hype(row) for row in statuses)
    proposed_hype = sum(is_proposed_hype(row) for row in statuses)
    _require_manifest_counts(
        adjudication_manifest,
        {
            "rows": len(statuses),
            "accepted": counts["accepted"],
            "accepted_mature": accepted_mature,
            "accepted_hype": accepted_hype,
            "proposed_hype": proposed_hype,
            "mature_pool": sum(row["proposed_label"] == "mature" for row in statuses),
            "hype_pool": sum(row["proposed_label"] == "hype" for row in statuses),
            "hype_evidence": len(evidence_rows),
        },
        "adjudication",
    )
    by_domain: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "accepted_mature": 0,
            "accepted_hype": 0,
            "proposed_hype": 0,
            "label_ready_hype": 0,
        }
    )
    reasons: Counter[str] = Counter()
    headline_only_hype = 0
    for row in statuses:
        domain = row["domain"]
        if row["decision_status"] == "accepted" and row["proposed_label"] == "mature":
            by_domain[domain]["accepted_mature"] += 1
        if row["decision_status"] == "accepted" and row["proposed_label"] == "hype":
            by_domain[domain]["accepted_hype"] += 1
            if row["evidence_contract"]["headline_only_rows"]:
                headline_only_hype += 1
        if row["decision_status"] == "proposed" and row["proposed_label"] == "hype":
            by_domain[domain]["proposed_hype"] += 1
        if row["label_ready"] and row["proposed_label"] == "hype":
            by_domain[domain]["label_ready_hype"] += 1
        reasons.update(row["evidence_contract"]["reasons"])
    ready_for_finalization = accepted_hype == 50 and all(
        by_domain[domain]["label_ready_hype"] >= quota["marketing_hype"]
        and by_domain[domain]["accepted_mature"] >= quota["mature"]
        for domain, quota in _MODEL_QUOTAS.items()
    )
    deficit_value = {
        "counts": {
            "rows": len(statuses),
            "accepted": counts["accepted"],
            "accepted_mature": accepted_mature,
            "accepted_hype": accepted_hype,
            "proposed": counts["proposed"],
            "proposed_hype": proposed_hype,
            "label_ready_mature": sum(
                row["label_ready"] and row["proposed_label"] == "mature"
                for row in statuses
            ),
            "label_ready_hype": sum(
                row["label_ready"] and row["proposed_label"] == "hype"
                for row in statuses
            ),
            "headline_only_hype": headline_only_hype,
            "review_queue_rows": len(queue_rows),
            "review_queue_hype_candidates": len(hype_candidate_ids),
        },
        "by_domain": {domain: by_domain[domain] for domain in sorted(by_domain)},
        "reasons": dict(sorted(reasons.items())),
        "ready_for_finalization": ready_for_finalization,
        "expected": {"accepted_mature": 57, "accepted_hype": 7, "proposed_hype": 36},
    }
    candidate_status_bytes = _jsonl_bytes(statuses)
    deficits_bytes = _json_bytes(deficit_value)
    manifest = {
        "schema_version": LABELING_CORPUS_READINESS_VERSION,
        "cutoff_date": "2026-09-15",
        "inputs": {
            "adjudication_manifest": _digest(adjudication_manifest_bytes),
            "review_queue_manifest": _digest(queue_manifest_bytes),
            "adjudication_files": adjudication_files,
            "review_queue_files": queue_files,
            "review_queue_schema_version": queue_schema,
        },
        "outputs": {
            CANDIDATE_STATUS_FILENAME: _digest(candidate_status_bytes),
            DEFICITS_FILENAME: _digest(deficits_bytes),
        },
        "totals": deficit_value["counts"],
    }
    return candidate_status_bytes, deficits_bytes, _json_bytes(manifest)


def export_corpus_readiness(
    *, adjudication_dir: Path, review_queue_dir: Path, output_dir: Path
) -> LabelingCorpusReadinessPaths:
    """Publish one immutable readiness bundle after a successful audit."""
    candidate_status, deficits, manifest = build_corpus_readiness(
        adjudication_dir=adjudication_dir, review_queue_dir=review_queue_dir
    )
    paths = publish_artifact_bundle(
        {
            CANDIDATE_STATUS_FILENAME: candidate_status,
            DEFICITS_FILENAME: deficits,
            MANIFEST_FILENAME: manifest,
        },
        output_dir,
    )
    return LabelingCorpusReadinessPaths(
        manifest=paths[MANIFEST_FILENAME],
        candidate_status=paths[CANDIDATE_STATUS_FILENAME],
        deficits=paths[DEFICITS_FILENAME],
    )
