"""Build a reviewed cross-corpus identity map for grouped evaluation."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nextwave.datasets.artifacts import publish_artifact_bundle

IDENTITY_REVIEW_VERSION = "candidate-identity-review-v1"
IDENTITY_DECISIONS_VERSION = "candidate-identity-decisions-v1"
_PLAN_VERSION = "labeling-enrichment-plan-v2"
_CUTOFF = "2026-09-15"
_RELATIONS = {"same_candidate", "same_family", "distinct"}
_GENERIC_TOKENS = {"ai", "model", "models", "system", "systems", "technology", "technologies"}
_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{2,127}\Z")


@dataclass(frozen=True, slots=True)
class IdentityReviewPaths:
    identities: Path
    pair_decisions: Path
    manifest: Path


def _digest(payload: bytes) -> dict[str, Any]:
    return {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def normalize_identity(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


def _load_plan(directory: Path, corpus: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = _read_object(directory / "manifest.json", f"{corpus} plan manifest")
    if manifest.get("schema_version") != _PLAN_VERSION:
        raise ValueError(f"{corpus} plan must use {_PLAN_VERSION}")
    output = (manifest.get("outputs") or {}).get("plan.json")
    if not isinstance(output, dict):
        raise ValueError(f"{corpus} plan manifest misses plan.json")
    payload = (directory / "plan.json").read_bytes()
    if output != _digest(payload):
        raise ValueError(f"{corpus} plan.json diverges from manifest")
    plan = json.loads(payload)
    if not isinstance(plan, dict) or plan.get("cutoff_date") != _CUTOFF:
        raise ValueError(f"{corpus} plan cutoff must be {_CUTOFF}")
    raw_candidates = plan.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError(f"{corpus} plan candidates must be a list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_candidates, 1):
        if not isinstance(raw, dict):
            raise ValueError(f"{corpus} candidate {index} must be an object")
        candidate_id = _require_text(raw.get("candidate_id"), "candidate_id")
        if _SAFE_ID.fullmatch(candidate_id) is None or candidate_id in seen:
            raise ValueError(f"invalid or duplicate candidate_id {candidate_id!r}")
        seen.add(candidate_id)
        canonical = _require_text(raw.get("canonical_name"), "canonical_name")
        domain = _require_text(raw.get("domain"), "domain")
        aliases = raw.get("aliases")
        search_terms = raw.get("search_terms")
        if not isinstance(aliases, list) or not isinstance(search_terms, list):
            raise ValueError(f"candidate {candidate_id} aliases/search_terms must be lists")
        names = [canonical]
        for label, values in (("alias", aliases), ("search term", search_terms)):
            for value in values:
                names.append(_require_text(value, f"candidate {candidate_id} {label}"))
        forms = sorted({normalize_identity(value) for value in names})
        if any(not form for form in forms):
            raise ValueError(f"candidate {candidate_id} has blank normalized identity")
        result.append(
            {
                "candidate_id": candidate_id,
                "canonical_name": canonical,
                "corpus": corpus,
                "domain": domain,
                "forms": forms,
            }
        )
    return sorted(result, key=lambda row: row["candidate_id"]), manifest


def _best_trigger(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any] | None:
    matches: list[tuple[int, float, str, str, str]] = []
    for left_form in left["forms"]:
        for right_form in right["forms"]:
            left_tokens, right_tokens = set(left_form.split()), set(right_form.split())
            common = left_tokens & right_tokens
            union = left_tokens | right_tokens
            jaccard = len(common) / len(union)
            exact = left_form == right_form
            shorter = min((left_form, right_form), key=len)
            substring = len(shorter) >= 5 and (
                left_form in right_form or right_form in left_form
            )
            fuzzy = (len(common - _GENERIC_TOKENS) >= 2 and jaccard >= 0.5) or substring
            if exact or fuzzy:
                matches.append(
                    (
                        2 if exact else 1,
                        jaccard,
                        left_form,
                        right_form,
                        "exact" if exact else "fuzzy",
                    )
                )
    if not matches:
        return None
    _, score, left_form, right_form, trigger = max(matches)
    return {
        "trigger": trigger,
        "left_form": left_form,
        "right_form": right_form,
        "token_jaccard": round(score, 6),
    }


def _suggestions(
    positives: list[dict[str, Any]], negatives: list[dict[str, Any]]
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for left in positives:
        for right in negatives:
            match = _best_trigger(left, right)
            if match is not None:
                result[(left["candidate_id"], right["candidate_id"])] = match
    return result


def _load_decisions(
    path: Path,
    positives: Mapping[str, dict[str, Any]],
    negatives: Mapping[str, dict[str, Any]],
    suggestions: Mapping[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    payload = _read_object(path, "identity decisions")
    if payload.get("schema_version") != IDENTITY_DECISIONS_VERSION:
        raise ValueError(f"identity decisions must use {IDENTITY_DECISIONS_VERSION}")
    raw_pairs = payload.get("pairs")
    if not isinstance(raw_pairs, list):
        raise ValueError("identity decision pairs must be a list")
    decisions: dict[tuple[str, str], dict[str, Any]] = {}
    for index, raw in enumerate(raw_pairs, 1):
        if not isinstance(raw, dict):
            raise ValueError(f"identity decision {index} must be an object")
        left = _require_text(raw.get("left_candidate_id"), "left_candidate_id")
        right = _require_text(raw.get("right_candidate_id"), "right_candidate_id")
        if left not in positives or right not in negatives:
            raise ValueError(f"identity decision {index} must be positive-to-negative")
        pair = (left, right)
        if pair in decisions:
            raise ValueError(f"duplicate identity decision {pair!r}")
        relation = _require_text(raw.get("relation"), "relation")
        if relation not in _RELATIONS:
            raise ValueError(f"identity decision {pair!r} has invalid relation")
        rationale = _require_text(raw.get("rationale"), "rationale")
        match = suggestions.get(pair)
        if match is not None and match["trigger"] == "exact" and relation == "distinct":
            raise ValueError(f"exact identity pair {pair!r} cannot be distinct")
        decisions[pair] = {
            "left_candidate_id": left,
            "right_candidate_id": right,
            "relation": relation,
            "rationale": rationale,
            **(
                match
                or {
                    "trigger": "manual",
                    "left_form": None,
                    "right_form": None,
                    "token_jaccard": None,
                }
            ),
        }
    missing = sorted(set(suggestions) - set(decisions))
    if missing:
        raise ValueError(f"identity decisions miss suggested pair {missing[0]!r}")
    return [decisions[pair] for pair in sorted(decisions)]


def build_identity_review(
    *, positive_plan_dir: str | Path, negative_plan_dir: str | Path, decisions_path: str | Path
) -> tuple[bytes, bytes, bytes]:
    positive_path, negative_path = Path(positive_plan_dir), Path(negative_plan_dir)
    positives, positive_manifest = _load_plan(positive_path, "positive")
    negatives, negative_manifest = _load_plan(negative_path, "negative")
    positive_index = {row["candidate_id"]: row for row in positives}
    negative_index = {row["candidate_id"]: row for row in negatives}
    suggestions = _suggestions(positives, negatives)
    decisions = _load_decisions(
        Path(decisions_path), positive_index, negative_index, suggestions
    )
    all_rows = positives + negatives
    parents = {row["candidate_id"]: row["candidate_id"] for row in all_rows}

    def find(value: str) -> str:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            first, second = sorted((left_root, right_root))
            parents[second] = first

    for decision in decisions:
        if decision["relation"] in {"same_candidate", "same_family"}:
            union(decision["left_candidate_id"], decision["right_candidate_id"])
    members: dict[str, list[str]] = {}
    for candidate_id in parents:
        members.setdefault(find(candidate_id), []).append(candidate_id)
    groups: dict[str, str] = {}
    for candidate_ids in members.values():
        identity = "|".join(sorted(candidate_ids))
        group_id = "cross-corpus-" + hashlib.sha256(identity.encode()).hexdigest()[:16]
        groups.update({candidate_id: group_id for candidate_id in candidate_ids})
    identity_rows = [
        {
            "candidate_id": row["candidate_id"],
            "canonical_name": row["canonical_name"],
            "corpus": row["corpus"],
            "group_id": groups[row["candidate_id"]],
            "identity_status": "reviewed",
            "schema_version": IDENTITY_REVIEW_VERSION,
        }
        for row in sorted(all_rows, key=lambda item: item["candidate_id"])
    ]
    identities_bytes = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in identity_rows
    ).encode()
    pairs_bytes = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in decisions
    ).encode()
    conflicts = sum(row["relation"] == "same_candidate" for row in decisions)
    manifest_value = {
        "schema_version": IDENTITY_REVIEW_VERSION,
        "cutoff_date": _CUTOFF,
        "ready_for_model": conflicts == 0,
        "totals": {
            "candidates": len(identity_rows),
            "pairs": len(decisions),
            "exact": sum(row["trigger"] == "exact" for row in decisions),
            "fuzzy": sum(row["trigger"] == "fuzzy" for row in decisions),
            "manual": sum(row["trigger"] == "manual" for row in decisions),
            "same_candidate": conflicts,
            "same_family": sum(row["relation"] == "same_family" for row in decisions),
            "distinct": sum(row["relation"] == "distinct" for row in decisions),
            "conflicts": conflicts,
        },
        "inputs": {
            "positive_plan_manifest": _digest((positive_path / "manifest.json").read_bytes()),
            "negative_plan_manifest": _digest((negative_path / "manifest.json").read_bytes()),
            "decisions": _digest(Path(decisions_path).read_bytes()),
            "positive_bundle_id": positive_manifest.get("bundle_id"),
            "negative_bundle_id": negative_manifest.get("bundle_id"),
        },
        "outputs": {
            "identities.jsonl": _digest(identities_bytes),
            "pair_decisions.jsonl": _digest(pairs_bytes),
        },
    }
    manifest_bytes = (
        json.dumps(manifest_value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode()
    return identities_bytes, pairs_bytes, manifest_bytes


def export_identity_review(
    *,
    positive_plan_dir: str | Path,
    negative_plan_dir: str | Path,
    decisions_path: str | Path,
    output_dir: str | Path,
) -> IdentityReviewPaths:
    identities, pairs, manifest = build_identity_review(
        positive_plan_dir=positive_plan_dir,
        negative_plan_dir=negative_plan_dir,
        decisions_path=decisions_path,
    )
    output = Path(output_dir)
    publish_artifact_bundle(
        {
            "identities.jsonl": identities,
            "pair_decisions.jsonl": pairs,
            "manifest.json": manifest,
        },
        output,
    )
    return IdentityReviewPaths(
        identities=output / "identities.jsonl",
        pair_decisions=output / "pair_decisions.jsonl",
        manifest=output / "manifest.json",
    )
