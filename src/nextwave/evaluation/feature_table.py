"""Build one leakage-safe development feature table for model evaluation.

The builder deliberately refuses to turn provisional labeling suggestions into
targets.  It can publish a development table from the currently accepted rows,
but marks it ineligible for qualification until the frozen 100/50/50 roster and
reviewed cross-corpus identities exist.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from nextwave.datasets.artifacts import publish_artifact_bundle

FEATURE_TABLE_VERSION = "candidate-feature-table-v2"
FEATURES_FILENAME = "features.jsonl"
MANIFEST_FILENAME = "manifest.json"
_CUTOFF = date(2026, 9, 15)
_WINDOW_START = date(2024, 9, 15)
_RECENT_START = date(2025, 9, 15)
_TEMPORAL_COUNT_RESULT_VERSION = "openalex-temporal-count-result-v4"
_IDENTITY_REVIEW_VERSION = "candidate-identity-review-v1"
_TEMPORAL_FEATURE_KEYS = (
    "scientific_previous_count_log1p",
    "scientific_recent_count_log1p",
    "scientific_count_log_growth",
    "scientific_scope_share_previous",
    "scientific_scope_share_recent",
    "scientific_scope_share_delta",
)


@dataclass(frozen=True, slots=True)
class FeatureTablePaths:
    features: Path
    manifest: Path


def _digest(payload: bytes) -> dict[str, Any]:
    return {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _manifest_digest(directory: Path) -> dict[str, Any]:
    return _digest((directory / MANIFEST_FILENAME).read_bytes())


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    result: list[dict[str, Any]] = []
    for lineno, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label} line {lineno} is invalid JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"{label} line {lineno} must be an object")
        result.append(value)
    return result


def _checked_file(directory: Path, manifest: Mapping[str, Any], filename: str) -> bytes:
    path = directory / filename
    outputs = manifest.get("outputs")
    if isinstance(outputs, dict):
        entry = outputs.get(filename)
    elif isinstance(outputs, list):
        matches = [
            item for item in outputs if isinstance(item, dict) and item.get("filename") == filename
        ]
        if len(matches) != 1:
            entry = None
        else:
            entry = {
                "size_bytes": matches[0].get("size_bytes"),
                "sha256": matches[0].get("sha256"),
            }
    else:
        entry = None
    if not isinstance(entry, dict) or not path.is_file():
        raise ValueError(f"{directory.name} misses {filename}")
    payload = path.read_bytes()
    if entry != _digest(payload):
        raise ValueError(f"{filename} diverges from {directory.name} manifest")
    return payload


def _manifest_output_rows(directory: Path, filename: str) -> list[dict[str, Any]]:
    manifest = _read_json(directory / MANIFEST_FILENAME, f"{directory.name} manifest")
    payload = _checked_file(directory, manifest, filename)
    return _rows_from_bytes(payload, filename)


def _rows_from_bytes(payload: bytes, label: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for lineno, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{label} line {lineno} must be an object")
        result.append(value)
    return result


def _index(rows: Iterable[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} needs {key}")
        if value in result:
            raise ValueError(f"{label} duplicates {key} {value!r}")
        result[value] = row
    return result


def _identity_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join("".join(char if char.isalnum() else " " for char in normalized).split())


def _identity_groups(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Join only exact normalized canonical/alias intersections.

    Semantic aliases remain for review; this conservative grouping prevents the
    same spelling from crossing folds without inventing equivalence.
    """

    parents = {row["candidate_id"]: row["candidate_id"] for row in rows}

    def find(value: str) -> str:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        parents[second] = first

    owners: dict[str, str] = {}
    for row in sorted(rows, key=lambda item: item["candidate_id"]):
        names = [row["canonical_name"], *row["aliases"]]
        for name in names:
            normalized = _identity_text(name)
            if not normalized:
                raise ValueError(f"candidate {row['candidate_id']} has blank identity text")
            previous = owners.setdefault(normalized, row["candidate_id"])
            union(previous, row["candidate_id"])

    members: dict[str, list[str]] = defaultdict(list)
    for candidate_id in parents:
        members[find(candidate_id)].append(candidate_id)
    result: dict[str, str] = {}
    for candidate_ids in members.values():
        identity = "|".join(sorted(candidate_ids))
        group_id = f"cross-corpus-{hashlib.sha256(identity.encode()).hexdigest()[:16]}"
        for candidate_id in candidate_ids:
            result[candidate_id] = group_id
    return result


def _reviewed_identity_groups(
    directory: Path,
    rows: list[dict[str, Any]],
) -> tuple[dict[str, str], bool, int]:
    manifest = _read_json(directory / MANIFEST_FILENAME, "identity review manifest")
    if manifest.get("schema_version") != _IDENTITY_REVIEW_VERSION:
        raise ValueError(f"identity review must use {_IDENTITY_REVIEW_VERSION}")
    identity_rows = _rows_from_bytes(
        _checked_file(directory, manifest, "identities.jsonl"),
        "identities",
    )
    pair_rows = _rows_from_bytes(
        _checked_file(directory, manifest, "pair_decisions.jsonl"),
        "identity pair decisions",
    )
    identity_index = _index(identity_rows, "candidate_id", "identity")
    expected = {row["candidate_id"]: row for row in rows}
    missing = sorted(set(expected) - set(identity_index))
    if missing:
        raise ValueError(f"identity review misses candidate {missing[0]!r}")
    groups: dict[str, str] = {}
    for candidate_id, candidate in expected.items():
        identity = identity_index[candidate_id]
        if identity.get("identity_status") != "reviewed":
            raise ValueError(f"identity {candidate_id!r} is not reviewed")
        if identity.get("canonical_name") != candidate["canonical_name"]:
            raise ValueError(f"identity {candidate_id!r} canonical_name differs")
        group_id = identity.get("group_id")
        if not isinstance(group_id, str) or not group_id.startswith("cross-corpus-"):
            raise ValueError(f"identity {candidate_id!r} has invalid group_id")
        groups[candidate_id] = group_id
    selected_ids = set(expected)
    conflicts = 0
    for index, pair in enumerate(pair_rows, 1):
        left = pair.get("left_candidate_id")
        right = pair.get("right_candidate_id")
        relation = pair.get("relation")
        if not isinstance(left, str) or not isinstance(right, str):
            raise ValueError(f"identity pair decision {index} has invalid IDs")
        if relation == "same_candidate" and {left, right} <= selected_ids:
            conflicts += 1
    return groups, conflicts == 0, conflicts


def _coverage_index(directory: Path, candidate_ids: set[str]) -> dict[str, dict[str, bool]]:
    manifest = _read_json(directory / MANIFEST_FILENAME, f"{directory.name} manifest")
    rows = _rows_from_bytes(_checked_file(directory, manifest, "coverage.jsonl"), "coverage")
    coverage = {
        candidate_id: {"scientific": False, "industry": False} for candidate_id in candidate_ids
    }
    seen: set[tuple[str, str]] = set()
    for row in rows:
        candidate_id = row.get("candidate_id")
        source_class = row.get("source_class")
        pair = (candidate_id, source_class)
        if candidate_id not in coverage or source_class not in {"scientific", "industry"}:
            raise ValueError("coverage contains an unknown candidate/source class")
        if pair in seen:
            raise ValueError("coverage duplicates candidate/source class")
        seen.add(pair)
        coverage[candidate_id][source_class] = row.get("status") == "complete"
    if len(seen) != 2 * len(candidate_ids):
        raise ValueError("coverage does not contain two source classes per candidate")
    return coverage


def _document_features(
    directory: Path,
    candidate_ids: set[str],
    *,
    window_start: date = _WINDOW_START,
    recent_start: date = _RECENT_START,
    cutoff: date = _CUTOFF,
) -> dict[str, dict[str, bool]]:
    manifest = _read_json(directory / MANIFEST_FILENAME, f"{directory.name} manifest")
    rows = _rows_from_bytes(_checked_file(directory, manifest, "documents.jsonl"), "documents")
    result = {
        candidate_id: {
            "scientific_previous_present": False,
            "scientific_recent_present": False,
            "industry_previous_present": False,
            "industry_recent_present": False,
            "unknown_date_present": False,
            "trust_ab_present": False,
            "multiple_origins_present": False,
            "multiple_organizations_present": False,
            "multiple_source_types_present": False,
        }
        for candidate_id in candidate_ids
    }
    origins: dict[str, set[str]] = defaultdict(set)
    organizations: dict[str, set[str]] = defaultdict(set)
    source_types: dict[str, set[str]] = defaultdict(set)
    seen_pairs: set[tuple[str, str]] = set()
    for row in rows:
        candidate_id = row.get("candidate_id")
        document_id = row.get("document_id")
        if candidate_id not in result or not isinstance(document_id, str):
            raise ValueError("document references an unknown candidate or lacks document_id")
        pair = (candidate_id, document_id)
        if pair in seen_pairs:
            raise ValueError("documents duplicate candidate/document")
        seen_pairs.add(pair)
        connector = row.get("connector")
        source_class = (
            "scientific"
            if connector == "openalex"
            else "industry"
            if connector in {"mediacloud", "exa"}
            else None
        )
        if source_class is None:
            raise ValueError(f"unsupported feature connector {connector!r}")
        published = row.get("published_at")
        if published is None:
            result[candidate_id]["unknown_date_present"] = True
        elif isinstance(published, str):
            published_date = date.fromisoformat(published)
            if not window_start <= published_date <= cutoff:
                raise ValueError(f"document {document_id} is outside the frozen feature window")
            window = "previous" if published_date < recent_start else "recent"
            result[candidate_id][f"{source_class}_{window}_present"] = True
        else:
            raise ValueError(f"document {document_id} has invalid published_at")
        if row.get("trust_tier") in {"A", "B"}:
            result[candidate_id]["trust_ab_present"] = True
        origin_id = row.get("origin_id")
        if isinstance(origin_id, str) and origin_id:
            origins[candidate_id].add(origin_id)
        for organization in row.get("organizations") or []:
            if isinstance(organization, str) and organization:
                organizations[candidate_id].add(organization.casefold())
        source_type = row.get("source_type")
        if isinstance(source_type, str) and source_type:
            source_types[candidate_id].add(source_type)
    for candidate_id in candidate_ids:
        result[candidate_id]["multiple_origins_present"] = len(origins[candidate_id]) >= 2
        result[candidate_id]["multiple_organizations_present"] = (
            len(organizations[candidate_id]) >= 2
        )
        result[candidate_id]["multiple_source_types_present"] = len(source_types[candidate_id]) >= 2
    return result


def _temporal_feature_index(
    directory: Path | None, candidate_ids: set[str]
) -> dict[str, dict[str, bool | float | None]]:
    empty = {
        candidate_id: {
            "temporal_count_coverage_complete": False,
            **{key: None for key in _TEMPORAL_FEATURE_KEYS},
        }
        for candidate_id in candidate_ids
    }
    if directory is None:
        return empty
    manifest = _read_json(directory / MANIFEST_FILENAME, "temporal count manifest")
    if (
        manifest.get("schema_version") != _TEMPORAL_COUNT_RESULT_VERSION
        or manifest.get("status") != "complete"
    ):
        raise ValueError("temporal count result must be complete v4")
    rows = _rows_from_bytes(
        _checked_file(
            directory,
            manifest,
            "candidate_temporal_features.jsonl",
        ),
        "candidate temporal features",
    )
    index = _index(rows, "candidate_id", "candidate temporal feature")
    missing = candidate_ids - set(index)
    if missing:
        raise ValueError(f"temporal count result misses candidates: {sorted(missing)}")
    result: dict[str, dict[str, bool | float | None]] = {}
    for candidate_id in candidate_ids:
        row = index[candidate_id]
        counts = row.get("counts")
        if row.get("coverage") != "complete" or not isinstance(counts, dict):
            raise ValueError(f"temporal count coverage is not complete for {candidate_id}")
        expected_count_keys = {
            "candidate_previous",
            "candidate_recent",
            "scope_previous",
            "scope_recent",
        }
        if set(counts) != expected_count_keys or any(
            type(value) is not int or value < 0 for value in counts.values()
        ):
            raise ValueError(f"temporal counts are invalid for {candidate_id}")
        for window in ("previous", "recent"):
            if counts[f"candidate_{window}"] > counts[f"scope_{window}"]:
                raise ValueError(
                    f"temporal candidate count exceeds scope count for {candidate_id} {window}"
                )
        expected_growth = math.log1p(counts["candidate_recent"]) - math.log1p(
            counts["candidate_previous"]
        )
        expected_shares = {
            window: (
                counts[f"candidate_{window}"] / counts[f"scope_{window}"]
                if counts[f"scope_{window}"] > 0
                else None
            )
            for window in ("previous", "recent")
        }
        expected_delta = (
            expected_shares["recent"] - expected_shares["previous"]
            if expected_shares["recent"] is not None
            and expected_shares["previous"] is not None
            else None
        )
        expected_derived = {
            "candidate_log_growth": expected_growth,
            "scope_share_previous": expected_shares["previous"],
            "scope_share_recent": expected_shares["recent"],
            "scope_share_delta": expected_delta,
        }
        for key, expected in expected_derived.items():
            actual = row.get(key)
            if expected is None:
                if actual is not None:
                    raise ValueError(f"temporal feature {key} diverges for {candidate_id}")
            elif type(actual) not in {int, float} or not math.isclose(
                actual, expected, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"temporal feature {key} diverges for {candidate_id}")
        numeric = {
            "scientific_previous_count_log1p": math.log1p(counts["candidate_previous"]),
            "scientific_recent_count_log1p": math.log1p(counts["candidate_recent"]),
            "scientific_count_log_growth": row.get("candidate_log_growth"),
            "scientific_scope_share_previous": row.get("scope_share_previous"),
            "scientific_scope_share_recent": row.get("scope_share_recent"),
            "scientific_scope_share_delta": row.get("scope_share_delta"),
        }
        for key, value in numeric.items():
            if value is not None and (type(value) not in {int, float} or not math.isfinite(value)):
                raise ValueError(f"temporal feature {key} is invalid for {candidate_id}")
        result[candidate_id] = {
            "temporal_count_coverage_complete": True,
            **numeric,
        }
    return result


def _candidate_row(
    raw: Mapping[str, Any],
    *,
    candidate_id: str,
    target: int,
    negative_class: str | None,
) -> dict[str, Any]:
    canonical = raw.get("canonical_name")
    aliases = raw.get("aliases")
    if not isinstance(canonical, str) or not canonical.strip():
        raise ValueError(f"candidate {candidate_id} needs canonical_name")
    if not isinstance(aliases, list) or any(not isinstance(item, str) for item in aliases):
        raise ValueError(f"candidate {candidate_id} aliases must be a list of strings")
    cutoff = raw.get("cutoff_date")
    if cutoff != _CUTOFF.isoformat():
        raise ValueError(f"candidate {candidate_id} cutoff is not frozen")
    domain = raw.get("domain")
    scope = raw.get("analysis_scope_key")
    if not isinstance(domain, str) or not isinstance(scope, str):
        raise ValueError(f"candidate {candidate_id} needs domain/scope")
    return {
        "candidate_id": candidate_id,
        "canonical_name": canonical.strip(),
        "aliases": aliases,
        "domain": domain,
        "analysis_scope_key": scope,
        "cutoff_date": cutoff,
        "target": target,
        "negative_class": negative_class,
    }


def build_feature_table(
    *,
    positive_dir: str | Path,
    positive_enrichment_dir: str | Path,
    negative_plan_dir: str | Path,
    negative_enrichment_dir: str | Path,
    adjudication_dir: str | Path,
    temporal_count_dir: str | Path | None = None,
    identity_review_dir: str | Path | None = None,
) -> tuple[bytes, bytes]:
    positive_path = Path(positive_dir)
    positive_manifest = _read_json(positive_path / MANIFEST_FILENAME, "positive manifest")
    positive_payload = _checked_file(positive_path, positive_manifest, "positive_candidates.jsonl")
    positives = _index(
        _rows_from_bytes(positive_payload, "positive candidates"),
        "record_id",
        "positive candidate",
    )

    negative_plan_path = Path(negative_plan_dir)
    negative_plan_manifest = _read_json(
        negative_plan_path / MANIFEST_FILENAME, "negative plan manifest"
    )
    plan_payload = _checked_file(negative_plan_path, negative_plan_manifest, "plan.json")
    plan = json.loads(plan_payload)
    if not isinstance(plan, dict) or not isinstance(plan.get("candidates"), list):
        raise ValueError("negative plan candidates must be a list")
    negatives = _index(plan["candidates"], "candidate_id", "negative candidate")

    adjudication_path = Path(adjudication_dir)
    adjudication_manifest = _read_json(
        adjudication_path / MANIFEST_FILENAME, "adjudication manifest"
    )
    adjudication_rows = _rows_from_bytes(
        _checked_file(adjudication_path, adjudication_manifest, "adjudication.jsonl"),
        "adjudication",
    )
    decisions = _index(adjudication_rows, "candidate_id", "adjudication")
    if set(decisions) != set(negatives):
        raise ValueError("adjudication roster differs from negative plan")

    candidate_rows: list[dict[str, Any]] = []
    positive_ids = set(positives)
    for candidate_id in sorted(positives):
        raw = positives[candidate_id]
        row = _candidate_row(raw, candidate_id=candidate_id, target=1, negative_class=None)
        row["identity_reviewed"] = raw.get("identity_status") == "reviewed"
        candidate_rows.append(row)

    accepted_negative_ids: set[str] = set()
    accepted_counts: Counter[str] = Counter()
    for candidate_id in sorted(negatives):
        decision = decisions[candidate_id]
        if decision.get("decision_status") != "accepted":
            continue
        proposed_label = decision.get("proposed_label")
        if proposed_label not in {"mature", "hype"}:
            raise ValueError(f"accepted candidate {candidate_id} has invalid label")
        accepted_negative_ids.add(candidate_id)
        accepted_counts[proposed_label] += 1
        row = _candidate_row(
            negatives[candidate_id],
            candidate_id=candidate_id,
            target=0,
            negative_class="marketing_hype" if proposed_label == "hype" else "mature",
        )
        row["identity_reviewed"] = False
        candidate_rows.append(row)

    positive_coverage = _coverage_index(Path(positive_enrichment_dir), positive_ids)
    negative_coverage_all = _coverage_index(Path(negative_enrichment_dir), set(negatives))
    positive_features = _document_features(Path(positive_enrichment_dir), positive_ids)
    negative_features_all = _document_features(Path(negative_enrichment_dir), set(negatives))
    if identity_review_dir is None:
        identity_groups = _identity_groups(candidate_rows)
        identity_ready = False
        identity_conflicts = 0
    else:
        identity_groups, identity_ready, identity_conflicts = _reviewed_identity_groups(
            Path(identity_review_dir), candidate_rows
        )
        for row in candidate_rows:
            row["identity_reviewed"] = True
    temporal_features = _temporal_feature_index(
        Path(temporal_count_dir) if temporal_count_dir is not None else None,
        positive_ids | accepted_negative_ids,
    )

    rows: list[dict[str, Any]] = []
    for candidate in sorted(candidate_rows, key=lambda item: item["candidate_id"]):
        candidate_id = candidate["candidate_id"]
        if candidate_id in positive_ids:
            coverage = positive_coverage[candidate_id]
            document_features = positive_features[candidate_id]
        else:
            coverage = negative_coverage_all[candidate_id]
            document_features = negative_features_all[candidate_id]
        rows.append(
            {
                "schema_version": FEATURE_TABLE_VERSION,
                **candidate,
                "group_id": identity_groups[candidate_id],
                "model_text": " ; ".join([candidate["canonical_name"], *candidate["aliases"]]),
                "features": {
                    "scientific_coverage_complete": coverage["scientific"],
                    "industry_coverage_complete": coverage["industry"],
                    **document_features,
                    **temporal_features[candidate_id],
                },
            }
        )

    qualification_eligible = (
        len(positives) == 100
        and accepted_counts == Counter({"mature": 50, "hype": 50})
        and identity_ready
        and all(row["identity_reviewed"] for row in rows)
        and all(
            all(
                row["features"][key]
                for key in (
                    "scientific_coverage_complete",
                    "industry_coverage_complete",
                )
            )
            for row in rows
        )
        and all(row["features"]["temporal_count_coverage_complete"] for row in rows)
    )
    features_bytes = (
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    ).encode("utf-8")
    manifest_value = {
        "schema_version": FEATURE_TABLE_VERSION,
        "cutoff_date": _CUTOFF.isoformat(),
        "release_status": "qualification" if qualification_eligible else "development_only",
        "qualification_eligible": qualification_eligible,
        "counts": {
            "positive": len(positives),
            "accepted_negative": len(accepted_negative_ids),
            "accepted_mature": accepted_counts["mature"],
            "accepted_marketing_hype": accepted_counts["hype"],
            "proposed_negative_excluded": len(negatives) - len(accepted_negative_ids),
            "feature_rows": len(rows),
            "identity_reviewed": sum(row["identity_reviewed"] for row in rows),
            "identity_conflicts": identity_conflicts,
        },
        "feature_policy": {
            "capped_document_counts_used": False,
            "uncapped_openalex_temporal_counts_used": temporal_count_dir is not None,
            "expert_annotation_fields_used": False,
            "proposed_labels_used": False,
            "exact_identity_only": identity_review_dir is None,
            "reviewed_identity_groups_used": identity_review_dir is not None,
            "temporal_window": [_WINDOW_START.isoformat(), _CUTOFF.isoformat()],
            "recent_window_from": _RECENT_START.isoformat(),
        },
        "deficits": {
            "accepted_mature": max(0, 50 - accepted_counts["mature"]),
            "accepted_marketing_hype": max(0, 50 - accepted_counts["hype"]),
            "unreviewed_identities": sum(not row["identity_reviewed"] for row in rows),
            "identity_conflicts": identity_conflicts,
        },
        "surpluses": {
            "accepted_mature": max(0, accepted_counts["mature"] - 50),
            "accepted_marketing_hype": max(0, accepted_counts["hype"] - 50),
        },
        "inputs": {
            "positive": _manifest_digest(positive_path),
            "positive_enrichment": _manifest_digest(Path(positive_enrichment_dir)),
            "negative_plan": _manifest_digest(negative_plan_path),
            "negative_enrichment": _manifest_digest(Path(negative_enrichment_dir)),
            "adjudication": _manifest_digest(adjudication_path),
            "temporal_counts": (
                _manifest_digest(Path(temporal_count_dir))
                if temporal_count_dir is not None
                else None
            ),
            "identity_review": (
                _manifest_digest(Path(identity_review_dir))
                if identity_review_dir is not None
                else None
            ),
        },
        "outputs": {FEATURES_FILENAME: _digest(features_bytes)},
    }
    manifest_bytes = (
        json.dumps(manifest_value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return features_bytes, manifest_bytes


def export_feature_table(
    *,
    positive_dir: str | Path,
    positive_enrichment_dir: str | Path,
    negative_plan_dir: str | Path,
    negative_enrichment_dir: str | Path,
    adjudication_dir: str | Path,
    temporal_count_dir: str | Path | None = None,
    identity_review_dir: str | Path | None = None,
    output_dir: str | Path,
) -> FeatureTablePaths:
    features, manifest = build_feature_table(
        positive_dir=positive_dir,
        positive_enrichment_dir=positive_enrichment_dir,
        negative_plan_dir=negative_plan_dir,
        negative_enrichment_dir=negative_enrichment_dir,
        adjudication_dir=adjudication_dir,
        temporal_count_dir=temporal_count_dir,
        identity_review_dir=identity_review_dir,
    )
    paths = publish_artifact_bundle(
        {FEATURES_FILENAME: features, MANIFEST_FILENAME: manifest}, output_dir
    )
    return FeatureTablePaths(features=paths[FEATURES_FILENAME], manifest=paths[MANIFEST_FILENAME])
