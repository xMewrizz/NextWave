from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from nextwave.evaluation.feature_table import (
    FEATURE_TABLE_VERSION,
    _document_features,
    _identity_groups,
    _temporal_feature_index,
    build_feature_table,
    export_feature_table,
)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(_json_bytes(row) for row in rows)


def _digest(payload: bytes) -> dict[str, object]:
    return {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _write_dict_bundle(directory: Path, files: dict[str, bytes]) -> None:
    directory.mkdir()
    for name, payload in files.items():
        (directory / name).write_bytes(payload)
    manifest = {"outputs": {name: _digest(payload) for name, payload in files.items()}}
    (directory / "manifest.json").write_bytes(_json_bytes(manifest))


def _candidate(candidate_id: str, name: str, aliases: list[str] | None = None) -> dict:
    return {
        "candidate_id": candidate_id,
        "canonical_name": name,
        "aliases": aliases or [],
        "domain": "Edge",
        "analysis_scope_key": "edge-v1",
        "cutoff_date": "2026-09-15",
    }


def _coverage(candidate_ids: list[str]) -> list[dict]:
    return [
        {
            "candidate_id": candidate_id,
            "source_class": source_class,
            "status": "complete",
        }
        for candidate_id in candidate_ids
        for source_class in ("scientific", "industry")
    ]


def _document(candidate_id: str, connector: str, suffix: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "document_id": f"doc-{candidate_id}-{suffix}",
        "connector": connector,
        "published_at": "2026-01-01",
        "trust_tier": "A" if connector == "openalex" else "unknown",
        "origin_id": f"origin-{candidate_id}-{suffix}",
        "organizations": [f"org-{suffix}"],
        "source_type": "scientific_publication" if connector == "openalex" else "other",
    }


class FeatureTableTests(unittest.TestCase):
    def test_exa_documents_supply_industry_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "combined"
            _write_dict_bundle(
                result,
                {
                    "documents.jsonl": _jsonl_bytes(
                        [_document("candidate-1", "exa", "industry")]
                    )
                },
            )
            features = _document_features(result, {"candidate-1"})
        self.assertTrue(features["candidate-1"]["industry_recent_present"])

    def _fixture(self, root: Path) -> dict[str, Path]:
        positive = root / "positive"
        positive.mkdir()
        positive_row = {
            "record_id": "organizer-001",
            "canonical_name": "Shared-Tech",
            "aliases": [],
            "domain": "Edge",
            "analysis_scope_key": "edge-v1",
            "cutoff_date": "2026-09-15",
            "identity_status": "pending_review",
        }
        positive_payload = _jsonl_bytes([positive_row])
        (positive / "positive_candidates.jsonl").write_bytes(positive_payload)
        (positive / "manifest.json").write_bytes(
            _json_bytes(
                {
                    "outputs": [
                        {
                            "filename": "positive_candidates.jsonl",
                            **_digest(positive_payload),
                        }
                    ]
                }
            )
        )

        candidates = [
            _candidate("team-negative-001", "Mature tech", ["Shared Tech"]),
            _candidate("team-negative-002", "Proposed hype"),
        ]
        negative_plan = root / "negative-plan"
        _write_dict_bundle(negative_plan, {"plan.json": _json_bytes({"candidates": candidates})})

        adjudication = root / "adjudication"
        decisions = [
            {
                "candidate_id": "team-negative-001",
                "decision_status": "accepted",
                "proposed_label": "mature",
            },
            {
                "candidate_id": "team-negative-002",
                "decision_status": "proposed",
                "proposed_label": "hype",
            },
        ]
        _write_dict_bundle(adjudication, {"adjudication.jsonl": _jsonl_bytes(decisions)})

        positive_result = root / "positive-result"
        _write_dict_bundle(
            positive_result,
            {
                "coverage.jsonl": _jsonl_bytes(_coverage(["organizer-001"])),
                "documents.jsonl": _jsonl_bytes(
                    [
                        _document("organizer-001", "openalex", "science"),
                        _document("organizer-001", "mediacloud", "media"),
                    ]
                ),
            },
        )
        negative_result = root / "negative-result"
        _write_dict_bundle(
            negative_result,
            {
                "coverage.jsonl": _jsonl_bytes(
                    _coverage(["team-negative-001", "team-negative-002"])
                ),
                "documents.jsonl": _jsonl_bytes(
                    [
                        _document("team-negative-001", "openalex", "science"),
                        _document("team-negative-002", "mediacloud", "media"),
                    ]
                ),
            },
        )
        return {
            "positive_dir": positive,
            "positive_enrichment_dir": positive_result,
            "negative_plan_dir": negative_plan,
            "negative_enrichment_dir": negative_result,
            "adjudication_dir": adjudication,
        }

    def test_proposed_label_is_excluded_and_manifest_is_development_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = self._fixture(Path(tmp))
            features, manifest = build_feature_table(**kwargs)
        rows = [json.loads(line) for line in features.decode().splitlines()]
        summary = json.loads(manifest)
        self.assertEqual(
            [row["candidate_id"] for row in rows],
            ["organizer-001", "team-negative-001"],
        )
        self.assertEqual(summary["schema_version"], FEATURE_TABLE_VERSION)
        self.assertEqual(summary["release_status"], "development_only")
        self.assertFalse(summary["qualification_eligible"])
        self.assertEqual(summary["counts"]["accepted_marketing_hype"], 0)
        self.assertEqual(summary["counts"]["proposed_negative_excluded"], 1)
        self.assertFalse(rows[0]["features"]["temporal_count_coverage_complete"])
        self.assertIsNone(rows[0]["features"]["scientific_count_log_growth"])

    def test_exact_alias_intersection_shares_cross_corpus_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            features, _ = build_feature_table(**self._fixture(Path(tmp)))
        rows = [json.loads(line) for line in features.decode().splitlines()]
        self.assertEqual(rows[0]["group_id"], rows[1]["group_id"])

    def test_reviewed_identity_artifact_replaces_exact_only_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kwargs = self._fixture(root)
            identity = root / "identity"
            identity_rows = [
                {
                    "candidate_id": "organizer-001",
                    "canonical_name": "Shared-Tech",
                    "corpus": "positive",
                    "group_id": "cross-corpus-reviewed0001",
                    "identity_status": "reviewed",
                },
                {
                    "candidate_id": "team-negative-001",
                    "canonical_name": "Mature tech",
                    "corpus": "negative",
                    "group_id": "cross-corpus-reviewed0001",
                    "identity_status": "reviewed",
                },
            ]
            _write_dict_bundle(
                identity,
                {
                    "identities.jsonl": _jsonl_bytes(identity_rows),
                    "pair_decisions.jsonl": b"",
                },
            )
            manifest_path = identity / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest.update(
                {
                    "schema_version": "candidate-identity-review-v1",
                    "ready_for_model": True,
                    "totals": {"conflicts": 0},
                }
            )
            manifest_path.write_bytes(_json_bytes(manifest))
            features, manifest_bytes = build_feature_table(
                **kwargs,
                identity_review_dir=identity,
            )
        rows = [json.loads(line) for line in features.decode().splitlines()]
        summary = json.loads(manifest_bytes)
        self.assertTrue(all(row["identity_reviewed"] for row in rows))
        self.assertEqual({row["group_id"] for row in rows}, {"cross-corpus-reviewed0001"})
        self.assertTrue(summary["feature_policy"]["reviewed_identity_groups_used"])
        self.assertEqual(summary["deficits"]["unreviewed_identities"], 0)

    def test_identity_does_not_merge_different_roots(self) -> None:
        rows = [
            _candidate("candidate-1", "RAG"),
            _candidate("candidate-2", "Retrieval-Augmented Generation"),
        ]
        groups = _identity_groups(rows)
        self.assertNotEqual(groups["candidate-1"], groups["candidate-2"])

    def test_future_document_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = self._fixture(Path(tmp))
            result = kwargs["positive_enrichment_dir"]
            documents = _jsonl_bytes(
                [
                    {
                        **_document("organizer-001", "openalex", "science"),
                        "published_at": "2026-09-16",
                    }
                ]
            )
            (result / "documents.jsonl").write_bytes(documents)
            manifest = json.loads((result / "manifest.json").read_text())
            manifest["outputs"]["documents.jsonl"] = _digest(documents)
            (result / "manifest.json").write_bytes(_json_bytes(manifest))
            with self.assertRaisesRegex(ValueError, "outside the frozen feature window"):
                build_feature_table(**kwargs)

    def test_manifest_checksum_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = self._fixture(Path(tmp))
            path = kwargs["adjudication_dir"] / "adjudication.jsonl"
            path.write_text(path.read_text() + "{}\n")
            with self.assertRaisesRegex(ValueError, "diverges"):
                build_feature_table(**kwargs)

    def test_export_is_atomic_and_refuses_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kwargs = self._fixture(root)
            output = root / "output"
            paths = export_feature_table(**kwargs, output_dir=output)
            self.assertTrue(paths.features.is_file())
            with self.assertRaisesRegex(ValueError, "already exists"):
                export_feature_table(**kwargs, output_dir=output)

    def test_temporal_counts_are_validated_and_transformed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "temporal"
            row = {
                "candidate_id": "organizer-001",
                "coverage": "complete",
                "counts": {
                    "candidate_previous": 2,
                    "candidate_recent": 5,
                    "scope_previous": 100,
                    "scope_recent": 200,
                },
                "candidate_log_growth": math.log1p(5) - math.log1p(2),
                "scope_share_previous": 0.02,
                "scope_share_recent": 0.025,
                "scope_share_delta": 0.005,
            }
            _write_dict_bundle(
                directory,
                {"candidate_temporal_features.jsonl": _jsonl_bytes([row])},
            )
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest.update(
                {
                    "schema_version": "openalex-temporal-count-result-v4",
                    "status": "complete",
                }
            )
            manifest_path.write_bytes(_json_bytes(manifest))
            result = _temporal_feature_index(directory, {"organizer-001"})
        self.assertTrue(result["organizer-001"]["temporal_count_coverage_complete"])
        self.assertAlmostEqual(
            result["organizer-001"]["scientific_previous_count_log1p"],
            1.0986122886681098,
        )

    def test_incomplete_temporal_counts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "temporal"
            _write_dict_bundle(
                directory,
                {"candidate_temporal_features.jsonl": _jsonl_bytes([])},
            )
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest.update(
                {
                    "schema_version": "openalex-temporal-count-result-v4",
                    "status": "partial",
                }
            )
            manifest_path.write_bytes(_json_bytes(manifest))
            with self.assertRaisesRegex(ValueError, "must be complete v4"):
                _temporal_feature_index(directory, {"organizer-001"})

    def test_temporal_candidate_count_cannot_exceed_scope_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "temporal"
            row = {
                "candidate_id": "organizer-001",
                "coverage": "complete",
                "counts": {
                    "candidate_previous": 11,
                    "candidate_recent": 1,
                    "scope_previous": 10,
                    "scope_recent": 10,
                },
                "candidate_log_growth": math.log1p(1) - math.log1p(11),
                "scope_share_previous": 1.1,
                "scope_share_recent": 0.1,
                "scope_share_delta": -1.0,
            }
            _write_dict_bundle(
                directory,
                {"candidate_temporal_features.jsonl": _jsonl_bytes([row])},
            )
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest.update(
                {
                    "schema_version": "openalex-temporal-count-result-v4",
                    "status": "complete",
                }
            )
            manifest_path.write_bytes(_json_bytes(manifest))
            with self.assertRaisesRegex(ValueError, "exceeds scope count"):
                _temporal_feature_index(directory, {"organizer-001"})


if __name__ == "__main__":
    unittest.main()
