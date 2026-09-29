from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from nextwave.evaluation.temporal_count_plan import (
    TEMPORAL_COUNT_PLAN_VERSION,
    build_analysis_temporal_count_plan,
    build_temporal_count_plan,
    export_analysis_temporal_count_plan,
    export_temporal_count_plan,
)


def _digest(payload: bytes) -> dict[str, object]:
    return {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _candidate(
    candidate_id: str,
    scope: str,
    term: str,
    *,
    cutoff_date: str = "2026-09-15",
    source_query: str | None = None,
) -> dict:
    row = {
        "candidate_id": candidate_id,
        "canonical_name": term,
        "aliases": [],
        "domain": "Edge",
        "analysis_scope_key": scope,
        "cutoff_date": cutoff_date,
        "search_terms": [term, f"{term} alias"],
    }
    if source_query is not None:
        row["origin"] = {"source_query": source_query}
    return row


def _write_plan(
    directory: Path,
    bundle_id: str,
    candidates: list[dict],
    *,
    plan_role: str | None = None,
    cutoff_date: str = "2026-09-15",
    analysis_scope: dict | None = None,
) -> None:
    directory.mkdir()
    value = {
        "schema_version": "labeling-enrichment-plan-v2",
        "cutoff_date": cutoff_date,
        "candidates": candidates,
    }
    if analysis_scope is not None:
        value["analysis_scope"] = analysis_scope
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()
    (directory / "plan.json").write_bytes(payload)
    manifest = {
        "schema_version": "labeling-enrichment-plan-v2",
        "bundle_id": bundle_id,
        "candidate_count": len(candidates),
        "outputs": {"plan.json": _digest(payload)},
    }
    if plan_role is not None:
        manifest["plan_role"] = plan_role
    (directory / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )


class TemporalCountPlanTests(unittest.TestCase):
    def _fixture(self, root: Path, *, reverse: bool = False) -> tuple[Path, Path]:
        positive = [
            _candidate("organizer-001", "edge-v1", "edge compression"),
            _candidate("organizer-002", "ai-security-v1", "agent identity"),
        ]
        negative = [
            _candidate("team-negative-001", "edge-v1", "in-memory computing"),
            _candidate("team-negative-002", "ai-security-v1", "security mesh"),
        ]
        if reverse:
            positive.reverse()
            negative.reverse()
        positive_dir = root / "positive"
        negative_dir = root / "negative"
        _write_plan(positive_dir, "bundle-positive", positive)
        _write_plan(negative_dir, "bundle-negative", negative)
        return positive_dir, negative_dir

    def test_builds_two_candidate_windows_and_six_scope_denominators(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            positive, negative = self._fixture(Path(tmp))
            plan_bytes, manifest_bytes = build_temporal_count_plan(
                positive_plan_dir=positive, negative_plan_dir=negative
            )
        plan = json.loads(plan_bytes)
        manifest = json.loads(manifest_bytes)
        self.assertEqual(plan["schema_version"], TEMPORAL_COUNT_PLAN_VERSION)
        self.assertEqual(
            manifest["counts"],
            {
                "candidate_tasks": 8,
                "candidates": 4,
                "scope_tasks": 12,
                "scopes": 6,
                "tasks": 20,
            },
        )
        self.assertEqual(len({task["count_id"] for task in plan["tasks"]}), 20)

    def test_windows_do_not_overlap_and_end_at_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            positive, negative = self._fixture(Path(tmp))
            plan = json.loads(
                build_temporal_count_plan(positive_plan_dir=positive, negative_plan_dir=negative)[0]
            )
        self.assertEqual(plan["windows"]["previous"]["until"], "2025-09-14")
        self.assertEqual(plan["windows"]["recent"]["from"], "2025-09-15")
        self.assertEqual(plan["windows"]["recent"]["until"], "2026-09-15")

    def test_candidate_uses_first_reviewed_term_and_uncapped_count_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            positive, negative = self._fixture(Path(tmp))
            plan = json.loads(
                build_temporal_count_plan(positive_plan_dir=positive, negative_plan_dir=negative)[0]
            )
        tasks = [
            task
            for task in plan["tasks"]
            if task["entity_type"] == "candidate" and task["entity_id"] == "organizer-001"
        ]
        self.assertEqual({task["search_text"] for task in tasks}, {"edge compression"})
        self.assertEqual({task["search_mode"] for task in tasks}, {"candidate_proximity_5"})
        self.assertEqual(
            {task["scope_search_text"] for task in tasks},
            {'"edge computing" OR "edge AI"'},
        )
        self.assertEqual({task["per_page"] for task in tasks}, {1})
        self.assertEqual({task["count_field"] for task in tasks}, {"meta.count"})

    def test_candidate_quotes_are_normalized_for_openalex_expression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            analysis = root / "analysis"
            _write_plan(
                analysis,
                "analysis-bundle",
                [
                    _candidate(
                        "alias-group-001",
                        "fintech-v1",
                        '"Buy Now, Pay Later" (BNPL) services',
                    )
                ],
                plan_role="analysis_candidates",
            )
            plan = json.loads(
                build_analysis_temporal_count_plan(analysis_plan_dir=analysis)[0]
            )

        tasks = [row for row in plan["tasks"] if row["entity_type"] == "candidate"]
        self.assertEqual(
            {row["search_text"] for row in tasks},
            {"Buy Now, Pay Later (BNPL) services"},
        )

    def test_candidate_order_does_not_change_plan_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            positive_a, negative_a = self._fixture(first)
            positive_b, negative_b = self._fixture(second, reverse=True)
            first_plan = build_temporal_count_plan(
                positive_plan_dir=positive_a, negative_plan_dir=negative_a
            )[0]
            second_plan = build_temporal_count_plan(
                positive_plan_dir=positive_b, negative_plan_dir=negative_b
            )[0]
        self.assertEqual(first_plan, second_plan)

    def test_checksum_and_schema_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            positive, negative = self._fixture(Path(tmp))
            with (positive / "plan.json").open("ab") as stream:
                stream.write(b" ")
            with self.assertRaisesRegex(ValueError, "diverges"):
                build_temporal_count_plan(positive_plan_dir=positive, negative_plan_dir=negative)

    def test_unknown_scope_and_duplicate_candidate_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            positive, negative = self._fixture(root)
            duplicate = [_candidate("organizer-001", "unknown-v1", "duplicate")]
            _write_plan(root / "replacement", "bundle-other", duplicate)
            with self.assertRaisesRegex(ValueError, "occurs in both plans"):
                build_temporal_count_plan(
                    positive_plan_dir=positive, negative_plan_dir=root / "replacement"
                )
            unknown = [_candidate("team-new-001", "unknown-v1", "unknown")]
            other = root / "unknown"
            _write_plan(other, "bundle-unknown", unknown)
            with self.assertRaisesRegex(ValueError, "unknown analysis scope"):
                build_temporal_count_plan(positive_plan_dir=positive, negative_plan_dir=other)

    def test_export_refuses_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            positive, negative = self._fixture(root)
            output = root / "output"
            paths = export_temporal_count_plan(
                positive_plan_dir=positive,
                negative_plan_dir=negative,
                output_dir=output,
            )
            self.assertTrue(paths.plan.is_file())
            with self.assertRaisesRegex(ValueError, "already exists"):
                export_temporal_count_plan(
                    positive_plan_dir=positive,
                    negative_plan_dir=negative,
                    output_dir=output,
                )

    def test_analysis_plan_uses_only_its_query_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            analysis = root / "analysis"
            _write_plan(
                analysis,
                "analysis-bundle",
                [
                    _candidate("alias-group-001", "ai-infrastructure-v1", "paged attention"),
                    _candidate("alias-group-002", "ai-infrastructure-v1", "AI accelerator"),
                ],
                plan_role="analysis_candidates",
            )
            plan_bytes, manifest_bytes = build_analysis_temporal_count_plan(
                analysis_plan_dir=analysis
            )
        plan = json.loads(plan_bytes)
        manifest = json.loads(manifest_bytes)
        self.assertEqual(set(plan["scope_queries"]), {"ai-infrastructure-v1"})
        self.assertEqual({row["role"] for row in plan["candidates"]}, {"analysis"})
        self.assertEqual(
            manifest["counts"],
            {
                "candidate_tasks": 4,
                "candidates": 2,
                "scope_tasks": 2,
                "scopes": 1,
                "tasks": 6,
            },
        )

    def test_historical_analysis_plan_uses_cutoff_relative_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            analysis = root / "analysis"
            _write_plan(
                analysis,
                "historical-bundle",
                [
                    _candidate(
                        "alias-group-001",
                        "ai-infrastructure-v1",
                        "paged attention",
                        cutoff_date="2025-09-15",
                    )
                ],
                plan_role="analysis_candidates",
                cutoff_date="2025-09-15",
            )
            plan_bytes, manifest_bytes = build_analysis_temporal_count_plan(
                analysis_plan_dir=analysis
            )

        plan = json.loads(plan_bytes)
        manifest = json.loads(manifest_bytes)
        self.assertEqual(plan["cutoff_date"], "2025-09-15")
        self.assertEqual(manifest["cutoff_date"], "2025-09-15")
        self.assertEqual(
            plan["windows"],
            {
                "previous": {
                    "from": "2023-09-16",
                    "until": "2024-09-14",
                    "inclusive": True,
                },
                "recent": {
                    "from": "2024-09-15",
                    "until": "2025-09-15",
                    "inclusive": True,
                },
            },
        )

    def test_analysis_plan_accepts_its_dynamic_user_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            analysis = root / "analysis"
            scope = "scope-7377ccb2b45c924c"
            query = "Infrastructure technologies for AI training and inference"
            _write_plan(
                analysis,
                "analysis-bundle",
                [
                    _candidate(
                        "alias-group-001",
                        scope,
                        "paged attention",
                        source_query=query,
                    ),
                    _candidate(
                        "alias-group-002",
                        scope,
                        "AI accelerator",
                        source_query=query,
                    ),
                ],
                plan_role="analysis_candidates",
                analysis_scope={
                    "scope_id": scope,
                    "normalized_query": query,
                    "search_texts": [query, "AI training infrastructure"],
                },
            )
            plan_bytes, _ = build_analysis_temporal_count_plan(
                analysis_plan_dir=analysis
            )

        plan = json.loads(plan_bytes)
        expected = f'"{query}" OR "AI training infrastructure"'
        self.assertEqual(plan["scope_queries"], {scope: expected})
        self.assertEqual(
            {task["scope_search_text"] for task in plan["tasks"]}, {expected}
        )

    def test_analysis_plan_rejects_inconsistent_dynamic_scope_queries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            analysis = root / "analysis"
            scope = "scope-dynamic"
            _write_plan(
                analysis,
                "analysis-bundle",
                [_candidate("one", scope, "one", source_query="first query")],
                plan_role="analysis_candidates",
                analysis_scope={
                    "scope_id": "another-scope",
                    "normalized_query": "first query",
                    "search_texts": ["first query"],
                },
            )
            with self.assertRaisesRegex(ValueError, "differs from analysis_scope"):
                build_analysis_temporal_count_plan(analysis_plan_dir=analysis)

    def test_analysis_plan_rejects_training_or_multiple_scopes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training = root / "training"
            _write_plan(training, "training", [_candidate("one", "edge-v1", "edge")])
            with self.assertRaisesRegex(ValueError, "analysis_candidates"):
                build_analysis_temporal_count_plan(analysis_plan_dir=training)

            mixed = root / "mixed"
            _write_plan(
                mixed,
                "mixed",
                [
                    _candidate("one", "edge-v1", "edge"),
                    _candidate("two", "robotics-v1", "robot"),
                ],
                plan_role="analysis_candidates",
            )
            with self.assertRaisesRegex(ValueError, "exactly one scope"):
                build_analysis_temporal_count_plan(analysis_plan_dir=mixed)

    def test_analysis_export_is_atomic_and_refuses_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            analysis = root / "analysis"
            _write_plan(
                analysis,
                "analysis",
                [_candidate("one", "edge-v1", "edge")],
                plan_role="analysis_candidates",
            )
            output = root / "output"
            paths = export_analysis_temporal_count_plan(
                analysis_plan_dir=analysis, output_dir=output
            )
            self.assertTrue(paths.plan.is_file())
            with self.assertRaisesRegex(ValueError, "already exists"):
                export_analysis_temporal_count_plan(
                    analysis_plan_dir=analysis, output_dir=output
                )


if __name__ == "__main__":
    unittest.main()
