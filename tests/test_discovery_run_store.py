"""Tests for discovery run persistence (no live network)."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from typing import Any

from nextwave.discovery import (
    ScopeGranularity,
    build_analysis_scope,
    build_discovery_plan,
)
from nextwave.discovery.run_store import (
    LABELING_CUTOFF_DATE_ISO,
    assert_labeling_cutoff,
    build_run_id,
    is_labeling_eligible,
    iter_nested_documents,
    load_discovery_run,
    save_discovery_run,
)


def _scope():
    return build_analysis_scope(
        raw_query="Технологии в ИИ",
        normalized_query="artificial intelligence",
        search_texts=("Технологии в ИИ",),
        languages=("en", "ru"),
        granularity=ScopeGranularity.DIRECTION,
        subfield_ids=("1702",),
    )


def _plan(analysis_id: str = "analysis-ai-001", cutoff: date = date(2026, 9, 15)):
    return build_discovery_plan(
        analysis_id=analysis_id,
        scope=_scope(),
        published_from=date(2025, 9, 15),
        cutoff_date=cutoff,
    )


class _StubResult:
    """Minimal stand-in with the same save/load interface as DiscoveryPipelineResult."""

    def __init__(self, plan_id: str, decisions: list[dict[str, Any]]) -> None:
        self.plan_id = plan_id
        self.pipeline_version = "discovery-pipeline-v6"
        self._decisions = decisions

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "pipeline_version": self.pipeline_version,
            "documents": ["d1", "d2"],
            "candidate_proposals": {
                "proposals": [{"proposal_id": "p1"}],
                "exclusions": [],
            },
            "candidate_gate": {"gate_id": "g1", "decisions": self._decisions},
            "alias_resolution": {"review_suggestions": []},
            "evidence_extraction": {"proposals": [], "issues": []},
            "text_extraction": None,
            "scientific": {
                "snapshot_path": "snap/oa",
                "documents": [
                    {"document_id": "d1", "title": "Doc one"},
                ],
            },
            "media": {"snapshot_path": "snap/mc", "documents": []},
            "verification": {"results": []},
        }


def _accepting_result(plan_id: str) -> _StubResult:
    return _StubResult(plan_id, [{"proposal_id": "p1", "decision": "accept"}])


class DiscoveryRunStoreTests(unittest.TestCase):
    def test_round_trip_reproduces_counts(self) -> None:
        plan = _plan()
        with tempfile.TemporaryDirectory() as directory:
            run_dir = save_discovery_run(
                plan,
                _accepting_result(plan.plan_id),  # type: ignore[arg-type]
                analysis_scope_key="ai-nlp-v1",
                domain="Инфраструктура ИИ",
                output_root=Path(directory),
            )
            loaded = load_discovery_run(run_dir)
            self.assertEqual(loaded.manifest["counts"]["documents"], 2)
            self.assertEqual(loaded.manifest["counts"]["proposals"], 1)
            self.assertEqual(loaded.manifest["counts"]["accepted"], 1)
            self.assertTrue(loaded.manifest["labeling_eligible"])
            self.assertEqual(loaded.plan["plan_id"], plan.plan_id)
            nested = iter_nested_documents(loaded.result)
            self.assertEqual([item["document_id"] for item in nested], ["d1"])

    def test_run_id_is_deterministic(self) -> None:
        plan = _plan()
        plan_dict = plan.to_dict()
        self.assertEqual(
            build_run_id("analysis-ai-001", plan_dict),
            build_run_id("analysis-ai-001", plan_dict),
        )

    def test_existing_directory_is_an_error(self) -> None:
        plan = _plan()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_discovery_run(
                plan,
                _accepting_result(plan.plan_id),  # type: ignore[arg-type]
                analysis_scope_key="ai-nlp-v1",
                domain="Инфраструктура ИИ",
                output_root=root,
            )
            with self.assertRaises(FileExistsError):
                save_discovery_run(
                    plan,
                    _accepting_result(plan.plan_id),  # type: ignore[arg-type]
                    analysis_scope_key="ai-nlp-v1",
                    domain="Инфраструктура ИИ",
                    output_root=root,
                )

    def test_checksum_mismatch_is_rejected(self) -> None:
        plan = _plan()
        with tempfile.TemporaryDirectory() as directory:
            run_dir = save_discovery_run(
                plan,
                _accepting_result(plan.plan_id),  # type: ignore[arg-type]
                analysis_scope_key="ai-nlp-v1",
                domain="Инфраструктура ИИ",
                output_root=Path(directory),
            )
            payload_path = run_dir / "pipeline_result.json"
            payload_path.write_text(
                json.dumps({"plan_id": plan.plan_id}, ensure_ascii=False),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_discovery_run(run_dir)

    def test_non_labeling_cutoff_is_demo_only(self) -> None:
        plan = _plan(cutoff=date(2026, 9, 20))
        with tempfile.TemporaryDirectory() as directory:
            run_dir = save_discovery_run(
                plan,
                _accepting_result(plan.plan_id),  # type: ignore[arg-type]
                analysis_scope_key="ai-nlp-v1",
                domain="Инфраструктура ИИ",
                output_root=Path(directory),
            )
            loaded = load_discovery_run(run_dir)
            self.assertFalse(loaded.manifest["labeling_eligible"])
            with self.assertRaises(ValueError):
                assert_labeling_cutoff(loaded.manifest["cutoff_date"])
            self.assertTrue(is_labeling_eligible(LABELING_CUTOFF_DATE_ISO))


if __name__ == "__main__":
    unittest.main()
