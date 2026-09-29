from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nextwave.application import (
    ANALYSIS_APPLICATION_VERSION,
    AnalysisApplication,
)


def _bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _digest(raw: bytes) -> dict[str, object]:
    return {"size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _environment() -> dict[str, str]:
    return {
        "NEXTWAVE_EXA_API_KEY": "exa-test",
        "NEXTWAVE_LLM_PROVIDER": "yandex",
        "NEXTWAVE_LLM_MODEL": "YandexGPT Lite 5",
        "NEXTWAVE_GATE_LLM_PROVIDER": "yandex",
        "NEXTWAVE_GATE_LLM_MODEL": "YandexGPT Pro 5",
        "NEXTWAVE_EVIDENCE_LLM_PROVIDER": "yandex",
        "NEXTWAVE_EVIDENCE_LLM_MODEL": "YandexGPT Pro 5.1",
        "NEXTWAVE_LLM_API_KEY": "llm-test",
        "NEXTWAVE_YANDEX_FOLDER_ID": "folder-test",
    }


def _bundle(path: Path, *, candidate_count: int | None = None, result=False) -> None:
    path.mkdir(parents=True)
    outputs: dict[str, dict[str, object]] = {}
    if result:
        raw = _bytes({"schema_version": "analysis-response-v1", "candidates": []})
        (path / "result.json").write_bytes(raw)
        outputs["result.json"] = _digest(raw)
    manifest = {
        "schema_version": "test-artifact-v1",
        "outputs": outputs,
    }
    if candidate_count is not None:
        manifest["candidate_count"] = candidate_count
    (path / "manifest.json").write_bytes(_bytes(manifest))


def _discovery_bundle(path: Path, *, query: str, analysis_id: str) -> None:
    path.mkdir(parents=True)
    plan = {
        "schema_version": "discovery-plan-v1",
        "analysis_id": analysis_id,
        "query": {"raw_query": query, "cutoff_date": "2026-09-15"},
    }
    plan_raw = _bytes(plan)
    result_raw = _bytes({"schema_version": "discovery-result-v1"})
    (path / "plan.json").write_bytes(plan_raw)
    (path / "pipeline_result.json").write_bytes(result_raw)
    (path / "manifest.json").write_bytes(
        _bytes(
            {
                "schema_version": "discovery-run-manifest-v2",
                "analysis_id": analysis_id,
                "raw_query": query,
                "cutoff_date": "2026-09-15",
                "outputs": [
                    {"filename": "plan.json", **_digest(plan_raw)},
                    {"filename": "pipeline_result.json", **_digest(result_raw)},
                ],
            }
        )
    )


class _Pipeline:
    def execute(self, plan, progress):
        progress("[discovery] sources: 10 documents (0.1s)")
        progress("[discovery] gate: 3 accepted (0.1s)")
        return object()


class AnalysisApplicationTests(unittest.TestCase):
    def _patches(self, calls: list[tuple[str, dict]]) -> list[patch]:
        def resolver(environment):
            scope = SimpleNamespace(scope_id="scope-ai", raw_query="AI infrastructure")
            return SimpleNamespace(resolve=lambda query: SimpleNamespace(scope=scope))

        def save_run(plan, result, **kwargs):
            run = Path(kwargs["output_root"]) / "analysis-job-run"
            _bundle(run)
            calls.append(("discovery", kwargs))
            return run

        def operation(name: str, *, inference=False, result=False):
            def execute(**kwargs):
                calls.append((name, kwargs))
                output = Path(kwargs["output_dir"])
                _bundle(
                    output,
                    candidate_count=3 if inference else None,
                    result=result,
                )
                if name == "temporal_counts":
                    (output / "manifest.json").write_bytes(
                        _bytes(
                            {
                                "schema_version": "openalex-temporal-count-result-v4",
                                "status": "complete",
                                "outputs": {},
                            }
                        )
                    )
                return object()

            return execute

        targets = {
            "build_query_resolver_from_environment": resolver,
            "build_discovery_plan": lambda **kwargs: object(),
            "build_discovery_pipeline_from_environment": lambda *args, **kwargs: _Pipeline(),
            "save_discovery_run": save_run,
            "export_analysis_enrichment_plan": operation("enrichment_plan"),
            "run_enrichment": operation("scientific_enrichment"),
            "export_exa_enrichment_plan": operation("exa_plan"),
            "run_exa_enrichment": operation("exa_enrichment"),
            "export_combined_enrichment": operation("combined_enrichment"),
            "export_analysis_temporal_count_plan": operation("temporal_plan"),
            "run_temporal_counts": operation("temporal_counts"),
            "export_analysis_feature_table": operation("features"),
            "export_analysis_inference": operation("inference", inference=True),
            "export_analysis_shortlist": operation("shortlist"),
            "export_analysis_evidence_input": operation("evidence_input"),
            "export_evidence_llm_plan": operation("evidence_plan"),
            "run_evidence_llm": operation("evidence_run"),
            "export_analysis_result": operation("result", result=True),
        }
        return [patch(f"nextwave.application.{name}", value) for name, value in targets.items()]

    def test_full_run_checkpoints_every_step_and_resumes_without_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model").mkdir()
            (root / "model" / "model.json").write_text("{}\n", encoding="utf-8")
            calls: list[tuple[str, dict]] = []
            progress: list[tuple[str, float, str]] = []
            patches = self._patches(calls)
            for item in patches:
                item.start()
            try:
                runner = AnalysisApplication(
                    workspace=root / "job",
                    model_dir=root / "model",
                    environment=_environment(),
                    cutoff_date=date(2026, 9, 15),
                    progress=lambda stage, value, message: progress.append(
                        (stage, value, message)
                    ),
                )
                paths = runner.run(query="  AI   infrastructure  ", analysis_id="job-1")
            finally:
                for item in reversed(patches):
                    item.stop()

            self.assertTrue(paths.result.is_file())
            manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], ANALYSIS_APPLICATION_VERSION)
            self.assertEqual(
                manifest["completed_steps"],
                [
                    "discovery",
                    "enrichment_plan",
                    "scientific_enrichment",
                    "exa_plan",
                    "exa_enrichment",
                    "combined_enrichment",
                    "temporal_plan",
                    "temporal_counts",
                    "features",
                    "inference",
                    "shortlist",
                    "evidence_input",
                    "evidence_plan",
                    "evidence_run",
                    "result",
                ],
            )
            shortlist = next(kwargs for name, kwargs in calls if name == "shortlist")
            self.assertEqual(shortlist["limit"], 3)
            self.assertEqual(
                next(kwargs for name, kwargs in calls if name == "scientific_enrichment")[
                    "connectors"
                ],
                ("openalex",),
            )
            self.assertEqual(
                next(kwargs for name, kwargs in calls if name == "scientific_enrichment")[
                    "concurrency"
                ],
                8,
            )
            self.assertEqual(
                next(kwargs for name, kwargs in calls if name == "temporal_counts")[
                    "concurrency"
                ],
                3,
            )
            self.assertIn("candidate_gate", {stage for stage, _, _ in progress})
            self.assertEqual(progress[-1][0:2], ("result", 1.0))

            resumed = AnalysisApplication(
                workspace=root / "job",
                model_dir=root / "model",
                environment=_environment(),
            ).run(query="AI infrastructure", analysis_id="job-1")
            self.assertEqual(resumed.result.read_bytes(), paths.result.read_bytes())

    def test_checkpoint_for_another_query_is_rejected_before_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model").mkdir()
            model = b"{}\n"
            (root / "model" / "model.json").write_bytes(model)
            app = AnalysisApplication(
                workspace=root,
                model_dir=root / "model",
                environment=_environment(),
            )
            state = {
                "schema_version": ANALYSIS_APPLICATION_VERSION,
                "analysis_id": "job-1",
                "query": "first",
                "cutoff_date": "2026-09-15",
                "model": _digest(model),
                "completed_steps": [],
                "artifacts": {},
            }
            root.mkdir(exist_ok=True)
            (root / "analysis_application.json").write_bytes(_bytes(state))
            with self.assertRaisesRegex(ValueError, "another job"):
                app.run(query="second", analysis_id="job-1")

    def test_saved_discovery_is_adopted_without_repeating_paid_search(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model").mkdir()
            (root / "model" / "model.json").write_text("{}\n", encoding="utf-8")
            saved = root / "job" / "artifacts" / "discovery" / "job-1-saved"
            _discovery_bundle(saved, query="AI infrastructure", analysis_id="job-1")
            calls: list[tuple[str, dict]] = []
            patches = self._patches(calls)
            for item in patches:
                item.start()
            try:
                result = AnalysisApplication(
                    workspace=root / "job",
                    model_dir=root / "model",
                    environment=_environment(),
                ).run(query="AI infrastructure", analysis_id="job-1")
            finally:
                for item in reversed(patches):
                    item.stop()

            self.assertTrue(result.result.is_file())
            self.assertNotIn("discovery", [name for name, _ in calls])
            state = json.loads(result.manifest.read_text(encoding="utf-8"))
            self.assertEqual(
                state["artifacts"]["discovery"],
                "artifacts/discovery/job-1-saved",
            )

    def test_partial_temporal_counts_are_retried_from_completed_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model").mkdir()
            (root / "model" / "model.json").write_text("{}\n", encoding="utf-8")
            calls: list[tuple[str, dict]] = []
            patches = self._patches(calls)
            for item in patches:
                item.start()

            def partial_temporal_counts(**kwargs):
                output = Path(kwargs["output_dir"])
                _bundle(output)
                (output / "manifest.json").write_bytes(
                    _bytes(
                        {
                            "schema_version": "openalex-temporal-count-result-v4",
                            "status": "partial",
                            "outputs": {},
                        }
                    )
                )

            try:
                runner = AnalysisApplication(
                    workspace=root / "job",
                    model_dir=root / "model",
                    environment=_environment(),
                )
                with patch(
                    "nextwave.application.run_temporal_counts",
                    partial_temporal_counts,
                ):
                    with self.assertRaisesRegex(ValueError, "must be complete"):
                        runner.run(query="AI infrastructure", analysis_id="job-1")
                state = json.loads(runner.manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(state["completed_steps"][-1], "temporal_plan")

                result = runner.run(query="AI infrastructure", analysis_id="job-1")
            finally:
                for item in reversed(patches):
                    item.stop()

            self.assertTrue(result.result.is_file())
            final_state = json.loads(runner.manifest_path.read_text(encoding="utf-8"))
            self.assertIn("temporal_counts", final_state["completed_steps"])

    def test_corrupt_completed_artifact_is_rejected_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model").mkdir()
            model = b"{}\n"
            (root / "model" / "model.json").write_bytes(model)
            discovery = root / "artifacts" / "discovery" / "run"
            _bundle(discovery)
            state = {
                "schema_version": ANALYSIS_APPLICATION_VERSION,
                "analysis_id": "job-1",
                "query": "query",
                "cutoff_date": "2026-09-15",
                "model": _digest(model),
                "completed_steps": ["discovery"],
                "artifacts": {"discovery": "artifacts/discovery/run"},
            }
            (root / "analysis_application.json").write_bytes(_bytes(state))
            (discovery / "manifest.json").write_text("{}\n", encoding="utf-8")
            app = AnalysisApplication(
                workspace=root,
                model_dir=root / "model",
                environment=_environment(),
            )
            with self.assertRaisesRegex(ValueError, "misses schema_version"):
                app.run(query="query", analysis_id="job-1")

    def test_changed_model_is_rejected_before_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model").mkdir()
            original = b'{"version":1}\n'
            (root / "model" / "model.json").write_bytes(original)
            state = {
                "schema_version": ANALYSIS_APPLICATION_VERSION,
                "analysis_id": "job-1",
                "query": "query",
                "cutoff_date": "2026-09-15",
                "model": _digest(original),
                "completed_steps": [],
                "artifacts": {},
            }
            (root / "analysis_application.json").write_bytes(_bytes(state))
            (root / "model" / "model.json").write_text(
                '{"version":2}\n', encoding="utf-8"
            )
            app = AnalysisApplication(
                workspace=root,
                model_dir=root / "model",
                environment=_environment(),
            )
            with self.assertRaisesRegex(ValueError, "another model"):
                app.run(query="query", analysis_id="job-1")

    def test_missing_exa_key_is_rejected_before_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "NEXTWAVE_EXA_API_KEY"):
                AnalysisApplication(
                    workspace=root / "job",
                    model_dir=root / "model",
                    environment={},
                )
            self.assertFalse((root / "job").exists())

    def test_nonqualification_cutoff_is_rejected_before_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "cutoff must be 2026-09-15"):
                AnalysisApplication(
                    workspace=root / "job",
                    model_dir=root / "model",
                    environment=_environment(),
                    cutoff_date=date(2026, 9, 16),
                )
            self.assertFalse((root / "job").exists())


if __name__ == "__main__":
    unittest.main()
