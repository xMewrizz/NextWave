from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from nextwave.__main__ import main
from nextwave.datasets import OrganizerDatasetPaths, OrganizerWorkbookError


class CommandLineTests(unittest.TestCase):
    @patch("nextwave.__main__.build_organizer_dataset")
    def test_dataset_build_uses_default_versioned_output(self, build) -> None:
        output = Path("data/processed/organizer-positive-2026-09-15-v1")
        build.return_value = OrganizerDatasetPaths(
            candidates=output / "positive_candidates.jsonl",
            annotations=output / "positive_annotations.jsonl",
            manifest=output / "manifest.json",
        )
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = main(
                ["dataset-build", "--input", "data/raw/organizer_signals.xlsx"]
            )

        self.assertEqual(exit_code, 0)
        build.assert_called_once_with(Path("data/raw/organizer_signals.xlsx"), output)
        self.assertIn("успешно собран", stdout.getvalue())
        self.assertIn("manifest.json", stdout.getvalue())

    @patch("nextwave.__main__.build_organizer_dataset")
    def test_dataset_build_reports_validation_error_without_traceback(self, build) -> None:
        build.side_effect = OrganizerWorkbookError("unexpected header in C2")
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            exit_code = main(
                [
                    "dataset-build",
                    "--input",
                    "broken.xlsx",
                    "--output",
                    "data/processed/check",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("unexpected header in C2", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    @patch("nextwave.__main__._runtime_environment")
    @patch("nextwave.__main__.build_query_resolver_from_environment")
    def test_query_resolve_prints_resolution_as_json(
        self, build_resolver, runtime_environment
    ) -> None:
        runtime_environment.return_value = {
            "NEXTWAVE_LLM_PROVIDER": "yandex",
            "NEXTWAVE_LLM_MODEL": "YandexGPT Lite 5",
            "NEXTWAVE_LLM_API_KEY": "temporary-secret",
        }
        build_resolver.return_value.resolve.return_value.to_dict.return_value = {
            "scope": {"normalized_query": "artificial intelligence"},
            "taxonomy_status": "matched",
        }
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = main(["query-resolve", "--query", "Технологии в ИИ"])

        self.assertEqual(exit_code, 0)
        runtime_environment.assert_called_once_with(Path("config/hackathon.env"))
        build_resolver.assert_called_once_with(runtime_environment.return_value)
        build_resolver.return_value.resolve.assert_called_once_with("Технологии в ИИ")
        self.assertIn('"taxonomy_status": "matched"', stdout.getvalue())

    @patch("nextwave.__main__.build_query_resolver_from_environment")
    def test_query_resolve_reports_configuration_error_without_traceback(
        self, build_resolver
    ) -> None:
        build_resolver.side_effect = ValueError("NEXTWAVE_LLM_API_KEY must not be blank")
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            exit_code = main(
                [
                    "query-resolve",
                    "--query",
                    "ИИ",
                    "--env-file",
                    "config/custom.env",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("NEXTWAVE_LLM_API_KEY", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_runtime_environment_reads_file_and_process_values_win(self) -> None:
        from nextwave.__main__ import _runtime_environment

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.env"
            path.write_text(
                "# runtime settings\n"
                "NEXTWAVE_LLM_PROVIDER=yandex\n"
                "NEXTWAVE_LLM_MODEL=YandexGPT Lite 5\n"
                "NEXTWAVE_LLM_API_KEY=file-secret\n",
                encoding="utf-8",
            )

            result = _runtime_environment(
                path,
                {"NEXTWAVE_LLM_API_KEY": "process-secret"},
                dotenv_path=Path(directory) / "no-such-env",
            )

        self.assertEqual(result["NEXTWAVE_LLM_PROVIDER"], "yandex")
        self.assertEqual(result["NEXTWAVE_LLM_MODEL"], "YandexGPT Lite 5")
        self.assertEqual(result["NEXTWAVE_LLM_API_KEY"], "process-secret")

    def test_runtime_environment_rejects_duplicate_keys(self) -> None:
        from nextwave.__main__ import _runtime_environment

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.env"
            path.write_text("KEY=first\nKEY=second\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate variable KEY"):
                _runtime_environment(path, {}, dotenv_path=Path(directory) / "no-such-env")

    def test_dotenv_overrides_file_and_process_wins(self) -> None:
        from nextwave.__main__ import _runtime_environment

        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / "base.env"
            env_file.write_text(
                "NEXTWAVE_LLM_PROVIDER=huggingface\nSHARED=from-file\n",
                encoding="utf-8",
            )
            dotenv_path = Path(directory) / ".env"
            dotenv_path.write_text(
                "NEXTWAVE_LLM_PROVIDER=yandex\nSHARED=from-dotenv\n",
                encoding="utf-8",
            )

            result = _runtime_environment(
                env_file, {"SHARED": "from-process"}, dotenv_path=dotenv_path
            )

            self.assertEqual(result["NEXTWAVE_LLM_PROVIDER"], "yandex")
            self.assertEqual(result["SHARED"], "from-process")

    @patch("nextwave.__main__.build_discovery_pipeline_from_environment")
    @patch("nextwave.__main__.build_query_resolver_from_environment")
    def test_discovery_run_saves_run_directory(
        self, build_resolver, build_pipeline
    ) -> None:
        from types import SimpleNamespace

        from nextwave.discovery import build_analysis_scope

        scope = build_analysis_scope(
            raw_query="Технологии в ИИ",
            normalized_query="artificial intelligence",
            search_texts=("Технологии в ИИ",),
            languages=("en", "ru"),
            subfield_ids=("1702",),
        )
        build_resolver.return_value.resolve.return_value = SimpleNamespace(scope=scope)

        class StubResult:
            pipeline_version = "discovery-pipeline-v6"

            def __init__(self, plan_id: str) -> None:
                self.plan_id = plan_id

            def to_dict(self):
                return {
                    "plan_id": self.plan_id,
                    "pipeline_version": self.pipeline_version,
                    "documents": ["d1"],
                    "candidate_proposals": {
                        "proposals": [{"proposal_id": "p1"}],
                        "exclusions": [],
                    },
                    "candidate_gate": {
                        "gate_id": "g1",
                        "decisions": [{"proposal_id": "p1", "decision": "accept"}],
                    },
                    "alias_resolution": {"review_suggestions": []},
                    "evidence_extraction": {"proposals": [], "issues": []},
                    "text_extraction": None,
                    "scientific": {"snapshot_path": "snap/oa", "documents": []},
                    "media": {"snapshot_path": "snap/mc", "documents": []},
                    "verification": {"results": []},
                }

        build_pipeline.return_value.execute.side_effect = (
            lambda plan, **kwargs: StubResult(plan.plan_id)
        )

        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / "runtime.env"
            env_file.write_text("", encoding="utf-8")
            output_root = Path(directory) / "runs"
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "discovery-run",
                        "--query",
                        "Технологии в ИИ",
                        "--analysis-id",
                        "analysis-ai-001",
                        "--analysis-scope-key",
                        "ai-nlp-v1",
                        "--domain",
                        "Инфраструктура ИИ",
                        "--env-file",
                        str(env_file),
                        "--output-root",
                        str(output_root),
                    ]
                )

            self.assertEqual(exit_code, 0)
            run_dirs = [path for path in output_root.iterdir() if path.is_dir()]
            self.assertEqual(len(run_dirs), 1)
            for filename in ("plan.json", "pipeline_result.json", "manifest.json"):
                self.assertTrue((run_dirs[0] / filename).is_file())
            self.assertIn("Каталог запуска", stdout.getvalue())

        build_resolver.return_value.resolve.assert_called_once_with("Технологии в ИИ")

    @patch("nextwave.__main__.build_query_resolver_from_environment")
    def test_discovery_run_reports_error_without_traceback(
        self, build_resolver
    ) -> None:
        build_resolver.side_effect = ValueError("NEXTWAVE_LLM_API_KEY must not be blank")
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            exit_code = main(
                [
                    "discovery-run",
                    "--query",
                    "ИИ",
                    "--analysis-id",
                    "analysis-ai-001",
                    "--analysis-scope-key",
                    "ai-nlp-v1",
                    "--domain",
                    "Инфраструктура ИИ",
                    "--env-file",
                    "config/custom.env",
                    "--output-root",
                    "data/development/discovery",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("NEXTWAVE_LLM_API_KEY", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_discovery_run_rejects_bad_date_without_traceback(self) -> None:
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            exit_code = main(
                [
                    "discovery-run",
                    "--query",
                    "ИИ",
                    "--analysis-id",
                    "analysis-ai-001",
                    "--analysis-scope-key",
                    "ai-nlp-v1",
                    "--domain",
                    "Инфраструктура ИИ",
                    "--cutoff-date",
                    "15.09.2026",
                    "--env-file",
                    "config/custom.env",
                    "--output-root",
                    "data/development/discovery",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("--cutoff-date", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    @patch("nextwave.__main__.save_discovery_run")
    @patch("nextwave.__main__.build_discovery_pipeline_from_environment")
    @patch("nextwave.__main__.build_query_resolver_from_environment")
    def test_discovery_run_derives_scope_key_and_domain_from_resolution(
        self, build_resolver, build_pipeline, save_run
    ) -> None:
        from types import SimpleNamespace

        from nextwave.discovery import build_analysis_scope

        scope = build_analysis_scope(
            raw_query="Технологии в ИИ",
            normalized_query="artificial intelligence",
            search_texts=("Технологии в ИИ",),
            languages=("en", "ru"),
            subfield_ids=("1702",),
        )
        build_resolver.return_value.resolve.return_value = SimpleNamespace(scope=scope)
        save_run.return_value = Path("runs") / "analysis-ai-009-abc123"

        with tempfile.TemporaryDirectory() as directory:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "discovery-run",
                        "--query",
                        "Технологии в ИИ",
                        "--analysis-id",
                        "analysis-ai-009",
                        "--env-file",
                        str(Path(directory) / "runtime.env"),
                        "--output-root",
                        str(Path(directory) / "runs"),
                    ]
                )

            self.assertEqual(exit_code, 0)
            _, kwargs = save_run.call_args
            self.assertEqual(kwargs["analysis_scope_key"], scope.scope_id)
            self.assertEqual(kwargs["domain"], "Технологии в ИИ")

    @patch("nextwave.__main__.export_enrichment_plan")
    def test_labeling_enrichment_plan_uses_versioned_default_output(
        self, export
    ) -> None:
        from nextwave.labeling.enrichment_plan import LABELING_ENRICHMENT_PLAN_VERSION

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(["labeling-enrichment-plan", "--bundle", "bundle"])

        self.assertEqual(exit_code, 0)
        _, kwargs = export.call_args
        self.assertEqual(
            kwargs["output_dir"],
            Path("data") / "development" / LABELING_ENRICHMENT_PLAN_VERSION,
        )
        self.assertEqual(
            kwargs["output_dir"],
            Path("data/development/labeling-enrichment-plan-v2"),
        )

    @patch("nextwave.__main__.run_enrichment")
    def test_labeling_enrichment_run_uses_versioned_default_output(
        self, run
    ) -> None:
        from nextwave.labeling.enrichment_run import (
            ENRICHMENT_RESULT_VERSION,
            LabelingEnrichmentRunPaths,
        )

        run.return_value = LabelingEnrichmentRunPaths(
            manifest=Path("out/manifest.json"),
            request_results=Path("out/request_results.jsonl"),
            documents=Path("out/documents.jsonl"),
            coverage=Path("out/coverage.jsonl"),
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                ["labeling-enrichment-run", "--plan", "plan", "--work", "work"]
            )

        self.assertEqual(exit_code, 0)
        _, kwargs = run.call_args
        self.assertEqual(
            kwargs["output_dir"],
            Path("data") / "development" / ENRICHMENT_RESULT_VERSION,
        )
        self.assertEqual(
            kwargs["output_dir"],
            Path("data/development/labeling-enrichment-result-v2"),
        )

    def test_enrichment_commands_have_no_stale_v1_defaults(self) -> None:
        from nextwave.__main__ import _build_parser

        help_text = _build_parser().format_help()
        self.assertNotIn("labeling-enrichment-plan-v1", help_text)
        self.assertNotIn("labeling-enrichment-result-v1", help_text)

    @patch("nextwave.__main__.export_evidence_input_plan")
    def test_evidence_input_plan_uses_versioned_default_output(
        self, export
    ) -> None:
        from nextwave.__main__ import _build_parser
        from nextwave.labeling.evidence_input_plan import (
            LABELING_EVIDENCE_INPUT_PLAN_VERSION,
            LabelingEvidenceInputPlanPaths,
        )

        export.return_value = LabelingEvidenceInputPlanPaths(
            manifest=Path("out/manifest.json"),
            media_reranked=Path("out/media_reranked.jsonl"),
            evidence_input_documents=Path("out/evidence_input_documents.jsonl"),
            coverage=Path("out/coverage.jsonl"),
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main([
                "labeling-evidence-input-plan",
                "--plan", "plan",
                "--result", "result",
                "--relevance", "relevance",
                "--media", "media",
            ])

        self.assertEqual(exit_code, 0)
        _, kwargs = export.call_args
        self.assertEqual(
            kwargs["output_dir"],
            Path("data") / "development" / LABELING_EVIDENCE_INPUT_PLAN_VERSION,
        )
        self.assertEqual(
            kwargs["output_dir"],
            Path("data/development/labeling-evidence-input-plan-v2"),
        )
        help_text = _build_parser().format_help()
        self.assertNotIn("labeling-evidence-input-plan-v1", help_text)


if __name__ == "__main__":
    unittest.main()
