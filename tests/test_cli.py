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
                _runtime_environment(path, {})

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


if __name__ == "__main__":
    unittest.main()
