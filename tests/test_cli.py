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
            "NEXTWAVE_LLM_PROVIDER": "openai",
            "NEXTWAVE_LLM_MODEL": "gpt-4.1",
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
                "NEXTWAVE_LLM_PROVIDER=openai\n"
                "NEXTWAVE_LLM_MODEL=gpt-4.1\n"
                "NEXTWAVE_LLM_API_KEY=file-secret\n",
                encoding="utf-8",
            )

            result = _runtime_environment(
                path,
                {"NEXTWAVE_LLM_API_KEY": "process-secret"},
            )

        self.assertEqual(result["NEXTWAVE_LLM_PROVIDER"], "openai")
        self.assertEqual(result["NEXTWAVE_LLM_MODEL"], "gpt-4.1")
        self.assertEqual(result["NEXTWAVE_LLM_API_KEY"], "process-secret")

    def test_runtime_environment_rejects_duplicate_keys(self) -> None:
        from nextwave.__main__ import _runtime_environment

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.env"
            path.write_text("KEY=first\nKEY=second\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate variable KEY"):
                _runtime_environment(path, {})


if __name__ == "__main__":
    unittest.main()
