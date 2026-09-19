from __future__ import annotations

import io
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


if __name__ == "__main__":
    unittest.main()
