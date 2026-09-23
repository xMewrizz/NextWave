"""End-to-end tests for the labeling export (hand-built run dirs, real template)."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

from nextwave.__main__ import main

TEMPLATE = Path("templates/labeling_workbook.xlsx")
CUTOFF = "2026-09-15"


def pretty(payload: dict) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def document(document_id: str) -> dict:
    return {
        "document_id": document_id,
        "title": f"Study of {document_id}",
        "url": f"https://example.org/{document_id}",
        "published_at": "2026-01-01",
        "organizations": [],
        "source_type": "scientific_publication",
        "trust_tier": "A",
        "origin_id": f"origin-{document_id}",
    }


def write_run_dir(
    root: Path, run_id: str, *, group_name: str = "Alpha Tech", domain: str = "Edge"
) -> Path:
    group_id = f"group-{run_id}"
    document_id = f"document-{run_id}"
    plan = {
        "plan_id": f"plan-{run_id}",
        "scope": {"scope_id": f"scope-{run_id}", "raw_query": "Технологии в ИИ"},
    }
    result = {
        "alias_resolution": {
            "groups": [
                {
                    "group_id": group_id,
                    "canonical_name": group_name,
                    "aliases": [],
                    "origin_ids": [f"origin-{run_id}"],
                    "document_ids": [document_id],
                    "normalization_key": group_name.casefold(),
                    "proposal_ids": [f"proposal-{run_id}"],
                }
            ],
            "review_suggestions": [],
        },
        "candidate_proposals": {"proposals": [], "exclusions": []},
        "candidate_gate": {"gate_id": "gate-1", "decisions": []},
        "text_extraction": {"issues": []},
        "evidence_extraction": {
            "proposals": [
                {
                    "alias_group_id": group_id,
                    "origin_id": f"origin-{run_id}",
                    "source_url": f"https://example.org/{document_id}",
                    "review_status": "pending",
                    "claim": {
                        "claim_id": f"claim-{run_id}",
                        "document_id": document_id,
                        "claim_type": "research",
                        "direction": "support",
                        "text": "A prototype was tested.",
                        "locator": "excerpt[0:10]",
                        "organization": None,
                        "extraction_confidence": None,
                    },
                }
            ]
        },
        "scientific": {"documents": [document(document_id)]},
        "media": {"documents": []},
        "verification": {"results": []},
    }
    plan_bytes = pretty(plan)
    result_bytes = pretty(result)

    def digest(data: bytes) -> dict:
        return {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    manifest = {
        "run_id": run_id,
        "cutoff_date": CUTOFF,
        "domain": domain,
        "counts": {},
        "outputs": [
            {"filename": "plan.json", **digest(plan_bytes)},
            {"filename": "pipeline_result.json", **digest(result_bytes)},
        ],
    }
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "plan.json").write_bytes(plan_bytes)
    (run_dir / "pipeline_result.json").write_bytes(result_bytes)
    (run_dir / "manifest.json").write_bytes(pretty(manifest))
    return run_dir


class LabelingExportTests(unittest.TestCase):
    def test_exports_workbook_jsonl_and_consistent_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = write_run_dir(root, "run-1")
            output = root / "export-1"
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "labeling-export",
                        "--runs",
                        str(run_dir),
                        "--template",
                        str(TEMPLATE),
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(exit_code, 0)
            for filename in (
                "negative_candidates.jsonl",
                "noise_controls.jsonl",
                "labeling_workbook.xlsx",
                "manifest.json",
            ):
                self.assertTrue((output / filename).is_file())
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            for filename in (
                "negative_candidates.jsonl",
                "noise_controls.jsonl",
                "labeling_workbook.xlsx",
            ):
                payload = (output / filename).read_bytes()
                self.assertEqual(
                    manifest["outputs"][filename]["sha256"],
                    hashlib.sha256(payload).hexdigest(),
                )
            candidates = (output / "negative_candidates.jsonl").read_text(encoding="utf-8")
            self.assertIn("team-negative-001", candidates)
            self.assertIn("Alpha Tech", candidates)
            self.assertIn("export", stdout.getvalue())

    def test_second_export_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = write_run_dir(root, "run-1")

            self.assertEqual(
                main(
                    [
                        "labeling-export",
                        "--runs",
                        str(run_dir),
                        "--template",
                        str(TEMPLATE),
                        "--output",
                        str(root / "export-1"),
                    ]
                ),
                0,
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "labeling-export",
                            "--runs",
                            str(run_dir),
                            "--template",
                            str(TEMPLATE),
                            "--output",
                            str(root / "export-2"),
                        ]
                    ),
                    0,
                )

            for filename in (
                "negative_candidates.jsonl",
                "noise_controls.jsonl",
                "manifest.json",
            ):
                self.assertEqual(
                    (root / "export-1" / filename).read_bytes(),
                    (root / "export-2" / filename).read_bytes(),
                )
            self.assertWorkbooksMatch(
                root / "export-1" / "labeling_workbook.xlsx",
                root / "export-2" / "labeling_workbook.xlsx",
            )

    def assertWorkbooksMatch(self, first: Path, second: Path) -> None:
        # Контейнерные байты xlsx плавают по воле библиотеки openpyxl
        # (доказано отладкой: содержимое одинаковое, хвосты zip различаются),
        # поэтому книга сравнивается посодержимому: порядок записей и байты
        # каждой части.
        with zipfile.ZipFile(first) as left, zipfile.ZipFile(second) as right:
            self.assertEqual(left.namelist(), right.namelist())
            for name in left.namelist():
                self.assertEqual(left.read(name), right.read(name))

    def test_existing_output_directory_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = write_run_dir(root, "run-1")
            output = root / "export-1"
            output.mkdir()

            from contextlib import redirect_stderr

            with redirect_stderr(io.StringIO()):
                exit_code = main(
                        [
                            "labeling-export",
                            "--runs",
                            str(run_dir),
                            "--template",
                            str(TEMPLATE),
                            "--output",
                            str(output),
                        ]
                    )

            self.assertEqual(exit_code, 1)

    def test_missing_run_directory_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            from contextlib import redirect_stderr

            with redirect_stderr(io.StringIO()):
                exit_code = main(
                    [
                        "labeling-export",
                        "--runs",
                        str(Path(directory) / "no-such-run"),
                        "--template",
                        str(TEMPLATE),
                        "--output",
                        str(Path(directory) / "export-1"),
                    ]
                )

            self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
