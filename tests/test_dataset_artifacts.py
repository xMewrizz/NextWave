from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from nextwave.datasets import (
    IdentityStatus,
    OrganizerAnnotationRecord,
    OrganizerArtifactError,
    ParsedOrganizerDataset,
    PositiveCandidateRecord,
    write_organizer_jsonl,
)


def make_dataset() -> ParsedOrganizerDataset:
    candidates: list[PositiveCandidateRecord] = []
    annotations: list[OrganizerAnnotationRecord] = []
    for number in range(1, 101):
        record_id = f"organizer-{number:03d}"
        candidates.append(
            PositiveCandidateRecord(
                record_id=record_id,
                source_row=number + 2,
                canonical_name=f"Технология {number}",
                domain="Финтех",
                analysis_scope_key="fintech-v1",
                aliases=(),
                group_id=record_id,
                identity_status=IdentityStatus.PENDING_REVIEW,
                cutoff_date=date(2026, 9, 15),
            )
        )
        annotations.append(
            OrganizerAnnotationRecord(
                record_id=record_id,
                source_number=number,
                companies_raw=f"Компания {number}",
                expert_rationale=f"Обоснование {number}",
                expert_stage="Исследование",
                expert_mention_trend="Рост упоминаний",
                expert_score=5,
                sources_raw=f"https://example.org/{number}",
            )
        )
    return ParsedOrganizerDataset(tuple(candidates), tuple(annotations))


class DatasetArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.dataset = make_dataset()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_writer_creates_two_separate_utf8_jsonl_files(self) -> None:
        paths = write_organizer_jsonl(self.dataset, self.root / "dataset")

        candidate_bytes = paths.candidates.read_bytes()
        annotation_bytes = paths.annotations.read_bytes()
        self.assertEqual(candidate_bytes.count(b"\n"), 100)
        self.assertEqual(annotation_bytes.count(b"\n"), 100)
        self.assertNotIn(b"\r\n", candidate_bytes)
        self.assertNotIn(b"\r\n", annotation_bytes)

        candidate = json.loads(candidate_bytes.splitlines()[0])
        annotation = json.loads(annotation_bytes.splitlines()[0])
        self.assertEqual(candidate["canonical_name"], "Технология 1")
        self.assertNotIn("expert_rationale", candidate)
        self.assertEqual(annotation["expert_rationale"], "Обоснование 1")
        self.assertNotIn("canonical_name", annotation)

    def test_identical_dataset_produces_identical_bytes(self) -> None:
        first = write_organizer_jsonl(self.dataset, self.root / "first")
        second = write_organizer_jsonl(self.dataset, self.root / "second")

        self.assertEqual(first.candidates.read_bytes(), second.candidates.read_bytes())
        self.assertEqual(first.annotations.read_bytes(), second.annotations.read_bytes())

    def test_existing_output_directory_is_not_modified(self) -> None:
        output = self.root / "existing"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("keep", encoding="utf-8")

        with self.assertRaisesRegex(OrganizerArtifactError, "already exists"):
            write_organizer_jsonl(self.dataset, output)

        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
        self.assertEqual(list(output.iterdir()), [marker])


if __name__ == "__main__":
    unittest.main()
