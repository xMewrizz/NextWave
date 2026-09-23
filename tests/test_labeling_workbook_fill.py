"""Tests for filling the labeling workbook template (blue cells only)."""

from __future__ import annotations

import hashlib
import unittest
from datetime import date
from io import BytesIO
from pathlib import Path

import openpyxl

from nextwave.labeling.contracts import EvidenceDirection, EvidenceKind, SourceType, TrustLevel
from nextwave.labeling.queue import (
    LabelingQueue,
    QueuedCandidate,
    QueuedEvidence,
    QueuedNoise,
)
from nextwave.labeling.workbook import (
    count_evidence_rows,
    fill_labeling_workbook,
    read_candidate_slots,
    read_noise_slots,
)

TEMPLATE = Path("templates/labeling_workbook.xlsx")


def empty_queue() -> LabelingQueue:
    return LabelingQueue(
        candidates=(),
        noise=(),
        evidence=(),
        deficits=(),
        overflow=(),
        dropped=(),
        unknown_trust_count=0,
        missing_date_count=0,
        future_evidence_count=0,
    )


def sample_queue() -> LabelingQueue:
    base = empty_queue()
    candidate_slots = read_candidate_slots(TEMPLATE)
    noise_slots = read_noise_slots(TEMPLATE)
    first_candidate = candidate_slots[0]
    first_noise = noise_slots[0]
    from dataclasses import replace

    candidates = (
        QueuedCandidate(
            candidate_id=first_candidate["candidate_id"],
            planned_class=first_candidate["planned_class"],
            canonical_name="Speculative Decoding",
            aliases=["speculative-decoding"],
            group_id="alias-group-1",
            source_query="Технологии в ИИ",
            domain=first_candidate["domain"],
            analysis_scope_key="edge-v1",
            run_id="run-1",
        ),
    )
    noise = (
        QueuedNoise(
            noise_id=first_noise["noise_id"],
            planned_noise_type=first_noise["planned_noise_type"],
            source_query="Технологии в ИИ",
            extracted_text="Digital Transformation",
            source_document_url="https://example.org/broad",
            domain=first_noise["domain"],
            analysis_scope_key="edge-v1",
            duplicate_of_candidate_id=None,
            origin_kind="exclusion",
        ),
    )
    evidence = (
        QueuedEvidence(
            evidence_id="evidence-001",
            candidate_id=first_candidate["candidate_id"],
            direction=EvidenceDirection.SUPPORT,
            kind=EvidenceKind.TECHNICAL_VALIDATION,
            source_type=SourceType.RESEARCH,
            trust_level=TrustLevel.A,
            title="Study",
            url="https://example.org/paper",
            published_at=date(2026, 1, 1),
            organization="",
            origin_id="doi:10.1234/x",
            claim="A prototype was tested.",
            locator="excerpt[0:10]",
            extraction_confidence=None,
        ),
        QueuedEvidence(
            evidence_id="evidence-002",
            candidate_id=first_candidate["candidate_id"],
            direction=EvidenceDirection.COUNTER,
            kind=EvidenceKind.PUBLICITY_WAVE,
            source_type=SourceType.INDUSTRY_MEDIA,
            trust_level=None,
            title="",
            url="https://example.org/news",
            published_at=None,
            organization="",
            origin_id="https://example.org/news",
            claim="Analysts hype the launch.",
            locator="excerpt[0:10]",
            extraction_confidence=None,
        ),
    )
    return replace(base, candidates=candidates, noise=noise, evidence=evidence)


def load_cells(data: bytes, sheet: str, rows: range, columns: str) -> list:
    workbook = openpyxl.load_workbook(BytesIO(data), read_only=True, data_only=True)
    try:
        target = workbook[sheet]
        return [
            [target[f"{column}{row}"].value for column in columns]
            for row in rows
        ]
    finally:
        workbook.close()


class WorkbookSlotTests(unittest.TestCase):
    def test_reads_template_owned_slots(self) -> None:
        candidates = read_candidate_slots(TEMPLATE)
        noise = read_noise_slots(TEMPLATE)

        self.assertEqual(len(candidates), 100)
        self.assertEqual(candidates[0]["candidate_id"], "team-negative-001")
        self.assertEqual(len(noise), 50)
        self.assertEqual(count_evidence_rows(TEMPLATE), 400)


class WorkbookFillTests(unittest.TestCase):
    def test_fills_blue_cells_and_preserves_grey_and_yellow(self) -> None:
        before = TEMPLATE.read_bytes()
        filled = fill_labeling_workbook(TEMPLATE, sample_queue())
        after = TEMPLATE.read_bytes()

        self.assertEqual(before, after)
        candidates = load_cells(filled, "Кандидаты", range(5, 105), "ABCDEFGHI")
        first_row = next(row for row in candidates if row[0] == "team-negative-001")
        self.assertEqual(first_row[2], "Speculative Decoding")
        self.assertEqual(first_row[3], "speculative-decoding")
        self.assertEqual(first_row[4], "alias-group-1")
        self.assertEqual(first_row[5], "Технологии в ИИ")
        # Grey cells untouched.
        self.assertEqual(first_row[1], "mature")
        self.assertEqual(first_row[6], "Edge")

        noise_rows = load_cells(filled, "Шум", range(5, 55), "ABCDEFGHI")
        first_noise = next(row for row in noise_rows if row[0] == "noise-001")
        self.assertEqual(first_noise[3], "Digital Transformation")
        self.assertEqual(first_noise[4], "https://example.org/broad")

        evidence_rows = load_cells(filled, "Доказательства", range(5, 7), "ABCDEFGHIJKLMN")
        self.assertEqual(evidence_rows[0][1], "team-negative-001")
        self.assertEqual(evidence_rows[0][2], "support")
        self.assertEqual(evidence_rows[0][5], "A")
        self.assertEqual(evidence_rows[1][2], "counter")
        # Unknown trust stays blank; unknown date stays blank.
        self.assertIsNone(evidence_rows[1][5])
        self.assertIsNone(evidence_rows[1][8])

    def test_untouched_sheets_stay_identical(self) -> None:
        filled = fill_labeling_workbook(TEMPLATE, sample_queue())
        for sheet in ("План", "Решения", "Справочник"):
            original = load_cells(TEMPLATE.read_bytes(), sheet, range(2, 30), "ABCDEF")
            rewritten = load_cells(filled, sheet, range(2, 30), "ABCDEF")
            self.assertEqual(original, rewritten)

    def test_unknown_candidate_id_is_rejected(self) -> None:
        from dataclasses import replace

        queue = replace(
            empty_queue(),
            candidates=(
                QueuedCandidate(
                    candidate_id="team-negative-999",
                    planned_class="mature",
                    canonical_name="Ghost",
                    aliases=(),
                    group_id="alias-group-9",
                    source_query="q",
                    domain="Edge",
                    analysis_scope_key="edge-v1",
                    run_id="run-1",
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "no template row"):
            fill_labeling_workbook(TEMPLATE, queue)

    def test_template_sha256_is_unchanged(self) -> None:
        digest = hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()
        fill_labeling_workbook(TEMPLATE, sample_queue())
        self.assertEqual(hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
