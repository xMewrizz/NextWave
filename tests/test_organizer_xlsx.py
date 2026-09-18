from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook, load_workbook

from nextwave.datasets import (
    EXPECTED_HEADERS,
    OrganizerWorkbookError,
    read_organizer_workbook,
)


def create_workbook(path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Слабые сигналы"
    sheet.cell(1, 2, "Тестовый набор")
    for column, header in enumerate(EXPECTED_HEADERS, start=2):
        sheet.cell(2, column, header)

    domains = (
        "Edge",
        "Защита ИИ",
        "Индустриальный ИИ",
        "Инфраструктура ИИ",
        "Роботы",
        "Финтех",
    )
    for number in range(1, 101):
        row = number + 2
        values = (
            number,
            f"  Технология   {number}  ",
            domains[(number - 1) % len(domains)],
            f"Компания {number}",
            f"Обоснование {number}",
            "Исследование",
            "Рост упоминаний",
            3 + number % 5,
            f"https://example.org/{number}",
        )
        for column, value in enumerate(values, start=2):
            sheet.cell(row, column, value)
    workbook.save(path)


class OrganizerWorkbookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary_directory.name) / "organizer.xlsx"
        create_workbook(self.path)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_valid_workbook_is_parsed_into_separate_records(self) -> None:
        dataset = read_organizer_workbook(self.path)

        self.assertEqual(len(dataset.candidates), 100)
        self.assertEqual(len(dataset.annotations), 100)
        self.assertEqual(dataset.candidates[0].record_id, "organizer-001")
        self.assertEqual(dataset.candidates[-1].record_id, "organizer-100")
        self.assertEqual(dataset.candidates[0].canonical_name, "Технология 1")
        self.assertEqual(dataset.candidates[0].analysis_scope_key, "edge-v1")
        self.assertEqual(dataset.annotations[0].expert_rationale, "Обоснование 1")

    def test_changed_header_is_rejected_with_cell_name(self) -> None:
        workbook = load_workbook(self.path)
        workbook.active.cell(2, 3, "Другое название")
        workbook.save(self.path)

        with self.assertRaisesRegex(OrganizerWorkbookError, "C2"):
            read_organizer_workbook(self.path)

    def test_missing_record_is_rejected(self) -> None:
        workbook = load_workbook(self.path)
        workbook.active.delete_rows(102)
        workbook.save(self.path)

        with self.assertRaisesRegex(OrganizerWorkbookError, "expected data range"):
            read_organizer_workbook(self.path)

    def test_broken_sequence_number_is_rejected(self) -> None:
        workbook = load_workbook(self.path)
        workbook.active.cell(12, 2, 99)
        workbook.save(self.path)

        with self.assertRaisesRegex(OrganizerWorkbookError, "B12"):
            read_organizer_workbook(self.path)

    def test_score_outside_contract_is_rejected(self) -> None:
        workbook = load_workbook(self.path)
        workbook.active.cell(3, 9, 8)
        workbook.save(self.path)

        with self.assertRaisesRegex(OrganizerWorkbookError, "I3"):
            read_organizer_workbook(self.path)

    def test_formula_is_rejected(self) -> None:
        workbook = load_workbook(self.path)
        workbook.active.cell(3, 9, "=3+4")
        workbook.save(self.path)

        with self.assertRaisesRegex(OrganizerWorkbookError, "formula.*I3"):
            read_organizer_workbook(self.path)


if __name__ == "__main__":
    unittest.main()
