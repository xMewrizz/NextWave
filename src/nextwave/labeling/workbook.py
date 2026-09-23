"""Fill the labeling workbook template from a review queue (blue cells only)."""

from __future__ import annotations

from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import openpyxl

from .queue import LabelingQueue

# Фиксированная метка сборки ради побайтовой воспроизводимости экспорта.
BUILD_STAMP = datetime(2026, 9, 15)

CANDIDATES_SHEET = "Кандидаты"
EVIDENCE_SHEET = "Доказательства"
NOISE_SHEET = "Шум"

HEADER_ROW = 4
FIRST_DATA_ROW = 5


def read_candidate_slots(template_path: str | Path) -> list[dict[str, Any]]:
    """Read template-owned candidate rows: ID, quota class and domain per row."""

    workbook = openpyxl.load_workbook(template_path, read_only=True, data_only=True)
    try:
        sheet = workbook[CANDIDATES_SHEET]
        slots = []
        for row in sheet.iter_rows(min_row=FIRST_DATA_ROW, values_only=True):
            if not row[0]:
                continue
            slots.append(
                {
                    "candidate_id": str(row[0]),
                    "planned_class": str(row[1]),
                    "domain": str(row[6]),
                    "row_number": len(slots) + FIRST_DATA_ROW,
                }
            )
        return slots
    finally:
        workbook.close()


def read_noise_slots(template_path: str | Path) -> list[dict[str, Any]]:
    """Read template-owned noise rows: ID, planned type and domain per row."""

    workbook = openpyxl.load_workbook(template_path, read_only=True, data_only=True)
    try:
        sheet = workbook[NOISE_SHEET]
        slots = []
        for row in sheet.iter_rows(min_row=FIRST_DATA_ROW, values_only=True):
            if not row[0]:
                continue
            slots.append(
                {
                    "noise_id": str(row[0]),
                    "planned_noise_type": str(row[1]),
                    "domain": str(row[5]),
                    "row_number": len(slots) + FIRST_DATA_ROW,
                }
            )
        return slots
    finally:
        workbook.close()


def count_evidence_rows(template_path: str | Path) -> int:
    """Count template-owned evidence rows available for queue proposals."""

    workbook = openpyxl.load_workbook(template_path, read_only=True, data_only=True)
    try:
        sheet = workbook[EVIDENCE_SHEET]
        total = 0
        for row in sheet.iter_rows(min_row=FIRST_DATA_ROW, values_only=True):
            if not row[0]:
                break
            total += 1
        return total
    finally:
        workbook.close()


def _set_cell(sheet: Any, row: int, column: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    sheet[f"{column}{row}"] = value


def fill_labeling_workbook(
    template_path: str | Path, queue: LabelingQueue
) -> bytes:
    """Fill a copy of the template with queue data; template file untouched.

    Only blue (pipeline) cells are written. Grey identifiers, quota classes,
    domains, cutoff dates and all yellow expert cells stay byte-identical.
    Evidence proposals map positionally onto the first template evidence rows.
    """

    workbook = openpyxl.load_workbook(template_path)
    workbook.properties.created = BUILD_STAMP
    workbook.properties.modified = BUILD_STAMP
    candidates = workbook[CANDIDATES_SHEET]
    for item in queue.candidates:
        row = _candidate_row(candidates, item.candidate_id)
        if row is None:
            raise ValueError(f"candidate {item.candidate_id} has no template row")
        _set_cell(candidates, row, "C", item.canonical_name)
        _set_cell(candidates, row, "D", "; ".join(item.aliases))
        _set_cell(candidates, row, "E", item.group_id)
        _set_cell(candidates, row, "F", item.source_query)

    noise = workbook[NOISE_SHEET]
    for item in queue.noise:
        row = _noise_row(noise, item.noise_id)
        if row is None:
            raise ValueError(f"noise {item.noise_id} has no template row")
        _set_cell(noise, row, "C", item.source_query)
        _set_cell(noise, row, "D", item.extracted_text)
        _set_cell(noise, row, "E", item.source_document_url)
        _set_cell(noise, row, "I", item.duplicate_of_candidate_id)

    evidence = workbook[EVIDENCE_SHEET]
    for index, item in enumerate(queue.evidence):
        row = FIRST_DATA_ROW + index
        cell_id = evidence[f"A{row}"].value
        if not cell_id:
            raise ValueError("queue evidence exceeds template evidence rows")
        _set_cell(evidence, row, "B", item.candidate_id)
        _set_cell(evidence, row, "C", item.direction.value)
        _set_cell(evidence, row, "D", item.kind.value)
        _set_cell(evidence, row, "E", item.source_type.value)
        _set_cell(
            evidence, row, "F", item.trust_level.value if item.trust_level else None
        )
        _set_cell(evidence, row, "G", item.title)
        _set_cell(evidence, row, "H", item.url)
        if isinstance(item.published_at, date):
            evidence[f"I{row}"] = item.published_at
        _set_cell(evidence, row, "J", item.organization)
        _set_cell(evidence, row, "K", item.origin_id)
        _set_cell(evidence, row, "L", item.claim)
        _set_cell(evidence, row, "M", item.locator)
        if item.extraction_confidence is not None:
            evidence[f"N{row}"] = item.extraction_confidence

    stream = BytesIO()
    workbook.save(stream)
    workbook.close()
    return stream.getvalue()


def _candidate_row(sheet: Any, candidate_id: str) -> int | None:
    for row in range(FIRST_DATA_ROW, sheet.max_row + 1):
        if sheet[f"A{row}"].value == candidate_id:
            return row
    return None


def _noise_row(sheet: Any, noise_id: str) -> int | None:
    for row in range(FIRST_DATA_ROW, sheet.max_row + 1):
        if sheet[f"A{row}"].value == noise_id:
            return row
    return None
