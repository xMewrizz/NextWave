"""Build the complete versioned organizer dataset artifact bundle."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .artifacts import (
    POSITIVE_ANNOTATIONS_FILENAME,
    POSITIVE_CANDIDATES_FILENAME,
    publish_artifact_bundle,
    render_jsonl,
)
from .contracts import FileDigest, HeaderMapping, OrganizerDatasetManifest
from .organizer_xlsx import (
    DATA_END_ROW,
    DATA_START_ROW,
    EXPECTED_HEADERS,
    EXPECTED_RECORD_COUNT,
    HEADER_ROW,
    ORGANIZER_CUTOFF_DATE,
    SHEET_NAME,
    read_organizer_workbook,
)

DATASET_VERSION = "organizer-positive-2026-09-15-v1"
ADAPTER_VERSION = "organizer-xlsx-v1"
MANIFEST_FILENAME = "manifest.json"

HEADER_OUTPUT_FIELDS = (
    "source_number",
    "canonical_name",
    "domain",
    "companies_raw",
    "expert_rationale",
    "expert_stage",
    "expert_mention_trend",
    "expert_score",
    "sources_raw",
)


@dataclass(frozen=True, slots=True)
class OrganizerDatasetPaths:
    """Paths of one complete organizer dataset build."""

    candidates: Path
    annotations: Path
    manifest: Path


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _digest_file(path: Path) -> FileDigest:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return FileDigest(filename=path.name, size_bytes=size, sha256=digest.hexdigest())


def _digest_bytes(filename: str, content: bytes) -> FileDigest:
    return FileDigest(
        filename=filename,
        size_bytes=len(content),
        sha256=_sha256_bytes(content),
    )


def _render_manifest(manifest: OrganizerDatasetManifest) -> bytes:
    text = json.dumps(
        manifest.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    )
    return (text + "\n").encode("utf-8")


def build_organizer_dataset(
    input_path: str | Path, output_directory: str | Path
) -> OrganizerDatasetPaths:
    """Validate the XLSX and publish two JSONL files with their manifest."""

    source_path = Path(input_path)
    dataset = read_organizer_workbook(source_path)
    candidates_bytes = render_jsonl(dataset.candidates)
    annotations_bytes = render_jsonl(dataset.annotations)

    output_digests = (
        _digest_bytes(POSITIVE_CANDIDATES_FILENAME, candidates_bytes),
        _digest_bytes(POSITIVE_ANNOTATIONS_FILENAME, annotations_bytes),
    )
    manifest = OrganizerDatasetManifest(
        dataset_version=DATASET_VERSION,
        adapter_version=ADAPTER_VERSION,
        cutoff_date=ORGANIZER_CUTOFF_DATE,
        source=_digest_file(source_path),
        sheet_name=SHEET_NAME,
        header_row=HEADER_ROW,
        data_start_row=DATA_START_ROW,
        data_end_row=DATA_END_ROW,
        input_record_count=EXPECTED_RECORD_COUNT,
        accepted_record_count=len(dataset.candidates),
        rejected_record_count=0,
        header_mapping=tuple(
            HeaderMapping(source_header, output_field)
            for source_header, output_field in zip(
                EXPECTED_HEADERS, HEADER_OUTPUT_FIELDS, strict=True
            )
        ),
        outputs=output_digests,
        validation_errors=(),
    )
    manifest_bytes = _render_manifest(manifest)

    paths = publish_artifact_bundle(
        {
            POSITIVE_CANDIDATES_FILENAME: candidates_bytes,
            POSITIVE_ANNOTATIONS_FILENAME: annotations_bytes,
            MANIFEST_FILENAME: manifest_bytes,
        },
        output_directory,
    )
    return OrganizerDatasetPaths(
        candidates=paths[POSITIVE_CANDIDATES_FILENAME],
        annotations=paths[POSITIVE_ANNOTATIONS_FILENAME],
        manifest=paths[MANIFEST_FILENAME],
    )
