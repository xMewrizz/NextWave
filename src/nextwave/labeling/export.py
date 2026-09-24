"""Export a labeling review queue into workbook, JSONL and manifest files."""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..datasets.artifacts import publish_artifact_bundle
from ..discovery.run_store import assert_run_labeling_eligible, load_discovery_run
from .contracts import LABELING_CUTOFF_DATE, RUBRIC_VERSION
from .queue import (
    QUEUE_SCHEMA_VERSION,
    CandidateSlot,
    NoiseSlot,
    build_labeling_queue,
    queue_to_jsonl,
    resolve_run_domains,
)
from .workbook import (
    count_evidence_rows,
    fill_labeling_workbook,
    read_candidate_slots,
    read_noise_slots,
)

LABELING_EXPORT_MANIFEST_VERSION = "labeling-export-manifest-v1"
NEGATIVE_CANDIDATES_FILENAME = "negative_candidates.jsonl"
NOISE_CONTROLS_FILENAME = "noise_controls.jsonl"
WORKBOOK_FILENAME = "labeling_workbook.xlsx"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True, slots=True)
class LabelingExportPaths:
    """Paths of one successfully published labeling export."""

    workbook: Path
    candidates: Path
    noise: Path
    manifest: Path


def _digest(data: bytes) -> dict[str, Any]:
    return {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _normalize_workbook_bytes(data: bytes) -> bytes:
    """Rewrite xlsx container bytes deterministically.

    openpyxl output floats across library versions and time boundaries:
    zip entry headers carry current timestamps, docProps/core.xml carries
    created/modified stamps, and entry order is not guaranteed. All three
    are canonicalized here, so identical content always yields identical
    container bytes and the manifest hash stays stable.
    """
    reader = zipfile.ZipFile(io.BytesIO(data))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as writer:
        for item in sorted(reader.infolist(), key=lambda entry: entry.filename):
            payload = reader.read(item.filename)
            if item.filename == "docProps/core.xml":
                payload = _pin_core_timestamps(payload)
            info = zipfile.ZipInfo(item.filename, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 0
            info.external_attr = 0
            info.extra = b""
            writer.writestr(info, payload)
    return buffer.getvalue()


_CORE_TIMESTAMP = re.compile(
    rb"<dcterms:(created|modified)[^>]*>.*?</dcterms:(created|modified)>"
)


def _pin_core_timestamps(payload: bytes) -> bytes:
    """Replace created/modified stamps in core.xml with a fixed value."""
    return _CORE_TIMESTAMP.sub(
        lambda match: b"<dcterms:%s xsi:type=\"dcterms:W3CDTF\">1980-01-01T00:00:00Z</dcterms:%s>"
        % (match.group(1), match.group(2)),
        payload,
    )


def export_labeling_bundle(
    *,
    run_dirs: tuple[str | Path, ...],
    template_path: str | Path,
    output_dir: str | Path,
    run_domains: Mapping[str, str] | None = None,
) -> LabelingExportPaths:
    """Build the queue from runs and publish workbook, JSONL and manifest."""

    if not run_dirs:
        raise ValueError("at least one discovery run directory is required")
    template = Path(template_path)
    if not template.is_file():
        raise FileNotFoundError(f"labeling template is missing: {template}")

    runs = tuple(load_discovery_run(run_dir) for run_dir in run_dirs)
    for run in runs:
        assert_run_labeling_eligible(run)

    candidate_rows = read_candidate_slots(template)
    noise_rows = read_noise_slots(template)
    candidate_slots = tuple(
        CandidateSlot(
            candidate_id=row["candidate_id"],
            domain=row["domain"],
        )
        for row in candidate_rows
    )
    noise_slots = tuple(
        NoiseSlot(
            noise_id=row["noise_id"],
            planned_noise_type=row["planned_noise_type"],
            domain=row["domain"],
        )
        for row in noise_rows
    )
    queue = build_labeling_queue(
        runs,
        candidate_slots=candidate_slots,
        noise_slots=noise_slots,
        run_domains=run_domains,
    )
    workbook_bytes = _normalize_workbook_bytes(fill_labeling_workbook(template, queue))
    negative_bytes, noise_bytes = queue_to_jsonl(queue)

    filled_candidates: dict[str, int] = {}
    for item in queue.candidates:
        filled_candidates[item.domain] = filled_candidates.get(item.domain, 0) + 1
    filled_candidate_strata: dict[str, int] = {}
    for item in queue.candidates:
        stratum = getattr(item, "selection_stratum", "single_origin") or "single_origin"
        filled_candidate_strata[stratum] = filled_candidate_strata.get(stratum, 0) + 1
    filled_noise: dict[str, int] = {}
    for item in queue.noise:
        filled_noise[item.planned_noise_type] = (
            filled_noise.get(item.planned_noise_type, 0) + 1
        )
    manifest_dict: dict[str, Any] = {
        "schema_version": LABELING_EXPORT_MANIFEST_VERSION,
        "export_version": QUEUE_SCHEMA_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "cutoff_date": LABELING_CUTOFF_DATE.isoformat(),
        "inputs": [
            {
                "run_id": run.run_id,
                "domain": resolve_run_domains((run,), run_domains)[run.run_id],
                "cutoff_date": run.manifest.get("cutoff_date"),
                "counts": run.manifest.get("counts"),
            }
            for run in runs
        ],
        "template": {
            "filename": template.name,
            **_digest(template.read_bytes()),
            "candidate_rows": len(candidate_rows),
            "noise_rows": len(noise_rows),
            "evidence_rows": count_evidence_rows(template),
        },
        "filled_candidates": filled_candidates,
        "filled_candidate_strata": filled_candidate_strata,
        "filled_noise": filled_noise,
        "evidence_rows": len(queue.evidence),
        "deficits": [
            {"area": item.area, "need": item.need, "missing": item.missing}
            for item in queue.deficits
        ],
        "overflow": [
            {
                "kind": item.kind,
                "key": item.key,
                "run_id": item.run_id,
                "reason": item.reason,
            }
            for item in queue.overflow
        ],
        "dropped": [
            {"reason": item.reason, "detail": item.detail} for item in queue.dropped
        ],
        "unknown_trust_count": queue.unknown_trust_count,
        "missing_date_count": queue.missing_date_count,
        "future_evidence_count": queue.future_evidence_count,
        "validation_errors": [],
        "outputs": {
            NEGATIVE_CANDIDATES_FILENAME: _digest(negative_bytes),
            NOISE_CONTROLS_FILENAME: _digest(noise_bytes),
            WORKBOOK_FILENAME: _digest(workbook_bytes),
        },
    }
    manifest_bytes = (
        json.dumps(manifest_dict, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    paths = publish_artifact_bundle(
        {
            NEGATIVE_CANDIDATES_FILENAME: negative_bytes,
            NOISE_CONTROLS_FILENAME: noise_bytes,
            WORKBOOK_FILENAME: workbook_bytes,
            MANIFEST_FILENAME: manifest_bytes,
        },
        output_dir,
    )
    return LabelingExportPaths(
        workbook=paths[WORKBOOK_FILENAME],
        candidates=paths[NEGATIVE_CANDIDATES_FILENAME],
        noise=paths[NOISE_CONTROLS_FILENAME],
        manifest=paths[MANIFEST_FILENAME],
    )
