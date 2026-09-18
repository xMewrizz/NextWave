"""Deterministic JSONL artifacts for the organizer dataset."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .organizer_xlsx import ParsedOrganizerDataset

POSITIVE_CANDIDATES_FILENAME = "positive_candidates.jsonl"
POSITIVE_ANNOTATIONS_FILENAME = "positive_annotations.jsonl"


class JsonRecord(Protocol):
    """A contract record that has a deterministic dictionary representation."""

    def to_dict(self) -> dict[str, Any]: ...


class OrganizerArtifactError(ValueError):
    """The requested artifact bundle cannot be safely published."""


@dataclass(frozen=True, slots=True)
class OrganizerJsonlPaths:
    """Paths of one successfully published JSONL bundle."""

    candidates: Path
    annotations: Path


def render_jsonl(records: Iterable[JsonRecord]) -> bytes:
    """Render records as compact UTF-8 JSONL with a final LF."""

    lines = [
        json.dumps(
            record.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        for record in records
    ]
    if not lines:
        raise OrganizerArtifactError("JSONL artifact must contain at least one record")
    return ("\n".join(lines) + "\n").encode("utf-8")


def write_organizer_jsonl(
    dataset: ParsedOrganizerDataset, output_directory: str | Path
) -> OrganizerJsonlPaths:
    """Publish both JSONL files together into a new directory."""

    candidates_bytes = render_jsonl(dataset.candidates)
    annotations_bytes = render_jsonl(dataset.annotations)

    target = Path(output_directory)
    if target.exists():
        raise OrganizerArtifactError(f"output directory already exists: {target}")

    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=parent))
    try:
        (staging / POSITIVE_CANDIDATES_FILENAME).write_bytes(candidates_bytes)
        (staging / POSITIVE_ANNOTATIONS_FILENAME).write_bytes(annotations_bytes)
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return OrganizerJsonlPaths(
        candidates=target / POSITIVE_CANDIDATES_FILENAME,
        annotations=target / POSITIVE_ANNOTATIONS_FILENAME,
    )
