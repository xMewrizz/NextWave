"""Deterministic JSONL artifacts for the organizer dataset."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterable, Mapping
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


def publish_artifact_bundle(
    files: Mapping[str, bytes], output_directory: str | Path
) -> dict[str, Path]:
    """Publish a complete set of files together into a new directory."""

    if not files:
        raise OrganizerArtifactError("artifact bundle must not be empty")
    for filename, content in files.items():
        if not filename or "/" in filename or "\\" in filename:
            raise OrganizerArtifactError(f"invalid artifact filename: {filename!r}")
        if not isinstance(content, bytes):
            raise OrganizerArtifactError(f"artifact {filename!r} must contain bytes")

    target = Path(output_directory)
    if target.exists():
        raise OrganizerArtifactError(f"output directory already exists: {target}")

    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=parent))
    try:
        for filename, content in files.items():
            (staging / filename).write_bytes(content)
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {filename: target / filename for filename in files}


def write_organizer_jsonl(
    dataset: ParsedOrganizerDataset, output_directory: str | Path
) -> OrganizerJsonlPaths:
    """Publish both JSONL files together into a new directory."""

    candidates_bytes = render_jsonl(dataset.candidates)
    annotations_bytes = render_jsonl(dataset.annotations)

    paths = publish_artifact_bundle(
        {
            POSITIVE_CANDIDATES_FILENAME: candidates_bytes,
            POSITIVE_ANNOTATIONS_FILENAME: annotations_bytes,
        },
        output_directory,
    )

    return OrganizerJsonlPaths(
        candidates=paths[POSITIVE_CANDIDATES_FILENAME],
        annotations=paths[POSITIVE_ANNOTATIONS_FILENAME],
    )
