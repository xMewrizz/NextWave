"""Atomic storage for immutable raw source snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path, PurePosixPath

from .contracts import ConnectorRequest, RawResponseArtifact, SnapshotManifest


class SnapshotWriter:
    """Build one snapshot in a staging directory and publish it atomically."""

    def __init__(self, root: Path, snapshot_id: str) -> None:
        self.root = root.resolve()
        self.snapshot_id = snapshot_id
        self.final_path = self.root / snapshot_id
        self.root.mkdir(parents=True, exist_ok=True)
        if self.final_path.exists():
            raise FileExistsError(f"snapshot already exists: {self.final_path}")
        self._staging_path = Path(
            tempfile.mkdtemp(prefix=f".{snapshot_id}-", suffix=".staging", dir=self.root)
        )
        self._finalized = False

    def write_response(
        self,
        request: ConnectorRequest,
        payload: bytes,
        media_type: str,
        retrieved_at: datetime,
    ) -> RawResponseArtifact:
        """Store the response bytes exactly as received from the connector."""

        self._require_open()
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        extension = ".json" if media_type.split(";", 1)[0].strip() == "application/json" else ".bin"
        relative_path = PurePosixPath(
            "raw",
            request.connector_id.value,
            f"{request.request_id}{extension}",
        )
        target = self._staging_path.joinpath(*relative_path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"response already stored: {relative_path}")
        target.write_bytes(payload)
        return RawResponseArtifact(
            uri=relative_path.as_posix(),
            media_type=media_type,
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            retrieved_at=retrieved_at,
        )

    def finalize(self, manifest: SnapshotManifest) -> Path:
        """Validate every referenced raw response and publish the snapshot directory."""

        self._require_open()
        if manifest.snapshot_id != self.snapshot_id:
            raise ValueError("manifest snapshot_id does not match writer snapshot_id")
        if self.final_path.exists():
            raise FileExistsError(f"snapshot already exists: {self.final_path}")

        referenced_uris = {
            run.artifact.uri for run in manifest.runs if run.artifact is not None
        }
        stored_uris = {
            path.relative_to(self._staging_path).as_posix()
            for path in self._staging_path.rglob("*")
            if path.is_file()
        }
        if stored_uris != referenced_uris:
            missing = sorted(referenced_uris - stored_uris)
            untracked = sorted(stored_uris - referenced_uris)
            raise ValueError(
                "snapshot artifacts do not match manifest: "
                f"missing={missing}, untracked={untracked}"
            )

        for run in manifest.runs:
            if run.artifact is None:
                continue
            artifact_path = self._staging_path.joinpath(*PurePosixPath(run.artifact.uri).parts)
            payload = artifact_path.read_bytes()
            if len(payload) != run.artifact.size_bytes:
                raise ValueError(f"artifact size mismatch: {run.artifact.uri}")
            if hashlib.sha256(payload).hexdigest() != run.artifact.sha256:
                raise ValueError(f"artifact checksum mismatch: {run.artifact.uri}")

        manifest_bytes = (
            json.dumps(
                manifest.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        (self._staging_path / "manifest.json").write_bytes(manifest_bytes)
        os.replace(self._staging_path, self.final_path)
        self._finalized = True
        return self.final_path

    def _require_open(self) -> None:
        if self._finalized:
            raise RuntimeError("snapshot writer is already finalized")
