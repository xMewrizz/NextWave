from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from nextwave.sources import (
    ConnectorId,
    ConnectorRequest,
    ConnectorRun,
    ConnectorStatus,
    QueryParameter,
    QueryPurpose,
    RetrievalChannel,
    SnapshotManifest,
    SnapshotWriter,
    SourceQuery,
)

NOW = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
RAW_PAYLOAD = b'{"meta":{"count":1},"results":[{"id":"W1"}]}'


def make_query() -> SourceQuery:
    return SourceQuery(
        query_id="query-ai-001",
        analysis_scope_id="scope-ai-001",
        purpose=QueryPurpose.DISCOVERY,
        raw_query="Технологии в ИИ",
        normalized_query="artificial intelligence",
        search_texts=("artificial intelligence",),
        published_from=date(2025, 9, 16),
        published_until=date(2026, 9, 15),
        cutoff_date=date(2026, 9, 15),
    )


def make_request() -> ConnectorRequest:
    return ConnectorRequest(
        request_id="request-openalex-001",
        query_id="query-ai-001",
        connector_id=ConnectorId.OPENALEX,
        channel=RetrievalChannel.TEXT,
        endpoint="https://api.openalex.org/works",
        parameters=(
            QueryParameter("filter", "from_publication_date:2025-09-16"),
            QueryParameter("search", "artificial intelligence"),
        ),
    )


def make_manifest(artifact) -> SnapshotManifest:
    run = ConnectorRun(
        run_id="run-openalex-001",
        request=make_request(),
        status=ConnectorStatus.SUCCESS,
        started_at=NOW,
        finished_at=NOW + timedelta(milliseconds=200),
        http_status=200,
        returned_records=1,
        artifact=artifact,
    )
    return SnapshotManifest(
        snapshot_id="snapshot-ai-001",
        snapshot_version="snapshot-v1",
        analysis_id="analysis-ai-001",
        created_at=NOW,
        query=make_query(),
        runs=(run,),
    )


class SnapshotWriterTests(unittest.TestCase):
    def test_response_bytes_are_preserved_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = SnapshotWriter(Path(temp_dir), "snapshot-ai-001")
            artifact = writer.write_response(
                make_request(),
                RAW_PAYLOAD,
                "application/json",
                NOW,
            )
            snapshot_path = writer.finalize(make_manifest(artifact))

            stored = snapshot_path / "raw" / "openalex" / "request-openalex-001.json"
            self.assertEqual(stored.read_bytes(), RAW_PAYLOAD)
            self.assertEqual(artifact.sha256, hashlib.sha256(RAW_PAYLOAD).hexdigest())
            self.assertEqual(artifact.size_bytes, len(RAW_PAYLOAD))

    def test_manifest_is_utf8_json_with_snapshot_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = SnapshotWriter(Path(temp_dir), "snapshot-ai-001")
            artifact = writer.write_response(
                make_request(),
                RAW_PAYLOAD,
                "application/json; charset=utf-8",
                NOW,
            )
            snapshot_path = writer.finalize(make_manifest(artifact))

            manifest_bytes = (snapshot_path / "manifest.json").read_bytes()
            manifest = json.loads(manifest_bytes.decode("utf-8"))
            self.assertTrue(manifest_bytes.endswith(b"\n"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["runs"][0]["returned_records"], 1)

    def test_existing_snapshot_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "snapshot-ai-001").mkdir()

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                SnapshotWriter(root, "snapshot-ai-001")

    def test_duplicate_response_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = SnapshotWriter(Path(temp_dir), "snapshot-ai-001")
            writer.write_response(make_request(), RAW_PAYLOAD, "application/json", NOW)

            with self.assertRaisesRegex(FileExistsError, "already stored"):
                writer.write_response(make_request(), RAW_PAYLOAD, "application/json", NOW)

    def test_untracked_raw_response_prevents_finalization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = SnapshotWriter(Path(temp_dir), "snapshot-ai-001")
            artifact = writer.write_response(
                make_request(),
                RAW_PAYLOAD,
                "application/json",
                NOW,
            )
            extra_request = ConnectorRequest(
                request_id="request-openalex-002",
                query_id="query-ai-001",
                connector_id=ConnectorId.OPENALEX,
                channel=RetrievalChannel.SEMANTIC,
                endpoint="https://api.openalex.org/works",
                parameters=(
                    QueryParameter("search.semantic", "artificial intelligence"),
                ),
            )
            writer.write_response(extra_request, b"{}", "application/json", NOW)

            with self.assertRaisesRegex(ValueError, "untracked"):
                writer.finalize(make_manifest(artifact))

            self.assertFalse((Path(temp_dir) / "snapshot-ai-001").exists())

    def test_changed_raw_response_prevents_finalization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            writer = SnapshotWriter(root, "snapshot-ai-001")
            artifact = writer.write_response(
                make_request(),
                RAW_PAYLOAD,
                "application/json",
                NOW,
            )
            staging_path = next(root.glob(".snapshot-ai-001-*.staging"))
            stored_path = (
                staging_path
                / "raw"
                / "openalex"
                / "request-openalex-001.json"
            )
            stored_path.write_bytes(b"changed")

            with self.assertRaisesRegex(ValueError, "size mismatch"):
                writer.finalize(make_manifest(artifact))

            self.assertFalse((Path(temp_dir) / "snapshot-ai-001").exists())

    def test_publish_staging_retries_transient_permission_errors(self) -> None:
        from unittest.mock import patch

        from nextwave.sources.snapshots import publish_staging

        real_replace = os.replace
        calls: list[int] = []

        def flaky_replace(source, target):
            calls.append(1)
            if len(calls) < 3:
                raise PermissionError(13, "transient OS lock")
            return real_replace(source, target)

        with tempfile.TemporaryDirectory() as temp_dir:
            staging = Path(temp_dir) / "staging"
            (staging / "inner").mkdir(parents=True)
            (staging / "inner" / "a.json").write_bytes(b"{}")
            final = Path(temp_dir) / "final"
            with patch(
                "nextwave.sources.snapshots.os.replace", side_effect=flaky_replace
            ):
                publish_staging(staging, final, pause_seconds=0)

            self.assertTrue((final / "inner" / "a.json").is_file())
            self.assertEqual(len(calls), 3)

    def test_publish_staging_rejects_other_os_errors_immediately(self) -> None:
        from unittest.mock import patch

        from nextwave.sources.snapshots import publish_staging

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "nextwave.sources.snapshots.os.replace",
                side_effect=OSError(28, "no space left"),
            ):
                with self.assertRaisesRegex(OSError, "no space left"):
                    publish_staging(
                        Path(temp_dir) / "staging",
                        Path(temp_dir) / "final",
                        pause_seconds=0,
                    )


if __name__ == "__main__":
    unittest.main()
