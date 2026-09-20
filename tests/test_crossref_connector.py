from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from nextwave.sources import (
    ConnectorStatus,
    CrossrefConnector,
    HttpResponse,
    QueryPurpose,
    RetrievalChannel,
    SnapshotWriter,
    SourceQuery,
    build_crossref_request,
    crossref_request_url,
)

NOW = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)


def make_query() -> SourceQuery:
    return SourceQuery(
        query_id="query-ai-001",
        analysis_scope_id="scope-ai-001",
        purpose=QueryPurpose.DISCOVERY,
        raw_query="Технологии в ИИ",
        normalized_query="artificial intelligence",
        search_texts=("artificial intelligence", "machine learning"),
        published_from=date(2025, 9, 16),
        published_until=date(2026, 9, 15),
        cutoff_date=date(2026, 9, 15),
    )


class FakeTransport:
    def __init__(self, response: HttpResponse | Exception) -> None:
        self.response = response
        self.calls: list[tuple[str, Mapping[str, str], float]] = []

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HttpResponse:
        self.calls.append((url, headers, timeout_seconds))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def fixed_clock():
    moments = iter((NOW, NOW + timedelta(milliseconds=125)))
    return lambda: next(moments)


class CrossrefRequestTests(unittest.TestCase):
    def test_text_request_contains_bibliographic_query_dates_and_cursor(self) -> None:
        request = build_crossref_request(
            make_query(),
            channel=RetrievalChannel.TEXT,
            search_text="artificial intelligence",
            rows=250,
            contact_email="team@example.org",
        )
        parameters = {item.name: item.value for item in request.parameters}

        self.assertEqual(parameters["query.bibliographic"], "artificial intelligence")
        self.assertEqual(parameters["cursor"], "*")
        self.assertEqual(parameters["rows"], "250")
        self.assertEqual(parameters["mailto"], "team@example.org")
        self.assertIn("from-pub-date:2025-09-16", parameters["filter"])
        self.assertIn("until-pub-date:2026-09-15", parameters["filter"])

    def test_identifier_request_normalizes_doi(self) -> None:
        request = build_crossref_request(
            make_query(),
            channel=RetrievalChannel.IDENTIFIER,
            doi="https://doi.org/10.1234/EXAMPLE.1",
        )
        parameters = {item.name: item.value for item in request.parameters}

        self.assertEqual(parameters["filter"], "doi:10.1234/example.1")
        self.assertEqual(parameters["rows"], "1")
        self.assertNotIn("query.bibliographic", parameters)

    def test_crossref_rejects_unsupported_semantic_channel(self) -> None:
        with self.assertRaisesRegex(ValueError, "text and identifier"):
            build_crossref_request(
                make_query(),
                channel=RetrievalChannel.SEMANTIC,
                search_text="artificial intelligence",
            )

    def test_request_id_changes_when_retry_attempt_changes(self) -> None:
        first = build_crossref_request(
            make_query(),
            channel=RetrievalChannel.TEXT,
            search_text="artificial intelligence",
            attempt=1,
        )
        second = build_crossref_request(
            make_query(),
            channel=RetrievalChannel.TEXT,
            search_text="artificial intelligence",
            attempt=2,
        )

        self.assertNotEqual(first.request_id, second.request_id)
        self.assertNotEqual(crossref_request_url(first), "")


class CrossrefConnectorTests(unittest.TestCase):
    def test_success_saves_exact_response_and_counts_items(self) -> None:
        body = json.dumps({"message": {"items": [{"DOI": "10.1/a"}, {"DOI": "10.1/b"}]}}).encode()
        transport = FakeTransport(
            HttpResponse(200, {"Content-Type": "application/json"}, body)
        )
        connector = CrossrefConnector(transport=transport, clock=fixed_clock())

        with tempfile.TemporaryDirectory() as temp_dir:
            writer = SnapshotWriter(Path(temp_dir), "snapshot-ai-001")
            run = connector.run_page(
                make_query(),
                writer,
                channel=RetrievalChannel.TEXT,
                search_text="artificial intelligence",
            )
            staging_path = next(Path(temp_dir).glob(".snapshot-ai-001-*.staging"))
            stored = staging_path.joinpath(*run.artifact.uri.split("/"))

            self.assertIs(run.status, ConnectorStatus.SUCCESS)
            self.assertEqual(run.returned_records, 2)
            self.assertEqual(stored.read_bytes(), body)

    def test_api_token_is_sent_only_in_header(self) -> None:
        body = b'{"message":{"items":[]}}'
        transport = FakeTransport(
            HttpResponse(200, {"Content-Type": "application/json"}, body)
        )
        connector = CrossrefConnector(
            transport=transport,
            api_token="secret-token",
            clock=fixed_clock(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            run = connector.run_page(
                make_query(),
                SnapshotWriter(Path(temp_dir), "snapshot-ai-001"),
                channel=RetrievalChannel.IDENTIFIER,
                doi="10.1234/example.1",
            )

        url, headers, _ = transport.calls[0]
        self.assertEqual(headers["Crossref-Plus-API-Token"], "Bearer secret-token")
        self.assertNotIn("secret-token", url)
        self.assertNotIn("secret-token", str(run.request.to_dict()))

    def test_http_error_preserves_body_and_marks_retryable(self) -> None:
        body = b'{"status":"error"}'
        transport = FakeTransport(
            HttpResponse(503, {"Content-Type": "application/json"}, body)
        )
        connector = CrossrefConnector(transport=transport, clock=fixed_clock())

        with tempfile.TemporaryDirectory() as temp_dir:
            run = connector.run_page(
                make_query(),
                SnapshotWriter(Path(temp_dir), "snapshot-ai-001"),
                channel=RetrievalChannel.TEXT,
                search_text="artificial intelligence",
            )

        self.assertIs(run.status, ConnectorStatus.FAILED)
        self.assertEqual(run.http_status, 503)
        self.assertIsNotNone(run.artifact)
        self.assertIsNone(run.returned_records)
        self.assertTrue(run.error.retryable)


if __name__ == "__main__":
    unittest.main()
