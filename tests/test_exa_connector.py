from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from nextwave.sources import (
    ConnectorStatus,
    ExaConnector,
    HttpResponse,
    QueryPurpose,
    SnapshotWriter,
    SourceQuery,
    build_exa_news_request,
    exa_request_body,
)

NOW = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)


def make_query() -> SourceQuery:
    return SourceQuery(
        query_id="query-exa-001",
        analysis_scope_id="scope-exa-001",
        purpose=QueryPurpose.HISTORICAL_ENRICHMENT,
        raw_query="Инфраструктура ИИ",
        normalized_query="ai infrastructure",
        search_texts=("AI inference accelerator",),
        published_from=date(2025, 9, 15),
        published_until=date(2026, 9, 15),
        cutoff_date=date(2026, 9, 15),
        languages=("en", "ru"),
    )


class FakeTransport:
    def __init__(self, response: HttpResponse | Exception) -> None:
        self.response = response
        self.calls: list[tuple[str, Mapping[str, str], bytes, float]] = []

    def post_json(self, url, *, headers, body, timeout_seconds):
        self.calls.append((url, headers, body, timeout_seconds))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def fixed_clock():
    moments = iter((NOW, NOW + timedelta(milliseconds=50)))
    return lambda: next(moments)


class ExaRequestTests(unittest.TestCase):
    def test_body_contains_news_dates_and_highlights(self) -> None:
        request = build_exa_news_request(
            make_query(), search_text="AI inference accelerator", num_results=12
        )
        body = json.loads(exa_request_body(request))
        self.assertEqual(body["category"], "news")
        self.assertEqual(body["numResults"], 12)
        self.assertEqual(body["startPublishedDate"], "2025-09-15T00:00:00.000Z")
        self.assertEqual(body["endPublishedDate"], "2026-09-15T23:59:59.999Z")
        self.assertEqual(body["contents"], {"highlights": True})

    def test_request_identity_changes_with_window(self) -> None:
        first = build_exa_news_request(make_query(), search_text="AI inference accelerator")
        other = make_query().to_dict()
        other["published_from"] = "2024-09-15"
        second_query = SourceQuery(
            query_id="query-exa-001",
            analysis_scope_id="scope-exa-001",
            purpose=QueryPurpose.HISTORICAL_ENRICHMENT,
            raw_query="Инфраструктура ИИ",
            normalized_query="ai infrastructure",
            search_texts=("AI inference accelerator",),
            published_from=date(2024, 9, 15),
            published_until=date(2026, 9, 15),
            cutoff_date=date(2026, 9, 15),
            languages=("en", "ru"),
        )
        second = build_exa_news_request(second_query, search_text="AI inference accelerator")
        self.assertNotEqual(first.request_id, second.request_id)


class ExaConnectorTests(unittest.TestCase):
    def test_success_saves_raw_response_and_hides_key(self) -> None:
        body = b'{"results":[{"id":"r1"}],"requestId":"remote"}'
        transport = FakeTransport(HttpResponse(200, {"Content-Type": "application/json"}, body))
        connector = ExaConnector(
            api_key="exa-secret", transport=transport, clock=fixed_clock()
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = SnapshotWriter(Path(temp_dir), "snapshot-exa-001")
            run = connector.run_search(
                make_query(), writer, search_text="AI inference accelerator"
            )
            stored = next(Path(temp_dir).glob(".snapshot-exa-001-*.staging"))
            self.assertEqual(stored.joinpath(*run.artifact.uri.split("/")).read_bytes(), body)
        url, headers, sent_body, _ = transport.calls[0]
        self.assertIs(run.status, ConnectorStatus.SUCCESS)
        self.assertEqual(run.returned_records, 1)
        self.assertEqual(headers["x-api-key"], "exa-secret")
        self.assertNotIn(b"exa-secret", sent_body)
        self.assertNotIn("exa-secret", url)
        self.assertNotIn("exa-secret", str(run.to_dict()))

    def test_network_exception_does_not_leak_message(self) -> None:
        connector = ExaConnector(
            api_key="exa-secret",
            transport=FakeTransport(RuntimeError("x-api-key: exa-secret")),
            clock=fixed_clock(),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            run = connector.run_search(
                make_query(),
                SnapshotWriter(Path(temp_dir), "snapshot-exa-001"),
                search_text="AI inference accelerator",
            )
        self.assertIs(run.status, ConnectorStatus.FAILED)
        self.assertTrue(run.error.retryable)
        self.assertNotIn("exa-secret", run.error.message)

    def test_429_is_retryable_and_keeps_retry_after(self) -> None:
        connector = ExaConnector(
            api_key="key",
            transport=FakeTransport(HttpResponse(429, {"Retry-After": "3"}, b"{}")),
            clock=fixed_clock(),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            run = connector.run_search(
                make_query(),
                SnapshotWriter(Path(temp_dir), "snapshot-exa-001"),
                search_text="AI inference accelerator",
            )
        self.assertTrue(run.error.retryable)
        self.assertEqual(run.error.retry_after_seconds, 3.0)


if __name__ == "__main__":
    unittest.main()
