from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from nextwave.sources import (
    ConnectorStatus,
    HttpResponse,
    OpenAlexConnector,
    RetrievalChannel,
    SnapshotManifest,
    SnapshotWriter,
    SourceQuery,
    build_openalex_request,
    openalex_request_url,
)

NOW = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)


def make_query(
    *,
    topic_ids: tuple[str, ...] = ("T10001",),
    subfield_ids: tuple[str, ...] = (),
) -> SourceQuery:
    from nextwave.sources import QueryPurpose

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
        languages=("en", "ru"),
        topic_ids=topic_ids,
        subfield_ids=subfield_ids,
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


class OpenAlexRequestTests(unittest.TestCase):
    def test_text_request_contains_query_scope_and_dates(self) -> None:
        request = build_openalex_request(
            make_query(),
            channel=RetrievalChannel.TEXT,
            search_text="artificial intelligence",
            per_page=25,
        )
        parameters = {item.name: item.value for item in request.parameters}

        self.assertEqual(parameters["search"], "artificial intelligence")
        self.assertEqual(parameters["per_page"], "25")
        self.assertIn("from_publication_date:2025-09-16", parameters["filter"])
        self.assertIn("to_publication_date:2026-09-15", parameters["filter"])
        self.assertIn("language:en|ru", parameters["filter"])

    def test_semantic_request_uses_semantic_parameter(self) -> None:
        request = build_openalex_request(
            make_query(),
            channel=RetrievalChannel.SEMANTIC,
            search_text="machine learning",
        )
        parameters = {item.name: item.value for item in request.parameters}

        self.assertNotIn("search", parameters)
        self.assertEqual(parameters["search.semantic"], "machine learning")

    def test_taxonomy_request_uses_topic_ids_without_search_text(self) -> None:
        request = build_openalex_request(
            make_query(topic_ids=("T10001", "T10002")),
            channel=RetrievalChannel.TAXONOMY,
        )
        parameters = {item.name: item.value for item in request.parameters}

        self.assertNotIn("search", parameters)
        self.assertNotIn("search.semantic", parameters)
        self.assertIn("topics.id:T10001|T10002", parameters["filter"])

    def test_taxonomy_request_uses_subfield_for_broad_scope(self) -> None:
        request = build_openalex_request(
            make_query(topic_ids=(), subfield_ids=("1702",)),
            channel=RetrievalChannel.TAXONOMY,
        )
        parameters = {item.name: item.value for item in request.parameters}

        self.assertIn("topics.subfield.id:1702", parameters["filter"])

    def test_request_url_is_stable(self) -> None:
        request = build_openalex_request(
            make_query(),
            channel=RetrievalChannel.TEXT,
            search_text="artificial intelligence",
        )

        self.assertEqual(openalex_request_url(request), openalex_request_url(request))
        self.assertIn("search=artificial+intelligence", openalex_request_url(request))

    def test_request_id_covers_page_size_and_retry_attempt(self) -> None:
        first = build_openalex_request(
            make_query(),
            channel=RetrievalChannel.TEXT,
            search_text="artificial intelligence",
            per_page=25,
            attempt=1,
        )
        changed_size = build_openalex_request(
            make_query(),
            channel=RetrievalChannel.TEXT,
            search_text="artificial intelligence",
            per_page=50,
            attempt=1,
        )
        retry = build_openalex_request(
            make_query(),
            channel=RetrievalChannel.TEXT,
            search_text="artificial intelligence",
            per_page=25,
            attempt=2,
        )

        self.assertNotEqual(first.request_id, changed_size.request_id)
        self.assertNotEqual(first.request_id, retry.request_id)


class OpenAlexConnectorTests(unittest.TestCase):
    def test_success_saves_exact_response_and_counts_page_records(self) -> None:
        body = json.dumps({"meta": {"count": 2}, "results": [{"id": "W1"}, {"id": "W2"}]}).encode()
        transport = FakeTransport(
            HttpResponse(200, {"Content-Type": "application/json"}, body)
        )
        connector = OpenAlexConnector(transport=transport, clock=fixed_clock())

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

    def test_contact_email_is_sent_as_mailto_param(self) -> None:
        body = b'{"meta":{"count":0},"results":[]}'
        transport = FakeTransport(
            HttpResponse(200, {"Content-Type": "application/json"}, body)
        )
        connector = OpenAlexConnector(
            transport=transport,
            contact_email="team@example.com",
            clock=fixed_clock(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            run = connector.run_page(
                make_query(),
                SnapshotWriter(Path(temp_dir), "snapshot-ai-001"),
                channel=RetrievalChannel.TEXT,
                search_text="artificial intelligence",
            )

        url, headers, _ = transport.calls[0]
        self.assertNotIn("Authorization", headers)
        self.assertIn("mailto=team%40example.com", url)
        self.assertIn("mailto", str(run.request.to_dict()))

    def test_retry_after_header_is_recorded_on_rate_limit(self) -> None:
        body = b'{"error":"Rate limit exceeded, retry in 39s"}'
        transport = FakeTransport(
            HttpResponse(
                429,
                {"Content-Type": "application/json", "Retry-After": "39"},
                body,
            )
        )
        connector = OpenAlexConnector(transport=transport, clock=fixed_clock())

        with tempfile.TemporaryDirectory() as temp_dir:
            run = connector.run_page(
                make_query(),
                SnapshotWriter(Path(temp_dir), "snapshot-ai-001"),
                channel=RetrievalChannel.TEXT,
                search_text="artificial intelligence",
            )

        self.assertEqual(run.error.code, "http_429")
        self.assertEqual(run.error.retry_after_seconds, 39.0)

    def test_garbage_retry_after_header_is_ignored(self) -> None:
        body = b'{"error":"Rate limit exceeded"}'
        transport = FakeTransport(
            HttpResponse(
                429,
                {"Content-Type": "application/json", "Retry-After": "soon"},
                body,
            )
        )
        connector = OpenAlexConnector(transport=transport, clock=fixed_clock())

        with tempfile.TemporaryDirectory() as temp_dir:
            run = connector.run_page(
                make_query(),
                SnapshotWriter(Path(temp_dir), "snapshot-ai-001"),
                channel=RetrievalChannel.TEXT,
                search_text="artificial intelligence",
            )

        self.assertIsNone(run.error.retry_after_seconds)

    def test_non_finite_retry_after_header_is_ignored(self) -> None:
        transport = FakeTransport(
            HttpResponse(
                429,
                {"Content-Type": "application/json", "Retry-After": "NaN"},
                b'{"error":"Rate limit exceeded"}',
            )
        )
        connector = OpenAlexConnector(transport=transport, clock=fixed_clock())

        with tempfile.TemporaryDirectory() as temp_dir:
            run = connector.run_page(
                make_query(),
                SnapshotWriter(Path(temp_dir), "snapshot-ai-001"),
                channel=RetrievalChannel.TEXT,
                search_text="artificial intelligence",
            )

        self.assertIsNone(run.error.retry_after_seconds)
        self.assertTrue(run.error.retryable)

    def test_excessive_retry_after_remains_retryable_later(self) -> None:
        body = b'{"error":"Rate limit exceeded, retry in 9612s"}'
        transport = FakeTransport(
            HttpResponse(
                429,
                {"Content-Type": "application/json", "Retry-After": "9612"},
                body,
            )
        )
        connector = OpenAlexConnector(transport=transport, clock=fixed_clock())

        with tempfile.TemporaryDirectory() as temp_dir:
            run = connector.run_page(
                make_query(),
                SnapshotWriter(Path(temp_dir), "snapshot-ai-001"),
                channel=RetrievalChannel.TEXT,
                search_text="artificial intelligence",
            )

        self.assertEqual(run.error.retry_after_seconds, 9612.0)
        self.assertTrue(run.error.retryable)

    def test_boundary_retry_after_stays_retryable(self) -> None:
        body = b'{"error":"Rate limit exceeded, retry in 120s"}'
        transport = FakeTransport(
            HttpResponse(
                429,
                {"Content-Type": "application/json", "Retry-After": "120"},
                body,
            )
        )
        connector = OpenAlexConnector(transport=transport, clock=fixed_clock())

        with tempfile.TemporaryDirectory() as temp_dir:
            run = connector.run_page(
                make_query(),
                SnapshotWriter(Path(temp_dir), "snapshot-ai-001"),
                channel=RetrievalChannel.TEXT,
                search_text="artificial intelligence",
            )

        self.assertTrue(run.error.retryable)

    def test_http_error_body_is_saved_and_coverage_is_unknown(self) -> None:
        body = b'{"error":"rate limit"}'
        transport = FakeTransport(
            HttpResponse(429, {"Content-Type": "application/json"}, body)
        )
        connector = OpenAlexConnector(transport=transport, clock=fixed_clock())

        with tempfile.TemporaryDirectory() as temp_dir:
            writer = SnapshotWriter(Path(temp_dir), "snapshot-ai-001")
            run = connector.run_page(
                make_query(),
                writer,
                channel=RetrievalChannel.TEXT,
                search_text="artificial intelligence",
            )

        self.assertIs(run.status, ConnectorStatus.FAILED)
        self.assertEqual(run.http_status, 429)
        self.assertIsNone(run.returned_records)
        self.assertIsNotNone(run.artifact)
        self.assertTrue(run.error.retryable)

    def test_network_error_returns_failure_without_artifact(self) -> None:
        connector = OpenAlexConnector(
            transport=FakeTransport(TimeoutError("deadline exceeded")),
            clock=fixed_clock(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            run = connector.run_page(
                make_query(),
                SnapshotWriter(Path(temp_dir), "snapshot-ai-001"),
                channel=RetrievalChannel.TEXT,
                search_text="artificial intelligence",
            )

        self.assertIs(run.status, ConnectorStatus.FAILED)
        self.assertIsNone(run.http_status)
        self.assertIsNone(run.returned_records)
        self.assertIsNone(run.artifact)
        self.assertEqual(run.error.code, "network_error")

    def test_invalid_json_is_saved_but_not_counted_as_coverage(self) -> None:
        transport = FakeTransport(
            HttpResponse(200, {"Content-Type": "application/json"}, b"not-json")
        )
        connector = OpenAlexConnector(transport=transport, clock=fixed_clock())

        with tempfile.TemporaryDirectory() as temp_dir:
            run = connector.run_page(
                make_query(),
                SnapshotWriter(Path(temp_dir), "snapshot-ai-001"),
                channel=RetrievalChannel.TEXT,
                search_text="artificial intelligence",
            )

        self.assertIs(run.status, ConnectorStatus.FAILED)
        self.assertIsNotNone(run.artifact)
        self.assertIsNone(run.returned_records)
        self.assertEqual(run.error.code, "invalid_response")


class OpenAlexApiKeyTests(unittest.TestCase):
    SECRET = "openalex-free-key-1"

    def _run(self, response, **kwargs):
        transport = FakeTransport(response)
        connector = OpenAlexConnector(
            transport=transport, clock=fixed_clock(), **kwargs
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            run = connector.run_page(
                make_query(),
                SnapshotWriter(Path(temp_dir), "snapshot-ai-001"),
                channel=RetrievalChannel.TEXT,
                search_text="artificial intelligence",
            )
        return transport, run

    @staticmethod
    def _ok_response():
        return HttpResponse(
            200, {"Content-Type": "application/json"}, b'{"results":[]}'
        )

    def test_anonymous_connector_sends_no_authorization(self) -> None:
        transport, _ = self._run(self._ok_response())

        _, headers, _ = transport.calls[0]
        self.assertNotIn("Authorization", headers)

    def test_api_key_sends_exact_bearer_header(self) -> None:
        transport, _ = self._run(self._ok_response(), api_key=self.SECRET)

        _, headers, _ = transport.calls[0]
        self.assertEqual(headers["Authorization"], f"Bearer {self.SECRET}")

    def test_blank_api_key_keeps_anonymous_behavior(self) -> None:
        for blank in (None, "", "   "):
            transport, _ = self._run(self._ok_response(), api_key=blank)

            _, headers, _ = transport.calls[0]
            self.assertNotIn("Authorization", headers)

    def test_api_key_is_stripped_before_use(self) -> None:
        transport, _ = self._run(
            self._ok_response(), api_key=f"  {self.SECRET}  "
        )

        _, headers, _ = transport.calls[0]
        self.assertEqual(headers["Authorization"], f"Bearer {self.SECRET}")

    def test_api_key_never_leaks_into_url_artifacts_or_errors(self) -> None:
        transport, run = self._run(
            self._ok_response(),
            api_key=self.SECRET,
            contact_email="team@example.com",
        )

        url, _, _ = transport.calls[0]
        self.assertIn("mailto=team%40example.com", url)
        self.assertNotIn(self.SECRET, url)
        self.assertNotIn(self.SECRET, json.dumps(run.request.to_dict()))
        self.assertNotIn(self.SECRET, json.dumps(run.to_dict()))

        manifest = SnapshotManifest(
            snapshot_id="snapshot-ai-001",
            snapshot_version="openalex-discovery-v1",
            analysis_id="analysis-ai-001",
            created_at=NOW,
            query=make_query(),
            runs=(run,),
        )
        self.assertNotIn(self.SECRET, json.dumps(manifest.to_dict()))

        _, failed = self._run(
            HttpResponse(500, {"Content-Type": "application/json"}, b"{}"),
            api_key=self.SECRET,
        )
        self.assertNotIn(self.SECRET, failed.error.message)

    def test_transport_exception_with_key_stays_out_of_error_and_manifest(self) -> None:
        transport, run = self._run(
            RuntimeError(f"Authorization: Bearer {self.SECRET}"),
            api_key=self.SECRET,
        )

        self.assertIs(run.status, ConnectorStatus.FAILED)
        self.assertEqual(run.error.code, "network_error")
        self.assertEqual(run.error.message, "RuntimeError: OpenAlex request failed")
        self.assertNotIn(self.SECRET, run.error.message)
        self.assertNotIn(self.SECRET, json.dumps(run.to_dict()))

        manifest = SnapshotManifest(
            snapshot_id="snapshot-ai-001",
            snapshot_version="openalex-discovery-v1",
            analysis_id="analysis-ai-001",
            created_at=NOW,
            query=make_query(),
            runs=(run,),
        )
        self.assertNotIn(self.SECRET, json.dumps(manifest.to_dict()))


if __name__ == "__main__":
    unittest.main()
