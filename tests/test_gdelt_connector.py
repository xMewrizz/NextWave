from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from nextwave.sources import (
    ConnectorStatus,
    GdeltConnector,
    HttpResponse,
    QueryPurpose,
    SnapshotWriter,
    SourceQuery,
    build_gdelt_request,
    build_gdelt_timeline_request,
    gdelt_request_url,
)

NOW = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)


def make_query(
    *,
    published_from: date = date(2026, 7, 1),
    published_until: date = date(2026, 9, 15),
    languages: tuple[str, ...] = ("en", "ru"),
) -> SourceQuery:
    return SourceQuery(
        query_id="query-ai-001",
        analysis_scope_id="scope-ai-001",
        purpose=QueryPurpose.DISCOVERY,
        raw_query="Технологии в ИИ",
        normalized_query="artificial intelligence",
        search_texts=("artificial intelligence",),
        published_from=published_from,
        published_until=published_until,
        cutoff_date=date(2026, 9, 15),
        languages=languages,
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


class FakeMonotonicClock:
    def __init__(self, *moments: float) -> None:
        self._moments = iter(moments)

    def __call__(self) -> float:
        return next(self._moments)


def fixed_clock():
    moments = iter((NOW, NOW + timedelta(milliseconds=125)))
    return lambda: next(moments)


class GdeltRequestTests(unittest.TestCase):
    def test_request_contains_phrase_languages_and_exact_window(self) -> None:
        request = build_gdelt_request(
            make_query(),
            search_text="artificial intelligence",
            max_records=200,
        )
        parameters = {item.name: item.value for item in request.parameters}

        self.assertEqual(parameters["startdatetime"], "20260701000000")
        self.assertEqual(parameters["enddatetime"], "20260915235959")
        self.assertEqual(parameters["maxrecords"], "200")
        self.assertEqual(parameters["mode"], "artlist")
        self.assertEqual(parameters["format"], "json")
        self.assertEqual(
            parameters["query"],
            '"artificial intelligence" (sourcelang:english OR sourcelang:russian)',
        )
        self.assertIn("query=%22artificial+intelligence%22", gdelt_request_url(request))

    def test_request_rejects_window_over_three_months(self) -> None:
        with self.assertRaisesRegex(ValueError, "three calendar months"):
            build_gdelt_request(
                make_query(
                    published_from=date(2026, 5, 1),
                    published_until=date(2026, 9, 15),
                ),
                search_text="artificial intelligence",
            )

    def test_request_rejects_unmapped_language(self) -> None:
        with self.assertRaisesRegex(ValueError, "not mapped"):
            build_gdelt_request(
                make_query(languages=("nl",)),
                search_text="artificial intelligence",
            )

    def test_timeline_request_is_separate_from_article_list(self) -> None:
        request = build_gdelt_timeline_request(
            make_query(),
            search_text="artificial intelligence",
        )
        parameters = {item.name: item.value for item in request.parameters}

        self.assertEqual(parameters["mode"], "timelinevolraw")
        self.assertNotIn("maxrecords", parameters)
        self.assertNotIn("sort", parameters)

    def test_timeline_request_allows_long_measurement_window(self) -> None:
        request = build_gdelt_timeline_request(
            make_query(
                published_from=date(2025, 9, 16),
                published_until=date(2026, 9, 15),
            ),
            search_text="artificial intelligence",
        )
        parameters = {item.name: item.value for item in request.parameters}

        self.assertEqual(parameters["startdatetime"], "20250916000000")
        self.assertEqual(parameters["enddatetime"], "20260915235959")


class GdeltConnectorTests(unittest.TestCase):
    def test_success_saves_exact_response_and_counts_articles(self) -> None:
        body = json.dumps({"articles": [{"url": "https://a.example"}]}).encode()
        transport = FakeTransport(
            HttpResponse(200, {"Content-Type": "application/json"}, body)
        )
        connector = GdeltConnector(transport=transport, clock=fixed_clock())

        with tempfile.TemporaryDirectory() as temp_dir:
            writer = SnapshotWriter(Path(temp_dir), "snapshot-ai-001")
            run = connector.run_window(
                make_query(),
                writer,
                search_text="artificial intelligence",
            )
            staging_path = next(Path(temp_dir).glob(".snapshot-ai-001-*.staging"))
            stored = staging_path.joinpath(*run.artifact.uri.split("/"))

            self.assertIs(run.status, ConnectorStatus.SUCCESS)
            self.assertEqual(run.returned_records, 1)
            self.assertEqual(stored.read_bytes(), body)

    def test_invalid_json_is_preserved_as_failed_run(self) -> None:
        transport = FakeTransport(
            HttpResponse(200, {"Content-Type": "text/plain"}, b"Please retry later")
        )
        connector = GdeltConnector(transport=transport, clock=fixed_clock())

        with tempfile.TemporaryDirectory() as temp_dir:
            run = connector.run_window(
                make_query(),
                SnapshotWriter(Path(temp_dir), "snapshot-ai-001"),
                search_text="artificial intelligence",
            )

        self.assertIs(run.status, ConnectorStatus.FAILED)
        self.assertEqual(run.error.code, "invalid_response")
        self.assertIsNotNone(run.artifact)
        self.assertIsNone(run.returned_records)

    def test_repeated_calls_respect_five_second_interval(self) -> None:
        body = b'{"articles":[]}'
        transport = FakeTransport(
            HttpResponse(200, {"Content-Type": "application/json"}, body)
        )
        waits: list[float] = []
        clock_moments = iter(
            (
                NOW,
                NOW + timedelta(milliseconds=10),
                NOW + timedelta(seconds=2),
                NOW + timedelta(seconds=2, milliseconds=10),
            )
        )
        connector = GdeltConnector(
            transport=transport,
            clock=lambda: next(clock_moments),
            monotonic_clock=FakeMonotonicClock(100.0, 102.0, 105.0),
            sleeper=waits.append,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            connector.run_window(
                make_query(),
                SnapshotWriter(Path(temp_dir), "snapshot-ai-001"),
                search_text="artificial intelligence",
            )
            connector.run_window(
                make_query(),
                SnapshotWriter(Path(temp_dir), "snapshot-ai-002"),
                search_text="artificial intelligence",
            )

        self.assertEqual(waits, [3.0])

    def test_timeline_success_counts_intervals_instead_of_articles(self) -> None:
        body = json.dumps(
            {
                "timeline": [
                    {
                        "series": "Volume Intensity",
                        "data": [
                            {"date": "20260914T000000Z", "value": 3, "norm": 1000},
                            {"date": "20260915T000000Z", "value": 5, "norm": 1200},
                        ],
                    }
                ]
            }
        ).encode()
        connector = GdeltConnector(
            transport=FakeTransport(
                HttpResponse(200, {"Content-Type": "application/json"}, body)
            ),
            clock=fixed_clock(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            run = connector.run_timeline(
                make_query(),
                SnapshotWriter(Path(temp_dir), "snapshot-ai-001"),
                search_text="artificial intelligence",
            )

        self.assertIs(run.status, ConnectorStatus.SUCCESS)
        self.assertEqual(run.returned_records, 2)


if __name__ == "__main__":
    unittest.main()
