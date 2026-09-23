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
    MediaCloudConnector,
    QueryPurpose,
    SnapshotWriter,
    SourceQuery,
    build_mediacloud_story_request,
    build_mediacloud_timeline_request,
    mediacloud_request_url,
)

NOW = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
US_NATIONAL = 34412234


def make_query(
    search_texts: tuple[str, ...] = (
        "artificial intelligence",
        "speculative decoding",
    ),
) -> SourceQuery:
    return SourceQuery(
        query_id="query-ai-001",
        analysis_scope_id="scope-ai-001",
        purpose=QueryPurpose.DISCOVERY,
        raw_query="Технологии в ИИ",
        normalized_query="artificial intelligence",
        search_texts=search_texts,
        published_from=date(2025, 9, 16),
        published_until=date(2026, 9, 15),
        cutoff_date=date(2026, 9, 15),
        languages=("en", "ru"),
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


class MediaCloudRequestTests(unittest.TestCase):
    def test_story_request_contains_exact_scope_and_collections(self) -> None:
        request = build_mediacloud_story_request(
            make_query(),
            search_text="artificial intelligence",
            collection_ids=(US_NATIONAL, 34412232),
            page_size=50,
        )
        parameters = {item.name: item.value for item in request.parameters}

        self.assertEqual(parameters["start"], "2025-09-16")
        self.assertEqual(parameters["end"], "2026-09-15")
        self.assertEqual(parameters["cs"], "34412234,34412232")
        self.assertEqual(parameters["page_size"], "50")
        self.assertEqual(parameters["sort_order"], "desc")
        self.assertEqual(
            parameters["q"],
            '"artificial intelligence" AND (language:en OR language:ru)',
        )
        self.assertIn("story-list", mediacloud_request_url(request))

    def test_next_page_has_distinct_request_id(self) -> None:
        first = build_mediacloud_story_request(
            make_query(),
            search_text="artificial intelligence",
            collection_ids=(US_NATIONAL,),
        )
        second = build_mediacloud_story_request(
            make_query(),
            search_text="artificial intelligence",
            collection_ids=(US_NATIONAL,),
            pagination_token="opaque-next-token",
            page_index=2,
        )

        self.assertNotEqual(first.request_id, second.request_id)
        self.assertIn("pagination_token=opaque-next-token", mediacloud_request_url(second))

    def test_timeline_request_is_separate_from_story_list(self) -> None:
        request = build_mediacloud_timeline_request(
            make_query(),
            search_text="speculative decoding",
            collection_ids=(US_NATIONAL,),
        )
        parameters = {item.name: item.value for item in request.parameters}

        self.assertIn("count-over-time", request.endpoint)
        self.assertNotIn("page_size", parameters)
        self.assertNotIn("sort_order", parameters)

    def test_global_request_omits_collection_and_can_limit_language(self) -> None:
        request = build_mediacloud_story_request(
            make_query(),
            search_text="artificial intelligence",
            collection_ids=(),
            languages=("ru",),
        )
        parameters = {item.name: item.value for item in request.parameters}

        self.assertNotIn("cs", parameters)
        self.assertEqual(parameters["q"], '"artificial intelligence" AND language:ru')

    def test_long_phrase_uses_and_of_content_words(self) -> None:
        request = build_mediacloud_story_request(
            make_query(
                search_texts=(
                    "artificial intelligence",
                    "peripheral artificial intelligence",
                )
            ),
            search_text="peripheral artificial intelligence",
            collection_ids=(US_NATIONAL,),
            languages=("en",),
        )
        parameters = {item.name: item.value for item in request.parameters}

        self.assertEqual(
            parameters["q"],
            "peripheral AND artificial AND intelligence AND language:en",
        )

    def test_single_character_particles_are_dropped(self) -> None:
        request = build_mediacloud_story_request(
            make_query(
                search_texts=(
                    "artificial intelligence",
                    "Перспективные решения в финтехе",
                )
            ),
            search_text="Перспективные решения в финтехе",
            collection_ids=(US_NATIONAL,),
            languages=("ru",),
        )
        parameters = {item.name: item.value for item in request.parameters}

        self.assertEqual(
            parameters["q"],
            "Перспективные AND решения AND финтехе AND language:ru",
        )


class MediaCloudConnectorTests(unittest.TestCase):
    def test_success_saves_exact_story_response(self) -> None:
        body = json.dumps(
            {"stories": [{"id": "story-1"}], "pagination_token": "next"}
        ).encode()
        transport = FakeTransport(
            HttpResponse(200, {"Content-Type": "application/json"}, body)
        )
        connector = MediaCloudConnector(
            api_key="secret-token",
            transport=transport,
            min_interval_seconds=0,
            clock=fixed_clock(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            writer = SnapshotWriter(Path(temp_dir), "snapshot-ai-001")
            run = connector.run_page(
                make_query(),
                writer,
                search_text="artificial intelligence",
                collection_ids=(US_NATIONAL,),
            )
            staging_path = next(Path(temp_dir).glob(".snapshot-ai-001-*.staging"))
            stored = staging_path.joinpath(*run.artifact.uri.split("/"))

            self.assertIs(run.status, ConnectorStatus.SUCCESS)
            self.assertEqual(run.returned_records, 1)
            self.assertEqual(stored.read_bytes(), body)

    def test_api_key_is_sent_only_in_header(self) -> None:
        transport = FakeTransport(
            HttpResponse(
                200,
                {"Content-Type": "application/json"},
                b'{"stories":[],"pagination_token":null}',
            )
        )
        connector = MediaCloudConnector(
            api_key="secret-token",
            transport=transport,
            min_interval_seconds=0,
            clock=fixed_clock(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            run = connector.run_page(
                make_query(),
                SnapshotWriter(Path(temp_dir), "snapshot-ai-001"),
                search_text="artificial intelligence",
                collection_ids=(US_NATIONAL,),
            )

        url, headers, _ = transport.calls[0]
        self.assertEqual(headers["Authorization"], "Token secret-token")
        self.assertNotIn("secret-token", url)
        self.assertNotIn("secret-token", str(run.request.to_dict()))

    def test_timeline_success_counts_points(self) -> None:
        body = json.dumps(
            {
                "count_over_time": {
                    "counts": [
                        {"date": "2026-09-14", "total_count": 1000, "count": 3},
                        {"date": "2026-09-15", "total_count": 1200, "count": 5},
                    ]
                }
            }
        ).encode()
        connector = MediaCloudConnector(
            api_key="secret-token",
            transport=FakeTransport(
                HttpResponse(200, {"Content-Type": "application/json"}, body)
            ),
            min_interval_seconds=0,
            clock=fixed_clock(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            run = connector.run_timeline(
                make_query(),
                SnapshotWriter(Path(temp_dir), "snapshot-ai-001"),
                search_text="artificial intelligence",
                collection_ids=(US_NATIONAL,),
            )

        self.assertIs(run.status, ConnectorStatus.SUCCESS)
        self.assertEqual(run.returned_records, 2)

    def test_http_401_is_not_retryable_and_body_is_saved(self) -> None:
        connector = MediaCloudConnector(
            api_key="invalid-token",
            transport=FakeTransport(
                HttpResponse(401, {"Content-Type": "application/json"}, b'{"detail":"bad"}')
            ),
            min_interval_seconds=0,
            clock=fixed_clock(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            run = connector.run_page(
                make_query(),
                SnapshotWriter(Path(temp_dir), "snapshot-ai-001"),
                search_text="artificial intelligence",
                collection_ids=(US_NATIONAL,),
            )

        self.assertIs(run.status, ConnectorStatus.FAILED)
        self.assertEqual(run.error.code, "http_401")
        self.assertFalse(run.error.retryable)
        self.assertIsNotNone(run.artifact)

    def test_repeated_calls_respect_thirty_second_interval(self) -> None:
        body = b'{"stories":[],"pagination_token":null}'
        waits: list[float] = []
        clock_moments = iter(
            (
                NOW,
                NOW + timedelta(milliseconds=10),
                NOW + timedelta(seconds=5),
                NOW + timedelta(seconds=5, milliseconds=10),
            )
        )
        connector = MediaCloudConnector(
            api_key="secret-token",
            transport=FakeTransport(
                HttpResponse(200, {"Content-Type": "application/json"}, body)
            ),
            clock=lambda: next(clock_moments),
            monotonic_clock=FakeMonotonicClock(100.0, 105.0, 130.0),
            sleeper=waits.append,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            connector.run_page(
                make_query(),
                SnapshotWriter(Path(temp_dir), "snapshot-ai-001"),
                search_text="artificial intelligence",
                collection_ids=(US_NATIONAL,),
            )
            connector.run_page(
                make_query(),
                SnapshotWriter(Path(temp_dir), "snapshot-ai-002"),
                search_text="artificial intelligence",
                collection_ids=(US_NATIONAL,),
            )

        self.assertEqual(waits, [25.0])


if __name__ == "__main__":
    unittest.main()
