from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from nextwave.discovery import (
    DiscoveryBudget,
    MediaDiscoveryExecutor,
    MediaFallbackReason,
    ScopeGranularity,
    build_analysis_scope,
    build_discovery_plan,
    build_media_search_schedule,
    parse_mediacloud_collection_ids,
)
from nextwave.sources import (
    ConnectorId,
    HttpResponse,
    NewsContentStatus,
    NewsDocumentEnrichment,
    SnapshotStatus,
)

NOW = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)


def plan():
    scope = build_analysis_scope(
        raw_query="Технологии в ИИ",
        normalized_query="artificial intelligence",
        search_texts=("Технологии в ИИ", "AI"),
        languages=("en", "ru"),
        granularity=ScopeGranularity.DIRECTION,
        subfield_ids=("1702",),
    )
    budgets = (
        DiscoveryBudget(ConnectorId.OPENALEX, True, 1, 1, 10, 10, 20),
        DiscoveryBudget(ConnectorId.MEDIACLOUD, False, 2, 2, 10, 10, 20),
        DiscoveryBudget(ConnectorId.GDELT, False, 1, 1, 10, 10, 20),
    )
    return build_discovery_plan(
        analysis_id="analysis-ai-001",
        scope=scope,
        published_from=date(2025, 9, 21),
        cutoff_date=date(2026, 9, 21),
        budgets=budgets,
    )


def mediacloud_story(number: int, *, language: str = "en") -> dict[str, object]:
    return {
        "id": f"story-{number}",
        "url": f"https://news.example/story-{number}",
        "title": f"New photonic inference accelerator {number}",
        "publish_date": "2026-09-20",
        "indexed_date": "2026-09-20T10:00:00Z",
        "language": language,
        "media_name": "Technology News",
    }


def mediacloud_response(*stories: dict[str, object], status: int = 200) -> HttpResponse:
    payload = {"stories": list(stories), "pagination_token": None}
    return HttpResponse(
        status,
        {"Content-Type": "application/json"},
        json.dumps(payload).encode(),
    )


def gdelt_response() -> HttpResponse:
    payload = {
        "articles": [
            {
                "url": "https://fallback.example/optical-chip",
                "title": "Laboratory demonstrates optical AI chip",
                "seendate": "20260920T120000Z",
                "domain": "fallback.example",
                "language": "English",
            }
        ]
    }
    return HttpResponse(
        200,
        {"Content-Type": "application/json"},
        json.dumps(payload).encode(),
    )


class SequenceTransport:
    def __init__(self, *responses: HttpResponse) -> None:
        self._responses = iter(responses)
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HttpResponse:
        self.calls.append(url)
        return next(self._responses)


class StubNewsEnricher:
    def enrich_many(self, documents, *, max_concurrency=6):
        return tuple(
            NewsDocumentEnrichment(
                document=replace(
                    document,
                    excerpt="The article describes a concrete prototype and pilot deployment.",
                ),
                status=NewsContentStatus.ARTICLE_TEXT,
                issue_code=None,
                message=None,
                content_sha256="a" * 64,
                fetched_bytes=100,
                excerpt_source="test.article",
                excerpt_truncated=False,
            )
            for document in documents
        )


class MediaSearchScheduleTests(unittest.TestCase):
    def test_separates_english_and_russian_retrieval(self) -> None:
        schedule = build_media_search_schedule(plan().query)

        self.assertEqual(
            [(step.search_text, step.languages) for step in schedule],
            [
                ("artificial intelligence", ("en",)),
                ("Технологии в ИИ", ("ru",)),
            ],
        )


class MediaDiscoveryExecutorTests(unittest.TestCase):
    def test_uses_mediacloud_without_calling_fallback(self) -> None:
        primary = SequenceTransport(
            mediacloud_response(mediacloud_story(1)),
            mediacloud_response(mediacloud_story(2, language="ru")),
        )
        fallback = SequenceTransport(gdelt_response())

        with tempfile.TemporaryDirectory() as directory:
            result = MediaDiscoveryExecutor(
                Path(directory),
                mediacloud_api_key="temporary-key",
                mediacloud_transport=primary,
                gdelt_transport=fallback,
                news_enricher=StubNewsEnricher(),
                clock=lambda: NOW,
                monotonic=lambda: 0.0,
                mediacloud_min_interval_seconds=0,
            ).execute(plan())

            manifest = json.loads(
                (result.snapshot_path / "manifest.json").read_text(encoding="utf-8")
            )

        self.assertIs(result.provider_used, ConnectorId.MEDIACLOUD)
        self.assertIs(result.fallback_reason, MediaFallbackReason.NOT_NEEDED)
        self.assertIs(result.status, SnapshotStatus.COMPLETE)
        self.assertEqual(len(result.documents), 2)
        self.assertEqual(len(primary.calls), 2)
        self.assertEqual(fallback.calls, [])
        first_query = parse_qs(urlparse(primary.calls[0]).query)
        second_query = parse_qs(urlparse(primary.calls[1]).query)
        self.assertNotIn("cs", first_query)
        self.assertIn("language:en", first_query["q"][0])
        self.assertIn("language:ru", second_query["q"][0])
        self.assertEqual(manifest["snapshot_version"], "media-discovery-v1")
        json.dumps(result.to_dict(), ensure_ascii=False)

    def test_failed_mediacloud_uses_recent_gdelt_window(self) -> None:
        primary = SequenceTransport(
            HttpResponse(429, {"Content-Type": "application/json"}, b'{"error":"limit"}'),
            HttpResponse(429, {"Content-Type": "application/json"}, b'{"error":"limit"}'),
        )
        fallback = SequenceTransport(gdelt_response())

        with tempfile.TemporaryDirectory() as directory:
            result = MediaDiscoveryExecutor(
                Path(directory),
                mediacloud_api_key="temporary-key",
                mediacloud_transport=primary,
                gdelt_transport=fallback,
                news_enricher=StubNewsEnricher(),
                clock=lambda: NOW,
                monotonic=lambda: 0.0,
                mediacloud_min_interval_seconds=0,
                gdelt_min_interval_seconds=0,
            ).execute(plan())

        self.assertIs(result.provider_used, ConnectorId.GDELT)
        self.assertIs(result.fallback_reason, MediaFallbackReason.PRIMARY_FAILED)
        self.assertIs(result.status, SnapshotStatus.PARTIAL)
        self.assertEqual(len(result.documents), 1)
        parameters = parse_qs(urlparse(fallback.calls[0]).query)
        self.assertEqual(parameters["startdatetime"], ["20260621000000"])
        self.assertEqual(parameters["enddatetime"], ["20260921235959"])

    def test_successful_empty_primary_is_known_zero_without_fallback(self) -> None:
        primary = SequenceTransport(mediacloud_response(), mediacloud_response())
        fallback = SequenceTransport(gdelt_response())

        with tempfile.TemporaryDirectory() as directory:
            result = MediaDiscoveryExecutor(
                Path(directory),
                mediacloud_api_key="temporary-key",
                mediacloud_transport=primary,
                gdelt_transport=fallback,
                news_enricher=StubNewsEnricher(),
                clock=lambda: NOW,
                monotonic=lambda: 0.0,
                mediacloud_min_interval_seconds=0,
            ).execute(plan())

        self.assertIs(result.provider_used, ConnectorId.MEDIACLOUD)
        self.assertEqual(result.documents, ())
        self.assertEqual(result.enrichment, ())
        self.assertEqual(fallback.calls, [])

    def test_missing_primary_key_uses_gdelt_without_ui_configuration(self) -> None:
        fallback = SequenceTransport(gdelt_response())

        with tempfile.TemporaryDirectory() as directory:
            result = MediaDiscoveryExecutor(
                Path(directory),
                gdelt_transport=fallback,
                news_enricher=StubNewsEnricher(),
                clock=lambda: NOW,
                monotonic=lambda: 0.0,
                gdelt_min_interval_seconds=0,
            ).execute(plan())

        self.assertIs(result.provider_used, ConnectorId.GDELT)
        self.assertIs(
            result.fallback_reason,
            MediaFallbackReason.PRIMARY_UNCONFIGURED,
        )

    def test_collection_ids_default_to_documented_pair_when_unconfigured(self) -> None:
        # Живой прогон 22.09.2026: story-list без ss/cs всегда отвечает 422,
        # поэтому пустая конфигурация даёт дефолтный охват, а не пустой кортеж.
        self.assertEqual(
            parse_mediacloud_collection_ids({}),
            (34412234, 34412118),
        )
        self.assertEqual(
            parse_mediacloud_collection_ids(
                {"NEXTWAVE_MEDIACLOUD_COLLECTION_IDS": "34412234, 34412118"}
            ),
            (34412234, 34412118),
        )
        with self.assertRaisesRegex(ValueError, "comma-separated integers"):
            parse_mediacloud_collection_ids(
                {"NEXTWAVE_MEDIACLOUD_COLLECTION_IDS": "global"}
            )


if __name__ == "__main__":
    unittest.main()
