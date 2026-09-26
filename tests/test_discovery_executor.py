from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from nextwave.discovery import (
    DiscoveryBudget,
    DiscoveryStopReason,
    OpenAlexDiscoveryExecutor,
    ScopeGranularity,
    build_analysis_scope,
    build_discovery_plan,
    build_openalex_search_schedule,
)
from nextwave.sources import ConnectorId, HttpResponse, RetrievalChannel, SnapshotStatus

NOW = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)


def work(
    external_id: str,
    *,
    doi: str | None,
    publication_date: str = "2026-08-10",
    keywords: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "id": f"https://openalex.org/{external_id}",
        "doi": f"https://doi.org/{doi}" if doi else None,
        "title": f"Work {external_id}",
        "publication_date": publication_date,
        "language": "en",
        "type": "article",
        "authorships": [],
        "abstract_inverted_index": {"Weak": [0], "signal": [1]},
        "keywords": keywords or [],
        "primary_location": {
            "landing_page_url": f"https://example.org/{external_id}",
            "source": {"display_name": "Example Journal"},
        },
    }


def response(*records: dict[str, object], status: int = 200) -> HttpResponse:
    body = json.dumps({"meta": {"count": len(records)}, "results": records}).encode()
    return HttpResponse(status, {"Content-Type": "application/json"}, body)


class SequenceTransport:
    def __init__(self, responses: list[HttpResponse]) -> None:
        self.responses = iter(responses)
        self.calls: list[tuple[str, Mapping[str, str], float]] = []

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HttpResponse:
        self.calls.append((url, headers, timeout_seconds))
        return next(self.responses)


def plan(
    *,
    max_requests: int = 6,
    max_pages: int = 4,
    max_documents: int = 400,
):
    scope = build_analysis_scope(
        raw_query="Технологии в ИИ",
        normalized_query="artificial intelligence",
        search_texts=("Технологии в ИИ", "AI"),
        languages=("en", "ru"),
        granularity=ScopeGranularity.DIRECTION,
        subfield_ids=("1702",),
    )
    budget = DiscoveryBudget(
        connector_id=ConnectorId.OPENALEX,
        required=True,
        max_requests=max_requests,
        max_pages=max_pages,
        max_documents=max_documents,
        request_timeout_seconds=12,
        max_elapsed_seconds=30,
    )
    return build_discovery_plan(
        analysis_id="analysis-ai-001",
        scope=scope,
        published_from=date(2025, 9, 20),
        cutoff_date=date(2026, 9, 20),
        budgets=(budget,),
    )


class OpenAlexSearchScheduleTests(unittest.TestCase):
    def test_uses_complementary_channels_and_skips_cyrillic_alias(self) -> None:
        schedule = build_openalex_search_schedule(plan().query)

        self.assertEqual(
            [(step.channel, step.search_text) for step in schedule],
            [
                (RetrievalChannel.TEXT, "artificial intelligence"),
                (RetrievalChannel.SEMANTIC, "artificial intelligence"),
                (RetrievalChannel.TAXONOMY, None),
                (RetrievalChannel.TEXT, "AI"),
            ],
        )


class OpenAlexDiscoveryExecutorTests(unittest.TestCase):
    def test_merges_discovery_hints_when_channels_return_the_same_origin(self) -> None:
        transport = SequenceTransport(
            [
                response(
                    work(
                        "W1",
                        doi="10.1234/shared",
                        keywords=[
                            {
                                "id": "https://openalex.org/keywords/speculative-decoding",
                                "display_name": "Speculative Decoding",
                                "score": 0.7,
                            }
                        ],
                    )
                ),
                response(
                    work(
                        "W9",
                        doi="10.1234/shared",
                        keywords=[
                            {
                                "id": "https://openalex.org/keywords/speculative-decoding",
                                "display_name": "Speculative Decoding",
                                "score": 0.9,
                            },
                            {
                                "id": "https://openalex.org/keywords/draft-model",
                                "display_name": "Draft Model",
                                "score": 0.8,
                            },
                        ],
                    )
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            result = OpenAlexDiscoveryExecutor(
                Path(directory),
                transport=transport,
                clock=lambda: NOW,
                monotonic=lambda: 0.0,
            ).execute(plan(max_pages=2))

        self.assertEqual(len(result.documents), 1)
        self.assertEqual(len(result.hints), 1)
        self.assertEqual(result.hints[0].document_id, result.documents[0].document_id)
        self.assertEqual(
            [(item.keyword_id, item.score) for item in result.hints[0].keywords],
            [("speculative-decoding", 0.9), ("draft-model", 0.8)],
        )

    def test_retryable_channel_failure_retries_with_backoff(self) -> None:
        transport = SequenceTransport(
            [
                HttpResponse(
                    429, {"Content-Type": "application/json"}, b'{"error":"limit"}'
                ),
                response(work("W1", doi="10.1234/one")),
                response(),
                response(),
                response(),
            ]
        )
        pauses: list[float] = []

        with tempfile.TemporaryDirectory() as directory:
            result = OpenAlexDiscoveryExecutor(
                Path(directory),
                transport=transport,
                clock=lambda: NOW,
                monotonic=lambda: 0.0,
                sleeper=pauses.append,
            ).execute(plan())

        self.assertEqual(result.usage.requests_used, 5)
        self.assertEqual(pauses, [5.0])
        self.assertEqual(len(result.documents), 1)
        self.assertEqual(result.documents[0].title, "Work W1")

    def test_retry_honors_server_retry_after_over_backoff(self) -> None:
        transport = SequenceTransport(
            [
                HttpResponse(
                    429,
                    {"Content-Type": "application/json", "Retry-After": "39"},
                    b'{"error":"limit"}',
                ),
                response(work("W1", doi="10.1234/one")),
                response(),
                response(),
                response(),
            ]
        )
        pauses: list[float] = []

        with tempfile.TemporaryDirectory() as directory:
            result = OpenAlexDiscoveryExecutor(
                Path(directory),
                transport=transport,
                clock=lambda: NOW,
                monotonic=lambda: 0.0,
                sleeper=pauses.append,
            ).execute(plan())

        self.assertEqual(pauses, [39.0])
        self.assertEqual(len(result.documents), 1)

    def test_long_retry_after_stops_current_attempt_without_sleeping(self) -> None:
        limited = HttpResponse(
            429,
            {"Content-Type": "application/json", "Retry-After": "9612"},
            b'{"error":"limit"}',
        )
        transport = SequenceTransport([limited, limited, limited, limited])
        pauses: list[float] = []

        with tempfile.TemporaryDirectory() as directory:
            result = OpenAlexDiscoveryExecutor(
                Path(directory),
                transport=transport,
                clock=lambda: NOW,
                monotonic=lambda: 0.0,
                sleeper=pauses.append,
            ).execute(plan())

        self.assertEqual(pauses, [])
        self.assertEqual(result.usage.requests_used, 4)
        self.assertTrue(all(run.error.retryable for run in result.manifest.runs))

    def test_executes_bounded_channels_deduplicates_and_publishes_snapshot(self) -> None:
        transport = SequenceTransport(
            [
                response(
                    work("W1", doi="10.1234/shared"),
                    work("W2", doi="10.1234/two"),
                    work("W9", doi="10.1234/future", publication_date="2026-09-21"),
                ),
                response(
                    work("W1", doi="10.1234/shared"),
                    work("W3", doi="10.1234/three"),
                ),
                response(work("W4", doi="10.1234/four")),
                response(work("W5", doi="10.1234/five")),
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            result = OpenAlexDiscoveryExecutor(
                Path(directory),
                transport=transport,
                clock=lambda: NOW,
                monotonic=lambda: 0.0,
            ).execute(plan())

            self.assertTrue((result.snapshot_path / "manifest.json").is_file())
            manifest = json.loads(
                (result.snapshot_path / "manifest.json").read_text(encoding="utf-8")
            )

        self.assertIs(result.status, SnapshotStatus.COMPLETE)
        self.assertEqual(result.usage.requests_used, 4)
        self.assertEqual(result.usage.pages_used, 4)
        self.assertEqual(result.usage.returned_records, 7)
        self.assertEqual(result.usage.accepted_records, 6)
        self.assertEqual(result.usage.unique_documents, 5)
        self.assertEqual(result.usage.duplicate_documents, 1)
        self.assertEqual(result.usage.rejected_records, 1)
        self.assertIs(result.usage.stop_reason, DiscoveryStopReason.CHANNELS_EXHAUSTED)
        self.assertEqual(len(result.documents), 5)
        self.assertEqual(result.issues[0].code, "after_cutoff")
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(len(manifest["runs"]), 4)
        self.assertEqual(len(transport.calls), 4)
        self.assertTrue(all(call[2] == 12 for call in transport.calls))
        requested_pages = [parse_qs(urlparse(call[0]).query)["page"] for call in transport.calls]
        self.assertEqual(requested_pages, [["1"], ["1"], ["1"], ["1"]])
        json.dumps(result.to_dict(), ensure_ascii=False)

    def test_document_budget_stops_later_channels(self) -> None:
        transport = SequenceTransport([response(work("W1", doi="10.1234/one"))])

        with tempfile.TemporaryDirectory() as directory:
            result = OpenAlexDiscoveryExecutor(
                Path(directory),
                transport=transport,
                clock=lambda: NOW,
                monotonic=lambda: 0.0,
            ).execute(plan(max_documents=1))

        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(result.usage.unique_documents, 1)
        self.assertIs(result.usage.stop_reason, DiscoveryStopReason.DOCUMENT_BUDGET)
        parameters = parse_qs(urlparse(transport.calls[0][0]).query)
        self.assertEqual(parameters["per_page"], ["1"])

    def test_hint_issue_keeps_the_request_that_produced_it(self) -> None:
        broken = work("W1", doi="10.1234/one")
        broken["topics"] = "not-a-list"
        transport = SequenceTransport([response(broken)])

        with tempfile.TemporaryDirectory() as directory:
            result = OpenAlexDiscoveryExecutor(
                Path(directory),
                transport=transport,
                clock=lambda: NOW,
                monotonic=lambda: 0.0,
            ).execute(plan(max_pages=1))

        self.assertEqual(len(result.documents), 1)
        self.assertEqual(len(result.hint_issues), 1)
        self.assertEqual(
            result.hint_issues[0].request_id,
            result.manifest.runs[0].request.request_id,
        )
        self.assertEqual(result.hints[0].topics, ())

    def test_failed_channel_makes_snapshot_partial_but_other_channels_continue(self) -> None:
        transport = SequenceTransport(
            [
                HttpResponse(429, {"Content-Type": "application/json"}, b'{"error":"limit"}'),
                HttpResponse(429, {"Content-Type": "application/json"}, b'{"error":"limit"}'),
                HttpResponse(429, {"Content-Type": "application/json"}, b'{"error":"limit"}'),
                response(),
                response(),
            ]
        )
        pauses: list[float] = []

        with tempfile.TemporaryDirectory() as directory:
            result = OpenAlexDiscoveryExecutor(
                Path(directory),
                transport=transport,
                clock=lambda: NOW,
                monotonic=lambda: 0.0,
                sleeper=pauses.append,
            ).execute(plan(max_pages=2))

        self.assertIs(result.status, SnapshotStatus.PARTIAL)
        self.assertEqual(len(result.manifest.runs), 5)
        self.assertEqual(result.manifest.runs[0].error.code, "http_429")
        self.assertEqual(pauses, [5.0, 15.0])
        self.assertEqual(result.usage.returned_records, 0)
        self.assertIs(result.usage.stop_reason, DiscoveryStopReason.PAGE_BUDGET)

    def test_elapsed_budget_stops_after_current_page(self) -> None:
        transport = SequenceTransport([response()])
        moments = iter((0.0, 31.0, 31.0, 31.0))

        with tempfile.TemporaryDirectory() as directory:
            result = OpenAlexDiscoveryExecutor(
                Path(directory),
                transport=transport,
                clock=lambda: NOW,
                monotonic=lambda: next(moments),
            ).execute(plan())

        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(result.usage.elapsed_ms, 31000)
        self.assertIs(result.usage.stop_reason, DiscoveryStopReason.ELAPSED_BUDGET)


class OpenAlexDiscoveryExecutorApiKeyTests(unittest.TestCase):
    def test_forwards_api_key_as_bearer_on_every_call(self) -> None:
        transport = SequenceTransport(
            [response(work("W1", doi="10.1234/one")), response(), response(), response()]
        )

        with tempfile.TemporaryDirectory() as directory:
            OpenAlexDiscoveryExecutor(
                Path(directory),
                transport=transport,
                api_key="  openalex-free-key-1  ",
                clock=lambda: NOW,
                monotonic=lambda: 0.0,
            ).execute(plan())

        self.assertTrue(transport.calls)
        for url, headers, _ in transport.calls:
            self.assertEqual(headers["Authorization"], "Bearer openalex-free-key-1")
            self.assertNotIn("openalex-free-key-1", url)

    def test_blank_api_key_keeps_anonymous_behavior(self) -> None:
        transport = SequenceTransport(
            [response(work("W1", doi="10.1234/one")), response(), response(), response()]
        )

        with tempfile.TemporaryDirectory() as directory:
            OpenAlexDiscoveryExecutor(
                Path(directory),
                transport=transport,
                api_key="   ",
                clock=lambda: NOW,
                monotonic=lambda: 0.0,
            ).execute(plan())

        self.assertTrue(transport.calls)
        for _, headers, _ in transport.calls:
            self.assertNotIn("Authorization", headers)


if __name__ == "__main__":
    unittest.main()
