from __future__ import annotations

import unittest
from datetime import UTC, date, datetime, timedelta

from nextwave.sources import (
    ConnectorError,
    ConnectorId,
    ConnectorRequest,
    ConnectorRun,
    ConnectorStatus,
    QueryParameter,
    QueryPurpose,
    RawResponseArtifact,
    RetrievalChannel,
    SnapshotManifest,
    SnapshotStatus,
    SourceQuery,
)

NOW = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
SHA256 = "a" * 64


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
        languages=("en", "ru"),
        topic_ids=("T10001",),
    )


def make_request(
    connector_id: ConnectorId = ConnectorId.OPENALEX,
    request_id: str = "request-openalex-001",
) -> ConnectorRequest:
    return ConnectorRequest(
        request_id=request_id,
        query_id="query-ai-001",
        connector_id=connector_id,
        channel=RetrievalChannel.TEXT,
        endpoint=f"https://api.example.org/{connector_id.value}",
        parameters=(
            QueryParameter("filter", "from_publication_date:2025-09-16"),
            QueryParameter("search", "artificial intelligence"),
        ),
    )


def make_artifact(uri: str = "openalex/request-openalex-001.json") -> RawResponseArtifact:
    return RawResponseArtifact(
        uri=uri,
        media_type="application/json",
        size_bytes=128,
        sha256=SHA256,
        retrieved_at=NOW,
    )


def make_success_run(
    connector_id: ConnectorId = ConnectorId.OPENALEX,
    run_id: str = "run-openalex-001",
    request_id: str = "request-openalex-001",
    artifact_uri: str = "openalex/request-openalex-001.json",
    returned_records: int = 12,
) -> ConnectorRun:
    return ConnectorRun(
        run_id=run_id,
        request=make_request(connector_id, request_id),
        status=ConnectorStatus.SUCCESS,
        started_at=NOW,
        finished_at=NOW + timedelta(milliseconds=250),
        http_status=200,
        returned_records=returned_records,
        artifact=make_artifact(artifact_uri),
    )


class SourceContractTests(unittest.TestCase):
    def test_query_preserves_user_text_and_normalized_search_plan(self) -> None:
        query = make_query()

        self.assertEqual(query.raw_query, "Технологии в ИИ")
        self.assertEqual(query.normalized_query, "artificial intelligence")
        self.assertEqual(query.to_dict()["published_until"], "2026-09-15")

    def test_query_rejects_documents_window_after_cutoff(self) -> None:
        with self.assertRaisesRegex(ValueError, "published_until"):
            SourceQuery(
                query_id="query-ai-001",
                analysis_scope_id="scope-ai-001",
                purpose=QueryPurpose.DISCOVERY,
                raw_query="ИИ",
                normalized_query="artificial intelligence",
                search_texts=("artificial intelligence",),
                published_from=date(2025, 1, 1),
                published_until=date(2026, 9, 16),
                cutoff_date=date(2026, 9, 15),
            )

    def test_query_rejects_duplicate_search_texts(self) -> None:
        with self.assertRaisesRegex(ValueError, "search_texts"):
            SourceQuery(
                query_id="query-ai-001",
                analysis_scope_id="scope-ai-001",
                purpose=QueryPurpose.DISCOVERY,
                raw_query="ИИ",
                normalized_query="artificial intelligence",
                search_texts=("Artificial Intelligence", "artificial intelligence"),
                published_from=date(2025, 1, 1),
                published_until=date(2026, 9, 15),
                cutoff_date=date(2026, 9, 15),
            )

    def test_query_rejects_topic_and_subfield_filters_together(self) -> None:
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            SourceQuery(
                query_id="query-ai-001",
                analysis_scope_id="scope-ai-001",
                purpose=QueryPurpose.DISCOVERY,
                raw_query="ИИ",
                normalized_query="artificial intelligence",
                search_texts=("artificial intelligence",),
                published_from=date(2025, 1, 1),
                published_until=date(2026, 9, 15),
                cutoff_date=date(2026, 9, 15),
                topic_ids=("T11636",),
                subfield_ids=("1702",),
            )

    def test_connector_parameters_must_have_deterministic_order(self) -> None:
        with self.assertRaisesRegex(ValueError, "sorted"):
            ConnectorRequest(
                request_id="request-openalex-001",
                query_id="query-ai-001",
                connector_id=ConnectorId.OPENALEX,
                channel=RetrievalChannel.TEXT,
                endpoint="https://api.openalex.org/works",
                parameters=(
                    QueryParameter("search", "artificial intelligence"),
                    QueryParameter("filter", "from_publication_date:2025-09-16"),
                ),
            )

    def test_successful_zero_results_are_complete_coverage(self) -> None:
        run = make_success_run(returned_records=0)

        self.assertTrue(run.coverage_complete)
        self.assertEqual(run.returned_records, 0)

    def test_failed_run_keeps_record_count_unknown(self) -> None:
        run = ConnectorRun(
            run_id="run-crossref-001",
            request=make_request(
                ConnectorId.CROSSREF,
                request_id="request-crossref-001",
            ),
            status=ConnectorStatus.FAILED,
            started_at=NOW,
            finished_at=NOW + timedelta(seconds=3),
            http_status=None,
            returned_records=None,
            artifact=None,
            error=ConnectorError(
                code="network_timeout",
                message="Crossref did not answer before the deadline",
                retryable=True,
            ),
        )

        self.assertFalse(run.coverage_complete)
        self.assertIsNone(run.returned_records)

    def test_connector_error_rejects_non_finite_retry_delay(self) -> None:
        for value in (float("nan"), float("inf"), -1.0):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "finite and non-negative"
            ):
                ConnectorError(
                    code="rate_limited",
                    message="Retry later",
                    retryable=True,
                    retry_after_seconds=value,
                )

    def test_failed_run_cannot_report_zero_records(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not report"):
            ConnectorRun(
                run_id="run-crossref-001",
                request=make_request(
                    ConnectorId.CROSSREF,
                    request_id="request-crossref-001",
                ),
                status=ConnectorStatus.FAILED,
                started_at=NOW,
                finished_at=NOW + timedelta(seconds=3),
                http_status=503,
                returned_records=0,
                artifact=None,
                error=ConnectorError(
                    code="service_unavailable",
                    message="Crossref returned 503",
                    retryable=True,
                ),
            )

    def test_snapshot_is_partial_when_one_connector_fails(self) -> None:
        failed_run = ConnectorRun(
            run_id="run-crossref-001",
            request=make_request(
                ConnectorId.CROSSREF,
                request_id="request-crossref-001",
            ),
            status=ConnectorStatus.FAILED,
            started_at=NOW,
            finished_at=NOW + timedelta(seconds=3),
            http_status=503,
            returned_records=None,
            artifact=make_artifact("crossref/request-crossref-001-error.json"),
            error=ConnectorError(
                code="service_unavailable",
                message="Crossref returned 503",
                retryable=True,
            ),
        )
        snapshot = SnapshotManifest(
            snapshot_id="snapshot-ai-001",
            snapshot_version="snapshot-v1",
            analysis_id="analysis-ai-001",
            created_at=NOW,
            query=make_query(),
            runs=(make_success_run(), failed_run),
        )

        self.assertIs(snapshot.status, SnapshotStatus.PARTIAL)
        self.assertEqual(snapshot.to_dict()["status"], "partial")

    def test_snapshot_rejects_run_for_another_query(self) -> None:
        request = ConnectorRequest(
            request_id="request-openalex-002",
            query_id="query-other-001",
            connector_id=ConnectorId.OPENALEX,
            channel=RetrievalChannel.TEXT,
            endpoint="https://api.openalex.org/works",
            parameters=(QueryParameter("search", "robotics"),),
        )
        run = ConnectorRun(
            run_id="run-openalex-002",
            request=request,
            status=ConnectorStatus.SUCCESS,
            started_at=NOW,
            finished_at=NOW + timedelta(milliseconds=200),
            http_status=200,
            returned_records=1,
            artifact=make_artifact("openalex/request-openalex-002.json"),
        )

        with self.assertRaisesRegex(ValueError, "snapshot query"):
            SnapshotManifest(
                snapshot_id="snapshot-ai-001",
                snapshot_version="snapshot-v1",
                analysis_id="analysis-ai-001",
                created_at=NOW,
                query=make_query(),
                runs=(run,),
            )


if __name__ == "__main__":
    unittest.main()
