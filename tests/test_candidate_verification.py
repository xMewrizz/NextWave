from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from nextwave.discovery import (
    AliasResolutionResult,
    CandidateVerificationExecutor,
    DiscoveryBudget,
    ResolvedAliasGroup,
    ScopeGranularity,
    VerificationStatus,
    build_analysis_scope,
    build_discovery_plan,
)
from nextwave.sources import ConnectorId, HttpResponse

NOW = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)


def plan():
    scope = build_analysis_scope(
        raw_query="Технологии в ИИ",
        normalized_query="artificial intelligence",
        search_texts=("AI",),
        languages=("en", "ru"),
        granularity=ScopeGranularity.DIRECTION,
        subfield_ids=("1702",),
    )
    return build_discovery_plan(
        analysis_id="analysis-ai-001",
        scope=scope,
        published_from=date(2025, 9, 21),
        cutoff_date=date(2026, 9, 21),
        budgets=(DiscoveryBudget(ConnectorId.OPENALEX, True, 1, 1, 20, 10, 20),),
    )


def aliases(scope_id: str, *names: str):
    groups = tuple(
        ResolvedAliasGroup(
            group_id=f"alias-group-{index}",
            analysis_scope_id=scope_id,
            canonical_name=name,
            aliases=(),
            proposal_ids=(f"proposal-{index}",),
            document_ids=(f"document-discovery-{index}",),
            origin_ids=(f"origin-discovery-{index}",),
            connector_ids=("mediacloud",),
            normalization_key=name.casefold(),
        )
        for index, name in enumerate(names, 1)
    )
    return AliasResolutionResult(
        scope_id,
        tuple(f"proposal-{index}" for index in range(1, len(names) + 1)),
        groups,
        (),
    )


def response(*titles: str):
    return HttpResponse(
        200,
        {"Content-Type": "application/json"},
        json.dumps({
            "meta": {"count": len(titles)},
            "results": [
                {
                    "id": f"https://openalex.org/W{index}",
                    "doi": f"https://doi.org/10.1234/work-{index}",
                    "title": title,
                    "publication_date": "2026-08-10",
                    "language": "en",
                    "type": "article",
                    "authorships": [],
                    "abstract_inverted_index": {"Technical": [0], "work": [1]},
                    "primary_location": {
                        "landing_page_url": f"https://science.example/work-{index}",
                        "source": {"display_name": "Example Journal"},
                    },
                }
                for index, title in enumerate(titles, 1)
            ],
        }).encode(),
    )


class SequenceTransport:
    def __init__(self, *responses):
        self._responses = iter(responses)
        self.calls = []
        self.timeouts = []

    def get(self, url, *, headers, timeout_seconds):
        self.calls.append(url)
        self.timeouts.append(timeout_seconds)
        return next(self._responses)


class CandidateVerificationTests(unittest.TestCase):
    def test_runs_candidate_query_without_scope_taxonomy_and_saves_snapshot(self):
        discovery_plan = plan()
        transport = SequenceTransport(response(
            "Speculative decoding reduces inference latency",
            "Unrelated data center paper",
        ))
        with tempfile.TemporaryDirectory() as directory:
            result = CandidateVerificationExecutor(
                Path(directory),
                transport=transport,
                clock=lambda: NOW,
                monotonic=lambda: 0.0,
            ).execute(
                discovery_plan,
                aliases(discovery_plan.scope.scope_id, "Speculative decoding"),
            )
            item = result.results[0]
            self.assertTrue((item.snapshot_path / "manifest.json").is_file())
            manifest = json.loads((item.snapshot_path / "manifest.json").read_text())

        self.assertEqual(item.status, VerificationStatus.SEARCHED)
        self.assertEqual(item.returned_records, 2)
        self.assertEqual(item.matching_origin_count, 1)
        self.assertEqual(manifest["query"]["purpose"], "verification")
        parameters = parse_qs(urlparse(transport.calls[0]).query)
        self.assertEqual(parameters["search"], ["Speculative decoding"])
        self.assertNotIn("topics.subfield.id", parameters["filter"][0])
        json.dumps(result.to_dict(), ensure_ascii=False)

    def test_successful_zero_does_not_mean_provider_failure(self):
        discovery_plan = plan()
        with tempfile.TemporaryDirectory() as directory:
            result = CandidateVerificationExecutor(
                Path(directory),
                transport=SequenceTransport(response()),
                clock=lambda: NOW,
                monotonic=lambda: 0.0,
            ).execute(
                discovery_plan,
                aliases(discovery_plan.scope.scope_id, "Speculative decoding"),
            )
        self.assertEqual(result.results[0].status, VerificationStatus.SEARCHED)
        self.assertEqual(result.results[0].returned_records, 0)
        self.assertEqual(result.results[0].matching_documents, ())

    def test_keeps_multiple_documents_with_the_same_doi_for_origin_review(self):
        discovery_plan = plan()
        payload = json.loads(response(
            "Speculative decoding in one study",
            "Speculative decoding in a second record",
        ).body)
        payload["results"][1]["doi"] = payload["results"][0]["doi"]
        transport = SequenceTransport(HttpResponse(
            200, {"Content-Type": "application/json"}, json.dumps(payload).encode()
        ))
        with tempfile.TemporaryDirectory() as directory:
            result = CandidateVerificationExecutor(
                Path(directory),
                transport=transport,
                clock=lambda: NOW,
                monotonic=lambda: 0.0,
            ).execute(
                discovery_plan,
                aliases(discovery_plan.scope.scope_id, "Speculative decoding"),
            )
        item = result.results[0]
        self.assertEqual(len(item.matching_documents), 2)
        self.assertEqual(item.matching_origin_count, 1)

    def test_failure_is_not_recorded_as_zero(self):
        discovery_plan = plan()
        transport = SequenceTransport(HttpResponse(429, {}, b"rate limited"))
        with tempfile.TemporaryDirectory() as directory:
            result = CandidateVerificationExecutor(
                Path(directory),
                transport=transport,
                clock=lambda: NOW,
                monotonic=lambda: 0.0,
            ).execute(
                discovery_plan,
                aliases(discovery_plan.scope.scope_id, "Speculative decoding"),
            )
        self.assertEqual(result.results[0].status, VerificationStatus.FAILED)
        self.assertIsNone(result.results[0].returned_records)
        self.assertEqual(result.results[0].error_code, "http_429")

    def test_unusable_records_are_reported_separately_from_matches(self):
        discovery_plan = plan()
        payload = json.loads(response("Speculative decoding works").body)
        payload["results"][0]["title"] = ""
        transport = SequenceTransport(HttpResponse(
            200, {"Content-Type": "application/json"}, json.dumps(payload).encode()
        ))
        with tempfile.TemporaryDirectory() as directory:
            result = CandidateVerificationExecutor(
                Path(directory),
                transport=transport,
                clock=lambda: NOW,
                monotonic=lambda: 0.0,
            ).execute(
                discovery_plan,
                aliases(discovery_plan.scope.scope_id, "Speculative decoding"),
            )
        item = result.results[0]
        self.assertEqual(item.status, VerificationStatus.SEARCHED)
        self.assertEqual(item.returned_records, 1)
        self.assertEqual(item.matching_documents, ())
        self.assertEqual(item.rejected_record_codes, ("missing_title",))

    def test_global_budget_marks_unsearched_groups(self):
        discovery_plan = plan()
        transport = SequenceTransport(response())
        with tempfile.TemporaryDirectory() as directory:
            result = CandidateVerificationExecutor(
                Path(directory),
                transport=transport,
                max_groups=1,
                clock=lambda: NOW,
                monotonic=lambda: 0.0,
            ).execute(
                discovery_plan,
                aliases(
                    discovery_plan.scope.scope_id,
                    "Speculative decoding",
                    "Photonic inference accelerator",
                ),
            )
        self.assertEqual(result.requests_used, 1)
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(result.results[1].status, VerificationStatus.SKIPPED_BUDGET)
        self.assertIsNone(result.results[1].returned_records)

    def test_request_timeout_respects_remaining_run_budget(self):
        discovery_plan = plan()
        transport = SequenceTransport(response())
        ticks = iter((0.0, 119.0))
        with tempfile.TemporaryDirectory() as directory:
            CandidateVerificationExecutor(
                Path(directory),
                transport=transport,
                clock=lambda: NOW,
                monotonic=lambda: next(ticks),
            ).execute(
                discovery_plan,
                aliases(discovery_plan.scope.scope_id, "Speculative decoding"),
            )
        self.assertEqual(transport.timeouts, [1.0])

    def test_short_acronym_does_not_match_inside_unrelated_word(self):
        discovery_plan = plan()
        with tempfile.TemporaryDirectory() as directory:
            result = CandidateVerificationExecutor(
                Path(directory),
                transport=SequenceTransport(response("Modern storage system")),
                clock=lambda: NOW,
                monotonic=lambda: 0.0,
            ).execute(discovery_plan, aliases(discovery_plan.scope.scope_id, "RAG"))
        self.assertEqual(result.results[0].matching_documents, ())


if __name__ == "__main__":
    unittest.main()
