from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

from nextwave.discovery import (
    CandidateMentionKind,
    CandidateProposal,
    CandidateProposalBatch,
    CandidateVerificationExecutor,
    DiscoveryBudget,
    DiscoveryPipeline,
    LlmProvider,
    LlmSelection,
    MediaDiscoveryExecutor,
    OpenAlexDiscoveryExecutor,
    ScopeGranularity,
    StructuredCandidateGate,
    StructuredCandidateMentionExtractor,
    StructuredEvidenceExtractor,
    build_analysis_scope,
    build_discovery_plan,
    split_gate_batch,
)
from nextwave.sources import (
    ConnectorId,
    HttpResponse,
    NewsContentStatus,
    NewsDocumentEnrichment,
)

NOW = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)


def plan():
    scope = build_analysis_scope(
        raw_query="Технологии в ИИ",
        normalized_query="artificial intelligence",
        search_texts=("Технологии в ИИ",),
        languages=("en", "ru"),
        granularity=ScopeGranularity.DIRECTION,
        subfield_ids=("1702",),
    )
    return build_discovery_plan(
        analysis_id="analysis-ai-001",
        scope=scope,
        published_from=date(2025, 9, 21),
        cutoff_date=date(2026, 9, 21),
        budgets=(
            DiscoveryBudget(ConnectorId.OPENALEX, True, 1, 1, 10, 10, 20),
            DiscoveryBudget(ConnectorId.MEDIACLOUD, False, 2, 2, 10, 10, 20),
            DiscoveryBudget(ConnectorId.GDELT, False, 1, 1, 10, 10, 20),
        ),
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
                    excerpt="The article reports a prototype tested in a data center.",
                ),
                status=NewsContentStatus.ARTICLE_TEXT,
                issue_code=None,
                message=None,
                content_sha256="b" * 64,
                fetched_bytes=200,
                excerpt_source="test.article",
                excerpt_truncated=False,
            )
            for document in documents
        )


class GroundedGenerator:
    def __call__(self, prompt: str) -> str:
        payload = json.loads(prompt.split("Input data as JSON:\n", 1)[1])
        results = []
        for document in payload["documents"]:
            phrase = "Photonic inference accelerator"
            mentions = [{"text": phrase, "field": "title"}] if phrase in document["title"] else []
            results.append({"document_id": document["document_id"], "mentions": mentions})
        return json.dumps({"documents": results})


class GateGenerator:
    def __call__(self, prompt: str) -> str:
        payload = json.loads(prompt.split("Input data as JSON:\n", 1)[1])
        return json.dumps(
            {
                "decisions": [
                    {
                        "proposal_id": proposal["proposal_id"],
                        "decision": "accept",
                        "reason": "concrete_technology",
                        "basis_document_ids": [proposal["documents"][0]["document_id"]],
                        "explanation": "A specific inference accelerator is described.",
                    }
                    for proposal in payload["proposals"]
                ]
            }
        )


class EvidenceGenerator:
    def __call__(self, prompt: str) -> str:
        payload = json.loads(prompt.split("Input data as JSON:\n", 1)[1])
        return json.dumps(
            {
                "documents": [
                    {
                        "alias_group_id": item["alias_group_id"],
                        "document_id": item["document_id"],
                        "claims": [
                            {
                                "quote": "The article reports a prototype tested in a data center.",
                                "kind": "prototype",
                                "direction": "support",
                            }
                        ]
                        if "The article reports a prototype tested in a data center."
                        in item["excerpt"]
                        else [],
                    }
                    for item in payload["documents"]
                ]
            }
        )


def openalex_response() -> HttpResponse:
    payload = {
        "meta": {"count": 1},
        "results": [
            {
                "id": "https://openalex.org/W1",
                "doi": "https://doi.org/10.1234/photonic",
                "title": "Photonic inference accelerator for artificial intelligence",
                "publication_date": "2026-08-10",
                "language": "en",
                "type": "article",
                "authorships": [],
                "abstract_inverted_index": {
                    "Photonic": [0],
                    "prototype": [1],
                },
                "keywords": [
                    {
                        "id": "https://openalex.org/keywords/photonic-inference-accelerator",
                        "display_name": "Photonic inference accelerator",
                        "score": 0.92,
                    }
                ],
                "primary_location": {
                    "landing_page_url": "https://science.example/photonic",
                    "source": {"display_name": "Example Journal"},
                },
            }
        ],
    }
    return HttpResponse(
        200,
        {"Content-Type": "application/json"},
        json.dumps(payload).encode(),
    )


def media_response(number: int, language: str) -> HttpResponse:
    payload = {
        "stories": [
            {
                "id": f"story-{number}",
                "url": f"https://news.example/photonic-{number}",
                "title": f"Photonic inference accelerator reaches pilot {number}",
                "publish_date": "2026-09-20",
                "indexed_date": "2026-09-20T10:00:00Z",
                "language": language,
                "media_name": "Technology News",
            }
        ],
        "pagination_token": None,
    }
    return HttpResponse(
        200,
        {"Content-Type": "application/json"},
        json.dumps(payload).encode(),
    )


class DiscoveryPipelineTests(unittest.TestCase):
    def test_builds_one_cross_source_candidate_proposal_pool(self) -> None:
        scientific_transport = SequenceTransport(openalex_response())
        media_transport = SequenceTransport(
            media_response(1, "en"),
            media_response(2, "ru"),
        )
        verification_transport = SequenceTransport(openalex_response())
        extractor = StructuredCandidateMentionExtractor(
            GroundedGenerator(),
            selection=LlmSelection(LlmProvider.YANDEX, "YandexGPT Lite 5"),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = DiscoveryPipeline(
                OpenAlexDiscoveryExecutor(
                    root,
                    transport=scientific_transport,
                    clock=lambda: NOW,
                    monotonic=lambda: 0.0,
                ),
                MediaDiscoveryExecutor(
                    root,
                    mediacloud_api_key="temporary-key",
                    mediacloud_transport=media_transport,
                    news_enricher=StubNewsEnricher(),
                    clock=lambda: NOW,
                    monotonic=lambda: 0.0,
                    mediacloud_min_interval_seconds=0,
                ),
                extractor,
                StructuredCandidateGate(
                    GateGenerator(),
                    selection=LlmSelection(LlmProvider.YANDEX, "YandexGPT Lite 5"),
                ),
                CandidateVerificationExecutor(
                    root,
                    transport=verification_transport,
                    clock=lambda: NOW,
                    monotonic=lambda: 0.0,
                ),
                StructuredEvidenceExtractor(
                    EvidenceGenerator(),
                    selection=LlmSelection(LlmProvider.YANDEX, "YandexGPT Lite 5"),
                ),
            ).execute(plan())

        proposal = next(
            item
            for item in result.candidate_proposals.proposals
            if item.normalized_name == "photonic inference accelerator"
        )
        self.assertEqual(len(result.documents), 3)
        self.assertEqual(proposal.origin_count, 3)
        self.assertEqual(proposal.connector_ids, ("mediacloud", "openalex"))
        self.assertEqual(result.text_extraction.batch_count, 1)
        self.assertEqual(len(scientific_transport.calls), 1)
        self.assertEqual(len(media_transport.calls), 2)
        self.assertIn(proposal.proposal_id, result.candidate_gate.accepted_proposal_ids)
        self.assertIn(
            proposal.proposal_id,
            result.alias_resolution.groups[0].proposal_ids,
        )
        self.assertEqual(result.verification.requests_used, 1)
        self.assertEqual(result.verification.results[0].matching_origin_count, 1)
        self.assertEqual(result.origin_resolution.candidates[0].document_count, 4)
        self.assertEqual(result.origin_resolution.candidates[0].exact_origin_count, 3)
        self.assertEqual(len(result.evidence_extraction.proposals), 2)
        self.assertTrue(
            all(item.review_status == "pending" for item in result.evidence_extraction.proposals)
        )
        self.assertEqual(len(verification_transport.calls), 1)
        json.dumps(result.to_dict(), ensure_ascii=False)


def make_proposal(
    proposal_id: str,
    *,
    origin_count: int = 1,
    connector_ids: tuple[str, ...] = ("openalex",),
) -> CandidateProposal:
    return CandidateProposal(
        proposal_id=proposal_id,
        analysis_scope_id="scope-ai-001",
        canonical_name=f"Tech {proposal_id}",
        normalized_name=f"tech {proposal_id}",
        aliases=(),
        mention_ids=(),
        source_kinds=(CandidateMentionKind.TITLE,),
        connector_ids=connector_ids,
        provider_term_ids=(),
        document_ids=tuple(
            f"document-{proposal_id}-{index}" for index in range(origin_count)
        ),
        origin_ids=tuple(f"origin-{proposal_id}-{index}" for index in range(origin_count)),
        max_provider_score=None,
        primary_provider_topic=False,
    )


class GateCapTests(unittest.TestCase):
    def test_split_redistributes_places_when_only_one_stratum_exists(self) -> None:
        batch = CandidateProposalBatch(
            analysis_scope_id="scope-ai-001",
            proposals=tuple(make_proposal(f"p{index}") for index in range(5)),
            exclusions=(),
        )

        sub, skipped = split_gate_batch(batch, 2)

        self.assertEqual(
            [item.proposal_id for item in sub.proposals], ["p0", "p1"]
        )
        self.assertEqual(sub.analysis_scope_id, "scope-ai-001")
        self.assertEqual(skipped, 3)

    def test_split_represents_each_visibility_stratum(self) -> None:
        batch = CandidateProposalBatch(
            analysis_scope_id="scope-ai-001",
            proposals=(
                make_proposal("established-1", origin_count=8),
                make_proposal("established-2", origin_count=5),
                make_proposal("emerging", origin_count=2),
                make_proposal("novel"),
                make_proposal(
                    "cross-source",
                    origin_count=2,
                    connector_ids=("mediacloud", "openalex"),
                ),
            ),
            exclusions=(),
        )

        sub, skipped = split_gate_batch(batch, 4)

        self.assertEqual(
            [item.proposal_id for item in sub.proposals],
            ["cross-source", "emerging", "novel", "established-1"],
        )
        self.assertEqual(skipped, 1)

    def test_split_without_shortage_skips_nothing(self) -> None:
        batch = CandidateProposalBatch(
            analysis_scope_id="scope-ai-001",
            proposals=(make_proposal("p0"),),
            exclusions=(),
        )

        sub, skipped = split_gate_batch(batch, 300)

        self.assertEqual(len(sub.proposals), 1)
        self.assertEqual(skipped, 0)

    def test_split_rejects_non_positive_cap(self) -> None:
        batch = CandidateProposalBatch(
            analysis_scope_id="scope-ai-001",
            proposals=(make_proposal("p0"),),
            exclusions=(),
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            split_gate_batch(batch, 0)


class DiscoveryProgressTests(unittest.TestCase):
    def test_progress_reports_timed_stages_in_order(self) -> None:
        scientific_transport = SequenceTransport(openalex_response())
        media_transport = SequenceTransport(
            media_response(1, "en"),
            media_response(2, "ru"),
        )
        verification_transport = SequenceTransport(openalex_response())
        extractor = StructuredCandidateMentionExtractor(
            GroundedGenerator(),
            selection=LlmSelection(LlmProvider.YANDEX, "YandexGPT Lite 5"),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events: list[str] = []
            DiscoveryPipeline(
                OpenAlexDiscoveryExecutor(
                    root,
                    transport=scientific_transport,
                    clock=lambda: NOW,
                    monotonic=lambda: 0.0,
                ),
                MediaDiscoveryExecutor(
                    root,
                    mediacloud_api_key="temporary-key",
                    mediacloud_transport=media_transport,
                    news_enricher=StubNewsEnricher(),
                    clock=lambda: NOW,
                    monotonic=lambda: 0.0,
                    mediacloud_min_interval_seconds=0,
                ),
                extractor,
                StructuredCandidateGate(
                    GateGenerator(),
                    selection=LlmSelection(LlmProvider.YANDEX, "YandexGPT Lite 5"),
                ),
                CandidateVerificationExecutor(
                    root,
                    transport=verification_transport,
                    clock=lambda: NOW,
                    monotonic=lambda: 0.0,
                ),
                StructuredEvidenceExtractor(
                    EvidenceGenerator(),
                    selection=LlmSelection(LlmProvider.YANDEX, "YandexGPT Lite 5"),
                ),
            ).execute(plan(), progress=events.append)

        stages = [event.split(":")[0] for event in events]
        self.assertEqual(
            stages,
            [
                "[discovery] sources",
                "[discovery] extraction",
                "[discovery] proposals",
                "[discovery] gate",
                "[discovery] aliases",
                "[discovery] verification",
                "[discovery] origins",
                "[discovery] evidence",
            ],
        )
        self.assertIn("3 documents", events[0])


if __name__ == "__main__":
    unittest.main()
