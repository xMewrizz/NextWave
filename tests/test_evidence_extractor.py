from __future__ import annotations

import json
import unittest
from dataclasses import replace
from typing import Any

from nextwave.contracts import SourceDocument, SourceType, TrustTier
from nextwave.discovery import (
    YANDEX_COMPLETION_ENDPOINT,
    AliasResolutionResult,
    CandidateOrigins,
    EvidenceCoverageStatus,
    EvidenceIssueCode,
    LlmProvider,
    LlmSelection,
    OriginGroup,
    OriginResolutionResult,
    ResolvedAliasGroup,
    StructuredEvidenceExtractor,
    build_evidence_extractor_from_environment,
)
from nextwave.sources import HttpResponse, NewsContentStatus, NewsDocumentEnrichment


def document(
    document_id: str,
    *,
    connector: str = "openalex",
    excerpt: str | None = "A prototype was tested at a university laboratory.",
    origin_id: str | None = None,
) -> SourceDocument:
    return SourceDocument(
        document_id=document_id,
        connector_id=connector,
        external_id=document_id,
        snapshot_id="snapshot-1",
        title="Photonic inference accelerator",
        url=f"https://example.org/{document_id}",
        canonical_url=f"https://example.org/{document_id}",
        source_type=SourceType.INDUSTRY_MEDIA
        if connector == "mediacloud"
        else SourceType.SCIENTIFIC_PUBLICATION,
        language="en",
        trust_tier=TrustTier.C if connector == "mediacloud" else TrustTier.A,
        origin_id=origin_id or document_id,
        excerpt=excerpt,
    )


def inputs(*documents: SourceDocument):
    group = ResolvedAliasGroup(
        "group-1",
        "scope-1",
        "Photonic inference accelerator",
        (),
        ("proposal-1",),
        tuple(item.document_id for item in documents),
        tuple(item.origin_id for item in documents),
        tuple(sorted({item.connector_id for item in documents})),
        "photonic inference accelerator",
    )
    aliases = AliasResolutionResult("scope-1", ("proposal-1",), (group,), ())
    origins = OriginResolutionResult(
        "plan-1",
        (
            CandidateOrigins(
                "group-1",
                tuple(
                    OriginGroup(
                        origin_id, tuple(item for item in documents if item.origin_id == origin_id)
                    )
                    for origin_id in dict.fromkeys(item.origin_id for item in documents)
                ),
                (),
            ),
        ),
    )
    return aliases, origins


def enrichment(item: SourceDocument, status: NewsContentStatus) -> NewsDocumentEnrichment:
    return NewsDocumentEnrichment(item, status, None, None, None, 0, None, False)


class QuoteGenerator:
    def __init__(
        self, *, quote: str = "A prototype was tested at a university laboratory."
    ) -> None:
        self.quote = quote

    def __call__(self, prompt: str) -> str:
        payload = json.loads(prompt.split("Input data as JSON:\n", 1)[1])
        return json.dumps(
            {
                "documents": [
                    {
                        "alias_group_id": item["alias_group_id"],
                        "document_id": item["document_id"],
                        "claims": [
                            {"quote": self.quote, "kind": "prototype", "direction": "support"}
                        ],
                    }
                    for item in payload["documents"]
                ]
            }
        )


class YandexTransport:
    def __init__(self) -> None:
        self.url: str | None = None
        self.payload: dict[str, Any] | None = None

    def post_json(
        self,
        url: str,
        *,
        headers,
        payload,
        timeout_seconds: float,
    ) -> HttpResponse:
        self.url = url
        self.payload = dict(payload)
        prompt = payload["messages"][0]["text"]
        output = QuoteGenerator()(prompt)
        body = {
            "result": {
                "alternatives": [{"message": {"role": "assistant", "text": output}}]
            }
        }
        self.assertions = (
            headers["Authorization"] == "Api-Key temporary-test-key",
            payload["modelUri"] == "gpt://folder-1/yandexgpt-5-lite",
            payload["jsonObject"] is True,
            len(payload["messages"]) == 1,
            timeout_seconds > 0,
        )
        return HttpResponse(200, {}, json.dumps(body).encode())


class EvidenceExtractorTests(unittest.TestCase):
    def extractor(self, generator=None) -> StructuredEvidenceExtractor:
        return StructuredEvidenceExtractor(
            generator or QuoteGenerator(),
            selection=LlmSelection(LlmProvider.YANDEX, "YandexGPT Lite 5"),
        )

    def test_verbatim_quote_is_a_pending_proposal_with_locator(self) -> None:
        item = document("science-1")
        result = self.extractor().extract(*inputs(item), ())
        self.assertEqual(len(result.proposals), 1)
        proposal = result.proposals[0]
        self.assertEqual(proposal.review_status, "pending")
        self.assertEqual(proposal.claim.locator, "excerpt[0:50]")
        self.assertEqual(proposal.origin_id, item.origin_id)
        self.assertEqual(result.coverage[0].status, EvidenceCoverageStatus.PROCESSED)
        json.dumps(result.to_dict())

    def test_paraphrase_cannot_become_a_claim(self) -> None:
        item = document("science-1")
        result = self.extractor(
            QuoteGenerator(quote="A different prototype was tested at a university laboratory.")
        ).extract(*inputs(item), ())
        self.assertEqual(result.proposals, ())
        self.assertEqual(result.issues[0].code, EvidenceIssueCode.NON_VERBATIM)

    def test_news_requires_confirmed_article_text(self) -> None:
        items = tuple(document(f"news-{index}", connector="mediacloud") for index in range(3))
        aliases, origins = inputs(*items)
        statuses = (
            NewsContentStatus.TITLE_ONLY,
            NewsContentStatus.META_DESCRIPTION,
            NewsContentStatus.ARTICLE_TEXT,
        )
        result = self.extractor().extract(
            aliases,
            origins,
            tuple(enrichment(item, status) for item, status in zip(items, statuses, strict=True)),
        )
        self.assertEqual(len(result.proposals), 1)
        self.assertEqual(result.proposals[0].claim.document_id, "news-2")
        self.assertEqual(
            sum(
                item.status is EvidenceCoverageStatus.NEWS_TEXT_UNAVAILABLE
                for item in result.coverage
            ),
            2,
        )

    def test_news_text_must_match_the_enriched_document(self) -> None:
        item = document("news-1", connector="mediacloud")
        stale = document("news-1", connector="mediacloud", excerpt="An unrelated page was fetched.")
        aliases, origins = inputs(item)
        result = self.extractor().extract(
            aliases,
            origins,
            (enrichment(stale, NewsContentStatus.ARTICLE_TEXT),),
        )
        self.assertEqual(result.proposals, ())
        self.assertEqual(result.coverage[0].status, EvidenceCoverageStatus.NEWS_TEXT_UNAVAILABLE)

    def test_duplicate_origin_is_proposed_once(self) -> None:
        first = document("science-1", origin_id="doi-1")
        second = document("science-2", origin_id="doi-1")
        result = self.extractor().extract(*inputs(first, second), ())
        self.assertEqual(len(result.proposals), 1)
        self.assertEqual(
            sum(item.status is EvidenceCoverageStatus.SAME_ORIGIN for item in result.coverage), 1
        )

    def test_model_failure_is_not_reported_as_zero_evidence(self) -> None:
        def fail(_prompt: str) -> str:
            raise RuntimeError("unavailable")

        result = self.extractor(fail).extract(*inputs(document("science-1")), ())
        self.assertEqual(result.proposals, ())
        self.assertEqual(result.issues[0].code, EvidenceIssueCode.MODEL_ERROR)
        self.assertEqual(result.coverage[0].status, EvidenceCoverageStatus.MODEL_FAILED)

    def test_billing_failure_is_visible_without_raw_response(self) -> None:
        def fail(_prompt: str) -> str:
            raise RuntimeError("Yandex completion API returned HTTP 429")

        result = self.extractor(fail).extract(*inputs(document("science-1")), ())
        self.assertEqual(result.proposals, ())
        self.assertEqual(result.issues[0].code, EvidenceIssueCode.MODEL_ERROR)
        self.assertEqual(
            result.issues[0].message,
            "Yandex completion API returned HTTP 429",
        )

    def test_shared_document_can_serve_two_candidates(self) -> None:
        item = document("science-1")
        aliases, origins = inputs(item)
        second_group = replace(
            aliases.groups[0], group_id="group-2", canonical_name="Optical inference chip"
        )
        aliases = replace(aliases, groups=(*aliases.groups, second_group))
        origins = replace(
            origins,
            candidates=(
                *origins.candidates,
                replace(origins.candidates[0], alias_group_id="group-2"),
            ),
        )
        result = self.extractor().extract(aliases, origins, ())
        self.assertEqual(len(result.proposals), 2)
        self.assertEqual(
            {proposal.alias_group_id for proposal in result.proposals},
            {"group-1", "group-2"},
        )

    def test_environment_builder_sends_and_parses_yandex_response(self) -> None:
        transport = YandexTransport()
        extractor = build_evidence_extractor_from_environment(
            {
                "NEXTWAVE_LLM_PROVIDER": "yandex",
                "NEXTWAVE_LLM_MODEL": "YandexGPT Lite 5",
                "NEXTWAVE_LLM_API_KEY": "temporary-test-key",
                "NEXTWAVE_YANDEX_FOLDER_ID": "folder-1",
            },
            llm_transport=transport,
        )
        result = extractor.extract(*inputs(document("science-1")), ())
        self.assertEqual(len(result.proposals), 1)
        self.assertEqual(transport.url, YANDEX_COMPLETION_ENDPOINT)
        self.assertEqual(transport.assertions, (True, True, True, True, True))
        self.assertTrue(transport.payload["jsonObject"])
        self.assertNotIn("temporary-test-key", json.dumps(transport.payload))


if __name__ == "__main__":
    unittest.main()
