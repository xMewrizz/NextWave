from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime
from http.client import IncompleteRead

from nextwave.contracts import SourceDocument, SourceType, TrustTier
from nextwave.sources import (
    MAX_NEWS_PAGE_BYTES,
    HttpResponse,
    NewsContentStatus,
    NewsDocumentEnricher,
    NewsEnrichmentIssueCode,
)


def news_document(*, excerpt: str | None = None, url: str = "https://news.example/story"):
    return SourceDocument(
        document_id="document-news-1",
        connector_id="mediacloud",
        external_id="story-1",
        snapshot_id="snapshot-news-1",
        title="Startup launches photonic inference accelerator",
        url=url,
        canonical_url=url,
        source_type=SourceType.INDUSTRY_MEDIA,
        language="en",
        trust_tier=TrustTier.C,
        origin_id=f"url:{url}",
        retrieved_at=datetime(2026, 9, 21, 10, 0, tzinfo=UTC),
        excerpt=excerpt,
    )


def numbered_news_document(number: int) -> SourceDocument:
    base = news_document(url=f"https://news.example/story-{number}")
    return SourceDocument(
        document_id=f"document-news-{number}",
        connector_id=base.connector_id,
        external_id=f"story-{number}",
        snapshot_id=base.snapshot_id,
        title=base.title,
        url=base.url,
        canonical_url=base.canonical_url,
        source_type=base.source_type,
        language=base.language,
        trust_tier=base.trust_tier,
        origin_id=base.origin_id,
        retrieved_at=base.retrieved_at,
    )


class FakeTransport:
    def __init__(self, response: HttpResponse) -> None:
        self.response = response
        self.calls: list[str] = []

    def get(self, url, *, headers, timeout_seconds):
        self.calls.append(url)
        return self.response


class BrokenChunkedTransport:
    def get(self, url, *, headers, timeout_seconds):
        raise IncompleteRead(b"partial response")


def enricher(response: HttpResponse) -> tuple[NewsDocumentEnricher, FakeTransport]:
    transport = FakeTransport(response)
    return (
        NewsDocumentEnricher(
            transport=transport,
            resolve_host=lambda _: ("93.184.216.34",),
        ),
        transport,
    )


class NewsDocumentEnricherTests(unittest.TestCase):
    def test_extracts_verbatim_article_body_from_json_ld(self) -> None:
        body = (
            "The company tested a photonic inference accelerator in two data centers. "
            "The pilot reduced latency while keeping model quality unchanged."
        )
        html = f"""
        <html><head><script type="application/ld+json">
        {{"@type":"NewsArticle","articleBody":{body!r}}}
        </script></head><body><p>Navigation text that should not win.</p></body></html>
        """.replace("'", '"').encode()
        service, _ = enricher(
            HttpResponse(200, {"Content-Type": "text/html; charset=utf-8"}, html)
        )

        result = service.enrich(news_document())

        self.assertIs(result.status, NewsContentStatus.ARTICLE_TEXT)
        self.assertEqual(result.document.excerpt, body)
        self.assertEqual(result.excerpt_source, "json_ld.articleBody")
        self.assertTrue(result.candidate_text_available)
        self.assertTrue(result.evidence_text_available)
        self.assertIsNotNone(result.content_sha256)
        json.dumps(result.to_dict())

    def test_uses_article_paragraphs_then_meta_description(self) -> None:
        paragraph = (
            "Engineers described a new optical interconnect and reported the first "
            "prototype measurements from an independent laboratory."
        )
        html = f"<html><body><article><p>{paragraph}</p></article></body></html>".encode()
        service, _ = enricher(HttpResponse(200, {"content-type": "text/html"}, html))

        result = service.enrich(news_document())

        self.assertEqual(result.document.excerpt, paragraph)
        self.assertEqual(result.excerpt_source, "html.article_paragraphs")

        description = (
            "A startup announced a limited pilot of a photonic inference accelerator "
            "for energy-efficient language model serving."
        )
        meta_html = (
            f'<html><head><meta property="og:description" content="{description}"></head></html>'
        ).encode()
        service, _ = enricher(
            HttpResponse(200, {"content-type": "text/html"}, meta_html)
        )

        meta_result = service.enrich(news_document())

        self.assertIs(meta_result.status, NewsContentStatus.META_DESCRIPTION)
        self.assertEqual(meta_result.document.excerpt, description)
        self.assertFalse(meta_result.evidence_text_available)

    def test_keeps_title_only_and_records_fetch_problem(self) -> None:
        service, _ = enricher(
            HttpResponse(403, {"content-type": "text/html"}, b"blocked")
        )

        result = service.enrich(news_document())

        self.assertIs(result.status, NewsContentStatus.TITLE_ONLY)
        self.assertIs(result.issue_code, NewsEnrichmentIssueCode.HTTP_ERROR)
        self.assertIsNone(result.document.excerpt)
        self.assertTrue(result.candidate_text_available)
        self.assertFalse(result.supplemental_text_available)

    def test_incomplete_chunked_response_becomes_network_error(self) -> None:
        service = NewsDocumentEnricher(
            transport=BrokenChunkedTransport(),
            resolve_host=lambda _: ("93.184.216.34",),
        )

        result = service.enrich(news_document())

        self.assertIs(result.status, NewsContentStatus.TITLE_ONLY)
        self.assertIs(result.issue_code, NewsEnrichmentIssueCode.NETWORK_ERROR)
        self.assertIsNone(result.http_status)

    def test_blocks_non_public_urls_before_request(self) -> None:
        transport = FakeTransport(HttpResponse(200, {}, b"unused"))
        service = NewsDocumentEnricher(
            transport=transport,
            resolve_host=lambda _: ("127.0.0.1",),
        )

        result = service.enrich(news_document(url="http://localhost/story"))

        self.assertIs(result.issue_code, NewsEnrichmentIssueCode.BLOCKED_URL)
        self.assertEqual(transport.calls, [])

    def test_rejects_oversized_page_and_skips_existing_excerpt(self) -> None:
        service, transport = enricher(
            HttpResponse(200, {"content-type": "text/html"}, b"x" * (MAX_NEWS_PAGE_BYTES + 1))
        )
        oversized = service.enrich(news_document())

        self.assertIs(
            oversized.issue_code,
            NewsEnrichmentIssueCode.RESPONSE_TOO_LARGE,
        )

        existing = service.enrich(news_document(excerpt="Provider supplied description"))

        self.assertIs(existing.status, NewsContentStatus.EXISTING_EXCERPT)
        self.assertEqual(len(transport.calls), 1)

    def test_enrich_many_preserves_input_order(self) -> None:
        paragraph = (
            b"<article><p>This article contains enough concrete technical content "
            b"to become an excerpt for candidate discovery and later review.</p></article>"
        )
        service, transport = enricher(
            HttpResponse(200, {"content-type": "text/html"}, paragraph)
        )
        documents = tuple(numbered_news_document(number) for number in range(10))

        results = service.enrich_many(documents, max_concurrency=4)

        self.assertEqual(
            [result.document.document_id for result in results],
            [document.document_id for document in documents],
        )
        self.assertEqual(len(transport.calls), 10)

    def test_http_status_follows_response_or_absence_of_request(self) -> None:
        paragraph = (
            b"<article><p>This article contains enough concrete technical content "
            b"to become an excerpt for candidate discovery and later review.</p></article>"
        )
        service, _ = enricher(
            HttpResponse(200, {"content-type": "text/html"}, paragraph)
        )
        self.assertEqual(service.enrich(news_document()).http_status, 200)

        missing, _ = enricher(HttpResponse(404, {}, b"missing"))
        failed = missing.enrich(news_document())
        self.assertEqual(failed.http_status, 404)
        self.assertEqual(failed.to_dict()["http_status"], 404)

        blocked_transport = FakeTransport(HttpResponse(200, {}, b"unused"))
        blocked = NewsDocumentEnricher(
            transport=blocked_transport,
            resolve_host=lambda _: ("127.0.0.1",),
        )
        self.assertIsNone(
            blocked.enrich(news_document(url="http://localhost/story")).http_status
        )

        existing, _ = enricher(HttpResponse(200, {}, b"unused"))
        self.assertIsNone(
            existing.enrich(news_document(excerpt="Provider supplied description"))
            .http_status
        )


if __name__ == "__main__":
    unittest.main()
