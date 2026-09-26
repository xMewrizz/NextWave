"""Resumable media page fetch tests: fake transport and sleeper only, no network."""

from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor as RealPool
from pathlib import Path
from unittest import mock

from nextwave.__main__ import main
from nextwave.labeling.media_fetch_run import (
    LABELING_MEDIA_FETCH_EXECUTOR_VERSION,
    LABELING_MEDIA_FETCH_RESULT_VERSION,
    run_media_fetch,
)
from nextwave.sources import (
    HttpResponse,
    NewsDocumentEnricher,
)

BUNDLE = "bundle-test-001"
SECRET_LIKE = "secret-xyz-123"


def doc_row(candidate: str, number: int, **overrides) -> dict:
    url = f"https://news.example/{candidate}-{number}"
    payload = {
        "candidate_id": candidate,
        "document_id": f"document-{candidate}-{number}",
        "connector": "mediacloud",
        "connector_id": "mediacloud",
        "external_id": f"story-{candidate}-{number}",
        "snapshot_id": "snapshot-1",
        "title": f"News story {candidate} {number}",
        "url": url,
        "canonical_url": url,
        "source_type": "industry_media",
        "language": "en",
        "trust_tier": "unknown",
        "origin_id": f"url:{url}",
        "doi": None,
        "authors": [],
        "organizations": [],
        "published_at": "2026-02-01",
        "observed_at": None,
        "retrieved_at": "2026-09-26T10:00:00+00:00",
        "publisher": "news.example",
        "excerpt": None,
        "automatic_translation": False,
        "generated_summary": False,
        "origin_method": "canonical_url",
        "origin_confidence": 1.0,
        "search_id": f"search-{candidate}",
        "request_id": f"request-{candidate}",
    }
    payload.update(overrides)
    return payload


def queue_item(document: dict, rank: int, **overrides) -> dict:
    payload = {
        "candidate_id": document["candidate_id"],
        "document_id": document["document_id"],
        "connector": "mediacloud",
        "search_id": document["search_id"],
        "request_id": document["request_id"],
        "selection_rank": rank,
        "url": document["url"],
        "document_identity": document["canonical_url"],
        "relevance_class": "weak",
    }
    payload.update(overrides)
    return payload


def digest(payload: bytes) -> dict:
    return {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def dump_jsonl(rows: list[dict]) -> bytes:
    return (
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    ).encode("utf-8")


def build_directories(
    root: Path,
    documents: list[dict],
    queue: list[dict],
    *,
    relevance_mutate=None,
    result_mutate=None,
) -> tuple[Path, Path]:
    result_dir = root / "result"
    result_dir.mkdir(parents=True)
    documents_bytes = dump_jsonl(documents)
    coverage_bytes = b'{"unrelated": true}\n'
    requests_bytes = b'{"unrelated": true}\n'
    (result_dir / "documents.jsonl").write_bytes(documents_bytes)
    (result_dir / "coverage.jsonl").write_bytes(coverage_bytes)
    (result_dir / "request_results.jsonl").write_bytes(requests_bytes)
    result_manifest = {
        "schema_version": "labeling-enrichment-result-v2",
        "bundle_id": BUNDLE,
        "outputs": {
            "coverage.jsonl": digest(coverage_bytes),
            "documents.jsonl": digest(documents_bytes),
            "request_results.jsonl": digest(requests_bytes),
        },
    }
    if result_mutate is not None:
        result_mutate(result_manifest)
    result_manifest_bytes = (
        json.dumps(result_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    (result_dir / "manifest.json").write_bytes(result_manifest_bytes)

    relevance_dir = root / "relevance"
    relevance_dir.mkdir(parents=True)
    queue_bytes = dump_jsonl(queue)
    ranked_bytes = b'{"unrelated": true}\n'
    shortlist_bytes = b'{"unrelated": true}\n'
    rel_coverage_bytes = b'{"unrelated": true}\n'
    (relevance_dir / "media_fetch_queue.jsonl").write_bytes(queue_bytes)
    (relevance_dir / "ranked_documents.jsonl").write_bytes(ranked_bytes)
    (relevance_dir / "scientific_shortlist.jsonl").write_bytes(shortlist_bytes)
    (relevance_dir / "coverage.jsonl").write_bytes(rel_coverage_bytes)
    relevance_manifest = {
        "schema_version": "labeling-relevance-plan-v1",
        "bundle_id": BUNDLE,
        "inputs": {
            "result_manifest": digest(result_manifest_bytes),
            "result_files": {
                "coverage.jsonl": digest(coverage_bytes),
                "documents.jsonl": digest(documents_bytes),
                "request_results.jsonl": digest(requests_bytes),
            },
        },
        "totals": {"media_fetch_queue_rows": len(queue)},
        "outputs": {
            "media_fetch_queue.jsonl": digest(queue_bytes),
            "ranked_documents.jsonl": digest(ranked_bytes),
            "scientific_shortlist.jsonl": digest(shortlist_bytes),
            "coverage.jsonl": digest(rel_coverage_bytes),
        },
    }
    if relevance_mutate is not None:
        relevance_mutate(relevance_manifest)
    (relevance_dir / "manifest.json").write_text(
        json.dumps(relevance_manifest, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return relevance_dir, result_dir


ARTICLE_BODY = (
    "The company tested a photonic inference accelerator in two data centers. "
    "The pilot reduced latency while keeping model quality unchanged."
)
PARAGRAPH = (
    "Engineers described a new optical interconnect and reported the first "
    "prototype measurements from an independent laboratory."
)
META_DESCRIPTION = (
    "A startup announced a limited pilot of a photonic inference accelerator "
    "for energy-efficient language model serving."
)


def article_html() -> bytes:
    return (
        "<html><head><script type=\"application/ld+json\">"
        "{\"@type\":\"NewsArticle\",\"articleBody\":" + json.dumps(ARTICLE_BODY) + "}"
        "</script></head><body><p>Short nav.</p></body></html>"
    ).encode()


def paragraphs_html() -> bytes:
    return f"<html><body><article><p>{PARAGRAPH}</p></article></body></html>".encode()


def meta_html() -> bytes:
    return (
        "<html><head>"
        f"<meta property=\"og:description\" content=\"{META_DESCRIPTION}\">"
        "</head><body></body></html>"
    ).encode()


def empty_html() -> bytes:
    return b"<html><body><p>Hi.</p></body></html>"


def ok_response(body: bytes) -> HttpResponse:
    return HttpResponse(200, {"Content-Type": "text/html; charset=utf-8"}, body)


class FakePageTransport:
    """Route URLs to response sequences; never touches the network."""

    def __init__(self, routes: dict[str, list]) -> None:
        self.routes = {url: list(items) for url, items in routes.items()}
        self.calls: list[str] = []

    def get(self, url: str, *, headers, timeout_seconds: float) -> HttpResponse:
        self.calls.append(url)
        items = self.routes.get(url, [ok_response(empty_html())])
        item = items.pop(0) if len(items) > 1 else items[0]
        if isinstance(item, Exception):
            raise item
        return item


class FakeSleeper:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def run_with_fakes(
    relevance: Path,
    result: Path,
    work: Path,
    output: Path,
    transport: FakePageTransport,
    sleeper: FakeSleeper | None = None,
):
    enricher = NewsDocumentEnricher(
        transport=transport,
        resolve_host=lambda _: ("93.184.216.34",),
        timeout_seconds=10.0,
    )
    return run_media_fetch(
        relevance_dir=relevance,
        result_dir=result,
        work_dir=work,
        output_dir=output,
        page_enricher=enricher,
        sleeper=sleeper or FakeSleeper(),
    )


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def two_candidate_fixture() -> tuple[list[dict], list[dict]]:
    docs = [doc_row("a", 1), doc_row("a", 2), doc_row("b", 1)]
    queue = [
        queue_item(docs[0], 1),
        queue_item(docs[1], 2),
        queue_item(docs[2], 1),
    ]
    return docs, queue


class MediaFetchVersionTests(unittest.TestCase):
    def test_version_constants(self) -> None:
        self.assertEqual(
            LABELING_MEDIA_FETCH_EXECUTOR_VERSION, "labeling-media-fetch-executor-v1"
        )
        self.assertEqual(
            LABELING_MEDIA_FETCH_RESULT_VERSION, "labeling-media-fetch-result-v1"
        )


class MediaFetchContentTests(unittest.TestCase):
    def test_json_ld_article_body_is_evidence(self) -> None:
        docs, queue = two_candidate_fixture()
        transport = FakePageTransport({docs[0]["url"]: [ok_response(article_html())]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = build_directories(root, [docs[0]], [queue[0]])
            paths = run_with_fakes(
                relevance, result, root / "work", root / "out", transport
            )
            enriched = read_jsonl(paths.enriched_documents)[0]
            pages = read_jsonl(paths.page_results)[0]

        self.assertEqual(enriched["content_status"], "article_text")
        self.assertTrue(enriched["evidence_text_available"])
        self.assertEqual(enriched["excerpt"], ARTICLE_BODY)
        self.assertEqual(enriched["http_status"], 200)
        self.assertEqual(pages["status"], "success")
        self.assertEqual(pages["http_status"], 200)

    def test_article_paragraphs_are_evidence(self) -> None:
        docs, queue = two_candidate_fixture()
        transport = FakePageTransport({docs[0]["url"]: [ok_response(paragraphs_html())]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = build_directories(root, [docs[0]], [queue[0]])
            paths = run_with_fakes(
                relevance, result, root / "work", root / "out", transport
            )
            enriched = read_jsonl(paths.enriched_documents)[0]

        self.assertEqual(enriched["content_status"], "article_text")
        self.assertTrue(enriched["evidence_text_available"])

    def test_meta_description_is_not_evidence(self) -> None:
        docs, queue = two_candidate_fixture()
        transport = FakePageTransport({docs[0]["url"]: [ok_response(meta_html())]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = build_directories(root, [docs[0]], [queue[0]])
            paths = run_with_fakes(
                relevance, result, root / "work", root / "out", transport
            )
            enriched = read_jsonl(paths.enriched_documents)[0]

        self.assertEqual(enriched["content_status"], "meta_description")
        self.assertFalse(enriched["evidence_text_available"])
        self.assertEqual(enriched["excerpt"], META_DESCRIPTION)

    def test_empty_page_is_title_only_without_evidence(self) -> None:
        docs, queue = two_candidate_fixture()
        transport = FakePageTransport({docs[0]["url"]: [ok_response(empty_html())]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = build_directories(root, [docs[0]], [queue[0]])
            paths = run_with_fakes(
                relevance, result, root / "work", root / "out", transport
            )
            enriched = read_jsonl(paths.enriched_documents)[0]

        self.assertEqual(enriched["content_status"], "title_only")
        self.assertFalse(enriched["evidence_text_available"])
        self.assertIsNone(enriched["excerpt"])


class MediaFetchRetryTests(unittest.TestCase):
    def test_http_404_is_single_terminal_attempt(self) -> None:
        docs, queue = two_candidate_fixture()
        transport = FakePageTransport(
            {docs[0]["url"]: [HttpResponse(404, {}, b"missing")]}
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = build_directories(root, [docs[0]], [queue[0]])
            paths = run_with_fakes(
                relevance, result, root / "work", root / "out", transport
            )
            pages = read_jsonl(paths.page_results)[0]

        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(pages["status"], "success")
        self.assertEqual(pages["content_status"], "title_only")
        self.assertEqual(pages["issue_code"], "http_error")
        self.assertEqual(pages["http_status"], 404)
        self.assertEqual(pages["attempts_count"], 1)
        self.assertFalse(pages["retryable_exhausted"])

    def test_http_429_is_retried(self) -> None:
        docs, queue = two_candidate_fixture()
        transport = FakePageTransport({
            docs[0]["url"]: [
                HttpResponse(429, {}, b"slow"),
                ok_response(article_html()),
            ]
        })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = build_directories(root, [docs[0]], [queue[0]])
            paths = run_with_fakes(
                relevance, result, root / "work", root / "out", transport
            )
            pages = read_jsonl(paths.page_results)[0]

        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(pages["status"], "success")
        self.assertEqual(pages["content_status"], "article_text")
        self.assertEqual(pages["attempts_count"], 2)

    def test_http_500_is_retried(self) -> None:
        docs, queue = two_candidate_fixture()
        transport = FakePageTransport({
            docs[0]["url"]: [
                HttpResponse(500, {}, b"boom"),
                ok_response(article_html()),
            ]
        })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = build_directories(root, [docs[0]], [queue[0]])
            paths = run_with_fakes(
                relevance, result, root / "work", root / "out", transport
            )
            pages = read_jsonl(paths.page_results)[0]

        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(pages["status"], "success")
        self.assertEqual(pages["attempts_count"], 2)

    def test_network_error_is_retried(self) -> None:
        docs, queue = two_candidate_fixture()
        transport = FakePageTransport({
            docs[0]["url"]: [TimeoutError("timed out"), ok_response(article_html())]
        })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = build_directories(root, [docs[0]], [queue[0]])
            paths = run_with_fakes(
                relevance, result, root / "work", root / "out", transport
            )
            pages = read_jsonl(paths.page_results)[0]

        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(pages["status"], "success")
        self.assertEqual(pages["issue_code"], None)

    def test_three_retryable_failures_stay_failed_unknown(self) -> None:
        docs, queue = two_candidate_fixture()
        transport = FakePageTransport(
            {docs[0]["url"]: [HttpResponse(500, {}, b"boom")]}
        )
        sleeper = FakeSleeper()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = build_directories(root, [docs[0]], [queue[0]])
            paths = run_with_fakes(
                relevance, result, root / "work", root / "out", transport, sleeper
            )
            pages = read_jsonl(paths.page_results)[0]
            coverage = read_jsonl(paths.coverage)[0]

            self.assertEqual(len(transport.calls), 3)
            self.assertEqual(pages["status"], "failed")
            self.assertEqual(pages["content_status"], "title_only")
            self.assertEqual(pages["issue_code"], "http_error")
            self.assertEqual(pages["http_status"], 500)
            self.assertEqual(pages["attempts_count"], 3)
            self.assertTrue(pages["retryable_exhausted"])
            self.assertEqual(coverage["status"], "unknown")
            self.assertEqual(coverage["failed"], 1)
            self.assertEqual(sleeper.calls, [2.0, 5.0])
            failures = list((root / "work" / "failures").rglob("result.json"))
            self.assertEqual(len(failures), 1)
            self.assertIn("cycle-001", failures[0].as_posix())
            completed = root / "work" / "completed"
            self.assertFalse(completed.is_dir() and any(completed.iterdir()))


class MediaFetchResumeTests(unittest.TestCase):
    def test_resume_reuses_completed_without_transport_calls(self) -> None:
        docs, queue = two_candidate_fixture()
        transport = FakePageTransport({})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = build_directories(root, docs, queue)
            run_with_fakes(relevance, result, root / "work", root / "out-1", transport)
            calls_after_first = len(transport.calls)
            before = {
                path.name: (path / "result.json").read_bytes()
                for path in (root / "work" / "completed").iterdir()
            }
            fresh = FakePageTransport({})
            paths = run_with_fakes(
                relevance, result, root / "work", root / "out-2", fresh
            )
            pages = read_jsonl(paths.page_results)

            self.assertEqual(len(fresh.calls), 0)
            self.assertTrue(pages and all(row["reused"] for row in pages))
            after = {
                path.name: (path / "result.json").read_bytes()
                for path in (root / "work" / "completed").iterdir()
            }
            self.assertEqual(before, after)
            self.assertEqual(calls_after_first, 3)

    def test_resume_retries_only_failed_pages(self) -> None:
        docs, queue = two_candidate_fixture()
        failing_url = docs[1]["url"]
        first = FakePageTransport({failing_url: [HttpResponse(500, {}, b"boom")]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = build_directories(root, docs, queue)
            run_with_fakes(relevance, result, root / "work", root / "out-1", first)
            second = FakePageTransport({})
            paths = run_with_fakes(
                relevance, result, root / "work", root / "out-2", second
            )
            pages = read_jsonl(paths.page_results)

            self.assertEqual(second.calls, [failing_url])
            by_document = {row["document_id"]: row for row in pages}
            self.assertFalse(by_document[docs[1]["document_id"]]["reused"])
            self.assertTrue(by_document[docs[0]["document_id"]]["reused"])
            self.assertTrue(by_document[docs[2]["document_id"]]["reused"])
            cycles = sorted(
                (root / "work" / "failures" / by_document[docs[1]["document_id"]]["fetch_id"])
                .iterdir()
            )
            self.assertEqual([path.name for path in cycles], ["cycle-001"])

    def test_corrupt_completed_cache_is_rejected(self) -> None:
        docs, queue = two_candidate_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = build_directories(root, [docs[0]], [queue[0]])
            run_with_fakes(
                relevance, result, root / "work", root / "out-1", FakePageTransport({})
            )
            completed = next((root / "work" / "completed").iterdir())
            (completed / "result.json").write_bytes(b"corrupt{{")
            with self.assertRaises(ValueError):
                run_with_fakes(
                    relevance,
                    result,
                    root / "work",
                    root / "out-2",
                    FakePageTransport({}),
                )

    def test_foreign_work_is_rejected_before_network(self) -> None:
        docs, queue = two_candidate_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = build_directories(root, docs, queue)
            other_docs = [docs[0]]
            other_queue = [queue[0]]
            other_relevance, _ = build_directories(root / "other", other_docs, other_queue)
            # Point the foreign relevance at the same result so only the
            # queue (and its manifest digests) differs.
            result_manifest_bytes = (result / "manifest.json").read_bytes()
            foreign_manifest = json.loads(
                (other_relevance / "manifest.json").read_text(encoding="utf-8")
            )
            foreign_manifest["inputs"]["result_manifest"] = digest(result_manifest_bytes)
            for name in ("coverage.jsonl", "documents.jsonl", "request_results.jsonl"):
                foreign_manifest["inputs"]["result_files"][name] = digest(
                    (result / name).read_bytes()
                )
            (other_relevance / "manifest.json").write_text(
                json.dumps(foreign_manifest, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            run_with_fakes(
                other_relevance, result, root / "work", root / "other-out",
                FakePageTransport({}),
            )
            transport = FakePageTransport({})
            with self.assertRaisesRegex(ValueError, "different inputs"):
                run_with_fakes(
                    relevance, result, root / "work", root / "other-out-2", transport
                )
            self.assertEqual(transport.calls, [])

    def test_orphan_staging_does_not_block_resume(self) -> None:
        docs, queue = two_candidate_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = build_directories(root, docs, queue)
            run_with_fakes(
                relevance, result, root / "work", root / "out-1", FakePageTransport({})
            )
            orphan = root / "work" / ".orphan-123.staging"
            orphan.mkdir()
            (orphan / "junk.txt").write_text("junk", encoding="utf-8")
            paths = run_with_fakes(
                relevance, result, root / "work", root / "out-2", FakePageTransport({})
            )
            pages = read_jsonl(paths.page_results)

            self.assertTrue(all(row["reused"] for row in pages))


def tamper_completed(work: Path, mutate) -> str:
    """Rewrite a completed result.json and repair its cache digest."""

    completed = next((work / "completed").iterdir())
    result_path = completed / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    mutate(result)
    result_bytes = (
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    result_path.write_bytes(result_bytes)
    cache_path = completed / "cache_manifest.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    cache["result"] = {
        "sha256": hashlib.sha256(result_bytes).hexdigest(),
        "size_bytes": len(result_bytes),
    }
    cache_path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return completed.name


def set_both(result: dict, name: str, value) -> None:
    result[name] = value
    result["enrichment"][name] = value


class MediaFetchCacheTamperTests(unittest.TestCase):
    def _run_once(self, root: Path, routes: dict) -> tuple[Path, Path]:
        docs, queue = two_candidate_fixture()
        relevance, result = build_directories(root, [docs[0]], [queue[0]])
        run_with_fakes(
            relevance, result, root / "work", root / "out-1", FakePageTransport(routes)
        )
        return relevance, result

    def _resume_rejected(self, root, relevance, result, pattern: str) -> None:
        transport = FakePageTransport({})
        with self.assertRaisesRegex(ValueError, pattern):
            run_with_fakes(
                relevance, result, root / "work", root / "out-2", transport
            )
        self.assertEqual(transport.calls, [])
        self.assertFalse((root / "out-2").exists())

    def test_nested_document_url_swap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = self._run_once(root, {})
            tamper_completed(
                root / "work",
                lambda item: item["enrichment"]["document"].update(
                    url="https://news.example/evil"
                ),
            )
            self._resume_rejected(root, relevance, result, "nested document url mismatch")

    def test_nested_document_id_swap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = self._run_once(root, {})
            tamper_completed(
                root / "work",
                lambda item: item["enrichment"]["document"].update(
                    document_id="document-evil-1"
                ),
            )
            self._resume_rejected(
                root, relevance, result, "nested document document_id mismatch"
            )

    def test_string_http_status_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = self._run_once(root, {})
            tamper_completed(
                root / "work", lambda item: set_both(item, "http_status", "200")
            )
            self._resume_rejected(root, relevance, result, "http_status is invalid")

    def test_string_truncated_flag_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = self._run_once(root, {})
            tamper_completed(
                root / "work", lambda item: set_both(item, "excerpt_truncated", "false")
            )
            self._resume_rejected(root, relevance, result, "must be a boolean")

    def test_empty_attempts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = self._run_once(root, {})
            tamper_completed(root / "work", lambda item: item.update(attempts=[]))
            self._resume_rejected(root, relevance, result, "attempts must list")

    def test_misnumbered_attempt_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = self._run_once(root, {})
            tamper_completed(
                root / "work", lambda item: item["attempts"][0].update(attempt=2)
            )
            self._resume_rejected(root, relevance, result, "attempts must run 1")

    def test_article_with_false_evidence_is_rejected(self) -> None:
        docs, _ = two_candidate_fixture()
        routes = {docs[0]["url"]: [ok_response(article_html())]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = self._run_once(root, routes)
            tamper_completed(
                root / "work", lambda item: item.update(evidence_text_available=False)
            )
            self._resume_rejected(root, relevance, result, "evidence flag disagrees")

    def test_article_with_http_404_is_rejected(self) -> None:
        docs, _ = two_candidate_fixture()
        routes = {docs[0]["url"]: [ok_response(article_html())]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = self._run_once(root, routes)
            tamper_completed(
                root / "work", lambda item: set_both(item, "http_status", 404)
            )
            self._resume_rejected(root, relevance, result, "clean HTTP 200")

    def test_meta_with_true_evidence_is_rejected(self) -> None:
        docs, _ = two_candidate_fixture()
        routes = {docs[0]["url"]: [ok_response(meta_html())]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = self._run_once(root, routes)
            tamper_completed(
                root / "work", lambda item: item.update(evidence_text_available=True)
            )
            self._resume_rejected(root, relevance, result, "evidence flag disagrees")

    def test_completed_network_error_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = self._run_once(root, {})
            tamper_completed(
                root / "work",
                lambda item: set_both(item, "issue_code", "network_error"),
            )
            self._resume_rejected(root, relevance, result, "retryable network error")

    def test_diverging_content_status_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = self._run_once(root, {})
            tamper_completed(
                root / "work",
                lambda item: item.update(
                    content_status="article_text",
                    evidence_text_available=True,
                    excerpt_source="html.article_paragraphs",
                ),
            )
            self._resume_rejected(
                root, relevance, result, "content_status disagrees with enrichment"
            )


class MediaFetchValidationTests(unittest.TestCase):
    def test_unknown_queue_document_is_rejected(self) -> None:
        docs, queue = two_candidate_fixture()
        ghost = queue_item(docs[0], 1, document_id="document-ghost-1")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = build_directories(root, docs, [ghost])
            with self.assertRaisesRegex(ValueError, "unknown candidate/document"):
                run_with_fakes(
                    relevance, result, root / "work", root / "out", FakePageTransport({})
                )
            self.assertFalse((root / "work").exists())
            self.assertFalse((root / "out").exists())

    def test_queue_url_mismatch_is_rejected(self) -> None:
        docs, queue = two_candidate_fixture()
        tampered = queue_item(docs[0], 1, url="https://news.example/elsewhere")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = build_directories(root, docs, [tampered])
            with self.assertRaises(ValueError):
                run_with_fakes(
                    relevance, result, root / "work", root / "out", FakePageTransport({})
                )
            self.assertFalse((root / "work").exists())

    def test_queue_checksum_mismatch_is_rejected(self) -> None:
        docs, queue = two_candidate_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = build_directories(root, docs, queue)
            with (relevance / "media_fetch_queue.jsonl").open("ab") as handle:
                handle.write(b" ")
            with self.assertRaisesRegex(ValueError, "mismatch with manifest"):
                run_with_fakes(
                    relevance, result, root / "work", root / "out", FakePageTransport({})
                )
            self.assertFalse((root / "work").exists())

    def test_existing_output_fails_before_work_and_network(self) -> None:
        docs, queue = two_candidate_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = build_directories(root, docs, queue)
            (root / "out").mkdir()
            transport = FakePageTransport({})
            with self.assertRaisesRegex(ValueError, "already exists"):
                run_with_fakes(
                    relevance, result, root / "work", root / "out", transport
                )
            self.assertFalse((root / "work").exists())
            self.assertEqual(transport.calls, [])

    def test_gdelt_queue_row_is_rejected(self) -> None:
        docs, queue = two_candidate_fixture()
        gdelt = queue_item(docs[0], 1, connector="gdelt")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = build_directories(root, docs, [gdelt])
            with self.assertRaisesRegex(ValueError, "mediacloud"):
                run_with_fakes(
                    relevance, result, root / "work", root / "out", FakePageTransport({})
                )

    def test_invalid_source_type_creates_no_work_and_no_calls(self) -> None:
        docs, queue = two_candidate_fixture()
        bad = doc_row("a", 1, source_type="blog_post")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = build_directories(root, [bad], [queue[0]])
            transport = FakePageTransport({})
            with self.assertRaisesRegex(ValueError, "unknown source_type"):
                run_with_fakes(
                    relevance, result, root / "work", root / "out", transport
                )
            self.assertFalse((root / "work").exists())
            self.assertFalse((root / "out").exists())
            self.assertEqual(transport.calls, [])

    def test_naive_datetime_creates_no_work(self) -> None:
        docs, queue = two_candidate_fixture()
        bad = doc_row("a", 1, retrieved_at="2026-09-26T10:00:00")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = build_directories(root, [bad], [queue[0]])
            transport = FakePageTransport({})
            with self.assertRaisesRegex(ValueError, "must include a timezone"):
                run_with_fakes(
                    relevance, result, root / "work", root / "out", transport
                )
            self.assertFalse((root / "work").exists())
            self.assertEqual(transport.calls, [])

    def test_duplicate_document_rows_are_rejected_with_line(self) -> None:
        docs, queue = two_candidate_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = build_directories(
                root, [docs[0], docs[0]], [queue[0]]
            )
            transport = FakePageTransport({})
            with self.assertRaisesRegex(ValueError, "line 2.*duplicates"):
                run_with_fakes(
                    relevance, result, root / "work", root / "out", transport
                )
            self.assertFalse((root / "work").exists())
            self.assertEqual(transport.calls, [])

    def test_duplicate_selection_rank_is_rejected(self) -> None:
        docs, queue = two_candidate_fixture()
        doubled = [queue_item(docs[0], 1), queue_item(docs[1], 1)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = build_directories(root, docs[:2], doubled)
            transport = FakePageTransport({})
            with self.assertRaisesRegex(ValueError, "1\\.\\.N"):
                run_with_fakes(
                    relevance, result, root / "work", root / "out", transport
                )
            self.assertFalse((root / "work").exists())
            self.assertEqual(transport.calls, [])

    def test_rank_gap_is_rejected(self) -> None:
        docs, queue = two_candidate_fixture()
        gapped = [queue_item(docs[0], 1), queue_item(docs[1], 3)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = build_directories(root, docs[:2], gapped)
            transport = FakePageTransport({})
            with self.assertRaisesRegex(ValueError, "1\\.\\.N"):
                run_with_fakes(
                    relevance, result, root / "work", root / "out", transport
                )
            self.assertFalse((root / "work").exists())
            self.assertEqual(transport.calls, [])

    def test_fetch_id_collision_is_rejected_before_work(self) -> None:
        docs, queue = two_candidate_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = build_directories(root, docs, queue)
            transport = FakePageTransport({})
            with mock.patch(
                "nextwave.labeling.media_fetch_run._fetch_id",
                return_value="fetch-collision",
            ):
                with self.assertRaisesRegex(ValueError, "duplicate fetch_id"):
                    run_with_fakes(
                        relevance, result, root / "work", root / "out", transport
                    )
            self.assertFalse((root / "work").exists())
            self.assertFalse((root / "out").exists())
            self.assertEqual(transport.calls, [])

    def test_permuted_inputs_give_identical_jsonl(self) -> None:
        # fetch_id embeds the input manifest digests, so page rows are
        # compared with fetch_id stripped; content and order must still match.
        docs, queue = two_candidate_fixture()
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            relevance_a, result_a = build_directories(first, docs, queue)
            run_with_fakes(
                relevance_a, result_a, first / "work", first / "out",
                FakePageTransport({}),
            )
            second = Path(directory) / "second"
            relevance_b, result_b = build_directories(
                second, list(reversed(docs)), list(reversed(queue))
            )
            run_with_fakes(
                relevance_b, result_b, second / "work", second / "out",
                FakePageTransport({}),
            )
            for name in ("enriched_documents.jsonl", "coverage.jsonl"):
                self.assertEqual(
                    (first / "out" / name).read_bytes(),
                    (second / "out" / name).read_bytes(),
                    name,
                )
            pages_a = read_jsonl(first / "out" / "page_results.jsonl")
            pages_b = read_jsonl(second / "out" / "page_results.jsonl")

            def without_fetch_id(rows: list[dict]) -> list[dict]:
                return [
                    {key: value for key, value in row.items() if key != "fetch_id"}
                    for row in rows
                ]

            self.assertEqual(without_fetch_id(pages_a), without_fetch_id(pages_b))
            self.assertEqual(len(pages_a), len(pages_b))

    def test_extra_queue_fields_do_not_change_spec_digest(self) -> None:
        # Unknown queue fields must not affect the spec digest (fetch_id
        # still embeds the changed input manifest digest by design).
        docs, queue = two_candidate_fixture()
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            relevance_a, result_a = build_directories(first, docs, queue)
            run_with_fakes(
                relevance_a, result_a, first / "work", first / "out",
                FakePageTransport({}),
            )
            second = Path(directory) / "second"
            extra = [dict(item, future_field="x") for item in queue]
            relevance_b, result_b = build_directories(second, docs, extra)
            run_with_fakes(
                relevance_b, result_b, second / "work", second / "out",
                FakePageTransport({}),
            )

            def spec_map(work: Path) -> dict[str, str]:
                mapping = {}
                for completed in (work / "completed").iterdir():
                    cache = json.loads(
                        (completed / "cache_manifest.json").read_text(encoding="utf-8")
                    )
                    mapping[cache["document_id"]] = cache["spec_digest"]
                return mapping

            self.assertEqual(
                spec_map(first / "work"), spec_map(second / "work")
            )

    def test_secret_like_transport_error_stays_out_of_output(self) -> None:
        docs, queue = two_candidate_fixture()
        transport = FakePageTransport({
            docs[0]["url"]: [
                RuntimeError(
                    f"headers={{'Authorization': 'Bearer {SECRET_LIKE}'}} cookies=x"
                )
            ]
        })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = build_directories(root, [docs[0]], [queue[0]])
            paths = run_with_fakes(
                relevance, result, root / "work", root / "out", transport
            )
            blob = b"".join(
                path.read_bytes()
                for path in sorted((root / "out").rglob("*"))
                if path.is_file()
            )
            pages = read_jsonl(paths.page_results)[0]

            self.assertNotIn(SECRET_LIKE.encode(), blob)
            self.assertEqual(pages["status"], "failed")
            self.assertNotIn(SECRET_LIKE, pages["note"])

    def test_no_headers_or_cookies_in_output(self) -> None:
        docs, queue = two_candidate_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = build_directories(root, docs, queue)
            paths = run_with_fakes(
                relevance, result, root / "work", root / "out", FakePageTransport({})
            )
            for name in (
                "page_results.jsonl",
                "enriched_documents.jsonl",
                "coverage.jsonl",
                "manifest.json",
            ):
                if name == "manifest.json":
                    rows = [
                        json.loads(paths.manifest.read_text(encoding="utf-8"))
                    ]
                else:
                    rows = read_jsonl(paths.manifest.parent / name)
                for row in rows:
                    stack = [row]
                    while stack:
                        item = stack.pop()
                        if isinstance(item, dict):
                            for key, value in item.items():
                                folded = str(key).casefold()
                                self.assertNotIn("header", folded)
                                self.assertNotIn("cookie", folded)
                                stack.append(value)
                        elif isinstance(item, list):
                            stack.extend(item)

    def test_only_queue_page_urls_are_fetched(self) -> None:
        docs, queue = two_candidate_fixture()
        transport = FakePageTransport({})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = build_directories(root, docs, queue)
            run_with_fakes(
                relevance, result, root / "work", root / "out", transport
            )

        allowed = {item["url"] for item in queue}
        self.assertTrue(transport.calls)
        self.assertEqual(set(transport.calls), allowed)
        for url in transport.calls:
            self.assertNotIn("api.openalex.org", url)
            self.assertNotIn("mediacloud", url)

    def test_cli_error_returns_1_without_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs, queue = two_candidate_fixture()
            relevance, result = build_directories(root, docs, queue)
            (relevance / "media_fetch_queue.jsonl").write_bytes(b"corrupt{{")
            exit_code = main([
                "labeling-media-fetch-run",
                "--relevance", str(relevance),
                "--result", str(result),
                "--work", str(root / "work"),
                "--output", str(root / "out"),
            ])

            self.assertEqual(exit_code, 1)
            self.assertFalse((root / "out").exists())
            self.assertFalse((root / "work").exists())


class MediaFetchConcurrencyTests(unittest.TestCase):
    def test_max_concurrency_is_six(self) -> None:
        docs, queue = two_candidate_fixture()
        seen: dict[str, int] = {}
        real_pool = RealPool

        def factory(*args, **kwargs):
            seen.update(kwargs)
            return real_pool(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = build_directories(root, docs, queue)
            with mock.patch(
                "nextwave.labeling.media_fetch_run.ThreadPoolExecutor", factory
            ):
                run_with_fakes(
                    relevance, result, root / "work", root / "out",
                    FakePageTransport({}),
                )

        self.assertEqual(seen.get("max_workers"), 6)

    def test_completed_pages_publish_one_by_one(self) -> None:
        docs, queue = two_candidate_fixture()
        gate = threading.Event()

        class GatedTransport(FakePageTransport):
            def get(self, url: str, *, headers, timeout_seconds: float) -> HttpResponse:
                if url == docs[0]["url"]:
                    self.calls.append(url)
                    if not gate.wait(timeout=15):
                        raise TimeoutError("test gate timed out")
                    return ok_response(article_html())
                return super().get(url, headers=headers, timeout_seconds=timeout_seconds)

        transport = GatedTransport({})
        errors: list[BaseException] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relevance, result = build_directories(root, docs, queue)

            def target() -> None:
                try:
                    run_with_fakes(
                        relevance, result, root / "work", root / "out", transport
                    )
                except BaseException as error:  # noqa: BLE001 - rethrown to the test
                    errors.append(error)

            worker = threading.Thread(target=target)
            worker.start()
            try:
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    completed = root / "work" / "completed"
                    if completed.is_dir() and len(list(completed.iterdir())) >= 2:
                        break
                    time.sleep(0.05)
                else:
                    self.fail("other pages were not published while one page blocked")
            finally:
                gate.set()
                worker.join(timeout=30)
            self.assertFalse(errors)
            self.assertFalse(worker.is_alive())


if __name__ == "__main__":
    unittest.main()
