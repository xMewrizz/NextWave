"""Resumable enrichment executor tests: fake transport, clock and sleeper only."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path
from unittest import mock

from nextwave.__main__ import main
from nextwave.labeling import enrichment_run as enrichment_run_module
from nextwave.labeling.enrichment_plan import export_enrichment_plan
from nextwave.labeling.enrichment_run import _deduplicate_document_rows, run_enrichment
from nextwave.sources import (
    HttpResponse,
    MediaCloudConnector,
    OpenAlexConnector,
)

CUTOFF = "2026-09-15"
FAKE_KEY = "test-key-abc-123"
FAKE_FOLDER = "test-folder-9"
FAKE_MAILTO = "test@example.org"
FIXED_NOW = datetime(2026, 9, 20, 12, 0, 0, tzinfo=UTC)


class DocumentDeduplicationTests(unittest.TestCase):
    def test_same_document_from_two_requests_keeps_stable_provenance(self) -> None:
        later = {
            "candidate_id": "team-negative-001",
            "document_id": "document-openalex-1",
            "title": "Same study",
            "search_id": "search-1",
            "request_id": "request-z",
            "retrieved_at": "2026-09-20T12:00:01+00:00",
        }
        earlier = later | {
            "request_id": "request-a",
            "retrieved_at": "2026-09-20T12:00:00+00:00",
        }

        self.assertEqual(
            _deduplicate_document_rows([later, earlier]),
            [earlier],
        )
        self.assertEqual(
            _deduplicate_document_rows([earlier, later]),
            [earlier],
        )

    def test_conflicting_duplicate_document_is_rejected(self) -> None:
        first = {
            "candidate_id": "team-negative-001",
            "document_id": "document-openalex-1",
            "title": "First title",
            "search_id": "search-1",
            "request_id": "request-a",
            "retrieved_at": "2026-09-20T12:00:00+00:00",
        }
        second = first | {"title": "Conflicting title", "request_id": "request-b"}

        with self.assertRaisesRegex(ValueError, "conflicting copies"):
            _deduplicate_document_rows([first, second])


class FakeTransport:
    """Route by marker substrings; never touches the network."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, float]] = []
        self._hits: dict[str, int] = {}

    def get(self, url: str, *, headers, timeout_seconds: float) -> HttpResponse:
        self.calls.append((url, dict(headers), timeout_seconds))
        if "fail-always" in url:
            return HttpResponse(500, {}, b'{"error": true}')
        if "bad-request" in url:
            return HttpResponse(400, {}, b'{"error": true}')
        if "syntax-fallback" in url:
            self._hits["syntax-fallback"] = self._hits.get("syntax-fallback", 0) + 1
            if self._hits["syntax-fallback"] == 1:
                return HttpResponse(400, {}, b'{"error": true}')
        if "corrupt-body" in url:
            return HttpResponse(
                200, {"Content-Type": "application/json"}, b"not json{["
            )
        if "retry-once" in url:
            if sum(1 for call in self.calls if "retry-once" in call[0]) == 1:
                return HttpResponse(429, {}, b'{"error": true}')
        if "timeout-once" in url:
            if sum(1 for call in self.calls if "timeout-once" in call[0]) == 1:
                raise TimeoutError("timed out")
        if "retry-after-45" in url or "retry-after-bad" in url or "retry-after-3600" in url:
            self._hits[url] = self._hits.get(url, 0) + 1
            if self._hits[url] == 1:
                if "retry-after-45" in url:
                    headers = {"Retry-After": "45"}
                elif "retry-after-3600" in url:
                    headers = {"Retry-After": "3600"}
                else:
                    headers = {"Retry-After": "soon"}
                return HttpResponse(429, headers, b'{"error": true}')
        if "partial-parse" in url:
            if "openalex" in url:
                body = {
                    "results": [
                        {
                            "id": "https://openalex.org/W1234567",
                            "title": "Study of enrichment",
                            "publication_date": "2026-01-01",
                        },
                        {"id": "https://openalex.org/W7654321"},
                    ]
                }
            else:
                body = {
                    "stories": [
                        {
                            "id": "story-good",
                            "url": "https://example.org/news/good",
                            "title": "News of enrichment",
                            "publish_date": "2026-01-01",
                            "indexed_date": "2026-01-02T00:00:00Z",
                            "language": "en",
                        },
                        {"id": "story-bad"},
                    ]
                }
            return HttpResponse(
                200, {"Content-Type": "application/json"}, json.dumps(body).encode()
            )
        if "empty-results" in url:
            body = {"results": []} if "openalex" in url else {"stories": []}
            return HttpResponse(
                200, {"Content-Type": "application/json"}, json.dumps(body).encode()
            )
        if "openalex" in url:
            body = {
                "results": [
                    {
                        "id": "https://openalex.org/W1234567",
                        "title": "Study of enrichment",
                        "publication_date": "2026-01-01",
                    }
                ]
            }
        else:
            digest = hashlib.sha256(url.encode()).hexdigest()[:8]
            body = {
                "stories": [
                    {
                        "id": f"story-{digest}",
                        "url": f"https://example.org/news/{digest}",
                        "title": "News of enrichment",
                        "publish_date": "2026-01-01",
                        "indexed_date": "2026-01-02T00:00:00Z",
                        "language": "en",
                    }
                ]
            }
        return HttpResponse(
            200, {"Content-Type": "application/json"}, json.dumps(body).encode()
        )


class ConcurrentFakeTransport(FakeTransport):
    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def get(self, url: str, *, headers, timeout_seconds: float) -> HttpResponse:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.02)
            return super().get(url, headers=headers, timeout_seconds=timeout_seconds)
        finally:
            with self.lock:
                self.active -= 1


class FakeClock:
    def __init__(self) -> None:
        self.now = FIXED_NOW

    def __call__(self) -> datetime:
        current = self.now
        self.now = self.now + timedelta(seconds=1)
        return current


class FakeMonotonic:
    def __init__(self, frozen: float = 1000.0) -> None:
        self.now = frozen

    def __call__(self) -> float:
        return self.now


class FakeSleeper:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def candidate_row(number: int, **overrides) -> dict:
    payload = {
        "schema_version": "negative-candidate-v1",
        "candidate_id": f"team-negative-{number:03d}",
        "canonical_name": f"ok-alpha-{number:03d}",
        "aliases": [],
        "group_id": f"group-{number:03d}",
        "source_query": "Технологии в ИИ",
        "domain": "Edge",
        "analysis_scope_key": "edge-v1",
        "cutoff_date": CUTOFF,
    }
    payload.update(overrides)
    return payload


def write_bundle(root: Path, name: str, rows: list[dict]) -> Path:
    body = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    manifest = {
        "schema_version": "labeling-export-manifest-v1",
        "cutoff_date": CUTOFF,
        "filled_candidates": {"Edge": len(rows)},
        "outputs": {
            "negative_candidates.jsonl": {
                "size_bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }
        },
    }
    bundle = root / name
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (bundle / "negative_candidates.jsonl").write_bytes(body)
    return bundle


def build_plan_output(root: Path, name: str, rows: list[dict]) -> Path:
    bundle = write_bundle(root, f"{name}-bundle", rows)
    output = root / f"{name}-plan"
    export_enrichment_plan(bundle_dir=bundle, output_dir=output)
    return output


def environment() -> dict[str, str]:
    return {
        "NEXTWAVE_MEDIACLOUD_API_KEY": FAKE_KEY,
        "NEXTWAVE_YANDEX_FOLDER_ID": FAKE_FOLDER,
        "NEXTWAVE_OPENALEX_MAILTO": FAKE_MAILTO,
    }


def run_with_fakes(
    plan: Path,
    work: Path,
    output: Path,
    transport: FakeTransport | None = None,
    clock=None,
    monotonic=None,
    sleeper=None,
    openalex_transport=None,
    mediacloud_transport=None,
    env: dict[str, str] | None = None,
    connectors=None,
    concurrency=1,
):
    return run_enrichment(
        plan_dir=plan,
        work_dir=work,
        output_dir=output,
        environment=env if env is not None else environment(),
        openalex_transport=openalex_transport or transport,
        mediacloud_transport=mediacloud_transport or transport,
        clock=clock or FakeClock(),
        monotonic_clock=monotonic or FakeMonotonic(),
        sleeper=sleeper or FakeSleeper(),
        connectors=connectors,
        concurrency=concurrency,
    )


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def scan_bytes(root: Path) -> bytes:
    blob = b""
    for path in sorted(root.rglob("*")):
        if path.is_file():
            blob += path.read_bytes()
    return blob


class EnrichmentRunTests(unittest.TestCase):
    def test_openalex_only_requests_run_concurrently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(root, "parallel", [candidate_row(1)])
            transport = ConcurrentFakeTransport()
            run_with_fakes(
                plan,
                root / "work",
                root / "output",
                openalex_transport=transport,
                connectors=("openalex",),
                concurrency=4,
            )
            self.assertGreater(transport.max_active, 1)

    def test_parallel_mode_rejects_rate_limited_mediacloud(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(root, "parallel-media", [candidate_row(1)])
            with self.assertRaisesRegex(ValueError, "only for OpenAlex"):
                run_with_fakes(
                    plan,
                    root / "work",
                    root / "output",
                    transport=FakeTransport(),
                    concurrency=2,
                )

    def test_openalex_only_skips_mediacloud_without_its_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(root, "run", [candidate_row(1)])
            transport = FakeTransport()
            run_with_fakes(
                plan,
                root / "work",
                root / "out-1",
                transport,
                env={"NEXTWAVE_OPENALEX_MAILTO": FAKE_MAILTO},
                connectors=("openalex",),
            )

            manifest = json.loads(
                (root / "out-1" / "manifest.json").read_text(encoding="utf-8")
            )
            results = read_jsonl(root / "out-1" / "request_results.jsonl")
            coverage = read_jsonl(root / "out-1" / "coverage.jsonl")
            self.assertEqual(manifest["requested_connectors"], ["openalex"])
            self.assertTrue(results)
            self.assertEqual({row["connector"] for row in results}, {"openalex"})
            scientific = next(
                row for row in coverage if row["source_class"] == "scientific"
            )
            industry = next(
                row for row in coverage if row["source_class"] == "industry"
            )
            self.assertEqual(scientific["status"], "complete")
            self.assertEqual(industry["status"], "unknown")
            self.assertFalse(any("mediacloud" in call[0] for call in transport.calls))

    def test_openalex_only_ignores_unselected_mediacloud_query_syntax(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(
                root,
                "quoted-media-term",
                [candidate_row(1, canonical_name='"Buy Now, Pay Later" services')],
            )
            transport = FakeTransport()

            run_with_fakes(
                plan,
                root / "work",
                root / "output",
                transport,
                env={"NEXTWAVE_OPENALEX_MAILTO": FAKE_MAILTO},
                connectors=("openalex",),
            )

            results = read_jsonl(root / "output" / "request_results.jsonl")
            self.assertTrue(results)
            self.assertEqual({row["connector"] for row in results}, {"openalex"})

    def test_all_success_including_empty_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(
                root, "run", [candidate_row(1, aliases=["empty-results"])]
            )
            transport = FakeTransport()
            run_with_fakes(plan, root / "work", root / "out-1", transport)

            coverage = read_jsonl(root / "out-1" / "coverage.jsonl")
            self.assertEqual(len(coverage), 2)
            self.assertTrue(all(row["status"] == "complete" for row in coverage))
            results = read_jsonl(root / "out-1" / "request_results.jsonl")
            empty = [row for row in results if "empty-results" in row["search_text"]]
            self.assertTrue(empty and all(row["status"] == "success" for row in empty))
            self.assertTrue(
                all(row["returned_records"] == 0 for row in empty)
            )

    def test_mixed_success_and_failure_is_partial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(
                root, "run", [candidate_row(1, aliases=["fail-always"])]
            )
            transport = FakeTransport()
            run_with_fakes(plan, root / "work", root / "out-1", transport)

            coverage = read_jsonl(root / "out-1" / "coverage.jsonl")
            self.assertTrue(all(row["status"] == "partial" for row in coverage))
            row = next(r for r in coverage if r["connector"] == "openalex")
            self.assertEqual(row["successful_requests"], 2)
            self.assertEqual(row["failed_requests"], 2)
            self.assertEqual(len(row["failed_request_ids"]), 2)
            self.assertTrue(row["incompleteness_reasons"])
            self.assertIn("http_500", row["incompleteness_reasons"][0])

    def test_all_failed_is_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(
                root, "run", [candidate_row(1, canonical_name="fail-always", aliases=[])]
            )
            transport = FakeTransport()
            run_with_fakes(plan, root / "work", root / "out-1", transport)

            coverage = read_jsonl(root / "out-1" / "coverage.jsonl")
            self.assertTrue(all(row["status"] == "unknown" for row in coverage))
            self.assertTrue(
                all(row["successful_requests"] == 0 for row in coverage)
            )

    def test_rerun_reuses_completed_without_new_transport_calls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(root, "run", [candidate_row(1)])
            transport = FakeTransport()
            run_with_fakes(plan, root / "work", root / "out-1", transport)
            calls_after_first = len(transport.calls)
            run_with_fakes(plan, root / "work", root / "out-2", transport)

            self.assertEqual(len(transport.calls), calls_after_first)
            results = read_jsonl(root / "out-2" / "request_results.jsonl")
            self.assertTrue(results and all(row["reused"] for row in results))
            for first, second in zip(
                read_jsonl(root / "out-1" / "request_results.jsonl"),
                results,
                strict=True,
            ):
                self.assertEqual(first["request_id"], second["request_id"])
                self.assertEqual(first["status"], "success")

    def test_failed_request_retried_while_success_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(
                root, "run", [candidate_row(1, aliases=["fail-always"])]
            )
            transport = FakeTransport()
            run_with_fakes(plan, root / "work", root / "out-1", transport)
            first_fail = sum(1 for call in transport.calls if "fail-always" in call[0])
            first_ok = sum(1 for call in transport.calls if "fail-always" not in call[0])
            run_with_fakes(plan, root / "work", root / "out-2", transport)

            self.assertEqual(
                sum(1 for call in transport.calls if "fail-always" in call[0]),
                first_fail * 2,
            )
            self.assertEqual(
                sum(1 for call in transport.calls if "fail-always" not in call[0]),
                first_ok,
            )

    def test_corrupt_completed_cache_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(root, "run", [candidate_row(1)])
            transport = FakeTransport()
            run_with_fakes(plan, root / "work", root / "out-1", transport)
            plan_data = json.loads((plan / "plan.json").read_text(encoding="utf-8"))
            request_id = plan_data["candidates"][0]["searches"][0]["requests"][0][
                "request_id"
            ]
            raw_files = list(
                (root / "work" / "completed" / request_id / "snapshot").rglob("*.json")
            )
            raw = next(p for p in raw_files if p.name != "manifest.json")
            payload = bytearray(raw.read_bytes())
            payload[len(payload) // 2] ^= 0xFF
            raw.write_bytes(bytes(payload))
            with self.assertRaisesRegex(ValueError, "checksum"):
                run_with_fakes(plan, root / "work", root / "out-2", transport)
            self.assertFalse((root / "out-2").exists())

    def test_work_for_other_plan_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_a = build_plan_output(root, "a", [candidate_row(1)])
            plan_b = build_plan_output(root, "b", [candidate_row(2)])
            transport = FakeTransport()
            run_with_fakes(plan_a, root / "work", root / "out-a", transport)
            with self.assertRaisesRegex(ValueError, "different plan"):
                run_with_fakes(plan_b, root / "work", root / "out-b", transport)
            self.assertFalse((root / "out-b").exists())

    def test_retryable_repeats_and_non_retryable_stops(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(
                root,
                "run",
                [
                    candidate_row(
                        1,
                        aliases=["retry-once", "timeout-once", "bad-request"],
                    )
                ],
            )
            oa_transport, mc_transport = FakeTransport(), FakeTransport()
            run_with_fakes(
                plan, root / "work", root / "out-1", None,
                openalex_transport=oa_transport, mediacloud_transport=mc_transport,
            )

            expected = {
                "openalex": {"retry-once": 3, "timeout-once": 3, "bad-request": 2},
                "mediacloud": {"retry-once": 2, "timeout-once": 2, "bad-request": 2},
            }
            for connector_transport, connector in (
                (oa_transport, "openalex"),
                (mc_transport, "mediacloud"),
            ):
                counts = {
                    marker: sum(
                        1 for call in connector_transport.calls if marker in call[0]
                    )
                    for marker in ("retry-once", "timeout-once", "bad-request")
                }
                self.assertEqual(counts, expected[connector])
            results = {
                row["search_text"]: row
                for row in read_jsonl(root / "out-1" / "request_results.jsonl")
            }
            self.assertEqual(results["retry-once"]["status"], "success")
            self.assertEqual(results["timeout-once"]["status"], "success")
            self.assertEqual(results["bad-request"]["status"], "failed")
            self.assertEqual(results["bad-request"]["error"]["code"], "http_400")

    def test_mediacloud_http_400_retries_with_safe_query_syntax(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unsafe = "syntax-fallback «engineering intelligence»: + ИИ"
            plan = build_plan_output(
                root,
                "run",
                [candidate_row(1, canonical_name=unsafe, aliases=[])],
            )
            oa_transport, mc_transport = FakeTransport(), FakeTransport()

            run_with_fakes(
                plan,
                root / "work",
                root / "out-1",
                None,
                openalex_transport=oa_transport,
                mediacloud_transport=mc_transport,
            )

            calls = [url for url, _, _ in mc_transport.calls]
            self.assertEqual(len(calls), 2)
            self.assertNotEqual(calls[0], calls[1])
            self.assertIn("%C2%AB", calls[0])
            self.assertNotIn("%C2%AB", calls[1])
            result = next(
                row
                for row in read_jsonl(root / "out-1" / "request_results.jsonl")
                if row["connector"] == "mediacloud"
            )
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["attempts"], 2)

    def test_single_mediacloud_connector_keeps_global_interval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(
                root, "run", [candidate_row(1, aliases=["Second"])]
            )
            transport = FakeTransport()
            sleeper = FakeSleeper()
            run_with_fakes(
                plan, root / "work", root / "out-1", transport,
                monotonic=FakeMonotonic(), sleeper=sleeper,
            )
            mediacloud_calls = [c for c in transport.calls if "mediacloud" in c[0]]

            self.assertEqual(len(mediacloud_calls), 2)
            self.assertEqual(sleeper.calls, [30.0])

    def test_rerun_keeps_output_bytes_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(
                root, "run", [candidate_row(1, aliases=["fail-always"])]
            )
            first_transport, second_transport = FakeTransport(), FakeTransport()
            run_with_fakes(plan, root / "work-a", root / "out-a", first_transport)
            run_with_fakes(plan, root / "work-b", root / "out-b", second_transport)

            for name in ("request_results.jsonl", "documents.jsonl", "coverage.jsonl"):
                self.assertEqual(
                    (root / "out-a" / name).read_bytes(),
                    (root / "out-b" / name).read_bytes(),
                )
            transport = FakeTransport()
            run_with_fakes(plan, root / "work", root / "out-1", transport)
            run_with_fakes(plan, root / "work", root / "out-2", transport)

            def normalized(name: str, output: Path) -> list[dict]:
                rows = read_jsonl(output / name)
                for row in rows:
                    row.pop("reused", None)
                    row.pop("reused_requests", None)
                return rows

            for name in ("request_results.jsonl", "documents.jsonl", "coverage.jsonl"):
                self.assertEqual(
                    normalized(name, root / "out-1"), normalized(name, root / "out-2")
                )
            second_reused = read_jsonl(root / "out-2" / "request_results.jsonl")
            self.assertTrue(
                all(
                    row["reused"]
                    for row in second_reused
                    if row["status"] == "success"
                )
            )
            self.assertTrue(
                all(
                    not row["reused"]
                    for row in second_reused
                    if row["status"] != "success"
                )
            )

    def test_existing_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(root, "run", [candidate_row(1)])
            transport = FakeTransport()
            run_with_fakes(plan, root / "work", root / "out-1", transport)
            before = {
                name: (root / "out-1" / name).read_bytes()
                for name in (
                    "manifest.json",
                    "request_results.jsonl",
                    "documents.jsonl",
                    "coverage.jsonl",
                )
            }
            with self.assertRaises(ValueError):
                run_with_fakes(plan, root / "work", root / "out-1", transport)
            for name, payload in before.items():
                self.assertEqual((root / "out-1" / name).read_bytes(), payload)

    def test_error_before_publish_leaves_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(root, "run", [candidate_row(1)])
            (plan / "plan.json").write_bytes(b'{"broken": true}\n')
            env_file = root / "empty.env"
            env_file.write_text("", encoding="utf-8")
            with redirect_stderr(StringIO()):
                exit_code = main(
                    [
                        "labeling-enrichment-run",
                        "--plan",
                        str(plan),
                        "--work",
                        str(root / "work"),
                        "--output",
                        str(root / "out-1"),
                        "--env-file",
                        str(env_file),
                    ]
                )
            self.assertEqual(exit_code, 1)
            self.assertFalse((root / "out-1").exists())

    def test_no_secrets_in_results_or_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(
                root, "run", [candidate_row(1, aliases=["fail-always"])]
            )
            transport = FakeTransport()
            stdout, stderr = StringIO(), StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                run_with_fakes(plan, root / "work", root / "out-1", transport)
            blob = scan_bytes(root / "work") + scan_bytes(root / "out-1")
            blob += stdout.getvalue().encode() + stderr.getvalue().encode()

            for secret in (FAKE_KEY, FAKE_FOLDER, "Authorization", "Api-Key", "Token "):
                self.assertNotIn(secret.encode(), blob)
            self.assertTrue(
                any("Authorization" in headers for _, headers, _ in transport.calls)
            )

    def test_gdelt_is_never_called(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(root, "run", [candidate_row(1)])
            transport = FakeTransport()
            run_with_fakes(plan, root / "work", root / "out-1", transport)
            blob = scan_bytes(root / "work") + scan_bytes(root / "out-1")

            self.assertFalse(any("gdelt" in call[0] for call in transport.calls))
            self.assertNotIn(b"gdelt", blob.lower())

    def test_zero_differs_from_api_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(
                root,
                "run",
                [
                    candidate_row(1, canonical_name="empty-results", aliases=[]),
                    candidate_row(2, canonical_name="fail-always", aliases=[]),
                ],
            )
            transport = FakeTransport()
            run_with_fakes(plan, root / "work", root / "out-1", transport)
            results = {
                row["search_text"]: row
                for row in read_jsonl(root / "out-1" / "request_results.jsonl")
            }

            self.assertEqual(results["empty-results"]["status"], "success")
            self.assertEqual(results["empty-results"]["returned_records"], 0)
            self.assertEqual(results["fail-always"]["status"], "failed")
            self.assertIsNone(results["fail-always"]["returned_records"])

    def test_documents_carry_full_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(root, "run", [candidate_row(1)])
            transport = FakeTransport()
            run_with_fakes(plan, root / "work", root / "out-1", transport)
            documents = read_jsonl(root / "out-1" / "documents.jsonl")

            self.assertTrue(documents)
            for document in documents:
                for field in (
                    "candidate_id",
                    "search_id",
                    "request_id",
                    "connector",
                    "snapshot_id",
                    "document_id",
                    "title",
                    "url",
                ):
                    self.assertTrue(document.get(field), field)

    def test_cli_partial_run_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(
                root, "run", [candidate_row(1, aliases=["fail-always"])]
            )
            oa_fake, mc_fake = FakeTransport(), FakeTransport()
            clock, monotonic, sleeper = FakeClock(), FakeMonotonic(), FakeSleeper()
            env_file = root / "test.env"
            env_file.write_text(
                f"NEXTWAVE_MEDIACLOUD_API_KEY={FAKE_KEY}\n"
                f"NEXTWAVE_YANDEX_FOLDER_ID={FAKE_FOLDER}\n"
                f"NEXTWAVE_OPENALEX_MAILTO={FAKE_MAILTO}\n",
                encoding="utf-8",
            )
            with (
                mock.patch.object(
                    enrichment_run_module,
                    "OpenAlexConnector",
                    lambda **kwargs: OpenAlexConnector(
                        transport=oa_fake,
                        clock=clock,
                        timeout_seconds=kwargs.get("timeout_seconds", 20),
                    ),
                ),
                mock.patch.object(
                    enrichment_run_module,
                    "MediaCloudConnector",
                    lambda **kwargs: MediaCloudConnector(
                        api_key=FAKE_KEY,
                        transport=mc_fake,
                        clock=clock,
                        monotonic_clock=monotonic,
                        sleeper=sleeper,
                        timeout_seconds=kwargs.get("timeout_seconds", 60),
                        min_interval_seconds=kwargs.get("min_interval_seconds", 30),
                    ),
                ),
                mock.patch.dict(
                    os.environ,
                    {
                        "NEXTWAVE_MEDIACLOUD_API_KEY": FAKE_KEY,
                        "NEXTWAVE_YANDEX_FOLDER_ID": FAKE_FOLDER,
                        "NEXTWAVE_OPENALEX_MAILTO": FAKE_MAILTO,
                    },
                ),
                mock.patch("time.sleep", sleeper),
            ):
                stdout = StringIO()
                with redirect_stdout(stdout), redirect_stderr(StringIO()):
                    exit_code = main(
                        [
                            "labeling-enrichment-run",
                            "--plan",
                            str(plan),
                            "--work",
                            str(root / "work"),
                            "--output",
                            str(root / "out-1"),
                            "--env-file",
                            str(env_file),
                        ]
                    )
            self.assertEqual(exit_code, 0)
            coverage = read_jsonl(root / "out-1" / "coverage.jsonl")
            self.assertTrue(any(row["status"] == "partial" for row in coverage))
            blob = scan_bytes(root / "out-1") + stdout.getvalue().encode()
            for secret in (FAKE_KEY, FAKE_FOLDER):
                self.assertNotIn(secret.encode(), blob)


            self.assertEqual(exit_code, 0)
            coverage = read_jsonl(root / "out-1" / "coverage.jsonl")
            self.assertTrue(any(row["status"] == "partial" for row in coverage))
            blob = scan_bytes(root / "out-1") + stdout.getvalue().encode()
            for secret in (FAKE_KEY, FAKE_FOLDER):
                self.assertNotIn(secret.encode(), blob)


class FailAllOpenAlexTransport(FakeTransport):
    """Fail every OpenAlex call; Media Cloud behaves like the fake default."""

    def get(self, url: str, *, headers, timeout_seconds: float) -> HttpResponse:
        if "openalex" in url:
            self.calls.append((url, dict(headers), timeout_seconds))
            return HttpResponse(500, {}, b'{"error": true}')
        return super().get(url, headers=headers, timeout_seconds=timeout_seconds)


def completed_spec_map(work: Path) -> dict[str, str]:
    mapping = {}
    for result_path in sorted((work / "completed").rglob("result.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        mapping[payload["request_id"]] = payload["spec_digest"]
    return mapping


def openalex_calls(transport: FakeTransport) -> list[tuple]:
    return [call for call in transport.calls if "openalex" in call[0]]


def mediacloud_calls(transport: FakeTransport) -> list[tuple]:
    return [call for call in transport.calls if "openalex" not in call[0]]


class OpenAlexApiKeyEnrichmentTests(unittest.TestCase):
    SECRET = "openalex-free-key-1"

    def test_key_sends_bearer_and_keeps_request_ids_and_digests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(root, "run", [candidate_row(1)])
            anonymous_transport = FakeTransport()
            run_with_fakes(plan, root / "work-anon", root / "out-anon", anonymous_transport)
            keyed_env = environment() | {"NEXTWAVE_OPENALEX_API_KEY": f"  {self.SECRET}  "}
            keyed_transport = FakeTransport()
            run_with_fakes(
                plan, root / "work-key", root / "out-key", keyed_transport, env=keyed_env
            )

            keyed_openalex = openalex_calls(keyed_transport)
            self.assertTrue(keyed_openalex)
            for _, headers, _ in keyed_openalex:
                self.assertEqual(headers["Authorization"], f"Bearer {self.SECRET}")
            for url, _, _ in keyed_transport.calls:
                self.assertNotIn(self.SECRET, url)
            for _, headers, _ in openalex_calls(anonymous_transport):
                self.assertNotIn("Authorization", headers)

            self.assertEqual(
                completed_spec_map(root / "work-key"),
                completed_spec_map(root / "work-anon"),
            )
            anon_results = read_jsonl(root / "out-anon" / "request_results.jsonl")
            keyed_results = read_jsonl(root / "out-key" / "request_results.jsonl")
            self.assertEqual(
                [row["request_id"] for row in keyed_results],
                [row["request_id"] for row in anon_results],
            )

    def test_resume_reuses_completed_and_retries_only_failed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(root, "run", [candidate_row(1)])
            failing = FailAllOpenAlexTransport()
            run_with_fakes(
                plan,
                root / "work",
                root / "out-1",
                openalex_transport=failing,
                mediacloud_transport=FakeTransport(),
            )
            first = read_jsonl(root / "out-1" / "request_results.jsonl")
            failed_openalex = {
                row["request_id"]
                for row in first
                if row["connector"] == "openalex" and row["status"] == "failed"
            }
            self.assertTrue(failed_openalex)
            self.assertTrue(
                all(
                    row["status"] == "success"
                    for row in first
                    if row["connector"] == "mediacloud"
                )
            )

            keyed_env = environment() | {"NEXTWAVE_OPENALEX_API_KEY": self.SECRET}
            second_transport = FakeTransport()
            run_with_fakes(
                plan, root / "work", root / "out-2", second_transport, env=keyed_env
            )

            self.assertEqual(mediacloud_calls(second_transport), [])
            self.assertEqual(len(openalex_calls(second_transport)), len(failed_openalex))
            for _, headers, _ in openalex_calls(second_transport):
                self.assertEqual(headers["Authorization"], f"Bearer {self.SECRET}")

            second = read_jsonl(root / "out-2" / "request_results.jsonl")
            for row in second:
                if row["connector"] == "mediacloud":
                    self.assertTrue(row["reused"])
                else:
                    self.assertFalse(row["reused"])
            coverage = read_jsonl(root / "out-2" / "coverage.jsonl")
            self.assertTrue(all(row["status"] == "complete" for row in coverage))

    def test_blank_key_keeps_anonymous_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(root, "run", [candidate_row(1)])
            transport = FakeTransport()
            blank_env = environment() | {"NEXTWAVE_OPENALEX_API_KEY": "   "}
            run_with_fakes(plan, root / "work", root / "out-1", transport, env=blank_env)

            for _, headers, _ in openalex_calls(transport):
                self.assertNotIn("Authorization", headers)

    def test_mediacloud_unaffected_by_openalex_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(root, "run", [candidate_row(1)])
            transport = FakeTransport()
            keyed_env = environment() | {"NEXTWAVE_OPENALEX_API_KEY": self.SECRET}
            run_with_fakes(plan, root / "work", root / "out-1", transport, env=keyed_env)

            media = mediacloud_calls(transport)
            self.assertTrue(media)
            for url, headers, _ in media:
                self.assertEqual(headers["Authorization"], f"Token {FAKE_KEY}")
                self.assertNotIn("Bearer", headers["Authorization"])
                self.assertNotIn(self.SECRET, url)
            blob = scan_bytes(root / "out-1")
            self.assertNotIn(self.SECRET.encode(), blob)


def build_tampered_plan(root: Path, name: str, rows: list[dict], mutate) -> Path:
    """Build a valid plan, then mutate plan.json keeping its manifest valid."""

    plan_dir = build_plan_output(root, name, rows)
    plan_data = json.loads((plan_dir / "plan.json").read_text(encoding="utf-8"))
    mutate(plan_data)
    raw = (
        json.dumps(plan_data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    (plan_dir / "plan.json").write_bytes(raw)
    manifest = json.loads((plan_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["outputs"]["plan.json"] = {
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    (plan_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return plan_dir


class StrictValidationTests(unittest.TestCase):
    def test_changed_manifest_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(root, "run", [candidate_row(1)])
            manifest = json.loads((plan / "manifest.json").read_text(encoding="utf-8"))
            manifest["schema_version"] = "labeling-enrichment-plan-v0"
            (plan / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                run_with_fakes(plan, root / "work", root / "out-1", FakeTransport())
            self.assertFalse((root / "work").exists())
            self.assertFalse((root / "out-1").exists())

    def test_tampered_policy_window_class_language_collections_rejected(self) -> None:
        cases = [
            (
                "policy",
                lambda search: search.update(
                    {"retrieval_policy": dict(search["retrieval_policy"], per_page=21)}
                ),
                "retrieval policy",
            ),
            (
                "window",
                lambda search: search.update(
                    {"planned_window": {"from": "2024-09-14", "until": "2026-09-15"}}
                ),
                "window",
            ),
            (
                "class",
                lambda search: search.update({"source_class": "industry"}),
                "source_class",
            ),
            (
                "language",
                lambda search: search.update(
                    {
                        "requests": [
                            dict(search["requests"][0], language="de"),
                            *search["requests"][1:],
                        ]
                    }
                ),
                "retired singular field",
            ),
        ]
        for name, mutate_search, pattern in cases:
            with self.subTest(tamper=name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    plan_dir = build_plan_output(root, "run", [candidate_row(1)])
                    plan_data = json.loads(
                        (plan_dir / "plan.json").read_text(encoding="utf-8")
                    )
                    entry = plan_data["candidates"][0]
                    search = next(
                        s for s in entry["searches"] if s["connector"] == "openalex"
                    )
                    mutate_search(search)
                    raw = (
                        json.dumps(
                            plan_data, ensure_ascii=False, indent=2, sort_keys=True
                        )
                        + "\n"
                    ).encode("utf-8")
                    (plan_dir / "plan.json").write_bytes(raw)
                    manifest = json.loads(
                        (plan_dir / "manifest.json").read_text(encoding="utf-8")
                    )
                    manifest["outputs"]["plan.json"] = {
                        "size_bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                    (plan_dir / "manifest.json").write_text(
                        json.dumps(
                            manifest, ensure_ascii=False, indent=2, sort_keys=True
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(ValueError, pattern):
                        run_with_fakes(
                            plan_dir, root / "work", root / "out-1", FakeTransport()
                        )
                    self.assertFalse((root / "work").exists())
                    self.assertFalse((root / "out-1").exists())

    def test_tampered_collections_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def mutate(plan_data) -> None:
                entry = plan_data["candidates"][0]
                target = next(
                    s for s in entry["searches"] if s["connector"] == "mediacloud"
                )
                target["collection_ids"] = [1]

            plan_dir = build_tampered_plan(root, "run", [candidate_row(1)], mutate)
            with self.assertRaisesRegex(ValueError, "collections"):
                run_with_fakes(plan_dir, root / "work", root / "out-1", FakeTransport())
            self.assertFalse((root / "work").exists())
            self.assertFalse((root / "out-1").exists())

    def test_path_traversal_request_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def mutate(plan_data) -> None:
                entry = plan_data["candidates"][0]
                target = next(
                    s for s in entry["searches"] if s["connector"] == "openalex"
                )
                target["requests"][0]["request_id"] = "../escape"

            plan_dir = build_tampered_plan(root, "run", [candidate_row(1)], mutate)
            with self.assertRaisesRegex(ValueError, "unsafe format"):
                run_with_fakes(plan_dir, root / "work", root / "out-1", FakeTransport())
            self.assertFalse((root / "work").exists())

    def test_recomputed_id_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def mutate(plan_data) -> None:
                entry = plan_data["candidates"][0]
                target = next(
                    s for s in entry["searches"] if s["connector"] == "openalex"
                )
                target["requests"][0]["search_text"] = target["requests"][0][
                    "search_text"
                ] + "X"

            plan_dir = build_tampered_plan(root, "run", [candidate_row(1)], mutate)
            with self.assertRaisesRegex(ValueError, "recomputation"):
                run_with_fakes(plan_dir, root / "work", root / "out-1", FakeTransport())
            self.assertFalse((root / "work").exists())

    def test_unbuildable_mediacloud_text_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                candidate_row(1, canonical_name="а б в", aliases=[]),
            ]
            plan = build_plan_output(root, "run", rows)
            transport = FakeTransport()
            with self.assertRaisesRegex(ValueError, "cannot be built"):
                run_with_fakes(plan, root / "work", root / "out-1", transport)
            self.assertEqual(transport.calls, [])
            self.assertFalse((root / "work").exists())

    def test_result_json_flip_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(root, "run", [candidate_row(1)])
            transport = FakeTransport()
            run_with_fakes(plan, root / "work", root / "out-1", transport)
            plan_data = json.loads((plan / "plan.json").read_text(encoding="utf-8"))
            request_id = plan_data["candidates"][0]["searches"][0]["requests"][0][
                "request_id"
            ]
            result_path = root / "work" / "completed" / request_id / "result.json"
            result_path.write_bytes(
                result_path.read_bytes().replace(b'"status": "success"', b'"status": "succesS"')
            )
            with self.assertRaisesRegex(ValueError, "checksum"):
                run_with_fakes(plan, root / "work", root / "out-2", transport)
            self.assertFalse((root / "out-2").exists())

    def test_snapshot_manifest_flip_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(root, "run", [candidate_row(1)])
            transport = FakeTransport()
            run_with_fakes(plan, root / "work", root / "out-1", transport)
            plan_data = json.loads((plan / "plan.json").read_text(encoding="utf-8"))
            request_id = plan_data["candidates"][0]["searches"][0]["requests"][0][
                "request_id"
            ]
            manifest_path = (
                root / "work" / "completed" / request_id / "snapshot" / "manifest.json"
            )
            manifest_path.write_bytes(
                manifest_path.read_bytes().replace(b'"snapshot"', b'"snapshoT"')
            )
            with self.assertRaisesRegex(ValueError, "checksum"):
                run_with_fakes(plan, root / "work", root / "out-2", transport)
            self.assertFalse((root / "out-2").exists())

    def test_cached_result_field_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(root, "run", [candidate_row(1)])
            transport = FakeTransport()
            run_with_fakes(plan, root / "work", root / "out-1", transport)
            plan_data = json.loads((plan / "plan.json").read_text(encoding="utf-8"))
            request_id = plan_data["candidates"][0]["searches"][0]["requests"][0][
                "request_id"
            ]
            completed = root / "work" / "completed" / request_id
            result_path = completed / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["candidate_id"] = "team-negative-999"
            result_bytes = (
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            result_path.write_bytes(result_bytes)
            cache_path = completed / "cache_manifest.json"
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            cache["result"] = {
                "size_bytes": len(result_bytes),
                "sha256": hashlib.sha256(result_bytes).hexdigest(),
            }
            cache_path.write_text(
                json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "candidate_id mismatch"):
                run_with_fakes(plan, root / "work", root / "out-2", transport)
            self.assertFalse((root / "out-2").exists())

    def test_atomic_work_creation_publishes_all_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(root, "run", [candidate_row(1)])
            with self.assertRaisesRegex(ValueError, "NEXTWAVE_MEDIACLOUD_API_KEY"):
                run_enrichment(
                    plan_dir=plan,
                    work_dir=root / "work",
                    output_dir=root / "out-1",
                    environment={},
                    openalex_transport=FakeTransport(),
                    mediacloud_transport=FakeTransport(),
                    clock=FakeClock(),
                    monotonic_clock=FakeMonotonic(),
                    sleeper=FakeSleeper(),
                )
            work = root / "work"
            self.assertTrue((work / "work_manifest.json").is_file())
            self.assertTrue((work / "completed").is_dir())
            self.assertTrue((work / "failures").is_dir())

    def test_existing_output_creates_no_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(root, "run", [candidate_row(1)])
            output = root / "out-1"
            output.mkdir()
            (output / "marker.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "already exists"):
                run_with_fakes(plan, root / "work", output, FakeTransport())
            self.assertFalse((root / "work").exists())
            self.assertEqual((output / "marker.txt").read_text(encoding="utf-8"), "keep")

    def test_orphan_staging_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(root, "run", [candidate_row(1)])
            transport = FakeTransport()
            run_with_fakes(plan, root / "work", root / "out-1", transport)
            orphan = root / "work" / ".orphan-staging-1"
            orphan.mkdir()
            (orphan / "junk.txt").write_text("junk", encoding="utf-8")
            run_with_fakes(plan, root / "work", root / "out-2", transport)

            self.assertTrue((orphan / "junk.txt").is_file())
            self.assertTrue((root / "out-2" / "coverage.jsonl").is_file())

    def test_partial_parse_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(
                root, "run", [candidate_row(1, canonical_name="partial-parse")]
            )
            transport = FakeTransport()
            run_with_fakes(plan, root / "work", root / "out-1", transport)
            results = {
                row["search_text"]: row
                for row in read_jsonl(root / "out-1" / "request_results.jsonl")
            }

            for row in results.values():
                self.assertEqual(row["status"], "success")
                self.assertEqual(row["returned_records"], 2)
                self.assertEqual(row["returned_documents"], 1)
                self.assertEqual(row["parse_issue_count"], 1)
            coverage = {
                row["connector"]: row
                for row in read_jsonl(root / "out-1" / "coverage.jsonl")
            }
            for row in coverage.values():
                self.assertEqual(row["status"], "complete")
            self.assertEqual(coverage["openalex"]["returned_records"], 4)
            self.assertEqual(coverage["openalex"]["returned_documents"], 1)
            self.assertEqual(coverage["openalex"]["parse_issue_count"], 2)
            self.assertEqual(coverage["mediacloud"]["returned_records"], 2)
            self.assertEqual(coverage["mediacloud"]["returned_documents"], 1)
            self.assertEqual(coverage["mediacloud"]["parse_issue_count"], 1)

    def test_contract_error_has_no_assertion(self) -> None:
        from nextwave.labeling.enrichment_run import _execute_request
        from nextwave.sources import MediaCloudConnector, OpenAlexConnector

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transport = FakeTransport()
            outcome, record, staging = _execute_request(
                candidate={
                    "candidate_id": "team-negative-001",
                    "canonical_name": "y" * 300,
                    "analysis_scope_key": "edge-v1",
                },
                search={
                    "connector": "openalex",
                    "retrieval_policy": {"max_attempts": 3, "per_page": 20},
                },
                request={
                    "request_id": "request-0123456789abcdef",
                    "search_text": "y" * 300,
                    "languages": ["en"],
                },
                window={"from": "2024-09-15", "until": "2026-09-15"},
                openalex_connector=OpenAlexConnector(transport=transport),
                mediacloud_connector=MediaCloudConnector(
                    api_key=FAKE_KEY, transport=transport
                ),
                staging_parent=root,
                snapshot_version="openalex-discovery-v1",
                bundle_id="bundle-0123456789abcdef",
                clock=FakeClock(),
                sleeper=FakeSleeper(),
            )

            self.assertEqual(outcome, "failed")
            self.assertEqual(record["error"]["code"], "invalid_request")
            self.assertIsInstance(staging, Path)
            self.assertEqual(transport.calls, [])

    def test_numeric_retry_after_increases_pause(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(
                root, "run", [candidate_row(1, canonical_name="retry-after-45")]
            )
            transport = FakeTransport()
            sleeper = FakeSleeper()
            run_with_fakes(
                plan, root / "work", root / "out-1", transport, sleeper=sleeper
            )

            self.assertEqual(sleeper.calls.count(45.0), 3)
            results = read_jsonl(root / "out-1" / "request_results.jsonl")
            self.assertTrue(all(row["status"] == "success" for row in results))

    def test_invalid_retry_after_keeps_backoff(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(
                root, "run", [candidate_row(1, canonical_name="retry-after-bad")]
            )
            transport = FakeTransport()
            sleeper = FakeSleeper()
            run_with_fakes(
                plan, root / "work", root / "out-1", transport, sleeper=sleeper
            )

            self.assertNotIn(45.0, sleeper.calls)
            self.assertEqual(sleeper.calls.count(5.0), 3)
            results = read_jsonl(root / "out-1" / "request_results.jsonl")
            self.assertTrue(all(row["status"] == "success" for row in results))

    def test_rewritten_result_wrong_spec_digest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(root, "run", [candidate_row(1)])
            transport = FakeTransport()
            run_with_fakes(plan, root / "work", root / "out-1", transport)
            plan_data = json.loads((plan / "plan.json").read_text(encoding="utf-8"))
            request_id = plan_data["candidates"][0]["searches"][0]["requests"][0][
                "request_id"
            ]
            completed = root / "work" / "completed" / request_id
            result_path = completed / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["spec_digest"] = "0" * 64
            result_bytes = (
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            result_path.write_bytes(result_bytes)
            cache_path = completed / "cache_manifest.json"
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            cache["result"] = {
                "size_bytes": len(result_bytes),
                "sha256": hashlib.sha256(result_bytes).hexdigest(),
            }
            cache_path.write_text(
                json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "spec_digest mismatch"):
                run_with_fakes(plan, root / "work", root / "out-2", transport)
            self.assertFalse((root / "out-2").exists())

    def test_wrong_document_snapshot_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(root, "run", [candidate_row(1)])
            transport = FakeTransport()
            run_with_fakes(plan, root / "work", root / "out-1", transport)
            plan_data = json.loads((plan / "plan.json").read_text(encoding="utf-8"))
            request_id = plan_data["candidates"][0]["searches"][0]["requests"][0][
                "request_id"
            ]
            completed = root / "work" / "completed" / request_id
            result_path = completed / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["documents"][0]["snapshot_id"] = "other"
            result_bytes = (
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            result_path.write_bytes(result_bytes)
            cache_path = completed / "cache_manifest.json"
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            cache["result"] = {
                "size_bytes": len(result_bytes),
                "sha256": hashlib.sha256(result_bytes).hexdigest(),
            }
            cache_path.write_text(
                json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "snapshot mismatch"):
                run_with_fakes(plan, root / "work", root / "out-2", transport)
            self.assertFalse((root / "out-2").exists())

    def test_broken_parse_issue_element_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(root, "run", [candidate_row(1)])
            transport = FakeTransport()
            run_with_fakes(plan, root / "work", root / "out-1", transport)
            plan_data = json.loads((plan / "plan.json").read_text(encoding="utf-8"))
            request_id = plan_data["candidates"][0]["searches"][0]["requests"][0][
                "request_id"
            ]
            completed = root / "work" / "completed" / request_id
            result_path = completed / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["parse_issues"] = ["broken"]
            result["parse_issue_count"] = 1
            result_bytes = (
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            result_path.write_bytes(result_bytes)
            cache_path = completed / "cache_manifest.json"
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            cache["result"] = {
                "size_bytes": len(result_bytes),
                "sha256": hashlib.sha256(result_bytes).hexdigest(),
            }
            cache_path.write_text(
                json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-object parse issue"):
                run_with_fakes(plan, root / "work", root / "out-2", transport)
            self.assertFalse((root / "out-2").exists())

    def test_non_object_search_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def mutate(plan_data) -> None:
                plan_data["candidates"][0]["searches"][1] = "oops"

            plan_dir = build_tampered_plan(root, "run", [candidate_row(1)], mutate)
            with self.assertRaisesRegex(ValueError, "exactly two"):
                run_with_fakes(plan_dir, root / "work", root / "out-1", FakeTransport())
            self.assertFalse((root / "work").exists())
            self.assertFalse((root / "out-1").exists())

    def test_missing_connector_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def mutate(plan_data) -> None:
                entry = plan_data["candidates"][0]
                target = next(
                    s for s in entry["searches"] if s["connector"] == "openalex"
                )
                del target["connector"]

            plan_dir = build_tampered_plan(root, "run", [candidate_row(1)], mutate)
            with self.assertRaisesRegex(ValueError, "exactly one"):
                run_with_fakes(plan_dir, root / "work", root / "out-1", FakeTransport())
            self.assertFalse((root / "work").exists())
            self.assertFalse((root / "out-1").exists())

    def test_missing_completed_dir_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(root, "run", [candidate_row(1)])
            transport = FakeTransport()
            run_with_fakes(plan, root / "work", root / "out-1", transport)
            shutil.rmtree(root / "work" / "completed")
            with self.assertRaisesRegex(ValueError, "service directory is missing"):
                run_with_fakes(plan, root / "work", root / "out-2", transport)
            self.assertFalse((root / "work" / "completed").exists())
            self.assertFalse((root / "out-2").exists())

    def test_retry_after_3600_defers_without_sleep(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(
                root, "run", [candidate_row(1, canonical_name="retry-after-3600")]
            )
            transport = FakeTransport()
            sleeper = FakeSleeper()
            run_with_fakes(
                plan, root / "work", root / "out-1", transport, sleeper=sleeper
            )

            self.assertEqual(
                sum(1 for call in transport.calls if "retry-after-3600" in call[0]), 3
            )
            self.assertTrue(all(delay < 120.0 for delay in sleeper.calls))
            self.assertNotIn(3600.0, sleeper.calls)
            results = {
                row["search_text"]: row
                for row in read_jsonl(root / "out-1" / "request_results.jsonl")
            }
            for row in results.values():
                self.assertEqual(row["status"], "failed")
                self.assertEqual(row["error"]["retry_after_seconds"], 3600.0)
                self.assertTrue(row["error"]["retry_deferred"])
            coverage = read_jsonl(root / "out-1" / "coverage.jsonl")
            self.assertTrue(all(row["status"] == "unknown" for row in coverage))

    def test_v1_plan_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(root, "run", [candidate_row(1)])
            for name in ("plan.json", "manifest.json"):
                path = plan / name
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["schema_version"] = "labeling-enrichment-plan-v1"
                path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
                    + "\n",
                    encoding="utf-8",
                )
            with self.assertRaisesRegex(ValueError, "does not match"):
                run_with_fakes(plan, root / "work", root / "out-1", FakeTransport())
            self.assertFalse((root / "work").exists())
            self.assertFalse((root / "out-1").exists())

    def test_v2_plan_accepted_without_singular_language(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(root, "run", [candidate_row(1)])
            transport = FakeTransport()
            run_with_fakes(plan, root / "work", root / "out-1", transport)

            rows = read_jsonl(root / "out-1" / "request_results.jsonl")
            self.assertTrue(all(isinstance(row["languages"], list) for row in rows))
            self.assertTrue(all("language" not in row for row in rows))

    def test_bad_languages_shapes_are_rejected(self) -> None:
        cases = [
            ("empty", []),
            ("duplicate", ["en", "en"]),
            ("unknown", ["xx"]),
            ("wrong_order", ["ru", "en"]),
        ]
        for name, languages in cases:
            with self.subTest(shape=name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)

                    def mutate(plan_data, _languages=languages) -> None:
                        entry = plan_data["candidates"][0]
                        target = next(
                            s
                            for s in entry["searches"]
                            if s["connector"] == "mediacloud"
                        )
                        target["requests"][0]["languages"] = _languages

                    plan_dir = build_tampered_plan(
                        root, "run", [candidate_row(1)], mutate
                    )
                    with self.assertRaises(ValueError):
                        run_with_fakes(
                            plan_dir, root / "work", root / "out-1", FakeTransport()
                        )
                    self.assertFalse((root / "work").exists())
                    self.assertFalse((root / "out-1").exists())

    def test_split_bilingual_request_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def mutate(plan_data) -> None:
                entry = plan_data["candidates"][0]
                target = next(
                    s for s in entry["searches"] if s["connector"] == "mediacloud"
                )
                first = target["requests"][0]
                target["requests"] = [
                    dict(
                        first,
                        request_id=first["request_id"] + "-en",
                        languages=["en"],
                    ),
                    dict(
                        first,
                        request_id=first["request_id"] + "-ru",
                        languages=["ru"],
                    ),
                ]

            plan_dir = build_tampered_plan(root, "run", [candidate_row(1)], mutate)
            with self.assertRaises(ValueError):
                run_with_fakes(plan_dir, root / "work", root / "out-1", FakeTransport())
            self.assertFalse((root / "work").exists())
            self.assertFalse((root / "out-1").exists())

    def test_languages_substitution_in_cache_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(root, "run", [candidate_row(1)])
            transport = FakeTransport()
            run_with_fakes(plan, root / "work", root / "out-1", transport)
            plan_data = json.loads((plan / "plan.json").read_text(encoding="utf-8"))
            request_id = plan_data["candidates"][0]["searches"][0]["requests"][0][
                "request_id"
            ]
            completed = root / "work" / "completed" / request_id
            result_path = completed / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["languages"] = ["ru"]
            result_bytes = (
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            result_path.write_bytes(result_bytes)
            cache_path = completed / "cache_manifest.json"
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            cache["result"] = {
                "size_bytes": len(result_bytes),
                "sha256": hashlib.sha256(result_bytes).hexdigest(),
            }
            cache_path.write_text(
                json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "mismatch"):
                run_with_fakes(plan, root / "work", root / "out-2", transport)
            self.assertFalse((root / "out-2").exists())

    def test_mediacloud_planned_count_is_per_term(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_plan_output(
                root, "run", [candidate_row(1, aliases=["Second", "Third"])]
            )
            transport = FakeTransport()
            run_with_fakes(plan, root / "work", root / "out-1", transport)
            coverage = {
                row["connector"]: row
                for row in read_jsonl(root / "out-1" / "coverage.jsonl")
            }

            self.assertEqual(coverage["openalex"]["planned_requests"], 6)
            self.assertEqual(coverage["mediacloud"]["planned_requests"], 3)


if __name__ == "__main__":
    unittest.main()
