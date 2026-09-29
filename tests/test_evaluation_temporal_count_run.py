from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
import unittest
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from nextwave.evaluation.temporal_count_plan import export_temporal_count_plan
from nextwave.evaluation.temporal_count_run import _query_from_task, run_temporal_counts
from nextwave.sources import HttpResponse, OpenAlexConnector


def _digest(payload: bytes) -> dict[str, object]:
    return {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _candidate(candidate_id: str, scope: str, term: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "canonical_name": term,
        "aliases": [],
        "domain": "Edge",
        "analysis_scope_key": scope,
        "cutoff_date": "2026-09-15",
        "search_terms": [term, f"{term} alias"],
    }


def _write_plan(directory: Path, bundle_id: str, candidates: list[dict]) -> None:
    directory.mkdir()
    value = {
        "schema_version": "labeling-enrichment-plan-v2",
        "cutoff_date": "2026-09-15",
        "candidates": candidates,
    }
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()
    (directory / "plan.json").write_bytes(payload)
    manifest = {
        "schema_version": "labeling-enrichment-plan-v2",
        "bundle_id": bundle_id,
        "candidate_count": len(candidates),
        "outputs": {"plan.json": _digest(payload)},
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )


class CountTransport:
    def __init__(self, count: int = 5) -> None:
        self.count = count
        self.calls: list[tuple[str, Mapping[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HttpResponse:
        self.calls.append((url, headers))
        body = json.dumps({"meta": {"count": self.count}, "results": []}).encode()
        return HttpResponse(200, {"Content-Type": "application/json"}, body)


class InvalidCountTransport(CountTransport):
    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HttpResponse:
        self.calls.append((url, headers))
        body = json.dumps({"meta": {"count": "five"}, "results": []}).encode()
        return HttpResponse(200, {"Content-Type": "application/json"}, body)


class ConcurrentCountTransport(CountTransport):
    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HttpResponse:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.02)
            return super().get(url, headers=headers, timeout_seconds=timeout_seconds)
        finally:
            with self.lock:
                self.active -= 1


class CandidateExceedsScopeTransport(CountTransport):
    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> HttpResponse:
        self.calls.append((url, headers))
        search = parse_qs(urlparse(url).query)["search"][0]
        count = 6 if " AND " in search else 5
        body = json.dumps({"meta": {"count": count}, "results": []}).encode()
        return HttpResponse(200, {"Content-Type": "application/json"}, body)


class TemporalCountRunTests(unittest.TestCase):
    def _plan(self, root: Path) -> Path:
        positive = root / "positive"
        negative = root / "negative"
        _write_plan(
            positive,
            "bundle-positive",
            [_candidate("organizer-001", "edge-v1", "edge compression")],
        )
        _write_plan(
            negative,
            "bundle-negative",
            [_candidate("team-negative-001", "edge-v1", "in-memory computing")],
        )
        plan = root / "count-plan"
        export_temporal_count_plan(
            positive_plan_dir=positive,
            negative_plan_dir=negative,
            output_dir=plan,
        )
        return plan

    def test_concurrency_runs_independent_openalex_counts_in_parallel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transport = ConcurrentCountTransport()
            run_temporal_counts(
                plan_dir=self._plan(root),
                work_dir=root / "work",
                output_dir=root / "output",
                connector=OpenAlexConnector(transport=transport),
                concurrency=4,
            )
            self.assertGreater(transport.max_active, 1)

    def test_long_dynamic_scope_expression_stays_in_search_texts(self) -> None:
        expression = " OR ".join(f'"scope term {index}"' for index in range(20))
        query = _query_from_task(
            {
                "count_id": "count-dynamic-scope",
                "search_text": expression,
                "scope_search_text": expression,
                "analysis_scope_key": "scope-dynamic",
                "search_mode": "boolean_scope",
                "published_from": "2025-09-15",
                "published_until": "2026-09-15",
            },
            cutoff=date(2026, 9, 15),
        )

        self.assertGreater(len(expression), 200)
        self.assertEqual(query.raw_query, "scope-dynamic")
        self.assertEqual(query.search_texts, (expression,))

    def test_full_run_uses_meta_count_and_builds_candidate_ratios(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = self._plan(root)
            transport = CountTransport(5)
            paths = run_temporal_counts(
                plan_dir=plan,
                work_dir=root / "work",
                output_dir=root / "result",
                connector=OpenAlexConnector(transport=transport),
                sleeper=lambda _seconds: None,
            )
            manifest = json.loads(paths["manifest.json"].read_text())
            rows = [
                json.loads(line)
                for line in paths["candidate_temporal_features.jsonl"].read_text().splitlines()
            ]
        self.assertEqual(len(transport.calls), 16)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["counts"]["complete"], 16)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["coverage"] == "complete" for row in rows))
        self.assertTrue(all(row["scope_share_recent"] == 1.0 for row in rows))

    def test_successful_zero_is_covered_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = run_temporal_counts(
                plan_dir=self._plan(root),
                work_dir=root / "work",
                output_dir=root / "result",
                connector=OpenAlexConnector(transport=CountTransport(0)),
                sleeper=lambda _seconds: None,
            )
            rows = [
                json.loads(line)
                for line in paths["candidate_temporal_features.jsonl"].read_text().splitlines()
            ]
        self.assertTrue(all(row["coverage"] == "complete" for row in rows))
        self.assertTrue(all(row["counts"]["candidate_recent"] == 0 for row in rows))
        self.assertTrue(all(row["scope_share_recent"] is None for row in rows))

    def test_invalid_meta_count_is_unknown_not_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = run_temporal_counts(
                plan_dir=self._plan(root),
                work_dir=root / "work",
                output_dir=root / "result",
                connector=OpenAlexConnector(transport=InvalidCountTransport()),
                sleeper=lambda _seconds: None,
                max_new_tasks=1,
            )
            results = [
                json.loads(line) for line in paths["count_results.jsonl"].read_text().splitlines()
            ]
        self.assertEqual(results[0]["status"], "unknown")
        self.assertIsNone(results[0]["count"])
        self.assertEqual(results[1]["status"], "not_run")

    def test_resume_reuses_successes_without_network_and_keeps_data_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = self._plan(root)
            first_transport = CountTransport(3)
            first = run_temporal_counts(
                plan_dir=plan,
                work_dir=root / "work",
                output_dir=root / "first-result",
                connector=OpenAlexConnector(transport=first_transport),
                sleeper=lambda _seconds: None,
            )
            second_transport = CountTransport(999)
            second = run_temporal_counts(
                plan_dir=plan,
                work_dir=root / "work",
                output_dir=root / "second-result",
                connector=OpenAlexConnector(transport=second_transport),
                sleeper=lambda _seconds: None,
            )
            first_counts = first["count_results.jsonl"].read_bytes()
            second_counts = second["count_results.jsonl"].read_bytes()
            first_features = first["candidate_temporal_features.jsonl"].read_bytes()
            second_features = second["candidate_temporal_features.jsonl"].read_bytes()
            second_manifest = json.loads(second["manifest.json"].read_text())
        self.assertEqual(len(first_transport.calls), 16)
        self.assertEqual(second_transport.calls, [])
        self.assertEqual(first_counts, second_counts)
        self.assertEqual(first_features, second_features)
        self.assertEqual(second_manifest["counts"]["reused"], 16)

    def test_completed_cache_publish_retries_transient_windows_lock(self) -> None:
        real_replace = os.replace
        failed_once = False

        def flaky_replace(source, target):
            nonlocal failed_once
            source_path = Path(source)
            target_path = Path(target)
            is_completed_cache = (
                target_path.parent.name == "completed"
                and source_path.name.startswith(".count-")
            )
            if is_completed_cache and not failed_once:
                failed_once = True
                raise PermissionError(13, "transient Windows lock")
            return real_replace(source, target)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch(
                "nextwave.sources.snapshots.os.replace",
                side_effect=flaky_replace,
            ):
                paths = run_temporal_counts(
                    plan_dir=self._plan(root),
                    work_dir=root / "work",
                    output_dir=root / "result",
                    connector=OpenAlexConnector(transport=CountTransport()),
                    sleeper=lambda _seconds: None,
                    max_new_tasks=1,
                )
            manifest = json.loads(paths["manifest.json"].read_text())

        self.assertTrue(failed_once)
        self.assertEqual(manifest["counts"]["complete"], 1)
        self.assertEqual(manifest["counts"]["not_run"], 15)

    def test_corrupt_completed_cache_is_rejected_before_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = self._plan(root)
            run_temporal_counts(
                plan_dir=plan,
                work_dir=root / "work",
                output_dir=root / "first-result",
                connector=OpenAlexConnector(transport=CountTransport()),
                sleeper=lambda _seconds: None,
            )
            first_cache = next((root / "work" / "completed").iterdir())
            (first_cache / "result.json").write_text("{}\n", encoding="utf-8")
            transport = CountTransport()
            with self.assertRaisesRegex(ValueError, "cache .* diverges"):
                run_temporal_counts(
                    plan_dir=plan,
                    work_dir=root / "work",
                    output_dir=root / "second-result",
                    connector=OpenAlexConnector(transport=transport),
                    sleeper=lambda _seconds: None,
                )
        self.assertEqual(transport.calls, [])

    def test_api_key_is_only_in_authorization_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transport = CountTransport()
            paths = run_temporal_counts(
                plan_dir=self._plan(root),
                work_dir=root / "work",
                output_dir=root / "result",
                connector=OpenAlexConnector(transport=transport, api_key="temporal-secret"),
                sleeper=lambda _seconds: None,
                max_new_tasks=1,
            )
            serialized = b"".join(path.read_bytes() for path in paths.values())
        self.assertEqual(transport.calls[0][1]["Authorization"], "Bearer temporal-secret")
        search = parse_qs(urlparse(transport.calls[0][0]).query)["search"][0]
        self.assertEqual(
            search,
            '("edge compression"~5) AND ("edge computing" OR "edge AI")',
        )
        self.assertNotIn("temporal-secret", transport.calls[0][0])
        self.assertNotIn(b"temporal-secret", serialized)

    def test_candidate_count_cannot_exceed_scope_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "candidate count exceeds scope count"):
                run_temporal_counts(
                    plan_dir=self._plan(root),
                    work_dir=root / "work",
                    output_dir=root / "result",
                    connector=OpenAlexConnector(transport=CandidateExceedsScopeTransport()),
                    sleeper=lambda _seconds: None,
                )
        self.assertFalse((root / "result").exists())

    def test_missing_task_is_rejected_before_work_or_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_dir = self._plan(root)
            plan_path = plan_dir / "plan.json"
            plan = json.loads(plan_path.read_text())
            plan["tasks"].pop()
            payload = (
                json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode()
            plan_path.write_bytes(payload)
            manifest_path = plan_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["outputs"]["plan.json"] = {
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
            transport = CountTransport()
            with self.assertRaisesRegex(ValueError, "do not cover"):
                run_temporal_counts(
                    plan_dir=plan_dir,
                    work_dir=root / "work",
                    output_dir=root / "result",
                    connector=OpenAlexConnector(transport=transport),
                    sleeper=lambda _seconds: None,
                )
            self.assertFalse((root / "work").exists())
            self.assertEqual(transport.calls, [])


if __name__ == "__main__":
    unittest.main()
