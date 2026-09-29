from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections.abc import Mapping
from datetime import date
from pathlib import Path

from nextwave.evaluation.exa_enrichment_plan import EXA_ENRICHMENT_PLAN_VERSION
from nextwave.evaluation.exa_enrichment_run import run_exa_enrichment
from nextwave.sources import HttpResponse, QueryPurpose, SourceQuery, build_exa_news_request


def digest(data: bytes):
    return {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def make_plan(root: Path):
    tasks = []
    for candidate_id in ("candidate-a", "candidate-b"):
        for window, start, end in (
            ("previous", date(2024, 9, 15), date(2025, 9, 14)),
            ("recent", date(2025, 9, 15), date(2026, 9, 15)),
        ):
            retrieval_query = "News coverage and industry reporting about AI accelerator"
            query = SourceQuery(
                query_id=f"query-{hashlib.sha256(f'{candidate_id}-{window}'.encode()).hexdigest()[:16]}",
                analysis_scope_id=f"scope-{hashlib.sha256(candidate_id.encode()).hexdigest()[:16]}",
                purpose=QueryPurpose.HISTORICAL_ENRICHMENT,
                raw_query=retrieval_query,
                normalized_query=retrieval_query.casefold(),
                search_texts=(retrieval_query,),
                published_from=start,
                published_until=end,
                cutoff_date=date(2026, 9, 15),
                languages=("en", "ru"),
            )
            tasks.append(
                {
                    "candidate_id": candidate_id,
                    "request": build_exa_news_request(
                        query, search_text=retrieval_query, num_results=10
                    ).to_dict(),
                    "retrieval_query": retrieval_query,
                    "search_text": "AI accelerator",
                    "term_rank": 1,
                    "window": window,
                }
            )
    tasks.sort(key=lambda item: (item["candidate_id"], item["window"]))
    plan = {
        "schema_version": EXA_ENRICHMENT_PLAN_VERSION,
        "bundle_id": "bundle-1",
        "cutoff_date": "2026-09-15",
        "policy": {},
        "totals": {"candidates": 2, "requests": 4},
        "tasks": tasks,
    }
    raw = (json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    root.mkdir()
    (root / "plan.json").write_bytes(raw)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": EXA_ENRICHMENT_PLAN_VERSION,
                "outputs": {"plan.json": digest(raw)},
            }
        ),
        encoding="utf-8",
    )


class FakeTransport:
    def __init__(self):
        self.calls: list[str] = []

    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes,
        timeout_seconds: float,
    ) -> HttpResponse:
        self.calls.append(json.loads(body)["startPublishedDate"])
        published = (
            "2025-01-10T00:00:00Z"
            if json.loads(body)["startPublishedDate"].startswith("2024")
            else "2026-01-10T00:00:00Z"
        )
        response = {
            "results": [
                {
                    "id": f"https://example.com/{len(self.calls)}",
                    "url": f"https://example.com/{len(self.calls)}",
                    "title": "AI accelerator deployment",
                    "publishedDate": published,
                    "highlights": ["The accelerator entered production."],
                }
            ]
        }
        return HttpResponse(
            200, {"Content-Type": "application/json"}, json.dumps(response).encode()
        )


class ExaEnrichmentRunTests(unittest.TestCase):
    def test_complete_run_and_resume_without_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, work = root / "plan", root / "work"
            make_plan(plan)
            transport = FakeTransport()
            first = run_exa_enrichment(
                plan_dir=plan,
                work_dir=work,
                output_dir=root / "result-1",
                environment={"NEXTWAVE_EXA_API_KEY": "secret"},
                transport=transport,
                sleeper=lambda _: None,
            )
            self.assertEqual(len(transport.calls), 4)
            coverage = [json.loads(line) for line in first.coverage.read_text().splitlines()]
            self.assertTrue(all(row["status"] == "complete" for row in coverage))
            second_transport = FakeTransport()
            second = run_exa_enrichment(
                plan_dir=plan,
                work_dir=work,
                output_dir=root / "result-2",
                environment={"NEXTWAVE_EXA_API_KEY": "secret"},
                transport=second_transport,
                sleeper=lambda _: None,
            )
            self.assertEqual(second_transport.calls, [])
            manifest = json.loads(second.manifest.read_text())
            self.assertEqual(manifest["totals"]["reused_requests"], 4)

    def test_limit_preserves_not_run_and_partial_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = root / "plan"
            make_plan(plan)
            result = run_exa_enrichment(
                plan_dir=plan,
                work_dir=root / "work",
                output_dir=root / "result",
                environment={"NEXTWAVE_EXA_API_KEY": "secret"},
                max_new_requests=1,
                concurrency=1,
                transport=FakeTransport(),
                sleeper=lambda _: None,
            )
            manifest = json.loads(result.manifest.read_text())
            self.assertEqual(manifest["totals"]["successful_requests"], 1)
            self.assertEqual(manifest["totals"]["not_run_requests"], 3)

    def test_wrong_key_never_enters_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = root / "plan"
            make_plan(plan)
            result = run_exa_enrichment(
                plan_dir=plan,
                work_dir=root / "work",
                output_dir=root / "result",
                environment={"NEXTWAVE_EXA_API_KEY": "secret-value"},
                max_new_requests=1,
                concurrency=1,
                transport=FakeTransport(),
                sleeper=lambda _: None,
            )
            for path in result.manifest.parent.iterdir():
                self.assertNotIn("secret-value", path.read_text(encoding="utf-8"))

    def test_corrupt_snapshot_is_rejected_before_network(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, work = root / "plan", root / "work"
            make_plan(plan)
            run_exa_enrichment(
                plan_dir=plan,
                work_dir=work,
                output_dir=root / "result-1",
                environment={"NEXTWAVE_EXA_API_KEY": "secret"},
                max_new_requests=1,
                concurrency=1,
                transport=FakeTransport(),
                sleeper=lambda _: None,
            )
            cached = next((work / "completed").iterdir())
            raw = next((cached / "snapshots").rglob("*.json"))
            raw.write_bytes(raw.read_bytes() + b" ")
            transport = FakeTransport()
            with self.assertRaisesRegex(ValueError, "snapshots are corrupted"):
                run_exa_enrichment(
                    plan_dir=plan,
                    work_dir=work,
                    output_dir=root / "result-2",
                    environment={"NEXTWAVE_EXA_API_KEY": "secret"},
                    transport=transport,
                    sleeper=lambda _: None,
                )
            self.assertEqual(transport.calls, [])


if __name__ == "__main__":
    unittest.main()
