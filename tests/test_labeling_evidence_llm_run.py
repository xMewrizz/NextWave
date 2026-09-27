"""Evidence LLM executor tests: fake generator only, no network, no model calls."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor as RealExecutor
from pathlib import Path
from unittest import mock

from nextwave.contracts import ClaimType, EvidenceDirection
from nextwave.discovery.llm import (
    LlmProvider,
    LlmRuntimeSettings,
    LlmSelection,
    YandexContentFilterError,
    YandexTruncationError,
    build_json_generator,
)
from nextwave.labeling import evidence_llm_plan as plan_module
from nextwave.labeling.evidence_llm_plan import (
    LABELING_EVIDENCE_LLM_PLAN_VERSION,
    build_evidence_prompt,
)
from nextwave.labeling.evidence_llm_run import (
    EVIDENCE_EXTRACTOR_VERSION,
    EVIDENCE_MAX_OUTPUT_TOKENS,
    EVIDENCE_RESPONSE_JSON_SCHEMA,
    LABELING_EVIDENCE_LLM_CACHE_VERSION,
    LABELING_EVIDENCE_LLM_EXECUTOR_VERSION,
    LABELING_EVIDENCE_LLM_RESULT_VERSION,
    LABELING_EVIDENCE_LLM_WORK_VERSION,
    _claim_match_is_sufficient,
    _extractor_id,
    run_evidence_llm,
)
from nextwave.sources import HttpResponse

BUNDLE = "bundle-test-001"
INPUT_MANIFEST_SHA = hashlib.sha256(b"fixture-evidence-input-manifest").hexdigest()
INPUT_MANIFEST_SIZE = 1234
SECRET_LIKE = "secret-xyz-123"

ENV = {
    "NEXTWAVE_LLM_PROVIDER": "yandex",
    "NEXTWAVE_LLM_MODEL": "YandexGPT Lite 5",
    "NEXTWAVE_LLM_API_KEY": "fake-key",
    "NEXTWAVE_YANDEX_FOLDER_ID": "fake-folder",
}

EXCERPT = (
    "Quantum error correction with surface codes improves devices "
    "and reported prototype measurements from a laboratory team."
)
QUOTE = "Quantum error correction with surface codes"


def digest(payload: bytes) -> dict:
    return {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def dump_json(payload) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def dump_jsonl(rows: list[dict]) -> bytes:
    return (
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    ).encode("utf-8")


def task_doc_row(candidate: str, number: int, connector: str, rank: int, **overrides) -> dict:
    source_class = "scientific" if connector == "openalex" else "industry"
    payload = {
        "candidate_id": candidate,
        "document_id": f"document-{candidate}-{number}",
        "connector": connector,
        "source_class": source_class,
        "final_rank": rank,
        "title": f"Study of {candidate} mechanisms",
        "url": f"https://example.org/{candidate}-{number}",
        "origin_id": f"doi:10.1/{candidate}-{number}",
        "published_at": "2025-04-01",
        "trust_tier": "A",
        "matched_term": "quantum error correction",
        "relevance_score": 80,
        "relevance_class": "strong",
        "excerpt": EXCERPT,
    }
    payload.update(overrides)
    return payload


def make_task_row(candidate: str, docs: list[dict]) -> dict:
    """Build a genuine task row with real prompt SHA and task ID."""
    from nextwave.labeling.evidence_llm_plan import select_passage

    task_documents = []
    for doc in docs:
        passage, start, end, _, source_sha, passage_sha = select_passage(
            doc["excerpt"], doc["matched_term"]
        )
        task_documents.append({
            "document_id": doc["document_id"],
            "final_rank": doc["final_rank"],
            "source_class": doc["source_class"],
            "connector": doc["connector"],
            "title": doc["title"],
            "url": doc["url"],
            "origin_id": doc["origin_id"],
            "published_at": doc["published_at"],
            "trust_tier": doc["trust_tier"],
            "matched_term": doc["matched_term"],
            "relevance_score": doc["relevance_score"],
            "relevance_class": doc["relevance_class"],
            "passage": passage,
            "passage_start": start,
            "passage_end": end,
            "passage_truncated": start != 0 or end != len(doc["excerpt"]),
            "source_excerpt_sha256": source_sha,
            "passage_sha256": passage_sha,
        })
    prompt = build_evidence_prompt(
        {"candidate_id": candidate, "documents": task_documents}
    )
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    task_id = plan_module._task_id(
        INPUT_MANIFEST_SHA,
        candidate,
        sorted(doc["document_id"] for doc in task_documents),
        sorted(doc["passage_sha256"] for doc in task_documents),
        sorted((doc["passage_start"], doc["passage_end"]) for doc in task_documents),
        prompt_sha,
    )
    return {
        "task_id": task_id,
        "candidate_id": candidate,
        "documents": task_documents,
        "document_count": len(task_documents),
        "prompt_chars": len(prompt),
        "estimated_input_tokens": math.ceil(len(prompt) / 3),
        "input_manifest_sha256": INPUT_MANIFEST_SHA,
        "prompt_sha256": prompt_sha,
    }


def run_coverage_row(candidate: str, task: dict | None) -> dict:
    if task is None:
        return {
            "candidate_id": candidate,
            "status": "no_input",
            "task_id": None,
            "input_documents": 0,
            "scientific_documents": 0,
            "industry_documents": 0,
            "prompt_chars": 0,
            "estimated_input_tokens": 0,
            "llm_call_planned": False,
            "empty_reasons": ["no_final_input"],
        }
    scientific = sum(1 for doc in task["documents"] if doc["source_class"] == "scientific")
    return {
        "candidate_id": candidate,
        "status": "planned",
        "task_id": task["task_id"],
        "input_documents": len(task["documents"]),
        "scientific_documents": scientific,
        "industry_documents": len(task["documents"]) - scientific,
        "prompt_chars": task["prompt_chars"],
        "estimated_input_tokens": task["estimated_input_tokens"],
        "llm_call_planned": True,
    }


def build_input(root: Path, tasks: list[dict], coverage: list[dict]) -> Path:
    input_dir = root / "input"
    input_dir.mkdir(parents=True)
    tasks_bytes = dump_jsonl(tasks)
    coverage_bytes = dump_jsonl(coverage)
    (input_dir / "tasks.jsonl").write_bytes(tasks_bytes)
    (input_dir / "coverage.jsonl").write_bytes(coverage_bytes)
    manifest = {
        "schema_version": LABELING_EVIDENCE_LLM_PLAN_VERSION,
        "bundle_id": BUNDLE,
        "cutoff_date": "2026-09-15",
        "input_manifest": {
            "sha256": INPUT_MANIFEST_SHA,
            "size_bytes": INPUT_MANIFEST_SIZE,
        },
        "totals": {
            "candidates": len(coverage),
            "planned_tasks": len(tasks),
        },
        "outputs": {
            "tasks.jsonl": digest(tasks_bytes),
            "coverage.jsonl": digest(coverage_bytes),
        },
    }
    (input_dir / "manifest.json").write_bytes(dump_json(manifest))
    return input_dir


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def answer(candidate: str, entries: list[tuple[str, list[tuple[str, str, str]]]]) -> str:
    return json.dumps({
        "candidate_id": candidate,
        "documents": [
            {
                "document_id": document_id,
                "claims": [
                    {"quote": quote, "kind": kind, "direction": direction}
                    for quote, kind, direction in claims
                ],
            }
            for document_id, claims in entries
        ],
    })


class FakeGenerator:
    def __init__(self, handler) -> None:
        self.handler = handler
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.handler(prompt)


def scripted(responses: dict) -> FakeGenerator:
    def handle(prompt: str) -> str:
        for candidate_id, response in responses.items():
            if f"candidate_id: {candidate_id}\n" in prompt:
                if isinstance(response, Exception):
                    raise response
                return response
        raise AssertionError("fake generator received an unexpected prompt")

    return FakeGenerator(handle)


def success_handler(entries: dict[str, list]) -> FakeGenerator:
    return scripted({
        candidate_id: answer(candidate_id, docs) for candidate_id, docs in entries.items()
    })


def run_with_fake(plan, work, out, generator, env=None, **kwargs):
    return run_evidence_llm(
        plan_dir=plan,
        work_dir=work,
        output_dir=out,
        environment=dict(ENV, **(env or {})),
        generator=generator,
        **kwargs,
    )


def two_task_fixture() -> tuple[list[dict], list[dict]]:
    docs_c1 = [task_doc_row("c1", number, "openalex", number) for number in (1, 2)]
    docs_c2 = [task_doc_row("c2", 1, "mediacloud", 1)]
    task_c1 = make_task_row("c1", docs_c1)
    task_c2 = make_task_row("c2", docs_c2)
    return [task_c1, task_c2], [
        run_coverage_row("c1", task_c1),
        run_coverage_row("c2", task_c2),
    ]


class ExecutorCallTests(unittest.TestCase):
    def test_six_documents_make_one_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [
                task_doc_row("c1", number, "openalex" if number % 2 else "mediacloud", number)
                for number in range(1, 7)
            ]
            task = make_task_row("c1", docs)
            plan = build_input(root, [task], [run_coverage_row("c1", task)])
            generator = success_handler({
                "c1": [(doc["document_id"], [(QUOTE, "research", "support")]) for doc in docs],
            })
            paths = run_with_fake(plan, root / "work", root / "out", generator)
            requests = read_jsonl(paths.request_results)
            claims = read_jsonl(paths.claims)
            manifest = json.loads((paths.manifest).read_text(encoding="utf-8"))

        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(requests[0]["status"], "success")
        self.assertEqual(len(claims), 6)
        self.assertEqual(manifest["totals"]["called"], 1)
        self.assertEqual(manifest["totals"]["success"], 1)

    def test_runtime_builder_receives_server_schema_and_2000_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [task_doc_row("c1", 1, "openalex", 1)]
            task = make_task_row("c1", docs)
            plan = build_input(root, [task], [run_coverage_row("c1", task)])
            generator = success_handler({"c1": [(docs[0]["document_id"], [])]})
            with mock.patch(
                "nextwave.labeling.evidence_llm_run.build_json_generator",
                return_value=generator,
            ) as builder:
                run_evidence_llm(
                    plan_dir=plan,
                    work_dir=root / "work",
                    output_dir=root / "out",
                    environment=ENV,
                )

        kwargs = builder.call_args.kwargs
        self.assertEqual(kwargs["max_output_tokens"], 2000)
        self.assertEqual(kwargs["schema_name"], "evidence_response")
        self.assertTrue(kwargs["server_side_json_schema"])
        self.assertEqual(kwargs["json_schema"], EVIDENCE_RESPONSE_JSON_SCHEMA)

    def test_no_input_makes_zero_calls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = build_input(
                root, [], [run_coverage_row("c1", None), run_coverage_row("c2", None)]
            )
            generator = success_handler({})
            paths = run_with_fake(plan, root / "work", root / "out", generator)
            manifest = json.loads((paths.manifest).read_text(encoding="utf-8"))

        self.assertEqual(len(generator.calls), 0)
        self.assertEqual(manifest["analysis_status"], "complete")
        self.assertEqual(manifest["totals"]["no_input"], 2)

    def test_empty_claims_are_cached_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [task_doc_row("c1", 1, "openalex", 1)]
            task = make_task_row("c1", docs)
            plan = build_input(root, [task], [run_coverage_row("c1", task)])
            generator = success_handler({"c1": [(docs[0]["document_id"], [])]})
            run_with_fake(plan, root / "work", root / "out-1", generator)
            fresh = success_handler({"c1": [(docs[0]["document_id"], [])]})
            paths = run_with_fake(plan, root / "work", root / "out-2", fresh)
            requests = read_jsonl(paths.request_results)
            coverage = read_jsonl(paths.coverage)

        self.assertEqual(len(fresh.calls), 0)
        self.assertTrue(requests[0]["reused"])
        self.assertEqual(requests[0]["status"], "success")
        self.assertEqual(coverage[0]["status"], "complete")


class EvidenceSchemaContractTests(unittest.TestCase):
    def test_schema_has_exact_bounds_and_enums(self) -> None:
        root = EVIDENCE_RESPONSE_JSON_SCHEMA
        documents = root["properties"]["documents"]
        document = documents["items"]
        claims = document["properties"]["claims"]
        claim = claims["items"]
        quote = claim["properties"]["quote"]

        self.assertFalse(root["additionalProperties"])
        self.assertEqual(root["required"], ["candidate_id", "documents"])
        self.assertEqual((documents["minItems"], documents["maxItems"]), (1, 6))
        self.assertFalse(document["additionalProperties"])
        self.assertEqual(document["required"], ["document_id", "claims"])
        self.assertEqual((claims["minItems"], claims["maxItems"]), (0, 1))
        self.assertFalse(claim["additionalProperties"])
        self.assertEqual(claim["required"], ["quote", "kind", "direction"])
        self.assertEqual((quote["minLength"], quote["maxLength"]), (20, 500))
        self.assertEqual(
            claim["properties"]["kind"]["enum"],
            [item.value for item in ClaimType],
        )
        self.assertEqual(
            claim["properties"]["direction"]["enum"],
            [item.value for item in EvidenceDirection],
        )

    def test_versions_change_only_executor_and_extractor_policy(self) -> None:
        self.assertEqual(EVIDENCE_MAX_OUTPUT_TOKENS, 2000)
        self.assertEqual(EVIDENCE_EXTRACTOR_VERSION, "evidence-llm-v5")
        self.assertEqual(
            LABELING_EVIDENCE_LLM_EXECUTOR_VERSION,
            "labeling-evidence-llm-executor-v6",
        )
        self.assertEqual(LABELING_EVIDENCE_LLM_WORK_VERSION, "labeling-evidence-llm-work-v1")
        self.assertEqual(LABELING_EVIDENCE_LLM_CACHE_VERSION, "labeling-evidence-llm-cache-v1")
        self.assertEqual(LABELING_EVIDENCE_LLM_RESULT_VERSION, "labeling-evidence-llm-result-v1")
        self.assertEqual(
            _extractor_id("yandex", "YandexGPT Lite 5"),
            "yandex-yandexgpt-lite-5-evidence-llm-v5",
        )


class ClaimValidationTests(unittest.TestCase):
    def _run_one(self, root: Path, claims):
        docs = [task_doc_row("c1", 1, "openalex", 1)]
        task = make_task_row("c1", docs)
        plan = build_input(root, [task], [run_coverage_row("c1", task)])
        generator = success_handler({"c1": [(docs[0]["document_id"], claims)]})
        paths = run_with_fake(plan, root / "work", root / "out", generator)
        return read_jsonl(paths.claims), read_jsonl(paths.issues), read_jsonl(
            paths.document_results
        )

    def test_verbatim_quote_gets_global_locator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            claims, _, documents = self._run_one(
                Path(directory), [(QUOTE, "research", "support")]
            )

        self.assertEqual(len(claims), 1)
        claim = claims[0]
        self.assertTrue(claim["claim_id"].startswith("claim-"))
        self.assertEqual(len(claim["claim_id"]), len("claim-") + 20)
        start = EXCERPT.find(QUOTE)
        self.assertEqual(claim["locator_start"], start)
        self.assertEqual(claim["locator_end"], start + len(QUOTE))
        self.assertEqual(claim["locator"], f"excerpt[{start}:{start + len(QUOTE)}]")
        self.assertEqual(documents[0]["claim_count"], 1)
        self.assertEqual(documents[0]["issue_count"], 0)
        expected = "claim-" + hashlib.sha256(
            f"c1|document-c1-1|research|support|{QUOTE}".encode()
        ).hexdigest()[:20]
        self.assertEqual(claim["claim_id"], expected)

    def test_title_only_quote_is_non_verbatim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            claims, issues, documents = self._run_one(
                Path(directory),
                [("Study of c1 mechanisms and methods here", "research", "support")],
            )

        self.assertEqual(claims, [])
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["code"], "non_verbatim")
        self.assertEqual(documents[0]["status"], "processed")
        self.assertEqual(documents[0]["claim_count"], 0)

    def test_verbatim_but_off_topic_quote_is_rejected(self) -> None:
        quote = "reported prototype measurements from a laboratory team"
        with tempfile.TemporaryDirectory() as directory:
            claims, issues, documents = self._run_one(
                Path(directory), [(quote, "prototype", "support")]
            )

        self.assertEqual(claims, [])
        self.assertEqual([issue["code"] for issue in issues], ["insufficient_term_match"])
        self.assertEqual(documents[0]["claim_count"], 0)
        self.assertEqual(documents[0]["issue_count"], 1)

    def test_claim_term_gate_boundaries(self) -> None:
        self.assertTrue(
            _claim_match_is_sufficient(
                "An air gapped sovereign system has implementation details",
                "air-gapped sovereign AI cloud",
            )
        )
        self.assertFalse(
            _claim_match_is_sufficient(
                "A self-powered electronic bandage may aid wound healing",
                "self-healing electronic skin",
            )
        )
        self.assertFalse(
            _claim_match_is_sufficient(
                "alpha and beta appear separately in factual implementation details",
                "alpha beta gamma",
            )
        )
        self.assertFalse(
            _claim_match_is_sufficient(
                "Alpha beta implementation details are reported",
                "alpha beta gamma",
            )
        )
        self.assertTrue(
            _claim_match_is_sufficient(
                "Alpha beta gamma implementation details are reported",
                "alpha beta gamma delta epsilon",
            )
        )

    def test_short_and_long_quotes_are_invalid(self) -> None:
        for quote in ("short", "x" * 501):
            with self.subTest(length=len(quote)), tempfile.TemporaryDirectory() as directory:
                claims, issues, _ = self._run_one(
                    Path(directory), [(quote, "research", "support")]
                )

            self.assertEqual(claims, [])
            self.assertEqual([issue["code"] for issue in issues], ["invalid_claim"])

    def test_unknown_kind_and_direction_are_invalid(self) -> None:
        cases = ((QUOTE, "magic", "support"), (QUOTE, "research", "sideways"))
        for claim in cases:
            with self.subTest(claim=claim), tempfile.TemporaryDirectory() as directory:
                claims, issues, _ = self._run_one(Path(directory), [claim])

            self.assertEqual(claims, [])
            self.assertEqual([issue["code"] for issue in issues], ["invalid_claim"])

    def test_multiple_claims_are_invalid_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            claims, issues, documents = self._run_one(
                Path(directory),
                [(QUOTE, "research", "support"), (QUOTE, "research", "support")],
            )

        self.assertEqual(claims, [])
        self.assertEqual([issue["code"] for issue in issues], ["invalid_response"])
        self.assertEqual(documents, [])


class ResponseFailureTests(unittest.TestCase):
    def _plan(self, root: Path):
        docs = [task_doc_row("c1", 1, "openalex", 1)]
        task = make_task_row("c1", docs)
        return build_input(root, [task], [run_coverage_row("c1", task)]), task

    def _failed(self, root, plan, generator):
        paths = run_with_fake(plan, root / "work", root / "out", generator)
        requests = read_jsonl(paths.request_results)
        issues = read_jsonl(paths.issues)
        coverage = read_jsonl(paths.coverage)
        manifest = json.loads((paths.manifest).read_text(encoding="utf-8"))
        return requests[0], issues, coverage[0], manifest

    def _answer_cases(self):
        docs = [task_doc_row("c1", 1, "openalex", 1)]
        task = make_task_row("c1", docs)
        full = answer("c1", [(docs[0]["document_id"], [(QUOTE, "research", "support")])])
        cases = {
            "missing": json.dumps({"candidate_id": "c1", "documents": []}),
            "extra": json.dumps({
                "candidate_id": "c1",
                "documents": [
                    {"document_id": docs[0]["document_id"], "claims": []},
                    {"document_id": "document-c1-999", "claims": []},
                ],
            }),
            "duplicate": json.dumps({
                "candidate_id": "c1",
                "documents": [
                    {"document_id": docs[0]["document_id"], "claims": []},
                    {"document_id": docs[0]["document_id"], "claims": []},
                ],
            }),
            "foreign": json.dumps({"candidate_id": "zzz", "documents": []}),
            "nonjson": "this is not json{{{",
        }
        return task, full, cases

    def test_missing_document_fails_whole_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task, _, cases = self._answer_cases()
            plan = build_input(
                root, [task],
                [run_coverage_row("c1", task)],
            )
            request, issues, coverage, manifest = self._failed(
                root, plan, scripted({"c1": cases["missing"]})
            )

            self.assertEqual(request["status"], "failed")
            self.assertEqual(request["error"], "invalid_response")
            self.assertFalse((root / "work" / "completed").exists() and any(
                (root / "work" / "completed").iterdir()))
            cycles = list((root / "work" / "failures").rglob("result.json"))
            self.assertEqual(len(cycles), 1)
            self.assertEqual(coverage["status"], "failed")
            self.assertEqual(manifest["analysis_status"], "partial")

    def test_extra_duplicate_foreign_documents_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for name in ("extra", "duplicate", "foreign"):
                case_dir = Path(directory) / name
                task, _, cases = self._answer_cases()
                plan = build_input(
                    case_dir, [task], [run_coverage_row("c1", task)]
                )
                request, _, _, _ = self._failed(
                    case_dir, plan, scripted({"c1": cases[name]})
                )
                self.assertEqual(request["status"], "failed", name)
                self.assertEqual(request["error"], "invalid_response", name)

    def test_non_json_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task, _, cases = self._answer_cases()
            plan = build_input(root, [task], [run_coverage_row("c1", task)])
            request, _, _, _ = self._failed(
                root, plan, scripted({"c1": cases["nonjson"]})
            )

        self.assertEqual(request["error"], "invalid_response")

    def test_model_error_classes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for name, error, code in (
                ("trunc", YandexTruncationError("cut"), "truncation"),
                ("filt", YandexContentFilterError("nope"), "content_filter"),
                ("boom", RuntimeError("down"), "model_error"),
            ):
                case_dir = Path(directory) / name
                task, _, _ = self._answer_cases()
                plan = build_input(
                    case_dir, [task], [run_coverage_row("c1", task)]
                )
                request, issues, _, _ = self._failed(
                    case_dir, plan, scripted({"c1": error})
                )
                self.assertEqual(request["error"], code, name)
                self.assertEqual(issues[0]["code"], code, name)


class ResumeTests(unittest.TestCase):
    def test_success_reused_without_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks, coverage = two_task_fixture()
            plan = build_input(root, tasks, coverage)
            first = success_handler({
                "c1": [(f"document-c1-{n}", [(QUOTE, "research", "support")])
                       for n in (1, 2)],
                "c2": [("document-c2-1", [(QUOTE, "pilot", "counter")])],
            })
            run_with_fake(plan, root / "work", root / "out-1", first)
            before = {
                path.name: (path / "result.json").read_bytes()
                for path in (root / "work" / "completed").iterdir()
            }
            fresh = success_handler({})
            paths = run_with_fake(plan, root / "work", root / "out-2", fresh)
            requests = read_jsonl(paths.request_results)

            self.assertEqual(len(fresh.calls), 0)
            self.assertTrue(all(row["reused"] for row in requests))
            after = {
                path.name: (path / "result.json").read_bytes()
                for path in (root / "work" / "completed").iterdir()
            }
            self.assertEqual(before, after)

    def test_failed_task_retried_next_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks, coverage = two_task_fixture()
            plan = build_input(root, tasks, coverage)
            failing = scripted({
                "c1": RuntimeError("down"),
                "c2": answer("c2", [("document-c2-1", [(QUOTE, "pilot", "counter")])]),
            })
            run_with_fake(plan, root / "work", root / "out-1", failing)
            fixed = success_handler({
                "c1": [(f"document-c1-{n}", [(QUOTE, "research", "support")])
                       for n in (1, 2)],
                "c2": [("document-c2-1", [(QUOTE, "pilot", "counter")])],
            })
            paths = run_with_fake(plan, root / "work", root / "out-2", fixed)
            requests = {row["candidate_id"]: row for row in read_jsonl(paths.request_results)}

        self.assertEqual(len(fixed.calls), 1)
        self.assertIn("c1", fixed.calls[0])
        self.assertTrue(requests["c2"]["reused"])
        self.assertFalse(requests["c1"]["reused"])
        self.assertEqual(requests["c1"]["status"], "success")

    def test_corrupt_caches_rejected_before_call(self) -> None:
        for variant in ("result", "raw", "manifest"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                tasks, coverage = two_task_fixture()
                plan = build_input(root, tasks, coverage)
                good = success_handler({
                    "c1": [(f"document-c1-{n}", []) for n in (1, 2)],
                    "c2": [("document-c2-1", [])],
                })
                run_with_fake(plan, root / "work", root / "out-1", good)
                completed = next((root / "work" / "completed").iterdir())
                if variant == "result":
                    (completed / "result.json").write_bytes(b"corrupt{{")
                elif variant == "raw":
                    raw_bytes = b"changed response text"
                    (completed / "raw_response.txt").write_bytes(raw_bytes)
                    cache_path = completed / "cache_manifest.json"
                    cache = json.loads(cache_path.read_text(encoding="utf-8"))
                    cache["raw_response"] = {
                        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                        "size_bytes": len(raw_bytes),
                    }
                    cache_path.write_text(
                        json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True)
                        + "\n",
                        encoding="utf-8",
                    )
                else:
                    (completed / "cache_manifest.json").write_bytes(b"corrupt{{")
                fresh = success_handler({})
                with self.assertRaises(ValueError):
                    run_with_fake(plan, root / "work", root / "out-2", fresh)
                self.assertEqual(len(fresh.calls), 0)

    def test_foreign_work_model_plan_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks, coverage = two_task_fixture()
            plan = build_input(root, tasks, coverage)
            good = success_handler({
                "c1": [(f"document-c1-{n}", []) for n in (1, 2)],
                "c2": [("document-c2-1", [])],
            })
            run_with_fake(plan, root / "work", root / "out-1", good)
            other_docs = [task_doc_row("c1", 1, "openalex", 1,
                                       excerpt="Different article text here.")]
            other_task = make_task_row("c1", other_docs)
            other_plan = build_input(
                root / "other", [other_task], [run_coverage_row("c1", other_task)]
            )
            fresh = success_handler({})
            with self.assertRaisesRegex(ValueError, "different inputs"):
                run_with_fake(other_plan, root / "work", root / "other-out", fresh)
            self.assertEqual(len(fresh.calls), 0)
            manifest_path = root / "work" / "work_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["model"] = "YandexGPT Pro 5"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "different inputs"):
                run_with_fake(plan, root / "work", root / "other-out-2", fresh)
            self.assertEqual(len(fresh.calls), 0)


    def test_old_extractor_work_fingerprint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks, coverage = two_task_fixture()
            plan = build_input(root, tasks, coverage)
            good = success_handler({
                "c1": [(f"document-c1-{n}", []) for n in (1, 2)],
                "c2": [("document-c2-1", [])],
            })
            run_with_fake(plan, root / "work", root / "out-1", good)
            manifest_path = root / "work" / "work_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["extractor_id"] = manifest["extractor_id"].replace(
                "evidence-llm-v5", "evidence-llm-v4"
            )
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            fresh = success_handler({})

            with self.assertRaisesRegex(ValueError, "different inputs"):
                run_with_fake(plan, root / "work", root / "out-2", fresh)

        self.assertEqual(len(fresh.calls), 0)


class PilotFilterTests(unittest.TestCase):
    def _three_tasks(self, root: Path):
        tasks = []
        for candidate in ("c1", "c2", "c3"):
            docs = [task_doc_row(candidate, 1, "openalex", 1)]
            tasks.append(make_task_row(candidate, docs))
        coverage = [run_coverage_row(task["candidate_id"], task) for task in tasks]
        return build_input(root, tasks, coverage)

    def _ok_handler(self):
        return success_handler({
            cid: [(f"document-{cid}-1", [(QUOTE, "research", "support")])]
            for cid in ("c1", "c2", "c3")
        })

    def test_max_new_tasks_counts_new_calls_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self._three_tasks(root)
            first = self._ok_handler()
            paths = run_with_fake(
                plan, root / "work", root / "out-1", first, max_new_tasks=1
            )
            first_requests = read_jsonl(paths.request_results)
            second = self._ok_handler()
            paths = run_with_fake(plan, root / "work", root / "out-2", second)
            second_requests = {row["candidate_id"]: row for row in read_jsonl(
                paths.request_results
            )}
            manifest = json.loads((paths.manifest).read_text(encoding="utf-8"))

        self.assertEqual(len(first.calls), 1)
        self.assertEqual(
            sum(1 for row in first_requests if row["status"] == "success"), 1
        )
        self.assertEqual(
            sum(1 for row in first_requests if row["status"] == "not_run"), 2
        )
        self.assertEqual(len(second.calls), 2)
        self.assertTrue(second_requests["c1"]["reused"])
        self.assertEqual(manifest["totals"]["called"], 2)
        self.assertEqual(manifest["totals"]["reused"], 1)

    def test_candidate_filter_selects_subset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self._three_tasks(root)
            generator = self._ok_handler()
            paths = run_with_fake(
                plan, root / "work", root / "out", generator, candidate_ids=["c2"]
            )
            requests = {row["candidate_id"]: row for row in read_jsonl(
                paths.request_results
            )}

        self.assertTrue(len(generator.calls) > 0)
        self.assertTrue(
            all("candidate_id: c2" in call for call in generator.calls)
        )
        self.assertTrue(
            all("candidate_id: c1" not in call for call in generator.calls)
        )
        self.assertEqual(requests["c2"]["status"], "success")
        self.assertEqual(requests["c1"]["status"], "not_run")
        self.assertEqual(requests["c3"]["status"], "not_run")

    def test_unknown_candidate_id_rejected_before_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self._three_tasks(root)
            generator = self._ok_handler()
            with self.assertRaisesRegex(ValueError, "unknown candidate_id filter"):
                run_with_fake(
                    plan, root / "work", root / "out", generator,
                    candidate_ids=["zzz"],
                )
            self.assertFalse((root / "work").exists())
            self.assertFalse((root / "out").exists())
            self.assertEqual(len(generator.calls), 0)

    def test_pilot_output_marks_not_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self._three_tasks(root)
            generator = self._ok_handler()
            paths = run_with_fake(
                plan, root / "work", root / "out", generator, max_new_tasks=2
            )
            requests = read_jsonl(paths.request_results)
            coverage = read_jsonl(paths.coverage)
            manifest = json.loads((paths.manifest).read_text(encoding="utf-8"))

        statuses = sorted(row["status"] for row in requests)
        self.assertEqual(statuses, ["not_run", "success", "success"])
        self.assertEqual(manifest["analysis_status"], "partial")
        cov = {row["candidate_id"]: row for row in coverage}
        self.assertEqual(cov["c3"]["status"], "not_run")

    def test_full_fake_run_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self._three_tasks(root)
            generator = self._ok_handler()
            paths = run_with_fake(plan, root / "work", root / "out", generator)
            manifest = json.loads((paths.manifest).read_text(encoding="utf-8"))

        self.assertEqual(manifest["analysis_status"], "complete")
        self.assertEqual(manifest["totals"]["success"], 3)
        self.assertEqual(manifest["totals"]["failed"], 0)
        self.assertEqual(manifest["totals"]["not_run"], 0)


class ConcurrencyDeterminismTests(unittest.TestCase):
    def test_concurrency_is_three(self) -> None:

        seen: dict = {}

        def factory(*args, **kwargs):
            seen.update(kwargs)
            return RealExecutor(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks, coverage = two_task_fixture()
            plan = build_input(root, tasks, coverage)
            handler = success_handler({
                "c1": [(f"document-c1-{n}", [(QUOTE, "research", "support")])
                       for n in (1, 2)],
                "c2": [("document-c2-1", [(QUOTE, "pilot", "counter")])],
            })
            with mock.patch(
                "nextwave.labeling.evidence_llm_run.ThreadPoolExecutor", factory
            ):
                run_with_fake(plan, root / "work", root / "out", handler)

        self.assertEqual(seen.get("max_workers"), 3)

    def test_identical_fakes_give_identical_bytes(self) -> None:
        def run_once(base: Path):
            tasks, coverage = two_task_fixture()
            plan = build_input(base, tasks, coverage)
            handler = success_handler({
                "c1": [(f"document-c1-{n}", [(QUOTE, "research", "support")])
                       for n in (1, 2)],
                "c2": [("document-c2-1", [(QUOTE, "pilot", "counter")])],
            })
            return run_with_fake(plan, base / "work", base / "out", handler)

        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            run_once(first)
            second = Path(directory) / "second"
            run_once(second)
            # Manifest records byte-exact input digests, so it legitimately
            # differs; all five data files must be byte-identical.
            for name in (
                "request_results.jsonl", "claims.jsonl", "document_results.jsonl",
                "issues.jsonl", "coverage.jsonl",
            ):
                self.assertEqual(
                    (first / "out" / name).read_bytes(),
                    (second / "out" / name).read_bytes(),
                    name,
                )


class SafetyTests(unittest.TestCase):
    def test_existing_output_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks, coverage = two_task_fixture()
            plan = build_input(root, tasks, coverage)
            (root / "out").mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                run_with_fake(plan, root / "work", root / "out",
                              success_handler({}))

    def test_preflight_error_creates_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks, coverage = two_task_fixture()
            tasks[0]["candidate_id"] = "zzz"
            plan = build_input(root, tasks, coverage)
            with self.assertRaises(ValueError):
                run_with_fake(plan, root / "work", root / "out",
                              success_handler({}))
            self.assertFalse((root / "work").exists())
            self.assertFalse((root / "out").exists())

    def test_secret_in_exception_stays_out(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks, coverage = two_task_fixture()
            plan = build_input(root, tasks, coverage)
            evil = scripted({
                "c1": RuntimeError(f"headers={{'Authorization': 'Bearer {SECRET_LIKE}'}}"),
                "c2": RuntimeError(f"cookies={SECRET_LIKE}"),
            })
            paths = run_with_fake(plan, root / "work", root / "out", evil)
            blob = b"".join(
                path.read_bytes()
                for path in sorted((root / "work").rglob("*"))
                if path.is_file()
            ) + b"".join(
                path.read_bytes()
                for path in sorted((root / "out").rglob("*"))
                if path.is_file()
            )
            issues = read_jsonl(paths.issues)

            self.assertNotIn(SECRET_LIKE.encode(), blob)
            self.assertTrue(all(issue["code"] == "model_error" for issue in issues))

    def test_prompt_has_no_schema_duplication(self) -> None:
        from nextwave.labeling.evidence_llm_plan import build_evidence_prompt as real_prompt

        settings = LlmRuntimeSettings(
            selection=LlmSelection(LlmProvider.YANDEX, "YandexGPT Lite 5"),
            api_key="fake-key",
            yandex_folder_id="fake-folder",
        )

        captured: dict = {}

        class FakeJsonTransport:
            def post_json(self, url, *, headers, payload, timeout_seconds):
                captured["text"] = payload["messages"][0]["text"]
                captured["payload"] = payload
                body = json.dumps({"result": {"alternatives": [{
                    "message": {"text": "{}"},
                    "status": "ALTERNATIVE_STATUS_FINAL",
                }]}}).encode()
                return HttpResponse(200, {}, body)

        generator = build_json_generator(
            settings,
            transport=FakeJsonTransport(),
            json_schema=EVIDENCE_RESPONSE_JSON_SCHEMA,
            schema_name="evidence_response",
            max_output_tokens=2000,
            server_side_json_schema=True,
        )
        task = {
            "candidate_id": "c1",
            "documents": [{
                "document_id": "d1",
                "source_class": "scientific",
                "title": "T",
                "url": "https://example.org/x",
                "passage": "Quantum error correction improves devices.",
                "matched_term": "quantum error correction",
            }],
        }
        expected = real_prompt(task)
        self.assertEqual(generator(expected), "{}")
        self.assertEqual(captured["text"], expected)
        self.assertNotIn("matching this JSON Schema", captured["text"])
        self.assertEqual(captured["payload"]["completionOptions"]["maxTokens"], "2000")
        self.assertEqual(
            captured["payload"]["jsonSchema"],
            {"schema": EVIDENCE_RESPONSE_JSON_SCHEMA},
        )
        self.assertNotIn("jsonObject", captured["payload"])

    def test_no_api_or_network_usage(self) -> None:
        module_path = (
            Path(__file__).resolve().parent.parent
            / "src" / "nextwave" / "labeling" / "evidence_llm_run.py"
        )
        text = module_path.read_text(encoding="utf-8")
        lowered = text.casefold()
        for token in (
            "urllib", "requests", "socket", "http.client", "openai",
            "anthropic", "Api-Key", "Authorization", "Bearer",
            "folder_id", "folder id", "os.environ", "getenv",
        ):
            self.assertNotIn(token, text)
            self.assertNotIn(token, lowered)
        # The only secret-adjacent literal allowed is the leak scanner itself.
        self.assertIn("def _check_no_secret_keys", text)


class PromptContractTests(unittest.TestCase):
    def _task(self) -> dict:
        return {
            "task_id": "task-abc",
            "candidate_id": "c1",
            "documents": [
                {
                    "document_id": "document-c1-1",
                    "source_class": "scientific",
                    "title": "Study of c1 mechanisms",
                    "url": "https://example.org/c1-1",
                    "passage": "Quantum error correction improves devices.",
                    "matched_term": "quantum error correction",
                }
            ],
        }

    def test_prompt_forbids_labels_and_allows_empty_claims(self) -> None:
        prompt = build_evidence_prompt(self._task())
        lowered = prompt.casefold()

        for forbidden in (
            "target", "expert", "planned_class", "planned class",
            "canonical_name", "canonical name", "domain", "why weak",
            "annotation",
        ):
            self.assertNotIn(forbidden, lowered)
        self.assertIn("empty claims", lowered)
        self.assertIn("title is not evidence", lowered)
        self.assertIn("untrusted", lowered)
        self.assertIn("promotional_claim", lowered)
        self.assertIn("support", lowered)
        self.assertIn("counter", lowered)
        self.assertIn("only json", lowered)

    def test_schema_uses_existing_enums(self) -> None:
        prompt = build_evidence_prompt(self._task())
        schema_text = prompt.split("Output JSON schema:", 1)[1]
        schema = json.loads(schema_text.split("\n\n", 1)[0])

        claim = schema["properties"]["documents"]["items"]["properties"]["claims"]
        self.assertEqual(claim["maxItems"], 1)
        kinds = schema["properties"]["documents"]["items"]["properties"]["claims"][
            "items"
        ]["properties"]["kind"]["enum"]
        directions = schema["properties"]["documents"]["items"]["properties"]["claims"][
            "items"
        ]["properties"]["direction"]["enum"]
        self.assertEqual(sorted(kinds), sorted(item.value for item in ClaimType))
        self.assertEqual(
            sorted(directions), sorted(item.value for item in EvidenceDirection)
        )
        self.assertFalse(schema["additionalProperties"])

    def test_prompt_is_deterministic(self) -> None:
        self.assertEqual(
            build_evidence_prompt(self._task()), build_evidence_prompt(self._task())
        )


class PreflightTests(unittest.TestCase):
    def _valid(self, root: Path):
        docs = [task_doc_row("c1", 1, "openalex", 1)]
        task = make_task_row("c1", docs)
        return build_input(root, [task], [run_coverage_row("c1", task)])

    def _run_fails(self, root: Path, input_dir: Path, pattern: str) -> None:
        with self.assertRaisesRegex(ValueError, pattern):
            run_with_fake(
                input_dir, root / "work", root / "out", FakeGenerator(lambda prompt: "{}")
            )
        self.assertFalse((root / "work").exists())
        self.assertFalse((root / "out").exists())

    def test_old_plan_v1_is_rejected_before_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self._valid(root)
            manifest_path = plan / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema_version"] = "labeling-evidence-llm-plan-v1"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            self._run_fails(root, plan, "does not match")

    def test_final_rank_gap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [
                task_doc_row("c1", 1, "openalex", 1),
                task_doc_row("c1", 2, "openalex", 3),
            ]
            task = make_task_row("c1", docs)
            input_dir = build_input(root, [task], [run_coverage_row("c1", task)])
            self._run_fails(root, input_dir, "1\\.\\.N")

    def test_extra_document_for_no_input_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [task_doc_row("c1", 1, "openalex", 1)]
            task = make_task_row("c1", docs)
            coverage = [run_coverage_row("c1", None)]
            input_dir = build_input(root, [task], coverage)
            self._run_fails(root, input_dir, "no_input coverage 'c1' has a task")

    def test_unknown_candidate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [task_doc_row("ghost", 1, "openalex", 1)]
            task = make_task_row("ghost", docs)
            coverage = [run_coverage_row("c1", None)]
            input_dir = build_input(root, [task], coverage)
            self._run_fails(root, input_dir, "unknown candidate")

    def test_future_dated_document_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [task_doc_row("c1", 1, "openalex", 1, published_at="2026-09-20")]
            task = make_task_row("c1", docs)
            input_dir = build_input(root, [task], [run_coverage_row("c1", task)])
            self._run_fails(root, input_dir, "past cutoff")

    def test_connector_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [task_doc_row("c1", 1, "openalex", 1, source_class="industry")]
            task = make_task_row("c1", docs)
            input_dir = build_input(root, [task], [run_coverage_row("c1", task)])
            self._run_fails(root, input_dir, "connector does not match")

    def test_corrupt_checksum_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = self._valid(root)
            with (input_dir / "coverage.jsonl").open("ab") as handle:
                handle.write(b" ")
            self._run_fails(root, input_dir, "mismatch with manifest")

    def test_existing_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = self._valid(root)
            (root / "out").mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                run_with_fake(
                    input_dir, root / "work", root / "out", FakeGenerator(lambda prompt: "{}")
                )

    def test_cli_default_comes_from_version_constant(self) -> None:
        from nextwave.__main__ import _build_parser
        from nextwave.labeling.evidence_llm_plan import (
            LABELING_EVIDENCE_LLM_PLAN_VERSION,
        )

        parser = _build_parser()
        defaults: dict = {}
        for sub in parser._subparsers._group_actions:
            for name, subparser in sub.choices.items():
                if name == "labeling-evidence-llm-plan":
                    defaults = {
                        item.dest: item.default for item in subparser._actions
                    }
        self.assertEqual(
            defaults["output"],
            Path("data") / "development" / LABELING_EVIDENCE_LLM_PLAN_VERSION,
        )


class TaskProvenanceTests(unittest.TestCase):
    def test_final_rank_present_and_ordered_in_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [
                task_doc_row("c1", number, "openalex", number) for number in range(1, 4)
            ]
            task = make_task_row("c1", docs)
            plan = build_input(root, [task], [run_coverage_row("c1", task)])
            generator = success_handler({
                "c1": [(doc["document_id"], [(QUOTE, "research", "support")])
                       for doc in docs],
            })
            run_with_fake(plan, root / "work", root / "out", generator)
            prompt = generator.calls[0]

        positions = [
            prompt.index(f"document_id: document-c1-{number}") for number in (1, 2, 3)
        ]
        self.assertEqual(positions, sorted(positions))
        for number in (1, 2, 3):
            self.assertIn(f"document-c1-{number}", prompt)

    def test_permutation_preserves_ranks_and_content(self) -> None:
        def run_once(base: Path, tasks, coverage):
            plan = build_input(base, tasks, coverage)
            handler = success_handler({
                task["candidate_id"]: [
                    (doc["document_id"], [(QUOTE, "research", "support")])
                    for doc in task["documents"]
                ]
                for task in tasks
            })
            return run_with_fake(plan, base / "work", base / "out", handler)

        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            docs_c1 = [task_doc_row("c1", n, "openalex", n) for n in (1, 2)]
            docs_c2 = [task_doc_row("c2", 1, "mediacloud", 1)]
            task_c1 = make_task_row("c1", docs_c1)
            task_c2 = make_task_row("c2", docs_c2)
            run_once(first, [task_c1, task_c2],
                     [run_coverage_row("c1", task_c1), run_coverage_row("c2", task_c2)])
            second = Path(directory) / "second"
            run_once(second, [task_c2, task_c1],
                     [run_coverage_row("c2", task_c2), run_coverage_row("c1", task_c1)])
            # Manifest records byte-exact input digests, so it legitimately
            # differs; all five data files must be byte-identical.
            names = (
                "request_results.jsonl", "claims.jsonl", "document_results.jsonl",
                "issues.jsonl", "coverage.jsonl",
            )
            for name in names:
                self.assertEqual(
                    (first / "out" / name).read_bytes(),
                    (second / "out" / name).read_bytes(),
                    name,
                )

    def test_no_input_with_count_one_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coverage = [{
                "candidate_id": "c1",
                "status": "no_input",
                "task_id": None,
                "input_documents": 1,
                "scientific_documents": 1,
                "industry_documents": 0,
                "prompt_chars": 0,
                "estimated_input_tokens": 0,
                "llm_call_planned": False,
                "empty_reasons": ["no_final_input"],
            }]
            input_dir = build_input(root, [], coverage)
            with self.assertRaisesRegex(ValueError, "must be 0"):
                run_with_fake(
                    input_dir, root / "work", root / "out",
                    FakeGenerator(lambda prompt: "{}"),
                )
            self.assertFalse((root / "work").exists())
            self.assertFalse((root / "out").exists())

    def test_false_flag_with_positive_count_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coverage = [{
                "candidate_id": "c1",
                "status": "planned",
                "task_id": "task-abc",
                "input_documents": 0,
                "scientific_documents": 1,
                "industry_documents": 0,
                "prompt_chars": 0,
                "estimated_input_tokens": 0,
                "llm_call_planned": True,
            }]
            input_dir = build_input(root, [], coverage)
            with self.assertRaisesRegex(ValueError, "must be fully empty|no task"):
                run_with_fake(
                    input_dir, root / "work", root / "out",
                    FakeGenerator(lambda prompt: "{}"),
                )
            self.assertFalse((root / "work").exists())

    def test_selected_sum_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [task_doc_row("c1", 1, "openalex", 1)]
            task = make_task_row("c1", docs)
            row = run_coverage_row("c1", task)
            row["scientific_documents"] = 0
            row["media_selected"] = 0
            input_dir = build_input(root, [task], [row])
            with self.assertRaisesRegex(ValueError, "do not add up"):
                run_with_fake(
                    input_dir, root / "work", root / "out",
                    FakeGenerator(lambda prompt: "{}"),
                )
            self.assertFalse((root / "work").exists())

    def test_bool_count_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [task_doc_row("c1", 1, "openalex", 1)]
            task = make_task_row("c1", docs)
            row = run_coverage_row("c1", task)
            row["input_documents"] = True
            input_dir = build_input(root, [task], [row])
            with self.assertRaisesRegex(ValueError, "must be a plain int"):
                run_with_fake(
                    input_dir, root / "work", root / "out",
                    FakeGenerator(lambda prompt: "{}"),
                )
            self.assertFalse((root / "work").exists())

    def test_non_http_url_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [task_doc_row("c1", 1, "openalex", 1, url="ftp://example.org/x")]
            task = make_task_row("c1", docs)
            input_dir = build_input(root, [task], [run_coverage_row("c1", task)])
            with self.assertRaisesRegex(ValueError, "absolute HTTP"):
                run_with_fake(
                    input_dir, root / "work", root / "out",
                    FakeGenerator(lambda prompt: "{}"),
                )
            self.assertFalse((root / "work").exists())

    def test_bool_relevance_score_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [task_doc_row("c1", 1, "openalex", 1, relevance_score=True)]
            task = make_task_row("c1", docs)
            input_dir = build_input(root, [task], [run_coverage_row("c1", task)])
            with self.assertRaisesRegex(ValueError, "plain int 0\\.\\.100"):
                run_with_fake(
                    input_dir, root / "work", root / "out",
                    FakeGenerator(lambda prompt: "{}"),
                )
            self.assertFalse((root / "work").exists())

    def test_unknown_trust_tier_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [task_doc_row("c1", 1, "openalex", 1, trust_tier="S")]
            task = make_task_row("c1", docs)
            input_dir = build_input(root, [task], [run_coverage_row("c1", task)])
            with self.assertRaisesRegex(ValueError, "trust_tier"):
                run_with_fake(
                    input_dir, root / "work", root / "out",
                    FakeGenerator(lambda prompt: "{}"),
                )
            self.assertFalse((root / "work").exists())

    def test_schema_directions_come_from_shared_contracts(self) -> None:
        from nextwave.contracts import EvidenceDirection as SharedDirection
        from nextwave.labeling import evidence_llm_run as run_module

        self.assertIs(run_module.EvidenceDirection, SharedDirection)
        prompt = build_evidence_prompt({
            "task_id": "task-x",
            "candidate_id": "c1",
            "documents": [{
                "document_id": "d1",
                "source_class": "scientific",
                "title": "T",
                "url": "https://example.org/x",
                "passage": "Quantum error correction improves devices.",
                "matched_term": "quantum error correction",
            }],
        })
        schema_text = prompt.split("Output JSON schema:", 1)[1]
        schema = json.loads(schema_text.split("\n\n", 1)[0])
        directions = schema["properties"]["documents"]["items"]["properties"][
            "claims"]["items"]["properties"]["direction"]["enum"]
        self.assertEqual(sorted(directions), sorted(item.value for item in SharedDirection))
        module_path = (
            Path(__file__).resolve().parent.parent
            / "src" / "nextwave" / "labeling" / "evidence_llm_run.py"
        )
        self.assertNotIn(
            "labeling.contracts", module_path.read_text(encoding="utf-8")
        )


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


class TaskKeyWhitelistTests(unittest.TestCase):
    def test_passage_with_organizer_words_passes_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            excerpt = (
                "The target protein annotation study shows expert-level "
                "methods for quantum error correction in planned class trials."
            )
            docs = [task_doc_row("c1", 1, "openalex", 1, excerpt=excerpt)]
            task = make_task_row("c1", docs)
            plan = build_input(root, [task], [run_coverage_row("c1", task)])
            generator = success_handler({"c1": [(docs[0]["document_id"], [])]})
            paths = run_with_fake(plan, root / "work", root / "out", generator)
            requests = read_jsonl(paths.request_results)

        self.assertEqual(requests[0]["status"], "success")

    def test_extra_task_key_target_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [task_doc_row("c1", 1, "openalex", 1)]
            task = make_task_row("c1", docs)
            task["target"] = 1
            plan = build_input(root, [task], [run_coverage_row("c1", task)])
            with self.assertRaisesRegex(ValueError, "unexpected task keys"):
                run_with_fake(
                    plan, root / "work", root / "out",
                    FakeGenerator(lambda prompt: "{}"),
                )
            self.assertFalse((root / "work").exists())
            self.assertFalse((root / "out").exists())

    def test_extra_document_key_expert_label_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [task_doc_row("c1", 1, "openalex", 1)]
            task = make_task_row("c1", docs)
            task["documents"][0]["expert_label"] = "mature"
            plan = build_input(root, [task], [run_coverage_row("c1", task)])
            with self.assertRaisesRegex(ValueError, "unexpected keys"):
                run_with_fake(
                    plan, root / "work", root / "out",
                    FakeGenerator(lambda prompt: "{}"),
                )
            self.assertFalse((root / "work").exists())

    def test_organizer_fields_absent_from_prompt(self) -> None:
        prompt = build_evidence_prompt({
            "task_id": "task-x",
            "candidate_id": "c1",
            "documents": [{
                "document_id": "d1",
                "source_class": "scientific",
                "title": "T",
                "url": "https://example.org/x",
                "passage": "Quantum error correction improves devices.",
                "matched_term": "quantum error correction",
            }],
        })
        for token in ('"target"', '"expert_label"', '"planned_class"', '"canonical_name"'):
            self.assertNotIn(token, prompt)


class CacheReconstructionTests(unittest.TestCase):
    def _run_once(self, root: Path):
        tasks, coverage = two_task_fixture()
        plan = build_input(root, tasks, coverage)
        good = success_handler({
            "c1": [(f"document-c1-{n}", [(QUOTE, "research", "support")])
                   for n in (1, 2)],
            "c2": [("document-c2-1", [(QUOTE, "pilot", "counter")])],
        })
        run_with_fake(plan, root / "work", root / "out-1", good)
        return plan

    def _resume_rejected(self, root, plan, pattern: str) -> None:
        fresh = success_handler({})
        with self.assertRaisesRegex(ValueError, pattern):
            run_with_fake(plan, root / "work", root / "out-2", fresh)
        self.assertEqual(len(fresh.calls), 0)

    def test_tampered_document_results_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self._run_once(root)

            def mutate(result: dict) -> None:
                result["document_results"][0]["claim_count"] = 99

            tamper_completed(root / "work", mutate)
            self._resume_rejected(root, plan, "result diverges")

    def test_tampered_issues_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self._run_once(root)

            def mutate(result: dict) -> None:
                result["issues"].append({
                    "candidate_id": result["candidate_id"],
                    "task_id": result["task_id"],
                    "document_id": None,
                    "code": "invalid_claim",
                    "message": "forged",
                })

            tamper_completed(root / "work", mutate)
            self._resume_rejected(root, plan, "result diverges")

    def test_removed_claim_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self._run_once(root)

            def mutate(result: dict) -> None:
                del result["claims"][-1]
                result["document_results"][0]["claim_count"] -= 1

            tamper_completed(root / "work", mutate)
            self._resume_rejected(root, plan, "result diverges")

    def test_added_duplicate_claim_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self._run_once(root)

            def mutate(result: dict) -> None:
                result["claims"].append(dict(result["claims"][0]))
                result["document_results"][0]["claim_count"] += 1

            tamper_completed(root / "work", mutate)
            self._resume_rejected(root, plan, "result diverges")

    def test_changed_raw_response_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self._run_once(root)
            completed = next((root / "work" / "completed").iterdir())
            raw_path = completed / "raw_response.txt"
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            raw["documents"][0]["claims"].append({
                "quote": QUOTE,
                "kind": "growth",
                "direction": "counter",
            })
            raw_bytes = (
                json.dumps(raw, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            raw_path.write_bytes(raw_bytes)
            cache_path = completed / "cache_manifest.json"
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            cache["raw_response"] = {
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "size_bytes": len(raw_bytes),
            }
            cache_path.write_text(
                json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._resume_rejected(root, plan, "too many claims")

    def test_changed_provider_identity_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self._run_once(root)
            completed = next((root / "work" / "completed").iterdir())
            cache_path = completed / "cache_manifest.json"
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            cache["model"] = "YandexGPT Pro 5"
            cache_path.write_text(
                json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._resume_rejected(root, plan, "model diverges")

    def test_empty_claims_reused_without_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks, coverage = two_task_fixture()
            plan = build_input(root, tasks, coverage)
            good = success_handler({
                "c1": [(f"document-c1-{n}", []) for n in (1, 2)],
                "c2": [("document-c2-1", [])],
            })
            run_with_fake(plan, root / "work", root / "out-1", good)
            fresh = success_handler({})
            paths = run_with_fake(plan, root / "work", root / "out-2", fresh)
            requests = read_jsonl(paths.request_results)
            claims = read_jsonl(paths.claims)

        self.assertEqual(len(fresh.calls), 0)
        self.assertTrue(all(row["reused"] for row in requests))
        self.assertEqual(claims, [])


class CoverageUniquenessTests(unittest.TestCase):
    def test_duplicate_no_input_coverage_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks, coverage = two_task_fixture()
            plan = build_input(
                root, tasks,
                coverage + [run_coverage_row("c9", None), run_coverage_row("c9", None)],
            )
            with self.assertRaisesRegex(ValueError, "duplicates candidate"):
                run_with_fake(
                    plan, root / "work", root / "out",
                    FakeGenerator(lambda prompt: "{}"),
                )
            self.assertFalse((root / "work").exists())
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
