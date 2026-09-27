"""Evidence LLM plan tests: offline fixtures only, no network, no model calls."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from nextwave.contracts import ClaimType
from nextwave.labeling.contracts import EvidenceDirection
from nextwave.labeling.evidence_llm_plan import (
    LABELING_EVIDENCE_LLM_PLAN_VERSION,
    build_evidence_prompt,
    export_evidence_llm_plan,
    select_passage,
)
from nextwave.labeling.evidence_term_policy import text_supports_matched_term

BUNDLE = "bundle-test-001"


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


def doc_row(candidate: str, number: int, connector: str, rank: int, **overrides) -> dict:
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
        "search_id": f"search-{candidate}",
        "request_id": f"request-{candidate}",
        "snapshot_id": "snapshot-1",
        "matched_term": "quantum error correction",
        "matched_tokens": ["correction", "error", "quantum"],
        "relevance_score": 80,
        "relevance_class": "strong",
        "excerpt": (
            "Quantum error correction with surface codes improves devices "
            "and reported prototype measurements from a laboratory team."
        ),
        "evidence_text_available": True,
    }
    payload.update(overrides)
    return payload


def coverage_row(candidate: str, count: int, **overrides) -> dict:
    payload = {
        "candidate_id": candidate,
        "final_evidence_input_count": count,
        "has_evidence_input": count > 0,
        "scientific_selected": count,
        "media_selected": 0,
        "source_classes_present": ["scientific"] if count > 0 else [],
        "empty_reasons": [] if count > 0 else ["no_scientific_available"],
    }
    payload.update(overrides)
    return payload


def build_input(
    root: Path, documents: list[dict], coverage: list[dict]
) -> Path:
    input_dir = root / "input"
    input_dir.mkdir(parents=True)
    documents_bytes = dump_jsonl(documents)
    coverage_bytes = dump_jsonl(coverage)
    (input_dir / "evidence_input_documents.jsonl").write_bytes(documents_bytes)
    (input_dir / "coverage.jsonl").write_bytes(coverage_bytes)
    manifest = {
        "schema_version": "labeling-evidence-input-plan-v4",
        "bundle_id": BUNDLE,
        "cutoff_date": "2026-09-15",
        "totals": {
            "candidates": len(coverage),
            "evidence_input_documents": len(documents),
        },
        "outputs": {
            "evidence_input_documents.jsonl": digest(documents_bytes),
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


def six_doc_fixture() -> tuple[list[dict], list[dict]]:
    docs = [
        doc_row("c1", number, "openalex" if number % 2 else "mediacloud", number)
        for number in range(1, 7)
    ]
    return docs, [coverage_row("c1", 6)]


class PassageSelectionTests(unittest.TestCase):
    def test_shared_policy_requires_enough_tokens_and_a_term_bigram(self) -> None:
        self.assertTrue(
            text_supports_matched_term(
                "Quantum error correction improves devices.",
                "quantum error correction platform",
            )
        )
        self.assertFalse(
            text_supports_matched_term(
                "Quantum devices use error controls and correction methods.",
                "quantum error correction platform",
            )
        )
        self.assertFalse(
            text_supports_matched_term(
                "Quantum error methods.", "quantum error correction platform"
            )
        )

    def test_shared_policy_handles_two_and_one_token_terms(self) -> None:
        self.assertTrue(text_supports_matched_term("Edge AI runs here.", "edge ai"))
        self.assertFalse(text_supports_matched_term("AI at the edge.", "edge ai"))
        self.assertTrue(text_supports_matched_term("RAG is evaluated.", "RAG"))

    def test_short_excerpt_passes_through_whole(self) -> None:
        excerpt = "Quantum error correction improves devices."
        passage, start, end, truncated, source_sha, passage_sha = select_passage(
            excerpt, "quantum error correction"
        )

        self.assertEqual((passage, start, truncated), (excerpt, 0, False))
        self.assertEqual(end, len(excerpt))
        self.assertEqual(source_sha, hashlib.sha256(excerpt.encode()).hexdigest())
        self.assertEqual(passage_sha, source_sha)
        self.assertEqual(excerpt[start:end], passage)

    def test_long_excerpt_yields_exact_substring(self) -> None:
        excerpt = "Alpha processes. " + "x" * 4000 + " Quantum error correction works."
        passage, start, end, truncated, source_sha, passage_sha = select_passage(
            excerpt, "quantum error correction"
        )

        self.assertTrue(truncated)
        self.assertLessEqual(len(passage), 3000)
        self.assertEqual(excerpt[start:end], passage)
        self.assertIn("Quantum error correction", passage)
        self.assertEqual(
            passage_sha, hashlib.sha256(passage.encode("utf-8")).hexdigest()
        )
        self.assertEqual(
            source_sha, hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
        )

    def test_relevant_tail_beats_generic_head(self) -> None:
        head = "Introduction to devices. " + "Market overview. " * 200
        tail = "Quantum error correction with surface codes improves devices."
        excerpt = head + " " + tail
        self.assertGreater(len(excerpt), 3000)
        passage, start, _, _, _, _ = select_passage(
            excerpt, "quantum error correction"
        )

        self.assertGreater(start, 0)
        self.assertIn("surface codes", passage)

    def test_title_does_not_select_passage(self) -> None:
        # The term sits in the title-like head text, not as evidence-worthy text;
        # selection must depend on chunk content alone, identically with any title.
        excerpt = "Alpha processes. " + "y" * 4000 + " Quantum error correction works."
        first = select_passage(excerpt, "quantum error correction")
        # Same excerpt rescored: deterministic regardless of any title context.
        second = select_passage(excerpt, "quantum error correction")

        self.assertEqual(first, second)
        self.assertIn("Quantum error correction", first[0])

    def test_exact_phrase_beats_scattered_overlap(self) -> None:
        scattered = "quantum " + " ".join(f"pad{i}" for i in range(60)) + " error correction"
        exact = "Review of quantum error correction methods."
        excerpt = scattered + " " + ("Filler text. " * 200) + " " + exact
        self.assertGreater(len(excerpt), 3000)
        passage, _, _, _, _, _ = select_passage(
            excerpt, "quantum error correction"
        )

        self.assertIn("Review of quantum error correction", passage)

    def test_narrow_window_beats_medium_window(self) -> None:
        narrow = "quantum error correction improves devices"
        medium = "quantum " + " ".join(f"m{i}" for i in range(50)) + " error correction"
        excerpt = medium + " " + ("Padding words. " * 200) + " " + narrow
        self.assertGreater(len(excerpt), 3000)
        passage, _, _, _, _, _ = select_passage(
            excerpt, "quantum error correction"
        )

        self.assertIn("improves devices", passage)

    def test_medium_window_beats_scattered_tokens(self) -> None:
        medium = "quantum " + " ".join(f"m{i}" for i in range(50)) + " error correction"
        scattered = "quantum " + " ".join(f"s{i}" for i in range(400)) + " error correction"
        excerpt = scattered + " " + ("Padding words. " * 60) + " " + medium
        self.assertGreater(len(excerpt), 3000)
        passage, _, _, _, _, _ = select_passage(
            excerpt, "quantum error correction"
        )

        self.assertIn("m49", passage)
        self.assertNotIn("s399", passage)

    def test_eligible_later_chunk_beats_ineligible_earlier_chunk(self) -> None:
        excerpt = (
            "Quantum devices use error controls and correction methods. "
            + "x" * 3200
            + " eligible-marker quantum error correction works."
        )
        passage, _, _, _, _, _ = select_passage(
            excerpt, "quantum error correction"
        )

        self.assertIn("eligible-marker quantum error correction", passage)
        self.assertTrue(text_supports_matched_term(passage, "quantum error correction"))


class TaskAssemblyTests(unittest.TestCase):
    def test_six_documents_make_one_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs, coverage = six_doc_fixture()
            input_dir = build_input(root, docs, coverage)
            paths = export_evidence_llm_plan(
                input_dir=input_dir, output_dir=root / "out"
            )
            tasks = read_jsonl(paths.tasks)

        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task["candidate_id"], "c1")
        self.assertEqual(task["document_count"], 6)
        self.assertEqual(len(task["documents"]), 6)
        self.assertTrue(task["task_id"].startswith("task-"))
        self.assertEqual(len(task["task_id"]), len("task-") + 20)
        self.assertEqual(
            [doc["document_id"] for doc in task["documents"]],
            [f"document-c1-{number}" for number in range(1, 7)],
        )
        prompt = build_evidence_prompt(task)
        self.assertEqual(len(prompt), task["prompt_chars"])
        self.assertEqual(
            task["prompt_sha256"],
            hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(task["estimated_input_tokens"], math.ceil(len(prompt) / 3))

    def test_document_count_not_task_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [
                doc_row("c1", number, "openalex", number) for number in range(1, 4)
            ] + [
                doc_row("c2", number, "mediacloud", number) for number in range(1, 3)
            ]
            coverage = [coverage_row("c1", 3), coverage_row("c2", 2)]
            input_dir = build_input(root, docs, coverage)
            paths = export_evidence_llm_plan(
                input_dir=input_dir, output_dir=root / "out"
            )
            tasks = read_jsonl(paths.tasks)
            manifest = json.loads((paths.manifest).read_text(encoding="utf-8"))

        self.assertEqual(len(tasks), 2)
        self.assertNotEqual(len(tasks), 5)
        self.assertEqual(manifest["schema_version"], "labeling-evidence-llm-plan-v8")
        self.assertEqual(manifest["claim_policy"]["max_claims_per_document"], 1)
        self.assertEqual(manifest["totals"]["planned_tasks"], 2)
        self.assertEqual(manifest["totals"]["input_documents"], 5)

    def test_ineligible_passage_is_removed_and_remaining_rank_is_compacted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [
                doc_row(
                    "c1",
                    1,
                    "openalex",
                    1,
                    excerpt=(
                        "Quantum devices use error controls and correction methods."
                    ),
                ),
                doc_row("c1", 2, "openalex", 2),
            ]
            paths = export_evidence_llm_plan(
                input_dir=build_input(root, docs, [coverage_row("c1", 2)]),
                output_dir=root / "out",
            )
            tasks = read_jsonl(paths.tasks)
            manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))

        self.assertEqual(tasks[0]["document_count"], 1)
        self.assertEqual(tasks[0]["documents"][0]["document_id"], "document-c1-2")
        self.assertEqual(tasks[0]["documents"][0]["final_rank"], 1)
        self.assertEqual(manifest["totals"]["source_input_documents"], 2)
        self.assertEqual(manifest["totals"]["input_documents"], 1)
        self.assertEqual(manifest["totals"]["excluded_ineligible_passages"], 1)

    def test_all_ineligible_passages_create_no_input_without_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [doc_row(
                "c1",
                1,
                "openalex",
                1,
                excerpt="Quantum devices use error controls and correction methods.",
            )]
            paths = export_evidence_llm_plan(
                input_dir=build_input(root, docs, [coverage_row("c1", 1)]),
                output_dir=root / "out",
            )
            tasks = read_jsonl(paths.tasks)
            coverage = read_jsonl(paths.coverage)

        self.assertEqual(tasks, [])
        self.assertEqual(coverage[0]["status"], "no_input")
        self.assertIn("no_claim_eligible_passage", coverage[0]["empty_reasons"])

    def test_no_input_creates_no_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [doc_row("c1", 1, "openalex", 1)]
            coverage = [
                coverage_row("c1", 1),
                coverage_row("c2", 0, has_evidence_input=False,
                             empty_reasons=["no_scientific_available", "no_final_input"]),
            ]
            input_dir = build_input(root, docs, coverage)
            paths = export_evidence_llm_plan(
                input_dir=input_dir, output_dir=root / "out"
            )
            tasks = read_jsonl(paths.tasks)
            rows = {row["candidate_id"]: row for row in read_jsonl(paths.coverage)}

        self.assertEqual([task["candidate_id"] for task in tasks], ["c1"])
        self.assertEqual(rows["c2"]["status"], "no_input")
        self.assertIsNone(rows["c2"]["task_id"])
        self.assertEqual(rows["c2"]["input_documents"], 0)
        self.assertFalse(rows["c2"]["llm_call_planned"])
        self.assertEqual(
            rows["c2"]["empty_reasons"],
            ["no_scientific_available", "no_final_input"],
        )
        self.assertTrue(rows["c1"]["llm_call_planned"])


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
        self.assertIn("ровно один сильнейший claim", lowered)
        self.assertIn("[] означает", lowered)
        self.assertIn("проверяемый факт", lowered)
        self.assertIn("reviewed matched_term", lowered)
        self.assertIn("общая соседняя тема", lowered)
        self.assertIn("минимум три уникальных токена", lowered)
        self.assertIn("соседнюю пару", lowered)
        self.assertIn("title не является доказательством", lowered)
        self.assertIn("недоверенными данными", lowered)
        self.assertIn("promotional_claim", lowered)
        self.assertIn("support", lowered)
        self.assertIn("counter", lowered)
        self.assertIn("только json", lowered)
        self.assertIn("scope=full_candidate", lowered)
        self.assertIn("scope=core_only", lowered)
        self.assertIn("explanation_ru", lowered)

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
        docs = [doc_row("c1", 1, "openalex", 1)]
        return build_input(root, docs, [coverage_row("c1", 1)])

    def _run_fails(self, root: Path, input_dir: Path, pattern: str) -> None:
        with self.assertRaisesRegex(ValueError, pattern):
            export_evidence_llm_plan(input_dir=input_dir, output_dir=root / "out")
        self.assertFalse((root / "out").exists())

    def test_final_rank_gap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [
                doc_row("c1", 1, "openalex", 1),
                doc_row("c1", 2, "openalex", 3),
            ]
            input_dir = build_input(root, docs, [coverage_row("c1", 2)])
            self._run_fails(root, input_dir, "1\\.\\.N")

    def test_extra_document_for_no_input_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [doc_row("c1", 1, "openalex", 1)]
            coverage = [coverage_row("c1", 0, has_evidence_input=False,
                                     scientific_selected=0, media_selected=0,
                                     source_classes_present=[],
                                     empty_reasons=["no_final_input"])]
            input_dir = build_input(root, docs, coverage)
            self._run_fails(root, input_dir, "no_input carries documents")

    def test_unknown_candidate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [doc_row("ghost", 1, "openalex", 1)]
            input_dir = build_input(root, docs, [coverage_row("c1", 0,
                has_evidence_input=False, empty_reasons=["no_final_input"])])
            self._run_fails(root, input_dir, "unknown candidate")

    def test_future_dated_document_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [doc_row("c1", 1, "openalex", 1, published_at="2026-09-20")]
            input_dir = build_input(root, docs, [coverage_row("c1", 1)])
            self._run_fails(root, input_dir, "past cutoff")

    def test_connector_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # scientific rows must use openalex; industry rows must use mediacloud.
            bad = [doc_row("c1", 1, "openalex", 1, source_class="industry")]
            input_dir = build_input(root, bad, [coverage_row("c1", 1)])
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
                export_evidence_llm_plan(input_dir=input_dir, output_dir=root / "out")

    def test_permuted_rows_give_identical_outputs(self) -> None:
        # Provenance digests (input_manifest_sha256, task_id) track exact input
        # bytes by design, so they legitimately differ after permutation; every
        # other byte of tasks.jsonl and coverage.jsonl must be identical.
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            docs, coverage = six_doc_fixture()
            input_a = build_input(first, docs, coverage)
            export_evidence_llm_plan(input_dir=input_a, output_dir=first / "out")
            second = Path(directory) / "second"
            input_b = build_input(second, list(reversed(docs)), coverage)
            export_evidence_llm_plan(input_dir=input_b, output_dir=second / "out")
            tasks_a = read_jsonl(first / "out" / "tasks.jsonl")
            tasks_b = read_jsonl(second / "out" / "tasks.jsonl")
            self.assertEqual(len(tasks_a), len(tasks_b))
            for row_a, row_b in zip(tasks_a, tasks_b, strict=True):
                self.assertEqual(row_a["candidate_id"], row_b["candidate_id"])
                self.assertEqual(row_a["document_count"], row_b["document_count"])
                self.assertEqual(row_a["prompt_chars"], row_b["prompt_chars"])
                self.assertEqual(
                    row_a["estimated_input_tokens"], row_b["estimated_input_tokens"]
                )
                self.assertEqual(row_a["documents"], row_b["documents"])
                self.assertEqual(row_a["prompt_sha256"], row_b["prompt_sha256"])
                self.assertRegex(row_a["input_manifest_sha256"], r"\A[0-9a-f]{64}\Z")
                self.assertRegex(row_a["task_id"], r"\Atask-[0-9a-f]{20}\Z")
            rows_a = read_jsonl(first / "out" / "coverage.jsonl")
            rows_b = read_jsonl(second / "out" / "coverage.jsonl")
            self.assertEqual(len(rows_a), len(rows_b))
            for row_a, row_b in zip(rows_a, rows_b, strict=True):
                if row_a["task_id"] is not None:
                    self.assertRegex(row_a["task_id"], r"\Atask-[0-9a-f]{20}\Z")
                scrubbed_a = dict(row_a, task_id=None)
                scrubbed_b = dict(row_b, task_id=None)
                self.assertEqual(scrubbed_a, scrubbed_b)

    def test_cli_default_comes_from_version_constant(self) -> None:
        from nextwave.__main__ import _build_parser

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

    def test_no_transport_or_api_usage(self) -> None:
        module_path = (
            Path(__file__).resolve().parent.parent
            / "src" / "nextwave" / "labeling" / "evidence_llm_plan.py"
        )
        text = module_path.read_text(encoding="utf-8").casefold()
        for token in (
            "urllib", "requests", "socket", "http.client", "yandex", "openai",
            "anthropic", "transport", "sklearn", "sentence_transformers",
            "embedding", "llm_client", "api_key",
        ):
            self.assertNotIn(token, text)


class TaskProvenanceTests(unittest.TestCase):
    def test_final_rank_present_and_ordered_in_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs, coverage = six_doc_fixture()
            input_dir = build_input(root, docs, coverage)
            paths = export_evidence_llm_plan(
                input_dir=input_dir, output_dir=root / "out"
            )
            tasks = read_jsonl(paths.tasks)

        self.assertEqual(len(tasks), 1)
        ranks = [doc["final_rank"] for doc in tasks[0]["documents"]]
        self.assertEqual(ranks, [1, 2, 3, 4, 5, 6])
        self.assertTrue(
            all(isinstance(rank, int) and not isinstance(rank, bool) for rank in ranks)
        )

    def test_permutation_preserves_ranks_and_content_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            docs, coverage = six_doc_fixture()
            input_a = build_input(first, docs, coverage)
            export_evidence_llm_plan(input_dir=input_a, output_dir=first / "out")
            second = Path(directory) / "second"
            input_b = build_input(second, list(reversed(docs)), coverage)
            export_evidence_llm_plan(input_dir=input_b, output_dir=second / "out")
            tasks_a = read_jsonl(first / "out" / "tasks.jsonl")
            tasks_b = read_jsonl(second / "out" / "tasks.jsonl")

        for row in tasks_a + tasks_b:
            self.assertEqual(
                [doc["final_rank"] for doc in row["documents"]],
                list(range(1, row["document_count"] + 1)),
            )
        scrubbed_a = [
            {key: value for key, value in row.items()
             if key not in ("input_manifest_sha256", "task_id")}
            for row in tasks_a
        ]
        scrubbed_b = [
            {key: value for key, value in row.items()
             if key not in ("input_manifest_sha256", "task_id")}
            for row in tasks_b
        ]
        self.assertEqual(scrubbed_a, scrubbed_b)

    def test_no_input_with_count_one_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coverage = [coverage_row("c1", 1, has_evidence_input=False,
                                     scientific_selected=1, media_selected=0,
                                     source_classes_present=["scientific"],
                                     empty_reasons=["no_final_input"])]
            input_dir = build_input(root, [], coverage)
            with self.assertRaisesRegex(ValueError, "disagrees with final count"):
                export_evidence_llm_plan(input_dir=input_dir, output_dir=root / "out")
            self.assertFalse((root / "out").exists())

    def test_false_flag_with_positive_count_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coverage = [coverage_row("c1", 0, has_evidence_input=False,
                                     scientific_selected=0, media_selected=0,
                                     source_classes_present=["scientific"],
                                     empty_reasons=["no_final_input"])]
            input_dir = build_input(root, [], coverage)
            with self.assertRaisesRegex(ValueError, "no_input must be fully empty"):
                export_evidence_llm_plan(input_dir=input_dir, output_dir=root / "out")
            self.assertFalse((root / "out").exists())

    def test_selected_sum_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [doc_row("c1", 1, "openalex", 1)]
            coverage = [coverage_row("c1", 1, scientific_selected=0, media_selected=0)]
            input_dir = build_input(root, docs, coverage)
            with self.assertRaisesRegex(ValueError, "diverges from final count"):
                export_evidence_llm_plan(input_dir=input_dir, output_dir=root / "out")
            self.assertFalse((root / "out").exists())

    def test_bool_count_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [doc_row("c1", 1, "openalex", 1)]
            coverage = [coverage_row("c1", True)]
            input_dir = build_input(root, docs, coverage)
            with self.assertRaisesRegex(ValueError, "non-negative plain int"):
                export_evidence_llm_plan(input_dir=input_dir, output_dir=root / "out")
            self.assertFalse((root / "out").exists())

    def test_non_http_url_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [doc_row("c1", 1, "openalex", 1, url="ftp://example.org/x")]
            input_dir = build_input(root, docs, [coverage_row("c1", 1)])
            with self.assertRaisesRegex(ValueError, "absolute HTTP"):
                export_evidence_llm_plan(input_dir=input_dir, output_dir=root / "out")
            self.assertFalse((root / "out").exists())

    def test_bool_relevance_score_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [doc_row("c1", 1, "openalex", 1, relevance_score=True)]
            input_dir = build_input(root, docs, [coverage_row("c1", 1)])
            with self.assertRaisesRegex(ValueError, "plain int 0\\.\\.100"):
                export_evidence_llm_plan(input_dir=input_dir, output_dir=root / "out")
            self.assertFalse((root / "out").exists())

    def test_unknown_trust_tier_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = [doc_row("c1", 1, "openalex", 1, trust_tier="S")]
            input_dir = build_input(root, docs, [coverage_row("c1", 1)])
            with self.assertRaisesRegex(ValueError, "trust_tier"):
                export_evidence_llm_plan(input_dir=input_dir, output_dir=root / "out")
            self.assertFalse((root / "out").exists())

    def test_schema_directions_come_from_shared_contracts(self) -> None:
        from nextwave.contracts import EvidenceDirection as SharedDirection
        from nextwave.labeling import evidence_llm_plan as planner

        self.assertIs(planner.EvidenceDirection, SharedDirection)
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
            / "src" / "nextwave" / "labeling" / "evidence_llm_plan.py"
        )
        self.assertNotIn(
            "labeling.contracts", module_path.read_text(encoding="utf-8")
        )


if __name__ == "__main__":
    unittest.main()
