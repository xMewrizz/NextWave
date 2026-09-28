from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from nextwave.discovery import (
    QUALIFICATION_GATE_ID,
    CandidateGateDecision,
    CandidateGateResult,
    GateDecision,
    GateReason,
)
from nextwave.labeling.enrichment_plan import export_target_enrichment_plan
from nextwave.labeling.enrichment_run import ENRICHMENT_RESULT_VERSION
from nextwave.labeling.target_gate import run_target_gate


def _digest(payload: bytes) -> dict:
    return {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _jsonl(rows: list[dict]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _target(
    *, canonical_name: str = "AI-native bank", aliases: list[str] | None = None
) -> dict:
    return {
        "candidate_id": "target-001",
        "canonical_name": canonical_name,
        "aliases": [] if aliases is None else aliases,
        "domain": "Финтех",
        "analysis_scope_key": "fintech-v1",
        "source_query": "Emerging fintech technologies",
    }


def _write_plan(root: Path, target: dict) -> Path:
    source = root / "targets.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": "labeling-target-candidates-v1",
                "cutoff_date": "2026-09-15",
                "candidates": [target],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output = root / "plan"
    export_target_enrichment_plan(candidates_file=source, output_dir=output)
    return output


def _document(
    plan: dict,
    *,
    number: int = 1,
    title: str = "The AI native bank platform launches",
    excerpt: str | None = "A concrete AI-native bank system is described.",
    published_at: str | None = "2026-08-01",
) -> dict:
    candidate = plan["candidates"][0]
    search = next(
        item for item in candidate["searches"] if item["connector"] == "openalex"
    )
    request = search["requests"][0]
    return {
        "authors": [],
        "automatic_translation": False,
        "candidate_id": candidate["candidate_id"],
        "canonical_url": f"https://example.test/paper-{number}",
        "connector": "openalex",
        "connector_id": "openalex",
        "document_id": f"document-openalex-{number:03d}",
        "doi": f"10.1000/example-{number}",
        "excerpt": excerpt,
        "external_id": f"W{number}",
        "generated_summary": False,
        "language": "en",
        "observed_at": "2026-08-02T00:00:00+00:00",
        "organizations": [],
        "origin_confidence": 1.0,
        "origin_id": f"doi:10.1000/example-{number}",
        "origin_method": "doi",
        "published_at": published_at,
        "publisher": "Example Journal",
        "request_id": request["request_id"],
        "retrieved_at": "2026-09-14T00:00:00+00:00",
        "search_id": search["search_id"],
        "snapshot_id": "snapshot-test",
        "source_type": "scientific_publication",
        "title": title,
        "trust_tier": "A",
        "url": f"https://example.test/paper-{number}",
    }


def _write_result(
    root: Path,
    plan_dir: Path,
    documents: list[dict],
    *,
    incomplete: bool = False,
) -> Path:
    output = root / "result"
    output.mkdir()
    plan_raw = (plan_dir / "plan.json").read_bytes()
    plan = json.loads(plan_raw)
    candidate = plan["candidates"][0]
    coverage = []
    for search in candidate["searches"]:
        connector = search["connector"]
        count = sum(row["connector_id"] == connector for row in documents)
        planned = len(search["requests"])
        coverage.append(
            {
                "candidate_id": candidate["candidate_id"],
                "connector": connector,
                "failed_request_ids": ["failed"] if incomplete else [],
                "failed_requests": 1 if incomplete else 0,
                "incompleteness_reasons": ["test"] if incomplete else [],
                "parse_issue_count": 0,
                "planned_requests": planned,
                "returned_documents": count,
                "returned_records": count,
                "reused_requests": 0,
                "search_id": search["search_id"],
                "source_class": search["source_class"],
                "status": "partial" if incomplete else "complete",
                "successful_requests": planned - 1 if incomplete else planned,
            }
        )
    document_raw = _jsonl(documents)
    coverage_raw = _jsonl(coverage)
    (output / "documents.jsonl").write_bytes(document_raw)
    (output / "coverage.jsonl").write_bytes(coverage_raw)
    manifest = {
        "schema_version": ENRICHMENT_RESULT_VERSION,
        "bundle_id": plan["bundle"]["bundle_id"],
        "executor_version": "labeling-enrichment-executor-v2",
        "plan": _digest(plan_raw),
        "outputs": {
            "documents.jsonl": _digest(document_raw),
            "coverage.jsonl": _digest(coverage_raw),
        },
        "totals": {"candidates": 1, "returned_documents": len(documents)},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return output


class FakeGate:
    def __init__(self, *, gate_id: str = QUALIFICATION_GATE_ID) -> None:
        self.gate_id = gate_id
        self.calls = []

    def evaluate(self, scope, batch, documents, *, max_concurrency=5):
        self.calls.append((scope, batch, documents))
        decisions = tuple(
            CandidateGateDecision(
                proposal_id=proposal.proposal_id,
                decision=GateDecision.ACCEPT,
                reason=GateReason.CONCRETE_TECHNOLOGY,
                basis_document_ids=(proposal.document_ids[0],),
                explanation="The cited source names a concrete technology.",
            )
            for proposal in batch.proposals
        )
        return CandidateGateResult(
            scope.scope_id,
            self.gate_id,
            tuple(proposal.proposal_id for proposal in batch.proposals),
            decisions,
            (),
            1,
        )


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class TargetGateTests(unittest.TestCase):
    def test_canonical_exact_sequence_calls_gate_and_round_trips_locator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_dir = _write_plan(root, _target())
            plan = json.loads((plan_dir / "plan.json").read_text(encoding="utf-8"))
            result_dir = _write_result(root, plan_dir, [_document(plan)])
            gate = FakeGate()
            paths = run_target_gate(
                plan_dir=plan_dir,
                result_dir=result_dir,
                output_dir=root / "output",
                gate=gate,
            )
            result = _rows(paths.gate_results)[0]
            grounding = _rows(paths.groundings)[0]

        self.assertEqual(len(gate.calls), 1)
        self.assertEqual(result["promotion_status"], "evidence_review_required")
        self.assertEqual(result["gate_decision"], "accept")
        self.assertEqual(grounding["quote"], "AI native bank")
        self.assertEqual(
            grounding["locator"],
            f"title[{grounding['start']}:{grounding['end']}]",
        )

    def test_noncontiguous_reordered_and_partial_words_do_not_ground(self) -> None:
        titles = (
            "AI systems make every native service useful to a bank",
            "A bank uses native AI",
            "AI-native banking platform",
        )
        for title in titles:
            with self.subTest(title=title), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                plan_dir = _write_plan(root, _target())
                plan = json.loads((plan_dir / "plan.json").read_text(encoding="utf-8"))
                result_dir = _write_result(
                    root,
                    plan_dir,
                    [_document(plan, title=title, excerpt="No matching phrase here.")],
                )
                gate = FakeGate()
                paths = run_target_gate(
                    plan_dir=plan_dir,
                    result_dir=result_dir,
                    output_dir=root / "output",
                    gate=gate,
                )
                result = _rows(paths.gate_results)[0]
            self.assertEqual(gate.calls, [])
            self.assertEqual(result["promotion_status"], "not_grounded")

    def test_alias_only_requires_review_without_gate_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_dir = _write_plan(
                root,
                _target(
                    canonical_name="autonomous cyber immune system",
                    aliases=["ACIS", "digital immune system"],
                ),
            )
            plan = json.loads((plan_dir / "plan.json").read_text(encoding="utf-8"))
            result_dir = _write_result(
                root,
                plan_dir,
                [_document(plan, title="A digital immune system is proposed", excerpt=None)],
            )
            gate = FakeGate()
            paths = run_target_gate(
                plan_dir=plan_dir,
                result_dir=result_dir,
                output_dir=root / "output",
                gate=gate,
            )
            result = _rows(paths.gate_results)[0]

        self.assertEqual(gate.calls, [])
        self.assertEqual(result["promotion_status"], "alias_review_required")
        self.assertEqual(result["ineligible_short_terms"], ["ACIS"])
        self.assertFalse(result["gate_called"])

    def test_one_token_alias_never_grounds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_dir = _write_plan(
                root,
                _target(
                    canonical_name="autonomous cyber immune system", aliases=["ACIS"]
                ),
            )
            plan = json.loads((plan_dir / "plan.json").read_text(encoding="utf-8"))
            result_dir = _write_result(
                root,
                plan_dir,
                [_document(plan, title="The ACIS instrument launched", excerpt=None)],
            )
            gate = FakeGate()
            paths = run_target_gate(
                plan_dir=plan_dir,
                result_dir=result_dir,
                output_dir=root / "output",
                gate=gate,
            )
            result = _rows(paths.gate_results)[0]

        self.assertEqual(gate.calls, [])
        self.assertEqual(result["promotion_status"], "not_grounded")

    def test_gate_receives_only_canonical_grounded_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_dir = _write_plan(root, _target())
            plan = json.loads((plan_dir / "plan.json").read_text(encoding="utf-8"))
            documents = [
                _document(plan, number=1),
                _document(
                    plan,
                    number=2,
                    title="Unrelated finance story",
                    excerpt="No exact technology sequence.",
                ),
            ]
            result_dir = _write_result(root, plan_dir, documents)
            gate = FakeGate()
            run_target_gate(
                plan_dir=plan_dir,
                result_dir=result_dir,
                output_dir=root / "output",
                gate=gate,
            )

        supplied = gate.calls[0][2]
        self.assertEqual([item.document_id for item in supplied], ["document-openalex-001"])

    def test_undated_document_is_excluded_from_grounding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_dir = _write_plan(root, _target())
            plan = json.loads((plan_dir / "plan.json").read_text(encoding="utf-8"))
            result_dir = _write_result(
                root, plan_dir, [_document(plan, published_at=None)]
            )
            gate = FakeGate()
            paths = run_target_gate(
                plan_dir=plan_dir,
                result_dir=result_dir,
                output_dir=root / "output",
                gate=gate,
            )
            result = _rows(paths.gate_results)[0]

        self.assertEqual(gate.calls, [])
        self.assertEqual(result["promotion_status"], "not_grounded")

    def test_post_cutoff_document_is_rejected_before_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_dir = _write_plan(root, _target())
            plan = json.loads((plan_dir / "plan.json").read_text(encoding="utf-8"))
            result_dir = _write_result(
                root, plan_dir, [_document(plan, published_at="2026-09-16")]
            )
            gate = FakeGate()
            with self.assertRaisesRegex(ValueError, "after the labeling cutoff"):
                run_target_gate(
                    plan_dir=plan_dir,
                    result_dir=result_dir,
                    output_dir=root / "output",
                    gate=gate,
                )
            self.assertFalse((root / "output").exists())
        self.assertEqual(gate.calls, [])

    def test_incomplete_coverage_is_rejected_before_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_dir = _write_plan(root, _target())
            plan = json.loads((plan_dir / "plan.json").read_text(encoding="utf-8"))
            result_dir = _write_result(
                root, plan_dir, [_document(plan)], incomplete=True
            )
            gate = FakeGate()
            with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
                run_target_gate(
                    plan_dir=plan_dir,
                    result_dir=result_dir,
                    output_dir=root / "output",
                    gate=gate,
                )
        self.assertEqual(gate.calls, [])

    def test_tampered_documents_are_rejected_before_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_dir = _write_plan(root, _target())
            plan = json.loads((plan_dir / "plan.json").read_text(encoding="utf-8"))
            result_dir = _write_result(root, plan_dir, [_document(plan)])
            with (result_dir / "documents.jsonl").open("ab") as stream:
                stream.write(b" ")
            gate = FakeGate()
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                run_target_gate(
                    plan_dir=plan_dir,
                    result_dir=result_dir,
                    output_dir=root / "output",
                    gate=gate,
                )
        self.assertEqual(gate.calls, [])

    def test_non_qualification_gate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_dir = _write_plan(root, _target())
            plan = json.loads((plan_dir / "plan.json").read_text(encoding="utf-8"))
            result_dir = _write_result(root, plan_dir, [_document(plan)])
            gate = FakeGate(gate_id="yandex-yandexgpt-5.1-candidate-gate-v4")
            with self.assertRaisesRegex(ValueError, "qualification Gate"):
                run_target_gate(
                    plan_dir=plan_dir,
                    result_dir=result_dir,
                    output_dir=root / "output",
                    gate=gate,
                )
            self.assertEqual(gate.calls, [])


if __name__ == "__main__":
    unittest.main()
