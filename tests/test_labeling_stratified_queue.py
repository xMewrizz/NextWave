"""Stratified neutral queue: strata, dedup, honest coverage, deficits."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import openpyxl

from nextwave.discovery.run_store import DiscoveryRun
from nextwave.labeling.queue import (
    CandidateSlot,
    _queue_identity_key,
    build_labeling_queue,
    queue_to_jsonl,
)

CUTOFF = "2026-09-15"


def strat_group(
    group_id: str,
    canonical_name: str,
    *,
    connector_ids: list[str] | None = None,
    origins: list[str] | None = None,
    documents: list[str] | None = None,
) -> dict:
    origin_ids = origins if origins is not None else [f"origin-{group_id}-0"]
    document_ids = documents if documents is not None else [f"document-{group_id}-0"]
    return {
        "group_id": group_id,
        "canonical_name": canonical_name,
        "aliases": [],
        "origin_ids": origin_ids,
        "document_ids": document_ids,
        "connector_ids": connector_ids or [],
        "normalization_key": canonical_name.casefold(),
        "proposal_ids": [f"proposal-{group_id}"],
    }


def strat_run(
    run_id: str,
    *,
    groups: list | None = None,
    domain: str = "Edge",
    raw_query: str = "Технологии в ИИ",
    scientific_status: str = "complete",
    scientific_usage: dict | None = None,
    media_status: str = "failed",
    media_provider: str | None = None,
    media_usage: list | dict | None = None,
) -> DiscoveryRun:
    plan = {
        "plan_id": f"plan-{run_id}",
        "analysis_id": f"analysis-{run_id}",
        "scope": {"scope_id": f"scope-{run_id}", "raw_query": raw_query},
    }
    result = {
        "alias_resolution": {"groups": groups or [], "review_suggestions": []},
        "candidate_proposals": {"proposals": [], "exclusions": []},
        "candidate_gate": {"gate_id": "gate-1", "decisions": []},
        "text_extraction": {"issues": []},
        "evidence_extraction": {"proposals": []},
        "scientific": {"status": scientific_status, "documents": []}
        | ({"usage": scientific_usage} if scientific_usage is not None else {}),
        "media": {"status": media_status, "provider_used": media_provider, "documents": []}
        | ({"usage": media_usage} if media_usage is not None else {}),
        "verification": {"results": []},
    }
    manifest = {"run_id": run_id, "cutoff_date": CUTOFF, "domain": domain}
    return DiscoveryRun(
        run_id=run_id, run_dir=Path(run_id), plan=plan, result=result, manifest=manifest
    )


def sci_usage(requests: int, stop: str) -> dict:
    return {
        "connector_id": "openalex",
        "requests_used": requests,
        "pages_used": 1,
        "returned_records": 1,
        "accepted_records": 1,
        "unique_documents": 1,
        "duplicate_documents": 0,
        "rejected_records": 0,
        "elapsed_ms": 10,
        "stop_reason": stop,
    }


def media_usage_list(requests: int, stop: str, connector: str = "mediacloud") -> list:
    return [
        {
            "connector_id": connector,
            "requests_used": requests,
            "pages_used": 1,
            "returned_records": 1,
            "accepted_records": 1,
            "unique_documents": 1,
            "duplicate_documents": 0,
            "rejected_records": 0,
            "elapsed_ms": 10,
            "stop_reason": stop,
        }
    ]


class StratifiedSelectionTests(unittest.TestCase):
    def test_round_robin_takes_from_available_strata(self) -> None:
        run = strat_run(
            "run-1",
            groups=[
                strat_group(
                    "group-single", "Single Tech", connector_ids=["openalex"],
                    origins=["o1"], documents=["d1"],
                ),
                strat_group(
                    "group-multi", "Multi Tech", connector_ids=["openalex"],
                    origins=["o1", "o2"], documents=["d1", "d2"],
                ),
                strat_group(
                    "group-media", "Media Tech", connector_ids=["mediacloud"],
                    origins=["o1"], documents=["d1"],
                ),
                strat_group(
                    "group-cross", "Cross Tech", connector_ids=["openalex", "mediacloud"],
                    origins=["o1"], documents=["d1"],
                ),
            ],
        )
        slots = tuple(f"team-negative-{i:03d}" for i in range(1, 5))
        queue = build_labeling_queue(
            (run,),
            candidate_slots=tuple(CandidateSlot(cid, "Edge") for cid in slots),
        )
        self.assertEqual(
            [item.canonical_name for item in queue.candidates],
            ["Cross Tech", "Media Tech", "Multi Tech", "Single Tech"],
        )
        self.assertEqual(
            [item.selection_stratum for item in queue.candidates],
            ["cross_source", "media_present", "multi_origin", "single_origin"],
        )

    def test_empty_stratum_does_not_create_deficit(self) -> None:
        run = strat_run(
            "run-1",
            groups=[
                strat_group(
                    "group-multi", "Multi Tech", connector_ids=["openalex"],
                    origins=["o1", "o2"], documents=["d1"],
                ),
                strat_group(
                    "group-single", "Single Tech", connector_ids=["openalex"],
                    origins=["o1"], documents=["d1"],
                ),
            ],
        )
        slots = (
            CandidateSlot("team-negative-001", "Edge"),
            CandidateSlot("team-negative-002", "Edge"),
        )
        queue = build_labeling_queue((run,), candidate_slots=slots)
        self.assertEqual(len(queue.candidates), 2)
        self.assertEqual(queue.deficits, ())
        self.assertEqual(
            [item.canonical_name for item in queue.candidates],
            ["Multi Tech", "Single Tech"],
        )

    def test_result_deterministic_on_permuted_runs(self) -> None:
        run_a = strat_run(
            "run-a",
            groups=[strat_group("group-a", "Alpha Tech", connector_ids=["openalex"])],
        )
        run_b = strat_run(
            "run-b",
            groups=[
                strat_group(
                    "group-b", "Beta Tech", connector_ids=["mediacloud"],
                    origins=["o1"], documents=["d1"],
                )
            ],
        )
        slots = (
            CandidateSlot("team-negative-001", "Edge"),
            CandidateSlot("team-negative-002", "Edge"),
        )
        first = build_labeling_queue((run_a, run_b), candidate_slots=slots)
        second = build_labeling_queue((run_b, run_a), candidate_slots=slots)
        self.assertEqual(
            [item.canonical_name for item in first.candidates],
            [item.canonical_name for item in second.candidates],
        )
        self.assertEqual(
            [item.group_id for item in first.candidates],
            [item.group_id for item in second.candidates],
        )
        self.assertEqual(queue_to_jsonl(first)[0], queue_to_jsonl(second)[0])


class QueueDuplicateTests(unittest.TestCase):
    def test_transformer_variants_share_one_slot(self) -> None:
        run = strat_run(
            "run-1",
            groups=[
                strat_group("group-t1", "Transformer", connector_ids=["openalex"]),
                strat_group("group-t2", "transformers", connector_ids=["openalex"]),
            ],
        )
        slots = (
            CandidateSlot("team-negative-001", "Edge"),
            CandidateSlot("team-negative-002", "Edge"),
        )
        queue = build_labeling_queue((run,), candidate_slots=slots)
        self.assertEqual(len(queue.candidates), 1)
        self.assertEqual(queue.candidates[0].canonical_name, "Transformer")
        duplicates = [item for item in queue.overflow if item.kind == "candidate_duplicate"]
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(duplicates[0].reason, "equivalent queue identity")
        self.assertEqual(
            [(item.area, item.need, item.missing) for item in queue.deficits],
            [("Edge", "candidate_review", 1)],
        )

    def test_cnn_singular_plural_share_one_slot(self) -> None:
        run = strat_run(
            "run-1",
            groups=[
                strat_group(
                    "group-c1", "Convolutional Neural Network (CNN)",
                    connector_ids=["openalex"],
                ),
                strat_group(
                    "group-c2", "Convolutional Neural Networks (CNNs)",
                    connector_ids=["openalex"],
                ),
            ],
        )
        slots = (CandidateSlot("team-negative-001", "Edge"),)
        queue = build_labeling_queue((run,), candidate_slots=slots)
        self.assertEqual(len(queue.candidates), 1)
        self.assertTrue(
            any(item.kind == "candidate_duplicate" for item in queue.overflow)
        )

    def test_different_technologies_not_merged(self) -> None:
        run = strat_run(
            "run-1",
            groups=[
                strat_group("group-rag", "RAG", connector_ids=["openalex"]),
                strat_group(
                    "group-rag-long", "Retrieval-Augmented Generation",
                    connector_ids=["openalex"],
                ),
                strat_group("group-tr1", "Transformer", connector_ids=["openalex"]),
                strat_group("group-tr2", "Transducer", connector_ids=["openalex"]),
            ],
        )
        slots = tuple(CandidateSlot(f"team-negative-{i:03d}", "Edge") for i in range(1, 5))
        queue = build_labeling_queue((run,), candidate_slots=slots)
        self.assertEqual(len(queue.candidates), 4)
        self.assertFalse(
            any(item.kind == "candidate_duplicate" for item in queue.overflow)
        )

    def test_identity_key_plural_exceptions(self) -> None:
        self.assertEqual(_queue_identity_key("Transformer"), _queue_identity_key("transformers"))
        self.assertEqual(
            _queue_identity_key("Convolutional Neural Network (CNN)"),
            _queue_identity_key("Convolutional Neural Networks (CNNs)"),
        )
        self.assertEqual(_queue_identity_key("my_tech"), _queue_identity_key("my-tech"))
        self.assertEqual(_queue_identity_key("My Tech"), _queue_identity_key("my-tech"))
        # ss/us/is endings are never stripped.
        self.assertEqual(_queue_identity_key("Glass"), "glass")
        self.assertEqual(_queue_identity_key("Status"), "status")
        self.assertEqual(_queue_identity_key("Analysis"), "analysis")
        self.assertNotEqual(
            _queue_identity_key("RAG"),
            _queue_identity_key("Retrieval-Augmented Generation"),
        )
        self.assertNotEqual(_queue_identity_key("Transformer"), _queue_identity_key("Transducer"))

    def test_same_name_in_different_domains_keeps_both_candidates(self) -> None:
        edge = strat_run(
            "run-edge",
            domain="Edge",
            groups=[strat_group("group-edge", "Transformer", connector_ids=["openalex"])],
        )
        fintech = strat_run(
            "run-fintech",
            domain="Финтех",
            groups=[
                strat_group("group-fintech", "transformers", connector_ids=["openalex"])
            ],
        )
        queue = build_labeling_queue(
            (edge, fintech),
            candidate_slots=(
                CandidateSlot("team-negative-001", "Edge"),
                CandidateSlot("team-negative-002", "Финтех"),
            ),
        )

        self.assertEqual(
            [(item.domain, item.canonical_name) for item in queue.candidates],
            [("Edge", "Transformer"), ("Финтех", "transformers")],
        )
        self.assertFalse(
            any(item.kind == "candidate_duplicate" for item in queue.overflow)
        )


class HonestCoverageTests(unittest.TestCase):
    def test_partial_page_budget_counts_as_scientific_with_note(self) -> None:
        run = strat_run(
            "run-1",
            scientific_status="partial",
            scientific_usage=sci_usage(5, "page_budget"),
            media_status="failed",
        )
        coverage = build_labeling_queue((run,)).search_coverage[0]
        self.assertEqual(coverage.source_classes, ("scientific",))
        self.assertEqual(coverage.notes, "scientific=partial(page_budget)")

    def test_complete_with_requests_counts_and_notes_provider(self) -> None:
        run = strat_run(
            "run-1",
            scientific_status="complete",
            scientific_usage=sci_usage(5, "channels_exhausted"),
            media_status="complete",
            media_provider="mediacloud",
            media_usage=media_usage_list(2, "channels_exhausted"),
        )
        coverage = build_labeling_queue((run,)).search_coverage[0]
        self.assertEqual(coverage.source_classes, ("scientific", "industry"))
        self.assertEqual(
            coverage.notes,
            "scientific=complete(channels_exhausted);industry=complete(mediacloud)",
        )

    def test_failed_or_zero_request_is_not_coverage(self) -> None:
        failed = strat_run(
            "run-1",
            scientific_status="failed",
            scientific_usage=sci_usage(5, "page_budget"),
            media_status="complete",
            media_provider="mediacloud",
            media_usage=media_usage_list(2, "channels_exhausted"),
        )
        self.assertEqual(
            build_labeling_queue((failed,)).search_coverage[0].source_classes,
            ("industry",),
        )
        zero_sci = strat_run(
            "run-2",
            scientific_status="partial",
            scientific_usage=sci_usage(0, "page_budget"),
            media_status="failed",
        )
        self.assertEqual(
            build_labeling_queue((zero_sci,)).search_coverage[0].source_classes, ()
        )
        zero_media = strat_run(
            "run-3",
            scientific_status="failed",
            media_status="complete",
            media_provider="mediacloud",
            media_usage=media_usage_list(0, "channels_exhausted"),
        )
        self.assertEqual(
            build_labeling_queue((zero_media,)).search_coverage[0].source_classes, ()
        )
        unknown = strat_run(
            "run-4",
            scientific_status="failed",
            media_status="complete",
            media_provider="unexpected-provider",
            media_usage=media_usage_list(2, "channels_exhausted"),
        )
        self.assertEqual(
            build_labeling_queue((unknown,)).search_coverage[0].source_classes, ()
        )

    def test_complete_without_usage_is_not_coverage(self) -> None:
        run = strat_run(
            "run-1",
            scientific_status="complete",
            media_status="complete",
            media_provider="mediacloud",
        )
        coverage = build_labeling_queue((run,)).search_coverage[0]
        self.assertEqual(coverage.source_classes, ())


class DeficitAggregationTests(unittest.TestCase):
    def test_sixteen_missing_slots_become_one_deficit(self) -> None:
        run = strat_run("run-1", groups=[])
        slots = tuple(CandidateSlot(f"team-negative-{i:03d}", "Edge") for i in range(1, 17))
        queue = build_labeling_queue((run,), candidate_slots=slots)
        self.assertEqual(queue.candidates, ())
        self.assertEqual(
            [(item.area, item.need, item.missing) for item in queue.deficits],
            [("Edge", "candidate_review", 16)],
        )


class ContractPreservationTests(unittest.TestCase):
    def test_jsonl_format_has_no_stratum_field(self) -> None:
        run = strat_run(
            "run-1",
            groups=[strat_group("group-a", "Alpha Tech", connector_ids=["openalex"])],
        )
        queue = build_labeling_queue(
            (run,), candidate_slots=(CandidateSlot("team-negative-001", "Edge"),)
        )
        negative_bytes, _ = queue_to_jsonl(queue)
        payload = json.loads(negative_bytes.decode("utf-8").strip())
        self.assertNotIn("selection_stratum", payload)
        self.assertEqual(
            sorted(payload.keys()),
            sorted(
                [
                    "schema_version",
                    "candidate_id",
                    "canonical_name",
                    "aliases",
                    "group_id",
                    "source_query",
                    "domain",
                    "analysis_scope_key",
                    "cutoff_date",
                ]
            ),
        )
        self.assertEqual(payload["schema_version"], "negative-candidate-v1")
        self.assertEqual(queue.candidates[0].selection_stratum, "single_origin")


class WorkbookNotesTests(unittest.TestCase):
    def test_decision_row_carries_machine_coverage_note(self) -> None:
        from nextwave.labeling.workbook import fill_labeling_workbook

        run = strat_run(
            "run-1",
            groups=[strat_group("group-a", "Alpha Tech", connector_ids=["openalex"])],
            scientific_status="partial",
            scientific_usage=sci_usage(5, "page_budget"),
            media_status="complete",
            media_provider="mediacloud",
            media_usage=media_usage_list(2, "channels_exhausted"),
        )
        queue = build_labeling_queue(
            (run,), candidate_slots=(CandidateSlot("team-negative-001", "Edge"),)
        )
        filled = fill_labeling_workbook(Path("templates/labeling_workbook.xlsx"), queue)
        workbook = openpyxl.load_workbook(BytesIO(filled), read_only=True, data_only=True)
        try:
            decisions = workbook["Решения"]
            found = False
            for row in range(5, decisions.max_row + 1):
                if decisions[f"C{row}"].value == "team-negative-001":
                    self.assertEqual(decisions[f"M{row}"].value, "Технологии в ИИ")
                    self.assertEqual(decisions[f"N{row}"].value, "scientific|industry")
                    self.assertEqual(
                        decisions[f"P{row}"].value,
                        "scientific=partial(page_budget);industry=complete(mediacloud)",
                    )
                    found = True
                    break
            self.assertTrue(found)
        finally:
            workbook.close()


class ExportStrataTests(unittest.TestCase):
    def test_export_manifest_reports_strata_and_aggregated_deficits(self) -> None:
        from nextwave.labeling.export import export_labeling_bundle

        def pretty(payload: dict) -> bytes:
            return (
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_id = "run-strata"
            group_cross = strat_group(
                "group-cross", "Cross Tech",
                connector_ids=["openalex", "mediacloud"],
                origins=["o1"], documents=["doc-cross"],
            )
            group_single = strat_group(
                "group-single", "Single Tech",
                connector_ids=["openalex"],
                origins=["o1"], documents=["doc-single"],
            )
            plan = {
                "plan_id": f"plan-{run_id}",
                "scope": {"scope_id": f"scope-{run_id}", "raw_query": "Технологии в ИИ"},
                "query": {"cutoff_date": CUTOFF},
            }
            result = {
                "pipeline_version": "discovery-pipeline-v9",
                "alias_resolution": {
                    "groups": [group_cross, group_single],
                    "review_suggestions": [],
                },
                "candidate_proposals": {"proposals": [], "exclusions": []},
                "candidate_gate": {
                    "gate_id": "yandex-yandexgpt-lite-5-candidate-gate-v4",
                    "input_proposal_ids": [],
                    "decisions": [],
                },
                "gate_coverage": {
                    "status": "complete",
                    "total_proposals": 0,
                    "checked_proposals": 0,
                    "skipped_proposals": 0,
                    "skipped_proposal_ids": [],
                },
                "text_extraction": {"issues": []},
                "evidence_extraction": {"proposals": []},
                "scientific": {"status": "complete", "documents": []},
                "media": {
                    "status": "complete",
                    "provider_used": "mediacloud",
                    "documents": [],
                },
                "verification": {"results": []},
            }
            plan_bytes = pretty(plan)
            result_bytes = pretty(result)

            def digest(data: bytes) -> dict:
                return {
                    "size_bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }

            manifest = {
                "run_id": run_id,
                "cutoff_date": CUTOFF,
                "domain": "Edge",
                "analysis_status": "complete",
                "pipeline_version": "discovery-pipeline-v9",
                "gate_id": "yandex-yandexgpt-lite-5-candidate-gate-v4",
                "counts": {},
                "outputs": [
                    {"filename": "plan.json", **digest(plan_bytes)},
                    {"filename": "pipeline_result.json", **digest(result_bytes)},
                ],
            }
            run_dir = root / run_id
            run_dir.mkdir(parents=True)
            (run_dir / "plan.json").write_bytes(plan_bytes)
            (run_dir / "pipeline_result.json").write_bytes(result_bytes)
            (run_dir / "manifest.json").write_bytes(pretty(manifest))
            output = root / "export-strata"
            export_labeling_bundle(
                run_dirs=(run_dir,),
                template_path=Path("templates/labeling_workbook.xlsx"),
                output_dir=output,
            )
            self.assertTrue(output.is_dir())
            manifest_out = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest_out["filled_candidate_strata"],
                {"cross_source": 1, "single_origin": 1},
            )
            # The template has 100 candidate slots in total; only 2 are filled here.
            # Deficits for the same area/need must arrive aggregated.
            for deficit in manifest_out["deficits"]:
                if deficit["area"] == "Edge" and deficit["need"] == "candidate_review":
                    self.assertGreater(deficit["missing"], 1)
            edge_deficits = [
                item for item in manifest_out["deficits"]
                if item["area"] == "Edge" and item["need"] == "candidate_review"
            ]
            self.assertEqual(len(edge_deficits), 1)


if __name__ == "__main__":
    unittest.main()
