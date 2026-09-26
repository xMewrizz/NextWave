"""Tests for the offline labeling-enrichment plan (hand-built bundles)."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from datetime import date
from io import StringIO
from pathlib import Path

from nextwave.__main__ import main
from nextwave.labeling import enrichment_plan as enrichment_plan_module
from nextwave.labeling.enrichment_plan import (
    LABELING_ENRICHMENT_PLAN_VERSION,
    MEDIACLOUD_COLLECTION_IDS,
    MEDIACLOUD_RETRIEVAL_POLICY,
    OPENALEX_RETRIEVAL_POLICY,
    export_enrichment_plan,
)
from nextwave.sources import (
    QueryPurpose,
    RetrievalChannel,
    SourceQuery,
    build_mediacloud_story_request,
    build_openalex_request,
)

CUTOFF = "2026-09-15"


def candidate_row(number: int, **overrides) -> dict:
    payload = {
        "schema_version": "negative-candidate-v1",
        "candidate_id": f"team-negative-{number:03d}",
        "canonical_name": f"Tech {number}",
        "aliases": [f"Alias {number}"],
        "group_id": f"group-{number:03d}",
        "source_query": "Технологии в ИИ",
        "domain": "Edge",
        "analysis_scope_key": "edge-v1",
        "cutoff_date": CUTOFF,
    }
    payload.update(overrides)
    return payload


def write_bundle(
    root: Path,
    name: str,
    rows: list[dict],
    *,
    cutoff: str = CUTOFF,
    filled: dict | None = None,
    manifest_override=None,
    tamper_byte: bool = False,
    tamper_size: bool = False,
) -> Path:
    body = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    if manifest_override is None:
        manifest_override = {
            "schema_version": "labeling-export-manifest-v1",
            "cutoff_date": cutoff,
            "filled_candidates": filled if filled is not None else {"Edge": len(rows)},
            "outputs": {
                "negative_candidates.jsonl": {
                    "size_bytes": len(body) + (1 if tamper_size else 0),
                    "sha256": hashlib.sha256(body).hexdigest(),
                }
            },
        }
    if tamper_byte:
        body = body.replace(b"Tech 1", b"Texh 1")
    bundle = root / name
    bundle.mkdir(parents=True)
    manifest_text = (
        manifest_override
        if isinstance(manifest_override, str)
        else json.dumps(manifest_override, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    )
    (bundle / "manifest.json").write_text(manifest_text, encoding="utf-8")
    (bundle / "negative_candidates.jsonl").write_bytes(body)
    return bundle


def load_plan(output: Path) -> dict:
    return json.loads((output / "plan.json").read_text(encoding="utf-8"))


def walk_keys(payload):
    if isinstance(payload, dict):
        for key, value in payload.items():
            yield key
            yield from walk_keys(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from walk_keys(value)


class EnrichmentPlanTests(unittest.TestCase):
    def test_valid_bundle_plans_all_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_bundle(root, "bundle", [candidate_row(1), candidate_row(2)])
            output = root / "plan"

            paths = export_enrichment_plan(bundle_dir=bundle, output_dir=output)

            plan = load_plan(output)
            self.assertEqual(len(plan["candidates"]), 2)
            self.assertEqual(plan["totals"]["candidates"], 2)
            self.assertTrue(paths.plan.is_file())
            self.assertTrue(paths.manifest.is_file())

    def test_permuted_rows_give_identical_plan_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [candidate_row(1), candidate_row(2), candidate_row(3)]
            first = write_bundle(root, "bundle-a", rows)
            second = write_bundle(root, "bundle-b", list(reversed(rows)))
            export_enrichment_plan(bundle_dir=first, output_dir=root / "plan-a")
            export_enrichment_plan(bundle_dir=second, output_dir=root / "plan-b")

            self.assertEqual(
                (root / "plan-a" / "plan.json").read_bytes(),
                (root / "plan-b" / "plan.json").read_bytes(),
            )

    def test_every_candidate_has_scientific_and_industry_searches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_bundle(root, "bundle", [candidate_row(1)])
            export_enrichment_plan(bundle_dir=bundle, output_dir=root / "plan")
            entry = load_plan(root / "plan")["candidates"][0]
            classes = [(s["source_class"], s["connector"]) for s in entry["searches"]]

            self.assertIn(("scientific", "openalex"), classes)
            self.assertIn(("industry", "mediacloud"), classes)

    def test_request_counts_follow_connector_formula(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_bundle(
                root,
                "bundle",
                [
                    candidate_row(
                        1, canonical_name="Tech One", aliases=["Second", "Third"]
                    )
                ],
            )
            export_enrichment_plan(bundle_dir=bundle, output_dir=root / "plan")
            entry = load_plan(root / "plan")["candidates"][0]
            scientific = next(
                s for s in entry["searches"] if s["connector"] == "openalex"
            )
            industry = next(
                s for s in entry["searches"] if s["connector"] == "mediacloud"
            )

            # N terms -> N x 2 openalex requests, N mediacloud requests.
            self.assertEqual(
                [(r["search_text"], r["languages"]) for r in scientific["requests"]],
                [
                    ("Tech One", ["en"]),
                    ("Tech One", ["ru"]),
                    ("Second", ["en"]),
                    ("Second", ["ru"]),
                    ("Third", ["en"]),
                    ("Third", ["ru"]),
                ],
            )
            self.assertEqual(
                [(r["search_text"], r["languages"]) for r in industry["requests"]],
                [
                    ("Tech One", ["en", "ru"]),
                    ("Second", ["en", "ru"]),
                    ("Third", ["en", "ru"]),
                ],
            )
            for search in entry["searches"]:
                self.assertEqual(
                    len({r["request_id"] for r in search["requests"]}),
                    len(search["requests"]),
                )
                self.assertTrue(
                    all(
                        r["request_id"].startswith("request-")
                        for r in search["requests"]
                    )
                )
                self.assertFalse(any("language" in r for r in search["requests"]))
            export_enrichment_plan(bundle_dir=bundle, output_dir=root / "plan-2")
            again = load_plan(root / "plan-2")["candidates"][0]
            self.assertEqual(entry["searches"], again["searches"])

    def test_languages_change_request_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_bundle(root, "bundle", [candidate_row(1)])
            export_enrichment_plan(bundle_dir=bundle, output_dir=root / "plan")
            entry = load_plan(root / "plan")["candidates"][0]

            scientific = next(
                s for s in entry["searches"] if s["connector"] == "openalex"
            )
            by_text: dict[str, set[str]] = {}
            for request in scientific["requests"]:
                by_text.setdefault(request["search_text"], set()).add(
                    request["request_id"]
                )
            self.assertTrue(
                all(len(ids) == 2 for ids in by_text.values()),
                "en and ru requests must have distinct ids",
            )
            industry = next(
                s for s in entry["searches"] if s["connector"] == "mediacloud"
            )
            self.assertTrue(
                all(
                    request["languages"] == ["en", "ru"]
                    for request in industry["requests"]
                )
            )

    def test_totals_match_request_elements(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_bundle(
                root,
                "bundle",
                [
                    candidate_row(1, aliases=["Second"]),
                    candidate_row(2, aliases=[]),
                ],
            )
            export_enrichment_plan(bundle_dir=bundle, output_dir=root / "plan")
            plan = load_plan(root / "plan")

            def count(connector: str) -> int:
                return sum(
                    len(search["requests"])
                    for entry in plan["candidates"]
                    for search in entry["searches"]
                    if search["connector"] == connector
                )

            self.assertEqual(plan["totals"]["openalex_primary_requests"], count("openalex"))
            self.assertEqual(plan["totals"]["mediacloud_primary_requests"], count("mediacloud"))
            # OpenAlex: (canonical + aliases) x (en + ru); Media Cloud: one per term.
            self.assertEqual(plan["totals"]["openalex_primary_requests"], (2 + 1) * 2)
            self.assertEqual(plan["totals"]["mediacloud_primary_requests"], 2 + 1)

    def test_retrieval_policies_and_completion_rule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_bundle(root, "bundle", [candidate_row(1)])
            export_enrichment_plan(bundle_dir=bundle, output_dir=root / "plan")
            entry = load_plan(root / "plan")["candidates"][0]
            scientific = next(
                s for s in entry["searches"] if s["connector"] == "openalex"
            )
            industry = next(
                s for s in entry["searches"] if s["connector"] == "mediacloud"
            )

            self.assertEqual(entry["searches"][0]["languages"], ["en", "ru"])
            self.assertEqual(
                scientific["retrieval_policy"],
                {
                    "channel": "text",
                    "per_page": 20,
                    "max_pages_per_request": 1,
                    "max_attempts": 3,
                    "timeout_seconds": 20,
                },
            )
            self.assertEqual(industry["collection_ids"], [34412234, 34412118])
            self.assertEqual(
                industry["retrieval_policy"],
                {
                    "page_size": 40,
                    "max_pages_per_request": 1,
                    "max_attempts": 3,
                    "timeout_seconds": 60,
                    "min_interval_seconds": 30,
                },
            )
            for search in entry["searches"]:
                self.assertEqual(
                    search["completion_rule"], "all_planned_requests_successful"
                )

    def test_windows_equal_t_minus_730_365(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_bundle(root, "bundle", [candidate_row(1)])
            export_enrichment_plan(bundle_dir=bundle, output_dir=root / "plan")
            plan = load_plan(root / "plan")

            self.assertEqual(plan["cutoff_date"], "2026-09-15")
            self.assertEqual(plan["history_from"], "2024-09-15")
            self.assertEqual(plan["recent_window_from"], "2025-09-15")
            self.assertEqual(
                plan["windows"]["previous"],
                {
                    "from": "2024-09-15",
                    "until": "2025-09-15",
                    "start_inclusive": True,
                    "end_inclusive": False,
                },
            )
            self.assertEqual(
                plan["windows"]["recent"],
                {
                    "from": "2025-09-15",
                    "until": "2026-09-15",
                    "start_inclusive": True,
                    "end_inclusive": True,
                },
            )
            entry = plan["candidates"][0]
            window = {"from": "2024-09-15", "until": "2026-09-15"}
            for search in entry["searches"]:
                self.assertEqual(search["required_window"], window)
                self.assertEqual(search["planned_window"], window)
            industry = next(
                s for s in entry["searches"] if s["connector"] == "mediacloud"
            )
            self.assertEqual(
                industry["publicity_window"],
                {"from": "2025-09-15", "until": "2026-09-15"},
            )

    def test_terms_canonical_first_cleaned_and_deduped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_bundle(
                root,
                "bundle",
                [
                    candidate_row(
                        1,
                        canonical_name="  Ｔｅｃｈ One  ",
                        aliases=["TECH ONE", "Second"],
                    )
                ],
            )
            export_enrichment_plan(bundle_dir=bundle, output_dir=root / "plan")
            entry = load_plan(root / "plan")["candidates"][0]

            self.assertEqual(entry["search_terms"][0], "Tech One")
            self.assertEqual(entry["search_terms"], ["Tech One", "Second"])

    def test_same_name_kept_by_candidate_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = candidate_row(1, canonical_name="Same Tech")
            second = candidate_row(2, canonical_name="Same Tech")
            bundle = write_bundle(root, "bundle", [first, second])
            export_enrichment_plan(bundle_dir=bundle, output_dir=root / "plan")
            entries = load_plan(root / "plan")["candidates"]

            self.assertEqual(len(entries), 2)
            self.assertEqual(
                [entry["candidate_id"] for entry in entries],
                ["team-negative-001", "team-negative-002"],
            )

    def test_tampered_byte_is_rejected_by_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_bundle(root, "bundle", [candidate_row(1)], tamper_byte=True)
            with self.assertRaisesRegex(ValueError, "checksum"):
                export_enrichment_plan(bundle_dir=bundle, output_dir=root / "plan")
            self.assertFalse((root / "plan").exists())

    def test_wrong_size_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_bundle(root, "bundle", [candidate_row(1)], tamper_size=True)
            with self.assertRaisesRegex(ValueError, "size"):
                export_enrichment_plan(bundle_dir=bundle, output_dir=root / "plan")
            self.assertFalse((root / "plan").exists())

    def test_repeated_candidate_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_bundle(root, "bundle", [candidate_row(1), candidate_row(1)])
            with self.assertRaisesRegex(ValueError, "candidate_id"):
                export_enrichment_plan(bundle_dir=bundle, output_dir=root / "plan")
            self.assertFalse((root / "plan").exists())

    def test_repeated_group_id_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = candidate_row(1)
            second = candidate_row(2, group_id=first["group_id"])
            bundle = write_bundle(root, "bundle", [first, second])
            with self.assertRaisesRegex(ValueError, "group_id"):
                export_enrichment_plan(bundle_dir=bundle, output_dir=root / "plan")
            self.assertFalse((root / "plan").exists())

    def test_row_count_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_bundle(root, "bundle", [candidate_row(1)], filled={"Edge": 2})
            with self.assertRaisesRegex(ValueError, "filled_candidates"):
                export_enrichment_plan(bundle_dir=bundle, output_dir=root / "plan")
            self.assertFalse((root / "plan").exists())

    def test_wrong_cutoff_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_bundle(root, "bundle", [candidate_row(1)], cutoff="2026-09-16")
            with self.assertRaisesRegex(ValueError, "cutoff"):
                export_enrichment_plan(bundle_dir=bundle, output_dir=root / "plan")
            self.assertFalse((root / "plan").exists())

    def test_existing_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_bundle(root, "bundle", [candidate_row(1)])
            output = root / "plan"
            output.mkdir()
            with self.assertRaises(ValueError):
                export_enrichment_plan(bundle_dir=bundle, output_dir=output)
            self.assertEqual(list(output.iterdir()), [])

    def test_cli_error_returns_1_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_bundle(root, "bundle", [candidate_row(1)], tamper_byte=True)
            output = root / "plan"
            with redirect_stderr(StringIO()):
                exit_code = main(
                    [
                        "labeling-enrichment-plan",
                        "--bundle",
                        str(bundle),
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(exit_code, 1)
            self.assertFalse(output.exists())

    def test_manifest_as_list_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_bundle(root, "bundle", [candidate_row(1)], manifest_override=[])
            with self.assertRaises(ValueError):
                export_enrichment_plan(bundle_dir=bundle, output_dir=root / "plan")
            self.assertFalse((root / "plan").exists())

    def test_outputs_wrong_type_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_bundle(
                root,
                "bundle",
                [candidate_row(1)],
                manifest_override={
                    "schema_version": "labeling-export-manifest-v1",
                    "cutoff_date": CUTOFF,
                    "filled_candidates": {"Edge": 1},
                    "outputs": [],
                },
            )
            with self.assertRaisesRegex(ValueError, "outputs"):
                export_enrichment_plan(bundle_dir=bundle, output_dir=root / "plan")
            self.assertFalse((root / "plan").exists())

    def test_filled_candidates_wrong_type_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_bundle(root, "bundle", [candidate_row(1)])
            manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
            manifest["filled_candidates"] = ["Edge"]
            (bundle / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "filled_candidates"):
                export_enrichment_plan(bundle_dir=bundle, output_dir=root / "plan")
            self.assertFalse((root / "plan").exists())

    def test_string_count_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_bundle(
                root, "bundle", [candidate_row(1)], filled={"Edge": "1"}
            )
            with self.assertRaisesRegex(ValueError, "filled_candidates"):
                export_enrichment_plan(bundle_dir=bundle, output_dir=root / "plan")
            self.assertFalse((root / "plan").exists())

    def test_true_count_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_bundle(
                root, "bundle", [candidate_row(1)], filled={"Edge": True}
            )
            with self.assertRaisesRegex(ValueError, "filled_candidates"):
                export_enrichment_plan(bundle_dir=bundle, output_dir=root / "plan")
            self.assertFalse((root / "plan").exists())

    def test_wrong_row_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_bundle(
                root, "bundle", [candidate_row(1, schema_version="negative-candidate-v0")]
            )
            with self.assertRaisesRegex(ValueError, "schema"):
                export_enrichment_plan(bundle_dir=bundle, output_dir=root / "plan")
            self.assertFalse((root / "plan").exists())

    def test_string_aliases_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_bundle(
                root, "bundle", [candidate_row(1, aliases="Alias 1")]
            )
            with self.assertRaisesRegex(ValueError, "aliases"):
                export_enrichment_plan(bundle_dir=bundle, output_dir=root / "plan")
            self.assertFalse((root / "plan").exists())

    def test_non_string_alias_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_bundle(
                root, "bundle", [candidate_row(1, aliases=["Alias 1", 5])]
            )
            with self.assertRaisesRegex(ValueError, "aliases"):
                export_enrichment_plan(bundle_dir=bundle, output_dir=root / "plan")
            self.assertFalse((root / "plan").exists())

    def test_empty_file_with_zero_count_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_bundle(root, "bundle", [], filled={"Edge": 0})
            with self.assertRaisesRegex(ValueError, "no records"):
                export_enrichment_plan(bundle_dir=bundle, output_dir=root / "plan")
            self.assertFalse((root / "plan").exists())

    def test_domain_mix_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            second = candidate_row(2, domain="Финтех", analysis_scope_key="fintech-v1")
            bundle = write_bundle(
                root,
                "bundle",
                [candidate_row(1), second],
                filled={"Edge": 1, "Финтех": 1},
            )
            other = candidate_row(1, domain="Финтех", analysis_scope_key="fintech-v1")
            swapped = write_bundle(
                root,
                "swapped",
                [other, candidate_row(2)],
                filled={"Edge": 2},
            )
            with self.assertRaisesRegex(ValueError, "domains"):
                export_enrichment_plan(bundle_dir=swapped, output_dir=root / "plan")
            self.assertFalse((root / "plan").exists())
            export_enrichment_plan(bundle_dir=bundle, output_dir=root / "plan-ok")

    def test_no_gdelt_anywhere_in_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_bundle(root, "bundle", [candidate_row(1)])
            export_enrichment_plan(bundle_dir=bundle, output_dir=root / "plan")
            raw = (root / "plan" / "plan.json").read_text(encoding="utf-8")

            self.assertNotIn("gdelt", raw)

    def test_plan_claims_no_executed_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_bundle(root, "bundle", [candidate_row(1)])
            export_enrichment_plan(bundle_dir=bundle, output_dir=root / "plan")
            plan = load_plan(root / "plan")
            forbidden = {
                key
                for key in walk_keys(plan)
                if key
                in {
                    "status",
                    "searched",
                    "search_performed",
                    "results",
                    "returned_records",
                    "chunks",
                    "query_texts",
                    "fallback_connector",
                    "coverage_limit",
                    "qualifies_for_complete_industry_coverage",
                }
            }

            self.assertEqual(forbidden, set())
            self.assertEqual(plan["schema_version"], LABELING_ENRICHMENT_PLAN_VERSION)


class RequestIdentityTests(unittest.TestCase):
    WINDOW = ("2024-09-15", "2026-09-15")

    def test_openalex_per_page_changes_request_id(self) -> None:
        base = dict(OPENALEX_RETRIEVAL_POLICY)
        changed = dict(base, per_page=21)

        self.assertNotEqual(
            enrichment_plan_module._request_id(
                "team-negative-001", "openalex", "primary", "Tech", ["en"],
                self.WINDOW, base, (),
            ),
            enrichment_plan_module._request_id(
                "team-negative-001", "openalex", "primary", "Tech", ["en"],
                self.WINDOW, changed, (),
            ),
        )

    def test_mediacloud_page_size_changes_request_id(self) -> None:
        base = dict(MEDIACLOUD_RETRIEVAL_POLICY)
        changed = dict(base, page_size=21)

        self.assertNotEqual(
            enrichment_plan_module._request_id(
                "team-negative-001", "mediacloud", "primary", "Tech", ["en", "ru"],
                self.WINDOW, base, MEDIACLOUD_COLLECTION_IDS,
            ),
            enrichment_plan_module._request_id(
                "team-negative-001", "mediacloud", "primary", "Tech", ["en", "ru"],
                self.WINDOW, changed, MEDIACLOUD_COLLECTION_IDS,
            ),
        )

    def test_mediacloud_collections_change_request_id(self) -> None:
        self.assertNotEqual(
            enrichment_plan_module._request_id(
                "team-negative-001", "mediacloud", "primary", "Tech", ["en", "ru"],
                self.WINDOW, MEDIACLOUD_RETRIEVAL_POLICY, MEDIACLOUD_COLLECTION_IDS,
            ),
            enrichment_plan_module._request_id(
                "team-negative-001", "mediacloud", "primary", "Tech", ["en", "ru"],
                self.WINDOW, MEDIACLOUD_RETRIEVAL_POLICY, (34412234,),
            ),
        )

    def test_languages_change_request_id(self) -> None:
        self.assertNotEqual(
            enrichment_plan_module._request_id(
                "team-negative-001", "openalex", "primary", "Tech", ["en"],
                self.WINDOW, OPENALEX_RETRIEVAL_POLICY, (),
            ),
            enrichment_plan_module._request_id(
                "team-negative-001", "openalex", "primary", "Tech", ["ru"],
                self.WINDOW, OPENALEX_RETRIEVAL_POLICY, (),
            ),
        )
        self.assertNotEqual(
            enrichment_plan_module._request_id(
                "team-negative-001", "mediacloud", "primary", "Tech", ["en", "ru"],
                self.WINDOW, MEDIACLOUD_RETRIEVAL_POLICY, MEDIACLOUD_COLLECTION_IDS,
            ),
            enrichment_plan_module._request_id(
                "team-negative-001", "mediacloud", "primary", "Tech", ["ru", "en"],
                self.WINDOW, MEDIACLOUD_RETRIEVAL_POLICY, MEDIACLOUD_COLLECTION_IDS,
            ),
        )

    def test_completion_rule_changes_search_id(self) -> None:
        base_ids = ["request-aaa", "request-bbb"]
        coverage = {
            "successful_response_including_zero": "covered",
            "failed_timeout_429_or_skipped": "unknown",
        }

        self.assertNotEqual(
            enrichment_plan_module._search_id(
                "team-negative-001", "openalex", "primary", base_ids,
                self.WINDOW, self.WINDOW, None,
                OPENALEX_RETRIEVAL_POLICY, (), "all_planned_requests_successful",
                coverage,
            ),
            enrichment_plan_module._search_id(
                "team-negative-001", "openalex", "primary", base_ids,
                self.WINDOW, self.WINDOW, None,
                OPENALEX_RETRIEVAL_POLICY, (), "any_request_successful",
                coverage,
            ),
        )

    def test_ids_are_deterministic(self) -> None:
        self.assertEqual(
            enrichment_plan_module._request_id(
                "team-negative-001", "mediacloud", "primary", "Tech", ["en", "ru"],
                self.WINDOW, MEDIACLOUD_RETRIEVAL_POLICY, MEDIACLOUD_COLLECTION_IDS,
            ),
            enrichment_plan_module._request_id(
                "team-negative-001", "mediacloud", "primary", "Tech", ["en", "ru"],
                self.WINDOW, MEDIACLOUD_RETRIEVAL_POLICY, MEDIACLOUD_COLLECTION_IDS,
            ),
        )
        self.assertEqual(
            enrichment_plan_module._search_id(
                "team-negative-001", "openalex", "primary", ["request-aaa"],
                self.WINDOW, self.WINDOW, None,
                OPENALEX_RETRIEVAL_POLICY, (), "all_planned_requests_successful",
                {"successful_response_including_zero": "covered"},
            ),
            enrichment_plan_module._search_id(
                "team-negative-001", "openalex", "primary", ["request-aaa"],
                self.WINDOW, self.WINDOW, None,
                OPENALEX_RETRIEVAL_POLICY, (), "all_planned_requests_successful",
                {"successful_response_including_zero": "covered"},
            ),
        )


def probe_query(search_text: str, languages: list[str], index: int):
    return SourceQuery(
        query_id=f"probe-{index:04d}",
        analysis_scope_id="scope-probe",
        purpose=QueryPurpose.DISCOVERY,
        raw_query="enrichment probe",
        normalized_query="enrichment probe",
        search_texts=(search_text,),
        published_from=date(2024, 9, 15),
        published_until=date(2026, 9, 15),
        cutoff_date=date(2026, 9, 15),
        languages=tuple(languages),
    )


class ExecutabilityTests(unittest.TestCase):
    def test_every_request_builds_connector_requests_without_http(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_bundle(
                root,
                "bundle",
                [
                    candidate_row(1, canonical_name="Speculative decoding"),
                    candidate_row(2, canonical_name="Фотонный чип", aliases=["Оптика"]),
                ],
            )
            export_enrichment_plan(bundle_dir=bundle, output_dir=root / "plan")
            plan = load_plan(root / "plan")
            built = 0
            for entry in plan["candidates"]:
                for search in entry["searches"]:
                    for request in search["requests"]:
                        query = probe_query(
                            request["search_text"], request["languages"], built
                        )
                        if search["connector"] == "openalex":
                            built_request = build_openalex_request(
                                query,
                                channel=RetrievalChannel.TEXT,
                                search_text=request["search_text"],
                                per_page=search["retrieval_policy"]["per_page"],
                            )
                            params = {
                                p.name: p.value for p in built_request.parameters
                            }
                            self.assertEqual(params["per_page"], "20")
                        else:
                            built_request = build_mediacloud_story_request(
                                query,
                                search_text=request["search_text"],
                                collection_ids=tuple(search["collection_ids"]),
                                page_size=search["retrieval_policy"]["page_size"],
                                languages=tuple(request["languages"]),
                            )
                            params = {
                                p.name: p.value for p in built_request.parameters
                            }
                            self.assertEqual(params["page_size"], "40")
                            self.assertEqual(params["cs"], "34412234,34412118")
                        built += 1

            # OpenAlex: 2 candidates x (1 canonical + 1 alias) x 2 languages;
            # Media Cloud: 2 candidates x (1 canonical + 1 alias) x 1 bilingual.
            self.assertEqual(built, 2 * 2 * 2 + 2 * 2)


if __name__ == "__main__":
    unittest.main()
