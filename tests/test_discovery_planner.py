from __future__ import annotations

import json
import unittest
from datetime import date

from nextwave.discovery import (
    DEFAULT_DISCOVERY_BUDGETS,
    AnalysisScope,
    DiscoveryBudget,
    DiscoveryPlan,
    ScopeGranularity,
    build_analysis_scope,
    build_discovery_plan,
)
from nextwave.sources import ConnectorId


def make_scope() -> AnalysisScope:
    return build_analysis_scope(
        raw_query="  Технологии   в ИИ  ",
        normalized_query="Artificial Intelligence Technologies",
        search_texts=("Технологии в ИИ", "AI technologies"),
        languages=("RU", "en"),
        subfield_ids=("1702",),
    )


class AnalysisScopeTests(unittest.TestCase):
    def test_normalizes_and_preserves_reviewed_query_variants(self) -> None:
        scope = make_scope()

        self.assertEqual(scope.raw_query, "Технологии в ИИ")
        self.assertEqual(scope.normalized_query, "artificial intelligence technologies")
        self.assertEqual(
            scope.search_texts,
            (
                "artificial intelligence technologies",
                "Технологии в ИИ",
                "AI technologies",
            ),
        )
        self.assertEqual(scope.languages, ("ru", "en"))
        self.assertEqual(scope.subfield_ids, ("1702",))

    def test_equivalent_spacing_and_case_produce_same_scope_id(self) -> None:
        first = make_scope()
        second = build_analysis_scope(
            raw_query="Технологии в ИИ",
            normalized_query="ARTIFICIAL INTELLIGENCE TECHNOLOGIES",
            search_texts=("технологии в ии", "ai technologies"),
            languages=("ru", "EN"),
            subfield_ids=("1702",),
        )

        self.assertEqual(first.scope_id, second.scope_id)

    def test_broad_and_narrow_meanings_have_distinct_scope_ids(self) -> None:
        broad = make_scope()
        narrow = build_analysis_scope(
            raw_query="Спекулятивное декодирование",
            normalized_query="speculative decoding",
            search_texts=("спекулятивное декодирование",),
            languages=("ru", "en"),
            granularity=ScopeGranularity.TECHNOLOGY,
        )

        self.assertNotEqual(broad.scope_id, narrow.scope_id)


class DiscoveryPlanTests(unittest.TestCase):
    def test_builds_source_query_from_frozen_scope_and_cutoff(self) -> None:
        scope = make_scope()
        plan = build_discovery_plan(
            analysis_id="analysis-ai-001",
            scope=scope,
            published_from=date(2024, 1, 1),
            cutoff_date=date(2026, 9, 20),
        )

        self.assertEqual(plan.query.analysis_scope_id, scope.scope_id)
        self.assertEqual(plan.query.search_texts, scope.search_texts)
        self.assertEqual(plan.query.published_from, date(2024, 1, 1))
        self.assertEqual(plan.query.published_until, date(2026, 9, 20))
        self.assertEqual(plan.query.cutoff_date, date(2026, 9, 20))
        self.assertTrue(plan.budget_for(ConnectorId.OPENALEX).required)
        self.assertFalse(plan.budget_for(ConnectorId.MEDIACLOUD).required)
        self.assertEqual(
            plan.budget_for(ConnectorId.MEDIACLOUD).max_elapsed_seconds,
            120.0,
        )

    def test_same_inputs_produce_same_scope_query_and_plan_ids(self) -> None:
        first = build_discovery_plan(
            analysis_id="analysis-ai-001",
            scope=make_scope(),
            published_from=date(2024, 1, 1),
            cutoff_date=date(2026, 9, 20),
        )
        second = build_discovery_plan(
            analysis_id="analysis-ai-001",
            scope=make_scope(),
            published_from=date(2024, 1, 1),
            cutoff_date=date(2026, 9, 20),
        )

        self.assertEqual(first.scope.scope_id, second.scope.scope_id)
        self.assertEqual(first.query.query_id, second.query.query_id)
        self.assertEqual(first.plan_id, second.plan_id)
        json.dumps(first.to_dict(), ensure_ascii=False)

    def test_cutoff_changes_query_and_plan_but_not_scope(self) -> None:
        scope = make_scope()
        first = build_discovery_plan(
            analysis_id="analysis-ai-001",
            scope=scope,
            published_from=date(2024, 1, 1),
            cutoff_date=date(2026, 9, 19),
        )
        second = build_discovery_plan(
            analysis_id="analysis-ai-001",
            scope=scope,
            published_from=date(2024, 1, 1),
            cutoff_date=date(2026, 9, 20),
        )

        self.assertEqual(first.scope.scope_id, second.scope.scope_id)
        self.assertNotEqual(first.query.query_id, second.query.query_id)
        self.assertNotEqual(first.plan_id, second.plan_id)

    def test_rejects_duplicate_connector_budgets(self) -> None:
        openalex = DEFAULT_DISCOVERY_BUDGETS[0]
        with self.assertRaisesRegex(ValueError, "budgets must be unique"):
            DiscoveryPlan(
                plan_id="plan-duplicate-budget",
                analysis_id="analysis-ai-001",
                scope=make_scope(),
                query=build_discovery_plan(
                    analysis_id="analysis-ai-001",
                    scope=make_scope(),
                    published_from=date(2024, 1, 1),
                    cutoff_date=date(2026, 9, 20),
                ).query,
                budgets=(openalex, openalex),
            )

    def test_rejects_plan_without_required_openalex(self) -> None:
        optional_openalex = DiscoveryBudget(
            connector_id=ConnectorId.OPENALEX,
            required=False,
            max_requests=1,
            max_pages=1,
            max_documents=10,
            request_timeout_seconds=10,
            max_elapsed_seconds=20,
        )
        with self.assertRaisesRegex(ValueError, "OpenAlex must be present and required"):
            build_discovery_plan(
                analysis_id="analysis-ai-001",
                scope=make_scope(),
                published_from=date(2024, 1, 1),
                cutoff_date=date(2026, 9, 20),
                budgets=(optional_openalex,),
            )


if __name__ == "__main__":
    unittest.main()
