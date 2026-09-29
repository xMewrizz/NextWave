from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from nextwave.__main__ import main
from nextwave.datasets import OrganizerDatasetPaths, OrganizerWorkbookError


class CommandLineTests(unittest.TestCase):
    @patch("nextwave.__main__.build_organizer_dataset")
    def test_dataset_build_uses_default_versioned_output(self, build) -> None:
        output = Path("data/processed/organizer-positive-2026-09-15-v1")
        build.return_value = OrganizerDatasetPaths(
            candidates=output / "positive_candidates.jsonl",
            annotations=output / "positive_annotations.jsonl",
            manifest=output / "manifest.json",
        )
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = main(["dataset-build", "--input", "data/raw/organizer_signals.xlsx"])

        self.assertEqual(exit_code, 0)
        build.assert_called_once_with(Path("data/raw/organizer_signals.xlsx"), output)
        self.assertIn("успешно собран", stdout.getvalue())
        self.assertIn("manifest.json", stdout.getvalue())

    @patch("nextwave.__main__.build_organizer_dataset")
    def test_dataset_build_reports_validation_error_without_traceback(self, build) -> None:
        build.side_effect = OrganizerWorkbookError("unexpected header in C2")
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            exit_code = main(
                [
                    "dataset-build",
                    "--input",
                    "broken.xlsx",
                    "--output",
                    "data/processed/check",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("unexpected header in C2", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    @patch("nextwave.__main__._runtime_environment")
    @patch("nextwave.__main__.build_query_resolver_from_environment")
    def test_query_resolve_prints_resolution_as_json(
        self, build_resolver, runtime_environment
    ) -> None:
        runtime_environment.return_value = {
            "NEXTWAVE_LLM_PROVIDER": "yandex",
            "NEXTWAVE_LLM_MODEL": "YandexGPT Lite 5",
            "NEXTWAVE_LLM_API_KEY": "temporary-secret",
        }
        build_resolver.return_value.resolve.return_value.to_dict.return_value = {
            "scope": {"normalized_query": "artificial intelligence"},
            "taxonomy_status": "matched",
        }
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = main(["query-resolve", "--query", "Технологии в ИИ"])

        self.assertEqual(exit_code, 0)
        runtime_environment.assert_called_once_with(Path("config/hackathon.env"))
        build_resolver.assert_called_once_with(runtime_environment.return_value)
        build_resolver.return_value.resolve.assert_called_once_with("Технологии в ИИ")
        self.assertIn('"taxonomy_status": "matched"', stdout.getvalue())

    @patch("nextwave.__main__.build_query_resolver_from_environment")
    def test_query_resolve_reports_configuration_error_without_traceback(
        self, build_resolver
    ) -> None:
        build_resolver.side_effect = ValueError("NEXTWAVE_LLM_API_KEY must not be blank")
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            exit_code = main(
                [
                    "query-resolve",
                    "--query",
                    "ИИ",
                    "--env-file",
                    "config/custom.env",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("NEXTWAVE_LLM_API_KEY", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_runtime_environment_reads_file_and_process_values_win(self) -> None:
        from nextwave.__main__ import _runtime_environment

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.env"
            path.write_text(
                "# runtime settings\n"
                "NEXTWAVE_LLM_PROVIDER=yandex\n"
                "NEXTWAVE_LLM_MODEL=YandexGPT Lite 5\n"
                "NEXTWAVE_LLM_API_KEY=file-secret\n",
                encoding="utf-8",
            )

            result = _runtime_environment(
                path,
                {"NEXTWAVE_LLM_API_KEY": "process-secret"},
                dotenv_path=Path(directory) / "no-such-env",
            )

        self.assertEqual(result["NEXTWAVE_LLM_PROVIDER"], "yandex")
        self.assertEqual(result["NEXTWAVE_LLM_MODEL"], "YandexGPT Lite 5")
        self.assertEqual(result["NEXTWAVE_LLM_API_KEY"], "process-secret")

    def test_runtime_environment_rejects_duplicate_keys(self) -> None:
        from nextwave.__main__ import _runtime_environment

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.env"
            path.write_text("KEY=first\nKEY=second\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate variable KEY"):
                _runtime_environment(path, {}, dotenv_path=Path(directory) / "no-such-env")

    def test_dotenv_overrides_file_and_process_wins(self) -> None:
        from nextwave.__main__ import _runtime_environment

        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / "base.env"
            env_file.write_text(
                "NEXTWAVE_LLM_PROVIDER=huggingface\nSHARED=from-file\n",
                encoding="utf-8",
            )
            dotenv_path = Path(directory) / ".env"
            dotenv_path.write_text(
                "NEXTWAVE_LLM_PROVIDER=yandex\nSHARED=from-dotenv\n",
                encoding="utf-8",
            )

            result = _runtime_environment(
                env_file, {"SHARED": "from-process"}, dotenv_path=dotenv_path
            )

            self.assertEqual(result["NEXTWAVE_LLM_PROVIDER"], "yandex")
            self.assertEqual(result["SHARED"], "from-process")

    @patch("nextwave.__main__.build_discovery_pipeline_from_environment")
    @patch("nextwave.__main__.build_query_resolver_from_environment")
    def test_discovery_run_saves_run_directory(self, build_resolver, build_pipeline) -> None:
        from types import SimpleNamespace

        from nextwave.discovery import build_analysis_scope

        scope = build_analysis_scope(
            raw_query="Технологии в ИИ",
            normalized_query="artificial intelligence",
            search_texts=("Технологии в ИИ",),
            languages=("en", "ru"),
            subfield_ids=("1702",),
        )
        build_resolver.return_value.resolve.return_value = SimpleNamespace(scope=scope)

        class StubResult:
            pipeline_version = "discovery-pipeline-v6"

            def __init__(self, plan_id: str) -> None:
                self.plan_id = plan_id

            def to_dict(self):
                return {
                    "plan_id": self.plan_id,
                    "pipeline_version": self.pipeline_version,
                    "documents": ["d1"],
                    "candidate_proposals": {
                        "proposals": [{"proposal_id": "p1"}],
                        "exclusions": [],
                    },
                    "candidate_gate": {
                        "gate_id": "g1",
                        "decisions": [{"proposal_id": "p1", "decision": "accept"}],
                    },
                    "alias_resolution": {"review_suggestions": []},
                    "evidence_extraction": {"proposals": [], "issues": []},
                    "text_extraction": None,
                    "scientific": {"snapshot_path": "snap/oa", "documents": []},
                    "media": {"snapshot_path": "snap/mc", "documents": []},
                    "verification": {"results": []},
                }

        build_pipeline.return_value.execute.side_effect = lambda plan, **kwargs: StubResult(
            plan.plan_id
        )

        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / "runtime.env"
            env_file.write_text("", encoding="utf-8")
            output_root = Path(directory) / "runs"
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "discovery-run",
                        "--query",
                        "Технологии в ИИ",
                        "--analysis-id",
                        "analysis-ai-001",
                        "--analysis-scope-key",
                        "ai-nlp-v1",
                        "--domain",
                        "Инфраструктура ИИ",
                        "--env-file",
                        str(env_file),
                        "--output-root",
                        str(output_root),
                    ]
                )

            self.assertEqual(exit_code, 0)
            run_dirs = [path for path in output_root.iterdir() if path.is_dir()]
            self.assertEqual(len(run_dirs), 1)
            for filename in ("plan.json", "pipeline_result.json", "manifest.json"):
                self.assertTrue((run_dirs[0] / filename).is_file())
            self.assertIn("Каталог запуска", stdout.getvalue())

        build_resolver.return_value.resolve.assert_called_once_with("Технологии в ИИ")

    @patch("nextwave.__main__.build_query_resolver_from_environment")
    def test_discovery_run_reports_error_without_traceback(self, build_resolver) -> None:
        build_resolver.side_effect = ValueError("NEXTWAVE_LLM_API_KEY must not be blank")
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            exit_code = main(
                [
                    "discovery-run",
                    "--query",
                    "ИИ",
                    "--analysis-id",
                    "analysis-ai-001",
                    "--analysis-scope-key",
                    "ai-nlp-v1",
                    "--domain",
                    "Инфраструктура ИИ",
                    "--env-file",
                    "config/custom.env",
                    "--output-root",
                    "data/development/discovery",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("NEXTWAVE_LLM_API_KEY", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_discovery_run_rejects_bad_date_without_traceback(self) -> None:
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            exit_code = main(
                [
                    "discovery-run",
                    "--query",
                    "ИИ",
                    "--analysis-id",
                    "analysis-ai-001",
                    "--analysis-scope-key",
                    "ai-nlp-v1",
                    "--domain",
                    "Инфраструктура ИИ",
                    "--cutoff-date",
                    "15.09.2026",
                    "--env-file",
                    "config/custom.env",
                    "--output-root",
                    "data/development/discovery",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("--cutoff-date", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    @patch("nextwave.__main__.save_discovery_run")
    @patch("nextwave.__main__.build_discovery_pipeline_from_environment")
    @patch("nextwave.__main__.build_query_resolver_from_environment")
    def test_discovery_run_derives_scope_key_and_domain_from_resolution(
        self, build_resolver, build_pipeline, save_run
    ) -> None:
        from types import SimpleNamespace

        from nextwave.discovery import build_analysis_scope

        scope = build_analysis_scope(
            raw_query="Технологии в ИИ",
            normalized_query="artificial intelligence",
            search_texts=("Технологии в ИИ",),
            languages=("en", "ru"),
            subfield_ids=("1702",),
        )
        build_resolver.return_value.resolve.return_value = SimpleNamespace(scope=scope)
        save_run.return_value = Path("runs") / "analysis-ai-009-abc123"

        with tempfile.TemporaryDirectory() as directory:
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "discovery-run",
                        "--query",
                        "Технологии в ИИ",
                        "--analysis-id",
                        "analysis-ai-009",
                        "--env-file",
                        str(Path(directory) / "runtime.env"),
                        "--output-root",
                        str(Path(directory) / "runs"),
                    ]
                )

            self.assertEqual(exit_code, 0)
            _, kwargs = save_run.call_args
            self.assertEqual(kwargs["analysis_scope_key"], scope.scope_id)
            self.assertEqual(kwargs["domain"], "Технологии в ИИ")

    @patch("nextwave.__main__.export_enrichment_plan")
    def test_labeling_enrichment_plan_uses_versioned_default_output(self, export) -> None:
        from nextwave.labeling.enrichment_plan import LABELING_ENRICHMENT_PLAN_VERSION

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(["labeling-enrichment-plan", "--bundle", "bundle"])

        self.assertEqual(exit_code, 0)
        _, kwargs = export.call_args
        self.assertEqual(
            kwargs["output_dir"],
            Path("data") / "development" / LABELING_ENRICHMENT_PLAN_VERSION,
        )
        self.assertEqual(
            kwargs["output_dir"],
            Path("data/development/labeling-enrichment-plan-v2"),
        )

    @patch("nextwave.__main__.export_analysis_enrichment_plan")
    def test_analysis_enrichment_plan_is_query_specific(self, export) -> None:
        from nextwave.labeling.enrichment_plan import (
            LABELING_ENRICHMENT_PLAN_VERSION,
            LabelingEnrichmentPlanPaths,
        )

        export.return_value = LabelingEnrichmentPlanPaths(
            plan=Path("out/plan.json"), manifest=Path("out/manifest.json")
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(["analysis-enrichment-plan", "--run", "one-run"])

        self.assertEqual(exit_code, 0)
        _, kwargs = export.call_args
        self.assertEqual(kwargs["run_dir"], Path("one-run"))
        self.assertEqual(
            kwargs["output_dir"],
            Path("data")
            / "development"
            / f"analysis-{LABELING_ENRICHMENT_PLAN_VERSION}",
        )

    @patch("nextwave.__main__.run_enrichment")
    def test_labeling_enrichment_run_uses_versioned_default_output(self, run) -> None:
        from nextwave.labeling.enrichment_run import (
            ENRICHMENT_RESULT_VERSION,
            LabelingEnrichmentRunPaths,
        )

        run.return_value = LabelingEnrichmentRunPaths(
            manifest=Path("out/manifest.json"),
            request_results=Path("out/request_results.jsonl"),
            documents=Path("out/documents.jsonl"),
            coverage=Path("out/coverage.jsonl"),
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(["labeling-enrichment-run", "--plan", "plan", "--work", "work"])

        self.assertEqual(exit_code, 0)
        _, kwargs = run.call_args
        self.assertEqual(
            kwargs["output_dir"],
            Path("data") / "development" / ENRICHMENT_RESULT_VERSION,
        )
        self.assertEqual(
            kwargs["output_dir"],
            Path("data/development/labeling-enrichment-result-v2"),
        )

    @patch("nextwave.__main__.run_enrichment")
    def test_enrichment_run_passes_connector_filter(self, run) -> None:
        from nextwave.labeling.enrichment_run import LabelingEnrichmentRunPaths

        run.return_value = LabelingEnrichmentRunPaths(
            manifest=Path("out/manifest.json"),
            request_results=Path("out/request_results.jsonl"),
            documents=Path("out/documents.jsonl"),
            coverage=Path("out/coverage.jsonl"),
        )
        with redirect_stdout(io.StringIO()):
            exit_code = main(
                [
                    "labeling-enrichment-run",
                    "--plan",
                    "plan",
                    "--work",
                    "work",
                    "--connector",
                    "openalex",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(run.call_args.kwargs["connectors"], ["openalex"])

    def test_enrichment_commands_have_no_stale_v1_defaults(self) -> None:
        from nextwave.__main__ import _build_parser

        help_text = _build_parser().format_help()
        self.assertNotIn("labeling-enrichment-plan-v1", help_text)
        self.assertNotIn("labeling-enrichment-result-v1", help_text)

    @patch("nextwave.__main__.export_target_enrichment_plan")
    def test_target_enrichment_plan_uses_versioned_default_output(self, export) -> None:
        from nextwave.labeling.enrichment_plan import (
            LABELING_TARGET_ENRICHMENT_PLAN_VERSION,
            LabelingEnrichmentPlanPaths,
        )

        export.return_value = LabelingEnrichmentPlanPaths(
            plan=Path("out/plan.json"), manifest=Path("out/manifest.json")
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(["labeling-target-enrichment-plan", "--candidates", "targets.json"])

        self.assertEqual(exit_code, 0)
        _, kwargs = export.call_args
        self.assertEqual(kwargs["candidates_file"], Path("targets.json"))
        self.assertEqual(
            kwargs["output_dir"],
            Path("data") / "development" / LABELING_TARGET_ENRICHMENT_PLAN_VERSION,
        )

    @patch("nextwave.__main__._runtime_environment", return_value={})
    @patch("nextwave.__main__.run_target_gate")
    def test_target_gate_uses_versioned_default_output(self, run_gate, _environment) -> None:
        from nextwave.labeling.target_gate import (
            LABELING_TARGET_GATE_VERSION,
            LabelingTargetGatePaths,
        )

        run_gate.return_value = LabelingTargetGatePaths(
            gate_results=Path("out/gate_results.jsonl"),
            groundings=Path("out/groundings.jsonl"),
            issues=Path("out/issues.jsonl"),
            manifest=Path("out/manifest.json"),
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                [
                    "labeling-target-gate-run",
                    "--plan",
                    "plan",
                    "--result",
                    "result",
                ]
            )

        self.assertEqual(exit_code, 0)
        _, kwargs = run_gate.call_args
        self.assertEqual(kwargs["plan_dir"], Path("plan"))
        self.assertEqual(kwargs["result_dir"], Path("result"))
        self.assertEqual(
            kwargs["output_dir"],
            Path("data") / "development" / LABELING_TARGET_GATE_VERSION,
        )

    @patch("nextwave.__main__.export_evidence_input_plan")
    def test_evidence_input_plan_uses_versioned_default_output(self, export) -> None:
        from nextwave.__main__ import _build_parser
        from nextwave.labeling.evidence_input_plan import (
            LABELING_EVIDENCE_INPUT_PLAN_VERSION,
            LabelingEvidenceInputPlanPaths,
        )

        export.return_value = LabelingEvidenceInputPlanPaths(
            manifest=Path("out/manifest.json"),
            media_reranked=Path("out/media_reranked.jsonl"),
            evidence_input_documents=Path("out/evidence_input_documents.jsonl"),
            coverage=Path("out/coverage.jsonl"),
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                [
                    "labeling-evidence-input-plan",
                    "--plan",
                    "plan",
                    "--result",
                    "result",
                    "--relevance",
                    "relevance",
                    "--media",
                    "media",
                ]
            )

        self.assertEqual(exit_code, 0)
        _, kwargs = export.call_args
        self.assertEqual(
            kwargs["output_dir"],
            Path("data") / "development" / LABELING_EVIDENCE_INPUT_PLAN_VERSION,
        )
        self.assertEqual(
            kwargs["output_dir"],
            Path("data/development/labeling-evidence-input-plan-v4"),
        )
        help_text = _build_parser().format_help()
        self.assertNotIn("labeling-evidence-input-plan-v1", help_text)

    @patch("nextwave.__main__.export_evidence_llm_plan")
    def test_evidence_llm_plan_uses_versioned_default_output(self, export) -> None:
        from nextwave.__main__ import _build_parser
        from nextwave.labeling.evidence_llm_plan import (
            LABELING_EVIDENCE_LLM_PLAN_VERSION,
            LabelingEvidenceLlmPlanPaths,
        )

        export.return_value = LabelingEvidenceLlmPlanPaths(
            manifest=Path("out/manifest.json"),
            tasks=Path("out/tasks.jsonl"),
            coverage=Path("out/coverage.jsonl"),
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                [
                    "labeling-evidence-llm-plan",
                    "--input",
                    "input",
                ]
            )

        self.assertEqual(exit_code, 0)
        _, kwargs = export.call_args
        self.assertEqual(
            kwargs["output_dir"],
            Path("data") / "development" / LABELING_EVIDENCE_LLM_PLAN_VERSION,
        )
        self.assertNotIn(
            "labeling-evidence-llm-plan-v1",
            _build_parser().format_help(),
        )

    @patch("nextwave.__main__.merge_evidence_llm_results")
    def test_evidence_llm_merge_uses_versioned_default_output(self, merge) -> None:
        from nextwave.labeling.evidence_llm_merge import (
            LABELING_EVIDENCE_LLM_MERGED_RESULT_VERSION,
            LabelingEvidenceLlmMergePaths,
        )

        merge.return_value = LabelingEvidenceLlmMergePaths(
            manifest=Path("out/manifest.json"),
            request_results=Path("out/request_results.jsonl"),
            claims=Path("out/claims.jsonl"),
            document_results=Path("out/document_results.jsonl"),
            issues=Path("out/issues.jsonl"),
            coverage=Path("out/coverage.jsonl"),
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                [
                    "labeling-evidence-llm-merge",
                    "--primary",
                    "primary",
                    "--retry",
                    "retry",
                ]
            )

        self.assertEqual(exit_code, 0)
        _, kwargs = merge.call_args
        self.assertEqual(
            kwargs["output_dir"],
            Path("data") / "development" / LABELING_EVIDENCE_LLM_MERGED_RESULT_VERSION,
        )

    @patch("nextwave.__main__.export_feature_table")
    def test_feature_table_uses_versioned_default_output(self, export) -> None:
        from nextwave.evaluation import FEATURE_TABLE_VERSION, FeatureTablePaths

        export.return_value = FeatureTablePaths(
            features=Path("out/features.jsonl"), manifest=Path("out/manifest.json")
        )
        with redirect_stdout(io.StringIO()):
            exit_code = main(
                [
                    "evaluation-feature-table",
                    "--positive",
                    "positive",
                    "--positive-enrichment",
                    "positive-result",
                    "--negative-plan",
                    "negative-plan",
                    "--negative-enrichment",
                    "negative-result",
                    "--adjudication",
                    "adjudication",
                ]
            )
        self.assertEqual(exit_code, 0)
        _, kwargs = export.call_args
        self.assertEqual(kwargs["output_dir"], Path("data") / "development" / FEATURE_TABLE_VERSION)

    @patch("nextwave.__main__.export_identity_review")
    def test_identity_review_uses_versioned_default_output(self, export) -> None:
        from nextwave.evaluation import IDENTITY_REVIEW_VERSION, IdentityReviewPaths

        export.return_value = IdentityReviewPaths(
            identities=Path("out/identities.jsonl"),
            pair_decisions=Path("out/pair_decisions.jsonl"),
            manifest=Path("out/manifest.json"),
        )
        with redirect_stdout(io.StringIO()):
            exit_code = main(
                [
                    "evaluation-identity-review",
                    "--positive-plan",
                    "positive-plan",
                    "--negative-plan",
                    "negative-plan",
                    "--decisions",
                    "decisions.json",
                ]
            )
        self.assertEqual(exit_code, 0)
        _, kwargs = export.call_args
        self.assertEqual(
            kwargs["output_dir"],
            Path("data") / "development" / IDENTITY_REVIEW_VERSION,
        )

    @patch("nextwave.__main__.export_gate_noise_evaluation")
    def test_gate_noise_uses_versioned_default_output(self, export) -> None:
        from nextwave.evaluation import (
            GATE_NOISE_EVALUATION_VERSION,
            GateNoiseEvaluationPaths,
        )

        export.return_value = GateNoiseEvaluationPaths(
            rows=Path("out/noise_results.jsonl"),
            report=Path("out/report.json"),
            manifest=Path("out/manifest.json"),
        )
        with redirect_stdout(io.StringIO()):
            exit_code = main(
                [
                    "evaluation-gate-noise",
                    "--selection",
                    "selection.json",
                    "--discovery-root",
                    "discovery",
                ]
            )
        self.assertEqual(exit_code, 0)
        _, kwargs = export.call_args
        self.assertEqual(
            kwargs["output_dir"],
            Path("data") / "development" / GATE_NOISE_EVALUATION_VERSION,
        )

    @patch("nextwave.__main__.export_model_report")
    def test_model_report_uses_versioned_default_output(self, export) -> None:
        from nextwave.evaluation import MODEL_REPORT_VERSION, ModelReportPaths

        export.return_value = ModelReportPaths(
            predictions=Path("out/oof_predictions.jsonl"),
            report=Path("out/metrics.json"),
            model=Path("out/model.json"),
        )
        with redirect_stdout(io.StringIO()):
            exit_code = main(
                [
                    "evaluation-model-report",
                    "--features",
                    "features",
                ]
            )
        self.assertEqual(exit_code, 0)
        _, kwargs = export.call_args
        self.assertEqual(kwargs["output_dir"], Path("data") / "development" / MODEL_REPORT_VERSION)

    @patch("nextwave.__main__.export_analysis_feature_table")
    def test_analysis_feature_table_uses_unlabeled_inputs(self, export) -> None:
        from nextwave.evaluation import (
            ANALYSIS_FEATURE_TABLE_VERSION,
            AnalysisFeatureTablePaths,
        )

        export.return_value = AnalysisFeatureTablePaths(
            features=Path("out/features.jsonl"), manifest=Path("out/manifest.json")
        )
        with redirect_stdout(io.StringIO()):
            exit_code = main(
                [
                    "analysis-feature-table",
                    "--analysis-plan",
                    "plan",
                    "--enrichment-result",
                    "enrichment",
                    "--temporal-counts",
                    "temporal",
                ]
            )
        self.assertEqual(exit_code, 0)
        _, kwargs = export.call_args
        self.assertEqual(kwargs["analysis_plan_dir"], Path("plan"))
        self.assertEqual(kwargs["enrichment_result_dir"], Path("enrichment"))
        self.assertEqual(kwargs["temporal_count_dir"], Path("temporal"))
        self.assertEqual(
            kwargs["output_dir"],
            Path("data") / "development" / ANALYSIS_FEATURE_TABLE_VERSION,
        )

    @patch("nextwave.__main__.export_combined_enrichment")
    def test_analysis_enrichment_merge_uses_versioned_output(self, export) -> None:
        from nextwave.evaluation import (
            ANALYSIS_COMBINED_ENRICHMENT_VERSION,
            CombinedEnrichmentPaths,
        )

        export.return_value = CombinedEnrichmentPaths(
            documents=Path("out/documents.jsonl"),
            coverage=Path("out/coverage.jsonl"),
            excluded_documents=Path("out/excluded_documents.jsonl"),
            manifest=Path("out/manifest.json"),
        )
        with redirect_stdout(io.StringIO()):
            exit_code = main(
                [
                    "analysis-enrichment-merge",
                    "--analysis-plan",
                    "analysis-plan",
                    "--scientific-result",
                    "science",
                    "--exa-plan",
                    "exa-plan",
                    "--exa-result",
                    "exa-result",
                ]
            )
        self.assertEqual(exit_code, 0)
        _, kwargs = export.call_args
        self.assertEqual(kwargs["analysis_plan_dir"], Path("analysis-plan"))
        self.assertEqual(kwargs["scientific_result_dir"], Path("science"))
        self.assertEqual(kwargs["exa_plan_dir"], Path("exa-plan"))
        self.assertEqual(kwargs["exa_result_dir"], Path("exa-result"))
        self.assertEqual(
            kwargs["output_dir"],
            Path("data")
            / "development"
            / ANALYSIS_COMBINED_ENRICHMENT_VERSION,
        )

    @patch("nextwave.__main__.export_analysis_inference")
    def test_analysis_inference_uses_frozen_model(self, export) -> None:
        from nextwave.evaluation import ANALYSIS_INFERENCE_VERSION, AnalysisInferencePaths

        export.return_value = AnalysisInferencePaths(
            predictions=Path("out/predictions.jsonl"),
            feature_importance=Path("out/feature_importance.jsonl"),
            manifest=Path("out/manifest.json"),
        )
        with redirect_stdout(io.StringIO()):
            exit_code = main(
                [
                    "analysis-inference",
                    "--features",
                    "features",
                    "--model",
                    "model",
                ]
            )
        self.assertEqual(exit_code, 0)
        _, kwargs = export.call_args
        self.assertEqual(kwargs["feature_dir"], Path("features"))
        self.assertEqual(kwargs["model_dir"], Path("model"))
        self.assertEqual(
            kwargs["output_dir"],
            Path("data") / "development" / ANALYSIS_INFERENCE_VERSION,
        )

    @patch("nextwave.__main__.export_analysis_result")
    def test_analysis_result_accepts_all_evidence_pages(self, export) -> None:
        from nextwave.evaluation import ANALYSIS_RESULT_VERSION, AnalysisResultPaths

        export.return_value = AnalysisResultPaths(
            candidates=Path("out/candidates.jsonl"),
            top15=Path("out/top15.json"),
            summary=Path("out/summary.json"),
            manifest=Path("out/manifest.json"),
        )
        with redirect_stdout(io.StringIO()):
            exit_code = main(
                [
                    "analysis-result",
                    "--analysis-plan",
                    "plan",
                    "--combined-result",
                    "combined",
                    "--features",
                    "features",
                    "--inference",
                    "inference",
                    "--evidence-result",
                    "page-1",
                    "--evidence-result",
                    "page-2",
                ]
            )
        self.assertEqual(exit_code, 0)
        _, kwargs = export.call_args
        self.assertEqual(
            kwargs["evidence_result_dirs"], (Path("page-1"), Path("page-2"))
        )
        self.assertEqual(
            kwargs["output_dir"],
            Path("data") / "development" / ANALYSIS_RESULT_VERSION,
        )

    @patch("nextwave.__main__.export_analysis_shortlist")
    def test_analysis_evidence_shortlist_is_not_final_top15(self, export) -> None:
        from nextwave.evaluation import ANALYSIS_SHORTLIST_VERSION, AnalysisShortlistPaths

        export.return_value = AnalysisShortlistPaths(
            shortlist=Path("out/shortlist.jsonl"), manifest=Path("out/manifest.json")
        )
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                ["analysis-evidence-shortlist", "--inference", "inference", "--limit", "25"]
            )
        self.assertEqual(exit_code, 0)
        _, kwargs = export.call_args
        self.assertEqual(kwargs["inference_dir"], Path("inference"))
        self.assertEqual(kwargs["limit"], 25)
        self.assertEqual(kwargs["offset"], 0)
        self.assertEqual(
            kwargs["output_dir"],
            Path("data") / "development" / ANALYSIS_SHORTLIST_VERSION,
        )
        self.assertIn("не финальный TOP-15", stdout.getvalue())

    @patch("nextwave.__main__.export_temporal_count_plan")
    def test_temporal_count_plan_uses_versioned_default_output(self, export) -> None:
        from nextwave.evaluation import (
            TEMPORAL_COUNT_PLAN_VERSION,
            TemporalCountPlanPaths,
        )

        export.return_value = TemporalCountPlanPaths(
            plan=Path("out/plan.json"), manifest=Path("out/manifest.json")
        )
        with redirect_stdout(io.StringIO()):
            exit_code = main(
                [
                    "evaluation-temporal-count-plan",
                    "--positive-plan",
                    "positive",
                    "--negative-plan",
                    "negative",
                ]
            )
        self.assertEqual(exit_code, 0)
        _, kwargs = export.call_args
        self.assertEqual(
            kwargs["output_dir"],
            Path("data") / "development" / TEMPORAL_COUNT_PLAN_VERSION,
        )

    @patch("nextwave.__main__.export_analysis_temporal_count_plan")
    def test_analysis_temporal_count_plan_uses_one_analysis_plan(self, export) -> None:
        from nextwave.evaluation import (
            TEMPORAL_COUNT_PLAN_VERSION,
            TemporalCountPlanPaths,
        )

        export.return_value = TemporalCountPlanPaths(
            plan=Path("out/plan.json"), manifest=Path("out/manifest.json")
        )
        with redirect_stdout(io.StringIO()):
            exit_code = main(
                [
                    "analysis-temporal-count-plan",
                    "--analysis-plan",
                    "analysis-plan",
                ]
            )
        self.assertEqual(exit_code, 0)
        _, kwargs = export.call_args
        self.assertEqual(kwargs["analysis_plan_dir"], Path("analysis-plan"))
        self.assertEqual(
            kwargs["output_dir"],
            Path("data")
            / "development"
            / f"analysis-{TEMPORAL_COUNT_PLAN_VERSION}",
        )

    @patch("nextwave.__main__._runtime_environment", return_value={})
    @patch("nextwave.__main__.run_temporal_counts")
    def test_temporal_count_run_uses_versioned_default_output(self, run, _environment) -> None:
        from nextwave.evaluation import TEMPORAL_COUNT_RESULT_VERSION

        run.return_value = {
            "count_results.jsonl": Path("out/count_results.jsonl"),
            "candidate_temporal_features.jsonl": Path("out/candidate_temporal_features.jsonl"),
            "manifest.json": Path("out/manifest.json"),
        }
        with redirect_stdout(io.StringIO()):
            exit_code = main(
                [
                    "evaluation-temporal-count-run",
                    "--plan",
                    "plan",
                    "--work",
                    "work",
                ]
            )
        self.assertEqual(exit_code, 0)
        _, kwargs = run.call_args
        self.assertEqual(
            kwargs["output_dir"],
            Path("data") / "development" / TEMPORAL_COUNT_RESULT_VERSION,
        )


if __name__ == "__main__":
    unittest.main()
