from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path

from . import __version__
from .datasets import (
    DATASET_VERSION,
    OrganizerArtifactError,
    OrganizerWorkbookError,
    build_organizer_dataset,
)
from .discovery import (
    build_discovery_pipeline_from_environment,
    build_discovery_plan,
    build_query_resolver_from_environment,
    save_discovery_run,
)
from .evaluation import (
    ANALYSIS_COMBINED_ENRICHMENT_VERSION,
    ANALYSIS_FEATURE_TABLE_VERSION,
    ANALYSIS_INFERENCE_VERSION,
    ANALYSIS_SHORTLIST_VERSION,
    EXA_ENRICHMENT_PLAN_VERSION,
    EXA_ENRICHMENT_RESULT_VERSION,
    FEATURE_TABLE_VERSION,
    GATE_NOISE_EVALUATION_VERSION,
    IDENTITY_REVIEW_VERSION,
    MODEL_REPORT_VERSION,
    TEMPORAL_COUNT_PLAN_VERSION,
    TEMPORAL_COUNT_RESULT_VERSION,
    export_analysis_feature_table,
    export_analysis_inference,
    export_analysis_shortlist,
    export_analysis_temporal_count_plan,
    export_combined_enrichment,
    export_exa_enrichment_plan,
    export_feature_table,
    export_gate_noise_evaluation,
    export_identity_review,
    export_model_report,
    export_temporal_count_plan,
    run_exa_enrichment,
    run_temporal_counts,
)
from .labeling.corpus_readiness import (
    LABELING_CORPUS_READINESS_VERSION,
    export_corpus_readiness,
)
from .labeling.enrichment_plan import (
    LABELING_ENRICHMENT_PLAN_VERSION,
    LABELING_TARGET_ENRICHMENT_PLAN_VERSION,
    export_analysis_enrichment_plan,
    export_enrichment_plan,
    export_target_enrichment_plan,
)
from .labeling.enrichment_run import ENRICHMENT_RESULT_VERSION, run_enrichment
from .labeling.evidence_input_plan import (
    LABELING_EVIDENCE_INPUT_PLAN_VERSION,
    export_evidence_input_plan,
)
from .labeling.evidence_llm_merge import (
    LABELING_EVIDENCE_LLM_MERGED_RESULT_VERSION,
    merge_evidence_llm_results,
)
from .labeling.evidence_llm_plan import (
    LABELING_EVIDENCE_LLM_PLAN_VERSION,
    export_evidence_llm_plan,
)
from .labeling.evidence_llm_run import (
    LABELING_EVIDENCE_LLM_RESULT_VERSION,
    run_evidence_llm,
)
from .labeling.export import export_labeling_bundle
from .labeling.finalize import LABELING_FINALIZE_VERSION, finalize_labeling_bundle
from .labeling.hype_input_plan import (
    HYPE_INPUT_POLICY_VERSION,
    export_hype_evidence_input,
)
from .labeling.maturity_input_plan import (
    MATURITY_INPUT_POLICY_VERSION,
    export_maturity_evidence_input,
)
from .labeling.media_fetch_run import (
    LABELING_MEDIA_FETCH_RESULT_VERSION,
    run_media_fetch,
)
from .labeling.relevance_plan import (
    LABELING_RELEVANCE_PLAN_VERSION,
    export_relevance_plan,
)
from .labeling.rubric_audit import (
    LABELING_RUBRIC_AUDIT_VERSION,
    export_rubric_audit,
)
from .labeling.target_gate import LABELING_TARGET_GATE_VERSION, run_target_gate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nextwave",
        description="Инструменты воспроизводимого анализа NextWave",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    commands = parser.add_subparsers(dest="command")

    dataset_build = commands.add_parser(
        "dataset-build",
        help="проверить XLSX организаторов и собрать версионированные артефакты",
    )
    dataset_build.add_argument(
        "--input",
        type=Path,
        required=True,
        help="путь к исходному XLSX организаторов",
    )
    dataset_build.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "processed" / DATASET_VERSION,
        help="новый каталог результата внутри data/processed",
    )
    query_resolve = commands.add_parser(
        "query-resolve",
        help="интерпретировать запрос и сопоставить его с таксономией OpenAlex",
    )
    query_resolve.add_argument(
        "--query",
        required=True,
        help="технологическое направление или конкретная технология",
    )
    query_resolve.add_argument(
        "--env-file",
        type=Path,
        default=Path("config") / "hackathon.env",
        help="файл runtime-настроек; переменные процесса имеют приоритет",
    )
    discovery_run = commands.add_parser(
        "discovery-run",
        help="выполнить ограниченный live-поиск и сохранить запуск для разметки",
    )
    discovery_run.add_argument(
        "--query",
        required=True,
        help="технологическое направление или конкретная технология",
    )
    discovery_run.add_argument(
        "--analysis-id",
        required=True,
        help="устойчивый идентификатор анализа, например analysis-ai-001",
    )
    discovery_run.add_argument(
        "--analysis-scope-key",
        default=None,
        help="ключ широкой области; для labeling — один из 6 ключей организаторов, "
        "иначе берётся scope_id из разбора запроса",
    )
    discovery_run.add_argument(
        "--domain",
        default=None,
        help="человекочитаемая область; по умолчанию — исходный текст запроса",
    )
    discovery_run.add_argument(
        "--cutoff-date",
        default="2026-09-15",
        help="дата среза в формате YYYY-MM-DD; для labeling только 2026-09-15",
    )
    discovery_run.add_argument(
        "--published-from",
        default=None,
        help="начало окна поиска YYYY-MM-DD; по умолчанию срез минус 365 дней",
    )
    discovery_run.add_argument(
        "--env-file",
        type=Path,
        default=Path("config") / "hackathon.env",
        help="файл runtime-настроек; переменные процесса имеют приоритет",
    )
    discovery_run.add_argument(
        "--output-root",
        type=Path,
        default=Path("data") / "development" / "discovery",
        help="корень для каталогов запусков",
    )
    labeling_export = commands.add_parser(
        "labeling-export",
        help="собрать очередь разметки из запусков в книгу, JSONL и опись",
    )
    labeling_export.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="каталоги запусков discovery (plan.json + pipeline_result.json + manifest.json)",
    )
    labeling_export.add_argument(
        "--template",
        type=Path,
        default=Path("templates") / "labeling_workbook.xlsx",
        help="шаблон книги экспертной проверки",
    )
    labeling_export.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "development" / "labeling-export-v1",
        help="новый каталог результата",
    )
    labeling_export.add_argument(
        "--domain-map",
        action="append",
        default=[],
        metavar="RUN_ID=Domain",
        help="привязка запуска к контролируемой области "
        "(Edge, Защита ИИ, Индустриальный ИИ, Инфраструктура ИИ, Роботы, Финтех); "
        "нужна, если в сейфе свободный текст вместо области",
    )
    labeling_export.add_argument(
        "--candidate-selection",
        type=Path,
        help="reviewed JSON с точным group_id для каждого нейтрального слота",
    )
    labeling_export.add_argument(
        "--noise-selection",
        type=Path,
        help="reviewed JSON с точным источником для каждого noise-слота",
    )
    labeling_finalize = commands.add_parser(
        "labeling-finalize",
        help="проверить решения в книге и выпустить обучающий отрицательный корпус",
    )
    labeling_finalize.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="исходный каталог labeling-export с кандидатами и noise",
    )
    labeling_finalize.add_argument(
        "--workbook",
        type=Path,
        required=True,
        help="заполненная и проверенная книга разметки",
    )
    labeling_finalize.add_argument(
        "--enrichment-result",
        type=Path,
        required=True,
        help="кандидатский enrichment-result с полным scientific/industry coverage",
    )
    labeling_finalize.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "processed" / LABELING_FINALIZE_VERSION,
        help="новый каталог проверенного корпуса",
    )
    enrichment_plan = commands.add_parser(
        "labeling-enrichment-plan",
        help="проверить candidate bundle и выпустить неизменяемый план enrichment",
    )
    enrichment_plan.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="каталог labeling-экспорта или organizer positive dataset",
    )
    enrichment_plan.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "development" / LABELING_ENRICHMENT_PLAN_VERSION,
        help="новый каталог результата",
    )
    enrichment_plan.add_argument(
        "--search-terms",
        type=Path,
        help="проверенный JSON с короткими поисковыми терминами для organizer positives",
    )
    analysis_enrichment_plan = commands.add_parser(
        "analysis-enrichment-plan",
        help="построить enrichment-план кандидатов одного discovery-запроса",
    )
    analysis_enrichment_plan.add_argument(
        "--run",
        type=Path,
        required=True,
        help="каталог одного полного discovery-run",
    )
    analysis_enrichment_plan.add_argument(
        "--output",
        type=Path,
        default=(
            Path("data")
            / "development"
            / f"analysis-{LABELING_ENRICHMENT_PLAN_VERSION}"
        ),
        help="новый каталог query-specific enrichment-плана",
    )
    exa_enrichment_plan = commands.add_parser(
        "analysis-exa-enrichment-plan",
        help="построить быстрый Exa media enrichment для всех кандидатов анализа",
    )
    exa_enrichment_plan.add_argument(
        "--analysis-plan",
        type=Path,
        required=True,
        help="каталог query-specific enrichment-плана с кандидатами",
    )
    exa_enrichment_plan.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "development" / EXA_ENRICHMENT_PLAN_VERSION,
        help="новый каталог Exa-плана",
    )
    exa_enrichment_run = commands.add_parser(
        "analysis-exa-enrichment-run",
        help="выполнить Exa enrichment с resume и параллельными запросами",
    )
    exa_enrichment_run.add_argument("--plan", type=Path, required=True)
    exa_enrichment_run.add_argument("--work", type=Path, required=True)
    exa_enrichment_run.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "development" / EXA_ENRICHMENT_RESULT_VERSION,
    )
    exa_enrichment_run.add_argument(
        "--env-file", type=Path, default=Path("config") / "hackathon.env"
    )
    exa_enrichment_run.add_argument("--max-new-requests", type=int, default=None)
    exa_enrichment_run.add_argument("--concurrency", type=int, default=5)
    enrichment_merge = commands.add_parser(
        "analysis-enrichment-merge",
        help="объединить OpenAlex и очищенный Exa enrichment одного анализа",
    )
    enrichment_merge.add_argument("--analysis-plan", type=Path, required=True)
    enrichment_merge.add_argument("--scientific-result", type=Path, required=True)
    enrichment_merge.add_argument("--exa-plan", type=Path, required=True)
    enrichment_merge.add_argument("--exa-result", type=Path, required=True)
    enrichment_merge.add_argument(
        "--output",
        type=Path,
        default=(
            Path("data") / "development" / ANALYSIS_COMBINED_ENRICHMENT_VERSION
        ),
    )
    target_enrichment_plan = commands.add_parser(
        "labeling-target-enrichment-plan",
        help="построить enrichment для нейтральных целевых кандидатов дефицита",
    )
    target_enrichment_plan.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="JSON labeling-target-candidates-v1 без меток и вердиктов",
    )
    target_enrichment_plan.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "development" / LABELING_TARGET_ENRICHMENT_PLAN_VERSION,
        help="новый каталог target enrichment plan",
    )
    enrichment_run = commands.add_parser(
        "labeling-enrichment-run",
        help="выполнить план enrichment с возобновляемым work-хранилищем",
    )
    enrichment_run.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="неизменяемый каталог плана (plan.json + manifest.json)",
    )
    enrichment_run.add_argument(
        "--work",
        type=Path,
        required=True,
        help="постоянное рабочее хранилище выполненных запросов",
    )
    enrichment_run.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "development" / ENRICHMENT_RESULT_VERSION,
        help="новый каталог результата",
    )
    enrichment_run.add_argument(
        "--env-file",
        type=Path,
        default=Path("config") / "hackathon.env",
        help="файл runtime-настроек; переменные процесса имеют приоритет",
    )
    target_gate = commands.add_parser(
        "labeling-target-gate-run",
        help="проверить grounding нейтральных целей и выполнить Candidate Gate",
    )
    target_gate.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="каталог target enrichment plan",
    )
    target_gate.add_argument(
        "--result",
        type=Path,
        required=True,
        help="полный результат target enrichment",
    )
    target_gate.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "development" / LABELING_TARGET_GATE_VERSION,
        help="новый каталог решений target Gate",
    )
    target_gate.add_argument(
        "--env-file",
        type=Path,
        default=Path("config") / "hackathon.env",
        help="файл runtime-настроек; переменные процесса имеют приоритет",
    )
    relevance_plan = commands.add_parser(
        "labeling-relevance-plan",
        help="офлайн-ранжирование документов enrichment без API и LLM",
    )
    relevance_plan.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="неизменяемый каталог плана enrichment (plan.json + manifest.json)",
    )
    relevance_plan.add_argument(
        "--result",
        type=Path,
        required=True,
        help="неизменяемый каталог результата enrichment",
    )
    relevance_plan.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "development" / LABELING_RELEVANCE_PLAN_VERSION,
        help="новый каталог результата",
    )
    media_fetch_run = commands.add_parser(
        "labeling-media-fetch-run",
        help="загрузить страницы media-очереди с возобновляемым work-хранилищем",
    )
    media_fetch_run.add_argument(
        "--relevance",
        type=Path,
        required=True,
        help="каталог relevance-плана (media_fetch_queue.jsonl + manifest.json)",
    )
    media_fetch_run.add_argument(
        "--result",
        type=Path,
        required=True,
        help="неизменяемый каталог результата enrichment",
    )
    media_fetch_run.add_argument(
        "--work",
        type=Path,
        required=True,
        help="постоянное рабочее хранилище загруженных страниц",
    )
    media_fetch_run.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "development" / LABELING_MEDIA_FETCH_RESULT_VERSION,
        help="новый каталог результата",
    )
    evidence_input_plan = commands.add_parser(
        "labeling-evidence-input-plan",
        help="переоценить media-тексты и собрать единый вход Evidence LLM",
    )
    evidence_input_plan.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="неизменяемый каталог плана enrichment (plan.json + manifest.json)",
    )
    evidence_input_plan.add_argument(
        "--result",
        type=Path,
        required=True,
        help="неизменяемый каталог результата enrichment",
    )
    evidence_input_plan.add_argument(
        "--relevance",
        type=Path,
        required=True,
        help="каталог relevance-плана (shortlist + очередь + manifest.json)",
    )
    evidence_input_plan.add_argument(
        "--media",
        type=Path,
        required=True,
        help="каталог результата загрузки media-страниц",
    )
    evidence_input_plan.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "development" / LABELING_EVIDENCE_INPUT_PLAN_VERSION,
        help="новый каталог результата",
    )
    evidence_llm_plan = commands.add_parser(
        "labeling-evidence-llm-plan",
        help="собрать детерминированные задания Evidence LLM без вызова модели",
    )
    evidence_llm_plan.add_argument(
        "--input",
        type=Path,
        required=True,
        help="каталог evidence input plan (documents + coverage + manifest.json)",
    )
    evidence_llm_plan.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "development" / LABELING_EVIDENCE_LLM_PLAN_VERSION,
        help="новый каталог результата",
    )
    evidence_llm_run = commands.add_parser(
        "labeling-evidence-llm-run",
        help="выполнить задания Evidence LLM с возобновляемым work-хранилищем",
    )
    evidence_llm_run.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="каталог плана Evidence LLM (tasks.jsonl + coverage.jsonl + manifest.json)",
    )
    evidence_llm_run.add_argument(
        "--work",
        type=Path,
        required=True,
        help="постоянное рабочее хранилище выполненных заданий",
    )
    evidence_llm_run.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "development" / LABELING_EVIDENCE_LLM_RESULT_VERSION,
        help="новый каталог результата",
    )
    evidence_llm_run.add_argument(
        "--env-file",
        type=Path,
        default=Path("config") / "hackathon.env",
        help="файл runtime-настроек; переменные процесса имеют приоритет",
    )
    evidence_llm_run.add_argument(
        "--max-new-tasks",
        type=int,
        default=None,
        help="обработать не более N новых заданий (повторное использование не считается)",
    )
    evidence_llm_run.add_argument(
        "--candidate-id",
        action="append",
        default=None,
        dest="candidate_ids",
        help="обработать только указанного кандидата (можно повторять)",
    )
    evidence_llm_merge = commands.add_parser(
        "labeling-evidence-llm-merge",
        help="объединить полный Evidence-результат с одним целевым retry без API",
    )
    evidence_llm_merge.add_argument(
        "--primary", type=Path, required=True, help="каталог полного Evidence-прогона"
    )
    evidence_llm_merge.add_argument(
        "--retry", type=Path, required=True, help="каталог целевого retry"
    )
    evidence_llm_merge.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "development" / LABELING_EVIDENCE_LLM_MERGED_RESULT_VERSION,
        help="новый каталог объединённого результата",
    )
    rubric_audit = commands.add_parser(
        "labeling-rubric-audit",
        help="проверить доказуемость mature/hype без присвоения меток",
    )
    rubric_audit.add_argument("--plan", type=Path, required=True)
    rubric_audit.add_argument("--enrichment-result", type=Path, required=True)
    rubric_audit.add_argument("--evidence-input", type=Path, required=True)
    rubric_audit.add_argument("--evidence-result", type=Path, required=True)
    rubric_audit.add_argument("--relevance", type=Path, required=True)
    rubric_audit.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "development" / LABELING_RUBRIC_AUDIT_VERSION,
        help="новый каталог результата",
    )
    corpus_readiness = commands.add_parser(
        "labeling-corpus-readiness",
        help="проверить готовность корпуса без изменения экспертных меток",
    )
    corpus_readiness.add_argument("--adjudication", type=Path, required=True)
    corpus_readiness.add_argument("--review-queue", type=Path, required=True)
    corpus_readiness.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "development" / LABELING_CORPUS_READINESS_VERSION,
        help="новый каталог результата",
    )
    maturity_input = commands.add_parser(
        "labeling-maturity-evidence-input",
        help="собрать специальный Evidence-вход для проверки зрелости",
    )
    maturity_input.add_argument("--plan", type=Path, required=True)
    maturity_input.add_argument("--audit", type=Path, required=True)
    maturity_input.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "development" / MATURITY_INPUT_POLICY_VERSION,
        help="новый каталог результата",
    )
    hype_input = commands.add_parser(
        "labeling-hype-evidence-input",
        help="собрать Gate-aware Evidence-вход для проверки marketing hype",
    )
    hype_input.add_argument("--plan", type=Path, required=True)
    hype_input.add_argument("--gate", type=Path, required=True)
    hype_input.add_argument("--evidence-input", type=Path, required=True)
    hype_input.add_argument("--media-result", type=Path, required=True)
    hype_input.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "development" / HYPE_INPUT_POLICY_VERSION,
        help="новый каталог результата",
    )
    identity_review = commands.add_parser(
        "evaluation-identity-review",
        help="зафиксировать aliases и cross-corpus группы для grouped CV",
    )
    identity_review.add_argument("--positive-plan", type=Path, required=True)
    identity_review.add_argument("--negative-plan", type=Path, required=True)
    identity_review.add_argument("--decisions", type=Path, required=True)
    identity_review.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "development" / IDENTITY_REVIEW_VERSION,
        help="новый каталог проверенных identity",
    )
    gate_noise = commands.add_parser(
        "evaluation-gate-noise",
        help="измерить Gate и полное удержание на 50 reviewed noise controls",
    )
    gate_noise.add_argument("--selection", type=Path, required=True)
    gate_noise.add_argument("--discovery-root", type=Path, required=True)
    gate_noise.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "development" / GATE_NOISE_EVALUATION_VERSION,
        help="новый каталог отчёта",
    )
    feature_table = commands.add_parser(
        "evaluation-feature-table",
        help="собрать leakage-safe таблицу признаков из frozen corpus",
    )
    feature_table.add_argument("--positive", type=Path, required=True)
    feature_table.add_argument("--positive-enrichment", type=Path, required=True)
    feature_table.add_argument("--negative-plan", type=Path, required=True)
    feature_table.add_argument("--negative-enrichment", type=Path, required=True)
    feature_table.add_argument("--adjudication", type=Path, required=True)
    feature_table.add_argument(
        "--temporal-counts",
        type=Path,
        default=None,
        help="complete openalex-temporal-count-result-v4",
    )
    feature_table.add_argument(
        "--identity-review",
        type=Path,
        default=None,
        help="candidate-identity-review-v1 с проверенными cross-corpus группами",
    )
    feature_table.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "development" / FEATURE_TABLE_VERSION,
        help="новый каталог таблицы признаков",
    )
    analysis_features = commands.add_parser(
        "analysis-feature-table",
        help="собрать безметочные признаки кандидатов одного пользовательского запроса",
    )
    analysis_features.add_argument("--analysis-plan", type=Path, required=True)
    analysis_features.add_argument("--enrichment-result", type=Path, required=True)
    analysis_features.add_argument("--temporal-counts", type=Path, required=True)
    analysis_features.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "development" / ANALYSIS_FEATURE_TABLE_VERSION,
    )
    analysis_inference = commands.add_parser(
        "analysis-inference",
        help="применить frozen model к кандидатам одного пользовательского запроса",
    )
    analysis_inference.add_argument("--features", type=Path, required=True)
    analysis_inference.add_argument("--model", type=Path, required=True)
    analysis_inference.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "development" / ANALYSIS_INFERENCE_VERSION,
    )
    analysis_shortlist = commands.add_parser(
        "analysis-evidence-shortlist",
        help="выбрать query-specific очередь кандидатов для Evidence Duel",
    )
    analysis_shortlist.add_argument("--inference", type=Path, required=True)
    analysis_shortlist.add_argument("--limit", type=int, default=30)
    analysis_shortlist.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "development" / ANALYSIS_SHORTLIST_VERSION,
    )
    model_report = commands.add_parser(
        "evaluation-model-report",
        help="обучить baseline/LogReg и выпустить grouped OOF отчёт",
    )
    model_report.add_argument("--features", type=Path, required=True)
    model_report.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "development" / MODEL_REPORT_VERSION,
        help="новый каталог диагностического отчёта",
    )
    count_plan = commands.add_parser(
        "evaluation-temporal-count-plan",
        help="собрать offline-план uncapped OpenAlex temporal counts",
    )
    count_plan.add_argument("--positive-plan", type=Path, required=True)
    count_plan.add_argument("--negative-plan", type=Path, required=True)
    count_plan.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "development" / TEMPORAL_COUNT_PLAN_VERSION,
    )
    analysis_count_plan = commands.add_parser(
        "analysis-temporal-count-plan",
        help="собрать temporal counts для кандидатов одного пользовательского запроса",
    )
    analysis_count_plan.add_argument("--analysis-plan", type=Path, required=True)
    analysis_count_plan.add_argument(
        "--output",
        type=Path,
        default=(
            Path("data")
            / "development"
            / f"analysis-{TEMPORAL_COUNT_PLAN_VERSION}"
        ),
    )
    count_run = commands.add_parser(
        "evaluation-temporal-count-run",
        help="выполнить OpenAlex temporal counts с resume и raw snapshots",
    )
    count_run.add_argument("--plan", type=Path, required=True)
    count_run.add_argument("--work", type=Path, required=True)
    count_run.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "development" / TEMPORAL_COUNT_RESULT_VERSION,
    )
    count_run.add_argument(
        "--env-file",
        type=Path,
        default=Path("config") / "hackathon.env",
    )
    count_run.add_argument("--max-new-tasks", type=int, default=None)
    return parser


def _run_dataset_build(input_path: Path, output_path: Path) -> int:
    try:
        paths = build_organizer_dataset(input_path, output_path)
    except (OrganizerWorkbookError, OrganizerArtifactError, OSError, ValueError) as error:
        print(f"Не удалось собрать датасет: {error}", file=sys.stderr)
        return 1

    print("Датасет организаторов успешно собран.")
    print(f"Кандидаты: {paths.candidates}")
    print(f"Экспертные аннотации: {paths.annotations}")
    print(f"Manifest: {paths.manifest}")
    return 0


def _read_environment_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            raise ValueError(f"{path}:{line_number}: invalid variable name")
        if key in values:
            raise ValueError(f"{path}:{line_number}: duplicate variable {key}")
        values[key] = value.strip()
    return values


def _runtime_environment(
    env_file: Path,
    process_environment: Mapping[str, str] | None = None,
    *,
    dotenv_path: Path | None = None,
) -> dict[str, str]:
    """Layer settings: committed file < local .env < process environment.

    The local `.env` (git-ignored) overrides the committed settings file, and
    explicit process variables win over everything. Missing files are empty.
    """

    values = _read_environment_file(env_file)
    values.update(_read_environment_file(dotenv_path or Path(".env")))
    values.update(os.environ if process_environment is None else process_environment)
    return values


def _run_query_resolve(query: str, env_file: Path) -> int:
    try:
        environment = _runtime_environment(env_file)
        resolution = build_query_resolver_from_environment(environment).resolve(query)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Не удалось интерпретировать запрос: {error}", file=sys.stderr)
        return 1

    print(json.dumps(resolution.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _parse_iso_date(value: str, flag: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{flag} must use YYYY-MM-DD, got {value!r}") from None


def _run_discovery_run(
    query: str,
    analysis_id: str,
    analysis_scope_key: str | None,
    domain: str | None,
    cutoff_text: str,
    published_from_text: str | None,
    env_file: Path,
    output_root: Path,
) -> int:
    try:
        cutoff_date = _parse_iso_date(cutoff_text, "--cutoff-date")
        if published_from_text:
            published_from = _parse_iso_date(published_from_text, "--published-from")
        else:
            published_from = cutoff_date - timedelta(days=365)
        if published_from > cutoff_date:
            raise ValueError("--published-from must not be later than --cutoff-date")
        environment = _runtime_environment(env_file)
        resolution = build_query_resolver_from_environment(environment).resolve(query)
        # Пользователь домен не выбирает: область выводится из разбора запроса.
        # Явные флаги нужны только разметке, чтобы привязать запуск к квоте области.
        effective_scope_key = (analysis_scope_key or resolution.scope.scope_id).strip()
        effective_domain = (domain or resolution.scope.raw_query).strip()
        if not effective_scope_key or not effective_domain:
            raise ValueError("analysis scope key and domain must not be blank")
        plan = build_discovery_plan(
            analysis_id=analysis_id,
            scope=resolution.scope,
            published_from=published_from,
            cutoff_date=cutoff_date,
        )
        result = build_discovery_pipeline_from_environment(environment).execute(
            plan, progress=print
        )
        run_dir = save_discovery_run(
            plan,
            result,
            analysis_scope_key=effective_scope_key,
            domain=effective_domain,
            output_root=output_root,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Не удалось выполнить поиск: {error}", file=sys.stderr)
        return 1

    coverage = result.to_dict().get("gate_coverage") or {}
    if coverage.get("status") == "partial":
        print(
            "Поиск сохранён с неполным покрытием Candidate Gate: "
            f"проверено {coverage.get('checked_proposals', 0)} из "
            f"{coverage.get('total_proposals', 0)}. "
            "Запуск нельзя использовать для итогового рейтинга или разметки."
        )
    else:
        print("Поиск завершён, запуск сохранён.")
    print(f"Каталог запуска: {run_dir}")
    return 0


def _parse_domain_map(entries: list[str]) -> dict[str, str]:
    """Parse RUN_ID=Domain entries into an audited run-domain mapping."""
    mapping: dict[str, str] = {}
    for entry in entries:
        run_id, separator, domain = entry.partition("=")
        if not separator or not run_id.strip() or not domain.strip():
            raise ValueError(f"invalid --domain-map entry {entry!r}; expected RUN_ID=Domain")
        mapping[run_id.strip()] = domain.strip()
    return mapping


def _run_labeling_export(
    runs: list[str],
    template: Path,
    output: Path,
    domain_map: list[str],
    candidate_selection: Path | None,
    noise_selection: Path | None,
) -> int:
    try:
        paths = export_labeling_bundle(
            run_dirs=tuple(runs),
            template_path=template,
            output_dir=output,
            run_domains=_parse_domain_map(domain_map),
            candidate_selection_path=candidate_selection,
            noise_selection_path=noise_selection,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Не удалось экспортировать очередь: {error}", file=sys.stderr)
        return 1

    print("Очередь разметки успешно экспортирована.")
    print(f"Книга: {paths.workbook}")
    print(f"Кандидаты: {paths.candidates}")
    print(f"Шум: {paths.noise}")
    print(f"Manifest: {paths.manifest}")
    return 0


def _run_labeling_enrichment_plan(bundle: Path, output: Path, search_terms: Path | None) -> int:
    try:
        paths = export_enrichment_plan(
            bundle_dir=bundle,
            output_dir=output,
            search_terms_file=search_terms,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Не удалось построить план enrichment: {error}", file=sys.stderr)
        return 1

    print("План enrichment успешно построен.")
    print(f"План: {paths.plan}")
    print(f"Manifest: {paths.manifest}")
    return 0


def _run_labeling_target_enrichment_plan(candidates: Path, output: Path) -> int:
    try:
        paths = export_target_enrichment_plan(
            candidates_file=candidates,
            output_dir=output,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Не удалось построить target enrichment plan: {error}", file=sys.stderr)
        return 1

    print("Target enrichment plan успешно построен.")
    print(f"План: {paths.plan}")
    print(f"Manifest: {paths.manifest}")
    return 0


def _run_labeling_finalize(
    bundle: Path, workbook: Path, enrichment_result: Path, output: Path
) -> int:
    try:
        paths = finalize_labeling_bundle(
            bundle_dir=bundle,
            workbook_path=workbook,
            enrichment_result_dir=enrichment_result,
            output_dir=output,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Не удалось финализировать разметку: {error}", file=sys.stderr)
        return 1

    print("Проверенный корпус успешно собран.")
    print(f"Кандидаты: {paths.candidates}")
    print(f"Решения: {paths.decisions}")
    print(f"Шум: {paths.noise}")
    print(f"Manifest: {paths.manifest}")
    return 0


def _run_labeling_enrichment_run(plan: Path, work: Path, output: Path, env_file: Path) -> int:
    try:
        environment = _runtime_environment(env_file)
        paths = run_enrichment(
            plan_dir=plan,
            work_dir=work,
            output_dir=output,
            environment=environment,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Не удалось выполнить enrichment: {error}", file=sys.stderr)
        return 1

    print("Enrichment успешно выполнен.")
    print(f"Manifest: {paths.manifest}")
    print(f"Запросы: {paths.request_results}")
    print(f"Документы: {paths.documents}")
    print(f"Покрытие: {paths.coverage}")
    return 0


def _run_labeling_target_gate(plan: Path, result: Path, output: Path, env_file: Path) -> int:
    try:
        environment = _runtime_environment(env_file)
        paths = run_target_gate(
            plan_dir=plan,
            result_dir=result,
            output_dir=output,
            environment=environment,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Не удалось выполнить target Candidate Gate: {error}", file=sys.stderr)
        return 1

    print("Target Candidate Gate успешно выполнен.")
    print(f"Решения: {paths.gate_results}")
    print(f"Grounding: {paths.groundings}")
    print(f"Проблемы: {paths.issues}")
    print(f"Manifest: {paths.manifest}")
    return 0


def _run_labeling_relevance_plan(plan: Path, result: Path, output: Path) -> int:
    try:
        paths = export_relevance_plan(
            plan_dir=plan,
            result_dir=result,
            output_dir=output,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Не удалось построить relevance-план: {error}", file=sys.stderr)
        return 1

    print("Relevance-план успешно построен.")
    print(f"Ранжировка: {paths.ranked_documents}")
    print(f"Shortlist: {paths.scientific_shortlist}")
    print(f"Очередь: {paths.media_fetch_queue}")
    print(f"Покрытие: {paths.coverage}")
    print(f"Manifest: {paths.manifest}")
    return 0


def _run_labeling_media_fetch_run(relevance: Path, result: Path, work: Path, output: Path) -> int:
    try:
        paths = run_media_fetch(
            relevance_dir=relevance,
            result_dir=result,
            work_dir=work,
            output_dir=output,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Не удалось загрузить media-страницы: {error}", file=sys.stderr)
        return 1

    print("Media-страницы успешно загружены.")
    print(f"Страницы: {paths.page_results}")
    print(f"Документы: {paths.enriched_documents}")
    print(f"Покрытие: {paths.coverage}")
    print(f"Manifest: {paths.manifest}")
    return 0


def _run_labeling_evidence_input_plan(
    plan: Path, result: Path, relevance: Path, media: Path, output: Path
) -> int:
    try:
        paths = export_evidence_input_plan(
            plan_dir=plan,
            result_dir=result,
            relevance_dir=relevance,
            media_dir=media,
            output_dir=output,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Не удалось построить evidence-вход: {error}", file=sys.stderr)
        return 1

    print("Evidence-вход успешно построен.")
    print(f"Переоценка: {paths.media_reranked}")
    print(f"Документы: {paths.evidence_input_documents}")
    print(f"Покрытие: {paths.coverage}")
    print(f"Manifest: {paths.manifest}")
    return 0


def _run_labeling_evidence_llm_plan(input_dir: Path, output: Path) -> int:
    try:
        paths = export_evidence_llm_plan(
            input_dir=input_dir,
            output_dir=output,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Не удалось построить план Evidence LLM: {error}", file=sys.stderr)
        return 1

    print("План Evidence LLM успешно построен.")
    print(f"Задания: {paths.tasks}")
    print(f"Покрытие: {paths.coverage}")
    print(f"Manifest: {paths.manifest}")
    return 0


def _run_labeling_evidence_llm_run(
    plan: Path,
    work: Path,
    output: Path,
    env_file: Path,
    max_new_tasks: int | None,
    candidate_ids: list[str] | None,
) -> int:
    try:
        environment = _runtime_environment(env_file)
        paths = run_evidence_llm(
            plan_dir=plan,
            work_dir=work,
            output_dir=output,
            environment=environment,
            max_new_tasks=max_new_tasks,
            candidate_ids=candidate_ids,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Не удалось выполнить Evidence LLM: {error}", file=sys.stderr)
        return 1

    print("Evidence LLM успешно выполнен.")
    print(f"Запросы: {paths.request_results}")
    print(f"Утверждения: {paths.claims}")
    print(f"Покрытие: {paths.coverage}")
    print(f"Manifest: {paths.manifest}")
    return 0


def _run_labeling_evidence_llm_merge(primary: Path, retry: Path, output: Path) -> int:
    try:
        paths = merge_evidence_llm_results(primary_dir=primary, retry_dir=retry, output_dir=output)
    except (OSError, ValueError) as error:
        print(f"Не удалось объединить Evidence-результаты: {error}", file=sys.stderr)
        return 1

    print("Evidence-результаты успешно объединены.")
    print(f"Утверждения: {paths.claims}")
    print(f"Покрытие: {paths.coverage}")
    print(f"Manifest: {paths.manifest}")
    return 0


def _run_labeling_rubric_audit(
    plan: Path,
    enrichment_result: Path,
    evidence_input: Path,
    evidence_result: Path,
    relevance: Path,
    output: Path,
) -> int:
    try:
        paths = export_rubric_audit(
            plan_dir=plan,
            enrichment_result_dir=enrichment_result,
            evidence_input_dir=evidence_input,
            evidence_result_dir=evidence_result,
            relevance_dir=relevance,
            output_dir=output,
        )
    except (OSError, ValueError) as error:
        print(f"Не удалось построить rubric audit: {error}", file=sys.stderr)
        return 1

    print("Rubric audit успешно построен.")
    print(f"Кандидаты: {paths.candidates}")
    print(f"Очередь maturity evidence: {paths.maturity_queue}")
    print(f"Manifest: {paths.manifest}")
    return 0


def _run_labeling_corpus_readiness(
    adjudication: Path, review_queue: Path, output: Path
) -> int:
    try:
        paths = export_corpus_readiness(
            adjudication_dir=adjudication,
            review_queue_dir=review_queue,
            output_dir=output,
        )
    except (OSError, ValueError) as error:
        print(f"Не удалось проверить готовность корпуса: {error}", file=sys.stderr)
        return 1

    print("Готовность корпуса проверена.")
    print(f"Статусы кандидатов: {paths.candidate_status}")
    print(f"Дефициты: {paths.deficits}")
    print(f"Manifest: {paths.manifest}")
    return 0


def _run_labeling_maturity_evidence_input(plan: Path, audit: Path, output: Path) -> int:
    try:
        paths = export_maturity_evidence_input(plan_dir=plan, audit_dir=audit, output_dir=output)
    except (OSError, ValueError) as error:
        print(f"Не удалось построить maturity evidence input: {error}", file=sys.stderr)
        return 1

    print("Maturity evidence input успешно построен.")
    print(f"Документы: {paths.documents}")
    print(f"Покрытие: {paths.coverage}")
    print(f"Manifest: {paths.manifest}")
    return 0


def _run_labeling_hype_evidence_input(
    plan: Path,
    gate: Path,
    evidence_input: Path,
    media_result: Path,
    output: Path,
) -> int:
    try:
        paths = export_hype_evidence_input(
            plan_dir=plan,
            gate_dir=gate,
            evidence_input_dir=evidence_input,
            media_result_dir=media_result,
            output_dir=output,
        )
    except (OSError, ValueError) as error:
        print(f"Не удалось построить hype evidence input: {error}", file=sys.stderr)
        return 1

    print("Hype evidence input успешно построен.")
    print(f"Документы: {paths.documents}")
    print(f"Покрытие: {paths.coverage}")
    print(f"Manifest: {paths.manifest}")
    return 0


def _run_evaluation_feature_table(
    positive: Path,
    positive_enrichment: Path,
    negative_plan: Path,
    negative_enrichment: Path,
    adjudication: Path,
    temporal_counts: Path | None,
    identity_review: Path | None,
    output: Path,
) -> int:
    try:
        paths = export_feature_table(
            positive_dir=positive,
            positive_enrichment_dir=positive_enrichment,
            negative_plan_dir=negative_plan,
            negative_enrichment_dir=negative_enrichment,
            adjudication_dir=adjudication,
            temporal_count_dir=temporal_counts,
            identity_review_dir=identity_review,
            output_dir=output,
        )
    except (OSError, ValueError) as error:
        print(f"Не удалось собрать таблицу признаков: {error}", file=sys.stderr)
        return 1
    print("Таблица признаков успешно собрана.")
    print(f"Признаки: {paths.features}")
    print(f"Manifest: {paths.manifest}")
    return 0


def _run_analysis_feature_table(
    analysis_plan: Path,
    enrichment_result: Path,
    temporal_counts: Path,
    output: Path,
) -> int:
    try:
        paths = export_analysis_feature_table(
            analysis_plan_dir=analysis_plan,
            enrichment_result_dir=enrichment_result,
            temporal_count_dir=temporal_counts,
            output_dir=output,
        )
    except (OSError, ValueError) as error:
        print(f"Не удалось собрать query-specific признаки: {error}", file=sys.stderr)
        return 1
    print("Query-specific признаки успешно собраны.")
    print(f"Признаки: {paths.features}")
    print(f"Manifest: {paths.manifest}")
    return 0


def _run_analysis_inference(features: Path, model: Path, output: Path) -> int:
    try:
        paths = export_analysis_inference(
            feature_dir=features, model_dir=model, output_dir=output
        )
    except (OSError, ValueError) as error:
        print(f"Не удалось выполнить query-specific inference: {error}", file=sys.stderr)
        return 1
    print("Query-specific inference успешно выполнен.")
    print(f"Предсказания: {paths.predictions}")
    print(f"Manifest: {paths.manifest}")
    return 0


def _run_analysis_shortlist(
    inference: Path, limit: int, output: Path
) -> int:
    try:
        paths = export_analysis_shortlist(
            inference_dir=inference, limit=limit, output_dir=output
        )
    except (OSError, ValueError) as error:
        print(f"Не удалось собрать evidence-shortlist: {error}", file=sys.stderr)
        return 1
    print("Evidence-shortlist успешно собран; это ещё не финальный TOP-15.")
    print(f"Очередь: {paths.shortlist}")
    print(f"Manifest: {paths.manifest}")
    return 0


def _run_analysis_enrichment_plan(run: Path, output: Path) -> int:
    try:
        paths = export_analysis_enrichment_plan(run_dir=run, output_dir=output)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Не удалось построить query-specific enrichment-план: {error}", file=sys.stderr)
        return 1

    print("Query-specific enrichment-план успешно построен.")
    print(f"План: {paths.plan}")
    print(f"Manifest: {paths.manifest}")
    return 0


def _run_analysis_exa_enrichment_plan(analysis_plan: Path, output: Path) -> int:
    try:
        paths = export_exa_enrichment_plan(
            analysis_plan_dir=analysis_plan, output_dir=output
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Не удалось построить Exa enrichment-план: {error}", file=sys.stderr)
        return 1
    print("Exa enrichment-план успешно построен без фильтрации кандидатов.")
    print(f"План: {paths.plan}")
    print(f"Manifest: {paths.manifest}")
    return 0


def _run_analysis_exa_enrichment(
    plan: Path,
    work: Path,
    output: Path,
    env_file: Path,
    max_new_requests: int | None,
    concurrency: int,
) -> int:
    try:
        paths = run_exa_enrichment(
            plan_dir=plan,
            work_dir=work,
            output_dir=output,
            environment=_runtime_environment(env_file),
            max_new_requests=max_new_requests,
            concurrency=concurrency,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Не удалось выполнить Exa enrichment: {error}", file=sys.stderr)
        return 1
    print("Exa enrichment выполнен.")
    print(f"Документы: {paths.documents}")
    print(f"Покрытие: {paths.coverage}")
    print(f"Manifest: {paths.manifest}")
    return 0


def _run_analysis_enrichment_merge(
    analysis_plan: Path,
    scientific_result: Path,
    exa_plan: Path,
    exa_result: Path,
    output: Path,
) -> int:
    try:
        paths = export_combined_enrichment(
            analysis_plan_dir=analysis_plan,
            scientific_result_dir=scientific_result,
            exa_plan_dir=exa_plan,
            exa_result_dir=exa_result,
            output_dir=output,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Не удалось объединить enrichment: {error}", file=sys.stderr)
        return 1
    print("OpenAlex и source-typed Exa enrichment объединены.")
    print(f"Документы: {paths.documents}")
    print(f"Исключённые Exa-документы: {paths.excluded_documents}")
    print(f"Manifest: {paths.manifest}")
    return 0


def _run_evaluation_identity_review(
    positive_plan: Path,
    negative_plan: Path,
    decisions: Path,
    output: Path,
) -> int:
    try:
        paths = export_identity_review(
            positive_plan_dir=positive_plan,
            negative_plan_dir=negative_plan,
            decisions_path=decisions,
            output_dir=output,
        )
    except (OSError, ValueError) as error:
        print(f"Не удалось собрать identity review: {error}", file=sys.stderr)
        return 1
    print("Identity review успешно собран.")
    print(f"Группы: {paths.identities}")
    print(f"Решения: {paths.pair_decisions}")
    print(f"Manifest: {paths.manifest}")
    return 0


def _run_evaluation_gate_noise(
    selection: Path, discovery_root: Path, output: Path
) -> int:
    try:
        paths = export_gate_noise_evaluation(
            selection_path=selection,
            discovery_root=discovery_root,
            output_dir=output,
        )
    except (OSError, ValueError) as error:
        print(f"Не удалось оценить Candidate Gate: {error}", file=sys.stderr)
        return 1
    print("Оценка Candidate Gate успешно собрана.")
    print(f"Строки: {paths.rows}")
    print(f"Отчёт: {paths.report}")
    print(f"Manifest: {paths.manifest}")
    return 0


def _run_evaluation_model_report(features: Path, output: Path) -> int:
    try:
        paths = export_model_report(feature_dir=features, output_dir=output)
    except (OSError, ValueError) as error:
        print(f"Не удалось построить отчёт модели: {error}", file=sys.stderr)
        return 1
    print("Диагностический отчёт модели успешно построен.")
    print(f"OOF predictions: {paths.predictions}")
    print(f"Ошибки: {paths.errors}")
    print(f"Метрики: {paths.report}")
    print(f"Модель: {paths.model}")
    return 0


def _run_evaluation_temporal_count_plan(
    positive_plan: Path, negative_plan: Path, output: Path
) -> int:
    try:
        paths = export_temporal_count_plan(
            positive_plan_dir=positive_plan,
            negative_plan_dir=negative_plan,
            output_dir=output,
        )
    except (OSError, ValueError) as error:
        print(f"Не удалось собрать temporal count plan: {error}", file=sys.stderr)
        return 1
    print("Temporal count plan успешно собран.")
    print(f"План: {paths.plan}")
    print(f"Manifest: {paths.manifest}")
    return 0


def _run_analysis_temporal_count_plan(analysis_plan: Path, output: Path) -> int:
    try:
        paths = export_analysis_temporal_count_plan(
            analysis_plan_dir=analysis_plan, output_dir=output
        )
    except (OSError, ValueError) as error:
        print(f"Не удалось собрать query-specific temporal count plan: {error}", file=sys.stderr)
        return 1
    print("Query-specific temporal count plan успешно собран.")
    print(f"План: {paths.plan}")
    print(f"Manifest: {paths.manifest}")
    return 0


def _run_evaluation_temporal_count_run(
    plan: Path,
    work: Path,
    output: Path,
    env_file: Path,
    max_new_tasks: int | None,
) -> int:
    try:
        paths = run_temporal_counts(
            plan_dir=plan,
            work_dir=work,
            output_dir=output,
            environment=_runtime_environment(env_file),
            max_new_tasks=max_new_tasks,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Не удалось выполнить temporal counts: {error}", file=sys.stderr)
        return 1
    print("Temporal counts выполнены.")
    print(f"Counts: {paths['count_results.jsonl']}")
    print(f"Признаки: {paths['candidate_temporal_features.jsonl']}")
    print(f"Manifest: {paths['manifest.json']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "dataset-build":
        return _run_dataset_build(arguments.input, arguments.output)
    if arguments.command == "query-resolve":
        return _run_query_resolve(arguments.query, arguments.env_file)
    if arguments.command == "discovery-run":
        return _run_discovery_run(
            arguments.query,
            arguments.analysis_id,
            arguments.analysis_scope_key,
            arguments.domain,
            arguments.cutoff_date,
            arguments.published_from,
            arguments.env_file,
            arguments.output_root,
        )
    if arguments.command == "labeling-export":
        return _run_labeling_export(
            arguments.runs,
            arguments.template,
            arguments.output,
            arguments.domain_map,
            arguments.candidate_selection,
            arguments.noise_selection,
        )
    if arguments.command == "labeling-finalize":
        return _run_labeling_finalize(
            arguments.bundle,
            arguments.workbook,
            arguments.enrichment_result,
            arguments.output,
        )
    if arguments.command == "labeling-enrichment-plan":
        return _run_labeling_enrichment_plan(
            arguments.bundle, arguments.output, arguments.search_terms
        )
    if arguments.command == "labeling-target-enrichment-plan":
        return _run_labeling_target_enrichment_plan(arguments.candidates, arguments.output)
    if arguments.command == "labeling-enrichment-run":
        return _run_labeling_enrichment_run(
            arguments.plan, arguments.work, arguments.output, arguments.env_file
        )
    if arguments.command == "labeling-target-gate-run":
        return _run_labeling_target_gate(
            arguments.plan,
            arguments.result,
            arguments.output,
            arguments.env_file,
        )
    if arguments.command == "labeling-relevance-plan":
        return _run_labeling_relevance_plan(arguments.plan, arguments.result, arguments.output)
    if arguments.command == "labeling-media-fetch-run":
        return _run_labeling_media_fetch_run(
            arguments.relevance, arguments.result, arguments.work, arguments.output
        )
    if arguments.command == "labeling-evidence-input-plan":
        return _run_labeling_evidence_input_plan(
            arguments.plan,
            arguments.result,
            arguments.relevance,
            arguments.media,
            arguments.output,
        )
    if arguments.command == "labeling-evidence-llm-plan":
        return _run_labeling_evidence_llm_plan(
            arguments.input,
            arguments.output,
        )
    if arguments.command == "labeling-evidence-llm-run":
        return _run_labeling_evidence_llm_run(
            arguments.plan,
            arguments.work,
            arguments.output,
            arguments.env_file,
            arguments.max_new_tasks,
            arguments.candidate_ids,
        )
    if arguments.command == "labeling-evidence-llm-merge":
        return _run_labeling_evidence_llm_merge(
            arguments.primary, arguments.retry, arguments.output
        )
    if arguments.command == "labeling-rubric-audit":
        return _run_labeling_rubric_audit(
            arguments.plan,
            arguments.enrichment_result,
            arguments.evidence_input,
            arguments.evidence_result,
            arguments.relevance,
            arguments.output,
        )
    if arguments.command == "labeling-corpus-readiness":
        return _run_labeling_corpus_readiness(
            arguments.adjudication,
            arguments.review_queue,
            arguments.output,
        )
    if arguments.command == "labeling-maturity-evidence-input":
        return _run_labeling_maturity_evidence_input(
            arguments.plan, arguments.audit, arguments.output
        )
    if arguments.command == "labeling-hype-evidence-input":
        return _run_labeling_hype_evidence_input(
            arguments.plan,
            arguments.gate,
            arguments.evidence_input,
            arguments.media_result,
            arguments.output,
        )
    if arguments.command == "evaluation-identity-review":
        return _run_evaluation_identity_review(
            arguments.positive_plan,
            arguments.negative_plan,
            arguments.decisions,
            arguments.output,
        )
    if arguments.command == "analysis-enrichment-plan":
        return _run_analysis_enrichment_plan(arguments.run, arguments.output)
    if arguments.command == "analysis-exa-enrichment-plan":
        return _run_analysis_exa_enrichment_plan(
            arguments.analysis_plan, arguments.output
        )
    if arguments.command == "analysis-exa-enrichment-run":
        return _run_analysis_exa_enrichment(
            arguments.plan,
            arguments.work,
            arguments.output,
            arguments.env_file,
            arguments.max_new_requests,
            arguments.concurrency,
        )
    if arguments.command == "analysis-enrichment-merge":
        return _run_analysis_enrichment_merge(
            arguments.analysis_plan,
            arguments.scientific_result,
            arguments.exa_plan,
            arguments.exa_result,
            arguments.output,
        )
    if arguments.command == "evaluation-gate-noise":
        return _run_evaluation_gate_noise(
            arguments.selection, arguments.discovery_root, arguments.output
        )
    if arguments.command == "evaluation-feature-table":
        return _run_evaluation_feature_table(
            arguments.positive,
            arguments.positive_enrichment,
            arguments.negative_plan,
            arguments.negative_enrichment,
            arguments.adjudication,
            arguments.temporal_counts,
            arguments.identity_review,
            arguments.output,
        )
    if arguments.command == "analysis-feature-table":
        return _run_analysis_feature_table(
            arguments.analysis_plan,
            arguments.enrichment_result,
            arguments.temporal_counts,
            arguments.output,
        )
    if arguments.command == "analysis-inference":
        return _run_analysis_inference(
            arguments.features, arguments.model, arguments.output
        )
    if arguments.command == "analysis-evidence-shortlist":
        return _run_analysis_shortlist(
            arguments.inference, arguments.limit, arguments.output
        )
    if arguments.command == "evaluation-model-report":
        return _run_evaluation_model_report(arguments.features, arguments.output)
    if arguments.command == "evaluation-temporal-count-plan":
        return _run_evaluation_temporal_count_plan(
            arguments.positive_plan, arguments.negative_plan, arguments.output
        )
    if arguments.command == "analysis-temporal-count-plan":
        return _run_analysis_temporal_count_plan(
            arguments.analysis_plan, arguments.output
        )
    if arguments.command == "evaluation-temporal-count-run":
        return _run_evaluation_temporal_count_run(
            arguments.plan,
            arguments.work,
            arguments.output,
            arguments.env_file,
            arguments.max_new_tasks,
        )
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
