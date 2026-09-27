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
from .labeling.enrichment_plan import (
    LABELING_ENRICHMENT_PLAN_VERSION,
    export_enrichment_plan,
)
from .labeling.enrichment_run import ENRICHMENT_RESULT_VERSION, run_enrichment
from .labeling.evidence_input_plan import (
    LABELING_EVIDENCE_INPUT_PLAN_VERSION,
    export_evidence_input_plan,
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
from .labeling.media_fetch_run import (
    LABELING_MEDIA_FETCH_RESULT_VERSION,
    run_media_fetch,
)
from .labeling.relevance_plan import (
    LABELING_RELEVANCE_PLAN_VERSION,
    export_relevance_plan,
)


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
            raise ValueError(
                f"invalid --domain-map entry {entry!r}; expected RUN_ID=Domain"
            )
        mapping[run_id.strip()] = domain.strip()
    return mapping


def _run_labeling_export(
    runs: list[str], template: Path, output: Path, domain_map: list[str]
) -> int:
    try:
        paths = export_labeling_bundle(
            run_dirs=tuple(runs),
            template_path=template,
            output_dir=output,
            run_domains=_parse_domain_map(domain_map),
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


def _run_labeling_enrichment_plan(
    bundle: Path, output: Path, search_terms: Path | None
) -> int:
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


def _run_labeling_enrichment_run(
    plan: Path, work: Path, output: Path, env_file: Path
) -> int:
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


def _run_labeling_media_fetch_run(
    relevance: Path, result: Path, work: Path, output: Path
) -> int:
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
        )
    if arguments.command == "labeling-enrichment-plan":
        return _run_labeling_enrichment_plan(
            arguments.bundle, arguments.output, arguments.search_terms
        )
    if arguments.command == "labeling-enrichment-run":
        return _run_labeling_enrichment_run(
            arguments.plan, arguments.work, arguments.output, arguments.env_file
        )
    if arguments.command == "labeling-relevance-plan":
        return _run_labeling_relevance_plan(
            arguments.plan, arguments.result, arguments.output
        )
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
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
