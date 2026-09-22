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

    print("Поиск завершён, запуск сохранён.")
    print(f"Каталог запуска: {run_dir}")
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
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
