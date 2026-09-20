from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from . import __version__
from .datasets import (
    DATASET_VERSION,
    OrganizerArtifactError,
    OrganizerWorkbookError,
    build_organizer_dataset,
)
from .discovery import build_query_resolver_from_environment


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
) -> dict[str, str]:
    values = _read_environment_file(env_file)
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "dataset-build":
        return _run_dataset_build(arguments.input, arguments.output)
    if arguments.command == "query-resolve":
        return _run_query_resolve(arguments.query, arguments.env_file)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
