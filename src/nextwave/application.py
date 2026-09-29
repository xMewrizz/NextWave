"""One resumable application service for a query-local NextWave analysis."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from nextwave.discovery import (
    build_discovery_pipeline_from_environment,
    build_discovery_plan,
    build_query_resolver_from_environment,
    load_evidence_llm_settings,
    load_gate_llm_settings,
    load_llm_runtime_settings,
    save_discovery_run,
)
from nextwave.evaluation import (
    export_analysis_evidence_input,
    export_analysis_feature_table,
    export_analysis_inference,
    export_analysis_result,
    export_analysis_shortlist,
    export_analysis_temporal_count_plan,
    export_combined_enrichment,
    export_exa_enrichment_plan,
    run_exa_enrichment,
    run_temporal_counts,
)
from nextwave.labeling import (
    LABELING_CUTOFF_DATE,
    export_analysis_enrichment_plan,
    export_evidence_llm_plan,
    run_enrichment,
    run_evidence_llm,
)

ANALYSIS_APPLICATION_VERSION = "analysis-application-v1"
ANALYSIS_APPLICATION_MANIFEST = "analysis_application.json"
ProgressCallback = Callable[[str, float, str], None]

_STEPS = (
    "discovery",
    "enrichment_plan",
    "scientific_enrichment",
    "exa_plan",
    "exa_enrichment",
    "combined_enrichment",
    "temporal_plan",
    "temporal_counts",
    "features",
    "inference",
    "shortlist",
    "evidence_input",
    "evidence_plan",
    "evidence_run",
    "result",
)
_ANALYSIS_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True, slots=True)
class AnalysisApplicationPaths:
    workspace: Path
    manifest: Path
    result_dir: Path
    result: Path


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _atomic_json(path: Path, value: object) -> None:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    staging.write_bytes(raw)
    os.replace(staging, path)


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _digest(raw: bytes) -> dict[str, Any]:
    return {"size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _validate_bundle(path: Path) -> None:
    manifest = _read_object(path / "manifest.json", f"artifact manifest {path.name}")
    if not isinstance(manifest.get("schema_version"), str):
        raise ValueError(f"artifact {path.name} misses schema_version")
    outputs = manifest.get("outputs")
    if outputs is None:
        return
    entries: list[tuple[object, object]]
    if isinstance(outputs, dict):
        entries = list(outputs.items())
    elif isinstance(outputs, list):
        entries = [
            (item.get("filename"), item) if isinstance(item, dict) else (None, item)
            for item in outputs
        ]
    else:
        raise ValueError(f"artifact {path.name} outputs must be an object or list")
    seen: set[str] = set()
    for filename, expected in entries:
        if not isinstance(filename, str) or not filename or not isinstance(expected, dict):
            raise ValueError(f"artifact {path.name} has invalid output entry")
        if filename in seen:
            raise ValueError(f"artifact {path.name} repeats output {filename}")
        seen.add(filename)
        output_path = (path / filename).resolve()
        if path.resolve() not in output_path.parents:
            raise ValueError(f"artifact {path.name} output escapes its directory")
        try:
            raw = output_path.read_bytes()
        except OSError as error:
            raise ValueError(f"artifact {path.name} misses {filename}") from error
        actual = _digest(raw)
        if (
            expected.get("size_bytes") != actual["size_bytes"]
            or expected.get("sha256") != actual["sha256"]
        ):
            raise ValueError(f"artifact {path.name} output {filename} is corrupt")


def _relative(workspace: Path, path: Path) -> str:
    resolved_workspace = workspace.resolve()
    resolved = path.resolve()
    if resolved_workspace not in resolved.parents and resolved != resolved_workspace:
        raise ValueError("analysis artifact escaped the job workspace")
    return resolved.relative_to(resolved_workspace).as_posix()


def _artifact_path(workspace: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"analysis checkpoint misses {label}")
    path = (workspace / value).resolve()
    root = workspace.resolve()
    if root not in path.parents:
        raise ValueError(f"analysis checkpoint {label} escapes workspace")
    return path


class AnalysisApplication:
    """Execute every product stage once and resume from immutable checkpoints."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        model_dir: str | Path,
        environment: Mapping[str, str],
        cutoff_date: date = LABELING_CUTOFF_DATE,
        progress: ProgressCallback | None = None,
    ) -> None:
        if cutoff_date != LABELING_CUTOFF_DATE:
            raise ValueError(
                f"live analysis cutoff must be {LABELING_CUTOFF_DATE.isoformat()}"
            )
        if not str(environment.get("NEXTWAVE_EXA_API_KEY", "")).strip():
            raise ValueError("NEXTWAVE_EXA_API_KEY is required for live analysis")
        load_llm_runtime_settings(environment)
        load_gate_llm_settings(environment)
        load_evidence_llm_settings(environment)
        self.workspace = Path(workspace)
        self.model_dir = Path(model_dir)
        self.environment = dict(environment)
        self.cutoff_date = cutoff_date
        self.progress = progress or (lambda _stage, _value, _message: None)
        self.manifest_path = self.workspace / ANALYSIS_APPLICATION_MANIFEST

    def run(self, *, query: str, analysis_id: str) -> AnalysisApplicationPaths:
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise ValueError("analysis query must not be blank")
        if not _ANALYSIS_ID.fullmatch(analysis_id) or ".." in analysis_id:
            raise ValueError("analysis_id is invalid")
        state = self._load_state(normalized_query, analysis_id)
        artifacts = state["artifacts"]
        completed = state["completed_steps"]

        if "result" in completed:
            result_dir = _artifact_path(self.workspace, artifacts.get("result"), "result")
            _validate_bundle(result_dir)
            return self._paths(result_dir)

        discovery_run = self._discovery(normalized_query, analysis_id, state)
        analysis_plan = self._publish(
            state,
            "enrichment_plan",
            "enrichment",
            0.29,
            "Подготовка enrichment-плана",
            lambda output: export_analysis_enrichment_plan(
                run_dir=discovery_run, output_dir=output
            ),
        )
        scientific = self._publish(
            state,
            "scientific_enrichment",
            "enrichment",
            0.39,
            "Научное обогащение OpenAlex",
            lambda output: run_enrichment(
                plan_dir=analysis_plan,
                work_dir=self.workspace / "work" / "scientific_enrichment",
                output_dir=output,
                environment=self.environment,
                connectors=("openalex",),
            ),
        )
        exa_plan = self._publish(
            state,
            "exa_plan",
            "enrichment",
            0.41,
            "Подготовка отраслевого поиска",
            lambda output: export_exa_enrichment_plan(
                analysis_plan_dir=analysis_plan, output_dir=output
            ),
        )
        exa_result = self._publish(
            state,
            "exa_enrichment",
            "enrichment",
            0.50,
            "Отраслевое обогащение Exa",
            lambda output: run_exa_enrichment(
                plan_dir=exa_plan,
                work_dir=self.workspace / "work" / "exa_enrichment",
                output_dir=output,
                environment=self.environment,
            ),
        )
        combined = self._publish(
            state,
            "combined_enrichment",
            "enrichment",
            0.53,
            "Объединение научных и отраслевых источников",
            lambda output: export_combined_enrichment(
                analysis_plan_dir=analysis_plan,
                scientific_result_dir=scientific,
                exa_plan_dir=exa_plan,
                exa_result_dir=exa_result,
                output_dir=output,
            ),
        )
        temporal_plan = self._publish(
            state,
            "temporal_plan",
            "enrichment",
            0.55,
            "Подготовка временных признаков",
            lambda output: export_analysis_temporal_count_plan(
                analysis_plan_dir=analysis_plan, output_dir=output
            ),
        )
        temporal_counts = self._publish(
            state,
            "temporal_counts",
            "enrichment",
            0.65,
            "Расчёт временной динамики OpenAlex",
            lambda output: run_temporal_counts(
                plan_dir=temporal_plan,
                work_dir=self.workspace / "work" / "temporal_counts",
                output_dir=output,
                environment=self.environment,
            ),
        )
        features = self._publish(
            state,
            "features",
            "model",
            0.69,
            "Построение признаков",
            lambda output: export_analysis_feature_table(
                analysis_plan_dir=analysis_plan,
                enrichment_result_dir=combined,
                temporal_count_dir=temporal_counts,
                output_dir=output,
            ),
        )
        inference = self._publish(
            state,
            "inference",
            "model",
            0.73,
            "Расчёт model score и факторов",
            lambda output: export_analysis_inference(
                feature_dir=features,
                model_dir=self.model_dir,
                output_dir=output,
            ),
        )
        candidate_count = self._candidate_count(inference)
        shortlist = self._publish(
            state,
            "shortlist",
            "evidence_duel",
            0.75,
            "Evidence-очередь для всех кандидатов",
            lambda output: export_analysis_shortlist(
                inference_dir=inference,
                output_dir=output,
                limit=candidate_count,
            ),
        )
        evidence_input = self._publish(
            state,
            "evidence_input",
            "evidence_duel",
            0.78,
            "Отбор проверяемых документов",
            lambda output: export_analysis_evidence_input(
                analysis_plan_dir=analysis_plan,
                combined_result_dir=combined,
                shortlist_dir=shortlist,
                output_dir=output,
            ),
        )
        evidence_plan = self._publish(
            state,
            "evidence_plan",
            "evidence_duel",
            0.80,
            "Подготовка Evidence-промптов",
            lambda output: export_evidence_llm_plan(
                input_dir=evidence_input, output_dir=output
            ),
        )
        evidence_result = self._publish(
            state,
            "evidence_run",
            "evidence_duel",
            0.94,
            "Извлечение Evidence claims",
            lambda output: run_evidence_llm(
                plan_dir=evidence_plan,
                work_dir=self.workspace / "work" / "evidence",
                output_dir=output,
                environment=self.environment,
            ),
        )
        result_dir = self._publish(
            state,
            "result",
            "result",
            1.0,
            "Формирование единого результата",
            lambda output: export_analysis_result(
                analysis_plan_dir=analysis_plan,
                combined_result_dir=combined,
                feature_dir=features,
                inference_dir=inference,
                evidence_result_dirs=(evidence_result,),
                output_dir=output,
            ),
        )
        return self._paths(result_dir)

    def _load_state(self, query: str, analysis_id: str) -> dict[str, Any]:
        if self.manifest_path.exists():
            state = _read_object(self.manifest_path, "analysis application checkpoint")
            if state.get("schema_version") != ANALYSIS_APPLICATION_VERSION:
                raise ValueError("analysis application checkpoint version is unsupported")
            if state.get("query") != query or state.get("analysis_id") != analysis_id:
                raise ValueError("analysis application checkpoint belongs to another job")
            if state.get("cutoff_date") != self.cutoff_date.isoformat():
                raise ValueError("analysis application checkpoint uses another cutoff")
            if state.get("model") != self._model_digest():
                raise ValueError("analysis application checkpoint uses another model")
            completed = state.get("completed_steps")
            artifacts = state.get("artifacts")
            if not isinstance(completed, list) or not isinstance(artifacts, dict):
                raise ValueError("analysis application checkpoint is invalid")
            if any(step not in _STEPS for step in completed) or len(completed) != len(
                set(completed)
            ):
                raise ValueError("analysis application checkpoint has invalid steps")
            expected_prefix = list(_STEPS[: len(completed)])
            if completed != expected_prefix:
                raise ValueError("analysis application checkpoint steps are not sequential")
            for step in completed:
                path = _artifact_path(self.workspace, artifacts.get(step), step)
                _validate_bundle(path)
            return state
        self.workspace.mkdir(parents=True, exist_ok=True)
        state = {
            "schema_version": ANALYSIS_APPLICATION_VERSION,
            "analysis_id": analysis_id,
            "query": query,
            "cutoff_date": self.cutoff_date.isoformat(),
            "model": self._model_digest(),
            "completed_steps": [],
            "artifacts": {},
        }
        _atomic_json(self.manifest_path, state)
        return state

    def _model_digest(self) -> dict[str, Any]:
        try:
            raw = (self.model_dir / "model.json").read_bytes()
        except OSError as error:
            raise ValueError("cannot read frozen model artifact") from error
        return _digest(raw)

    def _discovery(
        self, query: str, analysis_id: str, state: dict[str, Any]
    ) -> Path:
        if "discovery" in state["completed_steps"]:
            return _artifact_path(
                self.workspace, state["artifacts"].get("discovery"), "discovery"
            )
        recovered = self._recover_discovery(query, analysis_id)
        if recovered is not None:
            self._complete_step(state, "discovery", recovered)
            self.progress("candidate_gate", 0.27, "Сохранённый Candidate Gate восстановлен")
            return recovered
        self.progress("source_search", 0.02, "Интерпретация запроса")
        resolution = build_query_resolver_from_environment(self.environment).resolve(query)
        plan = build_discovery_plan(
            analysis_id=analysis_id,
            scope=resolution.scope,
            published_from=self.cutoff_date - timedelta(days=365),
            cutoff_date=self.cutoff_date,
        )

        def pipeline_progress(message: str) -> None:
            stage = "candidate_gate" if "[discovery] gate:" in message else "source_search"
            value = 0.24 if stage == "candidate_gate" else 0.12
            self.progress(stage, value, message)

        result = build_discovery_pipeline_from_environment(
            self.environment,
            snapshot_root=self.workspace / "snapshots" / "discovery",
        ).execute(plan, progress=pipeline_progress)
        run_dir = save_discovery_run(
            plan,
            result,
            analysis_scope_key=resolution.scope.scope_id,
            domain=resolution.scope.raw_query,
            output_root=self.workspace / "artifacts" / "discovery",
        )
        self._complete_step(state, "discovery", run_dir)
        self.progress("candidate_gate", 0.27, "Candidate Gate завершён")
        return run_dir

    def _recover_discovery(self, query: str, analysis_id: str) -> Path | None:
        root = self.workspace / "artifacts" / "discovery"
        if not root.exists():
            return None
        if not root.is_dir():
            raise ValueError("analysis discovery artifact root is not a directory")
        candidates = sorted(
            path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")
        )
        if not candidates:
            return None
        if len(candidates) != 1:
            raise ValueError("analysis discovery recovery found multiple saved runs")
        run_dir = candidates[0]
        _validate_bundle(run_dir)
        manifest = _read_object(run_dir / "manifest.json", "saved discovery manifest")
        plan = _read_object(run_dir / "plan.json", "saved discovery plan")
        plan_query = plan.get("query")
        if not isinstance(plan_query, dict):
            raise ValueError("saved discovery plan misses query")
        raw_query = plan_query.get("raw_query")
        if (
            manifest.get("analysis_id") != analysis_id
            or plan.get("analysis_id") != analysis_id
            or not isinstance(raw_query, str)
            or " ".join(raw_query.split()) != query
            or manifest.get("raw_query") != raw_query
            or manifest.get("cutoff_date") != self.cutoff_date.isoformat()
            or plan_query.get("cutoff_date") != self.cutoff_date.isoformat()
        ):
            raise ValueError("saved discovery run belongs to another analysis")
        return run_dir

    def _publish(
        self,
        state: dict[str, Any],
        step: str,
        stage: str,
        progress: float,
        message: str,
        operation: Callable[[Path], object],
    ) -> Path:
        if step in state["completed_steps"]:
            return _artifact_path(self.workspace, state["artifacts"].get(step), step)
        output = self.workspace / "artifacts" / step
        if output.exists():
            _validate_bundle(output)
        else:
            self.progress(stage, max(0.0, progress - 0.02), message)
            operation(output)
            _validate_bundle(output)
        self._complete_step(state, step, output)
        self.progress(stage, progress, message)
        return output

    def _complete_step(self, state: dict[str, Any], step: str, path: Path) -> None:
        expected = _STEPS[len(state["completed_steps"])]
        if step != expected:
            raise ValueError(f"analysis step {step} cannot follow {expected}")
        state["artifacts"][step] = _relative(self.workspace, path)
        state["completed_steps"].append(step)
        _atomic_json(self.manifest_path, state)

    def _candidate_count(self, inference: Path) -> int:
        manifest = _read_object(inference / "manifest.json", "analysis inference manifest")
        count = manifest.get("candidate_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("analysis inference has invalid candidate_count")
        return count

    def _paths(self, result_dir: Path) -> AnalysisApplicationPaths:
        result = result_dir / "result.json"
        if not result.is_file():
            raise ValueError("analysis result bundle misses result.json")
        return AnalysisApplicationPaths(
            workspace=self.workspace,
            manifest=self.manifest_path,
            result_dir=result_dir,
            result=result,
        )
