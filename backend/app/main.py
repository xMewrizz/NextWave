import asyncio
import json
import os
from datetime import date
from pathlib import Path

import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from nextwave.application import AnalysisApplication

from . import pipeline
from .job_store import configured_analysis_job_store
from .models import (
    AnalysisJob,
    AnalysisRequest,
    AnalysisStageState,
    Coverage,
    Stage,
)
from .result_store import configured_result_dir, load_result_bundle

app = FastAPI(title="Радар зарождающихся технологий", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_tasks: dict[str, asyncio.Task[None]] = {}
_RESULT_STAGES = (
    ("source_search", "Поиск источников"),
    ("candidate_gate", "Отбор технологических кандидатов"),
    ("enrichment", "Обогащение кандидатов"),
    ("model", "Расчёт оценки модели"),
    ("evidence_duel", "Evidence Duel"),
    ("result", "Формирование результата"),
)
_STAGE_LABELS = dict(_RESULT_STAGES)


def _require_synthetic_demo() -> None:
    if os.getenv("NEXTWAVE_ENABLE_SYNTHETIC_DEMO") != "1":
        raise HTTPException(404, "Синтетический API отключён в release mode.")


def _store():
    return configured_analysis_job_store()


def _normalized_query(value: str) -> str:
    return " ".join(value.split()).casefold()


async def _execute(job_id: str) -> None:
    store = _store()
    job = store.get(job_id)
    if job is None:
        return
    job = store.mark_running(job)
    try:
        if job.mode == "live":
            result = await asyncio.to_thread(_run_live_analysis, job.id)
            stages = [
                AnalysisStageState(key=key, label=label, status="complete")
                for key, label in _RESULT_STAGES
            ]
        else:
            result = load_result_bundle()
            result_query = ((result.get("query") or {}).get("text"))
            if not isinstance(result_query, str) or _normalized_query(
                job.query
            ) != _normalized_query(result_query):
                raise ValueError(
                    "для нового запроса live-runner отключён; сохранённый результат "
                    f"относится к запросу {result_query!r}"
                )
            stages = [
                AnalysisStageState(key=key, label=label, status="reused")
                for key, label in _RESULT_STAGES
            ]
        latest = store.get(job.id)
        if latest is None:
            raise ValueError("analysis job disappeared during execution")
        store.complete(latest, result, stages)
    except (OSError, ValueError) as error:
        store.fail(store.get(job.id) or job, str(error))
    except Exception as error:  # noqa: BLE001 - background job must become terminal
        store.fail(
            store.get(job.id) or job,
            f"analysis runner failed ({type(error).__name__})",
        )


def _run_live_analysis(job_id: str) -> dict:
    store = _store()
    job = store.get(job_id)
    if job is None:
        raise ValueError("analysis job does not exist")
    cutoff_text = os.getenv("NEXTWAVE_ANALYSIS_CUTOFF", "2026-09-15")
    try:
        cutoff = date.fromisoformat(cutoff_text)
    except ValueError:
        raise ValueError("NEXTWAVE_ANALYSIS_CUTOFF must use YYYY-MM-DD") from None
    workspace_root = Path(
        os.getenv("NEXTWAVE_ANALYSIS_WORK_DIR", "runtime/analysis-jobs")
    )
    model_dir = Path(
        os.getenv(
            "NEXTWAVE_MODEL_DIR",
            "data/development/model-report-development-v6",
        )
    )

    def progress(stage: str, value: float, message: str) -> None:
        current = store.get(job_id)
        if current is None:
            raise ValueError("analysis job disappeared during execution")
        store.update_progress(
            current,
            stage=stage,
            stage_label=message.strip() or _STAGE_LABELS[stage],
            progress=value,
        )

    paths = AnalysisApplication(
        workspace=workspace_root / job_id,
        model_dir=model_dir,
        environment=os.environ,
        cutoff_date=cutoff,
        progress=progress,
    ).run(query=job.query, analysis_id=f"analysis-{job.id}")
    try:
        value = json.loads(paths.result.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("cannot read live analysis result") from error
    if not isinstance(value, dict) or value.get("schema_version") != "analysis-response-v1":
        raise ValueError("live analysis result schema is unsupported")
    return value


def _schedule(job: AnalysisJob) -> None:
    current = _tasks.get(job.id)
    if current is not None and not current.done():
        return
    task = asyncio.create_task(_execute(job.id))
    _tasks[job.id] = task
    task.add_done_callback(lambda _task, job_id=job.id: _tasks.pop(job_id, None))


@app.get("/api/coverage")
def get_coverage() -> Coverage:
    _require_synthetic_demo()
    return pipeline.coverage()


@app.get("/api/result/current")
def get_current_result() -> dict:
    """Return the checked immutable CLI result without recomputing model policy."""

    try:
        return load_result_bundle()
    except ValueError as error:
        raise HTTPException(
            503,
            f"Итоговый результат недоступен ({configured_result_dir()}): {error}",
        ) from error


@app.get("/api/health")
def get_health() -> dict[str, str]:
    try:
        _store().healthcheck()
        load_result_bundle()
    except (OSError, ValueError, psycopg.Error) as error:
        raise HTTPException(503, "Сервис анализа не готов.") from error
    return {"status": "ok"}


@app.get("/api/stages")
def get_stages() -> list[Stage]:
    _require_synthetic_demo()
    return pipeline.STAGES


@app.post("/api/analyses", status_code=201)
async def create_analysis(body: AnalysisRequest) -> AnalysisJob:
    query = body.query.strip()
    if not query:
        raise HTTPException(422, "Пустой запрос не запускает анализ.")
    stages = [
        AnalysisStageState(key=key, label=label, status="pending")
        for key, label in _RESULT_STAGES
    ]
    mode = "live" if os.getenv("NEXTWAVE_ANALYSIS_MODE") == "live" else "cached_snapshot"
    job = _store().create(query, stages, mode=mode)
    _schedule(job)
    return job


@app.get("/api/analyses")
def list_analyses() -> list[AnalysisJob]:
    return _store().list()


@app.get("/api/analyses/{analysis_id}")
async def get_analysis(analysis_id: str) -> AnalysisJob:
    try:
        job = _store().get(analysis_id)
    except ValueError as error:
        raise HTTPException(500, str(error)) from error
    if job is None:
        raise HTTPException(404, "Анализ не найден.")
    if job.status in {"pending", "running"}:
        _schedule(job)
    return job


@app.post("/api/analyses/{analysis_id}/retry", status_code=202)
async def retry_analysis(analysis_id: str) -> AnalysisJob:
    store = _store()
    try:
        job = store.get(analysis_id)
    except ValueError as error:
        raise HTTPException(500, str(error)) from error
    if job is None:
        raise HTTPException(404, "Анализ не найден.")
    if job.status != "error":
        raise HTTPException(409, "Повторить можно только анализ с ошибкой.")
    retried = store.retry(job)
    _schedule(retried)
    return retried


@app.get("/api/analyses/{analysis_id}/result")
async def get_analysis_result(analysis_id: str) -> dict:
    job = await get_analysis(analysis_id)
    if job.status != "complete":
        raise HTTPException(409, "Результат анализа ещё не готов.")
    try:
        return _store().load_result(job)
    except ValueError as error:
        raise HTTPException(500, str(error)) from error
