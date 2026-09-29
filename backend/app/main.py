import asyncio
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import pipeline
from .job_store import AnalysisJobStore
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
    ("candidate_gate", "Candidate Gate"),
    ("enrichment", "Обогащение кандидатов"),
    ("model", "Расчёт оценки модели"),
    ("evidence_duel", "Evidence Duel"),
    ("result", "Формирование результата"),
)


def _require_synthetic_demo() -> None:
    if os.getenv("NEXTWAVE_ENABLE_SYNTHETIC_DEMO") != "1":
        raise HTTPException(404, "Синтетический API отключён в release mode.")


def _store() -> AnalysisJobStore:
    return AnalysisJobStore()


def _normalized_query(value: str) -> str:
    return " ".join(value.split()).casefold()


async def _execute(job_id: str) -> None:
    store = _store()
    job = store.get(job_id)
    if job is None:
        return
    job = store.mark_running(job)
    try:
        result = load_result_bundle()
        result_query = ((result.get("query") or {}).get("text"))
        if not isinstance(result_query, str) or _normalized_query(job.query) != _normalized_query(
            result_query
        ):
            raise ValueError(
                "для нового запроса live-runner ещё не подключён; сохранённый результат "
                f"относится к запросу {result_query!r}"
            )
        stages = [
            AnalysisStageState(key=key, label=label, status="reused")
            for key, label in _RESULT_STAGES
        ]
        store.complete(job, result, stages)
    except (OSError, ValueError) as error:
        store.fail(job, str(error))
    except Exception as error:  # noqa: BLE001 - background job must become terminal
        store.fail(job, f"analysis runner failed ({type(error).__name__})")


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


@app.get("/api/stages")
def get_stages() -> list[Stage]:
    _require_synthetic_demo()
    return pipeline.STAGES


@app.post("/api/analyses", status_code=201)
async def create_analysis(body: AnalysisRequest) -> AnalysisJob:
    query = body.query.strip()
    if not query:
        raise HTTPException(422, "Пустой запрос не запускает анализ.")
    job = _store().create(query)
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


@app.get("/api/analyses/{analysis_id}/result")
async def get_analysis_result(analysis_id: str) -> dict:
    job = await get_analysis(analysis_id)
    if job.status != "complete":
        raise HTTPException(409, "Результат анализа ещё не готов.")
    try:
        return _store().load_result(job)
    except ValueError as error:
        raise HTTPException(500, str(error)) from error
