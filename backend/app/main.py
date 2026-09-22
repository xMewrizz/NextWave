import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from nextwave.analysis import AnalysisMetadata
from nextwave.contracts import CandidateStatus

from . import database, pipeline
from .models import Analysis, AnalysisRequest, AnalysisSummary, Coverage, Stage, Trend


@asynccontextmanager
async def lifespan(_: FastAPI):
    database.init_database()
    database.mark_interrupted_analyses()
    try:
        yield
    finally:
        for task in tuple(_tasks):
            task.cancel()
        await asyncio.gather(*_tasks, return_exceptions=True)


app = FastAPI(title="Радар зарождающихся технологий", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_tasks: set[asyncio.Task] = set()


async def _execute(analysis_id: str) -> None:
    analysis = database.get_analysis(analysis_id)
    if analysis is None:
        return
    analysis = analysis.model_copy(update={"started_at": datetime.now(UTC)})

    def on_stage(stage: Stage, progress: float) -> None:
        nonlocal analysis
        analysis = analysis.model_copy(
            update={"status": "running", "stage": stage.label, "progress": progress}
        )
        database.save_analysis(analysis)

    def on_metadata(metadata: AnalysisMetadata) -> None:
        nonlocal analysis
        analysis = analysis.model_copy(update=metadata.model_dump())
        database.save_analysis(analysis)

    try:
        trends = await pipeline.run(analysis.query, on_stage, on_metadata)
    except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001 — safe error boundary
        if isinstance(exc, pipeline.IntegrationUnavailable):
            code = "model_unavailable"
            notice = "Интеграция модели недоступна. Анализатор ещё не подключён или не готов к запуску."
        elif isinstance(exc, pipeline.InvalidAssessment):
            code = "invalid_model_result"
            notice = "Модель вернула результат, не соответствующий контракту анализа."
        elif isinstance(exc, asyncio.CancelledError):
            code = "interrupted"
            notice = "Анализ прерван остановкой сервера. Запустите его повторно."
        else:
            code = "analysis_failed"
            notice = "Не удалось выполнить анализ. Повторите запрос позже."
        # Exception text can include provider credentials or raw data; never persist it.
        analysis = analysis.model_copy(update={"status": "error", "error_code": code, "notice": notice})
    else:
        analysis = analysis.model_copy(
            update={
                "trends": trends,
                "status": "done" if trends else "empty",
                "notice": _notice(trends),
            }
        )
    analysis = analysis.model_copy(update={"progress": 1.0, "stage": None, "finished_at": datetime.now(UTC)})
    database.save_analysis(analysis)


def _notice(trends: list[Trend]) -> str | None:
    if not trends:
        return "Анализ завершён: анализатор не вернул кандидатов по этому запросу."
    main = [trend for trend in trends if trend.status is CandidateStatus.MAIN]
    if len(main) < pipeline.TOP_N:
        return (
            f"В основной список прошли {len(main)} кандидатов из {pipeline.TOP_N}: "
            "показано фактическое число. Причины решений доступны в карточках кандидатов."
        )
    if len(main) > pipeline.TOP_N:
        return (
            f"Показаны первые {pipeline.TOP_N} из {len(main)} кандидатов main. "
            "Все финальные оценки сохранены и доступны через API."
        )
    return None


@app.get("/api/coverage")
def get_coverage() -> Coverage:
    return pipeline.coverage()


@app.get("/api/stages")
def get_stages() -> list[Stage]:
    return pipeline.STAGES


@app.post("/api/analyses", status_code=201)
async def create_analysis(body: AnalysisRequest) -> Analysis:
    query = body.query.strip()
    if not query:
        raise HTTPException(422, "Пустой запрос не запускает анализ.")

    analysis = Analysis(
        id=uuid.uuid4().hex[:12],
        query=query,
        status="pending",
        created_at=datetime.now(UTC),
        corpus_version="unavailable",
        method_version="unavailable",
    )
    database.save_analysis(analysis)
    task = asyncio.create_task(_execute(analysis.id))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return analysis


@app.get("/api/analyses")
def list_analyses() -> list[AnalysisSummary]:
    return database.list_analyses()


@app.get("/api/analyses/{analysis_id}")
def get_analysis(analysis_id: str) -> Analysis:
    analysis = database.get_analysis(analysis_id)
    if analysis is None:
        raise HTTPException(404, "Анализ не найден.")
    return analysis


@app.get("/api/analyses/{analysis_id}/trends/{trend_id}")
def get_trend(analysis_id: str, trend_id: str) -> Trend:
    analysis = get_analysis(analysis_id)
    trend = next((item for item in analysis.trends if item.candidate_id == trend_id), None)
    if trend is None:
        raise HTTPException(404, "Тренд не найден в этой выдаче.")
    return trend
