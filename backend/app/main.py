import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from nextwave.contracts import CandidateStatus

from . import database, pipeline
from .models import Analysis, AnalysisRequest, AnalysisSummary, Coverage, Stage, Trend


@asynccontextmanager
async def lifespan(_: FastAPI):
    database.init_database()
    database.mark_interrupted_analyses()
    yield


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

    def on_stage(stage: Stage, progress: float) -> None:
        nonlocal analysis
        analysis = analysis.model_copy(
            update={"status": "running", "stage": stage.label, "progress": progress}
        )
        database.save_analysis(analysis)

    try:
        trends = await pipeline.run(analysis.query, on_stage)
    except Exception as exc:  # noqa: BLE001 — состояние ошибки видно пользователю
        analysis = analysis.model_copy(
            update={"status": "error", "notice": f"Анализ прерван: {exc}"}
        )
    else:
        analysis = analysis.model_copy(
            update={
                "trends": trends,
                "status": "done" if trends else "empty",
                "notice": _notice(trends),
            }
        )
    analysis = analysis.model_copy(
        update={"progress": 1.0, "stage": None, "finished_at": datetime.now(UTC)}
    )
    database.save_analysis(analysis)


def _notice(trends: list[Trend]) -> str | None:
    if not trends:
        return (
            "Направление вне покрытия корпуса. Доступные направления: "
            f"{', '.join(pipeline.DIRECTIONS)}. Нерелевантная выдача не подставляется."
        )
    main = [trend for trend in trends if trend.status is CandidateStatus.MAIN]
    if len(main) < pipeline.TOP_N:
        return (
            f"В основной список прошли {len(main)} кандидатов из {pipeline.TOP_N}: "
            "остальные темы не набрали достаточной доказательной базы или истории в корпусе. "
            "Они доступны во вкладках «Наблюдение» и «Отсеяны» с указанием причины."
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
        corpus_version=pipeline.CORPUS_VERSION,
        method_version=pipeline.METHOD_VERSION,
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
