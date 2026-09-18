import asyncio
import uuid
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import pipeline
from .models import Analysis, AnalysisRequest, AnalysisSummary, Coverage, Stage, Trend

app = FastAPI(title="Радар зарождающихся технологий", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Демонстрационное хранилище работает в памяти одного процесса.
# Интегрированная версия сохраняет анализы в PostgreSQL.
_analyses: dict[str, Analysis] = {}
_tasks: set[asyncio.Task] = set()


async def _execute(analysis: Analysis) -> None:
    def on_stage(stage: Stage, progress: float) -> None:
        analysis.status = "running"
        analysis.stage = stage.label
        analysis.progress = progress

    try:
        trends = await pipeline.run(analysis.query, on_stage)
    except Exception as exc:  # noqa: BLE001 — состояние ошибки видно пользователю (US-06)
        analysis.status = "error"
        analysis.notice = f"Анализ прерван: {exc}"
    else:
        analysis.trends = trends
        analysis.status = "done" if trends else "empty"
        analysis.notice = _notice(trends)
    analysis.progress = 1.0
    analysis.stage = None
    analysis.finished_at = datetime.now(UTC)


def _notice(trends: list[Trend]) -> str | None:
    if not trends:
        return (
            "Направление вне покрытия корпуса. Доступные направления: "
            f"{', '.join(pipeline.DIRECTIONS)}. Нерелевантная выдача не подставляется."
        )
    main = [t for t in trends if t.bucket == "main"]
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
    _analyses[analysis.id] = analysis
    task = asyncio.create_task(_execute(analysis))
    _tasks.add(task)  # без ссылки задачу может собрать сборщик мусора
    task.add_done_callback(_tasks.discard)
    return analysis


@app.get("/api/analyses")
def list_analyses() -> list[AnalysisSummary]:
    items = sorted(_analyses.values(), key=lambda a: a.created_at, reverse=True)
    return [
        AnalysisSummary(
            id=a.id, query=a.query, status=a.status, created_at=a.created_at, trend_count=len(a.trends)
        )
        for a in items
    ]


@app.get("/api/analyses/{analysis_id}")
def get_analysis(analysis_id: str) -> Analysis:
    analysis = _analyses.get(analysis_id)
    if analysis is None:
        raise HTTPException(404, "Анализ не найден. Возможно, сервер был перезапущен.")
    return analysis


@app.get("/api/analyses/{analysis_id}/trends/{trend_id}")
def get_trend(analysis_id: str, trend_id: str) -> Trend:
    analysis = get_analysis(analysis_id)
    trend = next((t for t in analysis.trends if t.id == trend_id), None)
    if trend is None:
        raise HTTPException(404, "Тренд не найден в этой выдаче.")
    return trend
