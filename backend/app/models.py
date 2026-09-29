"""Контракт API. Фронтенд повторяет эти типы в frontend/src/lib/api.ts."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

AnalysisStatus = Literal["pending", "running", "done", "empty", "error"]
# main — ТОП зарождающихся; watchlist — признаки есть, доказательств мало;
# excluded — тема не прошла проверку на новизну или зарождаемость
Bucket = Literal["main", "watchlist", "excluded"]
BUCKETS: tuple[Bucket, ...] = ("main", "watchlist", "excluded")
SourceType = Literal["preprint", "journal", "patent", "vendor", "conference", "report"]
FactorKey = Literal["growth", "novelty", "independence", "evidence"]


class SourceRef(BaseModel):
    title: str
    url: str
    source_type: SourceType
    # None = дата публикации неизвестна; дата загрузки её не подменяет (ARCHITECTURE.md)
    published_at: date | None = None


class TimelinePoint(BaseModel):
    period: str  # YYYY-MM
    documents: int
    share: float = Field(description="доля документов темы в корпусе направления за период")


class ScoreFactor(BaseModel):
    key: FactorKey
    value: float = Field(ge=0, le=1)
    explanation: str


class UseCase(BaseModel):
    title: str
    organization: str | None
    description: str
    url: str


class Trend(BaseModel):
    id: str
    rank: int = Field(description="порядковый номер внутри своей корзины")
    bucket: Bucket
    bucket_reason: str = Field(description="почему кандидат попал именно в эту корзину")
    title: str
    summary: str
    score: float = Field(ge=0, le=1, description="оценка для ранжирования, не вероятность успеха")
    factors: list[ScoreFactor]
    problem: str
    advantage: str
    hypothesis: str | None = Field(default=None, description="применение в банке, предположение команды")
    use_case: UseCase
    first_seen: str = Field(description="первое найденное упоминание в корпусе, YYYY-MM")
    timeline: list[TimelinePoint]
    sources: list[SourceRef]
    document_count: int
    independent_sources: int
    limitations: list[str]


class SourceStat(BaseModel):
    name: str
    source_type: SourceType
    documents: int


class Coverage(BaseModel):
    directions: list[str]
    examples: list[str]
    documents_from: date
    documents_to: date
    document_count: int
    sources: list[SourceStat]
    corpus_version: str
    method_version: str
    updated_at: datetime
    thresholds: dict[str, float] = Field(description="пороги отбора по корзинам, часть версии метода")


class Stage(BaseModel):
    key: str
    label: str


class Analysis(BaseModel):
    id: str
    query: str
    status: AnalysisStatus
    stage: str | None = None
    progress: float = Field(default=0, ge=0, le=1)
    notice: str | None = Field(default=None, description="причина пустой/неполной выдачи или текст ошибки")
    created_at: datetime
    finished_at: datetime | None = None
    corpus_version: str
    method_version: str
    trends: list[Trend] = Field(default_factory=list)


class AnalysisSummary(BaseModel):
    id: str
    query: str
    status: AnalysisStatus
    created_at: datetime
    trend_count: int


class AnalysisRequest(BaseModel):
    query: str = Field(min_length=1, max_length=200)


JobStatus = Literal["pending", "running", "complete", "error"]
JobMode = Literal["cached_snapshot", "live"]
StageStatus = Literal["pending", "running", "complete", "reused", "error"]


class AnalysisStageState(BaseModel):
    key: str
    label: str
    status: StageStatus


class AnalysisJob(BaseModel):
    schema_version: Literal["analysis-job-v1"]
    id: str
    query: str
    mode: JobMode
    status: JobStatus
    stage: str | None = None
    stage_label: str | None = None
    progress: float = Field(ge=0, le=1)
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None = None
    result_available: bool = False
    result_sha256: str | None = None
    stage_history: list[AnalysisStageState] = Field(default_factory=list)
    error: str | None = None
