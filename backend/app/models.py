"""Backend API models built on the canonical NextWave ML contracts."""

from datetime import date, datetime
from typing import Literal

from nextwave.contracts import CandidateAssessment, CandidateStatus, SourceType
from pydantic import BaseModel, Field

AnalysisStatus = Literal["pending", "running", "done", "empty", "error"]
Bucket = CandidateStatus
BUCKETS = tuple(CandidateStatus)
FactorKey = Literal["growth", "novelty", "independence", "evidence"]


class TimelinePoint(BaseModel):
    period: str
    documents: int = Field(ge=0)
    share: float = Field(ge=0)


class ScoreFactor(BaseModel):
    key: FactorKey
    value: float = Field(ge=0, le=1)
    explanation: str


class UseCase(BaseModel):
    title: str
    organization: str | None
    description: str
    url: str


class Trend(CandidateAssessment):
    """Canonical candidate plus presentation data required by the UI."""

    rank: int = Field(ge=1)
    summary: str | None = None
    factors: list[ScoreFactor] = Field(default_factory=list)
    problem: str | None = None
    advantage: str | None = None
    hypothesis: str | None = None
    use_case: UseCase | None = None
    first_seen: str | None = None
    timeline: list[TimelinePoint] = Field(default_factory=list)
    document_count: int | None = Field(default=None, ge=0)
    limitations: list[str] = Field(default_factory=list)


class SourceStat(BaseModel):
    name: str
    source_type: SourceType
    documents: int = Field(ge=0)


class Coverage(BaseModel):
    directions: list[str]
    examples: list[str]
    documents_from: date | None
    documents_to: date | None
    document_count: int | None
    sources: list[SourceStat]
    corpus_version: str | None
    method_version: str | None
    updated_at: datetime | None
    thresholds: dict[str, float]
    notice: str | None = None


class Stage(BaseModel):
    key: str
    label: str


class Analysis(BaseModel):
    id: str
    query: str
    status: AnalysisStatus
    stage: str | None = None
    progress: float = Field(default=0, ge=0, le=1)
    notice: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_code: str | None = None
    corpus_version: str
    method_version: str
    model_version: str | None = None
    feature_version: str | None = None
    trends: list[Trend] = Field(default_factory=list)


class AnalysisSummary(BaseModel):
    id: str
    query: str
    status: AnalysisStatus
    created_at: datetime
    trend_count: int


class AnalysisRequest(BaseModel):
    query: str = Field(min_length=1, max_length=200)
