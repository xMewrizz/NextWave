"""Backend API models built on the canonical NextWave ML contracts."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

from nextwave.contracts import CandidateAssessment, CandidateStatus, SourceType

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
    summary: str
    factors: list[ScoreFactor]
    problem: str
    advantage: str
    hypothesis: str | None = None
    use_case: UseCase
    first_seen: str
    timeline: list[TimelinePoint]
    document_count: int = Field(ge=0)
    limitations: list[str]


class SourceStat(BaseModel):
    name: str
    source_type: SourceType
    documents: int = Field(ge=0)


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
    thresholds: dict[str, float]


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
