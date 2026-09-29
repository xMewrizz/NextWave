"""Контракт API. Фронтенд повторяет эти типы в frontend/src/lib/api.ts;
backend/test_frontend_contract.py сверяет оба файла."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


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
