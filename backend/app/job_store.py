"""Durable filesystem store for backend analysis jobs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import AnalysisJob, AnalysisStageState

_JOB_ID = re.compile(r"^[0-9a-f]{12}$")
JOB_SCHEMA_VERSION = "analysis-job-v1"


def configured_job_dir() -> Path:
    value = os.getenv("NEXTWAVE_JOB_DIR")
    if value:
        return Path(value)
    return Path(__file__).resolve().parents[2] / "runtime" / "backend-jobs"


def _atomic_json(path: Path, value: object) -> bytes:
    raw = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    staging.write_bytes(raw)
    os.replace(staging, path)
    return raw


class AnalysisJobStore:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else configured_job_dir()

    def create(
        self, query: str, stages: list[AnalysisStageState] | None = None
    ) -> AnalysisJob:
        now = datetime.now(UTC)
        job = AnalysisJob(
            schema_version=JOB_SCHEMA_VERSION,
            id=uuid.uuid4().hex[:12],
            query=query,
            mode="cached_snapshot",
            status="pending",
            progress=0.0,
            created_at=now,
            updated_at=now,
            stage_history=list(stages or []),
        )
        self.save(job)
        return job

    def save(self, job: AnalysisJob) -> None:
        _atomic_json(self._job_path(job.id), job.model_dump(mode="json"))

    def get(self, job_id: str) -> AnalysisJob | None:
        if not _JOB_ID.fullmatch(job_id):
            return None
        path = self._job_path(job_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read analysis job {job_id}") from error
        try:
            return AnalysisJob.model_validate(value)
        except ValueError as error:
            raise ValueError(f"analysis job {job_id} is invalid") from error

    def list(self) -> list[AnalysisJob]:
        if not self.root.exists():
            return []
        jobs = [
            job
            for path in self.root.iterdir()
            if path.is_dir() and (job := self.get(path.name)) is not None
        ]
        return sorted(jobs, key=lambda item: item.created_at, reverse=True)

    def complete(
        self,
        job: AnalysisJob,
        result: dict[str, Any],
        stages: list[AnalysisStageState],
    ) -> AnalysisJob:
        result_raw = _atomic_json(self._result_path(job.id), result)
        now = datetime.now(UTC)
        complete = job.model_copy(
            update={
                "status": "complete",
                "stage": "result",
                "stage_label": "Формирование результата",
                "progress": 1.0,
                "updated_at": now,
                "finished_at": now,
                "result_available": True,
                "result_sha256": hashlib.sha256(result_raw).hexdigest(),
                "stage_history": stages,
                "error": None,
            }
        )
        self.save(complete)
        return complete

    def fail(self, job: AnalysisJob, message: str) -> AnalysisJob:
        now = datetime.now(UTC)
        failed = job.model_copy(
            update={
                "status": "error",
                "stage": None,
                "stage_label": None,
                "progress": 1.0,
                "updated_at": now,
                "finished_at": now,
                "result_available": False,
                "error": message,
            }
        )
        self.save(failed)
        return failed

    def mark_running(self, job: AnalysisJob) -> AnalysisJob:
        stage_history = [
            stage.model_copy(
                update={"status": "running" if stage.key == "result" else "reused"}
            )
            for stage in job.stage_history
        ]
        running = job.model_copy(
            update={
                "status": "running",
                "stage": "result",
                "stage_label": "Проверка сохранённого результата",
                "progress": 0.95,
                "updated_at": datetime.now(UTC),
                "stage_history": stage_history,
                "error": None,
            }
        )
        self.save(running)
        return running

    def load_result(self, job: AnalysisJob) -> dict[str, Any]:
        if not job.result_available or not job.result_sha256:
            raise ValueError("analysis result is not available")
        try:
            raw = self._result_path(job.id).read_bytes()
            value = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("cannot read persisted analysis result") from error
        if hashlib.sha256(raw).hexdigest() != job.result_sha256:
            raise ValueError("persisted analysis result checksum mismatch")
        if not isinstance(value, dict) or value.get("schema_version") != "analysis-response-v1":
            raise ValueError("persisted analysis result schema is not supported")
        return value

    def _job_path(self, job_id: str) -> Path:
        return self.root / job_id / "job.json"

    def _result_path(self, job_id: str) -> Path:
        return self.root / job_id / "result.json"
