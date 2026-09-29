"""Durable filesystem store for backend analysis jobs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

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


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _completed_job(
    job: AnalysisJob,
    result_raw: bytes,
    stages: list[AnalysisStageState],
) -> AnalysisJob:
    now = datetime.now(UTC)
    return job.model_copy(
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


def _failed_job(job: AnalysisJob, message: str) -> AnalysisJob:
    now = datetime.now(UTC)
    return job.model_copy(
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


def _running_job(job: AnalysisJob) -> AnalysisJob:
    stage_history = [
        stage.model_copy(
            update={"status": "running" if stage.key == "result" else "reused"}
        )
        for stage in job.stage_history
    ]
    return job.model_copy(
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
        complete = _completed_job(job, result_raw, stages)
        self.save(complete)
        return complete

    def fail(self, job: AnalysisJob, message: str) -> AnalysisJob:
        failed = _failed_job(job, message)
        self.save(failed)
        return failed

    def mark_running(self, job: AnalysisJob) -> AnalysisJob:
        running = _running_job(job)
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

    def healthcheck(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)


class PostgresAnalysisJobStore:
    """PostgreSQL-backed job store used by the integrated Docker runtime."""

    def __init__(self, connection_string: str) -> None:
        self.connection_string = connection_string
        self._ensure_schema()

    def _connect(self):
        return psycopg.connect(self.connection_string)

    def _ensure_schema(self) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_jobs (
                    id VARCHAR(12) PRIMARY KEY,
                    job JSONB NOT NULL,
                    result_raw BYTEA,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

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
        value = Jsonb(job.model_dump(mode="json"))
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO analysis_jobs (id, job, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (id) DO UPDATE
                SET job = EXCLUDED.job, updated_at = NOW()
                """,
                (job.id, value),
            )

    def get(self, job_id: str) -> AnalysisJob | None:
        if not _JOB_ID.fullmatch(job_id):
            return None
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT job FROM analysis_jobs WHERE id = %s", (job_id,))
            row = cursor.fetchone()
        if row is None:
            return None
        try:
            return AnalysisJob.model_validate(row[0])
        except ValueError as error:
            raise ValueError(f"analysis job {job_id} is invalid") from error

    def list(self) -> list[AnalysisJob]:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT job FROM analysis_jobs "
                "ORDER BY (job->>'created_at')::timestamptz DESC"
            )
            rows = cursor.fetchall()
        try:
            return [AnalysisJob.model_validate(row[0]) for row in rows]
        except ValueError as error:
            raise ValueError("stored analysis job is invalid") from error

    def complete(
        self,
        job: AnalysisJob,
        result: dict[str, Any],
        stages: list[AnalysisStageState],
    ) -> AnalysisJob:
        result_raw = _json_bytes(result)
        complete = _completed_job(job, result_raw, stages)
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE analysis_jobs
                SET job = %s, result_raw = %s, updated_at = NOW()
                WHERE id = %s
                """,
                (Jsonb(complete.model_dump(mode="json")), result_raw, job.id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"analysis job {job.id} does not exist")
        return complete

    def fail(self, job: AnalysisJob, message: str) -> AnalysisJob:
        failed = _failed_job(job, message)
        self.save(failed)
        return failed

    def mark_running(self, job: AnalysisJob) -> AnalysisJob:
        running = _running_job(job)
        self.save(running)
        return running

    def load_result(self, job: AnalysisJob) -> dict[str, Any]:
        if not job.result_available or not job.result_sha256:
            raise ValueError("analysis result is not available")
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT result_raw FROM analysis_jobs WHERE id = %s", (job.id,)
            )
            row = cursor.fetchone()
        if row is None or row[0] is None:
            raise ValueError("cannot read persisted analysis result")
        raw = bytes(row[0])
        if hashlib.sha256(raw).hexdigest() != job.result_sha256:
            raise ValueError("persisted analysis result checksum mismatch")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("cannot read persisted analysis result") from error
        if not isinstance(value, dict) or value.get("schema_version") != "analysis-response-v1":
            raise ValueError("persisted analysis result schema is not supported")
        return value

    def healthcheck(self) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            if cursor.fetchone() != (1,):
                raise ValueError("PostgreSQL healthcheck failed")


@lru_cache(maxsize=4)
def _postgres_store(connection_string: str) -> PostgresAnalysisJobStore:
    return PostgresAnalysisJobStore(connection_string)


def configured_analysis_job_store() -> AnalysisJobStore | PostgresAnalysisJobStore:
    connection_string = os.getenv("NEXTWAVE_DATABASE_URL", "").strip()
    if connection_string:
        return _postgres_store(connection_string)
    return AnalysisJobStore()
