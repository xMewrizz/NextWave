"""Filesystem persistence tests for analysis jobs."""

from pathlib import Path

import pytest

from app.job_store import AnalysisJobStore
from app.models import AnalysisStageState


def _result() -> dict:
    return {
        "schema_version": "analysis-response-v1",
        "query": {"text": "test"},
        "summary": {},
        "top15": [],
        "candidates": [],
    }


def test_round_trip_complete_job_and_result(tmp_path: Path):
    store = AnalysisJobStore(tmp_path / "jobs")
    job = store.create("test")
    running = store.mark_running(job)
    complete = store.complete(
        running,
        _result(),
        [AnalysisStageState(key="result", label="Result", status="reused")],
    )

    restored = store.get(job.id)
    assert restored == complete
    assert restored is not None
    assert store.load_result(restored) == _result()
    assert store.list() == [complete]


def test_invalid_job_id_cannot_escape_root(tmp_path: Path):
    store = AnalysisJobStore(tmp_path / "jobs")
    assert store.get("../escape") is None


def test_corrupt_job_is_rejected(tmp_path: Path):
    store = AnalysisJobStore(tmp_path / "jobs")
    job = store.create("test")
    (store.root / job.id / "job.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="is invalid"):
        store.get(job.id)


def test_tampered_result_is_rejected(tmp_path: Path):
    store = AnalysisJobStore(tmp_path / "jobs")
    job = store.complete(store.create("test"), _result(), [])
    (store.root / job.id / "result.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        store.load_result(job)
