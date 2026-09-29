"""Backend contract tests for durable analysis jobs and result delivery."""

import asyncio
import hashlib
import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app import main, pipeline, result_store
from app.job_store import AnalysisJobStore

QUERY = "Инфраструктурные технологии для обучения и инференса ИИ"


def _json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()


def _digest(raw: bytes) -> dict[str, object]:
    return {"size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _result_fixture(root: Path) -> Path:
    candidate = {
        "schema_version": result_store.RESULT_SCHEMA_VERSION,
        "candidate_id": "candidate-1",
        "status": "main",
        "top15_rank": 1,
    }
    candidates = _json(candidate)
    top15 = _json([candidate])
    summary = _json({"candidate_count": 1, "top15_count": 1})
    response_value = {
        "schema_version": result_store.RESPONSE_SCHEMA_VERSION,
        "query": {"text": QUERY},
        "summary": json.loads(summary),
        "top15": json.loads(top15),
        "candidates": [candidate],
    }
    response = _json(response_value)
    manifest = {
        "schema_version": result_store.RESULT_SCHEMA_VERSION,
        "candidate_count": 1,
        "top15_count": 1,
        "outputs": {
            "candidates.jsonl": _digest(candidates),
            "top15.json": _digest(top15),
            "summary.json": _digest(summary),
            "result.json": _digest(response),
        },
    }
    root.mkdir()
    (root / "candidates.jsonl").write_bytes(candidates)
    (root / "top15.json").write_bytes(top15)
    (root / "summary.json").write_bytes(summary)
    (root / "result.json").write_bytes(response)
    (root / "manifest.json").write_bytes(_json(manifest))
    return root


async def _clear_tasks() -> None:
    tasks = list(main._tasks.values())
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    main._tasks.clear()


@pytest.fixture
async def client(tmp_path: Path, monkeypatch):
    await _clear_tasks()
    result_dir = _result_fixture(tmp_path / "result")
    monkeypatch.setenv("NEXTWAVE_RESULT_DIR", str(result_dir))
    monkeypatch.setenv("NEXTWAVE_JOB_DIR", str(tmp_path / "jobs"))
    async with AsyncClient(
        transport=ASGITransport(main.app), base_url="http://test"
    ) as http_client:
        yield http_client
    await _clear_tasks()


async def _wait(client: AsyncClient, analysis_id: str) -> dict:
    for _ in range(100):
        response = await client.get(f"/api/analyses/{analysis_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] in {"complete", "error"}:
            return body
        await asyncio.sleep(0.01)
    raise AssertionError("analysis job did not reach a terminal state")


def test_ranking_is_reproducible():
    trends = pipeline._demo["trends"]
    first = pipeline.rank(trends)
    second = pipeline.rank(list(reversed(trends)))
    assert [trend.id for trend in first] == [trend.id for trend in second]
    assert all(0 <= trend.score <= 1 for trend in first)


def test_buckets_are_capped_numbered_and_explained():
    ranked = pipeline.rank(pipeline._demo["trends"])
    by_bucket = {
        bucket: [trend for trend in ranked if trend.bucket == bucket]
        for bucket in ("main", "watchlist", "excluded")
    }
    assert len(by_bucket["main"]) <= pipeline.TOP_N
    for bucket, trends in by_bucket.items():
        assert [trend.rank for trend in trends] == list(range(1, len(trends) + 1))
        assert [trend.score for trend in trends] == sorted(
            (trend.score for trend in trends), reverse=True
        )
        assert all(trend.bucket_reason for trend in trends), bucket


async def test_job_persists_result_and_stage_history(client: AsyncClient):
    created_response = await client.post("/api/analyses", json={"query": QUERY})
    assert created_response.status_code == 201
    created = created_response.json()
    assert created["status"] == "pending"
    assert created["mode"] == "cached_snapshot"

    job = await _wait(client, created["id"])
    assert job["status"] == "complete"
    assert job["progress"] == 1.0
    assert job["result_available"] is True
    assert [stage["key"] for stage in job["stage_history"]] == [
        "source_search",
        "candidate_gate",
        "enrichment",
        "model",
        "evidence_duel",
        "result",
    ]
    assert {stage["status"] for stage in job["stage_history"]} == {"reused"}

    result_response = await client.get(f"/api/analyses/{created['id']}/result")
    assert result_response.status_code == 200
    assert result_response.json()["schema_version"] == "analysis-response-v1"

    listed = (await client.get("/api/analyses")).json()
    assert [item["id"] for item in listed] == [created["id"]]


async def test_health_checks_store_and_result(client: AsyncClient):
    response = await client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_mismatched_query_fails_without_substituting_result(client: AsyncClient):
    created = (
        await client.post("/api/analyses", json={"query": "Технологии квантовой связи"})
    ).json()
    job = await _wait(client, created["id"])
    assert job["status"] == "error"
    assert "live-runner" in job["error"]
    assert job["result_available"] is False
    response = await client.get(f"/api/analyses/{created['id']}/result")
    assert response.status_code == 409


async def test_pending_job_is_resumed_after_process_restart(client: AsyncClient):
    store = AnalysisJobStore()
    job = store.create(QUERY)
    await _clear_tasks()

    response = await client.get(f"/api/analyses/{job.id}")
    assert response.status_code == 200
    completed = await _wait(client, job.id)
    assert completed["status"] == "complete"


async def test_unknown_job_returns_404(client: AsyncClient):
    assert (await client.get("/api/analyses/000000000000")).status_code == 404
    assert (await client.get("/api/analyses/not-an-id")).status_code == 404


async def test_tampered_persisted_result_is_rejected(client: AsyncClient):
    created = (await client.post("/api/analyses", json={"query": QUERY})).json()
    job = await _wait(client, created["id"])
    result_path = main._store().root / job["id"] / "result.json"
    result_path.write_text("{}\n", encoding="utf-8")

    response = await client.get(f"/api/analyses/{job['id']}/result")
    assert response.status_code == 500
    assert "checksum mismatch" in response.json()["detail"]


async def test_blank_query_is_rejected(client: AsyncClient):
    assert (await client.post("/api/analyses", json={"query": "   "})).status_code == 422
