"""Одна проверка: контракт API и правила ранжирования не разъехались.

Запуск: uv run pytest
"""

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from app import pipeline
from app.main import app


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as c:
        yield c


async def _wait(client: AsyncClient, analysis_id: str) -> dict:
    for _ in range(100):
        body = (await client.get(f"/api/analyses/{analysis_id}")).json()
        if body["status"] in {"done", "empty", "error"}:
            return body
        await asyncio.sleep(0.1)
    raise AssertionError("анализ не завершился")


def test_ranking_is_reproducible():
    trends = pipeline._demo["trends"]
    first = pipeline.rank(trends)
    second = pipeline.rank(list(reversed(trends)))
    assert [t.id for t in first] == [t.id for t in second], "порядок должен зависеть только от баллов"
    assert all(0 <= t.score <= 1 for t in first)


def test_buckets_are_capped_numbered_and_explained():
    ranked = pipeline.rank(pipeline._demo["trends"])
    by_bucket = {b: [t for t in ranked if t.bucket == b] for b in ("main", "watchlist", "excluded")}

    assert len(by_bucket["main"]) <= pipeline.TOP_N, "основной список не может быть длиннее ТОП-15"
    for bucket, trends in by_bucket.items():
        assert [t.rank for t in trends] == list(range(1, len(trends) + 1)), f"{bucket}: сквозная нумерация"
        assert [t.score for t in trends] == sorted((t.score for t in trends), reverse=True)
        assert all(t.bucket_reason for t in trends), f"{bucket}: карточка без причины попадания"

    # Кандидат не может оказаться в основном списке, не пройдя пороги
    for trend in by_bucket["main"]:
        factors = {f.key: f.value for f in trend.factors}
        assert factors["novelty"] >= pipeline.THRESHOLDS["novelty_min"]
        assert factors["growth"] >= pipeline.THRESHOLDS["growth_min"]
        assert factors["evidence"] >= pipeline.THRESHOLDS["evidence_min"]
        assert trend.independent_sources >= pipeline.THRESHOLDS["independent_min"]
        assert trend.document_count >= pipeline.THRESHOLDS["documents_min"]

    # Отсев — только по новизне или зарождаемости, а не по слабым доказательствам
    for trend in by_bucket["excluded"]:
        factors = {f.key: f.value for f in trend.factors}
        assert (
            factors["novelty"] < pipeline.THRESHOLDS["novelty_min"]
            or factors["growth"] < pipeline.THRESHOLDS["growth_min"]
        ), f"{trend.id}: отсеян без основания"


def test_every_card_has_evidence():
    for trend in pipeline.rank(pipeline._demo["trends"]):
        assert trend.sources, f"{trend.id}: карточка без источников не попадает в выдачу"
        assert trend.use_case.url, f"{trend.id}: кейс без подтверждающей ссылки"
        assert {f.key for f in trend.factors} == set(pipeline.FACTOR_WEIGHTS)


async def test_covered_query_returns_ranked_trends(client: AsyncClient):
    created = (await client.post("/api/analyses", json={"query": "технологии в ИИ"})).json()
    assert created["status"] == "pending"

    body = await _wait(client, created["id"])
    assert body["status"] == "done"
    assert body["trends"], "покрытое направление должно давать непустую выдачу"
    assert body["corpus_version"] == pipeline.CORPUS_VERSION

    trend_id = body["trends"][0]["id"]
    assert (await client.get(f"/api/analyses/{created['id']}/trends/{trend_id}")).status_code == 200


async def test_uncovered_query_is_empty_not_error(client: AsyncClient):
    created = (await client.post("/api/analyses", json={"query": "выращивание тюльпанов"})).json()
    body = await _wait(client, created["id"])
    assert body["status"] == "empty"
    assert body["trends"] == []
    assert body["notice"], "пустая выдача обязана объяснять причину (US-01, US-06)"


async def test_empty_query_is_rejected(client: AsyncClient):
    assert (await client.post("/api/analyses", json={"query": "   "})).status_code == 422
