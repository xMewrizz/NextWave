"""Final assessments → persistent storage → existing API. Fixtures are never production data."""

import asyncio
import copy
import json
import os
import sys
import threading
import types
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from nextwave.analysis import AnalysisMetadata
from nextwave.contracts import CandidateAssessment
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from app import database, pipeline
from app.main import _tasks, app
from app.models import Analysis
from generate_frontend_contracts import TARGET, render_typescript_contracts

EXAMPLE = Path(__file__).resolve().parents[1] / "docs/examples/ml-assessments.json"
METADATA = AnalysisMetadata(
    method_version="contract-test-policy-v1",
    corpus_version="contract-test-corpus-v1",
    model_version="contract-test-model-v1",
    feature_version="contract-test-features-v1",
)


@pytest.fixture
def payload():
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


@pytest.fixture
def storage(monkeypatch, tmp_path):
    # PostgreSQL tests use a fresh isolated schema, never drop a developer's tables.
    url = os.getenv("NEXTWAVE_TEST_DATABASE_URL")
    admin = None
    if url:
        admin = create_engine(url)
        schema = "test_" + uuid.uuid4().hex
        with admin.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
        engine = create_engine(url, connect_args={"options": f"-csearch_path={schema}"})
    else:
        engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'analysis.db'}")
    monkeypatch.setattr(database, "engine", engine)
    monkeypatch.setattr(database, "SessionLocal", sessionmaker(bind=engine, expire_on_commit=False))
    monkeypatch.delenv("NEXTWAVE_ANALYZER_FACTORY", raising=False)
    yield engine
    engine.dispose()
    if admin:
        with admin.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


@pytest.fixture
async def client(storage):
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            yield client
        if _tasks:
            await asyncio.gather(*tuple(_tasks))


@pytest.fixture
def install_analyzer(monkeypatch):
    def install(analyze, metadata=METADATA):
        module = types.ModuleType("test_ml_integration")
        module.build_analyzer = lambda: types.SimpleNamespace(metadata=metadata, analyze=analyze)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        monkeypatch.setenv("NEXTWAVE_ANALYZER_FACTORY", "test_ml_integration:build_analyzer")

    return install


async def wait_result(client, analysis_id):
    for _ in range(100):
        response = await client.get(f"/api/analyses/{analysis_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] in {"done", "empty", "error"}:
            assert body["finished_at"]
            assert body["progress"] == 1
            return body
        await asyncio.sleep(0.01)
    raise AssertionError("analysis did not finish")


async def start(client, query="Проверка интеграции"):
    response = await client.post("/api/analyses", json={"query": query})
    assert response.status_code == 201
    assert response.json()["status"] == "pending"
    return response.json()["id"]


def test_frontend_contract_is_generated_from_backend_models():
    assert TARGET.read_text(encoding="utf-8") == render_typescript_contracts()


def test_documented_api_response_matches_adapter(payload):
    example = Analysis.model_validate_json(EXAMPLE.with_name("api-analysis.json").read_text())
    assert example.trends == pipeline.rank(payload, example.query, METADATA)
    assert example.model_version == METADATA.model_version


def test_example_and_ranking_preserve_all_decisions(payload):
    items = []
    for index in range(17):
        item = copy.deepcopy(payload[0])
        item.update(candidate_id=f"main-{index:02}", status="main")
        items.append(item)
    ranked = pipeline.rank(items, "Проверка интеграции", METADATA)
    assert len(ranked) == 17  # Preserve all final assessments; TOP-15 is a UI selection.
    assert [item.rank for item in ranked] == list(range(1, 18))
    assert ranked == pipeline.rank(items[::-1], "Проверка интеграции", METADATA)
    assert all(item.status == "main" for item in ranked)
    assert all(item.summary is None and item.document_count is None for item in ranked)
    assert CandidateAssessment.model_validate(payload[0]).evidence[0].direction == "support"


async def test_all_buckets_persist_and_get_never_runs_model(client, storage, install_analyzer, payload):
    assessments = []
    for status in ("main", "watchlist", "excluded"):
        item = copy.deepcopy(payload[0])
        item.update(candidate_id=status, status=status)
        item["exclusion_reason"] = "mature" if status == "excluded" else None
        item["evidence"][0]["direction"] = "counter" if status == "excluded" else "support"
        assessments.append(item)
    calls = []

    def analyze(query):
        calls.append(query)
        return assessments

    install_analyzer(analyze)
    analysis_id = await start(client)
    body = await wait_result(client, analysis_id)
    assert body["status"] == "done"
    assert body["started_at"]
    assert body["notice"].startswith("В основной список прошли 1 кандидатов из 15")
    for key, value in METADATA.model_dump().items():
        assert body[key] == value
    for expected in assessments:
        card = (await client.get(f"/api/analyses/{analysis_id}/trends/{expected['candidate_id']}")).json()
        assert {key: card[key] for key in expected} == expected
    storage.dispose()  # Fresh DB connections still return the complete result.
    assert (await client.get(f"/api/analyses/{analysis_id}")).json() == body
    history = (await client.get("/api/analyses")).json()
    assert history[0]["trend_count"] == 3
    assert calls == ["Проверка интеграции"]
    assert (await client.get(f"/api/analyses/{analysis_id}/trends/missing")).status_code == 404


async def test_unavailable_is_explicit_and_no_demo(client):
    result = await wait_result(client, await start(client, "технологии в ИИ"))
    assert result["status"] == "error"
    assert result["error_code"] == "model_unavailable"
    assert "Интеграция модели недоступна" in result["notice"]
    assert result["trends"] == []
    coverage = (await client.get("/api/coverage")).json()
    assert coverage["document_count"] is None
    assert coverage["thresholds"] == {}


async def test_empty_result_keeps_versions(client, install_analyzer):
    install_analyzer(lambda query: [])
    result = await wait_result(client, await start(client))
    assert result["status"] == "empty"
    assert result["model_version"] == METADATA.model_version
    assert result["notice"] and result["trends"] == []


@pytest.mark.parametrize(
    "defect",
    [
        "score",
        "nonfinite",
        "query",
        "version",
        "duplicate",
        "gate",
        "discovery",
        "reason",
        "evidence",
        "direction",
    ],
)
async def test_invalid_output_is_atomic(client, install_analyzer, payload, defect):
    item = payload[0]
    if defect == "score":
        item["prediction"]["weak_signal_score"] = 2
    elif defect == "nonfinite":
        item["features"]["publication_momentum"] = float("nan")
    elif defect == "query":
        item["query"] = "another run"
    elif defect == "version":
        item["prediction"]["model_version"] = "another model"
    elif defect == "duplicate":
        payload.append(copy.deepcopy(item))
    elif defect == "gate":
        item["status"] = "reject"
    elif defect == "discovery":
        payload = {"candidate_proposals": []}
    elif defect == "reason":
        item["status"] = "excluded"
    elif defect == "evidence":
        item.update(status="main", evidence=[])
    elif defect == "direction":
        item["evidence"][0]["direction"] = "invented"
    install_analyzer(lambda query: payload)
    result = await wait_result(client, await start(client))
    assert result["error_code"] == "invalid_model_result"
    assert result["trends"] == []


async def test_error_does_not_leak_exception(client, install_analyzer):
    def analyze(query):
        raise RuntimeError("secret-api-key raw-private-data")

    install_analyzer(analyze)
    result = await wait_result(client, await start(client))
    assert result["error_code"] == "analysis_failed"
    assert "secret" not in json.dumps(result)
    assert "private" not in json.dumps(result)


async def test_running_is_persisted_without_blocking_api(client, install_analyzer):
    release = threading.Event()

    def analyze(query):
        if not release.wait(5):
            raise RuntimeError("test timed out")
        return []

    install_analyzer(analyze)
    analysis_id = await start(client)
    try:
        for _ in range(100):
            body = (await client.get(f"/api/analyses/{analysis_id}")).json()
            if body["model_version"]:
                break
            await asyncio.sleep(0.01)
        assert body["status"] == "running"
        assert body["started_at"] and body["progress"] < 1
        assert body["model_version"] == METADATA.model_version
    finally:
        release.set()
    assert (await wait_result(client, analysis_id))["status"] == "empty"


async def test_queries_and_unknown_ids(client):
    for query in ("", "   ", "x" * 201):
        assert (await client.post("/api/analyses", json={"query": query})).status_code == 422
    assert (await client.get("/api/analyses/missing")).status_code == 404


def test_restart_marks_interrupted_run(storage):
    database.init_database()
    database.save_analysis(
        Analysis(
            id="interrupted",
            query="test",
            status="running",
            created_at=datetime.now(UTC),
            **METADATA.model_dump(),
        )
    )
    database.mark_interrupted_analyses()
    result = database.get_analysis("interrupted")
    assert result.status == "error"
    assert result.error_code == "interrupted"
    assert result.finished_at and result.progress == 1


def test_existing_schema_migrates_idempotently(storage, payload):
    # Exact pre-V1-10 columns; old JSON cards remain intact after migration.
    card = pipeline.rank(payload, "Проверка интеграции", METADATA)[0].model_dump(mode="json")
    card["evidence"][0].pop("direction")
    with storage.begin() as connection:
        connection.execute(
            text("""
            CREATE TABLE analyses (
                id VARCHAR(32) PRIMARY KEY, query VARCHAR(200), status VARCHAR(16),
                stage VARCHAR(200), progress FLOAT, notice TEXT,
                created_at TIMESTAMP, finished_at TIMESTAMP,
                corpus_version VARCHAR(100), method_version VARCHAR(100), trends JSON
            )
        """)
        )
        connection.execute(
            text("""
            INSERT INTO analyses (id, query, status, progress, created_at,
                corpus_version, method_version, trends)
            VALUES ('legacy', 'Проверка интеграции', 'done', 1, :created,
                'demo-legacy', 'legacy-policy', :trends)
        """),
            {"created": datetime.now(UTC).isoformat(), "trends": json.dumps([card])},
        )
    database.init_database()
    database.init_database()
    assert {"started_at", "model_version", "feature_version", "error_code"} <= {
        column["name"] for column in inspect(storage).get_columns("analyses")
    }
    result = database.get_analysis("legacy")
    assert result.corpus_version == "demo-legacy"
    assert result.model_version is None
    assert result.trends[0].evidence[0].direction is None
    with storage.connect() as connection:
        raw = connection.execute(text("SELECT trends FROM analyses WHERE id='legacy'")).scalar_one()
    assert (json.loads(raw) if isinstance(raw, str) else raw) == [card]
