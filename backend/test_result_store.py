import hashlib
import json
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app import result_store
from app.main import app


def _json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()


def _digest(raw: bytes) -> dict[str, object]:
    return {"size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _fixture(root: Path) -> Path:
    candidate = {
        "schema_version": result_store.RESULT_SCHEMA_VERSION,
        "candidate_id": "candidate-1",
        "status": "main",
        "top15_rank": 1,
    }
    candidates = _json(candidate)
    top15 = _json([candidate])
    summary = _json({"candidate_count": 1, "top15_count": 1})
    result_value = {
        "schema_version": result_store.RESPONSE_SCHEMA_VERSION,
        "query": {"text": "Инфраструктурные технологии для обучения и инференса ИИ"},
        "summary": json.loads(summary),
        "top15": json.loads(top15),
        "candidates": [candidate],
    }
    result = _json(result_value)
    manifest = {
        "schema_version": result_store.RESULT_SCHEMA_VERSION,
        "candidate_count": 1,
        "top15_count": 1,
        "outputs": {
            "candidates.jsonl": _digest(candidates),
            "top15.json": _digest(top15),
            "summary.json": _digest(summary),
            "result.json": _digest(result),
        },
    }
    root.mkdir()
    (root / "candidates.jsonl").write_bytes(candidates)
    (root / "top15.json").write_bytes(top15)
    (root / "summary.json").write_bytes(summary)
    (root / "result.json").write_bytes(result)
    (root / "manifest.json").write_bytes(_json(manifest))
    return root


def test_loads_checked_result_bundle(tmp_path: Path):
    bundle = result_store.load_result_bundle(_fixture(tmp_path / "result"))
    assert bundle["summary"]["candidate_count"] == 1
    assert bundle["top15"][0]["candidate_id"] == "candidate-1"


def test_loads_watchlist_candidate_in_top15(tmp_path: Path):
    root = _fixture(tmp_path / "result")
    for filename in ("candidates.jsonl", "top15.json", "result.json"):
        path = root / filename
        value = path.read_text(encoding="utf-8").replace('"status": "main"', '"status": "watchlist"')
        path.write_text(value, encoding="utf-8")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for filename in ("candidates.jsonl", "top15.json", "result.json"):
        manifest["outputs"][filename] = _digest((root / filename).read_bytes())
    (root / "manifest.json").write_bytes(_json(manifest))

    bundle = result_store.load_result_bundle(root)

    assert bundle["top15"][0]["status"] == "watchlist"


def test_rejects_excluded_candidate_in_top15(tmp_path: Path):
    root = _fixture(tmp_path / "result")
    for filename in ("candidates.jsonl", "top15.json", "result.json"):
        path = root / filename
        value = path.read_text(encoding="utf-8").replace('"status": "main"', '"status": "excluded"')
        path.write_text(value, encoding="utf-8")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for filename in ("candidates.jsonl", "top15.json", "result.json"):
        manifest["outputs"][filename] = _digest((root / filename).read_bytes())
    (root / "manifest.json").write_bytes(_json(manifest))

    with pytest.raises(ValueError, match="top15 is inconsistent"):
        result_store.load_result_bundle(root)


def test_rejects_tampered_candidate_file(tmp_path: Path):
    root = _fixture(tmp_path / "result")
    (root / "candidates.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from manifest"):
        result_store.load_result_bundle(root)


async def test_api_serves_configured_result(tmp_path: Path, monkeypatch):
    root = _fixture(tmp_path / "result")
    monkeypatch.setenv("NEXTWAVE_RESULT_DIR", str(root))
    async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
        response = await client.get("/api/result/current")
    assert response.status_code == 200
    assert response.json()["schema_version"] == result_store.RESPONSE_SCHEMA_VERSION


async def test_api_returns_503_for_missing_result(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NEXTWAVE_RESULT_DIR", str(tmp_path / "missing"))
    async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
        response = await client.get("/api/result/current")
    assert response.status_code == 503


async def test_synthetic_routes_are_disabled_by_default(monkeypatch):
    monkeypatch.delenv("NEXTWAVE_ENABLE_SYNTHETIC_DEMO", raising=False)
    async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
        response = await client.get("/api/coverage")
    assert response.status_code == 404
