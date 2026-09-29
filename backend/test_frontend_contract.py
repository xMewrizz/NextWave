"""frontend/src/lib/api.ts обязан повторять контракт backend и снимок demo/result."""

import json
import re
import typing
from pathlib import Path

from app.models import AnalysisJob, AnalysisStageState, JobMode, JobStatus, StageStatus

ROOT = Path(__file__).resolve().parents[1]
TS = (ROOT / "frontend/src/lib/api.ts").read_text(encoding="utf-8")
RESULT = json.loads((ROOT / "demo/result/result.json").read_text(encoding="utf-8"))


def _interface(name: str) -> dict[str, tuple[bool, str]]:
    """поле -> (необязательное, тип) из `interface name {...}`; вложенных литералов нет."""
    body = re.search(rf"interface {name} \{{\n(.*?)\n\}}", TS, re.DOTALL)
    assert body, f"interface {name} не найден в api.ts"
    fields = re.findall(r"^  (\w+)(\?)?: (.+)$", body.group(1), re.MULTILINE)
    return {key: (bool(opt), kind) for key, opt, kind in fields}


def _union(name: str) -> set[str]:
    match = re.search(rf"type {name} = ([^\n]+)", TS)
    assert match, f"type {name} не найден в api.ts"
    return set(re.findall(r"'([^']+)'", match.group(1)))


def _keys_match(name: str, rows: list[dict]) -> None:
    """Ключи реальных данных = поля интерфейса; необязательные поля могут отсутствовать."""
    declared = _interface(name)
    required = {key for key, (optional, _) in declared.items() if not optional}
    seen = set().union(*(row.keys() for row in rows))
    assert seen <= declared.keys(), f"{name}: нет в api.ts: {sorted(seen - declared.keys())}"
    assert required <= seen, f"{name}: нет в данных: {sorted(required - seen)}"
    for row in rows:
        assert required <= row.keys(), f"{name}: у строки нет {sorted(required - row.keys())}"


def test_enums_match_backend():
    for name, alias in (("JobStatus", JobStatus), ("JobMode", JobMode), ("StageStatus", StageStatus)):
        assert _union(name) == set(typing.get_args(alias)), name


def test_job_fields_match_backend():
    for name, model in (("AnalysisJob", AnalysisJob), ("AnalysisStageState", AnalysisStageState)):
        declared = _interface(name)
        assert declared.keys() == model.model_fields.keys(), name
        for key, field in model.model_fields.items():
            nullable = type(None) in typing.get_args(field.annotation)
            assert declared[key][1].endswith("| null") == nullable, f"{name}.{key}: nullable"


def test_result_types_match_demo_snapshot():
    candidates = RESULT["candidates"]
    claims = [
        claim
        for row in candidates
        for claim in [
            *row["signal_case"],
            *row["skeptic_case"],
            *([row["case_example"]] if row["case_example"] else []),
        ]
    ]
    factors = [
        factor
        for row in candidates
        for factor in [*row["model"]["top_positive_factors"], *row["model"]["top_negative_factors"]]
    ]
    _keys_match("ResultBundle", [RESULT])
    _keys_match("ResultQuery", [RESULT["query"]])
    _keys_match("ResultSummary", [RESULT["summary"]])
    _keys_match("ResultCandidate", candidates)
    _keys_match("ResultCandidate", RESULT["top15"])
    _keys_match("ResultModel", [row["model"] for row in candidates])
    _keys_match("EvidenceReview", [row["evidence_review"] for row in candidates])
    _keys_match("EvidenceClaimView", claims)
    _keys_match("EvidenceSource", [claim["source"] for claim in claims])
    _keys_match("ModelFactor", factors)


def test_result_enums_match_demo_snapshot():
    assert {row["status"] for row in RESULT["candidates"]} <= _union("Bucket")
