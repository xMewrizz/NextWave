"""End-to-end smoke test for the persisted NextWave web analysis contract."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_QUERY = "Инфраструктурные технологии для обучения и инференса ИИ"
TERMINAL = {"complete", "error"}


def _request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    expected_status: int = 200,
) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}", data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            status = response.status
            raw = response.read()
    except urllib.error.HTTPError as error:
        status = error.code
        raw = error.read()
    if status != expected_status:
        raise RuntimeError(
            f"{method} {path} returned HTTP {status}, expected {expected_status}: "
            f"{raw.decode('utf-8', errors='replace')[:500]}"
        )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{method} {path} returned invalid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{method} {path} did not return a JSON object")
    return value


def _wait_job(base_url: str, job_id: str, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        job = _request(base_url, f"/api/analyses/{job_id}")
        if job.get("status") in TERMINAL:
            return job
        time.sleep(0.25)
    raise RuntimeError(f"analysis job {job_id} did not finish in {timeout_seconds:g}s")


def _validate_result(result: dict[str, Any], expected_query: str, expected_candidates: int) -> None:
    if result.get("schema_version") != "analysis-response-v1":
        raise RuntimeError("unexpected result schema")
    query = result.get("query")
    if not isinstance(query, dict) or query.get("text") != expected_query:
        raise RuntimeError("result query differs from submitted query")
    candidates = result.get("candidates")
    top15 = result.get("top15")
    summary = result.get("summary")
    if not isinstance(candidates, list) or len(candidates) != expected_candidates:
        raise RuntimeError(
            f"expected {expected_candidates} candidates, got "
            f"{len(candidates) if isinstance(candidates, list) else 'invalid'}"
        )
    if not isinstance(top15, list) or len(top15) != 15:
        raise RuntimeError("result does not contain a complete TOP-15")
    if not isinstance(summary, dict):
        raise RuntimeError("result summary is missing")
    counts = summary.get("status_counts")
    if (
        not isinstance(counts, dict)
        or not set(counts).issubset({"main", "watchlist", "excluded"})
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in counts.values()
        )
    ):
        raise RuntimeError("result status counts are incomplete")
    if sum(counts.values()) != len(candidates):
        raise RuntimeError("result status counts do not add up to candidate count")
    if any(
        row.get("status") not in {"main", "watchlist"} for row in top15 if isinstance(row, dict)
    ):
        raise RuntimeError("TOP-15 contains an excluded candidate")


def _verify_existing(
    base_url: str, job_id: str, expected_query: str, expected_candidates: int
) -> None:
    job = _request(base_url, f"/api/analyses/{job_id}")
    if job.get("status") != "complete" or job.get("result_available") is not True:
        raise RuntimeError(f"persisted job {job_id} is not complete")
    result = _request(base_url, f"/api/analyses/{job_id}/result")
    _validate_result(result, expected_query, expected_candidates)


def _run(args: argparse.Namespace) -> str:
    if args.job_id:
        _verify_existing(args.base_url, args.job_id, args.query, args.expected_candidates)
        return args.job_id

    created = _request(
        args.base_url,
        "/api/analyses",
        method="POST",
        payload={"query": args.query},
        expected_status=201,
    )
    job_id = created.get("id")
    if not isinstance(job_id, str) or created.get("status") != "pending":
        raise RuntimeError("POST /api/analyses returned an invalid job")
    job = _wait_job(args.base_url, job_id, args.timeout)
    if job.get("status") != "complete":
        raise RuntimeError(f"analysis job failed: {job.get('error')}")
    _verify_existing(args.base_url, job_id, args.query, args.expected_candidates)

    if not args.skip_negative:
        rejected = _request(
            args.base_url,
            "/api/analyses",
            method="POST",
            payload={"query": "Технологии квантовой связи"},
            expected_status=201,
        )
        rejected_id = rejected.get("id")
        if not isinstance(rejected_id, str):
            raise RuntimeError("negative rehearsal returned an invalid job ID")
        rejected_job = _wait_job(args.base_url, rejected_id, args.timeout)
        if rejected_job.get("status") != "error" or rejected_job.get("result_available"):
            raise RuntimeError("foreign query received a substituted result")
        _request(
            args.base_url,
            f"/api/analyses/{rejected_id}/result",
            expected_status=409,
        )
    return job_id


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--expected-candidates", type=int, default=120)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--job-id",
        help="проверить существующий job после restart вместо создания нового",
    )
    parser.add_argument("--skip-negative", action="store_true")
    return parser


def main() -> int:
    try:
        job_id = _run(_parser().parse_args())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Web rehearsal failed: {error}", file=sys.stderr)
        return 1
    print(f"Web rehearsal passed. Persisted job: {job_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
