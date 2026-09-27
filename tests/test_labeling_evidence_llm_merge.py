from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from nextwave.labeling.evidence_llm_merge import (
    LABELING_EVIDENCE_LLM_MERGED_RESULT_VERSION,
    merge_evidence_llm_results,
)
from nextwave.labeling.evidence_llm_run import (
    LABELING_EVIDENCE_LLM_RESULT_VERSION,
    QUALIFICATION_EVIDENCE_MODEL,
    QUALIFICATION_EVIDENCE_PROVIDER,
)

FILES = (
    "request_results.jsonl",
    "claims.jsonl",
    "document_results.jsonl",
    "issues.jsonl",
    "coverage.jsonl",
)


def render(rows: list[dict]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode()


def digest(payload: bytes) -> dict:
    return {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def request(candidate: str, status: str) -> dict:
    return {
        "attempts": 0 if status == "not_run" else 1,
        "candidate_id": candidate,
        "error": "model call failed" if status == "failed" else None,
        "reused": False,
        "status": status,
        "task_id": f"task-{candidate}",
    }


def coverage(candidate: str, status: str) -> dict:
    if status == "no_input":
        return {
            "candidate_id": candidate,
            "status": "no_input",
            "task_id": None,
            "input_documents": 0,
            "llm_call_planned": False,
        }
    return {
        "candidate_id": candidate,
        "status": "complete" if status == "success" else status,
        "task_id": f"task-{candidate}",
        "input_documents": 1,
        "llm_call_planned": True,
    }


def claim(candidate: str, suffix: str) -> dict:
    return {
        "candidate_id": candidate,
        "task_id": f"task-{candidate}",
        "document_id": f"doc-{candidate}",
        "claim_id": f"claim-{suffix}",
        "scope": "core_only",
    }


def document(candidate: str, claim_count: int) -> dict:
    return {
        "candidate_id": candidate,
        "task_id": f"task-{candidate}",
        "document_id": f"doc-{candidate}",
        "claim_count": claim_count,
        "issue_count": 0,
        "status": "processed",
    }


def issue(candidate: str, code: str) -> dict:
    return {
        "candidate_id": candidate,
        "task_id": f"task-{candidate}",
        "document_id": f"doc-{candidate}",
        "code": code,
        "message": code,
    }


def write_result(
    root: Path,
    name: str,
    *,
    statuses: dict[str, str],
    claims: dict[str, list[dict]],
    retry_filter: list[str] | None,
    model: str = QUALIFICATION_EVIDENCE_MODEL,
) -> Path:
    path = root / name
    path.mkdir()
    planned = [candidate for candidate, status in statuses.items() if status != "no_input"]
    rows = {
        "request_results.jsonl": [request(candidate, statuses[candidate]) for candidate in planned],
        "claims.jsonl": [row for candidate in sorted(claims) for row in claims[candidate]],
        "document_results.jsonl": [
            document(candidate, len(claims.get(candidate, [])))
            for candidate in planned
            if statuses[candidate] in {"success"}
        ],
        "issues.jsonl": [
            issue(candidate, "invalid_response")
            for candidate in planned
            if statuses[candidate] == "failed"
        ],
        "coverage.jsonl": [coverage(candidate, status) for candidate, status in statuses.items()],
    }
    payloads = {filename: render(rows[filename]) for filename in FILES}
    for filename, payload in payloads.items():
        (path / filename).write_bytes(payload)
    manifest = {
        "schema_version": LABELING_EVIDENCE_LLM_RESULT_VERSION,
        "analysis_status": "partial",
        "bundle_id": "bundle-1",
        "cutoff_date": "2026-09-15",
        "plan_manifest": digest(b"plan"),
        "plan_files": {"tasks.jsonl": digest(b"tasks"), "coverage.jsonl": digest(b"coverage")},
        "provider": QUALIFICATION_EVIDENCE_PROVIDER,
        "model": model,
        "extractor_id": "yandex-yandexgpt-pro-5.1-evidence-llm-v10-specific-core",
        "max_output_tokens": 2000,
        "concurrency": 3,
        "applied_candidate_filter": retry_filter,
        "applied_max_new_tasks": None,
        "totals": {
            "called": sum(
                status != "not_run"
                for status in statuses.values()
                if status != "no_input"
            )
        },
        "outputs": {filename: digest(payload) for filename, payload in payloads.items()},
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class EvidenceLlmMergeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.statuses = {"c1": "success", "c2": "success", "c3": "failed", "c4": "no_input"}

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_primary(self) -> Path:
        return write_result(
            self.root,
            "primary",
            statuses=self.statuses,
            claims={"c1": [claim("c1", "primary")]},
            retry_filter=None,
        )

    def test_retry_replaces_only_when_it_adds_claims(self) -> None:
        primary = self.make_primary()
        retry = write_result(
            self.root,
            "retry",
            statuses={"c1": "not_run", "c2": "success", "c3": "failed", "c4": "no_input"},
            claims={"c2": [claim("c2", "retry")]},
            retry_filter=["c2", "c3"],
        )
        paths = merge_evidence_llm_results(
            primary_dir=primary, retry_dir=retry, output_dir=self.root / "out"
        )
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        claims = read_rows(paths.claims)

        self.assertEqual([row["claim_id"] for row in claims], ["claim-primary", "claim-retry"])
        self.assertEqual(manifest["schema_version"], LABELING_EVIDENCE_LLM_MERGED_RESULT_VERSION)
        self.assertEqual(manifest["selected_retry_candidates"], ["c2"])
        self.assertEqual(manifest["totals"]["claims"], 2)
        self.assertEqual(manifest["totals"]["source_calls"], 5)

    def test_retry_zero_does_not_replace_successful_primary_zero(self) -> None:
        primary = self.make_primary()
        retry = write_result(
            self.root,
            "retry",
            statuses={"c1": "not_run", "c2": "success", "c3": "not_run", "c4": "no_input"},
            claims={},
            retry_filter=["c2"],
        )
        paths = merge_evidence_llm_results(
            primary_dir=primary, retry_dir=retry, output_dir=self.root / "out"
        )
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        self.assertEqual(manifest["selected_retry_candidates"], [])

    def test_retry_complete_recovers_failed_primary_even_without_claim(self) -> None:
        primary = self.make_primary()
        retry = write_result(
            self.root,
            "retry",
            statuses={"c1": "not_run", "c2": "not_run", "c3": "success", "c4": "no_input"},
            claims={},
            retry_filter=["c3"],
        )
        paths = merge_evidence_llm_results(
            primary_dir=primary, retry_dir=retry, output_dir=self.root / "out"
        )
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        requests = {row["candidate_id"]: row for row in read_rows(paths.request_results)}
        self.assertEqual(requests["c3"]["status"], "success")
        self.assertEqual(manifest["recovered_failed_candidates"], ["c3"])

    def test_failed_retry_keeps_primary(self) -> None:
        primary = self.make_primary()
        retry = write_result(
            self.root,
            "retry",
            statuses={"c1": "not_run", "c2": "success", "c3": "failed", "c4": "no_input"},
            claims={},
            retry_filter=["c2", "c3"],
        )
        paths = merge_evidence_llm_results(
            primary_dir=primary, retry_dir=retry, output_dir=self.root / "out"
        )
        manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        self.assertEqual(manifest["selected_retry_candidates"], [])
        self.assertEqual(manifest["totals"]["failed"], 1)

    def test_non_qualification_model_is_rejected_without_output(self) -> None:
        primary = self.make_primary()
        retry = write_result(
            self.root,
            "retry",
            statuses={"c1": "not_run", "c2": "success", "c3": "not_run", "c4": "no_input"},
            claims={},
            retry_filter=["c2"],
            model="YandexGPT Lite 5",
        )
        with self.assertRaisesRegex(ValueError, "qualification Evidence model"):
            merge_evidence_llm_results(
                primary_dir=primary, retry_dir=retry, output_dir=self.root / "out"
            )
        self.assertFalse((self.root / "out").exists())

    def test_corrupt_input_is_rejected_without_output(self) -> None:
        primary = self.make_primary()
        retry = write_result(
            self.root,
            "retry",
            statuses={"c1": "not_run", "c2": "success", "c3": "not_run", "c4": "no_input"},
            claims={},
            retry_filter=["c2"],
        )
        with (retry / "claims.jsonl").open("ab") as stream:
            stream.write(b" ")
        with self.assertRaisesRegex(ValueError, "checksum or size mismatch"):
            merge_evidence_llm_results(
                primary_dir=primary, retry_dir=retry, output_dir=self.root / "out"
            )
        self.assertFalse((self.root / "out").exists())

    def test_same_inputs_produce_same_jsonl_bytes(self) -> None:
        primary = self.make_primary()
        retry = write_result(
            self.root,
            "retry",
            statuses={"c1": "not_run", "c2": "success", "c3": "not_run", "c4": "no_input"},
            claims={"c2": [claim("c2", "retry")]},
            retry_filter=["c2"],
        )
        one = merge_evidence_llm_results(
            primary_dir=primary, retry_dir=retry, output_dir=self.root / "one"
        )
        two = merge_evidence_llm_results(
            primary_dir=primary, retry_dir=retry, output_dir=self.root / "two"
        )
        for name in FILES:
            self.assertEqual(
                (one.manifest.parent / name).read_bytes(),
                (two.manifest.parent / name).read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
