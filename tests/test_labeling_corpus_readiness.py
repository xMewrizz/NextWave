from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from nextwave.labeling.corpus_readiness import (
    LABELING_CORPUS_READINESS_VERSION,
    build_corpus_readiness,
    export_corpus_readiness,
)


def _digest(payload: bytes) -> dict[str, Any]:
    return {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _evidence(candidate: dict[str, Any], number: int) -> dict[str, Any]:
    name = candidate["canonical_name"]
    return {
        "schema_version": "negative-corpus-hype-evidence-v1",
        "evidence_id": f"evidence-{candidate['candidate_id']}-{number}",
        "candidate_id": candidate["candidate_id"],
        "canonical_name": name,
        "content_status": "article_text",
        "document_id": f"document-{candidate['candidate_id']}-{number}",
        "fact_kind": "publicity",
        "headline": f"{name} receives attention {number}",
        "locator": f"paragraph-{number}",
        "origin_id": f"origin-{candidate['candidate_id']}-{number}",
        "published_at": "2026-06-01",
        "publisher": f"Publisher {number}",
        "quote": f"{name} appears in this substantive article text {number}.",
        "quote_kind": "article_text",
        "source_class": "industry",
        "trust_tier": "A",
        "url": f"https://example.org/{candidate['candidate_id']}/{number}",
    }


def _fixture(
    root: Path,
    *,
    evidence_change: Any = None,
    queue_change: Any = None,
    adjudication_schema: str = "negative-corpus-adjudication-v6",
    reverse_rows: bool = False,
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    adjudication_dir = root / "adjudication"
    queue_dir = root / "queue"
    adjudication_dir.mkdir()
    queue_dir.mkdir()

    rows: list[dict[str, Any]] = []
    for number in range(57):
        rows.append({
            "schema_version": adjudication_schema,
            "candidate_id": f"candidate-{number:03d}",
            "canonical_name": f"Mature technology {number}",
            "domain": "Edge",
            "decision_status": "accepted",
            "proposed_label": "mature",
            "audit_status": "reviewed",
            "evidence_status": "verified",
        })
    for number in range(7):
        rows.append({
            "schema_version": adjudication_schema,
            "candidate_id": f"hype-accepted-{number:02d}",
            "canonical_name": f"Hype technology {number}",
            "domain": "AI",
            "decision_status": "accepted",
            "proposed_label": "hype",
            "audit_status": "reviewed",
            "evidence_status": "verified",
        })
    for number in range(36):
        rows.append({
            "schema_version": adjudication_schema,
            "candidate_id": f"hype-proposed-{number:02d}",
            "canonical_name": f"Proposed technology {number}",
            "domain": "AI",
            "decision_status": "proposed",
            "proposed_label": "hype",
            "audit_status": "pending",
            "evidence_status": "pending",
        })
    if reverse_rows:
        rows.reverse()

    evidence: list[dict[str, Any]] = []
    for row in rows:
        if row["decision_status"] == "accepted" and row["proposed_label"] == "hype":
            evidence.extend(_evidence(row, number) for number in range(3))
    if evidence_change is not None:
        evidence_change(evidence, rows)

    adjudication_payload = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")
    evidence_payload = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in evidence
    ).encode("utf-8")
    (adjudication_dir / "adjudication.jsonl").write_bytes(adjudication_payload)
    (adjudication_dir / "hype_evidence.jsonl").write_bytes(evidence_payload)
    hype_rows = sorted(
        (row for row in rows if row["proposed_label"] == "hype"),
        key=lambda row: row["candidate_id"],
    )
    replacement_rows = sorted(
        (row for row in rows if row["proposed_label"] == "mature"),
        key=lambda row: row["candidate_id"],
    )[:7]
    queue_rows = [
        {
            "schema_version": "negative-hype-review-item-v1",
            "candidate_id": row["candidate_id"],
            "canonical_name": row["canonical_name"],
            "domain": row["domain"],
            "exact_title_groups": 0,
            "has_three_title_groups_two_hosts": False,
            "media": [],
            "pilot_ab_origin_ids": [],
            "publisher_hosts": 0,
            "retrieved_recent_media": 0,
            "review_status": "replace_or_targeted_search",
            "search_coverage_complete": True,
            "technical_ab_origin_ids": [],
        }
        for row in [*hype_rows, *replacement_rows]
    ]
    if queue_change is not None:
        queue_change(queue_rows)
    queue_payload = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in queue_rows
    ).encode("utf-8")
    (queue_dir / "review_queue.jsonl").write_bytes(queue_payload)

    _write_json(
        adjudication_dir / "manifest.json",
        {
            "schema_version": "negative-corpus-adjudication-manifest-v6",
            "cutoff_date": "2026-09-15",
            "counts": {
                "rows": 100,
                "accepted": 64,
                "accepted_mature": 57,
                "accepted_hype": 7,
                "proposed_hype": 36,
                "mature_pool": 57,
                "hype_pool": 43,
                "hype_evidence": len(evidence),
            },
            "inputs": {"source.jsonl": _digest(b"synthetic input\n")},
            "outputs": {
                "adjudication.jsonl": _digest(adjudication_payload),
                "hype_evidence.jsonl": _digest(evidence_payload),
            },
        },
    )
    _write_json(
        queue_dir / "manifest.json",
        {
            "schema_version": "negative-hype-review-queue-manifest-v1",
            "cutoff_date": "2026-09-15",
            "recent_window_from": "2025-09-15",
            "counts": {
                "candidates": 50,
                "needs_manual_publicity_review": 0,
                "replace_or_targeted_search": 50,
                "with_retrieved_recent_media": 0,
                "with_three_title_groups_two_hosts": 0,
            },
            "inputs": {},
            "outputs": {"review_queue.jsonl": _digest(queue_payload)},
        },
    )
    return adjudication_dir, queue_dir


def _statuses(payload: bytes) -> dict[str, dict[str, Any]]:
    return {
        row["candidate_id"]: row
        for row in (json.loads(line) for line in payload.decode().splitlines())
    }


def _hype_status(adjudication: Path, queue: Path) -> dict[str, Any]:
    payload = build_corpus_readiness(
        adjudication_dir=adjudication, review_queue_dir=queue
    )[0]
    return _statuses(payload)["hype-accepted-00"]


class CorpusReadinessTests(unittest.TestCase):
    def test_hype_pool_distinguishes_accepted_and_proposed_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adjudication, queue = _fixture(Path(directory))
            result = build_corpus_readiness(
                adjudication_dir=adjudication, review_queue_dir=queue
            )
            counts = json.loads(result[1])["counts"]
            self.assertEqual(counts["accepted_hype"], 7)
            self.assertEqual(counts["proposed_hype"], 36)

    def test_proposed_hype_is_not_label_ready_even_with_valid_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adjudication, queue = _fixture(Path(directory))
            result = build_corpus_readiness(adjudication_dir=adjudication, review_queue_dir=queue)
            status = _statuses(result[0])["hype-proposed-00"]
            self.assertFalse(status["label_ready"])
            self.assertEqual(status["evidence_contract"]["reason"], "proposed_not_accepted")

    def test_accepted_hype_without_evidence_is_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adjudication, queue = _fixture(
                Path(directory),
                evidence_change=lambda evidence, rows: evidence.clear(),
            )
            status = _hype_status(adjudication, queue)
            self.assertFalse(status["label_ready"])
            self.assertEqual(status["evidence_contract"]["reason"], "insufficient_rows")

    def test_future_published_at_blocks_hype(self) -> None:
        def change(evidence: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
            evidence[0] = {**evidence[0], "published_at": "2026-09-16"}

        with tempfile.TemporaryDirectory() as directory:
            adjudication, queue = _fixture(
                Path(directory), evidence_change=change
            )
            status = _hype_status(adjudication, queue)
            self.assertFalse(status["label_ready"])
            self.assertEqual(status["evidence_contract"]["reason"], "bad_date")

    def test_single_origin_blocks_hype(self) -> None:
        def change(evidence: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
            for row in evidence[:3]:
                row["origin_id"] = "one-origin"

        with tempfile.TemporaryDirectory() as directory:
            adjudication, queue = _fixture(Path(directory), evidence_change=change)
            status = _hype_status(adjudication, queue)
            self.assertFalse(status["label_ready"])
            self.assertEqual(status["evidence_contract"]["reason"], "single_origin")

    def test_casefolded_single_publisher_blocks_hype(self) -> None:
        def change(evidence: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
            evidence[0]["publisher"], evidence[1]["publisher"], evidence[2]["publisher"] = (
                "Publisher", "publisher", "PUBLISHER"
            )

        with tempfile.TemporaryDirectory() as directory:
            adjudication, queue = _fixture(Path(directory), evidence_change=change)
            status = _hype_status(adjudication, queue)
            self.assertFalse(status["label_ready"])
            self.assertEqual(status["evidence_contract"]["reason"], "single_publisher")

    def test_foreign_candidate_evidence_does_not_count(self) -> None:
        def change(evidence: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
            evidence[0]["candidate_id"] = "other-candidate"

        with tempfile.TemporaryDirectory() as directory:
            adjudication, queue = _fixture(Path(directory), evidence_change=change)
            status = _hype_status(adjudication, queue)
            self.assertFalse(status["label_ready"])
            self.assertEqual(status["evidence_contract"]["rows_total"], 2)
            self.assertEqual(status["evidence_contract"]["reason"], "insufficient_rows")

    def test_non_http_url_blocks_hype(self) -> None:
        def change(evidence: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
            evidence[0]["url"] = "ftp://example.org/article"

        with tempfile.TemporaryDirectory() as directory:
            adjudication, queue = _fixture(Path(directory), evidence_change=change)
            status = _hype_status(adjudication, queue)
            self.assertFalse(status["label_ready"])
            self.assertEqual(status["evidence_contract"]["reason"], "bad_url")

    def test_empty_url_blocks_hype(self) -> None:
        def change(evidence: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
            evidence[0]["url"] = ""

        with tempfile.TemporaryDirectory() as directory:
            adjudication, queue = _fixture(Path(directory), evidence_change=change)
            with self.assertRaisesRegex(ValueError, "hype evidence row needs non-empty url"):
                build_corpus_readiness(adjudication_dir=adjudication, review_queue_dir=queue)

    def test_three_title_only_rows_have_separate_headline_count(self) -> None:
        def change(evidence: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
            for row in evidence[:3]:
                row["content_status"] = "title_only"

        with tempfile.TemporaryDirectory() as directory:
            adjudication, queue = _fixture(Path(directory), evidence_change=change)
            contract = _hype_status(adjudication, queue)["evidence_contract"]
            self.assertFalse(contract["reason"] is None)
            self.assertEqual(contract["headline_only_rows"], 3)
            self.assertEqual(contract["substantive_rows"], 0)
            self.assertEqual(contract["reason"], "no_substantive_evidence")
            self.assertEqual(contract["reasons"], ["no_substantive_evidence"])

    def test_valid_mix_is_label_ready(self) -> None:
        def change(evidence: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
            evidence[1]["content_status"] = "title_only"
            evidence[2]["content_status"] = "title_only"

        with tempfile.TemporaryDirectory() as directory:
            adjudication, queue = _fixture(Path(directory), evidence_change=change)
            status = _hype_status(adjudication, queue)
            self.assertTrue(status["label_ready"])
            self.assertEqual(status["evidence_contract"]["substantive_rows"], 1)
            self.assertEqual(status["evidence_contract"]["headline_only_rows"], 2)

    def test_downloaded_page_with_headline_quote_is_not_substantive(self) -> None:
        def change(evidence: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
            for row in evidence[:3]:
                row["content_status"] = "article_text"
                row["quote_kind"] = "verbatim_headline"
                row["locator"] = "headline"

        with tempfile.TemporaryDirectory() as directory:
            adjudication, queue = _fixture(Path(directory), evidence_change=change)
            contract = _hype_status(adjudication, queue)["evidence_contract"]
            self.assertEqual(contract["headline_only_rows"], 3)
            self.assertEqual(contract["substantive_rows"], 0)
            self.assertEqual(contract["reason"], "no_substantive_evidence")

    def test_review_queue_must_cover_every_hype_candidate(self) -> None:
        def change(queue_rows: list[dict[str, Any]]) -> None:
            queue_rows.pop(0)
            queue_rows.append({**queue_rows[-1], "candidate_id": "replacement-extra"})

        with tempfile.TemporaryDirectory() as directory:
            adjudication, queue = _fixture(Path(directory), queue_change=change)
            with self.assertRaisesRegex(ValueError, "review queue misses hype candidates"):
                build_corpus_readiness(adjudication_dir=adjudication, review_queue_dir=queue)

    def test_tampered_checksum_fails_before_output_directory_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adjudication, queue = _fixture(Path(directory))
            payload = adjudication / "adjudication.jsonl"
            payload.write_bytes(payload.read_bytes() + b"tampered\n")
            output = Path(directory) / "output"
            with self.assertRaisesRegex(ValueError, "adjudication.jsonl digest mismatch"):
                export_corpus_readiness(
                    adjudication_dir=adjudication, review_queue_dir=queue, output_dir=output
                )
            self.assertFalse(output.exists())

    def test_wrong_adjudication_schema_fails_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adjudication, queue = _fixture(Path(directory), adjudication_schema="wrong-v0")
            output = Path(directory) / "output"
            with self.assertRaisesRegex(
                ValueError, "adjudication row has the wrong schema_version"
            ):
                export_corpus_readiness(
                    adjudication_dir=adjudication, review_queue_dir=queue, output_dir=output
                )
            self.assertFalse(output.exists())

    def test_manifest_count_mismatch_fails_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adjudication, queue = _fixture(Path(directory))
            manifest_path = queue / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["counts"]["candidates"] = 49
            _write_json(manifest_path, manifest)
            output = Path(directory) / "output"
            with self.assertRaisesRegex(ValueError, "count candidates does not match"):
                export_corpus_readiness(
                    adjudication_dir=adjudication,
                    review_queue_dir=queue,
                    output_dir=output,
                )
            self.assertFalse(output.exists())

    def test_existing_output_directory_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adjudication, queue = _fixture(Path(directory))
            output = Path(directory) / "output"
            output.mkdir()
            marker = output / "marker"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "output directory already exists"):
                export_corpus_readiness(
                    adjudication_dir=adjudication, review_queue_dir=queue, output_dir=output
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_row_permutation_has_identical_candidate_status_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = _fixture(Path(directory) / "first")
            second = _fixture(Path(directory) / "second", reverse_rows=True)
            first_result = build_corpus_readiness(
                adjudication_dir=first[0], review_queue_dir=first[1]
            )
            second_result = build_corpus_readiness(
                adjudication_dir=second[0], review_queue_dir=second[1]
            )
            self.assertEqual(first_result[0], second_result[0])

    def test_small_fixture_not_ready_and_expected_block_present(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adjudication, queue = _fixture(Path(directory))
            candidate_status, deficits, manifest = build_corpus_readiness(
                adjudication_dir=adjudication, review_queue_dir=queue
            )
            self.assertFalse(json.loads(deficits)["ready_for_finalization"])
            self.assertEqual(
                json.loads(deficits)["expected"],
                {
                    "accepted_mature": 57,
                    "accepted_hype": 7,
                    "proposed_hype": 36,
                },
            )
            manifest_version = json.loads(manifest)["schema_version"]
            self.assertEqual(manifest_version, LABELING_CORPUS_READINESS_VERSION)
            self.assertEqual(len(candidate_status.splitlines()), 100)

    def test_export_publishes_manifest_status_and_deficits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adjudication, queue = _fixture(Path(directory))
            paths = export_corpus_readiness(
                adjudication_dir=adjudication,
                review_queue_dir=queue,
                output_dir=Path(directory) / "output",
            )
            self.assertTrue(paths.manifest.is_file())
            self.assertTrue(paths.candidate_status.is_file())
            self.assertTrue(paths.deficits.is_file())


if __name__ == "__main__":
    unittest.main()
