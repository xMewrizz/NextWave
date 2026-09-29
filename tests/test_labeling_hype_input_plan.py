from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from nextwave.discovery.candidate_gate import QUALIFICATION_GATE_ID
from nextwave.labeling.hype_input_plan import build_hype_evidence_input

BUNDLE = "bundle-test"


def dump_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def dump_jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()


def digest(payload: bytes) -> dict[str, object]:
    return {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def write_artifact(
    directory: Path,
    version: str,
    files: dict[str, bytes],
    *,
    extra_manifest: dict | None = None,
) -> dict:
    directory.mkdir()
    for name, payload in files.items():
        (directory / name).write_bytes(payload)
    manifest = {
        "schema_version": version,
        "bundle_id": BUNDLE,
        "outputs": {name: digest(payload) for name, payload in files.items()},
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    (directory / "manifest.json").write_bytes(dump_json(manifest))
    return manifest


def science(candidate_id: str, number: int) -> dict:
    return {
        "candidate_id": candidate_id,
        "document_id": f"science-{candidate_id}-{number}",
        "connector": "openalex",
        "source_class": "scientific",
        "final_rank": number,
        "title": "Research title",
        "url": f"https://example.org/science/{candidate_id}/{number}",
        "origin_id": f"doi:10.1/{candidate_id}/{number}",
        "snapshot_id": f"snapshot-{number}",
        "published_at": "2026-01-01",
        "trust_tier": "A",
        "matched_term": "alpha technology",
        "relevance_score": 70,
        "relevance_class": "strong",
        "evidence_text_available": True,
        "excerpt": "Alpha technology is evaluated in a controlled study.",
    }


def media_rank(candidate_id: str, number: int, *, strong: bool = True) -> dict:
    url = f"https://news.example/{candidate_id}/{number}"
    return {
        "candidate_id": candidate_id,
        "document_id": f"media-{candidate_id}-{number}",
        "selection_rank": number,
        "final_class": "strong" if strong else "weak",
        "final_score": 70 if strong else 40,
        "matched_term": "alpha technology",
        "published_at": "2026-02-01",
        "url": url,
        "origin_id": f"url:{url}",
        "identity": url,
    }


def media_document(candidate_id: str, number: int) -> dict:
    url = f"https://news.example/{candidate_id}/{number}"
    return {
        "candidate_id": candidate_id,
        "document_id": f"media-{candidate_id}-{number}",
        "evidence_text_available": True,
        "excerpt": "Alpha technology received a public launch announcement.",
        "title": "Alpha technology launch",
        "url": url,
        "canonical_url": url,
        "origin_id": f"url:{url}",
        "snapshot_id": f"media-snapshot-{number}",
        "published_at": "2026-02-01",
        "trust_tier": "unknown",
        "publisher": f"publisher-{number}.example",
        "content_status": "article_text",
        "extraction_method": "html.article_paragraphs",
    }


def build_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    candidates = [
        {
            "candidate_id": "accepted",
            "canonical_name": "Alpha technology",
            "domain": "Edge",
            "cutoff_date": "2026-09-15",
        },
        {
            "candidate_id": "rejected",
            "canonical_name": "Beta technology",
            "domain": "Edge",
            "cutoff_date": "2026-09-15",
        },
        {
            "candidate_id": "short",
            "canonical_name": "Gamma technology",
            "domain": "Edge",
            "cutoff_date": "2026-09-15",
        },
    ]
    plan_payload = dump_json({"cutoff_date": "2026-09-15", "candidates": candidates})
    plan = root / "plan"
    write_artifact(
        plan,
        "labeling-enrichment-plan-v2",
        {"plan.json": plan_payload},
        extra_manifest={"candidate_count": 3},
    )

    result_manifest_digest = digest(b"result-manifest")
    gate_rows = []
    for candidate in candidates:
        accepted = candidate["candidate_id"] != "rejected"
        gate_rows.append({
            "schema_version": "labeling-target-gate-v1",
            "candidate_id": candidate["candidate_id"],
            "canonical_name": candidate["canonical_name"],
            "domain": candidate["domain"],
            "canonical_grounded": accepted,
            "gate_called": accepted,
            "gate_id": QUALIFICATION_GATE_ID if accepted else None,
            "gate_decision": "accept" if accepted else "not_called",
            "promotion_status": "evidence_review_required" if accepted else "not_grounded",
        })
    gate = root / "gate"
    write_artifact(
        gate,
        "labeling-target-gate-v1",
        {"gate_results.jsonl": dump_jsonl(gate_rows)},
        extra_manifest={
            "cutoff_date": "2026-09-15",
            "gate_id": QUALIFICATION_GATE_ID,
            "inputs": {
                "plan.json": digest(plan_payload),
                "enrichment_manifest.json": result_manifest_digest,
            }
        },
    )

    media_rows = [media_document("accepted", number) for number in range(1, 5)]
    media_rows.extend(media_document("short", number) for number in range(1, 3))
    media = root / "media"
    write_artifact(
        media,
        "labeling-media-fetch-result-v1",
        {"enriched_documents.jsonl": dump_jsonl(media_rows)},
        extra_manifest={"inputs": {"result_manifest": result_manifest_digest}},
    )

    science_rows = [science("accepted", number) for number in range(1, 5)]
    science_rows.append(science("rejected", 1))
    science_rows.append(science("short", 1))
    ranked_rows = [media_rank("accepted", number) for number in range(1, 4)]
    ranked_rows.append(media_rank("accepted", 4, strong=False))
    ranked_rows.extend(media_rank("short", number) for number in range(1, 3))
    evidence = root / "evidence"
    write_artifact(
        evidence,
        "labeling-evidence-input-plan-v4",
        {
            "evidence_input_documents.jsonl": dump_jsonl(science_rows),
            "media_reranked.jsonl": dump_jsonl(ranked_rows),
        },
        extra_manifest={
            "cutoff_date": "2026-09-15",
            "inputs": {
                "plan_file": digest(plan_payload),
                "plan_manifest": digest((plan / "manifest.json").read_bytes()),
                "media_manifest": digest((media / "manifest.json").read_bytes()),
            },
        },
    )
    return plan, gate, evidence, media


def read_rows(payload: bytes) -> list[dict]:
    return [json.loads(line) for line in payload.decode().splitlines()]


class HypeEvidenceInputTests(unittest.TestCase):
    def test_selects_three_media_and_uses_remaining_science_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            paths = build_fixture(Path(temp))
            documents, coverage, manifest = build_hype_evidence_input(
                plan_dir=paths[0],
                gate_dir=paths[1],
                evidence_input_dir=paths[2],
                media_result_dir=paths[3],
            )

        rows = read_rows(documents)
        accepted = [row for row in rows if row["candidate_id"] == "accepted"]
        self.assertEqual(len(accepted), 6)
        self.assertEqual([row["final_rank"] for row in accepted], list(range(1, 7)))
        self.assertEqual(
            [row["source_class"] for row in accepted],
            ["industry", "industry", "industry", "scientific", "scientific", "scientific"],
        )
        coverage_by_id = {row["candidate_id"]: row for row in read_rows(coverage)}
        self.assertFalse(coverage_by_id["rejected"]["has_evidence_input"])
        self.assertEqual(
            coverage_by_id["rejected"]["empty_reasons"], ["target_gate:not_grounded"]
        )
        self.assertFalse(coverage_by_id["short"]["has_evidence_input"])
        self.assertEqual(
            coverage_by_id["short"]["empty_reasons"],
            ["fewer_than_three_strong_recent_media_documents"],
        )
        manifest_value = json.loads(manifest)
        self.assertEqual(manifest_value["totals"]["candidates_without_final_input"], 2)

    def test_four_media_leave_two_scientific_slots(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, gate, evidence, media = build_fixture(root)
            evidence_manifest = json.loads((evidence / "manifest.json").read_text())
            ranked = [media_rank("accepted", number) for number in range(1, 5)]
            payload = dump_jsonl(ranked)
            (evidence / "media_reranked.jsonl").write_bytes(payload)
            evidence_manifest["outputs"]["media_reranked.jsonl"] = digest(payload)
            (evidence / "manifest.json").write_bytes(dump_json(evidence_manifest))

            documents, _, _ = build_hype_evidence_input(
                plan_dir=plan,
                gate_dir=gate,
                evidence_input_dir=evidence,
                media_result_dir=media,
            )

        accepted = [
            row for row in read_rows(documents) if row["candidate_id"] == "accepted"
        ]
        self.assertEqual(
            [row["source_class"] for row in accepted],
            ["industry", "industry", "industry", "industry", "scientific", "scientific"],
        )

    def test_gate_must_be_bound_to_exact_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            plan, gate, evidence, media = build_fixture(Path(temp))
            manifest = json.loads((gate / "manifest.json").read_text())
            manifest["inputs"]["plan.json"] = digest(b"different")
            (gate / "manifest.json").write_bytes(dump_json(manifest))

            with self.assertRaisesRegex(ValueError, "not bound"):
                build_hype_evidence_input(
                    plan_dir=plan,
                    gate_dir=gate,
                    evidence_input_dir=evidence,
                    media_result_dir=media,
                )


if __name__ == "__main__":
    unittest.main()
