from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nextwave.evaluation.analysis_result import (
    ANALYSIS_RESULT_VERSION,
    _deduplicate_results,
    _obvious_identity_key,
    build_analysis_result,
    export_analysis_result,
)
from nextwave.evaluation.feature_table import _digest
from nextwave.labeling.evidence_llm_run import LABELING_EVIDENCE_LLM_RESULT_VERSION


def _json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()


def _jsonl(rows: list[dict[str, object]]) -> bytes:
    return b"".join(_json(row) for row in rows)


class AnalysisResultTests(unittest.TestCase):
    def test_obvious_identity_key_collapses_acronym_and_plural_variants(self) -> None:
        self.assertEqual(
            _obvious_identity_key("Digital Twins (DTw)"),
            _obvious_identity_key("digital twin"),
        )
        self.assertEqual(
            _obvious_identity_key("Large Language Models (LLMs)"),
            _obvious_identity_key("large-language-model"),
        )
        self.assertNotEqual(
            _obvious_identity_key("RAG"),
            _obvious_identity_key("Retrieval-Augmented Generation"),
        )

    def test_final_dedup_keeps_one_card_and_records_primary(self) -> None:
        def row(candidate_id: str, name: str, status: str, score: float) -> dict:
            return {
                "candidate_id": candidate_id,
                "canonical_name": name,
                "aliases": [],
                "status": status,
                "reason": "passed" if status == "main" else "insufficient_origins",
                "reason_ru": "reason",
                "model": {"score": score},
                "evidence_review": {"independent_origins": 2 if status == "main" else 1},
            }

        rows = [
            row("candidate-a", "Digital Twins", "watchlist", 0.3),
            row("candidate-b", "Digital Twins (DTw)", "main", 0.7),
        ]
        _deduplicate_results(rows)
        by_id = {item["candidate_id"]: item for item in rows}
        self.assertEqual(by_id["candidate-b"]["duplicate_candidate_ids"], ["candidate-a"])
        self.assertEqual(by_id["candidate-a"]["reason"], "duplicate")
        self.assertEqual(by_id["candidate-a"]["duplicate_of"], "candidate-b")
        self.assertEqual(by_id["candidate-a"]["duplicate_of_name"], "Digital Twins (DTw)")

    def test_final_dedup_does_not_hide_maturity_evidence(self) -> None:
        rows = [
            {
                "candidate_id": "candidate-main",
                "canonical_name": "Reinforcement Learnings",
                "aliases": [],
                "status": "main",
                "reason": "passed",
                "reason_ru": "passed",
                "model": {"score": 0.9},
                "evidence_review": {"independent_origins": 3},
            },
            {
                "candidate_id": "candidate-mature",
                "canonical_name": "reinforcement learning",
                "aliases": [],
                "status": "excluded",
                "reason": "mature",
                "reason_ru": "mature",
                "model": {"score": 0.2},
                "evidence_review": {"independent_origins": 2},
            },
        ]
        _deduplicate_results(rows)
        by_id = {item["candidate_id"]: item for item in rows}
        self.assertEqual(by_id["candidate-mature"]["reason"], "mature")
        self.assertEqual(by_id["candidate-main"]["reason"], "duplicate")

    def _fixture(self, root: Path) -> dict[str, object]:
        candidate_ids = [
            "candidate-main",
            "candidate-mature",
            "candidate-hype",
            "candidate-pending",
        ]
        plan_dir = root / "plan"
        plan_dir.mkdir()
        candidates = [
            {
                "candidate_id": candidate_id,
                "canonical_name": candidate_id,
                "aliases": [],
                "domain": "Инфраструктура ИИ",
                "origin": {
                    "bundle_id": "bundle-1",
                    "group_id": candidate_id,
                    "source_query": "Инфраструктура ИИ",
                },
            }
            for candidate_id in candidate_ids
        ]
        for candidate in candidates:
            candidate["cutoff_date"] = "2026-09-15"
        plan = _json(
            {
                "bundle": {"bundle_id": "bundle-1"},
                "cutoff_date": "2026-09-15",
                "candidates": candidates,
            }
        )
        (plan_dir / "plan.json").write_bytes(plan)
        plan_manifest = {
            "schema_version": "labeling-enrichment-plan-v2",
            "bundle_id": "bundle-1",
            "outputs": {"plan.json": _digest(plan)},
        }
        (plan_dir / "manifest.json").write_bytes(_json(plan_manifest))

        combined_dir = root / "combined"
        combined_dir.mkdir()
        documents: list[dict[str, object]] = []
        for candidate_id in candidate_ids:
            for index in (1, 2):
                documents.append(
                    {
                        "candidate_id": candidate_id,
                        "document_id": f"{candidate_id}-doc-{index}",
                        "connector_id": "openalex",
                        "title": f"Source {index}",
                        "url": f"https://example.com/{candidate_id}/{index}",
                        "excerpt": f"verbatim evidence {candidate_id} {index}",
                        "origin_id": f"{candidate_id}-origin-{index}",
                        "organizations": [f"Actor {index}"],
                        "publisher": f"Publisher {index}",
                        "trust_tier": "A" if index == 1 else "B",
                    }
                )
        document_bytes = _jsonl(documents)
        (combined_dir / "documents.jsonl").write_bytes(document_bytes)
        combined_manifest = {
            "schema_version": "analysis-combined-enrichment-result-v1",
            "bundle_id": "bundle-1",
            "plan": _digest(plan),
            "outputs": {"documents.jsonl": _digest(document_bytes)},
        }
        (combined_dir / "manifest.json").write_bytes(_json(combined_manifest))

        feature_dir = root / "features"
        feature_dir.mkdir()
        feature_rows = [
            {
                "candidate_id": candidate_id,
                "cutoff_date": "2026-09-15",
                "features": {"temporal_count_coverage_complete": True},
            }
            for candidate_id in candidate_ids
        ]
        feature_bytes = _jsonl(feature_rows)
        (feature_dir / "features.jsonl").write_bytes(feature_bytes)
        feature_manifest = {
            "schema_version": "analysis-feature-table-v1",
            "outputs": {"features.jsonl": _digest(feature_bytes)},
        }
        feature_manifest_bytes = _json(feature_manifest)
        (feature_dir / "manifest.json").write_bytes(feature_manifest_bytes)

        inference_dir = root / "inference"
        inference_dir.mkdir()
        prediction_rows = [
            {
                "candidate_id": candidate_id,
                "model_score": 0.8,
                "decision_threshold": 0.5,
                "prediction": 1,
                "explanation": {
                    "top_positive_factors": [{"feature": "recent_growth"}],
                    "top_negative_factors": [],
                },
            }
            for candidate_id in candidate_ids
        ]
        prediction_bytes = _jsonl(prediction_rows)
        (inference_dir / "predictions.jsonl").write_bytes(prediction_bytes)
        inference_manifest = {
            "schema_version": "analysis-inference-v2",
            "release_status": "development_only",
            "inputs": {"features": _digest(feature_manifest_bytes)},
            "outputs": {"predictions.jsonl": _digest(prediction_bytes)},
        }
        (inference_dir / "manifest.json").write_bytes(_json(inference_manifest))

        evidence_dir = root / "evidence"
        evidence_dir.mkdir()
        coverage = [
            {"candidate_id": candidate_id, "status": "complete"}
            for candidate_id in candidate_ids[:-1]
        ] + [{"candidate_id": "candidate-pending", "status": "failed"}]
        claims: list[dict[str, object]] = []
        for index in (1, 2):
            claims.append(self._claim("candidate-main", index, "research"))
            claims.append(self._claim("candidate-mature", index, "research"))
        claims.append(
            self._claim(
                "candidate-mature", 1, "adoption", direction="counter", suffix="m"
            )
        )
        claims.append(self._claim("candidate-hype", 1, "promotional_claim"))
        coverage_bytes = _jsonl(coverage)
        claims_bytes = _jsonl(claims)
        (evidence_dir / "coverage.jsonl").write_bytes(coverage_bytes)
        (evidence_dir / "claims.jsonl").write_bytes(claims_bytes)
        evidence_manifest = {
            "schema_version": LABELING_EVIDENCE_LLM_RESULT_VERSION,
            "bundle_id": "bundle-1",
            "cutoff_date": "2026-09-15",
            "outputs": {
                "coverage.jsonl": _digest(coverage_bytes),
                "claims.jsonl": _digest(claims_bytes),
            },
        }
        (evidence_dir / "manifest.json").write_bytes(_json(evidence_manifest))
        return {
            "plan": plan_dir,
            "combined": combined_dir,
            "features": feature_dir,
            "inference": inference_dir,
            "evidence": evidence_dir,
        }

    @staticmethod
    def _claim(
        candidate_id: str,
        document_index: int,
        kind: str,
        *,
        direction: str = "support",
        suffix: str = "",
    ) -> dict[str, object]:
        return {
            "candidate_id": candidate_id,
            "document_id": f"{candidate_id}-doc-{document_index}",
            "claim_id": f"claim-{candidate_id}-{document_index}-{kind}{suffix}",
            "quote": f"verbatim evidence {candidate_id} {document_index}",
            "scope": "full_candidate",
            "direction": direction,
            "kind": kind,
            "explanation_ru": f"Проверяемое основание: {kind}",
        }

    @staticmethod
    def _build(paths: dict[str, object]) -> dict[str, bytes]:
        return build_analysis_result(
            analysis_plan_dir=paths["plan"],
            combined_result_dir=paths["combined"],
            feature_dir=paths["features"],
            inference_dir=paths["inference"],
            evidence_result_dirs=(paths["evidence"],),
        )

    def test_builds_policy_result_top15_and_uncalibrated_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            files = self._build(self._fixture(Path(tmp)))
        rows = {
            row["candidate_id"]: row
            for row in map(json.loads, files["candidates.jsonl"].splitlines())
        }
        self.assertEqual(rows["candidate-main"]["status"], "main")
        self.assertEqual(rows["candidate-mature"]["reason"], "mature")
        self.assertEqual(rows["candidate-hype"]["reason"], "marketing_hype")
        self.assertEqual(rows["candidate-pending"]["reason"], "evidence_review_incomplete")
        self.assertIsNone(rows["candidate-main"]["model"]["confidence"])
        self.assertEqual(rows["candidate-main"]["top15_rank"], 1)
        summary = json.loads(files["summary.json"])
        unified = json.loads(files["result.json"])
        self.assertFalse(summary["confidence_available"])
        self.assertIsNone(summary["high_confidence_weak_signals"])
        self.assertEqual(summary["main_target"], 15)
        self.assertFalse(summary["main_target_met"])
        self.assertEqual(summary["analysis_status"], "insufficient_main")
        self.assertEqual(unified["status"], "insufficient_main")
        self.assertEqual(unified["summary"], summary)
        self.assertEqual(summary["candidate_gate"]["evaluated_proposals"], 4)
        self.assertEqual(len(unified["candidates"]), 4)
        self.assertEqual(len(unified["top15"]), 1)

    def test_maturity_kind_overrides_counter_direction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = self._build(self._fixture(Path(tmp)))["candidates.jsonl"]
            rows = list(map(json.loads, raw.splitlines()))
        mature = next(row for row in rows if row["candidate_id"] == "candidate-mature")
        self.assertEqual(mature["status"], "excluded")
        self.assertEqual(mature["reason"], "mature")

    def test_promotional_claim_with_independent_ab_support_is_not_hype(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp))
            evidence = paths["evidence"]
            claim_lines = (evidence / "claims.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            claims = [json.loads(line) for line in claim_lines]
            claims.extend(self._claim("candidate-hype", index, "research") for index in (1, 2))
            raw = _jsonl(claims)
            (evidence / "claims.jsonl").write_bytes(raw)
            manifest = json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))
            manifest["outputs"]["claims.jsonl"] = _digest(raw)
            (evidence / "manifest.json").write_bytes(_json(manifest))
            rows = list(map(json.loads, self._build(paths)["candidates.jsonl"].splitlines()))
        hype = next(row for row in rows if row["candidate_id"] == "candidate-hype")
        self.assertEqual(hype["status"], "main")

    def test_mirrors_of_same_work_are_one_origin_and_one_actor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp))
            combined = paths["combined"]
            document_path = combined / "documents.jsonl"
            documents = [
                json.loads(line)
                for line in document_path.read_text(encoding="utf-8").splitlines()
            ]
            for document in documents:
                if document["candidate_id"] == "candidate-main":
                    document["title"] = (
                        "The same artificial intelligence research work"
                        if document["document_id"].endswith("-1")
                        else "The same artificial intelligence research work for Journal Example"
                    )
                    document["organizations"] = ["The same laboratory"]
                    document["publisher"] = f"Repository {document['document_id']}"
            raw = _jsonl(documents)
            document_path.write_bytes(raw)
            manifest_path = combined / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["outputs"]["documents.jsonl"] = _digest(raw)
            manifest_path.write_bytes(_json(manifest))

            rows = list(map(json.loads, self._build(paths)["candidates.jsonl"].splitlines()))

        candidate = next(row for row in rows if row["candidate_id"] == "candidate-main")
        self.assertEqual(candidate["evidence_review"]["independent_origins"], 1)
        self.assertEqual(candidate["evidence_review"]["independent_actors"], 1)
        self.assertEqual(candidate["status"], "watchlist")
        self.assertEqual(candidate["reason"], "insufficient_origins")

    def test_rejects_non_verbatim_claim_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp))
            evidence = paths["evidence"]
            claim_lines = (evidence / "claims.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            claims = [json.loads(line) for line in claim_lines]
            claims[0]["quote"] = "invented quote"
            raw = _jsonl(claims)
            (evidence / "claims.jsonl").write_bytes(raw)
            manifest = json.loads((evidence / "manifest.json").read_text(encoding="utf-8"))
            manifest["outputs"]["claims.jsonl"] = _digest(raw)
            (evidence / "manifest.json").write_bytes(_json(manifest))
            output = Path(tmp) / "output"
            with self.assertRaisesRegex(ValueError, "not verbatim"):
                export_analysis_result(
                    analysis_plan_dir=paths["plan"],
                    combined_result_dir=paths["combined"],
                    feature_dir=paths["features"],
                    inference_dir=paths["inference"],
                    evidence_result_dirs=(paths["evidence"],),
                    output_dir=output,
                )
            self.assertFalse(output.exists())

    def test_rejects_duplicate_evidence_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, "invalid candidate or status"):
                build_analysis_result(
                    analysis_plan_dir=paths["plan"],
                    combined_result_dir=paths["combined"],
                    feature_dir=paths["features"],
                    inference_dir=paths["inference"],
                    evidence_result_dirs=(paths["evidence"], paths["evidence"]),
                )

    def test_output_is_byte_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp))
            first = self._build(paths)
            second = self._build(paths)
        self.assertEqual(first, second)
        self.assertEqual(
            json.loads(first["manifest.json"])["schema_version"],
            ANALYSIS_RESULT_VERSION,
        )

    def test_historical_cutoff_comes_from_plan_and_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._fixture(Path(tmp))
            plan_path = paths["plan"] / "plan.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["cutoff_date"] = "2025-09-15"
            for candidate in plan["candidates"]:
                candidate["cutoff_date"] = "2025-09-15"
            plan_raw = _json(plan)
            plan_path.write_bytes(plan_raw)
            plan_manifest_path = paths["plan"] / "manifest.json"
            plan_manifest = json.loads(
                plan_manifest_path.read_text(encoding="utf-8")
            )
            plan_manifest["outputs"]["plan.json"] = _digest(plan_raw)
            plan_manifest_path.write_bytes(_json(plan_manifest))

            combined_manifest_path = paths["combined"] / "manifest.json"
            combined_manifest = json.loads(
                combined_manifest_path.read_text(encoding="utf-8")
            )
            combined_manifest["plan"] = _digest(plan_raw)
            combined_manifest_path.write_bytes(_json(combined_manifest))

            feature_path = paths["features"] / "features.jsonl"
            feature_rows = [
                json.loads(line)
                for line in feature_path.read_text(encoding="utf-8").splitlines()
            ]
            for row in feature_rows:
                row["cutoff_date"] = "2025-09-15"
            feature_raw = _jsonl(feature_rows)
            feature_path.write_bytes(feature_raw)
            feature_manifest_path = paths["features"] / "manifest.json"
            feature_manifest = json.loads(
                feature_manifest_path.read_text(encoding="utf-8")
            )
            feature_manifest["outputs"]["features.jsonl"] = _digest(feature_raw)
            feature_manifest_raw = _json(feature_manifest)
            feature_manifest_path.write_bytes(feature_manifest_raw)

            inference_manifest_path = paths["inference"] / "manifest.json"
            inference_manifest = json.loads(
                inference_manifest_path.read_text(encoding="utf-8")
            )
            inference_manifest["inputs"]["features"] = _digest(feature_manifest_raw)
            inference_manifest_path.write_bytes(_json(inference_manifest))

            files = self._build(paths)

        manifest = json.loads(files["manifest.json"])
        summary = json.loads(files["summary.json"])
        self.assertEqual(manifest["cutoff_date"], "2025-09-15")
        self.assertEqual(summary["cutoff_date"], "2025-09-15")
        self.assertEqual(len(manifest["limitations"]), 1)
        self.assertFalse(
            manifest["inputs"]["evidence_results"][0][
                "cutoff_matches_analysis"
            ]
        )


if __name__ == "__main__":
    unittest.main()
