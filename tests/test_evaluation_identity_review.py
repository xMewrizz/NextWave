from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from nextwave.evaluation.identity_review import (
    IDENTITY_DECISIONS_VERSION,
    IDENTITY_REVIEW_VERSION,
    build_identity_review,
    export_identity_review,
    normalize_identity,
)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()


def _digest(payload: bytes) -> dict[str, object]:
    return {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _candidate(candidate_id: str, name: str, *, search: str | None = None) -> dict:
    return {
        "candidate_id": candidate_id,
        "canonical_name": name,
        "aliases": [],
        "search_terms": [search or name],
        "domain": "Edge",
    }


def _write_plan(path: Path, candidates: list[dict]) -> None:
    path.mkdir()
    plan = {
        "schema_version": "labeling-enrichment-plan-v2",
        "cutoff_date": "2026-09-15",
        "candidates": candidates,
    }
    payload = _json_bytes(plan)
    (path / "plan.json").write_bytes(payload)
    (path / "manifest.json").write_bytes(
        _json_bytes(
            {
                "schema_version": "labeling-enrichment-plan-v2",
                "bundle_id": f"bundle-{path.name}",
                "outputs": {"plan.json": _digest(payload)},
            }
        )
    )


def _write_decisions(path: Path, pairs: list[dict]) -> None:
    path.write_bytes(
        _json_bytes({"schema_version": IDENTITY_DECISIONS_VERSION, "pairs": pairs})
    )


def _decision(left: str, right: str, relation: str = "same_family") -> dict:
    return {
        "left_candidate_id": left,
        "right_candidate_id": right,
        "relation": relation,
        "rationale": "Пара проверена по полным названиям.",
    }


class IdentityReviewTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, Path]:
        root.mkdir(parents=True, exist_ok=True)
        positive = root / "positive"
        negative = root / "negative"
        decisions = root / "decisions.json"
        _write_plan(
            positive,
            [
                _candidate("organizer-001", "Analog in-memory computing"),
                _candidate("organizer-002", "Satellite inference"),
            ],
        )
        _write_plan(
            negative,
            [
                _candidate("team-negative-001", "in memory computing"),
                _candidate("team-negative-002", "Unrelated technology"),
            ],
        )
        _write_decisions(
            decisions,
            [_decision("organizer-001", "team-negative-001")],
        )
        return {
            "positive_plan_dir": positive,
            "negative_plan_dir": negative,
            "decisions_path": decisions,
        }

    def test_normalization_handles_nfkc_dash_slash_and_underscore(self) -> None:
        self.assertEqual(
            normalize_identity("Ａnalog–in/memory_computing"),
            "analog in memory computing",
        )

    def test_review_builds_shared_family_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identities, pairs, manifest = build_identity_review(**self._fixture(Path(tmp)))
        rows = [json.loads(line) for line in identities.decode().splitlines()]
        by_id = {row["candidate_id"]: row for row in rows}
        summary = json.loads(manifest)
        self.assertEqual(
            by_id["organizer-001"]["group_id"],
            by_id["team-negative-001"]["group_id"],
        )
        self.assertNotEqual(
            by_id["organizer-002"]["group_id"],
            by_id["team-negative-002"]["group_id"],
        )
        self.assertEqual(summary["schema_version"], IDENTITY_REVIEW_VERSION)
        self.assertTrue(summary["ready_for_model"])
        self.assertEqual(json.loads(pairs.decode().splitlines()[0])["trigger"], "fuzzy")

    def test_same_candidate_is_a_blocking_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = self._fixture(Path(tmp))
            _write_decisions(
                kwargs["decisions_path"],
                [_decision("organizer-001", "team-negative-001", "same_candidate")],
            )
            _, _, manifest = build_identity_review(**kwargs)
        summary = json.loads(manifest)
        self.assertFalse(summary["ready_for_model"])
        self.assertEqual(summary["totals"]["conflicts"], 1)

    def test_missing_suggested_pair_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = self._fixture(Path(tmp))
            _write_decisions(kwargs["decisions_path"], [])
            with self.assertRaisesRegex(ValueError, "miss suggested pair"):
                build_identity_review(**kwargs)

    def test_reversed_pair_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = self._fixture(Path(tmp))
            _write_decisions(
                kwargs["decisions_path"],
                [_decision("team-negative-001", "organizer-001")],
            )
            with self.assertRaisesRegex(ValueError, "positive-to-negative"):
                build_identity_review(**kwargs)

    def test_manual_semantic_pair_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = self._fixture(Path(tmp))
            _write_decisions(
                kwargs["decisions_path"],
                [
                    _decision("organizer-001", "team-negative-001"),
                    _decision("organizer-002", "team-negative-002", "distinct"),
                ],
            )
            _, pairs, _ = build_identity_review(**kwargs)
        pair_rows = [json.loads(line) for line in pairs.decode().splitlines()]
        self.assertEqual(pair_rows[1]["trigger"], "manual")

    def test_exact_pair_cannot_be_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            positive, negative, decisions = root / "p", root / "n", root / "d.json"
            _write_plan(positive, [_candidate("organizer-001", "Same Tech")])
            _write_plan(negative, [_candidate("team-negative-001", "same-tech")])
            _write_decisions(
                decisions,
                [_decision("organizer-001", "team-negative-001", "distinct")],
            )
            with self.assertRaisesRegex(ValueError, "cannot be distinct"):
                build_identity_review(
                    positive_plan_dir=positive,
                    negative_plan_dir=negative,
                    decisions_path=decisions,
                )

    def test_generic_single_token_does_not_require_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            positive, negative, decisions = root / "p", root / "n", root / "d.json"
            _write_plan(positive, [_candidate("organizer-001", "AI cooling")])
            _write_plan(negative, [_candidate("team-negative-001", "AI banking")])
            _write_decisions(decisions, [])
            _, pairs, _ = build_identity_review(
                positive_plan_dir=positive,
                negative_plan_dir=negative,
                decisions_path=decisions,
            )
        self.assertEqual(pairs, b"")

    def test_permuted_inputs_and_decisions_are_byte_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self._fixture(root / "first")
            second_root = root / "second"
            second_root.mkdir()
            second = self._fixture(second_root)
            for key in ("positive_plan_dir", "negative_plan_dir"):
                directory = second[key]
                plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
                plan["candidates"].reverse()
                payload = _json_bytes(plan)
                (directory / "plan.json").write_bytes(payload)
                manifest = json.loads((directory / "manifest.json").read_text())
                manifest["outputs"]["plan.json"] = _digest(payload)
                (directory / "manifest.json").write_bytes(_json_bytes(manifest))
            first_result = build_identity_review(**first)
            second_result = build_identity_review(**second)
        self.assertEqual(first_result[:2], second_result[:2])

    def test_checksum_failure_precedes_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kwargs = self._fixture(root)
            with (kwargs["positive_plan_dir"] / "plan.json").open("ab") as stream:
                stream.write(b" ")
            output = root / "output"
            with self.assertRaisesRegex(ValueError, "diverges"):
                export_identity_review(**kwargs, output_dir=output)
            self.assertFalse(output.exists())

    def test_export_refuses_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kwargs = self._fixture(root)
            output = root / "output"
            export_identity_review(**kwargs, output_dir=output)
            with self.assertRaisesRegex(ValueError, "already exists"):
                export_identity_review(**kwargs, output_dir=output)


if __name__ == "__main__":
    unittest.main()
