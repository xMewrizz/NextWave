from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from nextwave.evaluation.feature_table import FEATURE_TABLE_VERSION
from nextwave.evaluation.model import (
    ERRORS_FILENAME,
    MODEL_REPORT_VERSION,
    _assign_folds,
    _fit_vectorizer,
    build_model_report,
    export_model_report,
)

_FEATURE_KEYS = (
    "scientific_coverage_complete",
    "industry_coverage_complete",
    "scientific_previous_present",
    "scientific_recent_present",
    "industry_previous_present",
    "industry_recent_present",
    "unknown_date_present",
    "trust_ab_present",
    "multiple_origins_present",
    "multiple_organizations_present",
    "multiple_source_types_present",
    "temporal_count_coverage_complete",
)
_NUMERIC_KEYS = (
    "scientific_previous_count_log1p",
    "scientific_recent_count_log1p",
    "scientific_count_log_growth",
    "scientific_scope_share_previous",
    "scientific_scope_share_recent",
    "scientific_scope_share_delta",
)


def _digest(payload: bytes) -> dict[str, object]:
    return {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _rows() -> list[dict]:
    domains = ("Edge", "Защита ИИ", "Индустриальный ИИ", "Инфраструктура ИИ", "Роботы")
    rows: list[dict] = []
    for index in range(20):
        target = int(index < 10)
        features = {key: bool((index + offset) % 3) for offset, key in enumerate(_FEATURE_KEYS)}
        features.update({key: float(index + offset) for offset, key in enumerate(_NUMERIC_KEYS)})
        rows.append(
            {
                "schema_version": FEATURE_TABLE_VERSION,
                "candidate_id": f"candidate-{index:03d}",
                "canonical_name": f"{'Emerging' if target else 'Established'} method {index}",
                "aliases": [],
                "domain": domains[index % len(domains)],
                "analysis_scope_key": "scope-v1",
                "cutoff_date": "2026-09-15",
                "target": target,
                "negative_class": None if target else ("mature" if index % 2 else "hype"),
                "group_id": f"group-{index:03d}",
                "identity_reviewed": False,
                "model_text": (
                    f"shared technology "
                    f"{'novel signal' if target else 'stable market'} method {index}"
                ),
                "features": features,
            }
        )
    return rows


def _write_feature_bundle(directory: Path, rows: list[dict]) -> None:
    directory.mkdir()
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode()
    (directory / "features.jsonl").write_bytes(payload)
    manifest = {
        "schema_version": FEATURE_TABLE_VERSION,
        "release_status": "development_only",
        "qualification_eligible": False,
        "counts": {"feature_rows": len(rows)},
        "outputs": {"features.jsonl": _digest(payload)},
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class EvaluationModelTests(unittest.TestCase):
    def test_grouped_folds_keep_identity_together_and_balance_classes(self) -> None:
        rows = _rows()
        rows[1]["group_id"] = rows[0]["group_id"]
        assignments = _assign_folds(rows)
        self.assertEqual(assignments[rows[0]["group_id"]], assignments[rows[1]["group_id"]])
        per_fold = {fold: {0: 0, 1: 0} for fold in range(5)}
        for row in rows:
            per_fold[assignments[row["group_id"]]][row["target"]] += 1
        for counts in per_fold.values():
            self.assertGreater(counts[0], 0)
            self.assertGreater(counts[1], 0)
        self.assertLessEqual(max(sum(value.values()) for value in per_fold.values()), 5)

    def test_vectorizer_is_fitted_only_from_training_rows(self) -> None:
        train = _rows()[:10]
        held_out = {**_rows()[10], "model_text": "secret_holdout_token secret_holdout_token"}
        vectorizer = _fit_vectorizer(train, include_text=True, include_media=True)
        self.assertNotIn("secret", vectorizer.vocabulary)
        self.assertNotIn("holdout", vectorizer.vocabulary)
        self.assertNotIn("token", vectorizer.vocabulary)
        self.assertNotIn("secret::holdout", vectorizer.vocabulary)
        self.assertEqual(held_out["target"], 0)

    def test_permuting_feature_rows_keeps_predictions_and_model_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first"
            second = root / "second"
            rows = _rows()
            _write_feature_bundle(first, rows)
            _write_feature_bundle(second, list(reversed(rows)))
            first_predictions, first_errors, first_report, first_model = build_model_report(first)
            second_predictions, second_errors, second_report, second_model = build_model_report(
                second
            )
        self.assertEqual(first_predictions, second_predictions)
        self.assertEqual(first_errors, second_errors)
        first_report_value = json.loads(first_report)
        second_report_value = json.loads(second_report)
        first_model_value = json.loads(first_model)
        second_model_value = json.loads(second_model)
        self.assertNotEqual(
            first_report_value["feature_manifest"],
            second_report_value["feature_manifest"],
        )
        first_model_value.pop("feature_manifest")
        second_model_value.pop("feature_manifest")
        for key in ("variants", "subgroups", "leave_one_domain_out"):
            self.assertEqual(first_report_value[key], second_report_value[key])
        self.assertEqual(first_model_value, second_model_value)

    def test_report_is_explicitly_development_only_and_has_all_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            feature_dir = Path(tmp) / "features"
            _write_feature_bundle(feature_dir, _rows())
            predictions, errors, report_bytes, model_bytes = build_model_report(feature_dir)
        report = json.loads(report_bytes)
        model = json.loads(model_bytes)
        self.assertEqual(report["schema_version"], MODEL_REPORT_VERSION)
        self.assertEqual(report["evaluation_status"], "development_only")
        self.assertFalse(report["qualification_eligible"])
        self.assertEqual(report["threshold_policy"], "fixed-0.5-no-test-fold-tuning")
        self.assertEqual(
            set(report["variants"]),
            {
                "majority",
                "rule",
                "logreg_structured_only",
                "logreg_text_structured",
                "logreg_without_media",
            },
        )
        self.assertEqual(model["threshold"], 0.5)
        self.assertEqual(len(predictions.decode().splitlines()), 20)
        error_rows = [json.loads(line) for line in errors.decode().splitlines()]
        prediction_rows = [json.loads(line) for line in predictions.decode().splitlines()]
        expected_errors = [
            row for row in prediction_rows if row["target"] != row["prediction"]
        ]
        self.assertEqual(
            [row["candidate_id"] for row in error_rows],
            [row["candidate_id"] for row in expected_errors],
        )
        self.assertEqual(report["error_count"], len(error_rows))
        self.assertEqual(
            report["organizer_recall"],
            report["variants"]["logreg_text_structured"]["oof"]["recall"],
        )
        for row in error_rows:
            self.assertEqual(
                row["error_type"],
                "false_positive" if row["target"] == 0 else "false_negative",
            )
        self.assertEqual(report["outputs"][ERRORS_FILENAME], _digest(errors))

    def test_report_has_explicit_false_positive_rates_for_negative_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            feature_dir = Path(tmp) / "features"
            rows = _rows()
            for row in rows:
                if row["negative_class"] == "hype":
                    row["negative_class"] = "marketing_hype"
            _write_feature_bundle(feature_dir, rows)
            _, _, report_bytes, _ = build_model_report(feature_dir)
        report = json.loads(report_bytes)
        rates = report["negative_false_positive_rate"]
        self.assertEqual(set(rates), {"mature", "marketing_hype"})
        for negative_class, rate in rates.items():
            subgroup = report["subgroups"]["negative_class"][negative_class]
            confusion = subgroup["confusion_matrix"]
            expected = confusion["fp"] / (confusion["tn"] + confusion["fp"])
            self.assertEqual(rate, expected)
            self.assertEqual(subgroup["false_positive_rate"], expected)

    def test_manifest_checksum_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            feature_dir = Path(tmp) / "features"
            _write_feature_bundle(feature_dir, _rows())
            with (feature_dir / "features.jsonl").open("ab") as stream:
                stream.write(b"{}\n")
            with self.assertRaisesRegex(ValueError, "diverges"):
                build_model_report(feature_dir)

    def test_export_is_atomic_and_refuses_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_dir = root / "features"
            output_dir = root / "report"
            _write_feature_bundle(feature_dir, _rows())
            paths = export_model_report(feature_dir=feature_dir, output_dir=output_dir)
            self.assertTrue(paths.report.is_file())
            self.assertIsNotNone(paths.errors)
            self.assertTrue(paths.errors.is_file())
            with self.assertRaisesRegex(ValueError, "already exists"):
                export_model_report(feature_dir=feature_dir, output_dir=output_dir)


if __name__ == "__main__":
    unittest.main()
