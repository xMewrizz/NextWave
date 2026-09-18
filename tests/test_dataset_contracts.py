from __future__ import annotations

import unittest
from dataclasses import fields
from datetime import date

from nextwave.datasets import (
    FileDigest,
    HeaderMapping,
    IdentityStatus,
    OrganizerAnnotationRecord,
    OrganizerDatasetManifest,
    PositiveCandidateRecord,
)


def make_candidate(**overrides) -> PositiveCandidateRecord:
    values = {
        "record_id": "organizer-001",
        "source_row": 3,
        "canonical_name": "Спекулятивное декодирование",
        "domain": "Инфраструктура ИИ",
        "analysis_scope_key": "ai-infrastructure-v1",
        "aliases": (),
        "group_id": "organizer-001",
        "identity_status": IdentityStatus.PENDING_REVIEW,
        "cutoff_date": date(2026, 9, 15),
    }
    values.update(overrides)
    return PositiveCandidateRecord(**values)


def make_annotation(**overrides) -> OrganizerAnnotationRecord:
    values = {
        "record_id": "organizer-001",
        "source_number": 1,
        "companies_raw": "Example Research Lab",
        "expert_rationale": "Ограниченное число ранних исследований.",
        "expert_stage": "Прототип",
        "expert_mention_trend": "Рост числа публикаций",
        "expert_score": 5,
        "sources_raw": "https://example.org/paper",
    }
    values.update(overrides)
    return OrganizerAnnotationRecord(**values)


class DatasetContractTests(unittest.TestCase):
    def test_positive_record_serializes_fixed_label_and_iso_date(self) -> None:
        body = make_candidate().to_dict()

        self.assertEqual(body["schema_version"], "organizer-positive-v1")
        self.assertEqual(body["label"], "weak_signal")
        self.assertEqual(body["target"], 1)
        self.assertEqual(body["label_origin"], "organizer_confirmed")
        self.assertEqual(body["cutoff_date"], "2026-09-15")
        self.assertEqual(body["aliases"], [])
        self.assertEqual(
            list(body),
            [
                "schema_version",
                "record_id",
                "source_row",
                "canonical_name",
                "domain",
                "analysis_scope_key",
                "aliases",
                "group_id",
                "identity_status",
                "label",
                "target",
                "label_origin",
                "cutoff_date",
            ],
        )

    def test_positive_record_excludes_expert_fields(self) -> None:
        safe_fields = {item.name for item in fields(PositiveCandidateRecord)}
        forbidden = {
            "companies_raw",
            "expert_rationale",
            "expert_stage",
            "expert_mention_trend",
            "expert_score",
            "sources_raw",
        }

        self.assertTrue(safe_fields.isdisjoint(forbidden))

    def test_record_id_must_match_source_row(self) -> None:
        with self.assertRaisesRegex(ValueError, "match source_row"):
            make_candidate(record_id="organizer-002")

    def test_scope_must_match_domain(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match domain"):
            make_candidate(analysis_scope_key="fintech-v1")

    def test_aliases_are_unique_and_do_not_repeat_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            make_candidate(aliases=("Speculative decoding", "speculative decoding"))
        with self.assertRaisesRegex(ValueError, "canonical_name"):
            make_candidate(aliases=(" спекулятивное декодирование ",))

    def test_annotation_score_is_bounded(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 3 and 7"):
            make_annotation(expert_score=8)

    def test_annotation_links_to_same_source_number(self) -> None:
        with self.assertRaisesRegex(ValueError, "match record_id"):
            make_annotation(source_number=2)

    def test_file_digest_requires_plain_filename_and_sha256(self) -> None:
        with self.assertRaisesRegex(ValueError, "directory"):
            FileDigest(filename="data/output.jsonl", size_bytes=10, sha256="a" * 64)
        with self.assertRaisesRegex(ValueError, "directory"):
            FileDigest(filename=r"data\output.jsonl", size_bytes=10, sha256="a" * 64)
        with self.assertRaisesRegex(ValueError, "64 lowercase"):
            FileDigest(filename="output.jsonl", size_bytes=10, sha256="not-a-hash")

    def test_manifest_counts_must_balance(self) -> None:
        source = FileDigest(filename="organizer.xlsx", size_bytes=100, sha256="a" * 64)
        output = FileDigest(filename="positive.jsonl", size_bytes=200, sha256="b" * 64)

        with self.assertRaisesRegex(ValueError, "add up"):
            OrganizerDatasetManifest(
                dataset_version="organizer-positive-2026-09-15-v1",
                adapter_version="organizer-xlsx-v1",
                cutoff_date=date(2026, 9, 15),
                source=source,
                sheet_name="Слабые сигналы",
                header_row=2,
                data_start_row=3,
                data_end_row=102,
                input_record_count=100,
                accepted_record_count=99,
                rejected_record_count=0,
                header_mapping=(HeaderMapping("Технология", "canonical_name"),),
                outputs=(output,),
            )


if __name__ == "__main__":
    unittest.main()
