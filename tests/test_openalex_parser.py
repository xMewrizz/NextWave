from __future__ import annotations

import json
import unittest
from datetime import UTC, date, datetime

from nextwave.contracts import SourceType, TrustTier
from nextwave.sources import parse_openalex_response, reconstruct_openalex_abstract

RETRIEVED_AT = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
CUTOFF_DATE = date(2026, 9, 15)


def make_work(**overrides):
    work = {
        "id": "https://openalex.org/W123456789",
        "doi": "https://doi.org/10.1234/EXAMPLE.1",
        "title": "Speculative decoding for efficient language models",
        "display_name": "Speculative decoding for efficient language models",
        "publication_date": "2026-08-10",
        "language": "en",
        "type": "article",
        "authorships": [
            {
                "author": {"display_name": "Ada Lovelace"},
                "institutions": [{"display_name": "Example University"}],
            },
            {
                "author": {"display_name": "Alan Turing"},
                "institutions": [{"display_name": "Example University"}],
            },
        ],
        "abstract_inverted_index": {
            "A": [0],
            "faster": [2],
            "method": [1],
        },
        "primary_location": {
            "landing_page_url": "https://publisher.example/paper",
            "source": {"display_name": "Journal of Examples"},
        },
    }
    work.update(overrides)
    return work


def topic(
    topic_id: str = "T12345",
    name: str = "Speculative Decoding for Language Models",
    score: float = 0.91,
):
    return {
        "id": f"https://openalex.org/{topic_id}",
        "display_name": name,
        "score": score,
        "subfield": {
            "id": "https://openalex.org/subfields/1702",
            "display_name": "Artificial Intelligence",
        },
        "field": {
            "id": "https://openalex.org/fields/17",
            "display_name": "Computer Science",
        },
        "domain": {
            "id": "https://openalex.org/domains/3",
            "display_name": "Physical Sciences",
        },
    }


def keyword(
    keyword_id: str = "speculative-decoding",
    name: str = "Speculative Decoding",
    score: float = 0.88,
):
    return {
        "id": f"https://openalex.org/keywords/{keyword_id}",
        "display_name": name,
        "score": score,
    }


def parse(*records):
    return parse_openalex_response(
        json.dumps({"results": list(records)}).encode("utf-8"),
        snapshot_id="snapshot-ai-001",
        retrieved_at=RETRIEVED_AT,
        cutoff_date=CUTOFF_DATE,
    )


class OpenAlexParserTests(unittest.TestCase):
    def test_normalizes_complete_work(self) -> None:
        result = parse(make_work())

        self.assertEqual(result.total_records, 1)
        self.assertEqual(result.accepted_records, 1)
        self.assertEqual(result.rejected_records, 0)
        document = result.documents[0]
        self.assertEqual(document.external_id, "W123456789")
        self.assertEqual(document.doi, "10.1234/example.1")
        self.assertEqual(document.url, "https://publisher.example/paper")
        self.assertEqual(document.canonical_url, "https://doi.org/10.1234/example.1")
        self.assertEqual(document.origin_id, "doi:10.1234/example.1")
        self.assertEqual(document.origin_method, "doi")
        self.assertEqual(document.authors, ("Ada Lovelace", "Alan Turing"))
        self.assertEqual(document.organizations, ("Example University",))
        self.assertEqual(document.publisher, "Journal of Examples")
        self.assertEqual(document.excerpt, "A method faster")
        self.assertEqual(document.source_type, SourceType.SCIENTIFIC_PUBLICATION)
        self.assertEqual(document.trust_tier, TrustTier.A)

    def test_falls_back_to_work_url_without_doi_or_landing_page(self) -> None:
        result = parse(make_work(doi=None, primary_location=None))

        document = result.documents[0]
        self.assertEqual(document.url, "https://openalex.org/W123456789")
        self.assertEqual(document.canonical_url, "https://openalex.org/W123456789")
        self.assertEqual(
            document.origin_id,
            "url:https://openalex.org/w123456789",
        )
        self.assertEqual(document.origin_method, "canonical_url")

    def test_rejects_future_record_without_losing_valid_record(self) -> None:
        result = parse(
            make_work(),
            make_work(
                id="https://openalex.org/W987654321",
                publication_date="2026-09-16",
            ),
        )

        self.assertEqual(result.accepted_records, 1)
        self.assertEqual(result.rejected_records, 1)
        self.assertEqual(result.issues[0].external_id, "W987654321")
        self.assertEqual(result.issues[0].code, "after_cutoff")

    def test_rejects_malformed_record_without_losing_valid_record(self) -> None:
        result = parse(make_work(), {"title": "No identifier"})

        self.assertEqual(result.accepted_records, 1)
        self.assertEqual(result.rejected_records, 1)
        self.assertEqual(result.issues[0].record_index, 1)
        self.assertEqual(result.issues[0].code, "invalid_identifier")

    def test_rejects_invalid_response_envelope(self) -> None:
        with self.assertRaisesRegex(ValueError, "results list"):
            parse_openalex_response(
                b'{"meta": {"count": 1}}',
                snapshot_id="snapshot-ai-001",
                retrieved_at=RETRIEVED_AT,
                cutoff_date=CUTOFF_DATE,
            )

    def test_reconstructs_abstract_by_positions(self) -> None:
        abstract = reconstruct_openalex_abstract(
            {"signal": [3], "Weak": [0], "technology": [1], "is": [2]}
        )

        self.assertEqual(abstract, "Weak technology is signal")

    def test_document_id_is_stable_inside_snapshot(self) -> None:
        first = parse(make_work()).documents[0]
        second = parse(make_work()).documents[0]

        self.assertEqual(first.document_id, second.document_id)

    def test_preserves_scored_topics_and_keywords_as_discovery_hints(self) -> None:
        topic_record = topic()
        result = parse(
            make_work(
                primary_topic=topic_record,
                topics=[topic_record, topic("T54321", "Efficient LLM Inference", 0.72)],
                keywords=[keyword(), keyword("draft-model", "Draft Model", 0.77)],
            )
        )

        hints = result.hints[0]
        self.assertEqual(hints.document_id, result.documents[0].document_id)
        self.assertEqual(hints.topics[0].topic_id, "T12345")
        self.assertTrue(hints.topics[0].primary)
        self.assertEqual(hints.topics[0].subfield_id, "1702")
        self.assertEqual(hints.keywords[0].keyword_id, "speculative-decoding")
        self.assertEqual(hints.keywords[0].score, 0.88)
        self.assertEqual(result.hint_issues, ())

    def test_bad_hints_do_not_discard_an_otherwise_valid_document(self) -> None:
        result = parse(make_work(topics="not-a-list", keywords=[]))

        self.assertEqual(result.accepted_records, 1)
        self.assertEqual(result.rejected_records, 0)
        self.assertEqual(result.hints[0].topics, ())
        self.assertEqual(len(result.hint_issues), 1)
        self.assertIn("topics must be a list", result.hint_issues[0].message)


if __name__ == "__main__":
    unittest.main()
