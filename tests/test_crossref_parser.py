from __future__ import annotations

import json
import unittest
from datetime import UTC, date, datetime

from nextwave.contracts import SourceType, TrustTier
from nextwave.sources import parse_crossref_response, parse_openalex_work

RETRIEVED_AT = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
CUTOFF_DATE = date(2026, 9, 15)


def make_work(**overrides):
    work = {
        "DOI": "10.1234/EXAMPLE.1",
        "title": ["Speculative decoding for efficient language models"],
        "URL": "https://doi.org/10.1234/EXAMPLE.1",
        "published": {"date-parts": [[2026, 8, 10]]},
        "language": "en",
        "type": "journal-article",
        "author": [
            {
                "given": "Ada",
                "family": "Lovelace",
                "affiliation": [{"name": "Example University"}],
            },
            {
                "given": "Alan",
                "family": "Turing",
                "affiliation": [{"name": "Example University"}],
            },
        ],
        "publisher": "Example Publisher",
        "abstract": "<jats:p>A &amp; B <jats:bold>method</jats:bold>.</jats:p>",
    }
    work.update(overrides)
    return work


def parse(*records):
    return parse_crossref_response(
        json.dumps({"message": {"items": list(records)}}).encode("utf-8"),
        snapshot_id="snapshot-ai-001",
        retrieved_at=RETRIEVED_AT,
        cutoff_date=CUTOFF_DATE,
    )


class CrossrefParserTests(unittest.TestCase):
    def test_normalizes_complete_work(self) -> None:
        result = parse(make_work())

        self.assertEqual(result.total_records, 1)
        self.assertEqual(result.accepted_records, 1)
        self.assertEqual(result.rejected_records, 0)
        document = result.documents[0]
        self.assertEqual(document.external_id, "10.1234/example.1")
        self.assertEqual(document.doi, "10.1234/example.1")
        self.assertEqual(document.canonical_url, "https://doi.org/10.1234/example.1")
        self.assertEqual(document.origin_id, "doi:10.1234/example.1")
        self.assertEqual(document.authors, ("Ada Lovelace", "Alan Turing"))
        self.assertEqual(document.organizations, ("Example University",))
        self.assertEqual(document.publisher, "Example Publisher")
        self.assertEqual(document.excerpt, "A & B method .")
        self.assertEqual(document.source_type, SourceType.SCIENTIFIC_PUBLICATION)
        self.assertEqual(document.trust_tier, TrustTier.A)

    def test_openalex_and_crossref_share_origin_id_for_same_doi(self) -> None:
        crossref_document = parse(make_work()).documents[0]
        openalex_document = parse_openalex_work(
            {
                "id": "https://openalex.org/W123456789",
                "doi": "https://doi.org/10.1234/EXAMPLE.1",
                "title": "Speculative decoding",
                "publication_date": "2026-08-10",
                "language": "en",
                "type": "article",
            },
            snapshot_id="snapshot-ai-001",
            retrieved_at=RETRIEVED_AT,
            cutoff_date=CUTOFF_DATE,
        )

        self.assertEqual(crossref_document.origin_id, openalex_document.origin_id)
        self.assertNotEqual(crossref_document.document_id, openalex_document.document_id)

    def test_partial_publication_date_remains_unknown(self) -> None:
        document = parse(make_work(published={"date-parts": [[2026]]})).documents[0]

        self.assertIsNone(document.published_at)

    def test_rejects_future_record_without_losing_valid_record(self) -> None:
        result = parse(
            make_work(),
            make_work(DOI="10.1234/future", published={"date-parts": [[2026, 9, 16]]}),
        )

        self.assertEqual(result.accepted_records, 1)
        self.assertEqual(result.rejected_records, 1)
        self.assertEqual(result.issues[0].code, "after_cutoff")

    def test_rejects_record_without_doi(self) -> None:
        result = parse(make_work(DOI=None))

        self.assertEqual(result.accepted_records, 0)
        self.assertEqual(result.issues[0].code, "invalid_identifier")

    def test_rejects_invalid_response_envelope(self) -> None:
        with self.assertRaisesRegex(ValueError, "message object"):
            parse_crossref_response(
                b'{"status":"ok"}',
                snapshot_id="snapshot-ai-001",
                retrieved_at=RETRIEVED_AT,
                cutoff_date=CUTOFF_DATE,
            )


if __name__ == "__main__":
    unittest.main()
