from __future__ import annotations

import json
import unittest
from datetime import UTC, date, datetime

from nextwave.contracts import SourceType
from nextwave.sources import parse_exa_response

NOW = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
CUTOFF = date(2026, 9, 15)


def result(**overrides):
    value = {
        "id": "https://example.com/news/1",
        "url": "https://Example.com/news/1?utm_source=x&id=2",
        "title": "AI inference accelerator enters production",
        "publishedDate": "2026-09-10T08:30:00.000Z",
        "author": "Ada Example",
        "highlights": ["The accelerator is now deployed in two data centers."],
    }
    value.update(overrides)
    return value


def parse(*records):
    return parse_exa_response(
        json.dumps({"results": list(records), "requestId": "remote"}).encode(),
        snapshot_id="snapshot-exa-001",
        retrieved_at=NOW,
        cutoff_date=CUTOFF,
    )


class ExaParserTests(unittest.TestCase):
    def test_normalizes_news_and_keeps_highlight_as_excerpt(self) -> None:
        document = parse(result()).documents[0]
        self.assertEqual(document.connector_id, "exa")
        self.assertIs(document.source_type, SourceType.INDUSTRY_MEDIA)
        self.assertEqual(document.published_at, date(2026, 9, 10))
        self.assertEqual(document.publisher, "example.com")
        self.assertEqual(document.authors, ("Ada Example",))
        self.assertEqual(
            document.excerpt,
            "The accelerator is now deployed in two data centers.",
        )
        self.assertNotIn("utm_source", document.canonical_url)

    def test_missing_highlights_is_valid_but_not_evidence_text(self) -> None:
        document = parse(result(highlights=None)).documents[0]
        self.assertIsNone(document.excerpt)

    def test_future_and_invalid_rows_are_isolated(self) -> None:
        parsed = parse(
            result(),
            result(id="future", publishedDate="2026-09-16T00:00:00Z"),
            result(id="broken", title=""),
        )
        self.assertEqual(parsed.accepted_records, 1)
        self.assertEqual(parsed.rejected_records, 2)
        self.assertEqual([issue.code for issue in parsed.issues], ["after_cutoff", "missing_title"])

    def test_invalid_envelope_fails_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "results list"):
            parse_exa_response(
                b'{"requestId":"remote"}',
                snapshot_id="snapshot-exa-001",
                retrieved_at=NOW,
                cutoff_date=CUTOFF,
            )


if __name__ == "__main__":
    unittest.main()
