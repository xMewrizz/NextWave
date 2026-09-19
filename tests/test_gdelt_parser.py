from __future__ import annotations

import json
import unittest
from datetime import UTC, date, datetime

from nextwave.contracts import SourceType, TrustTier
from nextwave.sources import (
    canonicalize_article_url,
    parse_gdelt_response,
    parse_gdelt_timeline_response,
)

RETRIEVED_AT = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
CUTOFF_DATE = date(2026, 9, 15)


def make_article(**overrides):
    article = {
        "url": "HTTPS://News.Example/ai/pilot/?utm_source=newsletter&id=42#section",
        "url_mobile": "",
        "title": "Bank tests a new AI inference accelerator",
        "seendate": "20260914T103000Z",
        "socialimage": "https://news.example/image.jpg",
        "domain": "News.Example",
        "language": "English",
        "sourcecountry": "United States",
    }
    article.update(overrides)
    return article


def parse(*records):
    return parse_gdelt_response(
        json.dumps({"articles": list(records)}).encode("utf-8"),
        snapshot_id="snapshot-ai-001",
        retrieved_at=RETRIEVED_AT,
        cutoff_date=CUTOFF_DATE,
    )


class GdeltParserTests(unittest.TestCase):
    def test_normalizes_article_as_unverified_information_source(self) -> None:
        result = parse(make_article())

        self.assertEqual(result.total_records, 1)
        self.assertEqual(result.accepted_records, 1)
        document = result.documents[0]
        self.assertEqual(document.canonical_url, "https://news.example/ai/pilot?id=42")
        self.assertEqual(document.origin_id, "url:https://news.example/ai/pilot?id=42")
        self.assertIsNone(document.published_at)
        self.assertEqual(document.observed_at, datetime(2026, 9, 14, 10, 30, tzinfo=UTC))
        self.assertEqual(document.publisher, "news.example")
        self.assertEqual(document.language, "en")
        self.assertEqual(document.source_type, SourceType.OTHER)
        self.assertEqual(document.trust_tier, TrustTier.UNKNOWN)

    def test_tracking_parameters_do_not_create_new_origin(self) -> None:
        first = canonicalize_article_url("https://example.org/story?id=7&utm_source=a")
        second = canonicalize_article_url("https://EXAMPLE.org/story/?utm_medium=b&id=7#top")

        self.assertEqual(first, second)

    def test_rejects_article_observed_after_cutoff(self) -> None:
        result = parse(make_article(seendate="20260916T000000Z"))

        self.assertEqual(result.accepted_records, 0)
        self.assertEqual(result.rejected_records, 1)
        self.assertEqual(result.issues[0].code, "after_cutoff")

    def test_rejects_missing_title_without_losing_valid_article(self) -> None:
        result = parse(make_article(), make_article(title=""))

        self.assertEqual(result.accepted_records, 1)
        self.assertEqual(result.rejected_records, 1)
        self.assertEqual(result.issues[0].code, "missing_title")

    def test_rejects_invalid_response_envelope(self) -> None:
        with self.assertRaisesRegex(ValueError, "articles list"):
            parse_gdelt_response(
                b'{"status":"ok"}',
                snapshot_id="snapshot-ai-001",
                retrieved_at=RETRIEVED_AT,
                cutoff_date=CUTOFF_DATE,
            )


class GdeltTimelineParserTests(unittest.TestCase):
    def test_calculates_weighted_share_from_raw_volume(self) -> None:
        payload = json.dumps(
            {
                "timeline": [
                    {
                        "series": "Volume Intensity",
                        "data": [
                            {"date": "20260914T000000Z", "value": 10, "norm": 1000},
                            {"date": "20260915T000000Z", "value": 30, "norm": 3000},
                        ],
                    }
                ]
            }
        ).encode()

        result = parse_gdelt_timeline_response(payload, cutoff_date=CUTOFF_DATE)

        self.assertEqual(result.matched_articles, 40)
        self.assertEqual(result.monitored_articles, 4000)
        self.assertEqual(result.share, 0.01)
        self.assertEqual(result.points[0].share, 0.01)

    def test_empty_timeline_is_a_valid_zero_point_result(self) -> None:
        result = parse_gdelt_timeline_response(
            b'{"timeline":[]}',
            cutoff_date=CUTOFF_DATE,
        )

        self.assertEqual(result.points, ())
        self.assertIsNone(result.share)

    def test_rejects_impossible_timeline_count(self) -> None:
        payload = json.dumps(
            {
                "timeline": [
                    {
                        "data": [
                            {"date": "20260914T000000Z", "value": 101, "norm": 100}
                        ]
                    }
                ]
            }
        ).encode()

        with self.assertRaisesRegex(ValueError, "must not exceed"):
            parse_gdelt_timeline_response(payload, cutoff_date=CUTOFF_DATE)

    def test_rejects_timeline_point_after_cutoff(self) -> None:
        payload = json.dumps(
            {
                "timeline": [
                    {
                        "data": [
                            {"date": "20260916T000000Z", "value": 1, "norm": 100}
                        ]
                    }
                ]
            }
        ).encode()

        with self.assertRaisesRegex(ValueError, "after cutoff_date"):
            parse_gdelt_timeline_response(payload, cutoff_date=CUTOFF_DATE)


if __name__ == "__main__":
    unittest.main()
