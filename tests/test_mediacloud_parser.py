from __future__ import annotations

import json
import unittest
from datetime import UTC, date, datetime

from nextwave.contracts import SourceType, TrustTier
from nextwave.sources import (
    parse_mediacloud_response,
    parse_mediacloud_timeline_response,
)

RETRIEVED_AT = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)
CUTOFF_DATE = date(2026, 9, 15)


def make_story(**overrides):
    story = {
        "id": "5fb4938303e2421217382d14511aa7fda4b1a425b3e5e21ae532788894e57575",
        "indexed_date": "2026-09-14 10:30:00+00:00",
        "publish_date": "2026-09-14",
        "language": "en",
        "media_name": "Tech Example",
        "media_url": "tech.example",
        "title": "Bank tests a new AI inference accelerator",
        "url": "HTTPS://Tech.Example/ai/pilot/?utm_source=newsletter&id=42#section",
    }
    story.update(overrides)
    return story


def parse(*records, pagination_token="next-page"):
    return parse_mediacloud_response(
        json.dumps(
            {"stories": list(records), "pagination_token": pagination_token}
        ).encode(),
        snapshot_id="snapshot-ai-001",
        retrieved_at=RETRIEVED_AT,
        cutoff_date=CUTOFF_DATE,
    )


class MediaCloudParserTests(unittest.TestCase):
    def test_normalizes_story_as_unverified_information_source(self) -> None:
        result = parse(make_story())

        self.assertEqual(result.total_records, 1)
        self.assertEqual(result.pagination_token, "next-page")
        document = result.documents[0]
        self.assertEqual(document.connector_id, "mediacloud")
        self.assertEqual(document.canonical_url, "https://tech.example/ai/pilot?id=42")
        self.assertEqual(document.origin_id, "url:https://tech.example/ai/pilot?id=42")
        self.assertEqual(document.published_at, date(2026, 9, 14))
        self.assertEqual(document.observed_at, datetime(2026, 9, 14, 10, 30, tzinfo=UTC))
        self.assertEqual(document.publisher, "tech example")
        self.assertEqual(document.language, "en")
        self.assertIs(document.source_type, SourceType.OTHER)
        self.assertIs(document.trust_tier, TrustTier.UNKNOWN)
        self.assertIsNone(document.excerpt)

    def test_missing_publish_date_remains_unknown(self) -> None:
        document = parse(make_story(publish_date=None)).documents[0]

        self.assertIsNone(document.published_at)

    def test_rejects_future_story_without_losing_valid_story(self) -> None:
        result = parse(
            make_story(),
            make_story(id="future", publish_date="2026-09-16"),
        )

        self.assertEqual(result.accepted_records, 1)
        self.assertEqual(result.rejected_records, 1)
        self.assertEqual(result.issues[0].code, "after_cutoff")

    def test_rejects_missing_title(self) -> None:
        result = parse(make_story(title=""))

        self.assertEqual(result.accepted_records, 0)
        self.assertEqual(result.issues[0].code, "missing_title")

    def test_rejects_invalid_response_envelope(self) -> None:
        with self.assertRaisesRegex(ValueError, "stories list"):
            parse_mediacloud_response(
                b'{"status":"ok"}',
                snapshot_id="snapshot-ai-001",
                retrieved_at=RETRIEVED_AT,
                cutoff_date=CUTOFF_DATE,
            )


class MediaCloudTimelineParserTests(unittest.TestCase):
    def test_calculates_weighted_share(self) -> None:
        payload = json.dumps(
            {
                "count_over_time": {
                    "counts": [
                        {"date": "2026-09-14", "count": 10, "total_count": 1000},
                        {"date": "2026-09-15", "count": 30, "total_count": 3000},
                    ]
                }
            }
        ).encode()

        result = parse_mediacloud_timeline_response(
            payload,
            cutoff_date=CUTOFF_DATE,
        )

        self.assertEqual(result.matched_articles, 40)
        self.assertEqual(result.monitored_articles, 4000)
        self.assertEqual(result.share, 0.01)
        self.assertEqual(result.points[0].share, 0.01)

    def test_rejects_impossible_count(self) -> None:
        payload = json.dumps(
            {
                "count_over_time": {
                    "counts": [
                        {"date": "2026-09-14", "count": 101, "total_count": 100}
                    ]
                }
            }
        ).encode()

        with self.assertRaisesRegex(ValueError, "must not exceed"):
            parse_mediacloud_timeline_response(payload, cutoff_date=CUTOFF_DATE)

    def test_rejects_interval_after_cutoff(self) -> None:
        payload = json.dumps(
            {
                "count_over_time": {
                    "counts": [
                        {"date": "2026-09-16", "count": 1, "total_count": 100}
                    ]
                }
            }
        ).encode()

        with self.assertRaisesRegex(ValueError, "after cutoff_date"):
            parse_mediacloud_timeline_response(payload, cutoff_date=CUTOFF_DATE)


if __name__ == "__main__":
    unittest.main()
