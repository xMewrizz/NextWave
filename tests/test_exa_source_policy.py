from __future__ import annotations

import unittest

from nextwave.evaluation.exa_source_policy import classify_exa_source


class ExaSourcePolicyTests(unittest.TestCase):
    def test_excludes_academic_repositories_and_journal_articles(self):
        for url in (
            "https://arxiv.org/abs/1234.5678",
            "https://exa.ai/library/publication/abc",
            "https://www.nature.com/articles/s41586-024-08549-9",
            "https://dl.acm.org/doi/10.1145/example",
        ):
            with self.subTest(url=url):
                decision = classify_exa_source(url)
                self.assertFalse(decision.eligible_for_industry)
                self.assertEqual(decision.source_type, "scientific_publication")

    def test_keeps_nature_news_as_media(self):
        decision = classify_exa_source("https://www.nature.com/news/example")
        self.assertTrue(decision.eligible_for_industry)
        self.assertEqual(decision.source_type, "industry_media")

    def test_types_press_release_company_and_social_sources(self):
        cases = {
            "https://www.prnewswire.com/news-releases/example": "press_release",
            "https://developer.nvidia.com/blog/example": "company_technical",
            "https://www.linkedin.com/posts/example": "social_or_blog",
            "https://www.reuters.com/technology/example": "industry_media",
        }
        for url, expected in cases.items():
            with self.subTest(url=url):
                decision = classify_exa_source(url)
                self.assertTrue(decision.eligible_for_industry)
                self.assertEqual(decision.source_type, expected)


if __name__ == "__main__":
    unittest.main()
