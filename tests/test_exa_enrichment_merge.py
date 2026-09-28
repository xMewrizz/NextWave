from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from nextwave.evaluation.exa_enrichment_merge import build_combined_enrichment
from nextwave.evaluation.exa_enrichment_plan import EXA_ENRICHMENT_PLAN_VERSION
from nextwave.evaluation.exa_enrichment_run import EXA_ENRICHMENT_RESULT_VERSION


def digest(raw: bytes):
    return {"size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def write_bundle(root: Path, schema: str, files: dict[str, bytes], **manifest_fields):
    root.mkdir()
    for name, raw in files.items():
        (root / name).write_bytes(raw)
    manifest = {
        "schema_version": schema,
        **manifest_fields,
        "outputs": {name: digest(raw) for name, raw in files.items()},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def jsonl(*rows):
    return b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)


class ExaEnrichmentMergeTests(unittest.TestCase):
    def test_keeps_science_and_industry_but_audits_academic_exa(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            analysis = root / "analysis"
            analysis_plan = {
                "schema_version": "labeling-enrichment-plan-v2",
                "candidates": [{"candidate_id": "candidate-1"}],
            }
            analysis_raw = (json.dumps(analysis_plan) + "\n").encode()
            write_bundle(
                analysis,
                "labeling-enrichment-plan-v2",
                {"plan.json": analysis_raw},
                bundle_id="bundle-1",
                candidate_count=1,
                plan_role="analysis_candidates",
            )

            science = root / "science"
            science_docs = jsonl(
                {
                    "candidate_id": "candidate-1",
                    "connector": "openalex",
                    "document_id": "science-1",
                    "published_at": "2026-01-01",
                }
            )
            science_cov = jsonl(
                {
                    "candidate_id": "candidate-1",
                    "source_class": "scientific",
                    "status": "complete",
                }
            )
            write_bundle(
                science,
                "labeling-enrichment-result-v2",
                {"documents.jsonl": science_docs, "coverage.jsonl": science_cov},
                bundle_id="bundle-1",
                plan=digest(analysis_raw),
            )

            exa_plan = root / "exa-plan"
            exa_plan_raw = b'{"schema_version":"analysis-exa-enrichment-plan-v2"}\n'
            write_bundle(
                exa_plan,
                EXA_ENRICHMENT_PLAN_VERSION,
                {"plan.json": exa_plan_raw},
                bundle_id="bundle-1",
                inputs={"analysis_plan": digest(analysis_raw)},
            )
            exa = root / "exa"
            base_document = {
                "authors": [],
                "automatic_translation": False,
                "connector_id": "exa",
                "doi": None,
                "excerpt": "Evidence text",
                "external_id": "external",
                "generated_summary": False,
                "language": "en",
                "observed_at": "2026-09-15T00:00:00+00:00",
                "organizations": [],
                "origin_confidence": 1.0,
                "origin_method": "canonical_url",
                "published_at": "2026-01-01",
                "publisher": "publisher",
                "retrieved_at": "2026-09-15T00:00:00+00:00",
                "snapshot_id": "snapshot-1",
                "trust_tier": "unknown",
            }
            exa_docs = jsonl(
                {
                    "candidate_id": "candidate-1",
                    "request_id": "request-1",
                    "window": "recent",
                    "document": {
                        **base_document,
                        "document_id": "exa-news",
                        "canonical_url": "https://www.reuters.com/technology/example",
                        "origin_id": "url:https://www.reuters.com/technology/example",
                        "source_type": "industry_media",
                        "title": "Industry report",
                        "url": "https://www.reuters.com/technology/example",
                    },
                },
                {
                    "candidate_id": "candidate-1",
                    "request_id": "request-1",
                    "window": "recent",
                    "document": {
                        **base_document,
                        "document_id": "exa-paper",
                        "canonical_url": "https://arxiv.org/abs/1234.5678",
                        "origin_id": "url:https://arxiv.org/abs/1234.5678",
                        "source_type": "industry_media",
                        "title": "Research paper",
                        "url": "https://arxiv.org/abs/1234.5678",
                    },
                },
            )
            exa_cov = jsonl(
                {"candidate_id": "candidate-1", "status": "complete"}
            )
            write_bundle(
                exa,
                EXA_ENRICHMENT_RESULT_VERSION,
                {"documents.jsonl": exa_docs, "coverage.jsonl": exa_cov},
                bundle_id="bundle-1",
                plan=digest(exa_plan_raw),
            )
            output = build_combined_enrichment(
                analysis_plan_dir=analysis,
                scientific_result_dir=science,
                exa_plan_dir=exa_plan,
                exa_result_dir=exa,
            )
            documents = [json.loads(line) for line in output["documents.jsonl"].splitlines()]
            excluded = [
                json.loads(line) for line in output["excluded_documents.jsonl"].splitlines()
            ]
            self.assertEqual({row["document_id"] for row in documents}, {"science-1", "exa-news"})
            news = next(row for row in documents if row["document_id"] == "exa-news")
            self.assertEqual(news["source_type"], "industry_media")
            self.assertEqual(excluded[0]["document_id"], "exa-paper")
            self.assertEqual(excluded[0]["reason"], "academic_domain")


if __name__ == "__main__":
    unittest.main()
