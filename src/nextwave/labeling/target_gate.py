"""Ground neutral repair targets and run the qualification Candidate Gate.

This module is deliberately narrower than open discovery.  A target name is
allowed into Candidate Gate only when the exact sequence of at least two
Unicode word tokens occurs in a retrieved title or excerpt.  Gate acceptance
does not assign a training label; publicity and maturity evidence are reviewed
afterwards.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol

from nextwave.contracts import SourceDocument, SourceType, TrustTier
from nextwave.datasets.artifacts import publish_artifact_bundle
from nextwave.discovery import (
    QUALIFICATION_GATE_ID,
    AnalysisScope,
    CandidateGateResult,
    CandidateMentionKind,
    CandidateProposal,
    CandidateProposalBatch,
    GateDecision,
    ScopeGranularity,
    build_candidate_gate_from_environment,
    build_candidate_mention,
)

from .contracts import LABELING_CUTOFF_DATE
from .enrichment_run import (
    COVERAGE_FILENAME as ENRICHMENT_COVERAGE_FILENAME,
)
from .enrichment_run import (
    DOCUMENTS_FILENAME as ENRICHMENT_DOCUMENTS_FILENAME,
)
from .enrichment_run import ENRICHMENT_RESULT_VERSION, load_validated_plan

LABELING_TARGET_GATE_VERSION = "labeling-target-gate-v1"
TARGET_GATE_RESULTS_FILENAME = "gate_results.jsonl"
TARGET_GROUNDINGS_FILENAME = "groundings.jsonl"
TARGET_GATE_ISSUES_FILENAME = "issues.jsonl"
TARGET_GATE_MANIFEST_FILENAME = "manifest.json"
TARGET_GROUNDING_EXTRACTOR_ID = "target-exact-sequence-v1"

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_CONNECTOR_SOURCE_CLASS = {"openalex": "scientific", "mediacloud": "industry"}
_DOMAIN_SCOPE = {
    "Защита ИИ": (
        "target-ai-security-v1",
        "Emerging AI security technologies",
    ),
    "Индустриальный ИИ": (
        "target-industrial-ai-v1",
        "Emerging industrial AI technologies",
    ),
    "Финтех": ("target-fintech-v1", "Emerging fintech technologies"),
}
_DOCUMENT_KEYS = {
    "authors",
    "automatic_translation",
    "candidate_id",
    "canonical_url",
    "connector",
    "connector_id",
    "document_id",
    "doi",
    "excerpt",
    "external_id",
    "generated_summary",
    "language",
    "observed_at",
    "organizations",
    "origin_confidence",
    "origin_id",
    "origin_method",
    "published_at",
    "publisher",
    "request_id",
    "retrieved_at",
    "search_id",
    "snapshot_id",
    "source_type",
    "title",
    "trust_tier",
    "url",
}


class _Gate(Protocol):
    @property
    def gate_id(self) -> str: ...

    def evaluate(
        self,
        scope: AnalysisScope,
        batch: CandidateProposalBatch,
        documents: tuple[SourceDocument, ...],
        *,
        max_concurrency: int = 5,
    ) -> CandidateGateResult: ...


@dataclass(frozen=True, slots=True)
class LabelingTargetGatePaths:
    gate_results: Path
    groundings: Path
    issues: Path
    manifest: Path


@dataclass(frozen=True, slots=True)
class _Grounding:
    candidate_id: str
    document: SourceDocument
    term: str
    term_role: str
    term_index: int
    location: str
    start: int
    end: int
    quote: str

    def to_dict(self) -> dict[str, Any]:
        identity = "|".join(
            (
                LABELING_TARGET_GATE_VERSION,
                self.candidate_id,
                self.document.document_id,
                self.term,
                self.location,
            )
        )
        return {
            "grounding_id": "grounding-"
            + hashlib.sha256(identity.encode()).hexdigest()[:16],
            "candidate_id": self.candidate_id,
            "document_id": self.document.document_id,
            "connector_id": self.document.connector_id,
            "origin_id": self.document.origin_id,
            "term": self.term,
            "term_role": self.term_role,
            "location": self.location,
            "locator": f"{self.location}[{self.start}:{self.end}]",
            "start": self.start,
            "end": self.end,
            "quote": self.quote,
        }


def _digest(payload: bytes) -> dict[str, Any]:
    return {"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, raw


def _checked_file(path: Path, expected: Any, label: str) -> bytes:
    if not isinstance(expected, dict):
        raise ValueError(f"{label} digest is missing")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if expected.get("size_bytes") != len(raw):
        raise ValueError(f"{label} size mismatch with manifest")
    if expected.get("sha256") != hashlib.sha256(raw).hexdigest():
        raise ValueError(f"{label} checksum mismatch with manifest")
    return raw


def _read_jsonl(raw: bytes, label: str) -> list[dict[str, Any]]:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label} line {line_number} is not valid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{label} line {line_number} must be an object")
        rows.append(row)
    return rows


def _render_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _plain_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _optional_date(value: Any, label: str) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO date or null")
    try:
        result = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO date") from error
    if result > LABELING_CUTOFF_DATE:
        raise ValueError(f"{label} is after the labeling cutoff")
    return result


def _optional_datetime(value: Any, label: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO datetime or null")
    try:
        result = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO datetime") from error
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return result


def _source_document(row: Mapping[str, Any], line_number: int) -> SourceDocument:
    if set(row) != _DOCUMENT_KEYS:
        raise ValueError(
            f"documents.jsonl line {line_number} contains unexpected or missing fields"
        )
    if row.get("connector") != row.get("connector_id"):
        raise ValueError(f"documents.jsonl line {line_number} connector mismatch")
    for field in (
        "candidate_id",
        "connector",
        "connector_id",
        "document_id",
        "external_id",
        "language",
        "origin_id",
        "request_id",
        "search_id",
        "snapshot_id",
        "title",
        "url",
        "canonical_url",
    ):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(
                f"documents.jsonl line {line_number} field {field!r} must be text"
            )
    for field in ("authors", "organizations"):
        if not isinstance(row.get(field), list) or any(
            not isinstance(value, str) for value in row[field]
        ):
            raise ValueError(
                f"documents.jsonl line {line_number} field {field!r} must be strings"
            )
    for field in ("automatic_translation", "generated_summary"):
        if not isinstance(row.get(field), bool):
            raise ValueError(
                f"documents.jsonl line {line_number} field {field!r} must be bool"
            )
    for field in ("doi", "excerpt", "origin_method", "publisher"):
        value = row.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(
                f"documents.jsonl line {line_number} field {field!r} must be text or null"
            )
    confidence = row.get("origin_confidence")
    if (
        confidence is not None
        and (
            isinstance(confidence, bool)
            or not isinstance(confidence, int | float)
            or not 0 <= confidence <= 1
        )
    ):
        raise ValueError(
            f"documents.jsonl line {line_number} origin_confidence is invalid"
        )
    try:
        return SourceDocument(
            document_id=row["document_id"],
            connector_id=row["connector_id"],
            external_id=row["external_id"],
            snapshot_id=row["snapshot_id"],
            title=row["title"],
            url=row["url"],
            canonical_url=row["canonical_url"],
            source_type=SourceType(row["source_type"]),
            language=row["language"],
            trust_tier=TrustTier(row["trust_tier"]),
            origin_id=row["origin_id"],
            doi=row["doi"],
            authors=tuple(row["authors"]),
            organizations=tuple(row["organizations"]),
            published_at=_optional_date(row["published_at"], "published_at"),
            observed_at=_optional_datetime(row["observed_at"], "observed_at"),
            retrieved_at=_optional_datetime(row["retrieved_at"], "retrieved_at"),
            publisher=row["publisher"],
            excerpt=row["excerpt"],
            automatic_translation=row["automatic_translation"],
            generated_summary=row["generated_summary"],
            origin_method=row["origin_method"],
            origin_confidence=row["origin_confidence"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid document on line {line_number}: {error}") from error


def _load_complete_result(
    result_dir: Path,
    *,
    plan: Mapping[str, Any],
    plan_digest: Mapping[str, Any],
) -> dict[str, list[tuple[dict[str, Any], SourceDocument]]]:
    manifest, _ = _read_json(result_dir / "manifest.json", "enrichment manifest")
    if manifest.get("schema_version") != ENRICHMENT_RESULT_VERSION:
        raise ValueError("enrichment result version mismatch")
    if manifest.get("bundle_id") != plan["bundle"]["bundle_id"]:
        raise ValueError("enrichment result bundle_id mismatch")
    if manifest.get("plan") != plan_digest:
        raise ValueError("enrichment result plan digest mismatch")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("enrichment result outputs must be an object")
    document_raw = _checked_file(
        result_dir / ENRICHMENT_DOCUMENTS_FILENAME,
        outputs.get(ENRICHMENT_DOCUMENTS_FILENAME),
        ENRICHMENT_DOCUMENTS_FILENAME,
    )
    coverage_raw = _checked_file(
        result_dir / ENRICHMENT_COVERAGE_FILENAME,
        outputs.get(ENRICHMENT_COVERAGE_FILENAME),
        ENRICHMENT_COVERAGE_FILENAME,
    )
    candidates = {item["candidate_id"]: item for item in plan["candidates"]}
    search_index: dict[str, tuple[str, str, set[str]]] = {}
    for candidate in candidates.values():
        for search in candidate["searches"]:
            search_index[search["search_id"]] = (
                candidate["candidate_id"],
                search["connector"],
                {request["request_id"] for request in search["requests"]},
            )

    coverage = _read_jsonl(coverage_raw, ENRICHMENT_COVERAGE_FILENAME)
    seen_coverage: set[tuple[str, str]] = set()
    for row in coverage:
        candidate_id = row.get("candidate_id")
        source_class = row.get("source_class")
        key = (candidate_id, source_class)
        if candidate_id not in candidates or source_class not in {"scientific", "industry"}:
            raise ValueError("coverage contains an unknown candidate or source class")
        if key in seen_coverage:
            raise ValueError("coverage contains a duplicate candidate/source pair")
        seen_coverage.add(key)
        planned = _plain_int(row.get("planned_requests"), "planned_requests")
        successful = _plain_int(row.get("successful_requests"), "successful_requests")
        failed = _plain_int(row.get("failed_requests"), "failed_requests")
        if (
            row.get("status") != "complete"
            or successful != planned
            or failed != 0
            or row.get("failed_request_ids") != []
        ):
            raise ValueError(f"coverage is incomplete for {candidate_id}/{source_class}")
    expected_coverage = {
        (candidate_id, source_class)
        for candidate_id in candidates
        for source_class in ("scientific", "industry")
    }
    if seen_coverage != expected_coverage:
        raise ValueError("coverage does not contain both source classes for every target")

    rows = _read_jsonl(document_raw, ENRICHMENT_DOCUMENTS_FILENAME)
    totals = manifest.get("totals")
    if not isinstance(totals, dict) or _plain_int(
        totals.get("returned_documents"), "returned_documents"
    ) != len(rows):
        raise ValueError("enrichment returned_documents total mismatch")
    result: dict[str, list[tuple[dict[str, Any], SourceDocument]]] = defaultdict(list)
    seen_pairs: set[tuple[str, str]] = set()
    for line_number, row in enumerate(rows, start=1):
        candidate_id = row.get("candidate_id")
        if candidate_id not in candidates:
            raise ValueError(f"unknown candidate on documents line {line_number}")
        expected = search_index.get(row.get("search_id"))
        if expected is None or expected[0] != candidate_id:
            raise ValueError(f"invalid search provenance on documents line {line_number}")
        if row.get("connector_id") != expected[1] or row.get("request_id") not in expected[2]:
            raise ValueError(f"invalid request provenance on documents line {line_number}")
        pair = (candidate_id, row.get("document_id"))
        if pair in seen_pairs:
            raise ValueError(f"duplicate candidate/document pair on line {line_number}")
        seen_pairs.add(pair)
        result[candidate_id].append((row, _source_document(row, line_number)))
    return result


def _tokens(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(_TOKEN.findall(normalized))


def _find_sequence(
    text: str | None, sequence: tuple[str, ...]
) -> tuple[int, int, str] | None:
    """Return a locator that round-trips the original text, not normalized text."""

    if text is None or len(sequence) < 2:
        return None
    matches = list(_TOKEN.finditer(text))
    tokens = tuple(
        unicodedata.normalize("NFKC", match.group()).casefold() for match in matches
    )
    width = len(sequence)
    for index in range(len(tokens) - width + 1):
        if tokens[index : index + width] == sequence:
            start = matches[index].start()
            end = matches[index + width - 1].end()
            return start, end, text[start:end]
    return None


def _ground_candidate(
    candidate: Mapping[str, Any],
    rows: list[tuple[dict[str, Any], SourceDocument]],
) -> tuple[list[_Grounding], list[str]]:
    terms = [candidate["canonical_name"], *candidate["aliases"]]
    tokenized = [(index, term, _tokens(term)) for index, term in enumerate(terms)]
    eligible = [item for item in tokenized if len(item[2]) >= 2]
    ineligible = [term for term in terms if len(_tokens(term)) < 2]
    groundings: list[_Grounding] = []
    for _, document in rows:
        if document.published_at is None:
            continue
        for index, term, sequence in eligible:
            for location, text in (("title", document.title), ("excerpt", document.excerpt)):
                match = _find_sequence(text, sequence)
                if match is not None:
                    start, end, quote = match
                    groundings.append(
                        _Grounding(
                            candidate_id=candidate["candidate_id"],
                            document=document,
                            term=term,
                            term_role="canonical" if index == 0 else "alias",
                            term_index=index,
                            location=location,
                            start=start,
                            end=end,
                            quote=quote,
                        )
                    )
    groundings.sort(
        key=lambda item: (
            item.term_index,
            0 if item.location == "title" else 1,
            item.document.connector_id,
            item.document.document_id,
        )
    )
    return groundings, ineligible


def _scope(domain: str) -> AnalysisScope:
    try:
        scope_id, query = _DOMAIN_SCOPE[domain]
    except KeyError as error:
        raise ValueError(f"target Gate does not support domain {domain!r}") from error
    return AnalysisScope(
        scope_id=scope_id,
        raw_query=query,
        normalized_query=query,
        search_texts=(query,),
        languages=("en", "ru"),
        resolver_version=LABELING_TARGET_GATE_VERSION,
        granularity=ScopeGranularity.DIRECTION,
    )


def _proposal(
    candidate: Mapping[str, Any], scope_id: str, groundings: list[_Grounding]
) -> CandidateProposal:
    documents: dict[str, SourceDocument] = {}
    identities: set[str] = set()
    mention_ids: set[str] = set()
    source_kinds: set[CandidateMentionKind] = set()
    grounded_aliases: set[str] = set()
    for item in groundings:
        if item.term_role != "canonical":
            continue
        identity = (item.document.doi or item.document.canonical_url).strip().casefold()
        if item.document.document_id not in documents and identity in identities:
            continue
        identities.add(identity)
        documents.setdefault(item.document.document_id, item.document)
        kind = (
            CandidateMentionKind.TITLE
            if item.location == "title"
            else CandidateMentionKind.EXCERPT
        )
        mention = build_candidate_mention(
            item.document,
            text=item.term,
            kind=kind,
            locator=f"{item.location}[{item.start}:{item.end}]",
            extractor_id=TARGET_GROUNDING_EXTRACTOR_ID,
        )
        mention_ids.add(mention.mention_id)
        source_kinds.add(kind)
        if item.term_role == "alias":
            grounded_aliases.add(item.term)
    identity = f"{scope_id}|{candidate['candidate_id']}"
    return CandidateProposal(
        proposal_id="proposal-" + hashlib.sha256(identity.encode()).hexdigest()[:16],
        analysis_scope_id=scope_id,
        canonical_name=candidate["canonical_name"],
        normalized_name=" ".join(_tokens(candidate["canonical_name"])),
        aliases=tuple(sorted(grounded_aliases, key=lambda item: (item.casefold(), item))),
        mention_ids=tuple(sorted(mention_ids)),
        source_kinds=tuple(sorted(source_kinds, key=lambda item: item.value)),
        connector_ids=tuple(sorted({item.connector_id for item in documents.values()})),
        provider_term_ids=(),
        document_ids=tuple(documents),
        origin_ids=tuple(sorted({item.origin_id for item in documents.values()})),
        max_provider_score=None,
        primary_provider_topic=False,
    )


def _gate_document(
    document: SourceDocument, groundings: list[_Grounding]
) -> SourceDocument:
    """Keep a canonical excerpt match inside Gate's bounded 700-char context."""

    matches = [
        item
        for item in groundings
        if item.term_role == "canonical"
        and item.document.document_id == document.document_id
        and item.location == "excerpt"
    ]
    if not matches or document.excerpt is None:
        return document
    match = matches[0]
    start = max(0, match.start - 250)
    end = min(len(document.excerpt), start + 700)
    if match.end > end:
        end = match.end
        start = max(0, end - 700)
    return replace(document, excerpt=document.excerpt[start:end])


def run_target_gate(
    *,
    plan_dir: str | Path,
    result_dir: str | Path,
    output_dir: str | Path,
    environment: Mapping[str, str] | None = None,
    gate: _Gate | None = None,
) -> LabelingTargetGatePaths:
    """Ground neutral targets and atomically publish qualification Gate decisions."""

    plan, plan_bytes, plan_digest = load_validated_plan(plan_dir)
    result_path = Path(result_dir)
    result_manifest, result_manifest_raw = _read_json(
        result_path / "manifest.json", "enrichment manifest"
    )
    documents_by_candidate = _load_complete_result(
        result_path, plan=plan, plan_digest=plan_digest
    )
    candidates = sorted(plan["candidates"], key=lambda item: item["candidate_id"])

    groundings_by_candidate: dict[str, list[_Grounding]] = {}
    ineligible_by_candidate: dict[str, list[str]] = {}
    proposals_to_run: list[tuple[dict[str, Any], CandidateProposal]] = []
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        groundings, ineligible = _ground_candidate(
            candidate, documents_by_candidate.get(candidate_id, [])
        )
        groundings_by_candidate[candidate_id] = groundings
        ineligible_by_candidate[candidate_id] = ineligible
        canonical_grounded = any(item.term_role == "canonical" for item in groundings)
        if not canonical_grounded:
            continue
        scope = _scope(candidate["domain"])
        proposal = _proposal(candidate, scope.scope_id, groundings)
        proposals_to_run.append((candidate, proposal))

    runner = gate or build_candidate_gate_from_environment(environment)
    if runner.gate_id != QUALIFICATION_GATE_ID:
        raise ValueError(
            f"target Gate {runner.gate_id!r} does not match qualification Gate "
            f"{QUALIFICATION_GATE_ID!r}"
        )
    results: dict[str, tuple[CandidateGateResult, Any]] = {}
    gate_issues: list[dict[str, Any]] = []
    for candidate, proposal in sorted(
        proposals_to_run, key=lambda item: item[0]["candidate_id"]
    ):
        domain = candidate["domain"]
        scope = _scope(domain)
        by_id = {
            item.document.document_id: item.document
            for item in groundings_by_candidate[candidate["candidate_id"]]
            if item.term_role == "canonical"
        }
        documents = tuple(
            _gate_document(
                by_id[document_id], groundings_by_candidate[candidate["candidate_id"]]
            )
            for document_id in proposal.document_ids
        )
        gate_result = runner.evaluate(
            scope,
            CandidateProposalBatch(scope.scope_id, (proposal,), ()),
            documents,
        )
        if gate_result.gate_id != QUALIFICATION_GATE_ID:
            raise ValueError(
                f"target Gate {gate_result.gate_id!r} does not match qualification Gate "
                f"{QUALIFICATION_GATE_ID!r}"
            )
        decisions = {decision.proposal_id: decision for decision in gate_result.decisions}
        if set(decisions) != {proposal.proposal_id}:
            raise ValueError("target Gate did not return exactly one decision per proposal")
        results[candidate["candidate_id"]] = (
            gate_result,
            decisions[proposal.proposal_id],
        )
        gate_issues.extend(
            {"domain": domain, **issue.to_dict()} for issue in gate_result.issues
        )

    gate_rows: list[dict[str, Any]] = []
    grounding_rows = [
        grounding.to_dict()
        for candidate_id in sorted(groundings_by_candidate)
        for grounding in groundings_by_candidate[candidate_id]
    ]
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        groundings = groundings_by_candidate[candidate_id]
        canonical_grounded = any(item.term_role == "canonical" for item in groundings)
        grounded_terms = sorted(
            {item.term for item in groundings}, key=lambda item: (item.casefold(), item)
        )
        if not canonical_grounded:
            decision = None
            reason = None
            basis_document_ids: list[str] = []
            if groundings:
                explanation = (
                    "Only an eligible alias was grounded; canonical/alias equivalence "
                    "requires review before Candidate Gate."
                )
                promotion_status = "alias_review_required"
            else:
                explanation = (
                    "No eligible exact term sequence was found in a title or excerpt."
                )
                promotion_status = "not_grounded"
            gate_id = None
        else:
            gate_result, gate_decision = results[candidate_id]
            decision = gate_decision.decision.value
            reason = gate_decision.reason.value
            basis_document_ids = list(gate_decision.basis_document_ids)
            explanation = gate_decision.explanation
            gate_id = gate_result.gate_id
            if gate_decision.decision is GateDecision.ACCEPT:
                promotion_status = (
                    "evidence_review_required"
                    if canonical_grounded
                    else "alias_review_required"
                )
            elif gate_decision.decision is GateDecision.REJECT:
                promotion_status = "gate_rejected"
            else:
                promotion_status = "gate_review"
        gate_rows.append(
            {
                "schema_version": LABELING_TARGET_GATE_VERSION,
                "candidate_id": candidate_id,
                "canonical_name": candidate["canonical_name"],
                "domain": candidate["domain"],
                "canonical_grounded": canonical_grounded,
                "grounded_terms": grounded_terms,
                "ineligible_short_terms": ineligible_by_candidate[candidate_id],
                "grounded_document_count": len(
                    {item.document.document_id for item in groundings}
                ),
                "gate_called": canonical_grounded,
                "gate_id": gate_id,
                "gate_decision": decision,
                "gate_reason": reason,
                "basis_document_ids": basis_document_ids,
                "explanation": explanation,
                "promotion_status": promotion_status,
            }
        )

    gate_bytes = _render_jsonl(gate_rows)
    grounding_bytes = _render_jsonl(grounding_rows)
    issue_bytes = _render_jsonl(
        sorted(
            gate_issues,
            key=lambda item: (
                item["domain"].casefold(),
                item.get("proposal_id") or "",
                item["code"],
            ),
        )
    )
    statuses = defaultdict(int)
    decisions = defaultdict(int)
    for row in gate_rows:
        statuses[row["promotion_status"]] += 1
        decisions[row["gate_decision"] or "not_called"] += 1
    manifest_base = {
        "schema_version": LABELING_TARGET_GATE_VERSION,
        "cutoff_date": LABELING_CUTOFF_DATE.isoformat(),
        "gate_id": QUALIFICATION_GATE_ID,
        "policy": {
            "minimum_term_tokens": 2,
            "match": "exact_contiguous_nfkc_casefolded_unicode_tokens",
            "fields": ["title", "excerpt"],
            "gate_acceptance_assigns_label": False,
            "alias_only_requires_review": True,
            "undated_documents": "excluded_from_grounding",
        },
        "inputs": {
            "plan.json": _digest(plan_bytes),
            "enrichment_manifest.json": _digest(result_manifest_raw),
            "enrichment_documents.jsonl": result_manifest["outputs"][ENRICHMENT_DOCUMENTS_FILENAME],
            "enrichment_coverage.jsonl": result_manifest["outputs"][ENRICHMENT_COVERAGE_FILENAME],
        },
        "totals": {
            "candidates": len(gate_rows),
            "groundings": len(grounding_rows),
            "grounded_candidates": sum(
                bool(groundings_by_candidate[row["candidate_id"]])
                for row in gate_rows
            ),
            "ungrounded_candidates": sum(
                not groundings_by_candidate[row["candidate_id"]]
                for row in gate_rows
            ),
            "canonical_grounded_candidates": sum(
                bool(row["canonical_grounded"]) for row in gate_rows
            ),
            "alias_only_candidates": sum(
                row["promotion_status"] == "alias_review_required" for row in gate_rows
            ),
            "gate_decisions": dict(sorted(decisions.items())),
            "promotion_statuses": dict(sorted(statuses.items())),
            "issues": len(gate_issues),
        },
    }
    outputs = {
        TARGET_GATE_RESULTS_FILENAME: gate_bytes,
        TARGET_GROUNDINGS_FILENAME: grounding_bytes,
        TARGET_GATE_ISSUES_FILENAME: issue_bytes,
    }
    manifest_bytes = (
        json.dumps(
            {
                **manifest_base,
                "outputs": {name: _digest(payload) for name, payload in outputs.items()},
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    published = publish_artifact_bundle(
        {**outputs, TARGET_GATE_MANIFEST_FILENAME: manifest_bytes}, output_dir
    )
    return LabelingTargetGatePaths(
        gate_results=published[TARGET_GATE_RESULTS_FILENAME],
        groundings=published[TARGET_GROUNDINGS_FILENAME],
        issues=published[TARGET_GATE_ISSUES_FILENAME],
        manifest=published[TARGET_GATE_MANIFEST_FILENAME],
    )
