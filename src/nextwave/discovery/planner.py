"""Deterministic construction of analysis scopes and discovery plans."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from datetime import date

from nextwave.sources import ConnectorId, QueryPurpose, SourceQuery

from .contracts import AnalysisScope, DiscoveryBudget, DiscoveryPlan, ScopeGranularity

DEFAULT_RESOLVER_VERSION = "query-resolver-v1"

DEFAULT_DISCOVERY_BUDGETS = (
    DiscoveryBudget(
        connector_id=ConnectorId.OPENALEX,
        required=True,
        max_requests=6,
        max_pages=4,
        max_documents=400,
        request_timeout_seconds=30.0,
        max_elapsed_seconds=90.0,
    ),
    DiscoveryBudget(
        connector_id=ConnectorId.MEDIACLOUD,
        required=False,
        max_requests=4,
        max_pages=2,
        max_documents=200,
        request_timeout_seconds=60.0,
        max_elapsed_seconds=120.0,
    ),
    DiscoveryBudget(
        connector_id=ConnectorId.GDELT,
        required=False,
        max_requests=2,
        max_pages=1,
        max_documents=100,
        request_timeout_seconds=30.0,
        max_elapsed_seconds=45.0,
    ),
)


def _normalize_whitespace(value: str, field_name: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", value).split())
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    if len(normalized) > 200:
        raise ValueError(f"{field_name} must not exceed 200 characters")
    return normalized


def _normalize_unique(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize_whitespace(value, field_name)
        identity = normalized.casefold()
        if identity in seen:
            continue
        seen.add(identity)
        result.append(normalized)
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    return tuple(result)


def _digest(prefix: str, payload: dict[str, object]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"{prefix}-{hashlib.sha256(serialized).hexdigest()[:16]}"


def build_analysis_scope(
    *,
    raw_query: str,
    normalized_query: str,
    search_texts: tuple[str, ...],
    languages: tuple[str, ...],
    granularity: ScopeGranularity = ScopeGranularity.DIRECTION,
    topic_ids: tuple[str, ...] = (),
    subfield_ids: tuple[str, ...] = (),
    resolver_version: str = DEFAULT_RESOLVER_VERSION,
) -> AnalysisScope:
    """Freeze one reviewed multilingual interpretation of the user's query."""

    raw = _normalize_whitespace(raw_query, "raw_query")
    normalized = _normalize_whitespace(normalized_query, "normalized_query").casefold()
    texts = _normalize_unique((normalized, *search_texts), "search_texts")
    normalized_languages = tuple(
        value.casefold() for value in _normalize_unique(languages, "languages")
    )
    normalized_topics = _normalize_unique(topic_ids, "topic_ids") if topic_ids else ()
    normalized_subfields = (
        _normalize_unique(subfield_ids, "subfield_ids") if subfield_ids else ()
    )
    if normalized_topics and normalized_subfields:
        raise ValueError("topic_ids and subfield_ids are mutually exclusive")
    resolver = _normalize_whitespace(resolver_version, "resolver_version").casefold()
    scope_id = _digest(
        "scope",
        {
            "languages": normalized_languages,
            "normalized_query": normalized,
            "resolver_version": resolver,
            "granularity": granularity.value,
            "search_texts": tuple(value.casefold() for value in texts),
            "topic_ids": tuple(value.casefold() for value in normalized_topics),
            "subfield_ids": tuple(value.casefold() for value in normalized_subfields),
        },
    )
    return AnalysisScope(
        scope_id=scope_id,
        raw_query=raw,
        normalized_query=normalized,
        search_texts=texts,
        languages=normalized_languages,
        resolver_version=resolver,
        granularity=granularity,
        topic_ids=normalized_topics,
        subfield_ids=normalized_subfields,
    )


def build_discovery_plan(
    *,
    analysis_id: str,
    scope: AnalysisScope,
    published_from: date,
    cutoff_date: date,
    budgets: tuple[DiscoveryBudget, ...] = DEFAULT_DISCOVERY_BUDGETS,
) -> DiscoveryPlan:
    """Bind a semantic scope to one historical window and hard source budgets."""

    query_id = _digest(
        "query",
        {
            "cutoff_date": cutoff_date.isoformat(),
            "published_from": published_from.isoformat(),
            "scope_id": scope.scope_id,
        },
    )
    query = SourceQuery(
        query_id=query_id,
        analysis_scope_id=scope.scope_id,
        purpose=QueryPurpose.DISCOVERY,
        raw_query=scope.raw_query,
        normalized_query=scope.normalized_query,
        search_texts=scope.search_texts,
        published_from=published_from,
        published_until=cutoff_date,
        cutoff_date=cutoff_date,
        languages=scope.languages,
        topic_ids=scope.topic_ids,
        subfield_ids=scope.subfield_ids,
    )
    plan_id = _digest(
        "plan",
        {
            "analysis_id": analysis_id,
            "budgets": [budget.to_dict() for budget in budgets],
            "query_id": query.query_id,
        },
    )
    return DiscoveryPlan(
        plan_id=plan_id,
        analysis_id=analysis_id,
        scope=scope,
        query=query,
        budgets=budgets,
    )
