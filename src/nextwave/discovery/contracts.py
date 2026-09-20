"""Immutable contracts for the scope and bounded plan of one discovery run."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any

from nextwave.sources import ConnectorId, QueryPurpose, SourceQuery

ANALYSIS_SCOPE_SCHEMA_VERSION = "analysis-scope-v1"
DISCOVERY_BUDGET_SCHEMA_VERSION = "discovery-budget-v1"
DISCOVERY_PLAN_SCHEMA_VERSION = "discovery-plan-v1"

_STABLE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{2,99}\Z")
_LANGUAGE = re.compile(r"[a-z]{2,3}(?:-[a-z0-9]{2,8})*\Z")


def _require_stable_id(value: str, field_name: str) -> None:
    if _STABLE_ID.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase stable identifier")


def _require_text(value: str, field_name: str, *, max_length: int = 200) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    if len(value) > max_length:
        raise ValueError(f"{field_name} must not exceed {max_length} characters")


def _require_unique_text(values: tuple[str, ...], field_name: str) -> None:
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    normalized = [value.strip().casefold() for value in values]
    if any(not value for value in normalized):
        raise ValueError(f"{field_name} must not contain blank values")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must contain unique values")


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, date | datetime):
        return value.isoformat()
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


@dataclass(frozen=True, slots=True)
class AnalysisScope:
    """Resolved meaning of a user query, fixed before candidates are extracted."""

    scope_id: str
    raw_query: str
    normalized_query: str
    search_texts: tuple[str, ...]
    languages: tuple[str, ...]
    resolver_version: str
    topic_ids: tuple[str, ...] = ()
    schema_version: str = field(default=ANALYSIS_SCOPE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        _require_stable_id(self.scope_id, "scope_id")
        _require_text(self.raw_query, "raw_query")
        _require_text(self.normalized_query, "normalized_query")
        _require_unique_text(self.search_texts, "search_texts")
        _require_unique_text(self.languages, "languages")
        if any(_LANGUAGE.fullmatch(value) is None for value in self.languages):
            raise ValueError("languages must contain lowercase language tags")
        _require_stable_id(self.resolver_version, "resolver_version")
        if self.normalized_query.casefold() not in {
            value.casefold() for value in self.search_texts
        }:
            raise ValueError("search_texts must contain normalized_query")
        normalized_topics = [value.strip().casefold() for value in self.topic_ids]
        if any(not value for value in normalized_topics):
            raise ValueError("topic_ids must not contain blank values")
        if len(set(normalized_topics)) != len(normalized_topics):
            raise ValueError("topic_ids must contain unique values")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True, slots=True)
class DiscoveryBudget:
    """Hard resource ceiling for one connector during discovery."""

    connector_id: ConnectorId
    required: bool
    max_requests: int
    max_pages: int
    max_documents: int
    request_timeout_seconds: float
    max_elapsed_seconds: float
    schema_version: str = field(default=DISCOVERY_BUDGET_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.connector_id, ConnectorId):
            raise ValueError("connector_id must be a ConnectorId")
        for field_name, value in (
            ("max_requests", self.max_requests),
            ("max_pages", self.max_pages),
            ("max_documents", self.max_documents),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.max_pages > self.max_requests:
            raise ValueError("max_pages must not exceed max_requests")
        for field_name, value in (
            ("request_timeout_seconds", self.request_timeout_seconds),
            ("max_elapsed_seconds", self.max_elapsed_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.request_timeout_seconds > self.max_elapsed_seconds:
            raise ValueError("request timeout must not exceed the connector elapsed budget")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True, slots=True)
class DiscoveryPlan:
    """One reproducible query and the source budgets allowed to execute it."""

    plan_id: str
    analysis_id: str
    scope: AnalysisScope
    query: SourceQuery
    budgets: tuple[DiscoveryBudget, ...]
    schema_version: str = field(default=DISCOVERY_PLAN_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        _require_stable_id(self.plan_id, "plan_id")
        _require_stable_id(self.analysis_id, "analysis_id")
        if self.query.purpose is not QueryPurpose.DISCOVERY:
            raise ValueError("discovery plan requires a discovery SourceQuery")
        if self.query.analysis_scope_id != self.scope.scope_id:
            raise ValueError("query must reference the plan scope")
        for field_name in (
            "raw_query",
            "normalized_query",
            "search_texts",
            "languages",
            "topic_ids",
        ):
            if getattr(self.query, field_name) != getattr(self.scope, field_name):
                raise ValueError(f"query {field_name} must match the plan scope")
        if not self.budgets:
            raise ValueError("discovery plan must contain connector budgets")
        connector_ids = [budget.connector_id for budget in self.budgets]
        if len(set(connector_ids)) != len(connector_ids):
            raise ValueError("connector budgets must be unique")
        openalex = next(
            (budget for budget in self.budgets if budget.connector_id is ConnectorId.OPENALEX),
            None,
        )
        if openalex is None or not openalex.required:
            raise ValueError("OpenAlex must be present and required for discovery")

    def budget_for(self, connector_id: ConnectorId) -> DiscoveryBudget | None:
        return next(
            (budget for budget in self.budgets if budget.connector_id is connector_id),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))
