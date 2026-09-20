"""Immutable contracts for the scope and bounded plan of one discovery run."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum, StrEnum
from typing import Any

from nextwave.sources import ConnectorId, QueryPurpose, SourceQuery

ANALYSIS_SCOPE_SCHEMA_VERSION = "analysis-scope-v1"
DISCOVERY_BUDGET_SCHEMA_VERSION = "discovery-budget-v1"
DISCOVERY_PLAN_SCHEMA_VERSION = "discovery-plan-v1"
QUERY_INTERPRETATION_SCHEMA_VERSION = "query-interpretation-v1"
QUERY_RESOLUTION_SCHEMA_VERSION = "query-resolution-v1"

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


class ScopeGranularity(StrEnum):
    """Breadth of the user's intent before source-specific retrieval begins."""

    DIRECTION = "direction"
    TECHNOLOGY = "technology"


class TaxonomyLevel(StrEnum):
    TOPIC = "topic"
    SUBFIELD = "subfield"


class TaxonomyLookupStatus(StrEnum):
    MATCHED = "matched"
    NO_MATCH = "no_match"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class QueryInterpretation:
    """Validated language-model output before OpenAlex taxonomy resolution."""

    normalized_query: str
    search_texts: tuple[str, ...]
    languages: tuple[str, ...]
    granularity: ScopeGranularity
    interpreter_version: str
    schema_version: str = field(default=QUERY_INTERPRETATION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        _require_text(self.normalized_query, "normalized_query")
        if re.search(r"[А-Яа-яЁё]", self.normalized_query) or re.search(
            r"[a-z]", self.normalized_query.casefold()
        ) is None:
            raise ValueError("normalized_query must be an English retrieval query")
        _require_unique_text(self.search_texts, "search_texts")
        if len(self.search_texts) > 8:
            raise ValueError("search_texts must not contain more than 8 variants")
        if any(len(value) > 200 for value in self.search_texts):
            raise ValueError("search_texts values must not exceed 200 characters")
        _require_unique_text(self.languages, "languages")
        if any(_LANGUAGE.fullmatch(value) is None for value in self.languages):
            raise ValueError("languages must contain lowercase language tags")
        if "en" not in self.languages:
            raise ValueError("languages must contain en")
        if not isinstance(self.granularity, ScopeGranularity):
            raise ValueError("granularity must be a ScopeGranularity")
        _require_stable_id(self.interpreter_version, "interpreter_version")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True, slots=True)
class TaxonomyCandidate:
    """One OpenAlex topic or subfield considered while resolving the query."""

    entity_id: str
    level: TaxonomyLevel
    display_name: str
    description: str | None
    works_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.level, TaxonomyLevel):
            raise ValueError("level must be a TaxonomyLevel")
        if self.level is TaxonomyLevel.TOPIC:
            if re.fullmatch(r"T[0-9]+", self.entity_id) is None:
                raise ValueError("topic entity_id must use the OpenAlex T<number> format")
        elif re.fullmatch(r"[0-9]+", self.entity_id) is None:
            raise ValueError("subfield entity_id must contain digits")
        _require_text(self.display_name, "display_name")
        if self.description is not None:
            _require_text(self.description, "description", max_length=2000)
        if isinstance(self.works_count, bool) or not isinstance(self.works_count, int):
            raise ValueError("works_count must be an integer")
        if self.works_count < 0:
            raise ValueError("works_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True, slots=True)
class AnalysisScope:
    """Resolved meaning of a user query, fixed before candidates are extracted."""

    scope_id: str
    raw_query: str
    normalized_query: str
    search_texts: tuple[str, ...]
    languages: tuple[str, ...]
    resolver_version: str
    granularity: ScopeGranularity = ScopeGranularity.DIRECTION
    topic_ids: tuple[str, ...] = ()
    subfield_ids: tuple[str, ...] = ()
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
        if not isinstance(self.granularity, ScopeGranularity):
            raise ValueError("granularity must be a ScopeGranularity")
        if self.normalized_query.casefold() not in {
            value.casefold() for value in self.search_texts
        }:
            raise ValueError("search_texts must contain normalized_query")
        normalized_topics = [value.strip().casefold() for value in self.topic_ids]
        if any(not value for value in normalized_topics):
            raise ValueError("topic_ids must not contain blank values")
        if len(set(normalized_topics)) != len(normalized_topics):
            raise ValueError("topic_ids must contain unique values")
        normalized_subfields = [value.strip().casefold() for value in self.subfield_ids]
        if any(not value for value in normalized_subfields):
            raise ValueError("subfield_ids must not contain blank values")
        if len(set(normalized_subfields)) != len(normalized_subfields):
            raise ValueError("subfield_ids must contain unique values")
        if self.topic_ids and self.subfield_ids:
            raise ValueError("topic_ids and subfield_ids are mutually exclusive")
        if self.granularity is ScopeGranularity.DIRECTION and self.topic_ids:
            raise ValueError("direction scope cannot use topic_ids")
        if self.granularity is ScopeGranularity.TECHNOLOGY and self.subfield_ids:
            raise ValueError("technology scope cannot use subfield_ids")

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))


@dataclass(frozen=True, slots=True)
class QueryResolution:
    """Auditable result of interpretation and OpenAlex taxonomy matching."""

    scope: AnalysisScope
    interpretation: QueryInterpretation
    taxonomy_status: TaxonomyLookupStatus
    taxonomy_candidates: tuple[TaxonomyCandidate, ...]
    selected_taxonomy: TaxonomyCandidate | None
    taxonomy_error: str | None = None
    schema_version: str = field(default=QUERY_RESOLUTION_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.taxonomy_status, TaxonomyLookupStatus):
            raise ValueError("taxonomy_status must be a TaxonomyLookupStatus")
        if self.scope.granularity is not self.interpretation.granularity:
            raise ValueError("scope granularity must match the interpretation")
        if self.taxonomy_status is TaxonomyLookupStatus.MATCHED:
            if self.selected_taxonomy is None or self.taxonomy_error is not None:
                raise ValueError("matched resolution requires a selection and no error")
            if self.selected_taxonomy not in self.taxonomy_candidates:
                raise ValueError("selected taxonomy must be one of the candidates")
        elif self.taxonomy_status is TaxonomyLookupStatus.NO_MATCH:
            if self.selected_taxonomy is not None or self.taxonomy_error is not None:
                raise ValueError("no-match resolution cannot contain a selection or error")
        else:
            if self.selected_taxonomy is not None or not self.taxonomy_error:
                raise ValueError("unavailable resolution requires a taxonomy error")

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
            "subfield_ids",
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
