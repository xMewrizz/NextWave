"""Query scope resolution and bounded discovery planning."""

from .contracts import (
    ANALYSIS_SCOPE_SCHEMA_VERSION,
    DISCOVERY_BUDGET_SCHEMA_VERSION,
    DISCOVERY_PLAN_SCHEMA_VERSION,
    AnalysisScope,
    DiscoveryBudget,
    DiscoveryPlan,
)
from .planner import (
    DEFAULT_DISCOVERY_BUDGETS,
    DEFAULT_RESOLVER_VERSION,
    build_analysis_scope,
    build_discovery_plan,
)

__all__ = [
    "ANALYSIS_SCOPE_SCHEMA_VERSION",
    "DEFAULT_DISCOVERY_BUDGETS",
    "DEFAULT_RESOLVER_VERSION",
    "DISCOVERY_BUDGET_SCHEMA_VERSION",
    "DISCOVERY_PLAN_SCHEMA_VERSION",
    "AnalysisScope",
    "DiscoveryBudget",
    "DiscoveryPlan",
    "build_analysis_scope",
    "build_discovery_plan",
]
