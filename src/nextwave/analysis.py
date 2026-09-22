"""UI-independent integration point for the final ML analysis, after decision policy."""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from pydantic import Field, field_validator

from nextwave.contracts import CandidateAssessment, ContractModel


class AnalysisMetadata(ContractModel):
    """Versions declared before inference, including runs with no candidates."""

    method_version: str = Field(max_length=100)
    model_version: str
    feature_version: str
    corpus_version: str = Field(max_length=100)

    @field_validator("method_version", "model_version", "feature_version", "corpus_version")
    @classmethod
    def require_text(cls, value: str) -> str:
        if not value:
            raise ValueError("must not be blank")
        return value


class Analyzer(Protocol):
    """Factory returns one analyzer per run; blocking work runs outside the API loop.

    `analyze` performs discovery, features, inference and final decision policy.
    Return ALL final candidates, including watchlist and excluded. No UI fields.
    Evidence.excerpt is the source quote; direction is support/counter when known.
    See docs/ML_INTEGRATION.md for the exact payload and factory configuration.
    """

    metadata: AnalysisMetadata

    def analyze(self, query: str) -> Sequence[CandidateAssessment | Mapping[str, Any]]: ...
