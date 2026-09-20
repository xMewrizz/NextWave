"""Explicit LLM selection and the approved OpenAI JSON adapter."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from nextwave.sources import HttpResponse

OPENAI_RESPONSES_ENDPOINT = "https://api.openai.com/v1/responses"
OPENAI_QUERY_MODEL = "gpt-4.1"
OPENAI_ADAPTER_VERSION = "openai-responses-v1"


class LlmProvider(StrEnum):
    OPENAI = "openai"
    YANDEX = "yandex"
    QWEN = "qwen"
    GIGACHAT = "gigachat"
    HUGGINGFACE = "huggingface"


ALLOWED_CLOUD_MODELS: Mapping[LlmProvider, frozenset[str]] = {
    LlmProvider.OPENAI: frozenset({"gpt-4.1", "gpt-5.6-luna"}),
    LlmProvider.YANDEX: frozenset({"YandexGPT Lite 5", "YandexGPT Pro 5", "YandexGPT Pro 5.1"}),
    LlmProvider.QWEN: frozenset({"Qwen3.6 35B-A3B", "Qwen3 235B"}),
    LlmProvider.GIGACHAT: frozenset(
        {"GigaChat 2 Lite", "GigaChat 2 Pro", "GigaChat 2 Max"}
    ),
}


@dataclass(frozen=True, slots=True)
class LlmSelection:
    """One disclosed provider and model; automatic routing is deliberately absent."""

    provider: LlmProvider
    model: str

    def __post_init__(self) -> None:
        if not isinstance(self.provider, LlmProvider):
            raise ValueError("provider must be an LlmProvider")
        if not self.model.strip():
            raise ValueError("model must not be blank")
        if self.model.strip().casefold() == "auto":
            raise ValueError("automatic model selection is not approved")
        if self.provider is LlmProvider.HUGGINGFACE:
            return
        if self.model not in ALLOWED_CLOUD_MODELS[self.provider]:
            raise ValueError(
                f"model {self.model!r} is not approved for provider {self.provider.value!r}"
            )


@dataclass(frozen=True, slots=True)
class LlmRuntimeSettings:
    """Validated runtime settings whose secret is excluded from repr and comparison."""

    selection: LlmSelection
    api_key: str = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("NEXTWAVE_LLM_API_KEY must not be blank")


def load_llm_runtime_settings(environment: Mapping[str, str]) -> LlmRuntimeSettings:
    """Read one explicit model selection; no provider or model is chosen automatically."""

    provider_value = environment.get("NEXTWAVE_LLM_PROVIDER", "").strip()
    model = environment.get("NEXTWAVE_LLM_MODEL", "").strip()
    api_key = environment.get("NEXTWAVE_LLM_API_KEY", "").strip()
    if not provider_value:
        raise ValueError("NEXTWAVE_LLM_PROVIDER is required")
    if not model:
        raise ValueError("NEXTWAVE_LLM_MODEL is required")
    try:
        provider = LlmProvider(provider_value)
    except ValueError as error:
        raise ValueError(f"unknown LLM provider: {provider_value}") from error
    return LlmRuntimeSettings(
        selection=LlmSelection(provider, model),
        api_key=api_key,
    )


class JsonHttpTransport(Protocol):
    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> HttpResponse: ...


class UrllibJsonHttpTransport:
    """Small standard-library POST transport used by the LLM adapter."""

    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> HttpResponse:
        request = Request(
            url,
            headers=dict(headers),
            data=json.dumps(payload, ensure_ascii=False).encode(),
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                return HttpResponse(
                    status_code=response.status,
                    headers=dict(response.headers.items()),
                    body=response.read(),
                )
        except HTTPError as error:
            return HttpResponse(
                status_code=error.code,
                headers=dict(error.headers.items()) if error.headers is not None else {},
                body=error.read(),
            )


QUERY_INTERPRETATION_JSON_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "normalized_query": {"type": "string"},
        "search_texts": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 8,
        },
        "languages": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "granularity": {"type": "string", "enum": ["direction", "technology"]},
    },
    "required": ["normalized_query", "search_texts", "languages", "granularity"],
    "additionalProperties": False,
}


class OpenAIResponsesJsonGenerator:
    """Calls one disclosed GPT-4.1 model and extracts its structured JSON text."""

    def __init__(
        self,
        api_key: str,
        *,
        selection: LlmSelection | None = None,
        transport: JsonHttpTransport | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OpenAI API key must not be blank")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.selection = selection or LlmSelection(LlmProvider.OPENAI, OPENAI_QUERY_MODEL)
        if self.selection != LlmSelection(LlmProvider.OPENAI, OPENAI_QUERY_MODEL):
            raise ValueError("OpenAI query adapter supports only the approved gpt-4.1 model")
        self._api_key = api_key
        self._transport = transport or UrllibJsonHttpTransport()
        self._timeout_seconds = timeout_seconds

    def __call__(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("prompt must not be blank")
        response = self._transport.post_json(
            OPENAI_RESPONSES_ENDPOINT,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "NextWave/0.1",
            },
            payload={
                "model": self.selection.model,
                "input": prompt,
                "max_output_tokens": 500,
                "store": False,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "query_interpretation",
                        "strict": True,
                        "schema": QUERY_INTERPRETATION_JSON_SCHEMA,
                    }
                },
            },
            timeout_seconds=self._timeout_seconds,
        )
        if not 200 <= response.status_code <= 299:
            raise RuntimeError(f"OpenAI Responses API returned HTTP {response.status_code}")
        return parse_openai_output_text(response.body, self.selection.model)


def parse_openai_output_text(body: bytes, requested_model: str) -> str:
    """Extract one JSON text item and reject an undisclosed served model."""

    try:
        payload = json.loads(body.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("OpenAI response is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("OpenAI response must be a JSON object")
    served_model = payload.get("model")
    if not isinstance(served_model, str) or not (
        served_model == requested_model or served_model.startswith(f"{requested_model}-")
    ):
        raise ValueError("OpenAI response used an unexpected model")
    output = payload.get("output")
    if not isinstance(output, list):
        raise ValueError("OpenAI response must contain an output list")
    texts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "output_text"
                and isinstance(part.get("text"), str)
            ):
                texts.append(part["text"])
    if len(texts) != 1 or not texts[0].strip():
        raise ValueError("OpenAI response must contain exactly one non-empty output_text")
    return texts[0]
