"""Explicit LLM selection and schema-constrained JSON adapters."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from nextwave.sources import HttpResponse

from .local_llm import LOCAL_MODEL_ID, ensure_local_llm_server

YANDEX_COMPLETION_ENDPOINT = (
    "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
)
YANDEX_ADAPTER_VERSION = "yandex-completion-v1"
YANDEX_MODEL_URIS: Mapping[str, str] = {
    "YandexGPT Lite 5": "yandexgpt-lite/latest",
    "YandexGPT Pro 5": "yandexgpt/latest",
    # Pro 5.1 идёт в URI дословно: тихой подмены на latest нет, неверный
    # сегмент даст громкую 400 — подтвердить по консоли при первом живом вызове.
    "YandexGPT Pro 5.1": "yandexgpt/5.1",
}
LOCAL_ADAPTER_VERSION = "llama-cpp-qwen3-4b-instruct-2507-q4-v1"
_SCHEMA_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


class LlmProvider(StrEnum):
    YANDEX = "yandex"
    QWEN = "qwen"
    GIGACHAT = "gigachat"
    HUGGINGFACE = "huggingface"


ALLOWED_CLOUD_MODELS: Mapping[LlmProvider, frozenset[str]] = {
    LlmProvider.YANDEX: frozenset({"YandexGPT Lite 5", "YandexGPT Pro 5", "YandexGPT Pro 5.1"}),
    LlmProvider.QWEN: frozenset({"Qwen3.6 35B-A3B", "Qwen3 235B"}),
    LlmProvider.GIGACHAT: frozenset({"GigaChat 2 Lite", "GigaChat 2 Pro", "GigaChat 2 Max"}),
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
    api_key: str = field(default="", repr=False, compare=False)
    yandex_folder_id: str = ""

    def __post_init__(self) -> None:
        if self.selection.provider is not LlmProvider.HUGGINGFACE and not self.api_key.strip():
            raise ValueError("NEXTWAVE_LLM_API_KEY must not be blank")
        if (
            self.selection.provider is LlmProvider.YANDEX
            and not self.yandex_folder_id.strip()
        ):
            raise ValueError("NEXTWAVE_YANDEX_FOLDER_ID must not be blank")


def load_llm_runtime_settings(environment: Mapping[str, str]) -> LlmRuntimeSettings:
    """Read one explicit model selection; no provider or model is chosen automatically."""

    provider_value = environment.get("NEXTWAVE_LLM_PROVIDER", "").strip()
    model = environment.get("NEXTWAVE_LLM_MODEL", "").strip()
    api_key = environment.get("NEXTWAVE_LLM_API_KEY", "").strip()
    yandex_folder_id = environment.get("NEXTWAVE_YANDEX_FOLDER_ID", "").strip()
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
        yandex_folder_id=yandex_folder_id,
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


class YandexCompletionJsonGenerator:
    """Calls one disclosed YandexGPT model and extracts its single JSON text."""

    def __init__(
        self,
        api_key: str,
        *,
        folder_id: str,
        selection: LlmSelection | None = None,
        transport: JsonHttpTransport | None = None,
        timeout_seconds: float = 60.0,
        max_output_tokens: int = 500,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Yandex API key must not be blank")
        if not folder_id.strip():
            raise ValueError("Yandex folder ID must not be blank")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or not 1 <= max_output_tokens <= 10000
        ):
            raise ValueError("max_output_tokens must be between 1 and 10000")
        self.selection = selection or LlmSelection(
            LlmProvider.YANDEX, "YandexGPT Lite 5"
        )
        if self.selection.provider is not LlmProvider.YANDEX:
            raise ValueError("Yandex adapter supports only YandexGPT models")
        try:
            model_path = YANDEX_MODEL_URIS[self.selection.model]
        except KeyError as error:
            raise ValueError(
                f"Yandex adapter has no model URI for {self.selection.model!r}"
            ) from error
        self._api_key = api_key
        self._model_uri = f"gpt://{folder_id.strip()}/{model_path}"
        self._transport = transport or UrllibJsonHttpTransport()
        self._timeout_seconds = timeout_seconds
        self._max_output_tokens = max_output_tokens

    def __call__(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("prompt must not be blank")
        response = self._transport.post_json(
            YANDEX_COMPLETION_ENDPOINT,
            headers={
                "Accept": "application/json",
                "Authorization": f"Api-Key {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "NextWave/0.1",
            },
            payload={
                "modelUri": self._model_uri,
                "completionOptions": {
                    "stream": False,
                    "temperature": 0,
                    "maxTokens": str(self._max_output_tokens),
                },
                "messages": [{"role": "user", "text": prompt}],
                "jsonObject": True,
            },
            timeout_seconds=self._timeout_seconds,
        )
        if not 200 <= response.status_code <= 299:
            raise RuntimeError(
                f"Yandex completion API returned HTTP {response.status_code}"
            )
        return parse_yandex_completion_text(response.body, self.selection.model)


class LocalLlamaJsonGenerator:
    """Use the pinned local Qwen model through llama.cpp's JSON chat API."""

    def __init__(
        self,
        *,
        selection: LlmSelection,
        transport: JsonHttpTransport | None = None,
        server_start: Callable[[], str] = ensure_local_llm_server,
        json_schema: Mapping[str, Any] = QUERY_INTERPRETATION_JSON_SCHEMA,
        schema_name: str = "query_interpretation",
        max_output_tokens: int = 500,
        timeout_seconds: float = 300.0,
    ) -> None:
        if selection != LlmSelection(LlmProvider.HUGGINGFACE, LOCAL_MODEL_ID):
            raise ValueError(f"Local LLM adapter supports only {LOCAL_MODEL_ID}")
        if _SCHEMA_NAME.fullmatch(schema_name) is None:
            raise ValueError("schema_name must be a lowercase JSON schema identifier")
        if not 1 <= max_output_tokens <= 10000:
            raise ValueError("max_output_tokens must be between 1 and 10000")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.selection = selection
        self._transport = transport or UrllibJsonHttpTransport()
        self._server_start = server_start
        self._json_schema = dict(json_schema)
        self._schema_name = schema_name
        self._max_output_tokens = max_output_tokens
        self._timeout_seconds = timeout_seconds

    def __call__(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("prompt must not be blank")
        root = self._server_start()
        response = self._transport.post_json(
            f"{root}/v1/chat/completions",
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            payload={
                "model": self.selection.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": self._max_output_tokens,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": self._schema_name,
                        "strict": True,
                        "schema": self._json_schema,
                    },
                },
            },
            timeout_seconds=self._timeout_seconds,
        )
        if not 200 <= response.status_code <= 299:
            raise RuntimeError(f"Local LLM returned HTTP {response.status_code}")
        try:
            payload = json.loads(response.body.decode())
            choice = payload["choices"][0]
            content = choice["message"]["content"]
            if choice.get("finish_reason") == "length":
                raise ValueError("Local LLM output was truncated")
            if not isinstance(content, str) or not isinstance(json.loads(content), dict):
                raise ValueError("Local LLM output must be a JSON object")
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
            raise ValueError("Local LLM returned invalid structured JSON") from error
        return content


def build_json_generator(
    settings: LlmRuntimeSettings,
    *,
    transport: JsonHttpTransport | None = None,
    json_schema: Mapping[str, Any] = QUERY_INTERPRETATION_JSON_SCHEMA,
    schema_name: str = "query_interpretation",
    max_output_tokens: int = 500,
) -> YandexCompletionJsonGenerator | LocalLlamaJsonGenerator:
    """Build the selected supported adapter without exposing credentials."""
    if settings.selection.provider is LlmProvider.HUGGINGFACE:
        return LocalLlamaJsonGenerator(
            selection=settings.selection,
            transport=transport,
            json_schema=json_schema,
            schema_name=schema_name,
            max_output_tokens=max_output_tokens,
        )
    if settings.selection.provider is LlmProvider.YANDEX:
        return YandexCompletionJsonGenerator(
            settings.api_key,
            folder_id=settings.yandex_folder_id,
            selection=settings.selection,
            transport=transport,
            max_output_tokens=max_output_tokens,
        )
    raise ValueError(f"LLM adapter is not implemented for {settings.selection.provider.value!r}")


def parse_yandex_completion_text(body: bytes, requested_model: str) -> str:
    """Extract the single alternative text and require a JSON object payload."""

    try:
        payload = json.loads(body.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Yandex response is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("Yandex response must be a JSON object")
    result = payload.get("result")
    alternatives = result.get("alternatives") if isinstance(result, dict) else None
    if not isinstance(alternatives, list) or len(alternatives) != 1:
        raise ValueError("Yandex response must contain exactly one alternative")
    message = alternatives[0].get("message") if isinstance(alternatives[0], dict) else None
    text = message.get("text") if isinstance(message, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Yandex response must contain non-empty message text")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("Yandex response text must be a JSON object") from error
    if not isinstance(parsed, dict):
        raise ValueError("Yandex response text must be a JSON object")
    return text
