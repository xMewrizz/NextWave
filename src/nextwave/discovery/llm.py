"""Explicit LLM selection and schema-constrained JSON adapters."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from nextwave.sources import HttpResponse

from .local_llm import LOCAL_MODEL_ID, ensure_local_llm_server

YANDEX_COMPLETION_ENDPOINT = (
    "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
)
YANDEX_ADAPTER_VERSION = "yandex-completion-v1"
QWEN_ADAPTER_VERSION = "alibaba-model-studio-openai-v1"
QWEN_MODEL_IDS: Mapping[str, str] = {
    "Qwen3.6 35B-A3B": "qwen3.6-35b-a3b",
    "Qwen3 235B": "qwen3-235b-a22b-instruct-2507",
}
YANDEX_MODEL_URIS: Mapping[str, str] = {
    "YandexGPT Lite 5": "yandexgpt-5-lite",
    "YandexGPT Pro 5": "yandexgpt-5-pro",
    "YandexGPT Pro 5.1": "yandexgpt-5.1",
}
LOCAL_ADAPTER_VERSION = "llama-cpp-qwen3-4b-instruct-2507-q4-v1"
_SCHEMA_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


def _normalize_qwen_base_url(value: str) -> str:
    """Allow only direct Alibaba Model Studio HTTPS endpoints required by the brief."""

    base_url = value.strip().rstrip("/")
    parsed = urlsplit(base_url)
    host = (parsed.hostname or "").casefold()
    if parsed.scheme != "https" or parsed.query or parsed.fragment:
        raise ValueError("NEXTWAVE_QWEN_BASE_URL must be an HTTPS URL")
    if host != "dashscope-intl.aliyuncs.com" and not host.endswith(
        ".ap-southeast-1.maas.aliyuncs.com"
    ):
        raise ValueError(
            "NEXTWAVE_QWEN_BASE_URL must be a direct Alibaba Model Studio "
            "Singapore endpoint"
        )
    if parsed.path != "/compatible-mode/v1":
        raise ValueError(
            "NEXTWAVE_QWEN_BASE_URL must end with /compatible-mode/v1"
        )
    return base_url


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
    """Validated runtime settings whose secrets are excluded from repr and comparison."""

    selection: LlmSelection
    api_key: str = field(default="", repr=False, compare=False)
    yandex_folder_id: str = field(default="", repr=False, compare=False)
    qwen_base_url: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.selection.provider is not LlmProvider.HUGGINGFACE and not self.api_key.strip():
            raise ValueError("NEXTWAVE_LLM_API_KEY must not be blank")
        if (
            self.selection.provider is LlmProvider.YANDEX
            and not self.yandex_folder_id.strip()
        ):
            raise ValueError("NEXTWAVE_YANDEX_FOLDER_ID must not be blank")
        if self.selection.provider is LlmProvider.QWEN:
            _normalize_qwen_base_url(self.qwen_base_url)


def load_llm_runtime_settings(environment: Mapping[str, str]) -> LlmRuntimeSettings:
    """Read one explicit model selection; no provider or model is chosen automatically."""

    provider_value = environment.get("NEXTWAVE_LLM_PROVIDER", "").strip()
    model = environment.get("NEXTWAVE_LLM_MODEL", "").strip()
    api_key = environment.get("NEXTWAVE_LLM_API_KEY", "").strip()
    yandex_folder_id = environment.get("NEXTWAVE_YANDEX_FOLDER_ID", "").strip()
    qwen_base_url = environment.get("NEXTWAVE_QWEN_BASE_URL", "").strip()
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
        qwen_base_url=qwen_base_url,
    )


def load_gate_llm_settings(environment: Mapping[str, str]) -> LlmRuntimeSettings:
    """Read the explicit Candidate Gate model selection.

    Both ``NEXTWAVE_GATE_LLM_PROVIDER`` and ``NEXTWAVE_GATE_LLM_MODEL`` must be
    set together, or both must be absent (then the main ``NEXTWAVE_LLM_*`` pair
    is reused for backward compatibility). Credentials are reused from the
    server settings and never enter results, reprs or error text.
    """

    gate_provider = environment.get("NEXTWAVE_GATE_LLM_PROVIDER", "").strip()
    gate_model = environment.get("NEXTWAVE_GATE_LLM_MODEL", "").strip()
    if not gate_provider and not gate_model:
        return load_llm_runtime_settings(environment)
    if bool(gate_provider) != bool(gate_model):
        missing = (
            "NEXTWAVE_GATE_LLM_MODEL"
            if gate_provider
            else "NEXTWAVE_GATE_LLM_PROVIDER"
        )
        raise ValueError(
            f"{missing} is required when the other NEXTWAVE_GATE_LLM_* variable is set"
        )
    api_key = environment.get("NEXTWAVE_LLM_API_KEY", "").strip()
    yandex_folder_id = environment.get("NEXTWAVE_YANDEX_FOLDER_ID", "").strip()
    qwen_base_url = environment.get("NEXTWAVE_QWEN_BASE_URL", "").strip()
    try:
        provider = LlmProvider(gate_provider)
    except ValueError as error:
        raise ValueError(f"unknown LLM provider: {gate_provider}") from error
    return LlmRuntimeSettings(
        selection=LlmSelection(provider, gate_model),
        api_key=api_key,
        yandex_folder_id=yandex_folder_id,
        qwen_base_url=qwen_base_url,
    )


def load_evidence_llm_settings(environment: Mapping[str, str]) -> LlmRuntimeSettings:
    """Read the optional Evidence extraction model selection.

    Both evidence variables must be set together. When both are absent the
    main model is reused for backward compatibility. Credentials remain the
    shared server credentials and never enter reprs or comparisons.
    """

    evidence_provider = environment.get("NEXTWAVE_EVIDENCE_LLM_PROVIDER", "").strip()
    evidence_model = environment.get("NEXTWAVE_EVIDENCE_LLM_MODEL", "").strip()
    if not evidence_provider and not evidence_model:
        return load_llm_runtime_settings(environment)
    if bool(evidence_provider) != bool(evidence_model):
        missing = (
            "NEXTWAVE_EVIDENCE_LLM_MODEL"
            if evidence_provider
            else "NEXTWAVE_EVIDENCE_LLM_PROVIDER"
        )
        raise ValueError(
            f"{missing} is required when the other "
            "NEXTWAVE_EVIDENCE_LLM_* variable is set"
        )
    api_key = environment.get("NEXTWAVE_LLM_API_KEY", "").strip()
    yandex_folder_id = environment.get("NEXTWAVE_YANDEX_FOLDER_ID", "").strip()
    qwen_base_url = environment.get("NEXTWAVE_QWEN_BASE_URL", "").strip()
    try:
        provider = LlmProvider(evidence_provider)
    except ValueError as error:
        raise ValueError(f"unknown LLM provider: {evidence_provider}") from error
    return LlmRuntimeSettings(
        selection=LlmSelection(provider, evidence_model),
        api_key=api_key,
        yandex_folder_id=yandex_folder_id,
        qwen_base_url=qwen_base_url,
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

    #: Bounded retries for transient network failures (a live run died on one).
    MAX_ATTEMPTS = 3
    BACKOFF_SECONDS = (5.0, 15.0)

    def __init__(
        self,
        *,
        max_attempts: int = MAX_ATTEMPTS,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        if (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
            or not 1 <= max_attempts <= 5
        ):
            raise ValueError("max_attempts must be between 1 and 5")
        self._max_attempts = max_attempts
        self._sleeper = sleeper or time.sleep

    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> HttpResponse:
        attempts = 0
        while True:
            attempts += 1
            try:
                return self._post_once(
                    url,
                    headers=headers,
                    payload=payload,
                    timeout_seconds=timeout_seconds,
                )
            except OSError as error:
                if attempts >= self._max_attempts:
                    raise RuntimeError(
                        f"LLM request failed after {attempts} attempts: {error}"
                    ) from error
                self._sleeper(
                    self.BACKOFF_SECONDS[
                        min(attempts - 1, len(self.BACKOFF_SECONDS) - 1)
                    ]
                )

    def _post_once(
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
        json_schema: Mapping[str, Any] | None = None,
        schema_name: str = "response",
        server_side_json_schema: bool = False,
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
        if _SCHEMA_NAME.fullmatch(schema_name) is None:
            raise ValueError("schema_name must be a lowercase JSON schema identifier")
        if json_schema is not None and not isinstance(json_schema, Mapping):
            raise ValueError("json_schema must be an object")
        if not isinstance(server_side_json_schema, bool):
            raise ValueError("server_side_json_schema must be a bool")
        if server_side_json_schema and json_schema is None:
            raise ValueError("server-side JSON Schema requires json_schema")
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
        self._json_schema = dict(json_schema) if json_schema is not None else None
        self._schema_name = schema_name
        self._server_side_json_schema = server_side_json_schema

    def __call__(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("prompt must not be blank")
        request_text = prompt
        if self._json_schema is not None and not self._server_side_json_schema:
            schema_text = json.dumps(self._json_schema, ensure_ascii=False)
            request_text = (
                f"Return only a JSON object matching this JSON Schema "
                f"({self._schema_name}):\n{schema_text}\n\n{prompt}"
            )
        payload: dict[str, Any] = {
            "modelUri": self._model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": 0,
                "maxTokens": str(self._max_output_tokens),
            },
            "messages": [{"role": "user", "text": request_text}],
        }
        if self._server_side_json_schema:
            payload["jsonSchema"] = {"schema": self._json_schema}
        else:
            payload["jsonObject"] = True
        response = self._transport.post_json(
            YANDEX_COMPLETION_ENDPOINT,
            headers={
                "Accept": "application/json",
                "Authorization": f"Api-Key {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "NextWave/0.1",
            },
            payload=payload,
            timeout_seconds=self._timeout_seconds,
        )
        if not 200 <= response.status_code <= 299:
            raise RuntimeError(
                f"Yandex completion API returned HTTP {response.status_code}"
            )
        return parse_yandex_completion_text(response.body, self.selection.model)


class QwenCompletionJsonGenerator:
    """Call one disclosed Qwen model directly through Alibaba Model Studio."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str,
        selection: LlmSelection,
        transport: JsonHttpTransport | None = None,
        timeout_seconds: float = 120.0,
        max_output_tokens: int = 500,
        json_schema: Mapping[str, Any] | None = None,
        schema_name: str = "response",
    ) -> None:
        if not api_key.strip():
            raise ValueError("Qwen API key must not be blank")
        normalized_base_url = _normalize_qwen_base_url(base_url)
        if selection.provider is not LlmProvider.QWEN:
            raise ValueError("Qwen adapter supports only Qwen models")
        try:
            model_id = QWEN_MODEL_IDS[selection.model]
        except KeyError as error:
            raise ValueError(
                f"Qwen adapter has no model ID for {selection.model!r}"
            ) from error
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if (
            isinstance(max_output_tokens, bool)
            or not isinstance(max_output_tokens, int)
            or not 1 <= max_output_tokens <= 10000
        ):
            raise ValueError("max_output_tokens must be between 1 and 10000")
        if _SCHEMA_NAME.fullmatch(schema_name) is None:
            raise ValueError("schema_name must be a lowercase JSON schema identifier")
        if json_schema is not None and not isinstance(json_schema, Mapping):
            raise ValueError("json_schema must be an object")
        self.selection = selection
        self._api_key = api_key.strip()
        self._endpoint = f"{normalized_base_url}/chat/completions"
        self._model_id = model_id
        self._transport = transport or UrllibJsonHttpTransport()
        self._timeout_seconds = timeout_seconds
        self._max_output_tokens = max_output_tokens
        self._json_schema = dict(json_schema) if json_schema is not None else None
        self._schema_name = schema_name

    def __call__(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("prompt must not be blank")
        request_text = prompt
        if self._json_schema is not None:
            schema_text = json.dumps(self._json_schema, ensure_ascii=False)
            request_text = (
                f"Return only a JSON object matching this JSON Schema "
                f"({self._schema_name}):\n{schema_text}\n\n{prompt}"
            )
        response = self._transport.post_json(
            self._endpoint,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "NextWave/0.1",
            },
            payload={
                "model": self._model_id,
                "messages": [{"role": "user", "content": request_text}],
                "temperature": 0,
                "max_tokens": self._max_output_tokens,
                "response_format": {"type": "json_object"},
            },
            timeout_seconds=self._timeout_seconds,
        )
        if not 200 <= response.status_code <= 299:
            raise RuntimeError(
                f"Alibaba Model Studio returned HTTP {response.status_code}"
            )
        return parse_openai_completion_text(response.body, self.selection.model)


class YandexFallbackJsonGenerator:
    """Try Yandex models in order; report the model that answered last.

    Lite deterministically mangles JSON formatting on some inputs (correct
    values, destroyed keys), so retrying the same model is useless: a parse
    failure replays the prompt on the next model. Transport errors and
    truncation propagate untouched. `.provider`/`.model` mirror the
    answering selection, so result manifests stay truthful.
    """

    def __init__(
        self, generators: tuple[YandexCompletionJsonGenerator, ...]
    ) -> None:
        if len(generators) < 2:
            raise ValueError("fallback requires at least two models")
        self._generators = generators
        self._current = generators[0].selection

    @property
    def provider(self) -> LlmProvider:
        return self._current.provider

    @property
    def model(self) -> str:
        return self._current.model

    def __call__(self, prompt: str) -> str:
        errors: list[ValueError] = []
        for generator in self._generators:
            try:
                text = generator(prompt)
            except YandexContentFilterError:
                # A refusal judges the content, not the format: a stronger
                # model refuses the same way, so do not waste the call.
                raise
            except YandexTruncationError:
                raise
            except ValueError as error:
                errors.append(error)
                continue
            self._current = generator.selection
            return text
        raise ValueError(
            "every Yandex model returned unparsable JSON"
        ) from errors[-1]


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
    server_side_json_schema: bool = False,
) -> YandexCompletionJsonGenerator | QwenCompletionJsonGenerator | LocalLlamaJsonGenerator:
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
            json_schema=json_schema,
            schema_name=schema_name,
            server_side_json_schema=server_side_json_schema,
        )
    if settings.selection.provider is LlmProvider.QWEN:
        return QwenCompletionJsonGenerator(
            settings.api_key,
            base_url=settings.qwen_base_url,
            selection=settings.selection,
            transport=transport,
            max_output_tokens=max_output_tokens,
            json_schema=json_schema,
            schema_name=schema_name,
        )
    raise ValueError(f"LLM adapter is not implemented for {settings.selection.provider.value!r}")


def _strip_json_fences(text: str) -> str:
    """Remove one Markdown code fence around a JSON payload, if present."""

    stripped = text.strip()
    match = re.fullmatch(r"```[a-zA-Z]*\n(.*)\n```", stripped, flags=re.DOTALL)
    if match is not None:
        return match.group(1).strip()
    return stripped


class YandexTruncationError(ValueError):
    """The model stopped mid-response: callers may retry with a smaller batch."""


class YandexContentFilterError(ValueError):
    """The provider refused the content: retrying or splitting is useless.

    Deliberately NOT a YandexTruncationError: the refusal is deterministic
    for the same content, so batch splitting would only multiply spend.
    Callers must skip with honest unknown coverage instead.
    """


def parse_openai_completion_text(body: bytes, requested_model: str) -> str:
    """Extract one JSON object from an OpenAI-compatible completion response."""

    try:
        payload = json.loads(body.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Qwen response is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("Qwen response must be a JSON object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Qwen response must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("Qwen response choice must be an object")
    finish_reason = choice.get("finish_reason")
    if finish_reason == "length":
        raise YandexTruncationError(
            f"Qwen {requested_model} response was truncated"
        )
    if finish_reason in {"content_filter", "content-filter"}:
        raise YandexContentFilterError(
            f"Qwen {requested_model} response was refused by content filter"
        )
    if finish_reason not in {None, "stop"}:
        raise ValueError(
            f"Qwen {requested_model} returned unsupported finish reason"
        )
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Qwen response must contain non-empty message content")
    cleaned = _strip_json_fences(content)
    try:
        parsed = json.loads(cleaned, strict=False)
    except json.JSONDecodeError as error:
        raise ValueError("Qwen response content must be a JSON object") from error
    if not isinstance(parsed, dict):
        raise ValueError("Qwen response content must be a JSON object")
    return _dump_canonical(parsed)


def _is_wrapper_junk(value: object) -> bool:
    """Junk wrapper keys carry nothing: "" or {"": ""} (both observed live)."""

    return value == "" or value == {"": ""}


def _dump_canonical(payload: dict) -> str:
    """Serialize to compact canonical JSON.

    Every success path exits through here, so downstream consumers always
    receive strict-parseable text no matter what the model emitted
    (whitespace, raw control characters, wrapper keys).
    """

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _unwrap_wrapper_payload(parsed: dict) -> dict | None:
    """Drop junk wrapper keys, keep the single substantial field.

    Observed live: {"<junk>": {"": ""}, "documents": [...], "<junk>": ""}.
    Legit responses never contain junk values, so any junk key means the
    model wrapped the payload: keep the one substantial field, reject
    anything else.
    """

    substantial = [
        (key, value) for key, value in parsed.items() if not _is_wrapper_junk(value)
    ]
    if len(substantial) != 1:
        return None
    if not any(_is_wrapper_junk(value) for _, value in parsed.items()):
        return None
    key, value = substantial[0]
    return {key: value}


def _unwrap_single_key_payload(only_key: str) -> dict | None:
    """Recover the payload when the model wraps the whole JSON as one string key.

    Observed live: the entire intended object arrives as a single escaped key
    with an empty value, sometimes prefixed with a `>>` turn marker and
    sometimes without the surrounding braces. Unwrap only when that key
    parses to a dict; anything else stays rejected.
    """

    candidate = _strip_json_fences(only_key.strip())
    if candidate.startswith(">>"):
        candidate = _strip_json_fences(candidate[2:].strip())
    for variant in (candidate, "{" + candidate + "}"):
        try:
            inner = json.loads(variant, strict=False)
        except json.JSONDecodeError:
            continue
        if isinstance(inner, dict):
            return inner
    return None


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
    status = alternatives[0].get("status") if isinstance(alternatives[0], dict) else None
    if isinstance(status, str) and "CONTENT_FILTER" in status:
        raise YandexContentFilterError(
            f"Yandex response was refused by content filter (status: {status})"
        )
    if isinstance(status, str) and status != "ALTERNATIVE_STATUS_FINAL":
        raise YandexTruncationError(f"Yandex response was not final (status: {status})")
    cleaned = _strip_json_fences(text)
    try:
        # strict=False: the model sometimes emits raw control characters
        # (e.g. newlines) inside string values (observed live). Structure
        # errors still fail; downstream validators check the shape.
        parsed = json.loads(cleaned, strict=False)
    except json.JSONDecodeError as error:
        raise ValueError(
            "Yandex response text must be a JSON object "
            f"(preview: {cleaned[:200]!r})"
        ) from error
    if not isinstance(parsed, dict):
        raise ValueError(
            "Yandex response text must be a JSON object "
            f"(preview: {cleaned[:200]!r})"
        )
    if len(parsed) == 1:
        ((only_key, only_value),) = parsed.items()
        if _is_wrapper_junk(only_value):
            unwrapped = _unwrap_single_key_payload(only_key)
            if unwrapped is None:
                raise ValueError(
                    "Yandex response text must be a JSON object "
                    f"(preview: {cleaned[:200]!r})"
                )
            return _dump_canonical(unwrapped)
    unwrapped = _unwrap_wrapper_payload(parsed)
    if unwrapped is not None:
        return _dump_canonical(unwrapped)
    if any(_is_wrapper_junk(value) for value in parsed.values()):
        raise ValueError(
            "Yandex response text must be a JSON object "
            f"(preview: {cleaned[:200]!r})"
        )
    return _dump_canonical(parsed)
