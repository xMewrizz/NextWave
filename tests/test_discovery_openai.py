"""Official OpenAI adapter for the disclosed GPT-5.6 Luna model."""

from __future__ import annotations

import json
import unittest

from nextwave.discovery.llm import (
    LlmProvider,
    OpenAICompletionJsonGenerator,
    YandexContentFilterError,
    YandexTruncationError,
    build_json_generator,
    load_llm_runtime_settings,
    parse_openai_completion_text,
)
from nextwave.discovery.query_resolver import build_query_resolver_from_environment
from nextwave.sources import HttpResponse


class RecordingTransport:
    def __init__(self, response: HttpResponse) -> None:
        self.response = response
        self.calls: list[dict] = []

    def post_json(self, url, *, headers, payload, timeout_seconds):
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": dict(payload),
                "timeout_seconds": timeout_seconds,
            }
        )
        return self.response


def completion(content: str, *, finish_reason: str = "stop") -> bytes:
    return json.dumps(
        {
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {"role": "assistant", "content": content},
                }
            ]
        }
    ).encode()


class OpenAISettingsTests(unittest.TestCase):
    def test_openai_uses_dedicated_key_and_builds_adapter(self) -> None:
        settings = load_llm_runtime_settings(
            {
                "NEXTWAVE_LLM_PROVIDER": "openai",
                "NEXTWAVE_LLM_MODEL": "GPT-5.6 Luna",
                "NEXTWAVE_OPENAI_API_KEY": "openai-secret",
                "NEXTWAVE_LLM_API_KEY": "legacy-secret",
            }
        )
        self.assertEqual(settings.selection.provider, LlmProvider.OPENAI)
        self.assertIsInstance(build_json_generator(settings), OpenAICompletionJsonGenerator)
        self.assertEqual(settings.api_key, "openai-secret")
        self.assertNotIn("openai-secret", repr(settings))
        self.assertNotIn("legacy-secret", repr(settings))

    def test_openai_requires_dedicated_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "NEXTWAVE_OPENAI_API_KEY"):
            load_llm_runtime_settings(
                {
                    "NEXTWAVE_LLM_PROVIDER": "openai",
                    "NEXTWAVE_LLM_MODEL": "GPT-5.6 Luna",
                    "NEXTWAVE_LLM_API_KEY": "legacy-secret",
                }
            )

    def test_query_resolver_records_openai_adapter_version(self) -> None:
        resolver = build_query_resolver_from_environment(
            {
                "NEXTWAVE_LLM_PROVIDER": "openai",
                "NEXTWAVE_LLM_MODEL": "GPT-5.6 Luna",
                "NEXTWAVE_OPENAI_API_KEY": "openai-secret",
            }
        )
        self.assertEqual(
            resolver._interpreter._version,
            "openai-chat-completions-v1",
        )


class OpenAIAdapterTests(unittest.TestCase):
    def _settings(self):
        return load_llm_runtime_settings(
            {
                "NEXTWAVE_LLM_PROVIDER": "openai",
                "NEXTWAVE_LLM_MODEL": "GPT-5.6 Luna",
                "NEXTWAVE_OPENAI_API_KEY": "openai-secret",
            }
        )

    def test_request_uses_pinned_model_and_strict_schema(self) -> None:
        transport = RecordingTransport(
            HttpResponse(200, {}, completion('{"ok":true}'))
        )
        schema = {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        }
        generator = build_json_generator(
            self._settings(),
            transport=transport,
            json_schema=schema,
            schema_name="answer",
            max_output_tokens=123,
        )

        self.assertEqual(generator("Return JSON"), '{"ok":true}')
        call = transport.calls[0]
        self.assertEqual(call["url"], "https://api.openai.com/v1/chat/completions")
        self.assertEqual(call["headers"]["Authorization"], "Bearer openai-secret")
        self.assertEqual(call["payload"]["model"], "gpt-5.6-luna")
        self.assertEqual(call["payload"]["reasoning_effort"], "none")
        self.assertEqual(call["payload"]["max_completion_tokens"], 123)
        response_format = call["payload"]["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertEqual(response_format["json_schema"]["schema"], schema)
        self.assertEqual(call["payload"]["messages"][0]["content"], "Return JSON")
        self.assertNotIn("openai-secret", json.dumps(call["payload"]))

    def test_http_error_does_not_echo_body_or_secret(self) -> None:
        transport = RecordingTransport(
            HttpResponse(401, {}, b'{"message":"Bearer openai-secret"}')
        )
        generator = build_json_generator(self._settings(), transport=transport)
        with self.assertRaisesRegex(RuntimeError, "HTTP 401") as context:
            generator("Return JSON")
        self.assertNotIn("openai-secret", str(context.exception))

    def test_parser_names_openai_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "OpenAI response"):
            parse_openai_completion_text(
                b"broken", "GPT-5.6 Luna", provider_name="OpenAI"
            )
        with self.assertRaises(YandexTruncationError):
            parse_openai_completion_text(
                completion("{}", finish_reason="length"),
                "GPT-5.6 Luna",
                provider_name="OpenAI",
            )
        with self.assertRaises(YandexContentFilterError):
            parse_openai_completion_text(
                completion("{}", finish_reason="content_filter"),
                "GPT-5.6 Luna",
                provider_name="OpenAI",
            )


if __name__ == "__main__":
    unittest.main()
