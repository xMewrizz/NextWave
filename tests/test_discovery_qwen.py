"""Direct Alibaba Model Studio adapter for the disclosed Qwen model."""

from __future__ import annotations

import json
import unittest

from nextwave.discovery.llm import (
    LlmProvider,
    QwenCompletionJsonGenerator,
    YandexContentFilterError,
    YandexTruncationError,
    build_json_generator,
    load_llm_runtime_settings,
    parse_openai_completion_text,
)
from nextwave.sources import HttpResponse

BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


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


class QwenSettingsTests(unittest.TestCase):
    def test_direct_qwen_settings_build_adapter(self) -> None:
        settings = load_llm_runtime_settings(
            {
                "NEXTWAVE_LLM_PROVIDER": "qwen",
                "NEXTWAVE_LLM_MODEL": "Qwen3 235B",
                "NEXTWAVE_LLM_API_KEY": "secret-value",
                "NEXTWAVE_QWEN_BASE_URL": BASE_URL,
            }
        )
        self.assertEqual(settings.selection.provider, LlmProvider.QWEN)
        self.assertIsInstance(build_json_generator(settings), QwenCompletionJsonGenerator)
        self.assertNotIn("secret-value", repr(settings))

    def test_qwen_requires_official_https_compatible_base_url(self) -> None:
        base = {
            "NEXTWAVE_LLM_PROVIDER": "qwen",
            "NEXTWAVE_LLM_MODEL": "Qwen3 235B",
            "NEXTWAVE_LLM_API_KEY": "secret-value",
        }
        for value in (
            "",
            "http://example.test/compatible-mode/v1",
            "https://example.test/v1",
            "https://openrouter.ai/compatible-mode/v1",
        ):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "NEXTWAVE_QWEN_BASE_URL"
            ):
                load_llm_runtime_settings(base | {"NEXTWAVE_QWEN_BASE_URL": value})


class QwenAdapterTests(unittest.TestCase):
    def test_request_uses_pinned_model_bearer_and_json_object(self) -> None:
        transport = RecordingTransport(
            HttpResponse(200, {"Content-Type": "application/json"}, completion('{"ok":true}'))
        )
        settings = load_llm_runtime_settings(
            {
                "NEXTWAVE_LLM_PROVIDER": "qwen",
                "NEXTWAVE_LLM_MODEL": "Qwen3 235B",
                "NEXTWAVE_LLM_API_KEY": "secret-value",
                "NEXTWAVE_QWEN_BASE_URL": BASE_URL,
            }
        )
        generator = build_json_generator(
            settings,
            transport=transport,
            json_schema={"type": "object"},
            schema_name="answer",
            max_output_tokens=123,
        )

        self.assertEqual(generator("Return JSON"), '{"ok":true}')
        call = transport.calls[0]
        self.assertEqual(call["url"], f"{BASE_URL}/chat/completions")
        self.assertEqual(call["headers"]["Authorization"], "Bearer secret-value")
        self.assertEqual(call["payload"]["model"], "qwen3-235b-a22b-instruct-2507")
        self.assertEqual(call["payload"]["response_format"], {"type": "json_object"})
        self.assertEqual(call["payload"]["temperature"], 0)
        self.assertEqual(call["payload"]["max_tokens"], 123)
        serialized = json.dumps(call["payload"])
        self.assertNotIn("secret-value", serialized)
        self.assertIn("JSON Schema", call["payload"]["messages"][0]["content"])

    def test_http_error_does_not_echo_provider_body_or_secret(self) -> None:
        transport = RecordingTransport(
            HttpResponse(401, {}, b'{"message":"Bearer secret-value"}')
        )
        settings = load_llm_runtime_settings(
            {
                "NEXTWAVE_LLM_PROVIDER": "qwen",
                "NEXTWAVE_LLM_MODEL": "Qwen3 235B",
                "NEXTWAVE_LLM_API_KEY": "secret-value",
                "NEXTWAVE_QWEN_BASE_URL": BASE_URL,
            }
        )
        generator = build_json_generator(settings, transport=transport)
        with self.assertRaisesRegex(RuntimeError, "HTTP 401") as context:
            generator("Return JSON")
        self.assertNotIn("secret-value", str(context.exception))

    def test_parser_accepts_fenced_object(self) -> None:
        result = parse_openai_completion_text(
            completion('```json\n{"answer":"ok"}\n```'), "Qwen3 235B"
        )
        self.assertEqual(result, '{"answer":"ok"}')

    def test_parser_rejects_non_object(self) -> None:
        with self.assertRaisesRegex(ValueError, "JSON object"):
            parse_openai_completion_text(completion("[1,2]"), "Qwen3 235B")

    def test_parser_classifies_truncation_and_content_filter(self) -> None:
        with self.assertRaises(YandexTruncationError):
            parse_openai_completion_text(
                completion('{"ok":true}', finish_reason="length"), "Qwen3 235B"
            )
        with self.assertRaises(YandexContentFilterError):
            parse_openai_completion_text(
                completion('{"ok":true}', finish_reason="content_filter"),
                "Qwen3 235B",
            )


if __name__ == "__main__":
    unittest.main()
