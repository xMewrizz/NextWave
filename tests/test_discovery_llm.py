from __future__ import annotations

import json
import unittest
from collections.abc import Mapping
from typing import Any

from nextwave.discovery import (
    OPENAI_RESPONSES_ENDPOINT,
    LlmProvider,
    LlmSelection,
    OpenAIResponsesJsonGenerator,
    load_llm_runtime_settings,
    parse_openai_output_text,
)
from nextwave.discovery.llm import LocalLlamaJsonGenerator, build_json_generator
from nextwave.discovery.local_llm import LOCAL_MODEL_ID
from nextwave.sources import HttpResponse


class FakeJsonTransport:
    def __init__(self, response: HttpResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, Mapping[str, str], Mapping[str, Any], float]] = []

    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> HttpResponse:
        self.calls.append((url, headers, payload, timeout_seconds))
        return self.response


def response_body(*, model: str = "gpt-4.1-2025-04-14") -> bytes:
    return json.dumps(
        {
            "id": "resp_test",
            "model": model,
            "output": [
                {"type": "reasoning", "id": "reasoning_test"},
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"normalized_query":"artificial intelligence"}',
                        }
                    ],
                },
            ],
        }
    ).encode()


class LlmSelectionTests(unittest.TestCase):
    def test_accepts_models_explicitly_listed_in_the_hackathon_specification(self) -> None:
        approved = (
            (LlmProvider.OPENAI, "gpt-4.1"),
            (LlmProvider.OPENAI, "gpt-5.6-luna"),
            (LlmProvider.YANDEX, "YandexGPT Lite 5"),
            (LlmProvider.YANDEX, "YandexGPT Pro 5"),
            (LlmProvider.YANDEX, "YandexGPT Pro 5.1"),
            (LlmProvider.QWEN, "Qwen3.6 35B-A3B"),
            (LlmProvider.QWEN, "Qwen3 235B"),
            (LlmProvider.GIGACHAT, "GigaChat 2 Lite"),
            (LlmProvider.GIGACHAT, "GigaChat 2 Pro"),
            (LlmProvider.GIGACHAT, "GigaChat 2 Max"),
        )

        for provider, model in approved:
            with self.subTest(provider=provider, model=model):
                self.assertEqual(LlmSelection(provider, model).model, model)

    def test_rejects_unapproved_cloud_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "not approved"):
            LlmSelection(LlmProvider.OPENAI, "gpt-5-mini")

    def test_rejects_automatic_routing_for_cloud_and_local_providers(self) -> None:
        for provider in (LlmProvider.OPENAI, LlmProvider.HUGGINGFACE):
            with self.subTest(provider=provider):
                with self.assertRaisesRegex(ValueError, "automatic model selection"):
                    LlmSelection(provider, "auto")

    def test_allows_explicit_local_hugging_face_model(self) -> None:
        selection = LlmSelection(LlmProvider.HUGGINGFACE, "Qwen/Qwen3-4B")

        self.assertEqual(selection.model, "Qwen/Qwen3-4B")

    def test_runtime_settings_do_not_expose_api_key_in_repr(self) -> None:
        settings = load_llm_runtime_settings(
            {
                "NEXTWAVE_LLM_PROVIDER": "openai",
                "NEXTWAVE_LLM_MODEL": "gpt-4.1",
                "NEXTWAVE_LLM_API_KEY": "temporary-secret",
            }
        )

        self.assertEqual(settings.selection.model, "gpt-4.1")
        self.assertNotIn("temporary-secret", repr(settings))

    def test_runtime_settings_require_an_explicit_provider_model_and_key(self) -> None:
        for missing in (
            "NEXTWAVE_LLM_PROVIDER",
            "NEXTWAVE_LLM_MODEL",
            "NEXTWAVE_LLM_API_KEY",
        ):
            environment = {
                "NEXTWAVE_LLM_PROVIDER": "openai",
                "NEXTWAVE_LLM_MODEL": "gpt-4.1",
                "NEXTWAVE_LLM_API_KEY": "temporary-secret",
            }
            environment.pop(missing)
            with self.subTest(missing=missing):
                with self.assertRaisesRegex(ValueError, missing):
                    load_llm_runtime_settings(environment)

    def test_local_model_requires_no_cloud_key(self) -> None:
        settings = load_llm_runtime_settings(
            {
                "NEXTWAVE_LLM_PROVIDER": "huggingface",
                "NEXTWAVE_LLM_MODEL": LOCAL_MODEL_ID,
            }
        )

        self.assertEqual(settings.api_key, "")
        self.assertIsInstance(build_json_generator(settings), LocalLlamaJsonGenerator)


class LocalLlamaJsonGeneratorTests(unittest.TestCase):
    def test_uses_loopback_json_schema_without_an_api_key(self) -> None:
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {"content": '{"normalized_query":"artificial intelligence"}'},
                        "finish_reason": "stop",
                    }
                ]
            }
        ).encode()
        transport = FakeJsonTransport(HttpResponse(200, {}, body))
        generator = LocalLlamaJsonGenerator(
            selection=LlmSelection(LlmProvider.HUGGINGFACE, LOCAL_MODEL_ID),
            transport=transport,
            server_start=lambda: "http://127.0.0.1:18080",
        )

        result = generator("Interpret AI")

        url, headers, payload, timeout = transport.calls[0]
        self.assertEqual(result, '{"normalized_query":"artificial intelligence"}')
        self.assertEqual(url, "http://127.0.0.1:18080/v1/chat/completions")
        self.assertNotIn("Authorization", headers)
        self.assertEqual(payload["model"], LOCAL_MODEL_ID)
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertEqual(timeout, 300)

    def test_rejects_truncated_json(self) -> None:
        body = json.dumps(
            {"choices": [{"message": {"content": "{}"}, "finish_reason": "length"}]}
        ).encode()
        generator = LocalLlamaJsonGenerator(
            selection=LlmSelection(LlmProvider.HUGGINGFACE, LOCAL_MODEL_ID),
            transport=FakeJsonTransport(HttpResponse(200, {}, body)),
            server_start=lambda: "http://127.0.0.1:18080",
        )

        with self.assertRaisesRegex(ValueError, "truncated"):
            generator("Interpret AI")


class OpenAIResponsesJsonGeneratorTests(unittest.TestCase):
    def test_sends_strict_schema_without_putting_key_in_payload(self) -> None:
        transport = FakeJsonTransport(HttpResponse(200, {}, response_body()))
        generator = OpenAIResponsesJsonGenerator(
            "temporary-secret",
            transport=transport,
            timeout_seconds=12,
        )

        result = generator("Normalize this query")

        url, headers, payload, timeout = transport.calls[0]
        self.assertEqual(result, '{"normalized_query":"artificial intelligence"}')
        self.assertEqual(url, OPENAI_RESPONSES_ENDPOINT)
        self.assertEqual(headers["Authorization"], "Bearer temporary-secret")
        self.assertEqual(payload["model"], "gpt-4.1")
        self.assertFalse(payload["store"])
        self.assertTrue(payload["text"]["format"]["strict"])
        self.assertNotIn("temporary-secret", json.dumps(payload))
        self.assertEqual(timeout, 12)

    def test_accepts_a_task_specific_strict_schema_and_output_budget(self) -> None:
        transport = FakeJsonTransport(HttpResponse(200, {}, response_body()))
        schema = {
            "type": "object",
            "properties": {"documents": {"type": "array"}},
            "required": ["documents"],
            "additionalProperties": False,
        }
        generator = OpenAIResponsesJsonGenerator(
            "temporary-secret",
            transport=transport,
            json_schema=schema,
            schema_name="candidate_mentions",
            max_output_tokens=2000,
        )

        generator("Extract grounded mentions")

        payload = transport.calls[0][2]
        self.assertEqual(payload["max_output_tokens"], 2000)
        self.assertEqual(payload["text"]["format"]["name"], "candidate_mentions")
        self.assertEqual(payload["text"]["format"]["schema"], schema)

    def test_refuses_other_approved_models_until_their_adapter_is_implemented(self) -> None:
        with self.assertRaisesRegex(ValueError, "supports only"):
            OpenAIResponsesJsonGenerator(
                "temporary-secret",
                selection=LlmSelection(LlmProvider.OPENAI, "gpt-5.6-luna"),
            )

    def test_does_not_expose_error_body_or_secret_on_http_failure(self) -> None:
        transport = FakeJsonTransport(
            HttpResponse(401, {}, b'{"error":{"message":"temporary-secret"}}')
        )
        generator = OpenAIResponsesJsonGenerator("temporary-secret", transport=transport)

        with self.assertRaisesRegex(RuntimeError, "HTTP 401") as caught:
            generator("Normalize this query")

        self.assertNotIn("temporary-secret", str(caught.exception))

    def test_reports_billing_error_code_without_exposing_error_text(self) -> None:
        transport = FakeJsonTransport(
            HttpResponse(
                429,
                {},
                b'{"error":{"code":"credit_balance_exhausted","message":"temporary-secret"}}',
            )
        )
        generator = OpenAIResponsesJsonGenerator("temporary-secret", transport=transport)

        with self.assertRaisesRegex(RuntimeError, "credit_balance_exhausted") as caught:
            generator("Return a short answer")

        self.assertNotIn("temporary-secret", str(caught.exception))


class OpenAIResponseParserTests(unittest.TestCase):
    def test_scans_message_items_instead_of_assuming_first_output(self) -> None:
        result = parse_openai_output_text(response_body(), "gpt-4.1")

        self.assertEqual(result, '{"normalized_query":"artificial intelligence"}')

    def test_rejects_an_unexpected_served_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "unexpected model"):
            parse_openai_output_text(response_body(model="gpt-5-mini"), "gpt-4.1")


if __name__ == "__main__":
    unittest.main()
