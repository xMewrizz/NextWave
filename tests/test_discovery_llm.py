from __future__ import annotations

import json
import unittest
from collections.abc import Mapping
from typing import Any
from urllib.error import URLError

from nextwave.discovery import (
    LlmProvider,
    LlmSelection,
    UrllibJsonHttpTransport,
    load_llm_runtime_settings,
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


class LlmSelectionTests(unittest.TestCase):
    def test_accepts_models_explicitly_listed_in_the_hackathon_specification(self) -> None:
        approved = (
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
            LlmSelection(LlmProvider.YANDEX, "YandexGPT Ultra")

    def test_rejects_automatic_routing_for_cloud_and_local_providers(self) -> None:
        for provider in (LlmProvider.YANDEX, LlmProvider.HUGGINGFACE):
            with self.subTest(provider=provider):
                with self.assertRaisesRegex(ValueError, "automatic model selection"):
                    LlmSelection(provider, "auto")

    def test_allows_explicit_local_hugging_face_model(self) -> None:
        selection = LlmSelection(LlmProvider.HUGGINGFACE, "Qwen/Qwen3-4B")

        self.assertEqual(selection.model, "Qwen/Qwen3-4B")

    def test_runtime_settings_do_not_expose_secrets_in_repr(self) -> None:
        settings = load_llm_runtime_settings(
            {
                "NEXTWAVE_LLM_PROVIDER": "yandex",
                "NEXTWAVE_LLM_MODEL": "YandexGPT Lite 5",
                "NEXTWAVE_LLM_API_KEY": "temporary-secret",
                "NEXTWAVE_YANDEX_FOLDER_ID": "folder-1",
            }
        )

        self.assertEqual(settings.selection.model, "YandexGPT Lite 5")
        self.assertNotIn("temporary-secret", repr(settings))
        self.assertNotIn("folder-1", repr(settings))

    def test_runtime_settings_exclude_secrets_from_comparison(self) -> None:
        first = load_llm_runtime_settings(
            {
                "NEXTWAVE_LLM_PROVIDER": "yandex",
                "NEXTWAVE_LLM_MODEL": "YandexGPT Lite 5",
                "NEXTWAVE_LLM_API_KEY": "secret-one",
                "NEXTWAVE_YANDEX_FOLDER_ID": "folder-one",
            }
        )
        second = load_llm_runtime_settings(
            {
                "NEXTWAVE_LLM_PROVIDER": "yandex",
                "NEXTWAVE_LLM_MODEL": "YandexGPT Lite 5",
                "NEXTWAVE_LLM_API_KEY": "secret-two",
                "NEXTWAVE_YANDEX_FOLDER_ID": "folder-two",
            }
        )

        self.assertEqual(first, second)

    def test_runtime_settings_require_an_explicit_provider_model_and_key(self) -> None:
        for missing in (
            "NEXTWAVE_LLM_PROVIDER",
            "NEXTWAVE_LLM_MODEL",
            "NEXTWAVE_LLM_API_KEY",
        ):
            environment = {
                "NEXTWAVE_LLM_PROVIDER": "yandex",
                "NEXTWAVE_LLM_MODEL": "YandexGPT Lite 5",
                "NEXTWAVE_LLM_API_KEY": "temporary-secret",
                "NEXTWAVE_YANDEX_FOLDER_ID": "folder-1",
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


class UrllibTransportRetryTests(unittest.TestCase):
    def test_network_failures_retry_with_backoff_then_succeed(self) -> None:
        response = HttpResponse(200, {"Content-Type": "application/json"}, b"{}")
        calls: list[str] = []
        pauses: list[float] = []
        transport = UrllibJsonHttpTransport(sleeper=pauses.append)

        def flaky_post_once(url, *, headers, payload, timeout_seconds):
            calls.append(url)
            if len(calls) < 3:
                raise URLError("temporary DNS failure")
            return response

        transport._post_once = flaky_post_once
        result = transport.post_json(
            "https://example.org/api",
            headers={},
            payload={},
            timeout_seconds=5.0,
        )

        self.assertIs(result, response)
        self.assertEqual(len(calls), 3)
        self.assertEqual(pauses, [5.0, 15.0])

    def test_persistent_outage_fails_loudly_after_three_attempts(self) -> None:
        pauses: list[float] = []
        transport = UrllibJsonHttpTransport(sleeper=pauses.append)

        def dead_post_once(url, *, headers, payload, timeout_seconds):
            raise URLError("no route to host")

        transport._post_once = dead_post_once
        with self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
            transport.post_json(
                "https://example.org/api",
                headers={},
                payload={},
                timeout_seconds=5.0,
            )
        self.assertEqual(pauses, [5.0, 15.0])


if __name__ == "__main__":
    unittest.main()
