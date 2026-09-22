"""Tests for the YandexGPT cloud adapter (no live network)."""

from __future__ import annotations

import json
import unittest
from collections.abc import Mapping
from typing import Any

from nextwave.discovery import LlmProvider, LlmSelection
from nextwave.discovery.llm import (
    YANDEX_COMPLETION_ENDPOINT,
    YandexCompletionJsonGenerator,
    build_json_generator,
    load_llm_runtime_settings,
    parse_yandex_completion_text,
)
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


def yandex_body(
    text: str = '{"normalized_query":"artificial intelligence"}',
    *,
    alternatives: list[dict[str, Any]] | None = None,
) -> bytes:
    if alternatives is None:
        alternatives = [{"message": {"role": "assistant", "text": text}}]
    return json.dumps(
        {"result": {"alternatives": alternatives, "modelVersion": "5"}}
    ).encode()


def generator(
    response: HttpResponse,
    *,
    model: str = "YandexGPT Lite 5",
) -> tuple[YandexCompletionJsonGenerator, FakeJsonTransport]:
    transport = FakeJsonTransport(response)
    return (
        YandexCompletionJsonGenerator(
            "temporary-secret",
            folder_id="folder-1",
            selection=LlmSelection(LlmProvider.YANDEX, model),
            transport=transport,
        ),
        transport,
    )


class YandexModelUriTests(unittest.TestCase):
    def test_builds_documented_model_uris(self) -> None:
        cases = (
            ("YandexGPT Lite 5", "gpt://folder-1/yandexgpt-lite/latest"),
            ("YandexGPT Pro 5", "gpt://folder-1/yandexgpt/latest"),
            # 5.1 дословно в URI: тихой подмены нет, неверный сегмент даст громкую 400.
            ("YandexGPT Pro 5.1", "gpt://folder-1/yandexgpt/5.1"),
        )
        for model, expected_uri in cases:
            with self.subTest(model=model):
                adapter, transport = generator(
                    HttpResponse(200, {"Content-Type": "application/json"}, yandex_body()),
                    model=model,
                )
                adapter("Технологии в ИИ")
                self.assertEqual(transport.calls[0][0], YANDEX_COMPLETION_ENDPOINT)
                self.assertEqual(transport.calls[0][2]["modelUri"], expected_uri)

    def test_requires_folder_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "folder"):
            YandexCompletionJsonGenerator(
                "temporary-secret",
                folder_id="  ",
                selection=LlmSelection(LlmProvider.YANDEX, "YandexGPT Lite 5"),
            )

    def test_requires_api_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "API key"):
            YandexCompletionJsonGenerator(
                "  ",
                folder_id="folder-1",
                selection=LlmSelection(LlmProvider.YANDEX, "YandexGPT Lite 5"),
            )


class YandexRequestTests(unittest.TestCase):
    def test_sends_api_key_header_and_json_object_mode(self) -> None:
        adapter, transport = generator(
            HttpResponse(200, {"Content-Type": "application/json"}, yandex_body())
        )
        adapter("Технологии в ИИ")

        _, headers, payload, _ = transport.calls[0]
        self.assertEqual(headers["Authorization"], "Api-Key temporary-secret")
        self.assertTrue(payload["jsonObject"])
        self.assertEqual(
            payload["messages"], [{"role": "user", "text": "Технологии в ИИ"}]
        )
        self.assertFalse(payload["completionOptions"]["stream"])
        self.assertEqual(payload["completionOptions"]["temperature"], 0)
        self.assertNotIn("temporary-secret", json.dumps(payload, ensure_ascii=False))

    def test_rejects_blank_prompt(self) -> None:
        adapter, _ = generator(
            HttpResponse(200, {"Content-Type": "application/json"}, yandex_body())
        )
        with self.assertRaisesRegex(ValueError, "prompt"):
            adapter("   ")

    def test_raises_on_http_error_without_body_leak(self) -> None:
        adapter, _ = generator(
            HttpResponse(401, {"Content-Type": "application/json"}, b'{"error":"x"}')
        )
        with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
            adapter("Технологии в ИИ")


class YandexResponseTests(unittest.TestCase):
    def test_returns_single_text(self) -> None:
        adapter, _ = generator(
            HttpResponse(200, {"Content-Type": "application/json"}, yandex_body())
        )
        self.assertEqual(
            adapter("Технологии в ИИ"),
            '{"normalized_query":"artificial intelligence"}',
        )

    def test_rejects_non_dict_text(self) -> None:
        adapter, _ = generator(
            HttpResponse(200, {"Content-Type": "application/json"}, yandex_body("[1, 2]"))
        )
        with self.assertRaisesRegex(ValueError, "JSON object"):
            adapter("Технологии в ИИ")

    def test_rejects_zero_or_two_alternatives(self) -> None:
        for alternatives in ([], [{}, {}]):
            adapter, _ = generator(
                HttpResponse(
                    200,
                    {"Content-Type": "application/json"},
                    yandex_body(alternatives=alternatives),
                )
            )
            with self.subTest(alternatives=len(alternatives)):
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    adapter("Технологии в ИИ")

    def test_rejects_invalid_json_body(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            parse_yandex_completion_text(b"not json", "model")


class YandexSettingsTests(unittest.TestCase):
    def test_generator_wires_through_build_json_generator(self) -> None:
        settings = load_llm_runtime_settings(
            {
                "NEXTWAVE_LLM_PROVIDER": "yandex",
                "NEXTWAVE_LLM_MODEL": "YandexGPT Lite 5",
                "NEXTWAVE_LLM_API_KEY": "temporary-secret",
                "NEXTWAVE_YANDEX_FOLDER_ID": "folder-1",
            }
        )
        adapter = build_json_generator(settings)
        self.assertIsInstance(adapter, YandexCompletionJsonGenerator)

    def test_folder_is_required_for_yandex_only(self) -> None:
        with self.assertRaisesRegex(ValueError, "NEXTWAVE_YANDEX_FOLDER_ID"):
            load_llm_runtime_settings(
                {
                    "NEXTWAVE_LLM_PROVIDER": "yandex",
                    "NEXTWAVE_LLM_MODEL": "YandexGPT Lite 5",
                    "NEXTWAVE_LLM_API_KEY": "temporary-secret",
                }
            )
        settings = load_llm_runtime_settings(
            {
                "NEXTWAVE_LLM_PROVIDER": "qwen",
                "NEXTWAVE_LLM_MODEL": "Qwen3.6 35B-A3B",
                "NEXTWAVE_LLM_API_KEY": "temporary-secret",
            }
        )
        self.assertEqual(settings.selection.provider, LlmProvider.QWEN)


if __name__ == "__main__":
    unittest.main()
