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
    YandexContentFilterError,
    YandexFallbackJsonGenerator,
    YandexTruncationError,
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
            ("YandexGPT Lite 5", "gpt://folder-1/yandexgpt-5-lite"),
            ("YandexGPT Pro 5", "gpt://folder-1/yandexgpt-5-pro"),
            ("YandexGPT Pro 5.1", "gpt://folder-1/yandexgpt-5.1"),
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

    def test_never_returns_legacy_slash_uri(self) -> None:
        from nextwave.discovery.llm import YANDEX_MODEL_URIS

        for model, uri in YANDEX_MODEL_URIS.items():
            with self.subTest(model=model):
                self.assertNotEqual(uri, "yandexgpt/5.1")
                self.assertNotIn("yandexgpt/", uri)

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
        self.assertNotIn("jsonSchema", payload)
        self.assertEqual(
            payload,
            {
                "modelUri": "gpt://folder-1/yandexgpt-5-lite",
                "completionOptions": {
                    "stream": False,
                    "temperature": 0,
                    "maxTokens": "500",
                },
                "messages": [{"role": "user", "text": "Технологии в ИИ"}],
                "jsonObject": True,
            },
        )
        self.assertNotIn("temporary-secret", json.dumps(payload, ensure_ascii=False))

    def test_server_side_schema_keeps_prompt_and_replaces_json_object(self) -> None:
        transport = FakeJsonTransport(
            HttpResponse(200, {"Content-Type": "application/json"}, yandex_body())
        )
        schema = {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        }
        adapter = YandexCompletionJsonGenerator(
            "temporary-secret",
            folder_id="folder-1",
            selection=LlmSelection(LlmProvider.YANDEX, "YandexGPT Lite 5"),
            transport=transport,
            json_schema=schema,
            schema_name="server_response",
            server_side_json_schema=True,
        )

        adapter("Original prompt bytes: target protein annotation")

        payload = transport.calls[0][2]
        self.assertEqual(
            payload["messages"][0]["text"],
            "Original prompt bytes: target protein annotation",
        )
        self.assertEqual(payload["jsonSchema"], {"schema": schema})
        self.assertNotIn("jsonObject", payload)
        self.assertNotIn("matching this JSON Schema", payload["messages"][0]["text"])

    def test_server_side_schema_requires_schema(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires json_schema"):
            YandexCompletionJsonGenerator(
                "temporary-secret",
                folder_id="folder-1",
                selection=LlmSelection(LlmProvider.YANDEX, "YandexGPT Lite 5"),
                json_schema=None,
                server_side_json_schema=True,
            )

    def test_rejects_blank_prompt(self) -> None:
        adapter, _ = generator(
            HttpResponse(200, {"Content-Type": "application/json"}, yandex_body())
        )
        with self.assertRaisesRegex(ValueError, "prompt"):
            adapter("   ")

    def test_embeds_json_schema_in_prompt_text(self) -> None:
        transport = FakeJsonTransport(
            HttpResponse(200, {"Content-Type": "application/json"}, yandex_body())
        )
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        adapter = YandexCompletionJsonGenerator(
            "temporary-secret",
            folder_id="folder-1",
            selection=LlmSelection(LlmProvider.YANDEX, "YandexGPT Lite 5"),
            transport=transport,
            json_schema=schema,
            schema_name="candidate_mentions",
        )
        adapter("Технологии в ИИ")

        text = transport.calls[0][2]["messages"][0]["text"]
        self.assertIn('"type": "object"', text)
        self.assertIn("candidate_mentions", text)
        self.assertTrue(text.endswith("Технологии в ИИ"))

    def test_rejects_non_object_schema(self) -> None:
        with self.assertRaisesRegex(ValueError, "json_schema"):
            YandexCompletionJsonGenerator(
                "temporary-secret",
                folder_id="folder-1",
                selection=LlmSelection(LlmProvider.YANDEX, "YandexGPT Lite 5"),
                json_schema=["not", "an", "object"],
            )

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

    def test_content_filter_status_raises_distinct_error(self) -> None:
        body = yandex_body(
            alternatives=[
                {
                    "message": {"role": "assistant", "text": "refusal"},
                    "status": "ALTERNATIVE_STATUS_CONTENT_FILTER",
                }
            ]
        )
        with self.assertRaises(YandexContentFilterError) as context:
            parse_yandex_completion_text(body, "model")
        self.assertNotIsInstance(context.exception, YandexTruncationError)

    def test_rejects_invalid_json_body(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            parse_yandex_completion_text(b"not json", "model")

    def test_rejects_non_dict_alternative_with_value_error(self) -> None:
        body = json.dumps(
            {"result": {"alternatives": ["not-a-dict"]}}
        ).encode()
        with self.assertRaisesRegex(ValueError, "non-empty message text"):
            parse_yandex_completion_text(body, "model")

    def test_rejects_non_final_alternative_status(self) -> None:
        body = json.dumps(
            {
                "result": {
                    "alternatives": [
                        {
                            "message": {"role": "assistant", "text": "{}"},
                            "status": "ALTERNATIVE_STATUS_TRUNCATED_FINAL",
                        }
                    ]
                }
            }
        ).encode()
        with self.assertRaisesRegex(ValueError, "not final.*TRUNCATED_FINAL"):
            parse_yandex_completion_text(body, "model")

    def test_strips_markdown_fences_around_json(self) -> None:
        fenced = '```json\n{"normalized_query":"artificial intelligence"}\n```'
        adapter, _ = generator(
            HttpResponse(200, {"Content-Type": "application/json"}, yandex_body(fenced))
        )
        self.assertEqual(
            adapter("Технологии в ИИ"),
            '{"normalized_query":"artificial intelligence"}',
        )

    def test_error_shows_response_preview_for_live_diagnosis(self) -> None:
        adapter, _ = generator(
            HttpResponse(
                200, {"Content-Type": "application/json"}, yandex_body("soon ...")
            )
        )
        with self.assertRaisesRegex(ValueError, "preview"):
            adapter("Технологии в ИИ")

    def test_unwraps_single_key_wrapper_with_turn_marker(self) -> None:
        inner = (
            '>>\n  "normalized_query": "peripheral artificial intelligence",\n'
            '  "search_texts": ["peripheral artificial intelligence"],\n'
            '  "languages": ["en"],\n'
            '  "granularity": "direction"\n'
        )
        adapter, _ = generator(
            HttpResponse(
                200,
                {"Content-Type": "application/json"},
                yandex_body(json.dumps({inner: ""})),
            )
        )
        cleaned = adapter("Периферийный искусственный интеллект")
        self.assertEqual(json.loads(cleaned)["granularity"], "direction")

    def test_broken_wrapper_stays_rejected(self) -> None:
        adapter, _ = generator(
            HttpResponse(
                200,
                {"Content-Type": "application/json"},
                yandex_body(json.dumps({"not json at all": ""})),
            )
        )
        with self.assertRaisesRegex(ValueError, "must be a JSON object"):
            adapter("Технологии в ИИ")

    def test_drops_junk_keys_around_real_documents(self) -> None:
        documents = [
            {"document_id": "document-1", "mentions": []},
            {"document_id": "document-2", "mentions": []},
        ]
        wrapped = {
            "}\n{": {"": ""},
            "documents": documents,
            "}\n": "",
        }
        adapter, _ = generator(
            HttpResponse(
                200,
                {"Content-Type": "application/json"},
                yandex_body(json.dumps(wrapped)),
            )
        )
        cleaned = adapter("Технологии в ИИ")
        self.assertEqual(json.loads(cleaned), {"documents": documents})

    def test_clean_single_key_response_passes_through(self) -> None:
        adapter, _ = generator(
            HttpResponse(
                200,
                {"Content-Type": "application/json"},
                yandex_body('{"normalized_query":"artificial intelligence"}'),
            )
        )
        self.assertEqual(
            adapter("Технологии в ИИ"),
            '{"normalized_query":"artificial intelligence"}',
        )

    def test_raw_newline_inside_string_is_tolerated(self) -> None:
        text = (
            '{"documents": [{"document_id": "d1", "mentions": '
            '[{"text": "a\nb", "field": "title"}]}]}'
        )
        adapter, _ = generator(
            HttpResponse(
                200, {"Content-Type": "application/json"}, yandex_body(text)
            )
        )
        cleaned = adapter("Технологии в ИИ")
        self.assertEqual(
            json.loads(cleaned)["documents"][0]["mentions"][0]["text"], "a\nb"
        )


class YandexFallbackTests(unittest.TestCase):
    def test_primary_success_leaves_fallback_untouched(self) -> None:
        primary, _ = generator(
            HttpResponse(
                200, {"Content-Type": "application/json"}, yandex_body('{"ok":true}')
            )
        )
        fallback, fallback_transport = generator(
            HttpResponse(
                200, {"Content-Type": "application/json"}, yandex_body('{"ok":false}')
            ),
            model="YandexGPT Pro 5",
        )
        wrapper = YandexFallbackJsonGenerator((primary, fallback))

        self.assertEqual(wrapper("prompt"), '{"ok":true}')
        self.assertEqual(wrapper.model, "YandexGPT Lite 5")
        self.assertEqual(fallback_transport.calls, [])

    def test_parse_failure_falls_back_and_reports_model(self) -> None:
        primary, _ = generator(
            HttpResponse(
                200,
                {"Content-Type": "application/json"},
                yandex_body(json.dumps({"mangled": ""})),
            )
        )
        fallback, _ = generator(
            HttpResponse(
                200, {"Content-Type": "application/json"}, yandex_body('{"ok":true}')
            ),
            model="YandexGPT Pro 5",
        )
        wrapper = YandexFallbackJsonGenerator((primary, fallback))

        self.assertEqual(wrapper("prompt"), '{"ok":true}')
        self.assertEqual(wrapper.model, "YandexGPT Pro 5")
        self.assertEqual(wrapper.provider, LlmProvider.YANDEX)

    def test_all_mangled_raises(self) -> None:
        first, _ = generator(
            HttpResponse(
                200,
                {"Content-Type": "application/json"},
                yandex_body(json.dumps({"mangled": ""})),
            )
        )
        second, _ = generator(
            HttpResponse(
                200,
                {"Content-Type": "application/json"},
                yandex_body(json.dumps({"also mangled": ""})),
            ),
            model="YandexGPT Pro 5",
        )
        wrapper = YandexFallbackJsonGenerator((first, second))

        with self.assertRaisesRegex(ValueError, "every Yandex model"):
            wrapper("prompt")

    def test_truncation_propagates_without_fallback(self) -> None:
        primary, _ = generator(
            HttpResponse(
                200,
                {"Content-Type": "application/json"},
                yandex_body(
                    alternatives=[
                        {
                            "message": {"role": "assistant", "text": '{"ok":'},
                            "status": "ALTERNATIVE_STATUS_TRUNCATED_FINAL",
                        }
                    ]
                ),
            )
        )
        fallback, fallback_transport = generator(
            HttpResponse(
                200, {"Content-Type": "application/json"}, yandex_body('{"ok":true}')
            ),
            model="YandexGPT Pro 5",
        )
        wrapper = YandexFallbackJsonGenerator((primary, fallback))

        with self.assertRaises(YandexTruncationError):
            wrapper("prompt")
        self.assertEqual(fallback_transport.calls, [])

    def test_fallback_requires_two_models(self) -> None:
        primary, _ = generator(
            HttpResponse(
                200, {"Content-Type": "application/json"}, yandex_body('{"ok":true}')
            )
        )
        with self.assertRaisesRegex(ValueError, "at least two"):
            YandexFallbackJsonGenerator((primary,))


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
