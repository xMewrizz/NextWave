"""Separate labeling Evidence LLM configuration tests."""

from __future__ import annotations

import unittest

from nextwave.discovery.llm import (
    LlmProvider,
    load_evidence_llm_settings,
    load_gate_llm_settings,
    load_llm_runtime_settings,
)


def base_env() -> dict[str, str]:
    return {
        "NEXTWAVE_LLM_PROVIDER": "yandex",
        "NEXTWAVE_LLM_MODEL": "YandexGPT Lite 5",
        "NEXTWAVE_LLM_API_KEY": "fake-secret-key",
        "NEXTWAVE_YANDEX_FOLDER_ID": "fake-secret-folder",
    }


class EvidenceLlmConfigTests(unittest.TestCase):
    def test_absent_pair_inherits_main_model(self) -> None:
        settings = load_evidence_llm_settings(base_env())

        self.assertEqual(settings.selection.provider, LlmProvider.YANDEX)
        self.assertEqual(settings.selection.model, "YandexGPT Lite 5")

    def test_explicit_pair_uses_pro51_without_changing_main_or_gate(self) -> None:
        env = base_env() | {
            "NEXTWAVE_GATE_LLM_PROVIDER": "yandex",
            "NEXTWAVE_GATE_LLM_MODEL": "YandexGPT Pro 5",
            "NEXTWAVE_EVIDENCE_LLM_PROVIDER": "yandex",
            "NEXTWAVE_EVIDENCE_LLM_MODEL": "YandexGPT Pro 5.1",
        }

        self.assertEqual(
            load_evidence_llm_settings(env).selection.model, "YandexGPT Pro 5.1"
        )
        self.assertEqual(load_llm_runtime_settings(env).selection.model, "YandexGPT Lite 5")
        self.assertEqual(load_gate_llm_settings(env).selection.model, "YandexGPT Pro 5")

    def test_provider_without_model_is_rejected(self) -> None:
        env = base_env() | {"NEXTWAVE_EVIDENCE_LLM_PROVIDER": "yandex"}
        with self.assertRaisesRegex(ValueError, "NEXTWAVE_EVIDENCE_LLM_MODEL"):
            load_evidence_llm_settings(env)

    def test_model_without_provider_is_rejected(self) -> None:
        env = base_env() | {"NEXTWAVE_EVIDENCE_LLM_MODEL": "YandexGPT Pro 5.1"}
        with self.assertRaisesRegex(ValueError, "NEXTWAVE_EVIDENCE_LLM_PROVIDER"):
            load_evidence_llm_settings(env)

    def test_unapproved_evidence_model_is_rejected(self) -> None:
        env = base_env() | {
            "NEXTWAVE_EVIDENCE_LLM_PROVIDER": "yandex",
            "NEXTWAVE_EVIDENCE_LLM_MODEL": "YandexGPT Ultra",
        }
        with self.assertRaisesRegex(ValueError, "is not approved"):
            load_evidence_llm_settings(env)

    def test_credentials_do_not_enter_repr_or_equality(self) -> None:
        first = load_evidence_llm_settings(
            base_env()
            | {
                "NEXTWAVE_EVIDENCE_LLM_PROVIDER": "yandex",
                "NEXTWAVE_EVIDENCE_LLM_MODEL": "YandexGPT Pro 5.1",
            }
        )
        second_env = base_env() | {
            "NEXTWAVE_LLM_API_KEY": "different-key",
            "NEXTWAVE_YANDEX_FOLDER_ID": "different-folder",
            "NEXTWAVE_EVIDENCE_LLM_PROVIDER": "yandex",
            "NEXTWAVE_EVIDENCE_LLM_MODEL": "YandexGPT Pro 5.1",
        }
        second = load_evidence_llm_settings(second_env)

        self.assertEqual(first, second)
        self.assertNotIn("fake-secret", repr(first))


if __name__ == "__main__":
    unittest.main()
