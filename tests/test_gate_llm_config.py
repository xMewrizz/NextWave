"""Separate Gate LLM configuration: Lite stays default, Gate uses explicit Pro."""

from __future__ import annotations

import json
import unittest

from nextwave.discovery import (
    ScopeGranularity,
    build_analysis_scope,
    build_candidate_gate_from_environment,
    build_evidence_extractor_from_environment,
    build_query_resolver_from_environment,
    load_gate_llm_settings,
    load_llm_runtime_settings,
)
from nextwave.discovery.candidate_gate import (
    PRODUCT_GATE_ID,
    QUALIFICATION_GATE_ID,
    QUALIFICATION_GATE_VERSION,
)
from nextwave.discovery.candidates import CandidateProposalBatch


def base_env() -> dict[str, str]:
    return {
        "NEXTWAVE_LLM_PROVIDER": "yandex",
        "NEXTWAVE_LLM_MODEL": "YandexGPT Lite 5",
        "NEXTWAVE_LLM_API_KEY": "temporary-secret",
        "NEXTWAVE_YANDEX_FOLDER_ID": "folder-1",
    }


def gate_scope():
    return build_analysis_scope(
        raw_query="Технологии в ИИ",
        normalized_query="artificial intelligence",
        search_texts=("Технологии в ИИ",),
        languages=("en", "ru"),
        granularity=ScopeGranularity.DIRECTION,
        subfield_ids=("1702",),
    )


class GateLlmConfigTests(unittest.TestCase):
    def test_missing_both_gate_vars_inherits_main_model(self) -> None:
        settings = load_gate_llm_settings(base_env())
        self.assertEqual(settings.selection.model, "YandexGPT Lite 5")
        self.assertEqual(settings.selection.provider.value, "yandex")

    def test_both_gate_vars_select_pro(self) -> None:
        env = base_env() | {
            "NEXTWAVE_GATE_LLM_PROVIDER": "yandex",
            "NEXTWAVE_GATE_LLM_MODEL": "YandexGPT Pro 5",
        }
        settings = load_gate_llm_settings(env)
        self.assertEqual(settings.selection.model, "YandexGPT Pro 5")
        # Main pair is untouched.
        self.assertEqual(load_llm_runtime_settings(env).selection.model, "YandexGPT Lite 5")

    def test_only_gate_provider_is_an_error(self) -> None:
        env = base_env() | {"NEXTWAVE_GATE_LLM_PROVIDER": "yandex"}
        with self.assertRaisesRegex(ValueError, "NEXTWAVE_GATE_LLM_MODEL"):
            load_gate_llm_settings(env)

    def test_only_gate_model_is_an_error(self) -> None:
        env = base_env() | {"NEXTWAVE_GATE_LLM_MODEL": "YandexGPT Pro 5"}
        with self.assertRaisesRegex(ValueError, "NEXTWAVE_GATE_LLM_PROVIDER"):
            load_gate_llm_settings(env)

    def test_forbidden_gate_model_is_rejected(self) -> None:
        env = base_env() | {
            "NEXTWAVE_GATE_LLM_PROVIDER": "yandex",
            "NEXTWAVE_GATE_LLM_MODEL": "YandexGPT Ultra",
        }
        with self.assertRaisesRegex(ValueError, "not approved"):
            load_gate_llm_settings(env)

    def test_gate_id_contains_actual_pro_model(self) -> None:
        env = base_env() | {
            "NEXTWAVE_GATE_LLM_PROVIDER": "qwen",
            "NEXTWAVE_GATE_LLM_MODEL": "Qwen3 235B",
            "NEXTWAVE_QWEN_BASE_URL": (
                "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            ),
        }
        gate = build_candidate_gate_from_environment(env)
        scope = gate_scope()
        result = gate.evaluate(
            scope,
            CandidateProposalBatch(analysis_scope_id=scope.scope_id, proposals=(), exclusions=()),
            (),
        )
        self.assertEqual(result.gate_id, PRODUCT_GATE_ID)
        self.assertEqual(result.gate_id, "qwen-qwen3-235b-candidate-gate-v6")

    def test_resolver_and_evidence_stay_on_main_lite(self) -> None:
        env = base_env() | {
            "NEXTWAVE_GATE_LLM_PROVIDER": "yandex",
            "NEXTWAVE_GATE_LLM_MODEL": "YandexGPT Pro 5",
        }
        # Builders must not raise and must not pick up the Gate model.
        build_query_resolver_from_environment(env)
        evidence = build_evidence_extractor_from_environment(env)
        extractor = evidence.extractor_id.casefold()
        self.assertIn("lite 5", extractor)
        self.assertNotIn("pro 5", extractor)
        self.assertEqual(load_llm_runtime_settings(env).selection.model, "YandexGPT Lite 5")

    def test_frozen_qualification_builder_keeps_gate_v4(self) -> None:
        env = base_env() | {
            "NEXTWAVE_GATE_LLM_PROVIDER": "yandex",
            "NEXTWAVE_GATE_LLM_MODEL": "YandexGPT Pro 5",
        }
        gate = build_candidate_gate_from_environment(
            env,
            version=QUALIFICATION_GATE_VERSION,
        )
        self.assertEqual(gate.gate_id, QUALIFICATION_GATE_ID)

    def test_secrets_absent_from_repr_and_serialized_result(self) -> None:
        env = base_env() | {
            "NEXTWAVE_GATE_LLM_PROVIDER": "yandex",
            "NEXTWAVE_GATE_LLM_MODEL": "YandexGPT Pro 5",
        }
        settings = load_gate_llm_settings(env)
        self.assertNotIn("temporary-secret", repr(settings))
        self.assertNotIn("folder-1", repr(settings))
        gate = build_candidate_gate_from_environment(env)
        scope = gate_scope()
        result = gate.evaluate(
            scope,
            CandidateProposalBatch(analysis_scope_id=scope.scope_id, proposals=(), exclusions=()),
            (),
        )
        payload = json.dumps(result.to_dict(), ensure_ascii=False)
        self.assertNotIn("temporary-secret", payload)
        self.assertNotIn("folder-1", payload)
        self.assertNotIn("temporary-secret", str(result))
        self.assertNotIn("folder-1", str(result))


if __name__ == "__main__":
    unittest.main()
