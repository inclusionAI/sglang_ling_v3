"""Regression tests for Ming audio configs."""

import copy
import json
import tempfile
import unittest

from sglang.srt.configs.bailing_audio_v3 import BailingMoeV3AudioConfig
from sglang.srt.configs.bailing_hybrid import BailingHybridConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def audio_config(**text_overrides):
    text = {
        "max_position_embeddings": 131072,
        "rope_theta": 6000000,
        "partial_rotary_factor": 0.5,
        "qk_rope_head_dim": 64,
    }
    text.update(text_overrides)
    return BailingMoeV3AudioConfig(text_config=text, audio_config={})


class TestBailingAudioRopeConfig(CustomTestCase):
    def test_audio_defaults_to_one_dimensional_rope(self):
        config = audio_config()
        self.assertNotIn("mrope_section", config.llm_config.rope_parameters)
        self.assertNotIn("video_rope", config.llm_config.rope_parameters)
        self.assertIs(config.llm_config._sglang_audio_1d_rope, True)
        self.assertEqual(config.llm_config.rope_parameters["rope_type"], "default")
        self.assertEqual(config.llm_config.rope_parameters["rope_theta"], 6000000)

    def test_legacy_audio_sections_are_removed_without_mutating_input(self):
        rope = {
            "rope_type": "default",
            "mrope_section": [8, 12, 12],
            "video_rope": True,
        }
        original = copy.deepcopy(rope)
        config = audio_config(rope_scaling=rope)
        self.assertNotIn("mrope_section", config.llm_config.rope_parameters)
        self.assertNotIn("video_rope", config.llm_config.rope_parameters)
        self.assertEqual(rope, original)

    def test_top_level_yarn_and_length_update_actual_text_config(self):
        config = audio_config()
        text = config.llm_config
        override = {
            "max_position_embeddings": 262144,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 131072,
            },
        }
        original = copy.deepcopy(override)
        config.update(override)
        self.assertIs(config.get_text_config(), text)
        self.assertIsInstance(text, BailingHybridConfig)
        self.assertEqual(text.max_position_embeddings, 262144)
        self.assertEqual(text.rope_parameters["rope_type"], "yarn")
        self.assertEqual(text.rope_parameters["factor"], 2.0)
        self.assertEqual(text.rope_parameters["rope_theta"], 6000000)
        self.assertEqual(text.qk_rope_head_dim, 64)
        self.assertEqual(text.partial_rotary_factor, 0.5)
        self.assertNotIn("mrope_section", text.rope_parameters)
        self.assertEqual(override, original)

    def test_partial_nested_update_preserves_typed_config(self):
        config = audio_config()
        text = config.llm_config
        config.update({"llm_config": {"max_position_embeddings": 262144}})
        self.assertIs(config.llm_config, text)
        self.assertIsInstance(text, BailingHybridConfig)
        self.assertEqual(text.max_position_embeddings, 262144)

    def test_nested_audio_override_preserves_typed_config(self):
        config = audio_config()
        audio = config.audio_config
        config.update({"audio_config": {"chunk_frames": 1500}})
        self.assertIs(config.audio_config, audio)
        self.assertEqual(audio.chunk_frames, 1500)
        self.assertEqual(audio.whisper_encoder_config["n_ctx"], 3000)

    def test_nested_rope_override_wins_across_aliases(self):
        config = audio_config()
        config.update(
            {
                "rope_theta": 10000,
                "rope_parameters": {"rope_type": "yarn", "factor": 4.0},
                "llm_config": {
                    "rope_scaling": {
                        "type": "yarn",
                        "factor": 2.0,
                        "rope_theta": 6000000,
                    }
                },
            }
        )
        self.assertEqual(config.llm_config.rope_parameters["factor"], 2.0)
        self.assertEqual(config.llm_config.rope_parameters["type"], "yarn")
        self.assertEqual(config.llm_config.rope_parameters["rope_type"], "yarn")
        self.assertEqual(config.llm_config.rope_parameters["rope_theta"], 6000000)
        self.assertEqual(config.llm_config.rope_theta, 6000000)

    def test_null_scaling_resets_to_unscaled_rope(self):
        config = audio_config()
        config.update({"rope_scaling": {"rope_type": "yarn", "factor": 2.0}})
        config.update({"rope_scaling": None})
        self.assertEqual(config.llm_config.rope_parameters["rope_type"], "default")
        self.assertNotIn("factor", config.llm_config.rope_parameters)

    def test_invalid_mapping_fails_without_partial_mutation(self):
        for key, value in [
            ("rope_scaling", "yarn"),
            ("rope_parameters", []),
            ("llm_config", None),
            ("text_config", None),
        ]:
            with self.subTest(key=key, value=value):
                config = audio_config()
                original = copy.deepcopy(config.llm_config.to_dict())
                with self.assertRaises(TypeError):
                    config.update({"max_position_embeddings": 262144, key: value})
                self.assertEqual(config.llm_config.to_dict(), original)

    def test_scalar_theta_updates_both_representations(self):
        config = audio_config()
        config.update({"rope_theta": 123456})
        self.assertEqual(config.llm_config.rope_theta, 123456)
        self.assertEqual(config.llm_config.rope_parameters["rope_theta"], 123456)

    def test_explicit_scaling_survives_initialization(self):
        config = audio_config(rope_scaling={"rope_type": "yarn", "factor": 2.0})
        self.assertEqual(config.llm_config.rope_parameters["rope_type"], "yarn")
        self.assertEqual(config.llm_config.rope_parameters["factor"], 2.0)
        self.assertNotIn("mrope_section", config.llm_config.rope_parameters)

    def test_top_level_override_cannot_reenable_audio_mrope(self):
        config = audio_config()
        config.update(
            {
                "rope_parameters": {
                    "rope_type": "default",
                    "mrope_section": [8, 12, 12],
                    "video_rope": True,
                }
            }
        )
        self.assertNotIn("mrope_section", config.llm_config.rope_parameters)
        self.assertNotIn("video_rope", config.llm_config.rope_parameters)

    def test_language_override_alias_precedence(self):
        config = audio_config()
        config.update(
            {
                "max_position_embeddings": 65536,
                "rope_scaling": {"rope_type": "yarn", "factor": 4.0},
                "text_config": {
                    "max_position_embeddings": 262144,
                    "rope_parameters": {"rope_type": "yarn", "factor": 2.0},
                },
                "llm_config": {"max_position_embeddings": 393216},
            }
        )
        self.assertEqual(config.llm_config.max_position_embeddings, 393216)
        self.assertEqual(config.llm_config.rope_parameters["factor"], 2.0)
        self.assertNotIn("text_config", vars(config))

    def test_real_hf_loader_routes_audio_overrides_to_typed_text_config(self):
        """CLI aliases must change the effective LLM, not just the wrapper."""
        from sglang.srt.utils.hf_transformers import get_config, get_hf_text_config
        from sglang.srt.utils.hf_transformers.common import get_context_length

        temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_dir.cleanup)
        audio_config().save_pretrained(temporary_dir.name)
        language_override = {
            "max_position_embeddings": 262144,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 131072,
                "rope_theta": 6000000,
            },
        }
        for key in (None, "text_config", "llm_config"):
            with self.subTest(key=key):
                override = (
                    copy.deepcopy(language_override)
                    if key is None
                    else {key: copy.deepcopy(language_override)}
                )
                override["audio_config"] = {"chunk_frames": 1500}
                original = copy.deepcopy(override)
                config = get_config(
                    temporary_dir.name,
                    trust_remote_code=False,
                    model_override_args=override,
                )
                text = get_hf_text_config(config)
                self.assertIs(text, config.llm_config)
                self.assertIs(config.get_text_config(), text)
                self.assertIsInstance(text, BailingHybridConfig)
                self.assertEqual(text.max_position_embeddings, 262144)
                self.assertEqual(get_context_length(text), 262144)
                self.assertEqual(text.rope_parameters["rope_type"], "yarn")
                self.assertEqual(text.rope_parameters["factor"], 2.0)
                self.assertEqual(
                    text.rope_parameters["original_max_position_embeddings"], 131072
                )
                self.assertEqual(text.rope_parameters["rope_theta"], 6000000)
                self.assertNotIn("mrope_section", text.rope_parameters)
                self.assertEqual(config.audio_config.chunk_frames, 1500)
                self.assertEqual(
                    config.audio_config.whisper_encoder_config["n_ctx"], 3000
                )
                self.assertEqual(override, original)

    def test_server_args_use_effective_language_context_without_bypass(self):
        """262K must be accepted from the model config without a context bypass."""
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.environ import envs
        from sglang.srt.server_args import prepare_server_args

        temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_dir.cleanup)
        audio_config().save_pretrained(temporary_dir.name)
        language_override = {
            "max_position_embeddings": 262144,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 131072,
                "rope_theta": 6000000,
            },
        }
        for key in (None, "text_config", "llm_config"):
            with self.subTest(key=key):
                override = (
                    language_override if key is None else {key: language_override}
                )
                args = prepare_server_args(
                    [
                        "--model-path",
                        temporary_dir.name,
                        "--device",
                        "cpu",
                        "--enable-multimodal",
                        "--context-length",
                        "262144",
                        "--json-model-override-args",
                        json.dumps(override),
                    ]
                )
                with envs.SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN.override(False):
                    config = ModelConfig.from_server_args(args)
                self.assertEqual(config.context_len, 262144)
                self.assertEqual(config.hf_text_config.max_position_embeddings, 262144)
                self.assertEqual(
                    config.hf_text_config.rope_parameters["rope_type"], "yarn"
                )
                self.assertEqual(config.hf_text_config.rope_parameters["factor"], 2.0)
                self.assertIs(config.hf_text_config, config.hf_config.llm_config)


if __name__ == "__main__":
    unittest.main()
