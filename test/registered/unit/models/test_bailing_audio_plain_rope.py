"""Regression tests for Ming audio models."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.models import bailing_moe_v3
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestBailingAudioPlainRope(CustomTestCase):
    def _patch(self, target, attribute, replacement):
        patcher = patch.object(target, attribute, replacement)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def test_plain_rope_is_not_rewritten_by_deepseek_parent(self):
        for rope in [
            {"rope_type": "default", "rope_theta": 6000000},
            {"type": "default", "partial_rotary_factor": 0.5},
        ]:
            with self.subTest(rope=rope):
                captured = {}

                def parent_init(self, *args, **kwargs):
                    captured["rope_scaling"] = args[9]

                self._patch(
                    bailing_moe_v3.DeepseekV2AttentionMLA, "__init__", parent_init
                )
                self._patch(
                    bailing_moe_v3,
                    "get_parallel",
                    lambda: SimpleNamespace(attn_tp_rank=0, attn_tp_size=1),
                )
                bailing_moe_v3.DsV3MLA(
                    config=SimpleNamespace(_sglang_audio_1d_rope=True),
                    hidden_size=2560,
                    num_heads=32,
                    qk_nope_head_dim=128,
                    qk_rope_head_dim=64,
                    v_head_dim=128,
                    q_lora_rank=None,
                    kv_lora_rank=512,
                    rope_theta=6000000,
                    rope_scaling=rope,
                    max_position_embeddings=131072,
                )
                self.assertIs(captured["rope_scaling"], None)

    def test_explicit_yarn_is_forwarded_to_existing_parent_path(self):
        captured = {}
        rope = {"rope_type": "yarn", "factor": 2.0}

        def parent_init(self, *args, **kwargs):
            captured["rope_scaling"] = args[9]

        self._patch(bailing_moe_v3.DeepseekV2AttentionMLA, "__init__", parent_init)
        self._patch(
            bailing_moe_v3,
            "get_parallel",
            lambda: SimpleNamespace(attn_tp_rank=0, attn_tp_size=1),
        )
        bailing_moe_v3.DsV3MLA(
            config=SimpleNamespace(),
            hidden_size=2560,
            num_heads=32,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            q_lora_rank=None,
            kv_lora_rank=512,
            rope_theta=6000000,
            rope_scaling=rope,
            max_position_embeddings=262144,
        )
        self.assertIs(captured["rope_scaling"], rope)

    def test_existing_sectioned_paths_keep_their_factory(self):
        for video_rope in [False, True]:
            with self.subTest(video_rope=video_rope):
                calls = []
                rope = {"rope_type": "default", "mrope_section": [8, 12, 12]}
                if video_rope:
                    rope["video_rope"] = True

                def parent_init(self, *args, **kwargs):
                    args[9]["rope_type"] = "deepseek_yarn"

                def visual_factory(*args, **kwargs):
                    calls.append("visual")
                    return None

                def audio_factory(*args, **kwargs):
                    calls.append("sectioned")
                    return None

                self._patch(
                    bailing_moe_v3.DeepseekV2AttentionMLA, "__init__", parent_init
                )
                self._patch(bailing_moe_v3, "get_rope", visual_factory)
                self._patch(
                    bailing_moe_v3.BailingMlaRotaryEmbedding,
                    "get_cached",
                    audio_factory,
                )
                self._patch(
                    bailing_moe_v3,
                    "get_parallel",
                    lambda: SimpleNamespace(attn_tp_rank=0, attn_tp_size=1),
                )
                bailing_moe_v3.DsV3MLA(
                    config=SimpleNamespace(rope_interleave=True),
                    hidden_size=2560,
                    num_heads=32,
                    qk_nope_head_dim=128,
                    qk_rope_head_dim=64,
                    v_head_dim=128,
                    q_lora_rank=None,
                    kv_lora_rank=512,
                    rope_theta=6000000,
                    rope_scaling=rope,
                    max_position_embeddings=131072,
                )
                self.assertEqual(calls, ["visual" if video_rope else "sectioned"])
                self.assertEqual(rope["rope_type"], "default")

    def test_non_audio_language_rope_keeps_original_parent_arguments(self):
        for rope in [
            None,
            {"rope_type": "default", "rope_theta": 6000000},
            {"type": "default"},
            {"rope_type": "yarn", "factor": 2.0},
        ]:
            with self.subTest(rope=rope):
                captured = {}

                def parent_init(self, *args, **kwargs):
                    captured["rope_scaling"] = args[9]

                self._patch(
                    bailing_moe_v3.DeepseekV2AttentionMLA, "__init__", parent_init
                )
                self._patch(
                    bailing_moe_v3,
                    "get_parallel",
                    lambda: SimpleNamespace(attn_tp_rank=0, attn_tp_size=1),
                )
                bailing_moe_v3.DsV3MLA(
                    config=SimpleNamespace(),
                    hidden_size=2560,
                    num_heads=32,
                    qk_nope_head_dim=128,
                    qk_rope_head_dim=64,
                    v_head_dim=128,
                    q_lora_rank=None,
                    kv_lora_rank=512,
                    rope_theta=6000000,
                    rope_scaling=rope,
                    max_position_embeddings=262144,
                )
                self.assertIs(captured["rope_scaling"], rope)


if __name__ == "__main__":
    unittest.main()
