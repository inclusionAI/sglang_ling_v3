"""Unit tests for the ``--quantization blockwise_fp8`` CLI alias."""

import json
import unittest

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.serving_hook import handle_quantization_alias
from sglang.srt.server_args import ServerArgs, prepare_server_args
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestBlockwiseFp8Alias(CustomTestCase):
    def test_cli_alias_builds_online_fp8_override(self):
        args = prepare_server_args(
            [
                "--model-path",
                "dummy",
                "--quantization",
                "blockwise_fp8",
                "--tp-size",
                "8",
                "--ep",
                "1",
                "--json-model-override-args",
                '{"architectures": ["BailingMoeV3ForCausalLM"]}',
            ]
        )

        # The dummy-model resolution path intentionally exits before aliases;
        # drive this model-independent handler directly.
        handle_quantization_alias(args)

        self.assertEqual(resolution_result(args, "quantization"), "fp8")
        overrides = json.loads(resolution_result(args, "json_model_override_args"))
        self.assertEqual(overrides["architectures"], ["BailingMoeV3ForCausalLM"])
        self.assertEqual(
            overrides["quantization_config"],
            {
                "is_checkpoint_fp8_serialized": False,
                "is_online_block_quant": True,
                "tp_size": 8,
                "ep_size": 1,
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "scale_fmt": "float32",
                "weight_block_size": [128, 128],
                "modules_to_not_convert": [
                    "q_a_proj",
                    "q_b_proj",
                    "fused_qkv_a_proj_with_mqa",
                    "kv_a_proj_with_mqa",
                    "kv_b_proj",
                    "lm_head",
                    "word_embeddings",
                    "b_proj",
                    "g_proj",
                    "gate",
                ],
            },
        )

    def test_alias_requires_object_model_overrides(self):
        args = ServerArgs(
            model_path="dummy",
            quantization="blockwise_fp8",
            json_model_override_args="[]",
        )

        with self.assertRaisesRegex(ValueError, "decode to an object"):
            handle_quantization_alias(args)

    def test_other_quantization_is_unchanged(self):
        args = ServerArgs(
            model_path="dummy",
            quantization="fp8",
            json_model_override_args='{"sentinel": true}',
        )

        handle_quantization_alias(args)

        self.assertEqual(resolution_result(args, "quantization"), "fp8")
        self.assertEqual(
            json.loads(resolution_result(args, "json_model_override_args")),
            {"sentinel": True},
        )

    def test_draft_only_alias_is_rejected(self):
        args = ServerArgs(
            model_path="dummy",
            speculative_draft_model_quantization="blockwise_fp8",
        )

        with self.assertRaisesRegex(ValueError, "draft-only"):
            handle_quantization_alias(args)

    def test_target_alias_rejects_explicit_draft_quantization(self):
        args = ServerArgs(
            model_path="dummy",
            quantization="blockwise_fp8",
            speculative_draft_model_quantization="fp8",
        )

        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            handle_quantization_alias(args)


if __name__ == "__main__":
    unittest.main()
