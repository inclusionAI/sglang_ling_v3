"""CPU unit tests for Bailing serialized blockwise-FP8 TP requantization."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import sglang.srt.layers.quantization.fp8 as fp8_module  # noqa: E402
from sglang.srt.layers.moe import utils as moe_utils  # noqa: E402
from sglang.srt.layers.moe.utils import (  # noqa: E402
    MoeA2ABackend,
    MoeRunnerBackend,
)
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod  # noqa: E402
from sglang.srt.models.bailing_moe_v3 import BailingMoE  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestBailingMoeV3TpRequant(CustomTestCase):
    @staticmethod
    def _empty_moe():
        moe = BailingMoE.__new__(BailingMoE)
        torch.nn.Module.__init__(moe)
        return moe

    def _supports_tail(
        self,
        *,
        opt_in=True,
        runner=MoeRunnerBackend.TRITON,
        moe_ep_size=1,
        cpu_offload_gb=0,
    ):
        method = object.__new__(Fp8MoEMethod)
        method.quant_config = SimpleNamespace(
            is_checkpoint_fp8_serialized=True,
            weight_block_size=[128, 128],
        )
        method.is_online_block_quant = False
        method.use_mxfp8 = False

        layer = SimpleNamespace(
            allow_fp8_blockwise_tail=opt_in,
            moe_ep_size=moe_ep_size,
            moe_tp_size=8,
            moe_tp_rank=3,
        )
        parallel = SimpleNamespace(
            tp_size=8,
            tp_rank=3,
            moe_dp_size=1,
        )
        runtime = SimpleNamespace(
            moe=SimpleNamespace(
                enable_eplb=False,
                ep_num_redundant_experts=0,
                init_expert_location="trivial",
            ),
            offload=SimpleNamespace(
                cpu_offload_gb=cpu_offload_gb,
                offload_group_size=-1,
            ),
        )
        a2a = MoeA2ABackend.NONE
        with (
            patch.object(fp8_module, "_is_cuda", True),
            patch.object(fp8_module, "_is_hip", False),
            patch.object(fp8_module, "get_parallel", return_value=parallel),
            patch.object(fp8_module, "get_exec", return_value=runtime),
            patch.object(fp8_module, "get_moe_runner_backend", return_value=runner),
            patch.object(fp8_module, "get_moe_a2a_backend", return_value=a2a),
            patch.object(moe_utils, "get_moe_a2a_backend", return_value=a2a),
        ):
            return method._supports_blockwise_fp8_tail(layer)

    def test_fp8_moe_tail_shape_gate_matches_bailing_runtime_boundary(self):
        for runner in (MoeRunnerBackend.TRITON, MoeRunnerBackend.AUTO):
            with self.subTest(supported_runner=runner.value):
                self.assertTrue(self._supports_tail(runner=runner))

        cases = (
            ("not opted in", {"opt_in": False}),
            ("cutlass", {"runner": MoeRunnerBackend.CUTLASS}),
            ("expert parallel", {"moe_ep_size": 2}),
            ("weight offload", {"cpu_offload_gb": 1}),
        )
        for name, kwargs in cases:
            with self.subTest(rejected=name):
                self.assertFalse(self._supports_tail(**kwargs))

    def test_dequantize_uses_checkpoint_global_block_offsets(self):
        quantized = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        checkpoint_scale = torch.tensor(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32
        )

        output = BailingMoE._dequantize_serialized_fp8_tp_shard(
            quantized,
            checkpoint_scale,
            (2, 2),
            row_offset=1,
            column_offset=1,
        )

        # Global row indices are [1, 2] -> scale rows [0, 1], while global
        # columns [1, 2, 3] -> scale columns [0, 1, 1].
        expected = torch.tensor([[1, 4, 6], [16, 25, 30]], dtype=torch.bfloat16)
        torch.testing.assert_close(output, expected)

        with self.assertRaisesRegex(ValueError, "exceeds its checkpoint scale"):
            BailingMoE._dequantize_serialized_fp8_tp_shard(
                quantized,
                checkpoint_scale[:1, :1],
                (2, 2),
                row_offset=1,
                column_offset=1,
            )

    def test_cache_keeps_global_scales_until_all_shards_are_loaded(self):
        moe = self._empty_moe()
        moe.experts = SimpleNamespace(
            num_local_experts=2,
            w13_weight=torch.empty(1),
            _map_global_expert_id_to_local_expert_id=lambda expert_id: (
                1 if expert_id == 7 else -1
            ),
        )
        loaded_scale = torch.arange(6, dtype=torch.float32).reshape(3, 2)

        with patch.object(
            BailingMoE, "_requires_serialized_fp8_tp_requant", return_value=True
        ):
            self.assertTrue(
                moe.cache_serialized_fp8_scale_for_tp_requant(loaded_scale, "w1", 7)
            )
            self.assertTrue(
                moe.cache_serialized_fp8_scale_for_tp_requant(loaded_scale, "w1", 99)
            )
            self.assertFalse(
                moe.cache_serialized_fp8_scale_for_tp_requant(
                    loaded_scale, "not-an-expert-shard", 7
                )
            )

            self.assertEqual(
                tuple(moe._serialized_fp8_tp_scale_cache["w1"].shape), (2, 3, 2)
            )
            torch.testing.assert_close(
                moe._serialized_fp8_tp_scale_cache["w1"][1], loaded_scale
            )
            self.assertEqual(moe._serialized_fp8_tp_scale_loaded["w1"], {1})

            with self.assertRaisesRegex(ValueError, "Inconsistent.*scale shapes"):
                moe.cache_serialized_fp8_scale_for_tp_requant(torch.ones(4, 2), "w1", 7)
            with self.assertRaisesRegex(ValueError, "scales must be 2D"):
                moe.cache_serialized_fp8_scale_for_tp_requant(
                    torch.ones(1, 1, 1), "w2", 7
                )

    def test_requantize_reblocks_tp_local_weights_on_cpu(self):
        moe = self._empty_moe()
        moe.layer_id = 0
        moe.intermediate_size = 6
        moe.hidden_size = 4
        moe.moe_tp_size = 2
        moe.moe_tp_rank = 1
        moe.quant_config = SimpleNamespace(weight_block_size=[2, 2])

        fp8_dtype = torch.float8_e4m3fn
        w13_weight = torch.tensor(
            [
                [1, 2, 3, 4],
                [2, 3, 4, 5],
                [3, 4, 5, 6],
                [2, 1, 4, 3],
                [3, 2, 5, 4],
                [4, 3, 6, 5],
            ],
            dtype=torch.float32,
        ).to(fp8_dtype)
        w2_weight = torch.tensor(
            [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]],
            dtype=torch.float32,
        ).to(fp8_dtype)

        experts = SimpleNamespace(
            num_local_experts=1,
            intermediate_size_per_partition=3,
            w13_weight=torch.nn.Parameter(w13_weight.unsqueeze(0), requires_grad=False),
            w2_weight=torch.nn.Parameter(w2_weight.unsqueeze(0), requires_grad=False),
            # Serialized loading creates shard-local scale layouts. Requant must
            # preserve these Parameter objects while rebinding their data.
            w13_weight_scale_inv=torch.nn.Parameter(
                torch.empty(1, 4, 2), requires_grad=False
            ),
            w2_weight_scale_inv=torch.nn.Parameter(
                torch.empty(1, 2, 2), requires_grad=False
            ),
            quant_method=SimpleNamespace(load_up_proj_weight_first=False),
        )
        moe.experts = experts
        moe._serialized_fp8_tp_scale_cache = {
            "w1": torch.tensor([[[1, 2], [3, 4], [5, 6]]], dtype=torch.float32),
            "w3": torch.tensor([[[2, 1], [4, 3], [6, 5]]], dtype=torch.float32),
            "w2": torch.tensor([[[1, 2, 3], [4, 5, 6]]], dtype=torch.float32),
        }
        moe._serialized_fp8_tp_scale_loaded = {
            "w1": {0},
            "w2": {0},
            "w3": {0},
        }

        expected_w13 = torch.cat(
            (
                BailingMoE._dequantize_serialized_fp8_tp_shard(
                    w13_weight[:3],
                    moe._serialized_fp8_tp_scale_cache["w1"][0],
                    (2, 2),
                    row_offset=3,
                ),
                BailingMoE._dequantize_serialized_fp8_tp_shard(
                    w13_weight[3:],
                    moe._serialized_fp8_tp_scale_cache["w3"][0],
                    (2, 2),
                    row_offset=3,
                ),
            ),
            dim=0,
        )
        expected_w2 = BailingMoE._dequantize_serialized_fp8_tp_shard(
            w2_weight,
            moe._serialized_fp8_tp_scale_cache["w2"][0],
            (2, 2),
            column_offset=3,
        )
        original_w13_scale_param = experts.w13_weight_scale_inv
        original_w2_scale_param = experts.w2_weight_scale_inv

        with patch.object(
            BailingMoE, "_requires_serialized_fp8_tp_requant", return_value=True
        ):
            self.assertTrue(moe.requantize_serialized_fp8_tp_weights())

        self.assertIs(experts.w13_weight_scale_inv, original_w13_scale_param)
        self.assertIs(experts.w2_weight_scale_inv, original_w2_scale_param)
        self.assertEqual(tuple(experts.w13_weight_scale_inv.shape), (1, 3, 2))
        self.assertEqual(tuple(experts.w2_weight_scale_inv.shape), (1, 2, 2))
        self.assertFalse(experts.w13_weight_scale_inv.format_ue8m0)
        self.assertFalse(experts.w2_weight_scale_inv.format_ue8m0)
        self.assertFalse(hasattr(moe, "_serialized_fp8_tp_scale_cache"))
        self.assertFalse(hasattr(moe, "_serialized_fp8_tp_scale_loaded"))

        def dequantize_local(weight, scale):
            expanded_scale = scale.repeat_interleave(2, 0).repeat_interleave(2, 1)
            return weight.float() * expanded_scale[: weight.shape[0], : weight.shape[1]]

        torch.testing.assert_close(
            dequantize_local(experts.w13_weight[0], experts.w13_weight_scale_inv[0]),
            expected_w13.float(),
            rtol=0.03,
            atol=0.1,
        )
        torch.testing.assert_close(
            dequantize_local(experts.w2_weight[0], experts.w2_weight_scale_inv[0]),
            expected_w2.float(),
            rtol=0.03,
            atol=0.1,
        )


if __name__ == "__main__":
    unittest.main()
