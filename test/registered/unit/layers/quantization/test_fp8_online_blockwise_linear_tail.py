import unittest
from unittest import mock

import torch

from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.fp8_utils import Fp8GemmRunnerBackend
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import (
    assert_output_close,
    init_single_process_dist,
    load_linear_weights,
    make_tp1_column_parallel_linear,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")

_BLOCK = 128


@unittest.skipUnless(
    torch.cuda.is_available() and get_device_sm() >= 89,
    "Online blockwise FP8 linear requires NVIDIA FP8 support",
)
class TestFp8OnlineBlockwiseLinearTail(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist()

    def test_bf16_k160_weight_is_padded_quantized_and_applied(self):
        torch.manual_seed(11)
        batch_size, output_size, logical_k = 7, 256, 160
        padded_k = 256

        with mock.patch.object(
            fp8_utils,
            "FP8_GEMM_RUNNER_BACKEND",
            Fp8GemmRunnerBackend.TRITON,
        ):
            quant_config = Fp8Config(
                is_checkpoint_fp8_serialized=False,
                is_online_block_quant=True,
                activation_scheme="dynamic",
                weight_block_size=[_BLOCK, _BLOCK],
            )
            layer = make_tp1_column_parallel_linear(
                quant_config, output_size, logical_k
            )
            weight = (
                torch.randn(
                    (output_size, logical_k),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                / 10
            )
            load_linear_weights(layer, weight=weight)
            self.assertEqual(layer.weight.dtype, torch.bfloat16)
            self.assertEqual(tuple(layer.weight.shape), (output_size, logical_k))

            layer.quant_method.process_weights_after_loading(layer)

            self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
            self.assertEqual(tuple(layer.weight.shape), (output_size, padded_k))
            self.assertEqual(tuple(layer.weight_scale_inv.shape), (2, 2))
            self.assertEqual(layer._online_blockwise_fp8_input_padding, 96)
            self.assertTrue(layer._online_blockwise_fp8_quantized)
            torch.testing.assert_close(
                layer.weight[:, logical_k:].float(),
                torch.zeros(
                    (output_size, padded_k - logical_k),
                    device="cuda",
                    dtype=torch.float32,
                ),
                rtol=0,
                atol=0,
            )

            quantized_blocks = layer.weight.float().reshape(2, _BLOCK, 2, _BLOCK)
            dequantized_weight = (
                quantized_blocks * layer.weight_scale_inv[:, None, :, None]
            ).reshape(output_size, padded_k)

            x = (
                torch.randn(
                    (batch_size, logical_k), device="cuda", dtype=torch.bfloat16
                )
                / 10
            )
            actual = layer.quant_method.apply(layer, x)
            expected = x.float() @ dequantized_weight[:, :logical_k].T

        self.assertEqual(tuple(x.shape), (batch_size, logical_k))
        assert_output_close(self, actual, expected, rtol=5e-2, atol=1e-1)


if __name__ == "__main__":
    unittest.main()
