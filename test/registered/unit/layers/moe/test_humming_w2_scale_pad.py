"""FP8-activation w2 scales must match humming's padded K.

humming pads a sublayer's weight along K to its input alignment (Ling-3.0-plus
at TP8: the w2 intermediate 160 -> 192 for FP8 activations) and its launcher
then checks the quantized activation's group scales at the padded width while
the activation itself stays at the logical K. Without the extra zero groups the
first MoE forward dies in `check_tensor_common` (`as.size(1) 5 != 6`). Pin the
helper the indexed-gemm path routes w2 scales through.
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.test.test_utils import CustomTestCase

try:
    from sglang.srt.layers.moe.moe_runner.humming import pad_input_scale_to_padded_k

    _HAS_HUMMING = True
except ImportError:
    _HAS_HUMMING = False


@unittest.skipUnless(_HAS_HUMMING, "requires the humming package")
class TestHummingW2ScalePad(CustomTestCase):
    def test_tp8_fp8_w2_scales_gain_one_zero_group(self):
        # Ling-3.0-plus TP8: logical K 160 = 5 groups of 32, weight padded to 192.
        scale = torch.rand(7, 5, dtype=torch.float32)
        padded = pad_input_scale_to_padded_k(scale, padded_k=192, group_size=32)
        self.assertEqual(tuple(padded.shape), (7, 6))
        self.assertTrue(torch.equal(padded[:, :5], scale))
        self.assertTrue(torch.equal(padded[:, 5], torch.zeros(7)))

    def test_aligned_k_passes_through_untouched(self):
        # TP4: logical K 320 = 10 groups, humming needs no padding.
        scale = torch.rand(7, 10, dtype=torch.float32)
        self.assertIs(
            pad_input_scale_to_padded_k(scale, padded_k=320, group_size=32), scale
        )

    def test_unquantized_or_ungrouped_inputs_pass_through(self):
        self.assertIsNone(
            pad_input_scale_to_padded_k(None, padded_k=192, group_size=32)
        )
        scale = torch.rand(7, 5, dtype=torch.float32)
        self.assertIs(
            pad_input_scale_to_padded_k(scale, padded_k=192, group_size=0), scale
        )
        per_tensor = torch.rand(7, dtype=torch.float32)
        self.assertIs(
            pad_input_scale_to_padded_k(per_tensor, padded_k=192, group_size=32),
            per_tensor,
        )


if __name__ == "__main__":
    unittest.main()
