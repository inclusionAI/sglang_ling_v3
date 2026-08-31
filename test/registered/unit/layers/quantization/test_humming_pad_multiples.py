"""Weight-padding multiples for the Humming MoE path.

Runs against the real `humming` package -- the multiples exist only to satisfy
that package's own asserts, so a stubbed version would be testing this file's
arithmetic rather than the contract.
"""

from __future__ import annotations

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

try:
    from humming.layer import HummingInputSchema
    from humming.schema import BaseWeightSchema
    from humming.transform import prepare_layer_config

    from sglang.srt.layers.quantization.humming_utils import (
        humming_pad_multiples,
        make_humming_deepep_input_schema,
    )

    _HAS_HUMMING = True
except ImportError:
    _HAS_HUMMING = False

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

# Ling-3.0-plus: hidden 4096, moe_intermediate_size 1280 -> 320 per rank at TP4.
HIDDEN, INTER_PER_RANK = 4096, 320
W13_N, W13_K = INTER_PER_RANK * 2, HIDDEN
W2_N, W2_K = HIDDEN, INTER_PER_RANK


def _mxfp4_weight_schema():
    """The schema production builds, via Mxfp4WeightSchema.convert_humming."""
    schema, _ = BaseWeightSchema.from_config({"quant_method": "mxfp4"}).convert_humming(
        tensors={
            "weight": torch.zeros((1, 64, HIDDEN // 2), dtype=torch.uint8),
            "weight_scale": torch.zeros((1, 64, HIDDEN // 32), dtype=torch.uint8),
        },
        shape_n_stacks=[64],
        shape_k_stacks=[HIDDEN],
        param_dtype=torch.bfloat16,
        num_experts=1,
    )
    return schema


def _padded(shape_n, shape_k, input_schema, weight_schema):
    pad_n, pad_k = humming_pad_multiples(input_schema, weight_schema, torch.bfloat16)
    cfg = prepare_layer_config(
        shape_n=shape_n,
        shape_k=shape_k,
        weight_schema=weight_schema,
        input_schema=input_schema,
        num_experts=512,
        pad_n_to_multiple=pad_n,
        pad_k_to_multiple=pad_k,
        torch_dtype=torch.bfloat16,
    )
    return cfg, (pad_n, pad_k)


@unittest.skipUnless(_HAS_HUMMING, "requires the humming package")
class TestHummingPadMultiples(CustomTestCase):
    def test_tp4_expert_shards_are_not_padded(self):
        """Ling-3.0-plus at TP4 must pay no padding.

        w13 N=640 and w2 K=320 already satisfy humming's alignment, so any
        multiple large enough to round them up is compute spent on nothing --
        the kernel iterates the padded extents. A fixed 256/128 rounded both to
        1.2x, one sixth of every MoE GEMM.
        """
        ws, ins = _mxfp4_weight_schema(), HummingInputSchema()
        for name, (n, k) in (("w13", (W13_N, W13_K)), ("w2", (W2_N, W2_K))):
            cfg, _ = _padded(n, k, ins, ws)
            self.assertEqual((cfg.shape_n, cfg.shape_k), (n, k), name)
            self.assertEqual((cfg.pad_shape_n, cfg.pad_shape_k), (0, 0), name)

    def test_deepep_fp8_dispatch_keeps_its_128_k_alignment(self):
        """The activation scale group must win over the dtype-derived minimum.

        DeepEP fp8 dispatch quantizes activations in groups of 128 along K.
        fp8 activations alone would only force K % 64, so a derivation that
        ignores input_scale_group_size silently under-aligns that path.
        """
        ws = _mxfp4_weight_schema()
        fp8_in = make_humming_deepep_input_schema("w13", W13_K)
        self.assertEqual(fp8_in.a_dtype.num_bits, 8)
        self.assertEqual(humming_pad_multiples(fp8_in, ws, torch.bfloat16)[1], 128)
        # ... while bf16 dispatch on the same weights derives the smaller one.
        self.assertEqual(
            humming_pad_multiples(HummingInputSchema(), ws, torch.bfloat16)[1], 32
        )

    def test_unaligned_shard_is_still_rounded_up_to_something_legal(self):
        """Guards over-lowering: humming's transform asserts must still hold."""
        ws, ins = _mxfp4_weight_schema(), HummingInputSchema()
        cfg, _ = _padded(600, 300, ins, ws)
        a_bits = cfg.a_dtype.num_bits
        # humming/transform.py: transform_humming_weight
        self.assertEqual(cfg.shape_n % 64, 0)
        self.assertEqual(cfg.shape_k % (2 * (256 // a_bits)), 0)
        self.assertEqual(cfg.shape_k % ws.weight_scale_group_size, 0)
        self.assertGreaterEqual(cfg.shape_n, 600)
        self.assertGreaterEqual(cfg.shape_k, 300)


if __name__ == "__main__":
    unittest.main()
