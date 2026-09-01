"""Chunked MoE finalize (finalize+AR overlap) vs. one full finalize call.

SGLANG_OPT_MOE_FINALIZE_AR_OVERLAP runs moe_fused_mul_sum over token chunks so
each chunk's TP all-reduce can start while the next chunk finalizes. The
all-reduce total is untouched; the correctness claim to pin is that finalizing
row chunks independently changes no bits versus the single full call -- for the
plain case, with output_add, with routed_scaling_factor, and with a ragged
last chunk.
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.test.test_utils import CustomTestCase

try:
    from sglang.kernels.ops.moe.moe_fused_mul_sum import moe_fused_mul_sum
except ImportError:  # triton is not installed on CPU-only runners
    moe_fused_mul_sum = None


@unittest.skipUnless(
    torch.cuda.is_available() and moe_fused_mul_sum is not None,
    "CUDA + triton required",
)
class TestHummingFinalizeArChunkEquivalence(CustomTestCase):
    def _run(self, *, num_tokens, num_chunks, with_output_add, scaling, seed=0):
        torch.manual_seed(seed)
        device = "cuda"
        top_k, size, num_experts = 8, 512, 64
        dtype = torch.bfloat16
        inputs = torch.randn(num_tokens, top_k, size, dtype=dtype, device=device)
        topk_weights = torch.rand(num_tokens, top_k, dtype=dtype, device=device)
        topk_ids = torch.randint(
            0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device
        )
        output_add = (
            torch.randn(num_tokens, size, dtype=dtype, device=device)
            if with_output_add
            else None
        )

        full = torch.empty(num_tokens, size, dtype=dtype, device=device)
        moe_fused_mul_sum(
            inputs=inputs,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            is_ep=False,
            routed_scaling_factor=scaling,
            outputs=full,
            output_add=output_add,
        )

        # Same chunk arithmetic as HummingRunnerCore._finalize_with_ar_overlap.
        chunked = torch.empty_like(full)
        chunk = -(-num_tokens // num_chunks)
        for start in range(0, num_tokens, chunk):
            end = min(start + chunk, num_tokens)
            moe_fused_mul_sum(
                inputs=inputs[start:end],
                topk_weights=topk_weights[start:end],
                topk_ids=topk_ids[start:end],
                is_ep=False,
                routed_scaling_factor=scaling,
                outputs=chunked[start:end],
                output_add=None if output_add is None else output_add[start:end],
            )

        self.assertTrue(torch.equal(full, chunked))

    def test_even_chunks(self):
        self._run(num_tokens=4096, num_chunks=4, with_output_add=False, scaling=None)

    def test_ragged_last_chunk(self):
        self._run(num_tokens=4097, num_chunks=4, with_output_add=False, scaling=None)

    def test_with_output_add_and_scaling(self):
        self._run(num_tokens=5000, num_chunks=4, with_output_add=True, scaling=2.5)

    def test_chunk_larger_than_batch_degenerates_to_one_call(self):
        self._run(num_tokens=100, num_chunks=4, with_output_add=True, scaling=None)


if __name__ == "__main__":
    unittest.main()
