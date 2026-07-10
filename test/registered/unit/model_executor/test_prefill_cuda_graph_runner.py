"""Unit tests for PrefillCudaGraphRunner."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.model_executor.cuda_graph_buffer_registry import build_prefill_registry
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    _PREFILL_STATIC_FIELDS,
    PrefillCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPrefillCudaGraphRunnerMultimodalBCG(CustomTestCase):
    def _make_runner(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.device = torch.device("cpu")
        runner.max_bs = 1
        runner.capture_num_tokens = [8]
        runner.max_num_tokens = 8
        runner._capture_req_slots = 1
        runner.prefill_backend_name = Backend.BREAKABLE
        runner.require_mlp_tp_gather = False
        runner.require_attn_tp_gather = False
        runner.dp_size = 1
        runner.buffers = SimpleNamespace()
        runner.model_runner = SimpleNamespace(attn_backend=object())
        runner.capture_return_pooled_hidden_states = False
        runner.capture_hidden_mode = CaptureHiddenMode.NULL
        runner.static_draft_hidden_states = None
        runner._is_full_backend = False
        runner._full_cg_seq_lens_cpu = None
        runner.tbo_plugin = SimpleNamespace(
            capture_one_batch_size=lambda forward_batch, num_tokens: None
        )

        source = SimpleNamespace(
            input_ids=torch.zeros(8, dtype=torch.int64),
            positions=torch.zeros(8, dtype=torch.int64),
            out_cache_loc=torch.zeros(8, dtype=torch.int64),
            mrope_positions=torch.zeros((3, 8), dtype=torch.int64),
            input_embeds=torch.zeros((8, 4), dtype=torch.float32),
        )
        runner.buffer_registry = build_prefill_registry(
            device=runner.device,
            max_bs=runner.max_bs,
            max_num_token=runner.max_num_tokens,
            cache_loc_dtype=torch.int64,
            is_multimodal=True,
            hidden_size=4,
            embed_dtype=torch.float32,
            share_pool=False,
            source=source,
        )
        runner._prefill_static_buffers = {
            name: torch.zeros((runner.max_bs,), dtype=torch.int64)
            for name in _PREFILL_STATIC_FIELDS
        }
        runner._next_token_logits_buffer = lambda rows: None
        runner._prepare_forward_metadata_for_replay = (
            lambda forward_batch, static_forward_batch, num_tokens: None
        )
        return runner

    def _make_forward_batch(self):
        return ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            input_ids=torch.tensor([10, 11, 12], dtype=torch.int64),
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([3], dtype=torch.int64),
            out_cache_loc=torch.tensor([0, 1, 2], dtype=torch.int64),
            seq_lens_sum=3,
            orig_seq_lens=torch.tensor([3], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([3], dtype=torch.int64),
            extend_num_tokens=3,
            extend_seq_lens=torch.tensor([3], dtype=torch.int64),
            extend_prefix_lens=torch.tensor([0], dtype=torch.int64),
            extend_start_loc=torch.tensor([0], dtype=torch.int64),
            extend_prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
            extend_seq_lens_cpu=torch.tensor([3], dtype=torch.int64),
            extend_logprob_start_lens_cpu=torch.tensor([3], dtype=torch.int64),
            positions=torch.tensor([0, 1, 2], dtype=torch.int64),
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
        )

    def test_capture_prepare_keeps_input_embeds_slot_for_breakable_backend(self):
        runner = self._make_runner()

        forward_batch, _ = runner.capture_prepare(num_tokens=4)

        expected_slot = runner.buffer_registry.get_slot("input_embeds").slice_for(1, 4)
        self.assertIsNotNone(forward_batch.input_embeds)
        self.assertEqual(
            forward_batch.input_embeds.data_ptr(), expected_slot.data_ptr()
        )
        self.assertEqual(tuple(forward_batch.input_embeds.shape), (4, 4))

    def test_load_batch_keeps_input_embeds_slot_for_breakable_backend(self):
        runner = self._make_runner()
        forward_batch = self._make_forward_batch()

        static_forward_batch = runner.load_batch(forward_batch)

        expected_slot = runner.buffer_registry.get_slot("input_embeds").slice_for(1, 8)
        self.assertIsNotNone(static_forward_batch.input_embeds)
        self.assertEqual(
            static_forward_batch.input_embeds.data_ptr(), expected_slot.data_ptr()
        )
        self.assertEqual(tuple(static_forward_batch.input_embeds.shape), (8, 4))


if __name__ == "__main__":
    unittest.main()
