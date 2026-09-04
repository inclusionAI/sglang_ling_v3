"""A hybrid prefill ending at a Mamba fork preserves both state snapshots."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPrefillAdderMambaBranchSplit(unittest.TestCase):
    def setUp(self):
        self.enabled = patch(
            "sglang.srt.managers.schedule_policy.mamba_extra_buffer_enabled",
            return_value=True,
        )
        self.chunk = patch(
            "sglang.srt.managers.schedule_policy.mamba_cache_chunk_size",
            return_value=64,
        )
        self.enabled.start()
        self.chunk.start()
        self.addCleanup(self.enabled.stop)
        self.addCleanup(self.chunk.stop)

    def _adder(self, chunk_limit=32768):
        adder = object.__new__(PrefillAdder)
        adder.page_size = 1
        adder.is_hybrid_ssm_cache = True
        adder.dllm_config = None
        adder.rem_chunk_tokens = chunk_limit
        adder.can_run_list = []
        adder.rem_input_tokens = 100000
        adder.exact_chunk_fill = False
        adder.new_chunked_req = None
        adder._check_prefill_budget = lambda *args, **kwargs: (True, chunk_limit)
        adder._check_prefill_tile_budget = lambda _: None
        return adder

    def _req(self, prompt=10000, prefix=0, branch=2112):
        return SimpleNamespace(
            prefix_indices=list(range(prefix)),
            full_untruncated_fill_ids=list(range(prompt)),
            mamba_branching_seqlen=branch,
            sampling_params=SimpleNamespace(max_new_tokens=8),
        )

    def _select(self, adder, req, has_chunked_req=False):
        return adder._select_prefill_admission(
            req,
            total_tokens=100000,
            host_hit_length=0,
            swa_host_hit_length=0,
            truncation_align_size=None,
            has_chunked_req=has_chunked_req,
        )

    def test_fork_inside_admitted_prefill_becomes_chunk(self):
        admission = self._select(self._adder(), self._req())
        self.assertEqual(admission.extend_len, 2112)
        self.assertTrue(admission.is_chunked)
        self.assertEqual(admission.max_new_tokens, 0)

    def test_fork_beyond_budget_waits_for_later_chunk(self):
        admission = self._select(self._adder(4096), self._req(branch=5120))
        self.assertEqual(admission.extend_len, 4096)
        self.assertTrue(admission.is_chunked)

    def test_host_hit_and_unaligned_fork(self):
        adder = self._adder()
        req = self._req(branch=2112)
        admission = adder._select_prefill_admission(
            req,
            total_tokens=100000,
            host_hit_length=1024,
            swa_host_hit_length=0,
            truncation_align_size=None,
        )
        self.assertEqual(admission.prefix_len, 1024)
        self.assertEqual(admission.extend_len, 1088)

        admission = self._select(adder, self._req(branch=2100))
        self.assertEqual(admission.extend_len, 10000)
        self.assertFalse(admission.is_chunked)

    def test_disabled_or_existing_chunk_does_not_split(self):
        req = self._req()
        admission = self._select(self._adder(chunk_limit=None), req)
        self.assertFalse(admission.is_chunked)
        admission = self._select(self._adder(), req, has_chunked_req=True)
        self.assertFalse(admission.is_chunked)


if __name__ == "__main__":
    unittest.main()
