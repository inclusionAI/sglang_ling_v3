import unittest

import torch

from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
    DsparkDraftSampler,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase


class _FakeLmHead:
    org_vocab_size = 8
    weight = torch.empty((8, 4), dtype=torch.float32)


class _CountingMarkovHead:
    def __init__(self):
        self.calls = 0

    def sample_block(self, base_logits, **kwargs):
        self.calls += 1
        bs, gamma, _ = base_logits.shape
        tokens = torch.full(
            (bs, gamma), self.calls, dtype=torch.int64, device=base_logits.device
        )
        corrected = (
            torch.full_like(base_logits, float(self.calls))
            if kwargs.get("collect_corrected", False)
            else None
        )
        return tokens, corrected


class _FakeModel:
    sample_from_anchor = True
    _use_fp32_lm_head = False
    lm_head = _FakeLmHead()

    def __init__(self):
        self.markov_head = _CountingMarkovHead()

    def compute_base_logits(self, hidden_states):
        return torch.zeros(
            (hidden_states.shape[0], self.lm_head.org_vocab_size),
            dtype=torch.float32,
            device=hidden_states.device,
        ), None


register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDsparkDraftSampler(CustomTestCase):
    def test_markov_block_is_sampled_once(self):
        model = _FakeModel()
        sampler = DsparkDraftSampler(
            model=model,
            gamma=3,
            max_bs=2,
            device="cpu",
            folded_sampling=False,
        )

        sampler(torch.zeros((6, 4)), torch.zeros(6, dtype=torch.int64))

        self.assertEqual(model.markov_head.calls, 1)
        torch.testing.assert_close(sampler.out, torch.ones_like(sampler.out))

    def test_folded_tokens_and_corrected_logits_come_from_same_sample(self):
        model = _FakeModel()
        sampler = DsparkDraftSampler(
            model=model,
            gamma=3,
            max_bs=2,
            device="cpu",
            folded_sampling=True,
        )

        sampler(torch.zeros((6, 4)), torch.zeros(6, dtype=torch.int64))

        self.assertEqual(model.markov_head.calls, 1)
        torch.testing.assert_close(sampler.out, torch.ones_like(sampler.out))
        torch.testing.assert_close(
            sampler.corrected_out, torch.ones_like(sampler.corrected_out)
        )


if __name__ == "__main__":
    unittest.main()
