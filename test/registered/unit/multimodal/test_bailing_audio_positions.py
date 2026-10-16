"""Regression tests for Ming audio multimodal."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.multimodal.processors.bailing_audio import BailingAudioProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestBailingAudioPositions(CustomTestCase):
    def test_audio_uses_engine_positions_after_placeholder_expansion(self):
        processor = BailingAudioProcessor.__new__(BailingAudioProcessor)
        processor.spec = SimpleNamespace(token_count=lambda frames: 2)
        processor.audio_token_id = 2
        processor.audio_start_token = "<audio>"
        processor.audio_patch_token = "<audioPatch>"
        processor.audio_end_token = "</audio>"
        processor.tokenizer = lambda text, **kw: SimpleNamespace(
            input_ids=torch.tensor([[10, 1, 2, 2, 3, 20]])
        )
        result = processor.build_output("before<AUDIO>after", [torch.zeros(1, 8, 128)])
        self.assertEqual(result.input_ids, [10, 1, 2, 2, 3, 20])
        self.assertEqual(result.mm_items[0].offsets, [(2, 3)])
        self.assertIs(result.mrope_positions, None)
        self.assertIs(result.mrope_position_delta, None)


if __name__ == "__main__":
    unittest.main()
