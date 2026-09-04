"""Unit tests for the Bailing mRoPE output shape contract."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.rotary_embedding.bailing_mrope import (
    BailingMRotaryEmbedding,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _config():
    return SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=1),
        llm_config=SimpleNamespace(
            image_patch_token=99,
            video_patch_token=98,
            image_start_token=97,
            video_start_token=96,
            use_interleaved_frame_timestamp=False,
        ),
    )


class TestBailingMropeShape(CustomTestCase):
    def test_multimodal_positions_include_request_axis(self):
        positions, _ = (
            BailingMRotaryEmbedding.bailing_3drope_get_input_positions_tensor(
                input_ids=torch.tensor([97, 99]),
                hf_config=_config(),
                image_grid_thw=torch.tensor([[1, 1, 1]]),
                video_grid_thw=None,
            )
        )
        self.assertEqual(positions.shape, torch.Size([3, 1, 2]))

    def test_single_position_stays_two_dimensional_after_processor_squeeze(self):
        positions, _ = (
            BailingMRotaryEmbedding.bailing_3drope_get_input_positions_tensor(
                input_ids=torch.tensor([42]),
                hf_config=_config(),
                image_grid_thw=torch.tensor([[1, 1, 1]]),
                video_grid_thw=None,
            )
        )
        self.assertEqual(positions.shape, torch.Size([3, 1, 1]))
        self.assertEqual(positions.squeeze(1).shape, torch.Size([3, 1]))

    def test_text_only_positions_use_the_same_shape_contract(self):
        positions, _ = (
            BailingMRotaryEmbedding.bailing_3drope_get_input_positions_tensor(
                input_ids=torch.tensor([10, 11]),
                hf_config=_config(),
                image_grid_thw=None,
                video_grid_thw=None,
            )
        )
        self.assertEqual(positions.shape, torch.Size([3, 1, 2]))


if __name__ == "__main__":
    unittest.main()
