# coding=utf-8
# Adapted from vllm's bailingmm.py
# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SGLang BailingMM model with multi-modal support (image, video, audio)."""

import itertools
import logging
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.multi_gate import init_multi_gate_mm_indices
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.bailing_moe_v3 import BailingMoeV3ForCausalLM
from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, is_cuda

_is_cuda = is_cuda()
logger = logging.getLogger(__name__)


class BailingMoeV3VLForConditionalGeneration(nn.Module):
    """
    BailingMM model for multi-modal conditional generation.

    This model supports:
    - Image understanding via Qwen2.5 Vision Transformer
    - Video understanding via the same Vision Transformer
    - Audio understanding via Whisper Encoder

    The model uses a BailingMoE as the language model backbone.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.norm_query_embeds = getattr(config, "norm_query_embeds", False)
        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder

        self.model = BailingMoeV3ForCausalLM(
            config=config.llm_config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        assert config.vision_config is not None
        vision_config = config.vision_config

        # Under pipeline parallelism the vision encoder + projection only run
        # on the first stage (the language model's input embedding stage). On
        # every other stage they are replaced by PPMissingLayer placeholders to
        # avoid wasting GPU memory and loading unused weights. Use explicit
        # flags to track existence, because nn.Module placeholders are still
        # truthy (`if self.visual` would be True).
        self._build_mm_encoders = self.pp_group.is_first_rank

        if self._build_mm_encoders:
            self.visual = Qwen3VLMoeVisionModel(
                vision_config,
                quant_config=quant_config,
                prefix=add_prefix("visual", prefix),
                use_data_parallel=self.use_data_parallel,
            )
        else:
            self.visual = PPMissingLayer()

        # Vision projection layer: MLP
        self.disable_merger_proj = getattr(
            config.vision_config, "disable_merger_proj", False
        )
        if self.disable_merger_proj:
            vision_out_dim = config.vision_config.hidden_size * (
                config.vision_config.spatial_merge_size**2
            )
        else:
            vision_out_dim = config.vision_config.out_hidden_size
        self.vision_out_dim = vision_out_dim
        llm_hidden_size = config.llm_config.hidden_size
        if self._build_mm_encoders:
            self.linear_proj = nn.Sequential(
                nn.Linear(vision_out_dim, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size),
            )
        else:
            self.linear_proj = PPMissingLayer()
        self.audio = None

        # mrope support flag
        rope_scaling = config.llm_config.rope_parameters
        self.is_mrope_enabled = (
            rope_scaling is not None and "mrope_section" in rope_scaling
        )

        # Pattern for padding
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()

    def get_input_embeddings(self):
        """Get input embeddings for language model."""
        return self.model.get_input_embeddings()

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        """Pad input ids with multi-modal tokens."""
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    def _get_vision_feature(self, pixel_values, grid_thw):

        # Forward through vision encoder
        if self.use_data_parallel:
            from sglang.srt.multimodal.mm_utils import run_dp_sharded_mrope_vision_model

            vision_embeds = run_dp_sharded_mrope_vision_model(
                self.visual, pixel_values, grid_thw.tolist(), rope_type="rope_3d"
            )
        else:
            vision_embeds = self.visual(pixel_values, grid_thw=grid_thw)

        if "deepstack_visual_indexes" in self.config.vision_config:
            assert (
                len(self.config.vision_config.deepstack_visual_indexes) + 1
            ) * self.vision_out_dim == vision_embeds.shape[-1]
            vision_embeds = vision_embeds[..., : self.vision_out_dim]

        # Project to language model dimension
        vision_embeds = self.linear_proj(vision_embeds)
        if self.norm_query_embeds:
            vision_embeds = F.normalize(vision_embeds, dim=-1)
        return vision_embeds

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """
        Extract image features from input items.

        Args:
            items: List of MultimodalDataItem containing image features

        Returns:
            Projected image embeddings
        """
        assert (
            self._build_mm_encoders
        ), "Vision encoder is only available on the first PP stage"

        # Concatenate pixel values from all items
        pixel_values = torch.cat([item.feature for item in items], dim=0).to(
            dtype=self.visual.dtype
        )

        # Get grid information
        image_grid_thw = torch.concat([item.image_grid_thw for item in items], dim=0)
        image_embeds = self._get_vision_feature(pixel_values, image_grid_thw)
        return image_embeds

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """
        Extract video features from input items.

        Video uses the same vision encoder as images.

        Args:
            items: List of MultimodalDataItem containing video features

        Returns:
            Projected video embeddings
        """
        assert (
            self._build_mm_encoders
        ), "Vision encoder is only available on the first PP stage"

        # Concatenate pixel values from all items
        pixel_values = torch.cat([item.feature for item in items], dim=0).to(
            dtype=self.visual.dtype
        )

        # Get grid information
        video_grid_thw = torch.concat([item.video_grid_thw for item in items], dim=0)
        video_embeds = self._get_vision_feature(pixel_values, video_grid_thw)
        return video_embeds

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        """
        Forward pass for BailingMM model.

        Args:
            input_ids: Input token IDs
            positions: Position IDs (may be mrope positions)
            forward_batch: Batch information for forward pass
            input_embeds: Pre-computed input embeddings
            get_embedding: Whether to return embeddings instead of logits
            pp_proxy_tensors: Pipeline parallel proxy tensors

        Returns:
            Hidden states or logits depending on the mode
        """
        # Handle mrope positions
        if self.is_mrope_enabled:
            positions = forward_batch.mrope_positions

        with init_multi_gate_mm_indices(
            forward_batch,
            input_ids,
            (
                self.config.llm_config.image_patch_token
                if self.visual is not None and self._build_mm_encoders
                else -1
            ),
            (
                self.config.llm_config.video_patch_token
                if self.visual is not None and self._build_mm_encoders
                else -1
            ),
            (
                self.config.llm_config.audio_patch_token
                if self.audio is not None and self._build_mm_encoders
                else -1
            ),
        ):
            # Use SGLang's general multi-modal embedding routine
            logits_result = general_mm_embed_routine(
                input_ids=input_ids,
                forward_batch=forward_batch,
                language_model=self.model,
                multimodal_model=self,
                positions=positions,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        return logits_result

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        """Expose the language backbone's MoE topology to EPLB/DeepEP."""
        return BailingMoeV3ForCausalLM.get_model_config_for_expert_location(
            config.llm_config
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """
        Load model weights.

        Handles special cases:
        - audio.positional_embedding is loaded as buffer
        - Vision weights need name mapping
        - MoE weights use stacked params mapping
        """
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        # Use itertools.tee to create two independent iterators
        weights_for_text, weights_for_other = itertools.tee(weights, 2)

        # Generator for text model weights
        def filter_text_weights(weights_iter):
            for name, loaded_weight in weights_iter:
                if name.startswith("model.") and not name.startswith("model.visual"):
                    # Remove the extra "model." prefix for text model weights
                    new_name = name.replace("model.model.", "model.")
                    yield new_name, loaded_weight
                if name.startswith("lm_head.weight"):
                    yield name, loaded_weight

        # Load text model weights using the first iterator
        self.model.load_weights(filter_text_weights(weights_for_text))

        # Load remaining weights using the second iterator
        for name, loaded_weight in weights_for_other:
            # Skip text model weights
            if name.startswith("model.") and not name.startswith("model.visual"):
                continue
            if name.startswith("lm_head.weight"):
                continue
            if name.startswith("model.visual"):
                name = name.replace("model.visual", "visual")

            # Handle vision encoder weights
            if "visual." in name:
                # Adapt VisionAttention naming
                name = name.replace("attn.qkv.", "attn.qkv_proj.")

            # Default weight loading
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


# Entry point for model registry
EntryClass = [
    BailingMoeV3VLForConditionalGeneration,
]
