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
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.bailing_moe import BailingMoeV2ForCausalLM
from sglang.srt.models.qwen2_5_vl import Qwen2_5_VisionTransformer
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, is_cuda

_is_cuda = is_cuda()
logger = logging.getLogger(__name__)


class Transpose(nn.Module):
    """Transpose layer for swapping dimensions."""

    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim0, self.dim1)


class BailingMMNativeForConditionalGeneration(nn.Module):
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
        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder

        # 1. Language model - use BailingMoE
        self.model = BailingMoeV2ForCausalLM(
            config.llm_config,
            quant_config,
            prefix=add_prefix("model", prefix),
        )

        # 2. Vision encoder - use Qwen2.5 Vision Transformer
        # Under pipeline parallelism the multimodal encoders + projections only
        # run on the first PP stage (where the language model's input embedding
        # is computed). On every other stage they are replaced by
        # PPMissingLayer placeholders so that no GPU memory / weights are wasted.
        # Use explicit flags rather than `is None` checks, because nn.Module
        # placeholders are still truthy.
        self._build_mm_encoders = self.pp_group.is_first_rank

        self.vision = None
        if config.vision_config is not None:
            if self._build_mm_encoders:
                vision_config = config.vision_config

                # Determine vision encoder class based on architecture
                arch = vision_config.architectures[0]
                if arch in ["Qwen3MoeVisionTransformer", "Qwen3_VisionTransformer"]:
                    from sglang.srt.models.qwen3_vl import (
                        Qwen3VLMoeVisionModel as VisionCls,
                    )
                elif arch == "Qwen2_5_VisionTransformer":
                    VisionCls = Qwen2_5_VisionTransformer
                else:
                    raise ValueError(f"Unsupported vision architecture: {arch}")

                self.vision = VisionCls(
                    vision_config,
                    quant_config=quant_config,
                    prefix=add_prefix("vision", prefix),
                    use_data_parallel=self.use_data_parallel,
                )
                # Vision projection layer: MLP
                mlp_modules_img = [
                    nn.Linear(
                        self.config.vision_config.out_hidden_size,
                        self.model.config.hidden_size,
                    )
                ]
                for _ in range(1, self.config.mlp_depth):
                    mlp_modules_img.append(nn.GELU())
                    mlp_modules_img.append(
                        nn.Linear(
                            self.model.config.hidden_size,
                            self.model.config.hidden_size,
                        )
                    )
                self.linear_proj = nn.Sequential(*mlp_modules_img)
            else:
                self.vision = PPMissingLayer()
                self.linear_proj = PPMissingLayer()

        # 3. Audio encoder - based on Whisper
        self.audio = None
        self.linear_proj_audio = None
        if self.config.audio_config is not None:
            if self._build_mm_encoders:
                from sglang.srt.layers.modeling_whisper_encoder import (
                    WhisperAudioEncoder,
                )

                with set_default_torch_dtype(torch.float32):
                    self.audio = WhisperAudioEncoder(
                        **self.config.audio_config.whisper_encoder_config
                    )

                audio_encoder_proj = torch.nn.Conv1d(
                    self.audio.audio_emb_dim,
                    self.model.config.hidden_size,
                    kernel_size=self.config.audio_config.ds_kernel_size,
                    stride=self.config.audio_config.ds_stride,
                    padding=self.config.audio_config.ds_kernel_size // 2,
                )
                mlp_modules_audio = [
                    audio_encoder_proj,  # index should be 0 to match model weights
                    Transpose(-1, -2),
                ]
                for _ in range(1, self.config.mlp_depth):
                    mlp_modules_audio.append(nn.GELU())
                    mlp_modules_audio.append(
                        nn.Linear(
                            self.model.config.hidden_size,
                            self.model.config.hidden_size,
                        )
                    )
                self.linear_proj_audio = nn.Sequential(*mlp_modules_audio)
            else:
                self.audio = PPMissingLayer()
                self.linear_proj_audio = PPMissingLayer()

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
                self.vision, pixel_values, grid_thw.tolist(), rope_type="rope_3d"
            )
        else:
            vision_embeds = self.vision(pixel_values, grid_thw=grid_thw)

        if "deepstack_visual_indexes" in self.config.vision_config:
            assert (
                len(self.config.vision_config.deepstack_visual_indexes) + 1
            ) * self.config.vision_config.out_hidden_size == vision_embeds.shape[-1]
            vision_embeds = vision_embeds[
                ..., : self.config.vision_config.out_hidden_size
            ]

        # Project to language model dimension
        vision_embeds = self.linear_proj(vision_embeds)

        # Normalize as in original implementation
        vision_embeds = F.normalize(vision_embeds.float(), dim=-1)
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
            dtype=self.vision.dtype
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
            dtype=self.vision.dtype
        )

        # Get grid information
        video_grid_thw = torch.concat([item.video_grid_thw for item in items], dim=0)
        video_embeds = self._get_vision_feature(pixel_values, video_grid_thw)
        return video_embeds

    def _audio_pad_inputs(self, items: List[MultimodalDataItem]):
        if len(items) == 1 and items[0].audio_feature_lens.shape[-1] == 1:
            audio_feats = items[0].feature
            if len(audio_feats.shape) == 2:
                audio_feats = audio_feats.unsqueeze(0)
            return audio_feats, None
        feats_lens = torch.cat([item.audio_feature_lens.flatten() for item in items])
        feats_lens = feats_lens.cpu()
        audio_feats = []
        feat_idx = 0
        feat_inner_len = 0
        for feat_len in feats_lens:
            feat = items[feat_idx].feature
            # (1, T_i, D) -> (T_i, D)
            if len(feat.shape) == 3:
                assert feat.shape[0] == 1
                feat = feat.squeeze(0)
            audio_feats.append(feat[feat_inner_len : feat_inner_len + feat_len])
            feat_inner_len += feat_len
            if feat_inner_len >= feat.shape[0]:
                feat_idx += 1
                feat_inner_len = 0
        audio_feats = torch.nn.utils.rnn.pad_sequence(audio_feats, batch_first=True)
        # whisper encoder conv
        embeds_lens = (feats_lens - 3 + 2 * 1) // 2 + 1
        # project layer conv
        ds_kernel_size = self.config.audio_config.ds_kernel_size
        ds_stride = self.config.audio_config.ds_stride
        embeds_lens = (
            embeds_lens - ds_kernel_size + 2 * (ds_kernel_size // 2)
        ) // ds_stride + 1
        return audio_feats, embeds_lens

    def _audio_unpad_outputs(self, audio_embeds, embeds_lens):
        if embeds_lens is None:
            audio_embeds = audio_embeds.squeeze(0)
        else:
            audio_embeds = torch.cat(
                [audio_embeds[i, :len] for i, len in enumerate(embeds_lens)], dim=0
            )
        return audio_embeds

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """
        Extract audio features from input items.

        Args:
            items: List of MultimodalDataItem containing audio features

        Returns:
            Projected audio embeddings
        """
        # item.feature: [1, T, D]
        # item.audio_feature_lens: [1, n], sum(item.audio_feature_lens) = T
        assert (
            self._build_mm_encoders
        ), "Audio encoder is not available on this PP stage"
        assert self.audio is not None, "Audio encoder is not initialized"
        audio_feats, embeds_lens = self._audio_pad_inputs(items)
        audio_feats_seg = self.audio(audio_feats)
        audio_embeds = self.linear_proj_audio(audio_feats_seg.transpose(-1, -2))
        audio_embeds = self._audio_unpad_outputs(audio_embeds, embeds_lens)
        if self.config.audio_config.norm_query_embeds:
            audio_embeds = F.normalize(audio_embeds, dim=-1)  # [-1, 4096]
        return audio_embeds.to(items[0].feature.dtype)

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
                if self.vision is not None and self._build_mm_encoders
                else -1
            ),
            (
                self.config.llm_config.video_patch_token
                if self.vision is not None and self._build_mm_encoders
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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """
        Load model weights.

        Handles special cases:
        - audio.positional_embedding is loaded as buffer
        - Vision weights need name mapping
        - MoE weights use stacked params mapping
        """
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        # Also include buffers (e.g., audio.positional_embedding)
        buffers_dict = dict(self.named_buffers())

        # Use itertools.tee to create two independent iterators
        weights_for_text, weights_for_other = itertools.tee(weights, 2)

        # Generator for text model weights
        def filter_text_weights(weights_iter):
            for name, loaded_weight in weights_iter:
                if name.startswith("model."):
                    # Remove the outer "model." prefix for text model weights
                    # model.model.xxx -> model.xxx, model.lm_head.xxx -> lm_head.xxx
                    new_name = name[len("model.") :]
                    yield new_name, loaded_weight

        # Load text model weights using the first iterator
        self.model.load_weights(filter_text_weights(weights_for_text))

        for name, loaded_weight in weights_for_other:
            if name.startswith("model."):
                continue
            # Handle vision encoder weights
            if "vision." in name:
                # Adapt VisionAttention naming
                name = name.replace("attn.qkv.", "attn.qkv_proj.")

            # Try loading as parameter first, then as buffer
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            elif name in buffers_dict:
                buffers_dict[name].copy_(loaded_weight)


class BailingMM2NativeForConditionalGeneration(BailingMMNativeForConditionalGeneration):
    pass


# Entry point for model registry
EntryClass = [
    BailingMMNativeForConditionalGeneration,
    BailingMM2NativeForConditionalGeneration,
]
