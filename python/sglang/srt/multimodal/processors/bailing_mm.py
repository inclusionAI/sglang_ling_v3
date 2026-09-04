# Adapted from vllm's bailingmm.py processor
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
"""Multimodal processor for BailingMM model."""

import inspect

import torch

from sglang.srt.layers.rotary_embedding.bailing_mrope import BailingMRotaryEmbedding
from sglang.srt.managers.schedule_batch import Modality, MultimodalProcessorOutput
from sglang.srt.models.bailing_mm import (
    BailingMM2NativeForConditionalGeneration,
    BailingMMNativeForConditionalGeneration,
)
from sglang.srt.models.bailing_mm_v3 import BailingMoeV3VLForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultiModalProcessorOutput,
    MultimodalSpecialTokens,
)
from sglang.srt.utils import ImageData

# Default special tokens
DEFAULT_IMAGE_PATCH_TOKEN = "<|image_pad|>"
DEFAULT_IM_START_TOKEN = "<|vision_start|>"
DEFAULT_IM_END_TOKEN = "<|vision_end|>"
DEFAULT_VID_START_TOKEN = "<|video_start|>"
DEFAULT_VID_END_TOKEN = "<|video_end|>"
DEFAULT_FRAME_PATCH_TOKEN = "<|video_pad|>"

DEFAULT_AUDIO_PATCH_TOKEN = "<|audio_pad|>"
DEFAULT_AU_START_TOKEN = "<|audio_start|>"
DEFAULT_AU_END_TOKEN = "<|audio_end|>"


class BailingMMMultimodalProcessor(SGLangBaseProcessor):
    """
    Multimodal processor for BailingMM model.

    Handles image, video, and audio preprocessing for the BailingMM model.
    """

    models = [
        BailingMMNativeForConditionalGeneration,
        BailingMM2NativeForConditionalGeneration,
        BailingMoeV3VLForConditionalGeneration,
    ]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        # Get tokenizer from processor
        processor = _processor
        tokenizer = getattr(_processor, "tokenizer", None)
        if tokenizer is None:
            tokenizer = _processor

        def get_token_id_from_config(
            token_id_attr: str, patch_token_attr: str, fallback_token: str
        ) -> int | None:
            """Prefer checkpoint token IDs over legacy tokenizer token names."""
            token_id = getattr(hf_config, token_id_attr, None)
            if token_id is not None:
                return token_id

            # Bailing checkpoints have used both ``llm_config`` and
            # ``text_config`` for the text-model subconfig.
            for text_config_attr in ("llm_config", "text_config"):
                text_config = getattr(hf_config, text_config_attr, None)
                token_id = getattr(text_config, patch_token_attr, None)
                if token_id is not None:
                    return token_id

            # Backward compatibility for checkpoints whose config does not
            # expose the token IDs.
            return tokenizer.convert_tokens_to_ids(fallback_token)

        self.image_token_id = get_token_id_from_config(
            "image_token_id", "image_patch_token", DEFAULT_IMAGE_PATCH_TOKEN
        )
        self.video_token_id = get_token_id_from_config(
            "video_token_id", "video_patch_token", DEFAULT_FRAME_PATCH_TOKEN
        )
        self.audio_token_id = get_token_id_from_config(
            "audio_token_id", "audio_patch_token", DEFAULT_AUDIO_PATCH_TOKEN
        )

        def get_token_str(*attrs: str, fallback: str) -> str:
            for source in (processor, tokenizer):
                for attr in attrs:
                    token = getattr(source, attr, None)
                    if token is not None:
                        return token
            return fallback

        self.image_token = (
            get_token_str(
                "vision_start_token",
                "vision_bos_token",
                fallback="<|vision_start|>",
            )
            + get_token_str("image_token", fallback="<|image_pad|>")
            + get_token_str(
                "vision_end_token", "vision_eos_token", fallback="<|vision_end|>"
            )
        )
        self.video_token = (
            get_token_str(
                "video_start_token", "video_bos_token", fallback="<|video_start|>"
            )
            + get_token_str("video_token", fallback="<|video_pad|>")
            + get_token_str(
                "video_end_token", "video_eos_token", fallback="<|video_end|>"
            )
        )
        self.audio_token = (
            get_token_str(
                "audio_start_token", "audio_bos_token", fallback="<|audio_start|>"
            )
            + get_token_str("audio_token", fallback="<|audio_pad|>")
            + get_token_str(
                "audio_end_token", "audio_eos_token", fallback="<|audio_end|>"
            )
        )

        # Recognize only the canonical placeholder strings. Legacy
        # angle-bracket markers intentionally do not match.
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=self.image_token,
            video_token=self.video_token,
            audio_token=self.audio_token,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            audio_token_id=self.audio_token_id,
        ).build(_processor)

        # Update attribute to modality mapping
        self.ATTR_NAME_TO_MODALITY.update(
            {
                "audio_feats": Modality.AUDIO,
                "audio_feats_lengths": Modality.AUDIO,
                "encoder_feats_lengths": Modality.AUDIO,
            }
        )

    def process_mm_data(
        self, input_text, images=None, videos=None, audios=None, **kwargs
    ) -> dict:
        processor = self._processor
        images = images if images else None
        videos = videos if videos else None
        audios = audios if audios else None
        # if audios is not None:
        #    audios = [(torch.tensor(audios), processor.audio_processor.sample_rate)]
        device = "cuda"
        processor_kwargs = {
            "text": [input_text],
            "return_tensors": "pt",
            "device": device,
        }
        if images is not None:
            processor_kwargs["images"] = images
        if videos is not None:
            processor_kwargs["videos"] = videos

        signature_params = inspect.signature(processor.__call__).parameters
        if audios is not None and "audios" in signature_params:
            processor_kwargs["audios"] = audios
            if "audio_kwargs" in signature_params:
                processor_kwargs["audio_kwargs"] = {"use_whisper_encoder": True}

        batch_inputs = processor(**processor_kwargs)
        mm_key_map = {
            "audio_feats": "audio_features",
            "audio_feats_lengths": "audio_feature_lens",
        }
        mm_inputs = {}
        for k, v in batch_inputs.items():
            if k in mm_key_map:
                k = mm_key_map[k]
            if k in self.FEATURE_NAMES:
                v = v.to(dtype=torch.bfloat16, device=device)
            mm_inputs[k] = v
        return mm_inputs

    def _processor_fetch_mm_input(
        self,
        prompt: str = "",
        request_obj=None,
    ):
        if request_obj is None:
            return None

        # Convert prompt into str
        if isinstance(prompt, list):
            assert len(prompt) and isinstance(prompt[0], int)
            prompt_str = self._processor.tokenizer.decode(prompt)
        else:
            assert isinstance(prompt, str)
            prompt_str = prompt

        mm_sampling_kwargs = getattr(request_obj, "mm_sampling_kwargs", {})
        if mm_sampling_kwargs is None:
            mm_sampling_kwargs = {}
        # Launch-time global default for image resize (from --mm-process-config
        # {"image":{...}}). Only min_pixels/max_pixels are forwarded because those
        # are the only keys fetch_image reads for the smart_resize branch; other
        # image_config keys are deliberately dropped to avoid breaking fetch_image.
        # Request-level mm_sampling_kwargs takes precedence over the launch default.
        image_config = getattr(self, "image_config", None) or {}
        launch_image_defaults = {
            k: image_config[k]
            for k in ("min_pixels", "max_pixels")
            if k in image_config
        }
        merged_image_kwargs = {**launch_image_defaults, **mm_sampling_kwargs}
        contents = []
        if request_obj.image_data is not None:
            for item in request_obj.image_data:
                # The OpenAI endpoint wraps images in ImageData, while the native
                # /generate endpoint passes URL/data-URI strings through directly.
                url = item.url if isinstance(item, ImageData) else item
                contents.append(
                    dict(
                        type="image_url",
                        image_url=dict(url=url),
                        **merged_image_kwargs,
                    )
                )
        if request_obj.video_data is not None:
            for item in request_obj.video_data:
                assert isinstance(item, (dict, str))
                url = item["url"] if isinstance(item, dict) else item
                contents.append(
                    dict(
                        type="video_url",
                        video_url=dict(url=url),
                        **mm_sampling_kwargs,
                    )
                )
        if request_obj.audio_data is not None:
            for item in request_obj.audio_data:
                assert isinstance(item, (dict, str))
                url = item["url"] if isinstance(item, dict) else item
                contents.append(
                    dict(
                        type="audio_url",
                        audio_url=dict(url=url),
                    )
                )

        image_inputs, video_inputs, audio_inputs = self._processor.process_vision_info(
            conversations=[{"content": contents}]
        )
        # use when audio processor support gpu
        # if audio_inputs:
        #    audio_inputs = [(waveform.cuda(), sr) for waveform, sr in audio_inputs]

        return BaseMultiModalProcessorOutput(
            images=image_inputs if image_inputs is not None else [],
            audios=audio_inputs if audio_inputs is not None else [],
            videos=video_inputs if video_inputs is not None else [],
            input_text=prompt_str,
        )

    async def process_mm_data_async(
        self,
        input_text: str = "",
        request_obj=None,
        *args,
        **kwargs,
    ):
        """
        Process multimodal data for BailingMM model.

        Args:
            input_text: Input text prompt
            request_obj: Request object with additional parameters

        Returns:
            Dictionary with processed multimodal data
        """
        # Load multimodal data
        base_output = self._processor_fetch_mm_input(
            prompt=input_text,
            request_obj=request_obj,
        )

        if base_output is None:
            return None

        # Process and combine multimodal data
        # ret: pixel_values, image_grid_thw,
        #    : pixel_values_videos, video_grid_thw, video_timestamps_seconds
        #    : audio_feats -> audio_features, audio_feats_lengths -> audio_feature_lens, encoder_feats_lengths(x)
        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        input_ids_flat = input_ids.flatten()
        mrope_positions, mrope_position_delta = (
            BailingMRotaryEmbedding.bailing_3drope_get_input_positions_tensor(
                input_ids_flat,
                self.hf_config,
                image_grid_thw=ret.get("image_grid_thw", None),
                video_grid_thw=ret.get("video_grid_thw", None),
            )
        )
        mrope_positions = mrope_positions.squeeze(1)

        return MultimodalProcessorOutput(
            mm_items=mm_items,
            input_ids=input_ids.tolist(),
            im_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            audio_token_id=self.audio_token_id,
            mrope_positions=mrope_positions,
            mrope_position_delta=mrope_position_delta,
        )
