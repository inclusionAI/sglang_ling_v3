"""Local audio/text configuration for Ming-ASR-Tiny checkpoints."""

import copy

from transformers import AutoConfig, PretrainedConfig

from sglang.srt.configs.bailing_hybrid import BailingHybridConfig
from sglang.srt.configs.ming_audio_processing import SGLangMingAudioProcessor
from sglang.srt.multimodal.customized_mm_processor_utils import (
    register_customized_processor,
)


class BailingWhisperAudioConfig(PretrainedConfig):
    """Whisper encoder and projector settings stored by Bailing checkpoints."""

    model_type = "bailing_whisper_audio"

    def __init__(
        self,
        whisper_encoder_config=None,
        downsampler_kernel_size_list=None,
        downsampler_stride_list=None,
        downsampler_padding_list=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.whisper_encoder_config = whisper_encoder_config or {
            "n_mels": 128,
            "n_ctx": 3000,
            "n_state": 1280,
            "n_head": 20,
            "n_layer": 32,
        }
        self.downsampler_kernel_size_list = list(downsampler_kernel_size_list or [3])
        self.downsampler_stride_list = list(downsampler_stride_list or [2])
        self.downsampler_padding_list = list(
            downsampler_padding_list
            or [kernel // 2 for kernel in self.downsampler_kernel_size_list]
        )
        if not (
            len(self.downsampler_kernel_size_list)
            == len(self.downsampler_stride_list)
            == len(self.downsampler_padding_list)
        ):
            raise ValueError(
                "audio downsampler kernel/stride/padding lists must have the same length"
            )

    @classmethod
    def from_checkpoint_config(cls, config: dict) -> "BailingWhisperAudioConfig":
        if "whisper_encoder_config" in config:
            return cls(**config)
        return cls(
            whisper_encoder_config={
                "n_mels": config.get("n_mels", 128),
                "n_ctx": config.get("n_audio_ctx", 3000),
                "n_state": config.get("n_audio_state", 1280),
                "n_head": config.get("n_audio_head", 20),
                "n_layer": config.get("n_audio_layer", 32),
            },
            downsampler_kernel_size_list=config.get(
                "downsampler_kernel_size_list", [config.get("ds_kernel_size", 3)]
            ),
            downsampler_stride_list=config.get(
                "downsampler_stride_list", [config.get("ds_stride", 2)]
            ),
            downsampler_padding_list=config.get("downsampler_padding_list"),
            chunk_frames=config.get("chunk_frames", 3000),
        )


@register_customized_processor(SGLangMingAudioProcessor)
class BailingMoeV3AudioConfig(PretrainedConfig):
    """Composite config consumed by the local Ming V3 SGLang runtime."""

    model_type = "bailing_moe_v3_audio"
    sub_configs = {
        "llm_config": BailingHybridConfig,
        "audio_config": BailingWhisperAudioConfig,
    }

    def update(self, config_dict):
        """Apply Audio language overrides without replacing its typed config.

        Language overrides accept top-level fields and text_config/llm_config
        mappings, in increasing precedence. All target the same typed LLM.
        """
        updates = copy.deepcopy(config_dict)
        nested_updates = []
        for key in ("text_config", "llm_config"):
            value = updates.pop(key, {})
            if not isinstance(value, dict):
                raise TypeError(f"Audio {key} overrides must be a mapping")
            nested_updates.append(value)
        top_level_updates = {
            key: updates[key]
            for key in (
                "max_position_embeddings",
                "rope_theta",
                "rope_scaling",
                "rope_parameters",
            )
            if key in updates
        }
        override_groups = (top_level_updates, *nested_updates)
        for group in override_groups:
            for key in ("rope_scaling", "rope_parameters"):
                value = group.get(key)
                if value is not None and not isinstance(value, dict):
                    raise TypeError(f"Audio {key} overrides must be a mapping or null")

        text_config = self.llm_config
        rope_parameters = copy.deepcopy(text_config.rope_parameters)
        language_fields = {
            key: value
            for group in override_groups
            for key, value in group.items()
            if key not in ("rope_scaling", "rope_parameters")
        }
        text_config.update(language_fields)
        text_config._sglang_audio_1d_rope = True
        for group in override_groups:
            for key in ("rope_scaling", "rope_parameters"):
                if key not in group:
                    continue
                value = group[key]
                if value is None:
                    rope_parameters = {
                        "rope_theta": rope_parameters["rope_theta"],
                        "rope_type": "default",
                    }
                    if hasattr(text_config, "partial_rotary_factor"):
                        rope_parameters["partial_rotary_factor"] = (
                            text_config.partial_rotary_factor
                        )
                else:
                    rope_parameters.update(value)
                    rope_type = value.get("rope_type", value.get("type"))
                    if rope_type is not None:
                        rope_parameters["rope_type"] = rope_type
                rope_parameters["type"] = rope_parameters["rope_type"]
            if "rope_theta" in group:
                rope_parameters["rope_theta"] = group["rope_theta"]

        text_config.rope_theta = rope_parameters["rope_theta"]
        rope_parameters.pop("mrope_section", None)
        rope_parameters.pop("video_rope", None)
        text_config.rope_parameters = rope_parameters
        text_config.rope_scaling = copy.deepcopy(rope_parameters)

        # Keep typed sub-configs when applying partial nested overrides.
        for key, value in updates.items():
            current = getattr(self, key, None)
            if isinstance(value, dict) and isinstance(current, PretrainedConfig):
                current.update(value)
            else:
                setattr(self, key, value)
        self.mrope_section = None

    def __init__(
        self,
        text_config=None,
        llm_config=None,
        audio_config=None,
        audio_token_id=157168,
        mlp_depth=2,
        **kwargs,
    ):
        kwargs.pop("model_type", None)
        if kwargs.get("architectures") in (
            ["BailingMoeV3VLForConditionalGeneration"],
            ["MingAudioV3ForConditionalGeneration"],
        ):
            kwargs["architectures"] = ["BailingMoeV3AudioForConditionalGeneration"]
        # Consume old mixed-checkpoint metadata without retaining visual components.
        for key in (
            "vision_config",
            "image_token_id",
            "video_token_id",
            "video_rope",
            "vision_start_token_id",
            "vision_end_token_id",
            "mrope_section",
        ):
            kwargs.pop(key, None)
        if text_config is None:
            text_config = llm_config
        if isinstance(text_config, dict):
            text_config = copy.deepcopy(text_config)
            for key in (
                "image_patch_token",
                "video_patch_token",
                "image_start_token",
                "video_start_token",
            ):
                text_config.pop(key, None)
            text_config = BailingHybridConfig(**text_config)
        self.llm_config = text_config or BailingHybridConfig()
        self.llm_config._sglang_audio_1d_rope = True
        if isinstance(audio_config, dict):
            audio_config = BailingWhisperAudioConfig.from_checkpoint_config(
                audio_config
            )
        self.audio_config = audio_config
        self.audio_token_id = audio_token_id
        self.mlp_depth = mlp_depth

        if not hasattr(self.llm_config, "rope_parameters"):
            self.llm_config.rope_parameters = {}
        rope_parameters = self.llm_config.rope_parameters
        rope_parameters.setdefault("rope_theta", self.llm_config.rope_theta)
        rope_parameters.setdefault("rope_type", "default")
        # Audio/text uses engine-owned 1-D sequence positions. Legacy visual
        # selectors must not switch this audio-only model back to M-RoPE.
        rope_parameters.pop("mrope_section", None)
        rope_parameters.pop("video_rope", None)
        if getattr(self.llm_config, "audio_patch_token", None) is None:
            self.llm_config.audio_patch_token = audio_token_id
        if self.llm_config.architectures is None:
            self.llm_config.architectures = ["BailingMoeV3ForCausalLM"]
        # Ming's legacy Bailing runtime upcasts the checkpoint's BF16 LM head
        # and the final hidden states before the vocab GEMM. Keep that contract
        # checkpoint/config-owned rather than changing other Bailing V3 models.
        self.llm_config.lm_head_params_dtype = "float32"
        self.llm_config.lm_head_input_dtype = "float32"

        super().__init__(**kwargs)
        if self.architectures is None:
            self.architectures = ["BailingMoeV3AudioForConditionalGeneration"]
        # Only audio and text are supported, including for legacy checkpoints.
        self.supported_modalities = ("audio",)

    def get_text_config(self, decoder: bool = False) -> PretrainedConfig:
        return self.llm_config


AutoConfig.register(
    BailingMoeV3AudioConfig.model_type, BailingMoeV3AudioConfig, exist_ok=True
)
