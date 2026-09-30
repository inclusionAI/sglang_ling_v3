"""Ming-ASR-Tiny audio/text runtime with legacy checkpoint identifiers."""

import logging
import math
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sglang.srt.configs.bailing_audio_v3 import BailingMoeV3AudioConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.bailing_moe_v3 import BailingMoeV3ForCausalLM
from sglang.srt.multimodal.ming_audio import MingAudioMetadata, MingAudioSpec
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix
from torch import Tensor

logger = logging.getLogger(__name__)


# Adapted from OpenAI Whisper: https://github.com/openai/whisper
# MIT License
#
# Copyright (c) 2022 OpenAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


class _WhisperLayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).to(dtype=x.dtype)


class _WhisperLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(dtype=x.dtype),
            None if self.bias is None else self.bias.to(dtype=x.dtype),
        )


class _WhisperConv1d(nn.Conv1d):
    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        return super()._conv_forward(
            x,
            weight.to(dtype=x.dtype),
            None if bias is None else bias.to(dtype=x.dtype),
        )


def _sinusoids(length: int, channels: int, max_timescale: int = 10000) -> Tensor:
    assert channels % 2 == 0
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, None] * inv_timescales[None, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class _WhisperMultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = _WhisperLinear(n_state, n_state)
        self.key = _WhisperLinear(n_state, n_state, bias=False)
        self.value = _WhisperLinear(n_state, n_state)
        self.out = _WhisperLinear(n_state, n_state)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, sequence_length, n_state = x.shape
        head_dim = n_state // self.n_head
        q = self.query(x).view(batch_size, sequence_length, self.n_head, head_dim)
        k = self.key(x).view(batch_size, sequence_length, self.n_head, head_dim)
        v = self.value(x).view(batch_size, sequence_length, self.n_head, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.permute(0, 2, 1, 3).flatten(start_dim=2)
        return self.out(x)


class _WhisperResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.attn = _WhisperMultiHeadAttention(n_state, n_head)
        self.attn_ln = _WhisperLayerNorm(n_state)
        self.mlp = nn.Sequential(
            _WhisperLinear(n_state, n_state * 4),
            nn.GELU(),
            _WhisperLinear(n_state * 4, n_state),
        )
        self.mlp_ln = _WhisperLayerNorm(n_state)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_ln(x))
        return x + self.mlp(self.mlp_ln(x))


class WhisperAudioEncoder(nn.Module):
    """Whisper encoder with Ming's variable-length ``[B, T, n_mels]`` input."""

    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        super().__init__()
        self.conv1 = _WhisperConv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = _WhisperConv1d(
            n_state, n_state, kernel_size=3, stride=2, padding=1
        )
        self.register_buffer("positional_embedding", _sinusoids(n_ctx, n_state))
        self.blocks = nn.ModuleList(
            [_WhisperResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = _WhisperLayerNorm(n_state)
        self.audio_emb_dim = n_state

    def forward(self, x: Tensor) -> Tensor:
        """Encode a batch of mel spectrograms shaped ``[batch, frames, mels]``."""
        x = x.transpose(1, 2)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        positional_embedding = self.positional_embedding[: x.shape[1], :]
        assert x.shape[1:] == positional_embedding.shape, "incorrect audio shape"
        x = (x + positional_embedding).to(dtype=x.dtype)

        for block in self.blocks:
            x = block(x)

        return self.ln_post(x)


class _Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.transpose(self.dim0, self.dim1)


def _audio_downsampler_config(audio_config) -> tuple[list[int], list[int], list[int]]:
    spec = MingAudioSpec.from_config(audio_config)
    return list(spec.kernels), list(spec.strides), list(spec.paddings)


def _pad_audio_features(
    items: list[MultimodalDataItem],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover individual raw-mel clips from processor-owned audio items."""
    clips = []
    lengths = []
    for item in items:
        raw_lengths = getattr(item, "audio_mel_lengths", None)
        metadata = getattr(item, "ming_audio", None)
        if metadata is not None:
            metadata = MingAudioMetadata.from_dict(metadata)
            if (
                raw_lengths is not None
                and tuple(torch.as_tensor(raw_lengths).flatten().tolist())
                != metadata.mel_lengths
            ):
                raise ValueError(
                    "Ming metadata disagrees with legacy audio_mel_lengths"
                )
            raw_lengths = metadata.mel_lengths
        if raw_lengths is None:
            raise ValueError("Bailing audio items require audio_mel_lengths")
        raw_lengths = torch.as_tensor(raw_lengths, dtype=torch.long).flatten().cpu()
        if raw_lengths.numel() == 0 or torch.any(raw_lengths <= 0):
            raise ValueError("audio_mel_lengths must contain positive lengths")
        feature = item.feature
        if not isinstance(feature, torch.Tensor) or feature.ndim != 3:
            raise ValueError(
                "Bailing audio features must have shape [batch, frames, mels]"
            )
        if feature.shape[0] == raw_lengths.numel():
            rows = list(feature)
        elif feature.shape[0] == 1 and feature.shape[1] == int(raw_lengths.sum()):
            starts = torch.cat(
                [torch.zeros(1, dtype=torch.long), raw_lengths.cumsum(0)[:-1]]
            )
            rows = [
                feature[0, int(start) : int(start + length)]
                for start, length in zip(starts, raw_lengths)
            ]
        else:
            raise ValueError(
                "Cannot align Bailing audio feature rows with audio_mel_lengths"
            )
        for row, length in zip(rows, raw_lengths):
            length = int(length)
            if length > row.shape[0]:
                raise ValueError("audio_mel_lengths exceeds audio feature length")
            clips.append(row[:length])
            lengths.append(length)
    return torch.nn.utils.rnn.pad_sequence(clips, batch_first=True), torch.tensor(
        lengths, dtype=torch.long
    )


class BailingMoeV3AudioForConditionalGeneration(nn.Module):
    """Ming V3 runtime with local audio encoding, projection, and weight mapping."""

    @staticmethod
    def shared_experts_fusion_disable_reason(hf_config, quant_config):
        return BailingMoeV3ForCausalLM.shared_experts_fusion_disable_reason(
            hf_config.llm_config, quant_config
        )

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return BailingMoeV3ForCausalLM.get_model_config_for_expert_location(
            config.llm_config
        )

    def __init__(
        self,
        config: BailingMoeV3AudioConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.pp_group = get_parallel().pp_group
        self.model = BailingMoeV3ForCausalLM(
            config=config.llm_config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()

        self.audio = None
        self.linear_proj_audio = None
        if config.audio_config is not None:
            with set_default_torch_dtype(torch.float32):
                self.audio = WhisperAudioEncoder(
                    **config.audio_config.whisper_encoder_config
                )
            self.linear_proj_audio = self._build_audio_projector()

    def _build_audio_projector(self) -> nn.Sequential:
        assert self.audio is not None
        kernels, strides, paddings = _audio_downsampler_config(self.config.audio_config)
        hidden_size = self.config.llm_config.hidden_size
        modules: list[nn.Module] = []
        input_size = self.audio.audio_emb_dim
        for kernel, stride, padding in zip(kernels, strides, paddings):
            modules.extend(
                [
                    nn.Conv1d(input_size, hidden_size, kernel, stride, padding),
                    nn.GELU(),
                ]
            )
            input_size = hidden_size
        modules.extend([_Transpose(-1, -2), nn.Linear(hidden_size, hidden_size)])
        for _ in range(2, self.config.mlp_depth):
            modules.extend([nn.GELU(), nn.Linear(hidden_size, hidden_size)])
        return nn.Sequential(*modules)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def pad_input_ids(self, input_ids: list[int], mm_inputs: MultimodalInputs):
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    def _audio_embedding_lengths(self, mel_lengths: torch.Tensor) -> torch.Tensor:
        return MingAudioSpec.from_config(self.config.audio_config).embedding_length(
            mel_lengths
        )

    def get_audio_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        if self.audio is None or self.linear_proj_audio is None:
            raise ValueError("Audio encoder is not initialized")
        features, mel_lengths = _pad_audio_features(items)
        chunks = []
        spec = MingAudioSpec.from_config(self.config.audio_config)
        for item in items:
            metadata = getattr(item, "ming_audio", None)
            if metadata is not None:
                metadata = MingAudioMetadata.from_dict(metadata)
                expected = tuple(spec.token_count(n) for n in metadata.mel_lengths)
                if metadata.token_lengths != expected:
                    raise ValueError(
                        "Ming audio token length plan disagrees with model geometry"
                    )
        for feature, length in zip(features, mel_lengths):
            chunks.extend(feature[: int(length)].split(spec.chunk_frames))
        # Encode each chunk with a fixed batch shape. Besides padded attention,
        # real BF16 weights are sensitive to GEMM kernels selected by batch size.
        # A request must not change its embeddings when unrelated clips arrive.
        outputs = []
        for chunk in chunks:
            encoded = self.audio(chunk.unsqueeze(0))
            projected = self.linear_proj_audio(encoded.transpose(-1, -2))[0]
            if projected.shape[0] != spec.embedding_length(chunk.shape[0]):
                raise ValueError("Audio projector output violates the length contract")
            outputs.append(projected)
        return F.normalize(torch.cat(outputs), dim=-1).to(dtype=features.dtype)

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
        if "mrope_section" in self.config.llm_config.rope_parameters:
            positions = forward_batch.mrope_positions

        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            multimodal_model=self,
            data_embedding_funcs={
                Modality.AUDIO: self.get_audio_feature,
            },
            positions=positions,
            pp_proxy_tensors=pp_proxy_tensors,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Dispatch checkpoint tensors once without retaining the weight stream."""
        params = dict(self.named_parameters(remove_duplicate=False))
        buffers = dict(self.named_buffers())
        loaded = set()
        prefixes = ("audio.", "linear_proj_audio.")
        ignored_visual = []

        def language_weights():
            for checkpoint_name, weight in weights:
                name = checkpoint_name.removeprefix("model.")
                if name.startswith(("visual.", "linear_proj.")):
                    ignored_visual.append(checkpoint_name)
                    continue
                if name.startswith(prefixes):
                    name = name.replace("attn.qkv.", "attn.qkv_proj.")
                    if name in params:
                        loader = getattr(
                            params[name], "weight_loader", default_weight_loader
                        )
                        loader(params[name], weight)
                    elif name in buffers:
                        if buffers[name].shape != weight.shape:
                            raise ValueError(f"Shape mismatch for {checkpoint_name}")
                        buffers[name].copy_(weight)
                    else:
                        raise ValueError(
                            f"Unexpected multimodal weight: {checkpoint_name}"
                        )
                    loaded.add(name)
                elif (
                    checkpoint_name.startswith("model.")
                    or checkpoint_name == "lm_head.weight"
                ):
                    yield checkpoint_name.replace("model.model.", "model.", 1), weight

        language_loaded = self.model.load_weights(language_weights())
        self.ignored_visual_weight_names = tuple(ignored_visual)
        if ignored_visual:
            logger.info(
                "Audio-only Ming skipped %d legacy vision tensors", len(ignored_visual)
            )
        required_audio = {
            name
            for name in (*params, *buffers)
            if name.startswith(("audio.", "linear_proj_audio."))
        }
        missing = required_audio - loaded
        if missing:
            raise ValueError(f"Missing audio checkpoint weights: {sorted(missing)}")
        # Match the shared VL runtime's guard: missing routing tensors can
        # otherwise leave random expert selection after a successful startup.
        # Inspect only parameters present on this pipeline rank.
        required_router = {
            name
            for name, _ in self.model.named_parameters()
            if name.endswith((".mlp.gate.weight", ".mlp.gate.expert_bias"))
        }
        missing_router = required_router - set(language_loaded or ())
        if missing_router:
            raise ValueError(
                f"Missing audio language router weights: {sorted(missing_router)}"
            )
        loaded.update(f"model.{name}" for name in (language_loaded or ()))
        return loaded


EntryClass = [BailingMoeV3AudioForConditionalGeneration]
