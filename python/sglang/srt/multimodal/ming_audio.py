# Copyright 2026 SGLang Team
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
"""Ming audio preprocessing and clip geometry, shared by processor and encoder.

Like MiMoAudioPipeline, this module owns waveform preparation, Mel extraction,
and token-length rules. It has no scheduler or checkpoint-code dependency.
"""

from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class MingAudioSpec:
    """Immutable preprocessing and encoder/projector length contract.

    ``chunk_frames`` is measured before the encoder's stride-two convolution.
    Chunk output lengths must be rounded separately, including the final chunk.
    """

    n_mels: int = 128
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    chunk_frames: int = 3000
    kernels: tuple[int, ...] = (3,)
    strides: tuple[int, ...] = (2,)
    paddings: tuple[int, ...] = (1,)

    def __post_init__(self):
        if self.n_mels not in (80, 128):
            raise ValueError("Ming audio requires 80 or 128 mel bins")
        if (self.sample_rate, self.n_fft, self.hop_length) != (16000, 400, 160):
            raise ValueError("Ming checkpoints require the 16 kHz Whisper frontend")
        if self.chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive")
        if not (len(self.kernels) == len(self.strides) == len(self.paddings)):
            raise ValueError(
                "audio downsampler kernel/stride/padding lists "
                "must have the same length"
            )
        if any(k <= 0 for k in self.kernels) or any(s <= 0 for s in self.strides):
            raise ValueError("audio downsampler kernels and strides must be positive")
        if any(p < 0 for p in self.paddings):
            raise ValueError("audio downsampler paddings must be nonnegative")

    @classmethod
    def from_config(cls, config):
        def get(name, default=None):
            return (
                config.get(name, default)
                if isinstance(config, dict)
                else getattr(config, name, default)
            )

        encoder = get("whisper_encoder_config") or {}
        kernels = tuple(
            get("downsampler_kernel_size_list") or [get("ds_kernel_size", 3)]
        )
        strides = tuple(get("downsampler_stride_list") or [get("ds_stride", 2)])
        paddings = get("downsampler_padding_list")
        spec = cls(
            n_mels=encoder.get("n_mels", get("n_mels", 128)),
            chunk_frames=get("chunk_frames", 3000),
            kernels=kernels,
            strides=strides,
            paddings=tuple(
                paddings if paddings is not None else [k // 2 for k in kernels]
            ),
        )
        context = encoder.get("n_ctx", get("n_audio_ctx", 3000))
        if (spec.chunk_frames + 1) // 2 > context:
            raise ValueError("audio chunk exceeds encoder positional context")
        return spec

    def embedding_length(self, mel_length):
        """Length of one chunk; accepts an integer or an integer tensor."""
        length = (mel_length + 1) // 2
        for kernel, stride, padding in zip(self.kernels, self.strides, self.paddings):
            length = (length + 2 * padding - kernel) // stride + 1
        return length

    def chunks(self, mel_length: int) -> tuple[int, ...]:
        if mel_length <= 0:
            raise ValueError("audio mel length must be positive")
        full, remainder = divmod(mel_length, self.chunk_frames)
        return (self.chunk_frames,) * full + ((remainder,) if remainder else ())

    def token_count(self, mel_length: int) -> int:
        lengths = [self.embedding_length(n) for n in self.chunks(mel_length)]
        if any(n <= 0 for n in lengths):
            raise ValueError("audio chunk is too short for the configured downsampler")
        return sum(lengths)


@dataclass(frozen=True)
class MingAudioMetadata:
    """Versioned model metadata transported opaquely by the generic engine."""

    mel_lengths: tuple[int, ...]
    token_lengths: tuple[int, ...]
    version: int = 1

    def __post_init__(self):
        if self.version != 1:
            raise ValueError(f"Unsupported Ming audio metadata version: {self.version}")
        if not self.mel_lengths or len(self.mel_lengths) != len(self.token_lengths):
            raise ValueError("Ming audio metadata must describe each clip")
        if any(n <= 0 for n in (*self.mel_lengths, *self.token_lengths)):
            raise ValueError("Ming audio lengths must be positive")

    def to_dict(self):
        return {
            "version": self.version,
            "mel_lengths": list(self.mel_lengths),
            "token_lengths": list(self.token_lengths),
        }

    @classmethod
    def from_dict(cls, value):
        return cls(
            tuple(value["mel_lengths"]), tuple(value["token_lengths"]), value["version"]
        )


# Vendored from OpenAI Whisper's MIT-licensed assets.  Checkpoint parity is
# bit-sensitive: rebuilding these filters from the published formula differs
# by 1 ULP in a few float32 entries and can flip a BF16 mel value.
_WHISPER_MEL_FILTERS_PATH = Path(__file__).parent / "assets" / "mel_filters.npz"


@lru_cache(maxsize=2)
def _whisper_mel_filters(n_mels: int) -> Tensor:
    """Load OpenAI Whisper's exact 16 kHz mel filter bank."""
    if n_mels not in (80, 128):
        raise ValueError(f"Whisper-compatible n_mels must be 80 or 128, got {n_mels}")

    with np.load(_WHISPER_MEL_FILTERS_PATH) as filters_file:
        return torch.from_numpy(filters_file[f"mel_{n_mels}"]).float()


class BailingAudioFrontend:
    """Waveform normalization, Whisper-compatible log-mel, and length helpers."""

    sample_rate = 16000
    n_fft = 400
    hop_length = 160

    def __init__(
        self,
        n_mels: int = 128,
        *,
        device: str | torch.device | None = None,
        resample_device: str | torch.device | None = None,
    ):
        self.n_mels = n_mels
        # None preserves the caller's device. Constructing a frontend must not
        # initialize CUDA before the processor's decoding workers are started.
        self.device = torch.device(device) if device is not None else None
        self._spectral_constants = None
        self.resample_device = (
            torch.device(resample_device) if resample_device is not None else None
        )
        self._resampler = None
        _whisper_mel_filters(n_mels)

    def prepare_waveform(self, waveform: Tensor, sample_rate: int) -> Tensor:
        """Resample and downmix a waveform to contiguous mono float32 samples."""
        if waveform.ndim not in (1, 2):
            raise ValueError(
                "waveform must have shape [samples] or [channels, samples], "
                f"got {tuple(waveform.shape)}"
            )
        if waveform.numel() == 0:
            raise ValueError("waveform must not be empty")
        if (
            not isinstance(sample_rate, Integral)
            or isinstance(sample_rate, bool)
            or sample_rate <= 0
        ):
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if waveform.is_complex() or not torch.isfinite(waveform).all():
            raise ValueError("waveform must contain finite real samples")

        waveform = waveform.float()
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
        if sample_rate != self.sample_rate:
            import torchaudio

            if self.resample_device is None:
                waveform = torchaudio.functional.resample(
                    waveform, sample_rate, self.sample_rate
                )
            else:
                waveform = waveform.to(self.resample_device)
                resampler = self._resampler
                if resampler is None or resampler.orig_freq != sample_rate:
                    # Match the float32 CPU kernel construction, then reuse it
                    # on the selected device. Retain only the most recent rate.
                    resampler = torchaudio.transforms.Resample(
                        sample_rate, self.sample_rate, dtype=torch.float32
                    ).to(self.resample_device)
                    self._resampler = resampler
                waveform = resampler(waveform)
        return waveform.contiguous()

    def waveform_to_mel(
        self, waveform: Tensor, output_dtype: torch.dtype = torch.bfloat16
    ) -> Tensor:
        """Return Bailing's ``[1, frames, n_mels]`` Whisper-compatible log-mel."""
        if waveform.ndim != 1:
            raise ValueError(
                f"waveform must have shape [samples], got {tuple(waveform.shape)}"
            )
        if waveform.numel() == 0:
            raise ValueError("waveform must not be empty")
        if waveform.is_complex() or not torch.isfinite(waveform).all():
            raise ValueError("waveform must contain finite real samples")
        if waveform.numel() <= self.n_fft // 2:
            raise ValueError(
                "waveform is too short for Whisper reflect padding "
                "(need more than 200 samples)"
            )

        # Preserve the target processor's explicit, service-owned device
        # contract. The production processor always chooses the target device;
        # absent a choice, retain the input device rather than reading a
        # process-global current CUDA ordinal.
        output_device = self.device if self.device is not None else waveform.device
        waveform = waveform.to(device=output_device, dtype=torch.float32)

        # Whisper's reference builds the window on CPU and then moves it to
        # the selected device. Generating it directly on CUDA differs by a few
        # ULP and can flip a rounded BF16 mel value. Cache the small immutable
        # CPU constants; move them per call without retaining request samples.
        if self._spectral_constants is None:
            self._spectral_constants = (
                torch.hann_window(self.n_fft, dtype=torch.float32),
                _whisper_mel_filters(self.n_mels),
            )
        cpu_window, cpu_filters = self._spectral_constants
        window = cpu_window.to(device=waveform.device, dtype=torch.float32)
        filters = cpu_filters.to(device=waveform.device, dtype=torch.float32)
        stft = torch.stft(
            waveform,
            self.n_fft,
            self.hop_length,
            window=window,
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs().square()
        mel = filters @ magnitudes
        log_mel = torch.clamp(mel, min=1e-10).log10()
        log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
        return (
            ((log_mel + 4.0) / 4.0)
            .transpose(0, 1)
            .unsqueeze(0)
            .to(device=output_device, dtype=output_dtype)
        )

    @staticmethod
    def audio_patch_length(
        mel_length: int,
        kernel_sizes: Sequence[int],
        strides: Sequence[int],
        paddings: Sequence[int],
    ) -> int:
        """Compute Bailing audio patch count after the encoder and downsampler."""
        spec = MingAudioSpec(
            kernels=tuple(kernel_sizes),
            strides=tuple(strides),
            paddings=tuple(paddings),
        )
        return spec.token_count(mel_length)
