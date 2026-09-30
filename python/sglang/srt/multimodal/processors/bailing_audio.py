"""Local Bailing audio prompt and feature integration."""

import re
from typing import Iterable, Sequence

import torch
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.models.bailing_audio_v3 import (
    BailingMoeV3AudioForConditionalGeneration,
)
from sglang.srt.multimodal.bailing_audio_io import load_bailing_audio_native
from sglang.srt.multimodal.ming_audio import (
    BailingAudioFrontend,
    MingAudioMetadata,
    MingAudioSpec,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.runtime_context import get_device
from sglang.srt.utils.common import CLIENT_MEDIA_EXCEPTIONS

_AUDIO_FORMS = (
    ("<audio>", "<audioPatch>", "</audio>"),
    ("<|audio_start|>", "<|audio_pad|>", "<|audio_end|>"),
)
_AUDIO_PLACEHOLDER = "<AUDIO>"


def audio_frontend_device(server_args, *, gpu_enabled=True) -> torch.device:
    """Resolve the selected frontend device within this service."""
    if not gpu_enabled:
        return torch.device("cpu")
    # Raw ServerArgs.device can remain None after auto-detection; the runtime
    # publishes the resolved backend separately from the construction record.
    if get_device().device != "cuda" or not torch.cuda.is_available():
        raise ValueError("GPU audio preprocessing requires a CUDA service")
    return torch.device("cuda", server_args.base_gpu_id)


def normalize_bailing_audio_placeholders(input_text: str, *, has_audios: bool) -> str:
    """Map supported Bailing audio placeholders to one local marker."""
    if not has_audios:
        return input_text
    for start, patch, end in _AUDIO_FORMS:
        input_text = re.sub(
            re.escape(start) + f"(?:{re.escape(patch)})+" + re.escape(end),
            _AUDIO_PLACEHOLDER,
            input_text,
        )
    return input_text


def get_bailing_audio_downsampler_config(
    audio_config,
) -> tuple[list[int], list[int], list[int]]:
    """Read Bailing's list-based downsampler config with scalar fallbacks."""
    spec = MingAudioSpec.from_config(audio_config)
    return list(spec.kernels), list(spec.strides), list(spec.paddings)


class BailingAudioProcessor:
    """Own local Bailing audio placeholder expansion and MM output creation."""

    def __init__(self, tokenizer, audio_config):
        self.tokenizer = tokenizer
        self.audio_config = audio_config
        self.spec = MingAudioSpec.from_config(audio_config)
        self.audio_patch_token, self.audio_token_id = self._resolve_audio_patch_token()
        self.audio_start_token, _, self.audio_end_token = next(
            form for form in _AUDIO_FORMS if form[1] == self.audio_patch_token
        )
        self.downsampler_config = get_bailing_audio_downsampler_config(audio_config)

    def _resolve_audio_patch_token(self) -> tuple[str, int]:
        unk_token_id = getattr(self.tokenizer, "unk_token_id", None)
        for _, token, _ in _AUDIO_FORMS:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id != unk_token_id:
                return token, int(token_id)
        raise ValueError("tokenizer missing required Bailing audio patch token")

    def prepare_audio_features(
        self,
        audios: Iterable[tuple[torch.Tensor, int]],
        frontend: BailingAudioFrontend,
    ) -> list[torch.Tensor]:
        """Convert decoded waveforms into local Whisper-compatible mel features."""
        return [
            frontend.waveform_to_mel(frontend.prepare_waveform(waveform, sample_rate))
            for waveform, sample_rate in audios
        ]

    def build_output(
        self, input_text: str, audio_features: Sequence[torch.Tensor]
    ) -> MultimodalProcessorOutput:
        """Expand audio markers and construct the standard SGLang audio item."""
        if not audio_features:
            raise ValueError(
                "Bailing audio processor requires at least one audio feature"
            )

        mel_lengths = []
        for feature in audio_features:
            if feature.ndim != 3 or feature.shape[0] != 1:
                raise ValueError(
                    "Bailing audio features must have shape [1, frames, n_mels], "
                    f"got {tuple(feature.shape)}"
                )
            mel_lengths.append(int(feature.shape[1]))

        patch_lengths = [self.spec.token_count(length) for length in mel_lengths]
        input_text = normalize_bailing_audio_placeholders(input_text, has_audios=True)
        placeholder_count = input_text.count(_AUDIO_PLACEHOLDER)
        if placeholder_count != len(audio_features):
            raise ValueError(
                "Bailing audio input count does not match prompt placeholders: "
                f"{len(audio_features)} audio inputs, {placeholder_count} placeholders"
            )

        expanded = input_text
        for patch_length in patch_lengths:
            replacement = (
                self.audio_start_token
                + self.audio_patch_token * patch_length
                + self.audio_end_token
            )
            expanded = expanded.replace(_AUDIO_PLACEHOLDER, replacement, 1)
        input_ids = self.tokenizer(expanded, return_tensors="pt").input_ids.flatten()
        offsets = self._get_audio_offsets(input_ids, patch_lengths)

        items = []
        for feature, mel_length, patch_length, offset in zip(
            audio_features, mel_lengths, patch_lengths, offsets
        ):
            # One clip per item matches the standard multimodal cache contract.
            # Its feature byte length includes its raw length; no extra shared hash
            # metadata is needed to distinguish different clip boundaries.
            item = MultimodalDataItem(
                modality=Modality.AUDIO, feature=feature, offsets=[offset]
            )
            item.audio_feature_lens = torch.tensor([[patch_length]], dtype=torch.long)
            item.audio_mel_lengths = torch.tensor([mel_length], dtype=torch.long)
            item.ming_audio = MingAudioMetadata(
                (mel_length,), (patch_length,)
            ).to_dict()
            items.append(item)
        # Bailing audio follows Qwen-style ASR M-RoPE: all three axes share
        # the sequential token index. Keep delta at zero so decode positions
        # continue normally after the expanded audio placeholder tokens.
        mrope_positions = (
            torch.arange(input_ids.numel(), dtype=torch.long)
            .unsqueeze(0)
            .expand(3, -1)
            .clone()
        )
        return MultimodalProcessorOutput(
            mm_items=items,
            input_ids=input_ids.tolist(),
            audio_token_id=self.audio_token_id,
            mrope_positions=mrope_positions,
            mrope_position_delta=torch.tensor([0], dtype=torch.long),
        )

    def _get_audio_offsets(
        self, input_ids: torch.Tensor, patch_lengths: Sequence[int]
    ) -> list[tuple[int, int]]:
        mask = input_ids == self.audio_token_id
        starts = (mask & ~torch.roll(mask, 1)).nonzero(as_tuple=True)[0]
        ends = (mask & ~torch.roll(mask, -1)).nonzero(as_tuple=True)[0]
        offsets = list(zip(starts.tolist(), ends.tolist()))
        if len(offsets) != len(patch_lengths) or any(
            end - start + 1 != length
            for (start, end), length in zip(offsets, patch_lengths)
        ):
            raise ValueError(
                "Bailing audio placeholder lengths do not match mel features"
            )
        return offsets


class BailingMoeV3AudioMultimodalProcessor(BaseMultimodalProcessor):
    """Local audio request adapter for the Ming/Bailing V3 runtime."""

    models = [BailingMoeV3AudioForConditionalGeneration]
    # Hash real features before IPC replaces them with transport proxies. The
    # base finalizer also copies them to CPU when ordinary transport is selected.
    precompute_hash_before_cpu_transfer = True
    gpu_resample = False

    @classmethod
    def _load_single_item(
        cls,
        data,
        modality,
        frame_count_limit=None,
        audio_sample_rate=None,
        discard_alpha_channel=True,
    ):
        if (
            modality != Modality.AUDIO
            or audio_sample_rate is not None
            or cls._is_preprocessed_input(data)
        ):
            return super()._load_single_item(
                data,
                modality,
                frame_count_limit,
                audio_sample_rate,
                discard_alpha_channel,
            )
        # Only the Bailing GPU resampler needs native-rate, multichannel input.
        # Keep the base loader's exception contract for malformed client media.
        try:
            return load_bailing_audio_native(data)
        except CLIENT_MEDIA_EXCEPTIONS as error:
            raise ValueError(f"Error while loading Bailing audio: {error}") from error
        except Exception as error:
            raise RuntimeError(f"Error while loading Bailing audio: {error}") from error

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        audio_config = getattr(hf_config, "audio_config", None)
        if audio_config is None:
            raise ValueError("Bailing Moe V3 requires an audio_config")
        # The base constructor fingerprints enabled preprocessing caches.
        # Prepare that state first; the frontend still allocates CUDA tensors
        # lazily when audio is processed, not during construction.
        options = (server_args.mm_process_config or {}).get("audio", {})
        gpu_mel = options.get("gpu_mel", True)
        self.gpu_resample = options.get("gpu_resample", True)
        if not isinstance(gpu_mel, bool) or not isinstance(self.gpu_resample, bool):
            raise ValueError("audio.gpu_mel and audio.gpu_resample must be booleans")
        # TODO: validate GPU resampling hidden states/tokens and WER/CER against
        # the release reference; keeping this option is not accuracy sign-off.
        self.frontend = BailingAudioFrontend(
            n_mels=audio_config.whisper_encoder_config["n_mels"],
            device=audio_frontend_device(server_args, gpu_enabled=gpu_mel),
            resample_device=(
                audio_frontend_device(server_args, gpu_enabled=True)
                if self.gpu_resample
                else None
            ),
        )
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.bailing_audio_processor = BailingAudioProcessor(
            self._tokenizer, audio_config
        )
        self.mm_tokens = MultimodalSpecialTokens(
            audio_token=_AUDIO_PLACEHOLDER,
            audio_token_id=self.bailing_audio_processor.audio_token_id,
        ).build(_processor)
        # Checkpoint-owned ASR contract. Flash emits ``language\ttranscript``;
        # Tiny emits a plain transcript and must not use Flash's prompt.
        self.asr_output_format = getattr(hf_config, "asr_output_format", "language_tab")
        if self.asr_output_format not in ("plain", "language_tab"):
            raise ValueError("Ming asr_output_format must be plain or language_tab")

    def preprocess_fingerprint_payload(self):
        payload = super().preprocess_fingerprint_payload()
        payload["bailing_audio_frontend"] = {
            "device": str(self.frontend.device),
            "n_mels": self.frontend.n_mels,
            "resample_device": str(self.frontend.resample_device),
            "preserve_sample_rate": self.gpu_resample,
            "resample_allow_tf32": (
                torch.backends.cudnn.allow_tf32 if self.gpu_resample else None
            ),
            "matmul_precision": (
                torch.get_float32_matmul_precision()
                if self.frontend.device.type == "cuda"
                else None
            ),
        }
        return payload

    def _build_transcription_prompt(self, input_text):
        if isinstance(input_text, list):
            input_text = self._tokenizer.decode(input_text) if input_text else ""
        if input_text and input_text.strip():
            return input_text
        # ASR requests use the checkpoint's direct prompt, not the generic chat
        # template. Tiny emits EOS under the Flash language-header prompt.
        if self.asr_output_format == "plain":
            return "Transcribe this speech.<AUDIO>"
        return (
            "Please recognize the language of this speech and transcribe it with "
            "appropriate punctuation marks, placing a `\\t` between the identified "
            "language and the transcribe.<AUDIO>"
        )

    async def process_mm_data_async(
        self,
        image_data=None,
        video_data=None,
        audio_data=None,
        input_text=None,
        **kwargs,
    ):
        if image_data or video_data:
            raise ValueError(
                "The local Bailing Moe V3 processor currently "
                "accepts audio requests only"
            )
        if not audio_data:
            return None

        input_text = self._build_transcription_prompt(input_text)
        if not isinstance(input_text, str):
            raise TypeError(
                "Bailing audio requests require string or tokenized input_text"
            )
        prompt = normalize_bailing_audio_placeholders(input_text, has_audios=True)
        # The generic loader may discard surplus inputs on its legacy path.
        # Validate the original request before loading can change its count.
        placeholder_count = prompt.count(_AUDIO_PLACEHOLDER)
        if placeholder_count != len(audio_data):
            raise ValueError(
                "Bailing audio input count does not match prompt placeholders: "
                f"{len(audio_data)} audio inputs, {placeholder_count} placeholders"
            )
        base_output = await self.load_mm_data(
            prompt=prompt,
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
            audio_sample_rate=None if self.gpu_resample else self.frontend.sample_rate,
        )
        if base_output is None:
            return None

        if self.gpu_resample:
            audios = [
                (torch.as_tensor(waveform).movedim(-1, 0).contiguous(), sample_rate)
                for waveform, sample_rate in base_output.audios
            ]
        else:
            audios = [
                (torch.as_tensor(audio), self.frontend.sample_rate)
                for audio in base_output.audios
            ]
        audio_features = self.bailing_audio_processor.prepare_audio_features(
            audios,
            self.frontend,
        )
        output = self.bailing_audio_processor.build_output(
            base_output.input_text, audio_features
        )
        output.mm_items = self._finalize_mm_items(output.mm_items, images=None)
        return output
