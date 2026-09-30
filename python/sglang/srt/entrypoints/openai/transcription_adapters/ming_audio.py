# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Ming plain-text transcription adapter for the OpenAI ASR API."""

from sglang.srt.entrypoints.openai.protocol import TranscriptionVerboseResponse
from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    TranscriptionAdapter,
    register_transcription_adapter,
)


@register_transcription_adapter("BailingMoeV3Audio")
class MingAudioAdapter(TranscriptionAdapter):
    """Preserve raw transcripts without language-header or timestamp parsing."""

    def build_sampling_params(self, request):
        if getattr(request, "language", None) is not None:
            raise ValueError(
                "Ming ASR currently supports automatic language detection only; omit language"
            )
        if getattr(request, "timestamp_granularities", None):
            raise ValueError("Ming ASR does not produce word or segment timestamps")
        return {
            "temperature": request.temperature,
            "max_new_tokens": max(448, int((request.audio_duration_s or 0) * 15)),
        }

    def build_verbose_response(self, request, text, ret, tokenizer, usage):
        return TranscriptionVerboseResponse(
            language=None,
            duration=round(request.audio_duration_s or 0, 2),
            text=text,
            segments=[],
            usage=usage,
        )
