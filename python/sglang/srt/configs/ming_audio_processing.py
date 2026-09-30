# Copyright 2026 Ant Group. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Tokenizer/chat-template adapter; audio preprocessing belongs to SGLang's pipeline.

The standalone HF model owns its full Processor and FeatureExtractor. This
adapter deliberately does not import checkpoint processor code or vision code.
"""

from transformers import AutoTokenizer, ProcessorMixin


class SGLangMingAudioProcessor(ProcessorMixin):
    """Expose the trained tokenizer/template to SGLang's request handling."""

    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(self, tokenizer, chat_template=None):
        template = tokenizer.chat_template if chat_template is None else chat_template
        super().__init__(tokenizer, chat_template=template)
        vocabulary = tokenizer.get_vocab()
        forms = (
            ("<audio>", "<audioPatch>", "</audio>"),
            ("<|audio_start|>", "<|audio_pad|>", "<|audio_end|>"),
        )
        if not any(all(token in vocabulary for token in form) for form in forms):
            raise ValueError("tokenizer lacks Ming audio start/patch/end tokens")

    @classmethod
    def from_pretrained(cls, model_path, *args, **kwargs):
        template = kwargs.pop("chat_template", None)
        tokenizer = AutoTokenizer.from_pretrained(model_path, *args, **kwargs)
        return cls(tokenizer, chat_template=template)


def is_ming_audio_checkpoint(config):
    """Explicit audio-only metadata, or an audio tower without a vision tower."""
    declared = config.get("supported_modalities")
    return config.get("audio_config") is not None and (
        set(declared) == {"audio"}
        if declared is not None
        else config.get("vision_config") is None
    )


class SGLangBailingProcessorFactory:
    """Preserve VL processing while loading legacy audio with the local frontend."""

    @classmethod
    def from_pretrained(cls, model_path, *args, **kwargs):
        from transformers import AutoProcessor, PretrainedConfig

        config, _ = PretrainedConfig.get_config_dict(model_path, **kwargs)
        processor = (
            SGLangMingAudioProcessor
            if is_ming_audio_checkpoint(config)
            else AutoProcessor
        )
        return processor.from_pretrained(model_path, *args, **kwargs)
