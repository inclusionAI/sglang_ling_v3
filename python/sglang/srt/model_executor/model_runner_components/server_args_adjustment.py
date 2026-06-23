from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.utils import log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def model_specific_adjustment(
    *, server_args: ServerArgs, model_config: ModelConfig
) -> None:
    from sglang.srt.model_executor.model_runner import (
        CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS,
    )

    # HRM-Text needs bidirectional prompt attention (prefill), which only the
    # Triton backend honors and only with cuda graph / chunked prefill off
    # (TritonAttnBackend.allow_bidirectional_attention_in_extend). Radix cache
    # is also unsafe: the recurrent forward writes direction-dependent KV
    # across many slots.
    hf_config = model_config.hf_config
    is_hrm_text = getattr(
        hf_config, "model_type", None
    ) == "hrm_text" or "HrmTextForCausalLM" in getattr(hf_config, "architectures", [])
    # prefix_lm defaults to True upstream; defaulting False would skip the
    # bidirectional-attention forcing and silently produce junk output.
    is_prefix_lm_recurrent = is_hrm_text and getattr(hf_config, "prefix_lm", True)
    if is_prefix_lm_recurrent:
        if server_args.attention_backend not in (None, "triton"):
            logger.warning(
                f"Overriding --attention-backend "
                f"{server_args.attention_backend!r} -> 'triton': only the "
                "Triton backend supports HRM-Text's bidirectional prefix "
                "attention."
            )
        server_args.attention_backend = "triton"
        server_args.chunked_prefill_size = -1
        server_args.disable_radix_cache = True
        server_args.disable_cuda_graph = True
        logger.warning(
            "HRM-Text (prefix_lm) detected: forcing --attention-backend "
            "triton, --chunked-prefill-size -1, --disable-radix-cache, and "
            "--disable-cuda-graph for correctness of the bidirectional "
            "prompt attention."
        )

    if model_config.is_multimodal:
        if not model_config.is_multimodal_chunked_prefill_supported:
            server_args.chunked_prefill_size = -1
            logger.info(
                f"Automatically turn off --chunked-prefill-size as it is not supported for "
                f"{model_config.hf_config.model_type}"
            )

    use_mla_backend = model_config.attention_arch == AttentionArch.MLA
    if (
        not use_mla_backend
        or server_args.attention_backend
        not in CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS
    ):
        server_args.disable_chunked_prefix_cache = True

    if not server_args.disable_chunked_prefix_cache:
        log_info_on_rank0(logger, "Chunked prefix cache is turned on.")

    # The imperative adjustments above may overwrite fields the resolution passes
    # already declared (HRM-Text forces attention_backend); redeclare the
    # adjusted values so publish parity holds.
    from sglang.srt.arg_groups.overrides import refresh_declared_fields

    refresh_declared_fields(server_args, ("attention_backend",))
