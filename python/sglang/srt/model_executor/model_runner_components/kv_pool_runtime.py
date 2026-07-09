from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.distributed import get_world_group
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.platforms import current_platform
from sglang.srt.utils.common import get_available_gpu_memory, get_device_memory_capacity

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _should_enable_lazy_compaction() -> bool:
    """Lazy compaction default — ON unless
    `SGLANG_DISABLE_LAZY_COMPACTION=1` (escape hatch for A/B / rollback).
    Centralized here so both unified-memory-pool factory call sites stay in sync.
    """
    return not envs.SGLANG_DISABLE_LAZY_COMPACTION.get()


def is_post_capture_kv_active(model_runner: ModelRunner) -> bool:
    return (
        model_runner.server_args.post_capture_kv_sizing_planned()
        and current_platform.is_cuda()
        and not model_runner.is_draft_worker
    )


def post_capture_resize_kv_pool(model_runner: ModelRunner) -> None:
    """Resize the KV pool after capture.

    Writes the resized pool sizes back onto ``model_runner`` (the documented leaf
    exception to the read-only-god-object rule): the post-capture resize depends on
    live runtime state (``forward_stream``-adjacent pools, measured free memory) that
    the configurator does not carry, so it stays a runner-facing free function.
    """
    mr = model_runner
    pool = mr.token_to_kv_pool
    torch.cuda.synchronize()
    free_gb = get_available_gpu_memory(
        mr.device,
        mr.gpu_id,
        distributed=get_world_group().world_size > 1,
        cpu_group=get_world_group().cpu_group,
    )
    headroom_gb = mr.pre_model_load_memory * (1 - mr.mem_fraction_static)
    decode_cuda_graph_config = mr.server_args.cuda_graph_config.decode
    decode_max_bs = int(decode_cuda_graph_config.max_bs or 0)
    running_requests = int(mr.max_running_requests or decode_max_bs or 1)
    eager_decode_gap = (
        mr.server_args.disaggregation_mode != "prefill"
        and decode_cuda_graph_config.backend != Backend.DISABLED
        and decode_max_bs < running_requests
    )
    if eager_decode_gap:
        logger.warning(
            "Post-capture KV sizing: decode CUDA graph max_bs=%d < "
            "max_running_requests=%d; reserving activation headroom",
            decode_max_bs,
            running_requests,
        )
    if eager_decode_gap or mambaish_config(mr.model_config) is not None:
        headroom_gb = max(
            headroom_gb,
            mr.server_args.mamba_pre_capture_reserve_mb(
                get_device_memory_capacity(mr.device)
            )
            / 1024,
        )
    budget_bytes = (
        int(max(0.0, free_gb - headroom_gb) * (1 << 30))
        + pool.post_capture_backed_bytes
    )
    config = mr.kv_cache_configurator._config_from_budget(
        budget_bytes, cap_tokens=mr.max_total_num_tokens
    )
    pool.finalize_backing(config)
    mr.token_to_kv_pool_allocator.resize(config)

    # Set the new pool size
    mr.max_total_num_tokens = config.max_total_num_tokens
    if mr.is_hybrid_swa:
        mr.full_max_total_num_tokens = config.full_max_total_num_tokens
        mr.swa_max_total_num_tokens = config.swa_max_total_num_tokens
    if mr.memory_pool_config is not None:
        mr.memory_pool_config.max_total_num_tokens = config.max_total_num_tokens
        mr.memory_pool_config.full_max_total_num_tokens = (
            config.full_max_total_num_tokens
        )
        mr.memory_pool_config.swa_max_total_num_tokens = config.swa_max_total_num_tokens
    if mr.max_running_requests is not None:
        # Re-calculate max_running_requests for the now smaller pool
        capped_reqs = min(
            mr.max_running_requests,
            mr.kv_cache_configurator._resolve_max_num_reqs(config.max_total_num_tokens),
        )
        if capped_reqs < mr.max_running_requests:
            logger.warning(
                "Post-capture KV sizing: max_running_requests %d -> %d",
                mr.max_running_requests,
                capped_reqs,
            )
            mr.max_running_requests = capped_reqs
            if mr.memory_pool_config is not None:
                mr.memory_pool_config.max_running_requests = capped_reqs
    logger.info(
        "Post-capture KV sizing: max_total_num_tokens=%d, free memory=%.2f GB",
        config.max_total_num_tokens,
        get_available_gpu_memory(mr.device, mr.gpu_id),
    )


def build_unified_mamba_pools(model_runner: ModelRunner, max_num_reqs: int) -> None:
    """Build the shared-KV-pool stack for a hybrid-Mamba model:
    one byte buffer split between the full-attn MHA KV pool and the
    per-request Mamba state pool, with virtual slot ids above the
    allocator. Writes the pool trio back onto ``model_runner``; needs its
    live ``forward_stream``, so it cannot live on the configurator."""
    from sglang.srt.mem_cache.unified_memory_pool import init_unified_mamba_pools

    mr = model_runner
    config = mambaish_config(mr.model_config)
    assert config is not None
    assert (
        not mr.use_mla_backend
    ), "unified memory pool does not support MLA-hybrid-Mamba yet"
    # The full sub-pool is page-aware (via `MultiEndedAllocator(page_size=...)`);
    # the mamba sub-pool stays page=1.
    assert mr.page_size >= 1, f"page_size must be >= 1, got {mr.page_size}"
    # Mirror the non-shared path's extra_max_context_len computation.
    extra_max_context_len = 4
    if mr.server_args.speculative_num_draft_tokens is not None:
        extra_max_context_len += mr.server_args.speculative_num_draft_tokens

    mamba_layer_ids = [
        i
        for i in config.mamba2_cache_params.layers
        if mr.layer_info.start_layer <= i < mr.layer_info.end_layer
    ]
    full_attention_layer_ids = [
        i
        for i in config.full_attention_layer_ids
        if mr.layer_info.start_layer <= i < mr.layer_info.end_layer
    ]

    bundle = init_unified_mamba_pools(
        device=mr.device,
        kv_cache_dtype=mr.kv_cache_dtype,
        head_num=mr.model_config.get_num_kv_heads(get_attention_tp_size()),
        head_dim=mr.model_config.head_dim,
        page_size=mr.page_size,
        start_layer=mr.layer_info.start_layer,
        end_layer=mr.layer_info.end_layer,
        is_draft_worker=mr.is_draft_worker,
        use_mla_backend=mr.use_mla_backend,
        mamba_layer_ids=mamba_layer_ids,
        full_attention_layer_ids=full_attention_layer_ids,
        mamba2_cache_params=config.mamba2_cache_params,
        model_context_len=mr.model_config.context_len,
        extra_max_context_len=extra_max_context_len,
        max_total_num_tokens=mr.max_total_num_tokens,
        max_mamba_cache_size=mr.server_args.max_mamba_cache_size,
        max_num_reqs=max_num_reqs,
        enable_memory_saver=mr.server_args.enable_memory_saver,
        enable_mamba_extra_buffer=mr.server_args.enable_mamba_extra_buffer(),
        speculative_num_draft_tokens=mr.server_args.speculative_num_draft_tokens,
        disable_overlap_schedule=mr.server_args.disable_overlap_schedule,
        need_sort=mr.server_args.disaggregation_mode in ("decode", "prefill"),
        mamba_full_memory_ratio=mr.server_args.mamba_full_memory_ratio,
        # Overlap mode: the allocator's `free` drops a wait_stream(forward_stream)
        # barrier so eager compaction serializes after the in-flight forward's
        # v2p/KV reads. Near-no-op in normal mode.
        forward_stream=mr.forward_stream,
        # Lazy compaction: default ON, env-var escape hatch for rollback / A/B.
        lazy_compaction=_should_enable_lazy_compaction(),
    )
    mr.req_to_token_pool = bundle.req_to_token_pool
    mr.token_to_kv_pool = bundle.token_to_kv_pool
    mr.token_to_kv_pool_allocator = bundle.token_to_kv_pool_allocator
    # Keep a reference so the shared byte buffer is not GC'd.
    mr._unified_memory_pool = bundle.unified_memory_pool


def build_unified_swa_pools(model_runner: ModelRunner, max_num_reqs: int) -> None:
    """Build the unified-pool stack for a hybrid-SWA model (Triton): one byte
    buffer split between the full-attention and SWA KV pools."""
    from sglang.srt.mem_cache.unified_memory_pool import init_unified_swa_pools

    mr = model_runner
    assert mr.is_hybrid_swa, "build_unified_swa_pools called on a non-SWA model"
    # Both sub-pools are page-aware; the SWA composite runs alloc_extend_kernel
    # once in virtual space and binds the new pages on both sub-allocators.
    assert mr.page_size >= 1, f"page_size must be >= 1, got {mr.page_size}"
    assert (
        not mr.use_mla_backend
    ), "unified memory pool does not support MLA-SWA hybrid yet"
    # Mirror the non-shared path's extra_max_context_len computation.
    extra_max_context_len = 4
    if mr.server_args.speculative_num_draft_tokens is not None:
        extra_max_context_len += mr.server_args.speculative_num_draft_tokens
    mr.req_to_token_pool = ReqToTokenPool(
        size=max_num_reqs,
        max_context_len=mr.model_config.context_len + extra_max_context_len,
        device=mr.device,
        enable_memory_saver=mr.server_args.enable_memory_saver,
    )

    head_num = mr.model_config.get_num_kv_heads(get_attention_tp_size())
    head_dim = mr.model_config.head_dim
    if mr.is_hybrid_swa_compress:
        # Asymmetric head dims between full and SWA (NPU compress path):
        # pull SWA-specific dims from the hf text config.
        v_head_dim = mr.model_config.hf_text_config.v_head_dim
        swa_head_num = max(
            1,
            mr.model_config.hf_text_config.swa_num_key_value_heads
            // get_attention_tp_size(),
        )
        swa_head_dim = mr.model_config.hf_text_config.swa_head_dim
        swa_v_head_dim = mr.model_config.hf_text_config.swa_v_head_dim
    else:
        v_head_dim = head_dim
        swa_head_num = head_num
        swa_head_dim = head_dim
        swa_v_head_dim = head_dim

    # Filter layer ids to this worker's [start_layer, end_layer) range.
    swa_attention_layer_ids = [
        i
        for i in mr.model_config.swa_attention_layer_ids
        if mr.layer_info.start_layer <= i < mr.layer_info.end_layer
    ]
    full_attention_layer_ids = [
        i
        for i in mr.model_config.full_attention_layer_ids
        if mr.layer_info.start_layer <= i < mr.layer_info.end_layer
    ]

    bundle = init_unified_swa_pools(
        device=mr.device,
        kv_cache_dtype=mr.kv_cache_dtype,
        head_num=head_num,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        swa_head_num=swa_head_num,
        swa_head_dim=swa_head_dim,
        swa_v_head_dim=swa_v_head_dim,
        page_size=mr.page_size,
        start_layer=mr.layer_info.start_layer,
        end_layer=mr.layer_info.end_layer,
        swa_attention_layer_ids=swa_attention_layer_ids,
        full_attention_layer_ids=full_attention_layer_ids,
        full_max_total_num_tokens=mr.full_max_total_num_tokens,
        swa_max_total_num_tokens=mr.swa_max_total_num_tokens,
        enable_memory_saver=mr.server_args.enable_memory_saver,
        need_sort=mr.server_args.disaggregation_mode in ("decode", "prefill"),
        # Overlap mode: same wait_stream(forward_stream) rationale as
        # `build_unified_mamba_pools`.
        forward_stream=mr.forward_stream,
        # Lazy compaction: default ON, with env var escape hatch for rollback / A/B.
        lazy_compaction=_should_enable_lazy_compaction(),
    )
    mr.token_to_kv_pool = bundle.token_to_kv_pool
    mr.token_to_kv_pool_allocator = bundle.token_to_kv_pool_allocator
    # Keep a reference so the shared byte buffer is not GC'd.
    mr._unified_memory_pool = bundle.unified_memory_pool
