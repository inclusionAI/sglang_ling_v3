from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.configs.model_config import is_deepseek_v4
from sglang.srt.distributed.parallel_state import get_world_group
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.platforms import current_platform
from sglang.srt.utils.common import (
    get_available_gpu_memory,
    get_device_memory_capacity,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.model_runner_components.pool_configurator import (
        MemoryPoolConfig,
    )

logger = logging.getLogger(__name__)


def _should_enable_lazy_compaction() -> bool:
    """Lazy compaction default — ON unless
    `SGLANG_DISABLE_LAZY_COMPACTION=1` (escape hatch for A/B / rollback).
    Centralized here so both unified-memory-pool factory call sites stay in sync.
    """
    return not envs.SGLANG_DISABLE_LAZY_COMPACTION.get()


class ModelRunnerKVCacheMixin:
    def init_memory_pool(self: ModelRunner, pre_model_load_memory: int):
        """Temporary 1-line delegate — dropped in kvc-drop-mixin-inheritance."""
        # Unified-pool fast path: build req_to_token + token_to_kv pool + allocator
        # from one byte buffer, then return. Gated to the target worker
        # (req_to_token_pool is None); supports hybrid Mamba and hybrid SWA (not
        # DSV4). Stays on the mixin because the unified pool factories need
        # ModelRunner state (forward_stream) the configurator does not carry.
        if (
            self.server_args.enable_unified_memory
            and self.server_args.disaggregation_mode == "null"
            and self.req_to_token_pool is None
        ):
            self.memory_pool_config = (
                self.kv_cache_configurator._resolve_memory_pool_config(
                    pre_model_load_memory
                )
            )
            config = self.memory_pool_config
            self.max_total_num_tokens = config.max_total_num_tokens
            self.max_running_requests = config.max_running_requests
            if self.is_hybrid_swa:
                self.full_max_total_num_tokens = config.full_max_total_num_tokens
                self.swa_max_total_num_tokens = config.swa_max_total_num_tokens
            if mambaish_config(self.model_config) is not None:
                self._init_unified_mamba_pools(self.max_running_requests)
            elif self.is_hybrid_swa and not is_deepseek_v4(
                self.model_config.hf_config
            ):
                self._init_unified_swa_pools(self.max_running_requests)
            else:
                # Fail loud, not silently fall through to the normal pools (which
                # would leave the flag a no-op). The feature replaces the HYBRID
                # pools only.
                raise ValueError(
                    "--enable-unified-memory only supports hybrid Mamba and "
                    "hybrid sliding-window-attention models (DeepSeek-V4 excluded); "
                    f"the current model ({self.model_config.hf_config.architectures}) "
                    "is neither, so the unified memory pool cannot be built. Drop "
                    "--enable-unified-memory for this model."
                )
            logger.info(
                f"Memory pool end. "
                f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
            )
            return

        result = self.kv_cache_configurator.configure(
            pre_model_load_memory=pre_model_load_memory
        )
        self.max_total_num_tokens = result.max_total_num_tokens
        self.max_running_requests = result.max_running_requests
        self.req_to_token_pool = result.req_to_token_pool
        self.token_to_kv_pool = result.token_to_kv_pool
        self.token_to_kv_pool_allocator = result.token_to_kv_pool_allocator
        self.memory_pool_config = result.memory_pool_config
        if self.is_hybrid_swa:
            self.full_max_total_num_tokens = result.full_max_total_num_tokens
            self.swa_max_total_num_tokens = result.swa_max_total_num_tokens

    def _config_from_budget(
        self: ModelRunner, budget_bytes: int, *, cap_tokens: Optional[int] = None
    ) -> MemoryPoolConfig:
        return self.kv_cache_configurator._config_from_budget(
            budget_bytes, cap_tokens=cap_tokens
        )

    def _resolve_max_num_reqs(self: ModelRunner, token_capacity: int) -> int:
        return self.kv_cache_configurator._resolve_max_num_reqs(token_capacity)

    @property
    def post_capture_kv_active(self: ModelRunner) -> bool:
        return (
            self.server_args.post_capture_kv_sizing_planned()
            and current_platform.is_cuda()
            and not self.is_draft_worker
        )

    def post_capture_resize_kv_pool(self: ModelRunner) -> None:
        """Resize the KV pool after capture."""
        pool = self.token_to_kv_pool
        torch.cuda.synchronize()
        free_gb = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=get_world_group().world_size > 1,
            cpu_group=get_world_group().cpu_group,
        )
        headroom_gb = self.pre_model_load_memory * (1 - self.mem_fraction_static)
        decode_cuda_graph_config = self.server_args.cuda_graph_config.decode
        decode_max_bs = int(decode_cuda_graph_config.max_bs or 0)
        running_requests = int(self.max_running_requests or decode_max_bs or 1)
        eager_decode_gap = (
            self.server_args.disaggregation_mode != "prefill"
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
        if eager_decode_gap or mambaish_config(self.model_config) is not None:
            headroom_gb = max(
                headroom_gb,
                self.server_args.mamba_pre_capture_reserve_mb(
                    get_device_memory_capacity(self.device)
                )
                / 1024,
            )
        budget_bytes = (
            int(max(0.0, free_gb - headroom_gb) * (1 << 30))
            + pool.post_capture_backed_bytes
        )
        config = self._config_from_budget(
            budget_bytes, cap_tokens=self.max_total_num_tokens
        )
        pool.finalize_backing(config)
        self.token_to_kv_pool_allocator.resize(config)

        # Set the new pool size
        self.max_total_num_tokens = config.max_total_num_tokens
        if self.is_hybrid_swa:
            self.full_max_total_num_tokens = config.full_max_total_num_tokens
            self.swa_max_total_num_tokens = config.swa_max_total_num_tokens
        if self.memory_pool_config is not None:
            self.memory_pool_config.max_total_num_tokens = config.max_total_num_tokens
            self.memory_pool_config.full_max_total_num_tokens = (
                config.full_max_total_num_tokens
            )
            self.memory_pool_config.swa_max_total_num_tokens = (
                config.swa_max_total_num_tokens
            )
        if self.max_running_requests is not None:
            # Re-calculate max_running_requests for the now smaller pool
            capped_reqs = min(
                self.max_running_requests,
                self._resolve_max_num_reqs(config.max_total_num_tokens),
            )
            if capped_reqs < self.max_running_requests:
                logger.warning(
                    "Post-capture KV sizing: max_running_requests %d -> %d",
                    self.max_running_requests,
                    capped_reqs,
                )
                self.max_running_requests = capped_reqs
                if self.memory_pool_config is not None:
                    self.memory_pool_config.max_running_requests = capped_reqs
        logger.info(
            "Post-capture KV sizing: max_total_num_tokens=%d, free memory=%.2f GB",
            config.max_total_num_tokens,
            get_available_gpu_memory(self.device, self.gpu_id),
        )

    def _init_unified_mamba_pools(self: ModelRunner, max_num_reqs: int):
        """Build the shared-KV-pool stack for a hybrid-Mamba model:
        one byte buffer split between the full-attn MHA KV pool and the
        per-request Mamba state pool, with virtual slot ids above the
        allocator."""
        from sglang.srt.mem_cache.unified_memory_pool import init_unified_mamba_pools

        config = mambaish_config(self.model_config)
        assert config is not None
        assert (
            not self.use_mla_backend
        ), "unified memory pool does not support MLA-hybrid-Mamba yet"
        # The full sub-pool is page-aware (via `MultiEndedAllocator(page_size=...)`);
        # the mamba sub-pool stays page=1.
        assert self.page_size >= 1, f"page_size must be >= 1, got {self.page_size}"
        # Mirror the non-shared path's extra_max_context_len computation.
        extra_max_context_len = 4
        if self.server_args.speculative_num_draft_tokens is not None:
            extra_max_context_len += self.server_args.speculative_num_draft_tokens

        mamba_layer_ids = [
            i
            for i in config.mamba2_cache_params.layers
            if self.start_layer <= i < self.end_layer
        ]
        full_attention_layer_ids = [
            i
            for i in config.full_attention_layer_ids
            if self.start_layer <= i < self.end_layer
        ]

        bundle = init_unified_mamba_pools(
            device=self.device,
            kv_cache_dtype=self.kv_cache_dtype,
            head_num=self.model_config.get_num_kv_heads(get_attention_tp_size()),
            head_dim=self.model_config.head_dim,
            page_size=self.page_size,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            is_draft_worker=self.is_draft_worker,
            use_mla_backend=self.use_mla_backend,
            mamba_layer_ids=mamba_layer_ids,
            full_attention_layer_ids=full_attention_layer_ids,
            mamba2_cache_params=config.mamba2_cache_params,
            model_context_len=self.model_config.context_len,
            extra_max_context_len=extra_max_context_len,
            max_total_num_tokens=self.max_total_num_tokens,
            max_mamba_cache_size=self.server_args.max_mamba_cache_size,
            max_num_reqs=max_num_reqs,
            enable_memory_saver=self.server_args.enable_memory_saver,
            enable_mamba_extra_buffer=self.server_args.enable_mamba_extra_buffer(),
            speculative_num_draft_tokens=self.server_args.speculative_num_draft_tokens,
            disable_overlap_schedule=self.server_args.disable_overlap_schedule,
            need_sort=self.server_args.disaggregation_mode in ("decode", "prefill"),
            mamba_full_memory_ratio=self.server_args.mamba_full_memory_ratio,
            # Overlap mode: the allocator's `free` drops a wait_stream(forward_stream)
            # barrier so eager compaction serializes after the in-flight forward's
            # v2p/KV reads. Near-no-op in normal mode.
            forward_stream=self.forward_stream,
            # Lazy compaction: default ON, env-var escape hatch for rollback / A/B.
            lazy_compaction=_should_enable_lazy_compaction(),
        )
        self.req_to_token_pool = bundle.req_to_token_pool
        self.token_to_kv_pool = bundle.token_to_kv_pool
        self.token_to_kv_pool_allocator = bundle.token_to_kv_pool_allocator
        # Keep a reference so the shared byte buffer is not GC'd.
        self._unified_memory_pool = bundle.unified_memory_pool

    def _init_unified_swa_pools(self: ModelRunner, max_num_reqs: int):
        """Build the unified-pool stack for a hybrid-SWA model (Triton): one byte
        buffer split between the full-attention and SWA KV pools."""
        from sglang.srt.mem_cache.unified_memory_pool import init_unified_swa_pools

        assert self.is_hybrid_swa, "_init_unified_swa_pools called on a non-SWA model"
        # Both sub-pools are page-aware; the SWA composite runs alloc_extend_kernel
        # once in virtual space and binds the new pages on both sub-allocators.
        assert self.page_size >= 1, f"page_size must be >= 1, got {self.page_size}"
        assert (
            not self.use_mla_backend
        ), "unified memory pool does not support MLA-SWA hybrid yet"
        # Mirror the non-shared path's extra_max_context_len computation.
        extra_max_context_len = 4
        if self.server_args.speculative_num_draft_tokens is not None:
            extra_max_context_len += self.server_args.speculative_num_draft_tokens
        self.req_to_token_pool = ReqToTokenPool(
            size=max_num_reqs,
            max_context_len=self.model_config.context_len + extra_max_context_len,
            device=self.device,
            enable_memory_saver=self.server_args.enable_memory_saver,
        )

        head_num = self.model_config.get_num_kv_heads(get_attention_tp_size())
        head_dim = self.model_config.head_dim
        if self.is_hybrid_swa_compress:
            # Asymmetric head dims between full and SWA (NPU compress path):
            # pull SWA-specific dims from the hf text config.
            v_head_dim = self.model_config.hf_text_config.v_head_dim
            swa_head_num = max(
                1,
                self.model_config.hf_text_config.swa_num_key_value_heads
                // get_attention_tp_size(),
            )
            swa_head_dim = self.model_config.hf_text_config.swa_head_dim
            swa_v_head_dim = self.model_config.hf_text_config.swa_v_head_dim
        else:
            v_head_dim = head_dim
            swa_head_num = head_num
            swa_head_dim = head_dim
            swa_v_head_dim = head_dim

        # Filter layer ids to this worker's [start_layer, end_layer) range.
        swa_attention_layer_ids = [
            i
            for i in self.model_config.swa_attention_layer_ids
            if self.start_layer <= i < self.end_layer
        ]
        full_attention_layer_ids = [
            i
            for i in self.model_config.full_attention_layer_ids
            if self.start_layer <= i < self.end_layer
        ]

        bundle = init_unified_swa_pools(
            device=self.device,
            kv_cache_dtype=self.kv_cache_dtype,
            head_num=head_num,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            swa_head_num=swa_head_num,
            swa_head_dim=swa_head_dim,
            swa_v_head_dim=swa_v_head_dim,
            page_size=self.page_size,
            start_layer=self.start_layer,
            end_layer=self.end_layer,
            swa_attention_layer_ids=swa_attention_layer_ids,
            full_attention_layer_ids=full_attention_layer_ids,
            full_max_total_num_tokens=self.full_max_total_num_tokens,
            swa_max_total_num_tokens=self.swa_max_total_num_tokens,
            enable_memory_saver=self.server_args.enable_memory_saver,
            need_sort=self.server_args.disaggregation_mode in ("decode", "prefill"),
            # Overlap mode: same wait_stream(forward_stream) rationale as
            # `_init_unified_mamba_pools`.
            forward_stream=self.forward_stream,
            # Lazy compaction: default ON, with env var escape hatch for rollback / A/B.
            lazy_compaction=_should_enable_lazy_compaction(),
        )
        self.token_to_kv_pool = bundle.token_to_kv_pool
        self.token_to_kv_pool_allocator = bundle.token_to_kv_pool_allocator
        # Keep a reference so the shared byte buffer is not GC'd.
        self._unified_memory_pool = bundle.unified_memory_pool
