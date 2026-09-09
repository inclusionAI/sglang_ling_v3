from typing import Optional

import torch

from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
    is_unified_kv_triton,
)
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    BuildCommitInjectLayout,
    build_unified_commit_inject_layout,
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.speculative.dspark_components.dspark_planner import VerifyWindow
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout


class TargetHiddenKvInjector:
    def __init__(
        self,
        *,
        draft_model,
        draft_model_runner,
        model_runner,
        device,
        verify_num_draft_tokens: int,
        block_pos_offsets: torch.Tensor,
    ) -> None:
        self.draft_model = draft_model
        self.draft_model_runner = draft_model_runner
        self.model_runner = model_runner
        self.device = device
        self.verify_num_draft_tokens = verify_num_draft_tokens
        self._block_pos_offsets = block_pos_offsets

    def inject_target_hidden(
        self,
        *,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
        state_slot: Optional[torch.Tensor] = None,
        final_pos: Optional[torch.Tensor] = None,
        target_hidden_is_projected: bool = False,
    ) -> None:
        if target_hidden is None or target_hidden.numel() == 0:
            return
        device = self.model_runner.device
        cache_loc = cache_loc.to(device=device, dtype=torch.int64, non_blocking=True)
        positions = positions.to(device=device, dtype=torch.int64, non_blocking=True)
        target_hidden = target_hidden.to(device=device, non_blocking=True)
        n_real = positions.shape[0]
        if target_hidden.shape[0] > n_real:
            target_hidden = target_hidden[:n_real]
        if cache_loc_2d is not None:
            cache_loc_2d = cache_loc_2d.to(
                device=device, dtype=torch.int64, non_blocking=True
            )
        if commit_lens is not None:
            commit_lens = commit_lens.to(
                device=device, dtype=torch.int32, non_blocking=True
            )
        if state_slot is not None:
            state_slot = state_slot.to(
                device=device, dtype=torch.int64, non_blocking=True
            )
        if final_pos is not None:
            final_pos = final_pos.to(
                device=device, dtype=torch.int64, non_blocking=True
            )

        pool = self.draft_model_runner.token_to_kv_pool
        if hasattr(pool, "set_swa_key_buffer_radix_fused_norm_rope"):
            if target_hidden_is_projected:
                raise RuntimeError(
                    "Pre-gather target-hidden projection is not supported by the "
                    "DSpark MLA KV injection path."
                )
            self._inject_mla(
                pool=pool,
                target_hidden=target_hidden,
                cache_loc=cache_loc,
                positions=positions,
                cache_loc_2d=cache_loc_2d,
                commit_lens=commit_lens,
                state_slot=state_slot,
                final_pos=final_pos,
            )
            return

        with torch.inference_mode():
            self.draft_model.write_target_hidden_kv(
                target_hidden=target_hidden,
                pool=pool,
                positions=positions,
                cache_loc=cache_loc,
                cache_loc_2d=cache_loc_2d,
                commit_lens=commit_lens,
                target_hidden_is_projected=target_hidden_is_projected,
            )

    def inject_target_hidden_static(
        self,
        *,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> bool:
        """Commit a uniform verify window without host-visible compaction.

        Returns False when the fixed-row protocol is disabled or the draft pool
        does not honor the reserved skip slot; the caller then compacts eagerly.
        """
        pool = self.draft_model_runner.token_to_kv_pool
        if (
            not envs.SGLANG_DSPARK_ENABLE_STATIC_COMMIT.get()
            or is_unified_kv_triton()
            or hasattr(pool, "set_swa_key_buffer_radix_fused_norm_rope")
            or not hasattr(self.draft_model, "write_target_hidden_kv_static")
            or not hasattr(pool, "set_mla_kv_buffer")
        ):
            return False

        device = self.model_runner.device
        target_hidden = target_hidden.to(device=device, non_blocking=True)
        cache_loc = cache_loc.to(device=device, dtype=torch.int64, non_blocking=True)
        positions = positions.to(device=device, dtype=torch.int64, non_blocking=True)
        commit_lens = commit_lens.to(
            device=device, dtype=torch.int32, non_blocking=True
        )
        verify_len = self.verify_num_draft_tokens
        if target_hidden.shape[0] != commit_lens.numel() * verify_len:
            raise ValueError(
                "DSpark static target-hidden commit shape mismatch: "
                f"hidden={target_hidden.shape[0]}, "
                f"commit_lens={commit_lens.numel()}, verify_len={verify_len}."
            )
        with torch.inference_mode():
            self.draft_model.write_target_hidden_kv_static(
                target_hidden=target_hidden,
                pool=pool,
                positions=positions,
                cache_loc=cache_loc,
                commit_lens=commit_lens,
                verify_len=verify_len,
            )
        return True

    def _inject_mla(
        self,
        *,
        pool,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor],
        commit_lens: Optional[torch.Tensor],
        state_slot: Optional[torch.Tensor] = None,
        final_pos: Optional[torch.Tensor] = None,
    ) -> None:
        if is_unified_kv_triton():
            swa_loc = self._unified_inject_loc(
                pool=pool,
                positions=positions,
                cache_loc_2d=cache_loc_2d,
                commit_lens=commit_lens,
                state_slot=state_slot,
                final_pos=final_pos,
            )
        elif (
            cache_loc.is_cuda
            and cache_loc.is_contiguous()
            and commit_lens is not None
            and cache_loc_2d is not None
            and commit_lens.is_contiguous()
            and cache_loc.numel() == cache_loc_2d.numel()
        ):
            from sglang.kernels.ops.speculative.dspark.commit_swa import (
                committed_swa_locations,
            )

            swa_loc = committed_swa_locations(
                cache_loc,
                pool.full_to_swa_index_mapping,
                commit_lens,
                cache_loc_2d.shape[1],
            )
        else:
            swa_loc = pool.translate_loc_from_full_to_swa(cache_loc).to(torch.int32)
            if commit_lens is not None and cache_loc_2d is not None:
                bs, verify_len = cache_loc_2d.shape
                col = torch.arange(verify_len, device=cache_loc.device).view(1, -1)
                committed_mask = (col < commit_lens.to(torch.long).view(-1, 1)).reshape(
                    -1
                )
                swa_loc = torch.where(
                    committed_mask, swa_loc, torch.full_like(swa_loc, -1)
                )

        with torch.inference_mode():
            self.draft_model.write_target_hidden_kv(
                main_hidden=target_hidden,
                swa_loc=swa_loc,
                positions=positions,
                pool=pool,
            )

    def _unified_inject_loc(
        self,
        *,
        pool,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor],
        commit_lens: Optional[torch.Tensor],
        state_slot: Optional[torch.Tensor],
        final_pos: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Ring row for target-hidden injection under unified_kv.

        loc = state_slot * ring + pos % ring, with two skip (-1) rules:
          * SWA window: only the last ``win`` tokens per req land in the ring;
            older tokens share a ring slot (pos % ring) and would race, so drop
            them (needed for long prefill chunks).
          * commit gate: uncommitted verify tokens (col >= commit_len) are dropped.
        """
        if state_slot is None:
            raise RuntimeError(
                "unified_kv target-hidden injection requires state_slot "
                "(per-token draft req_pool_indices)."
            )
        ring = pool.unified_swa_ring_size
        win = pool.unified_swa_window
        pos = positions.to(torch.int64)
        loc = state_slot.to(torch.int64) * ring + pos % ring
        if final_pos is not None:
            keep = pos > (final_pos.to(torch.int64) - win)
            loc = torch.where(keep, loc, torch.full_like(loc, -1))
        if commit_lens is not None and cache_loc_2d is not None:
            bs, verify_len = cache_loc_2d.shape
            col = torch.arange(verify_len, device=positions.device).view(1, -1)
            committed = (col < commit_lens.to(torch.long).view(-1, 1)).reshape(-1)
            loc = torch.where(committed, loc, torch.full_like(loc, -1))
        return loc.to(torch.int32)

    def inject_ragged(
        self,
        *,
        batch: ScheduleBatch,
        layout: RaggedVerifyLayout,
        hidden_strided: torch.Tensor,
        verify_window: VerifyWindow,
        commit_lens: torch.Tensor,
        bs: int,
    ) -> None:
        stride = self.verify_num_draft_tokens
        prefix_lens = batch.seq_lens
        hidden = hidden_strided.view(bs, stride, -1)

        pool = self.draft_model_runner.token_to_kv_pool
        if hasattr(pool, "set_swa_key_buffer_radix_fused_norm_rope"):
            if hidden_strided.numel() == 0:
                return
            if is_unified_kv_triton():
                inject_layout = build_unified_commit_inject_layout(
                    req_pool_indices=batch.req_pool_indices,
                    prefix_lens=prefix_lens,
                    block_pos_offsets=self._block_pos_offsets[:stride],
                    commit_lens=commit_lens,
                    stride=stride,
                    ring_stride=pool.unified_swa_ring_size,
                )
            else:
                inject_layout = BuildCommitInjectLayout.execute(
                    req_pool_indices=batch.req_pool_indices,
                    req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                    prefix_lens=prefix_lens,
                    block_pos_offsets=self._block_pos_offsets[:stride],
                    full_to_swa_mapping=pool.full_to_swa_index_mapping,
                    commit_lens=commit_lens,
                    stride=stride,
                )
            with torch.inference_mode():
                self.draft_model.write_target_hidden_kv(
                    main_hidden=hidden.reshape(-1, hidden.shape[-1]),
                    swa_loc=inject_layout.swa_loc,
                    positions=inject_layout.positions,
                    pool=pool,
                )
            return

        # Compact verify already scattered its rows back to one stride per
        # request, so the fixed-row protocol applies unchanged: rows at or past
        # commit_lens (rejected drafts and never-verified padding) land on the
        # reserved skip slot instead of being gathered through a host-visible
        # boolean mask.
        target_hidden = hidden.reshape(-1, hidden.shape[-1])
        positions = verify_window.positions_2d.reshape(-1)
        if self.inject_target_hidden_static(
            target_hidden=target_hidden,
            cache_loc=verify_window.verify_cache_loc,
            positions=positions,
            commit_lens=commit_lens,
        ):
            return
        self.inject_target_hidden(
            target_hidden=target_hidden,
            cache_loc=verify_window.verify_cache_loc,
            cache_loc_2d=verify_window.verify_cache_loc_2d,
            positions=positions,
            commit_lens=commit_lens,
        )
