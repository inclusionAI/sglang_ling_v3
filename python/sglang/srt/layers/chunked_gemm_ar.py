"""Chunked row-parallel GEMM + all-reduce overlap (``SGLANG_OPT_CHUNKED_GEMM_AR``).

On a large-M prefill, a row-parallel linear (attention o_proj, dense-MLP
down_proj) runs quant -> GEMM -> all-reduce strictly serialized on the compute
stream, and the NCCL ring kernel only occupies a small fraction of the SMs
(24/78 on H20 TP4) -- the rest of the GPU idles for the whole AR (~1 ms x 76
layers on a 25k-token Ling-3.0-plus TP4 prefill).

This module splits the token dim into a pipeline of row chunks:

  compute stream: [quant+GEMM c0 (all SMs)] [quant+GEMM c1 (capped SMs)] ...
  c10d nccl stream:                [all_reduce c0]        [all_reduce c1] ...

Chunk i's all-reduce is issued as an async c10d op (it runs on
ProcessGroupNCCL's internal stream, ordered after the chunk's GEMM by c10d's
own stream sync) and overlaps chunk i+1's GEMM. Two details make the overlap
real instead of nominal:

- deep_gemm launches a persistent grid of exactly ``num_sms`` blocks whose
  smem footprint allows only one block per SM, so a concurrent NCCL kernel
  would otherwise just serialize behind it. Overlapped chunks therefore run
  under ``configure_deep_gemm_num_sms(total_sms - reserve_sms)`` to leave the
  NCCL blocks room to co-schedule. The first chunk has no AR to hide and runs
  at full SM count.
- Chunk sizes decay geometrically (``ratio``) so that each chunk's AR time
  matches the next, smaller chunk's SM-capped GEMM time, instead of paying
  the SM reservation for longer than the AR needs.

Numerics: the GEMM stage is bit-identical to the unchunked call (per-token-
group quant depends only on the row and the deep_gemm K-loop accumulates in a
fixed order per row). The all-reduce stage sums the same per-element operands,
but NCCL's cross-rank summation order depends on where an element falls in the
buffer, so chunking can reassociate that sum -- the same class of
last-bit difference as changing NCCL's algorithm or chunk size, deterministic
run to run and identical across ranks. Every TP rank derives the same chunk
plan from (M, env), so the collectives line up.

Two entry points, both fall through (return ``None``) on any ineligibility:

- ``try_forward_chunked_gemm_ar``: for ``RowParallelLinear.forward`` when the
  layer reduces its own output (``reduce_results=True``).
- ``try_forward_chunked_gemm_ar_inline``: for a layer built with
  ``reduce_results=False`` whose all-reduce is normally issued later by the
  model (the attention stage boundary). The caller MUST then tell that
  boundary the tensor is already reduced.

Scope guard shared by both: 2-D CUDA input with at least
``SGLANG_OPT_CHUNKED_GEMM_AR_MIN_TOKENS`` rows, plain (non-symmetric,
non-quantized, non-DP-attention) all-reduce, a quant method that exposes
``apply_into``, and outside torch.compile / CUDA-graph capture.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import TYPE_CHECKING, List, Optional, Tuple

import msgspec
import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.layers.linear import RowParallelLinear
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class _Knobs(msgspec.Struct, frozen=True):
    min_tokens: int
    num_chunks: int
    ratio: float
    reserve_sms: int


_ENABLED: Optional[bool] = None
_KNOBS: Optional[_Knobs] = None
_TOTAL_SMS: Optional[int] = None
_LOGGED = False


def chunked_gemm_ar_enabled() -> bool:
    """Cached master switch; the first gate on every hot path that may chunk."""
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = envs.SGLANG_OPT_CHUNKED_GEMM_AR.get()
    return _ENABLED


def _knobs() -> _Knobs:
    global _KNOBS
    if _KNOBS is None:
        _KNOBS = _Knobs(
            min_tokens=envs.SGLANG_OPT_CHUNKED_GEMM_AR_MIN_TOKENS.get(),
            num_chunks=envs.SGLANG_OPT_CHUNKED_GEMM_AR_NUM_CHUNKS.get(),
            ratio=envs.SGLANG_OPT_CHUNKED_GEMM_AR_RATIO.get(),
            reserve_sms=envs.SGLANG_OPT_CHUNKED_GEMM_AR_RESERVE_SMS.get(),
        )
    return _KNOBS


def compute_chunk_plan(
    m: int,
    num_chunks: int,
    ratio: float,
    align: int = 256,
    min_rows: int = 512,
) -> Optional[List[Tuple[int, int]]]:
    """Split ``m`` rows into up to ``num_chunks`` geometric chunks.

    Chunk i+1 holds ~``ratio`` times the rows of chunk i (the first chunk is
    the largest: it is the only GEMM that cannot hide an AR, so it should run
    as long as possible at full SM count). Boundaries are aligned to
    ``align`` rows for GEMM tile friendliness; chunks smaller than
    ``min_rows`` are merged into their neighbor. Returns ``None`` when fewer
    than 2 chunks survive -- callers fall back to the unchunked path.

    Pure function of its arguments: all TP ranks must (and do) compute the
    same plan for the collectives to line up.
    """
    if num_chunks < 2 or m < 2 * min_rows:
        return None
    ratio = min(max(ratio, 0.05), 1.0)
    weights = [ratio**i for i in range(num_chunks)]
    total = sum(weights)
    bounds = [0]
    acc = 0.0
    for w in weights[:-1]:
        acc += m * w / total
        b = int(round(acc / align)) * align
        b = max(b, bounds[-1] + min_rows)
        if b > m - min_rows:
            break
        bounds.append(b)
    bounds.append(m)
    if len(bounds) < 3:
        return None
    return list(zip(bounds[:-1], bounds[1:]))


def _get_total_sms(device: torch.device) -> int:
    global _TOTAL_SMS
    if _TOTAL_SMS is None:
        _TOTAL_SMS = torch.cuda.get_device_properties(device).multi_processor_count
    return _TOTAL_SMS


def try_forward_chunked_gemm_ar(
    layer: RowParallelLinear,
    input_parallel: torch.Tensor,
    bias: Optional[torch.Tensor],
    forward_batch: Optional[ForwardBatch],
) -> Optional[torch.Tensor]:
    """``RowParallelLinear.forward`` entry: GEMM + the layer's own TP all-reduce
    as an overlapped chunk pipeline, or ``None`` to fall through.

    The caller has already checked: env switch, ``reduce_results``,
    ``tp_size > 1``, no dp-attention reduce, no ``skip_all_reduce`` /
    ``output_tensor`` override. The reduce group is the one
    ``tensor_model_parallel_all_reduce`` would use.
    """
    from sglang.srt.distributed import get_tp_group

    return _try_forward_pipeline(
        layer=layer,
        x=input_parallel,
        bias=bias,
        forward_batch=forward_batch,
        group=get_tp_group(),
    )


def try_forward_chunked_gemm_ar_inline(
    layer: RowParallelLinear,
    input_parallel: torch.Tensor,
    forward_batch: Optional[ForwardBatch],
    reduce_on_attn_tp: bool,
) -> Optional[torch.Tensor]:
    """Entry for a layer built with ``reduce_results=False`` whose all-reduce
    is normally issued downstream by the model (e.g. Ling-V3's attention
    o_proj, reduced by the following stage boundary).

    Runs GEMM + all-reduce as the chunk pipeline over the group the downstream
    site would have reduced on (``attn_tp_group`` when ``reduce_on_attn_tp``,
    else the TP group) and returns the fully reduced ``[m, output_size]``
    tensor. The caller must then make the downstream site skip its all-reduce
    for this tensor. Returns ``None`` -- and touches nothing -- whenever the
    layer's regular forward should run instead (the downstream all-reduce then
    happens as usual).
    """
    if not chunked_gemm_ar_enabled():
        return None
    # A layer that reduces itself is served from RowParallelLinear.forward.
    if layer.reduce_results or layer.tp_size <= 1 or not layer.input_is_parallel:
        return None
    if layer.skip_bias_add and layer.bias is not None:
        return None
    if layer.use_dp_attention_reduce:
        return None

    from sglang.srt.distributed import get_attn_tp_group, get_tp_group

    group = get_attn_tp_group() if reduce_on_attn_tp else get_tp_group()
    # Bias is fused into rank 0's GEMM only, so it is added once after the AR.
    bias = None if layer.tp_rank > 0 else layer.bias
    return _try_forward_pipeline(
        layer=layer,
        x=input_parallel,
        bias=bias,
        forward_batch=forward_batch,
        group=group,
    )


def _try_forward_pipeline(
    layer: RowParallelLinear,
    x: torch.Tensor,
    bias: Optional[torch.Tensor],
    forward_batch: Optional[ForwardBatch],
    group: GroupCoordinator,
) -> Optional[torch.Tensor]:
    if not isinstance(x, torch.Tensor) or x.dim() != 2 or not x.is_cuda:
        return None
    m = x.shape[0]
    knobs = _knobs()
    if m < knobs.min_tokens:
        return None
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        return None

    # Modes where a plain all_reduce would diverge from the regular collective.
    try:
        from sglang.srt.distributed.device_communicators.pynccl_allocator import (
            is_allocation_symmetric,
        )
    except ImportError:
        # Older trees: the group-level switch is the equivalent "stay away" signal.
        from sglang.srt.distributed.device_communicators.pynccl_allocator import (
            is_symmetric_memory_enabled as is_allocation_symmetric,
        )
    from sglang.srt.layers.moe.utils import should_skip_mlp_all_reduce
    from sglang.srt.runtime_context import get_exec

    if should_skip_mlp_all_reduce():
        return None
    if is_allocation_symmetric():
        return None
    if (
        forward_batch is not None
        and not forward_batch.forward_mode.is_decode_or_idle()
        and get_exec().comm.enable_quant_communications
    ):
        return None

    apply_into = getattr(layer.quant_method, "apply_into", None)
    if apply_into is None:
        return None
    if group.world_size != layer.tp_size:
        return None

    plan = compute_chunk_plan(m=m, num_chunks=knobs.num_chunks, ratio=knobs.ratio)
    if plan is None:
        return None

    from sglang.srt.layers.deep_gemm_wrapper import configure_deep_gemm_num_sms

    capped_sms = None
    if knobs.reserve_sms > 0:
        total_sms = _get_total_sms(x.device)
        if total_sms - knobs.reserve_sms >= 8:
            capped_sms = total_sms - knobs.reserve_sms

    global _LOGGED
    if not _LOGGED:
        _LOGGED = True
        logger.info(
            "Chunked GEMM+AR overlap engaged: m=%d chunks=%s capped_sms=%s group=%s",
            m,
            [hi - lo for lo, hi in plan],
            capped_sms,
            group.unique_name,
        )

    output = torch.empty((m, layer.output_size), dtype=x.dtype, device=x.device)
    device_group = group.device_group
    works = []
    for i, (lo, hi) in enumerate(plan):
        # Chunk 0 has no AR in flight yet; give it the full GPU.
        sm_ctx = configure_deep_gemm_num_sms(capped_sms) if i > 0 else nullcontext()
        with sm_ctx:
            apply_into(layer, x[lo:hi], output[lo:hi], bias=bias)
        # Runs on PG-NCCL's internal stream, after this chunk's GEMM and
        # concurrent with the next chunk's GEMM on the compute stream.
        works.append(
            torch.distributed.all_reduce(
                output[lo:hi], group=device_group, async_op=True
            )
        )
    for work in works:
        work.wait()
    return output
