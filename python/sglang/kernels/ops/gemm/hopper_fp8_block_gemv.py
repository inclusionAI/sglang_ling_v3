from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Hopper small-M blockwise-FP8 GEMV (see csrc/gemm/hopper_fp8_block_gemv.cuh).
#
# Decode-time dense projections of a blockwise-FP8 model (128x128 weight
# blocks, 1x128 activation groups) run at M = tokens-in-flight, i.e. 1..8 for
# bs <= 2 with a few draft tokens. DeepGEMM's TMA/WGMMA pipeline moves those
# few-MB weights at ~1 TB/s on H20 (18 us for a 21 MB projection that streams
# in 6 us). Streaming the weights straight into mma.m16n8k32 e4m3 fragments,
# one warp per 16-row tile for all 8 token columns, runs near the bandwidth
# floor at these M. M <= 16 (two n8 column groups) covers bs <= 4 verify.

_MAX_M = 16
_MAX_K = 4096  # 16 x K bytes of activation slab in shared memory


def _config(m: int, n: int, k: int) -> tuple[int, int, int, int]:
    """(max_m, row_tiles_per_block, k_split_warps, k_unroll)."""
    max_m = 8 if m <= 8 else 16
    tiles = (n + 15) // 16
    nkb = k // 128
    row_tiles = 2 if tiles >= 256 else 1
    # Split K across warps until the grid covers the SMs a few times over;
    # each warp keeps >= 2 scale blocks so the unrolled loads still pipeline.
    k_split = 1
    for cand in (8, 4, 2):
        if nkb % cand == 0 and nkb // cand >= 2 and cand * row_tiles <= 8:
            k_split = cand
            break
    unroll = 2 if (nkb // k_split) % 2 == 0 else 1
    return max_m, row_tiles, k_split, unroll


@cache_once
def _jit_hopper_fp8_block_gemv_module(
    max_m: int, row_tiles: int, k_split: int, unroll: int
) -> Module:
    args = make_cpp_args(max_m, row_tiles, k_split, unroll)
    return load_jit(
        "hopper_fp8_block_gemv",
        *args,
        cuda_files=["gemm/hopper_fp8_block_gemv.cuh"],
        cuda_wrappers=[("run", f"sglang::HopperFp8BlockGemvKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def use_hopper_fp8_block_gemv(
    m: int, n: int, k: int, block_size: Sequence[int]
) -> bool:
    if not (
        1 <= m <= _MAX_M
        and tuple(block_size) == (128, 128)
        and k % 128 == 0
        and 128 <= k <= _MAX_K
        and n >= 16
    ):
        return False
    # Two column groups (M > 8) double the MMA and slab traffic; measured on
    # H20 they still beat DeepGEMM only on the narrow projections (N <= 1536:
    # 1024x4096 9.4 -> 7.0 us), and lose on the wide ones (5120x4096 18.5 -> 23 us).
    return m <= 8 or n <= 1536


def hopper_fp8_block_gemv(
    xq: torch.Tensor,
    xs: torch.Tensor,
    wq: torch.Tensor,
    ws: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """out[M, N] (bf16) = xq[M, K] @ wq[N, K]^T with 1x128 / 128x128 fp8 block scales.

    xq / wq are float8_e4m3fn, xs is fp32 [M, K/128] (row-major), ws is fp32
    [ceil(N/128), K/128]; all contiguous.
    """
    m, k = xq.shape
    n = wq.shape[0]
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=xq.device)
    module = _jit_hopper_fp8_block_gemv_module(*_config(m, n, k))
    module.run(xq, xs, wq, ws, out)
    return out
