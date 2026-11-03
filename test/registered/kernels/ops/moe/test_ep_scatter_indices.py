"""Expert-offset initialization must not race index-buffer writes."""

import sys

import pytest
import torch
import triton

from sglang.kernels.ops.moe.ep_moe_kernels import _fwd_kernel_ep_scatter_1, ep_scatter
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize(
    "num_experts,num_tokens,alignment",
    [
        (1, 1, 128),
        (512, 0, 128),
        (64, 32, 128),
        (64, 1839, 128),
        (511, 1839, 128),
        (512, 1, 128),
        (512, 1839, 128),
        (512, 8192, 128),
        (512, 1839, 32),
    ],
)
def test_expert_indices_preserve_bounds(num_experts, num_tokens, alignment):
    generator = torch.Generator().manual_seed(20261103)
    routes = torch.randint(num_experts, (num_tokens * 8,), generator=generator)
    valid = torch.bincount(routes, minlength=num_experts).to(torch.int32)
    padded = ((valid + alignment - 1) // alignment) * alignment
    assignments = routes.numel()
    nonempty_bound = min(assignments, num_experts)
    total = alignment * (nonempty_bound + (assignments - nonempty_bound) // alignment)
    padded[-1] += total - int(padded.sum())
    offsets = padded.cumsum(0) - padded
    expected = torch.full((total,), -1, dtype=torch.int32)
    for expert in range(num_experts):
        start = int(offsets[expert])
        expected[start : start + int(valid[expert])] = expert

    expected = expected.cuda()
    padded, valid, offsets = padded.cuda(), valid.cuda(), offsets.cuda()
    guard = 2 * total + 8192
    storage = torch.empty(total + 2 * guard, device="cuda", dtype=torch.int32)
    indices = storage[guard : guard + total]
    starts = torch.empty_like(padded)
    for attempt in range(200):
        storage.fill_(-777)
        # A premature global read points into an inspectable outer guard,
        # rather than corrupting unrelated allocations or the CUDA context.
        starts.fill_(total + 4096)
        _fwd_kernel_ep_scatter_1[(num_experts,)](
            padded,
            valid,
            starts,
            indices,
            num_experts=num_experts,
            BLOCK_E=alignment,
            BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
            num_warps=8,
        )
        torch.cuda.synchronize()
        assert bool(storage[:guard].eq(-777).all()), f"leading guard, {attempt=}"
        assert bool(storage[guard + total :].eq(-777).all()), f"trailing guard, {attempt=}"
        torch.testing.assert_close(indices, expected, rtol=0, atol=0)
        torch.testing.assert_close(starts, offsets.to(torch.int32), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("num_experts,expert_start", [(64, 0), (512, 1024)])
def test_scatter_consumes_expert_offsets(dtype, num_experts, expert_start):
    local = torch.tensor(
        [
            [0, num_experts - 1],
            [num_experts // 2, -1],
            [-1, num_experts],
            [0, num_experts // 2],
        ],
        dtype=torch.int64,
    )
    routes = (local + expert_start).cuda()
    live_routes = local[(local >= 0) & (local < num_experts)]
    valid = torch.bincount(live_routes, minlength=num_experts).to(torch.int32)
    padded = ((valid + 127) // 128) * 128
    total = int(padded.sum())
    expected_start = padded.cumsum(0) - padded
    x = (torch.arange(4 * 256).reshape(4, 256) % 16).cuda().to(dtype)
    is_fp8 = dtype == torch.float8_e4m3fn
    scales = (
        torch.arange(1, 9, device="cuda", dtype=torch.float32).reshape(4, 2)
        if is_fp8
        else None
    )
    out = torch.empty((total, 256), device="cuda", dtype=dtype)
    out_scales = torch.empty((total, 2), device="cuda") if is_fp8 else None
    starts = torch.empty(num_experts, device="cuda", dtype=torch.int32)
    indices = torch.empty(total, device="cuda", dtype=torch.int32)
    destinations = torch.empty_like(routes)
    ep_scatter(
        x,
        scales,
        routes,
        padded.cuda(),
        valid.cuda(),
        starts,
        out,
        out_scales,
        indices,
        destinations,
        expert_start=expert_start,
    )
    torch.cuda.synchronize()
    for token in range(4):
        for slot in range(2):
            expert = int(local[token, slot])
            dest = int(destinations[token, slot])
            if not 0 <= expert < num_experts:
                assert dest == -1
                continue
            assert int(expected_start[expert]) <= dest < int(
                expected_start[expert] + valid[expert]
            )
            assert int(indices[dest]) == expert
            torch.testing.assert_close(out[dest].float(), x[token].float(), rtol=0, atol=0)
            if is_fp8:
                torch.testing.assert_close(out_scales[dest], scales[token], rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
