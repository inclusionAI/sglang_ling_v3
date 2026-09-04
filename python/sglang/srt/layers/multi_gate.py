from contextlib import contextmanager
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

multi_gate_block_m = 64


@triton.jit
def multi_gate_kernel(
    # Pointers to matrices
    a_ptr,
    c_ptr,
    expert_bias_ptr,
    text_b_ptr,
    image_b_ptr,
    audio_b_ptr,
    text_bias_ptr,
    image_bias_ptr,
    audio_bias_ptr,
    mm_token_ids_ptr,
    mm_id_ptr,
    num_valid_tokens: tl.constexpr,
    compute_type: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,  # num_experts
    K: tl.constexpr,  # hidden_size
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(mm_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens
    off_mm = tl.load(mm_id_ptr + pid_m).to(tl.int64)

    if off_mm == 1:
        b_ptr = image_b_ptr
        bias_ptr = image_bias_ptr
    elif off_mm == 2:
        b_ptr = audio_b_ptr
        bias_ptr = audio_bias_ptr
    else:
        b_ptr = text_b_ptr
        bias_ptr = text_bias_ptr

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_SIZE_K):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k_start),
            other=0.0,
        ).to(compute_type)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_offset = stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptr + c_offset, accumulator, mask=c_mask)

    bias = tl.load(bias_ptr + offs_cn[None, :], mask=(offs_cn[None, :] < N), other=0.0)
    tl.store(expert_bias_ptr + c_offset, bias, mask=c_mask)


def get_config(tokens):
    configs = {
        1024: {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 64,
            "num_warps": 4,
            "num_stages": 3,
        },
        2048: {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 8,
            "num_stages": 3,
        },
        4096: {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 3,
        },
        8192: {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 64,
            "num_warps": 8,
            "num_stages": 3,
        },
    }
    config = configs[min(configs.keys(), key=lambda x: abs(x - tokens))]
    return config


@torch.compiler.disable
def multi_gate_triton_kernel(
    hidden_states,
    multi_gate_indices,
    text_w,
    image_w,
    audio_w,
    text_bias,
    image_bias,
    audio_bias,
    config=None,
):
    assert (
        multi_gate_indices is not None
    ), "Please call with init_multi_gate_mm_indices first"
    mm_token_ids, mm_id = multi_gate_indices
    num_tokens = hidden_states.shape[0]

    # transpose to [hidden_size, num_experts]
    text_w = text_w.transpose(0, 1)
    image_w = image_w.transpose(0, 1)
    audio_w = audio_w.transpose(0, 1)
    num_experts = text_w.shape[-1]
    assert (
        image_w.shape[-1] == num_experts
    ), f"Expect image_w.shape[-1] == num_experts, but got {image_w.shape=} and {num_experts=}"
    assert (
        audio_w.shape[-1] == num_experts
    ), f"Expect audio_w.shape[-1] == num_experts, but got {audio_w.shape=} and {num_experts=}"
    assert (
        text_bias.shape[-1] == num_experts
    ), f"Expect text_bias.shape[-1] == num_experts, but got {text_bias.shape=} and {num_experts=}"
    assert (
        image_bias.shape[-1] == num_experts
    ), f"Expect image_bias.shape[-1] == num_experts, but got {image_bias.shape=} and {num_experts=}"
    assert (
        audio_bias.shape[-1] == num_experts
    ), f"Expect audio_bias.shape[-1] == num_experts, but got {audio_bias.shape=} and {num_experts=}"

    router_logits = torch.empty(
        [num_tokens, num_experts],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    dynamic_expert_bias = torch.empty(
        [num_tokens, num_experts], dtype=torch.float32, device=hidden_states.device
    )

    if text_w.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif text_w.dtype == torch.float16:
        compute_type = tl.float16
    elif text_w.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {text_w.dtype}")

    M = mm_token_ids.shape[0]
    N = num_experts
    config = config or get_config(M)
    assert (
        config["BLOCK_SIZE_M"] == multi_gate_block_m
    ), f"Expect config['BLOCK_SIZE_M'] == {multi_gate_block_m}, but got {config['BLOCK_SIZE_M']}"

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    multi_gate_kernel[grid](
        hidden_states,
        router_logits,
        dynamic_expert_bias,
        text_w,
        image_w,
        audio_w,
        text_bias,
        image_bias,
        audio_bias,
        mm_token_ids,
        mm_id,
        num_valid_tokens=num_tokens,
        compute_type=compute_type,
        stride_am=hidden_states.stride(0),
        stride_ak=hidden_states.stride(1),
        stride_bk=text_w.stride(0),
        stride_bn=text_w.stride(1),
        stride_cm=router_logits.stride(0),
        stride_cn=router_logits.stride(1),
        M=M,
        N=N,
        K=hidden_states.shape[-1],
        **config,
    )
    return router_logits, dynamic_expert_bias


def create_multi_gate_mm_indices(
    input_ids, image_patch_token, video_patch_token, audio_patch_token
):
    vision_mask = (input_ids == image_patch_token) + (input_ids == video_patch_token)
    vision_indices = vision_mask.nonzero(as_tuple=False).squeeze(-1)
    audio_mask = input_ids == audio_patch_token
    audio_indices = audio_mask.nonzero(as_tuple=False).squeeze(-1)
    text_indices = (~vision_mask & ~audio_mask).nonzero(as_tuple=False).squeeze(-1)
    mm_indices = [text_indices, vision_indices, audio_indices]

    block_m = multi_gate_block_m
    mm_blocks = list((ind.shape[0] + block_m - 1) // block_m for ind in mm_indices)
    total_tokens = input_ids.shape[0]
    mm_token_ids = torch.full(
        (sum(mm_blocks) * block_m,),
        total_tokens,
        dtype=torch.int32,
        device=input_ids.device,
    )
    mm_id = torch.empty(sum(mm_blocks), dtype=torch.int32, device=input_ids.device)
    for i, ind in enumerate(mm_indices):
        block_beg = sum(mm_blocks[:i])
        mm_token_ids[block_beg * block_m :][: ind.shape[0]] = ind
        mm_id[block_beg : block_beg + mm_blocks[i]] = i
    return mm_token_ids, mm_id


_multi_gate_indices: Optional[Tuple[torch.Tensor, torch.Tensor]] = None


@contextmanager
def init_multi_gate_mm_indices(
    forward_batch, input_ids, image_patch_token, video_patch_token, audio_patch_token
):
    try:
        global _multi_gate_indices
        if forward_batch.contains_mm_inputs():
            _multi_gate_indices = create_multi_gate_mm_indices(
                input_ids, image_patch_token, video_patch_token, audio_patch_token
            )
        yield
    finally:
        _multi_gate_indices = None


def get_multi_gate_indices():
    return _multi_gate_indices
