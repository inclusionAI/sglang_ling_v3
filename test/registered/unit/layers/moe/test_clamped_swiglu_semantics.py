from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.moe_runner import humming as humming_runner
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.humming import HummingRunnerCore
from sglang.srt.models.bailing_moe_v3 import BailingMoE
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _humming_activation(
    inputs: torch.Tensor,
    *,
    gemm1_clamp_limit: float | None,
) -> torch.Tensor:
    runner = HummingRunnerCore(
        MoeRunnerConfig(
            num_experts=2,
            num_local_experts=2,
            activation="silu",
            gemm1_clamp_limit=gemm1_clamp_limit,
        )
    )
    outputs = torch.empty((*inputs.shape[:-1], inputs.shape[-1] // 2))
    runner.apply_activation(inputs, outputs)
    return outputs


def test_humming_matches_bailing_limit_only_swiglu():
    inputs = torch.tensor([[2.0, -1.0, 3.0, -4.0]])
    gate, up = inputs.chunk(2, dim=-1)
    limit = 1.5

    actual = _humming_activation(inputs, gemm1_clamp_limit=limit)
    torch.testing.assert_close(
        actual,
        F.silu(gate).clamp(max=limit) * up.clamp(min=-limit, max=limit),
    )


def test_bailing_v3_resolves_per_layer_limits_from_config_lists():
    expert_limits = [0] * 35 + [4] * 7
    shared_limits = [0] * 34 + [5] * 6 + [7] * 2

    assert BailingMoE._get_swiglu_limit(expert_limits, 34) is None
    assert BailingMoE._get_swiglu_limit(expert_limits, 35) == 4
    assert BailingMoE._get_swiglu_limit(shared_limits, 33) is None
    assert BailingMoE._get_swiglu_limit(shared_limits, 34) == 5
    assert BailingMoE._get_swiglu_limit(shared_limits, 40) == 7


def _run_masked_act_quant(*, gemm1_clamp_limit: float | None):
    """Drive `_grouped_masked_act_quant` and report which branch it took.

    Returns `(fused_call, activation_output)`; `fused_call` is None when the
    kernel was not reached.
    """
    num_experts, max_tokens, intermediate = 2, 4, 256
    runner = HummingRunnerCore(
        MoeRunnerConfig(
            num_experts=num_experts,
            num_local_experts=num_experts,
            activation="silu",
            top_k=8,
            gemm1_clamp_limit=gemm1_clamp_limit,
        )
    )
    fp8 = object()
    runner.layer = SimpleNamespace(
        humming_metas={"w2": SimpleNamespace(a_dtype=fp8, input_scale_group_size=128)}
    )

    torch.manual_seed(0)
    gate_up = torch.randn(num_experts, max_tokens, 2 * intermediate).bfloat16()
    buffers = {
        "gate_up_output": gate_up,
        "activation_output": torch.zeros(
            num_experts, max_tokens, intermediate, dtype=torch.bfloat16
        ),
    }
    expert_num_tokens = torch.tensor([max_tokens, 2], dtype=torch.int32)

    with (
        # `dtypes` / `HummingMethod` only exist when the closed-source humming
        # package is installed; the branch decision itself does not need it.
        patch.object(
            humming_runner, "dtypes", SimpleNamespace(float8e4m3=fp8), create=True
        ),
        patch.object(
            humming_runner,
            "HummingMethod",
            SimpleNamespace(may_quant_input=lambda **kwargs: ("fallback", kwargs)),
            create=True,
        ),
        patch(
            "sglang.kernels.ops.attention.dsv4.moe.silu_and_mul_masked_post_quant"
        ) as fused,
    ):
        runner._grouped_masked_act_quant(gate_up, expert_num_tokens, buffers)

    return (
        fused.call_args if fused.call_count else None,
        buffers["activation_output"],
    )


def test_masked_path_takes_the_fused_kernel_without_a_clamp():
    fused_call, activation_output = _run_masked_act_quant(gemm1_clamp_limit=None)

    assert fused_call is not None
    assert fused_call.kwargs["swiglu_limit"] is None
    # The fallback buffer must be untouched when the kernel ran.
    assert not activation_output.any()


def test_masked_path_falls_back_rather_than_dropping_the_gemm1_clamp():
    """The kernel only knows `swiglu_limit`, a different clamp semantics.

    Ling-3.0-plus sets `expert_swiglu_limit_list` on its late layers, so taking
    the fused path there would silently serve unclamped SwiGLU.
    """
    limit = 1.5
    fused_call, activation_output = _run_masked_act_quant(gemm1_clamp_limit=limit)

    assert fused_call is None

    torch.manual_seed(0)
    gate_up = torch.randn(2, 4, 512).bfloat16()
    gate, up = gate_up.chunk(2, dim=-1)
    torch.testing.assert_close(
        activation_output,
        F.silu(gate).clamp(max=limit) * up.clamp(min=-limit, max=limit),
    )
