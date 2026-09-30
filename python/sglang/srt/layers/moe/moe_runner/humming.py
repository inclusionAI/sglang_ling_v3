from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional
from weakref import WeakValueDictionary

import torch

from sglang.kernels.ops.moe.ep_moe_kernels import (
    moe_permute,
    moe_permute_with_scale,
    moe_unpermute,
)
from sglang.kernels.ops.moe.moe_fused_mul_sum import moe_fused_mul_sum
from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.utils import (
    MoeRunnerBackend,
    should_skip_post_experts_all_reduce,
)
from sglang.srt.runtime_context import get_forward, get_parallel, get_stream
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLCombineInput,
        DeepEPLLDispatchOutput,
        DeepEPNormalCombineInput,
        DeepEPNormalDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


logger = logging.getLogger(__name__)


try:
    from humming import dtypes
    from humming.config import GemmType as HummingGemmType
    from humming.layer import HummingMethod

    _humming_available = True
except ModuleNotFoundError:
    _humming_available = False


def get_standard_humming_moe_gemm_type() -> HummingGemmType:
    env_gemm_type_str = envs.SGLANG_HUMMING_MOE_GEMM_TYPE.get().lower()
    if env_gemm_type_str == "grouped":
        gemm_type = HummingGemmType.GROUPED_CONTIGUOUS
    elif env_gemm_type_str == "indexed":
        gemm_type = HummingGemmType.INDEXED
    else:
        gemm_type = HummingGemmType.INDEXED

    logger.info_once(f"Using {gemm_type.value} gemm for humming moe")

    return gemm_type


@dataclass
class HummingRunnerInput(RunnerInput):
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    gemm_type: HummingGemmType
    expert_num_tokens: torch.Tensor | None = None
    expected_m: int | None = None
    hidden_states_scale: torch.Tensor | None = None
    apply_routed_scaling_factor: bool = True
    output_add: torch.Tensor | None = None

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.HUMMING


@dataclass
class HummingRunnerOutput(RunnerOutput):
    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.HUMMING


@dataclass
class HummingMoeQuantInfo(MoeQuantInfo):
    layer: torch.nn.Module


def _humming_moe_runner_core_run_fake(
    moe_runner_id: int,
    gemm_type: str,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_num_tokens: torch.Tensor | None = None,
    expected_m: int | None = None,
    hidden_states_scale: torch.Tensor | None = None,
    apply_routed_scaling_factor: bool = True,
    output_add: torch.Tensor | None = None,
) -> torch.Tensor:
    runner = HummingRunnerCore.runner_cores[moe_runner_id]
    output_dtype = runner.config.params_dtype or hidden_states.dtype
    return torch.empty_like(hidden_states, dtype=output_dtype)


@register_custom_op(fake_impl=_humming_moe_runner_core_run_fake)
def humming_moe_runner_core_run(
    moe_runner_id: int,
    gemm_type: str,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_num_tokens: torch.Tensor | None = None,
    expected_m: int | None = None,
    hidden_states_scale: torch.Tensor | None = None,
    apply_routed_scaling_factor: bool = True,
    output_add: torch.Tensor | None = None,
) -> torch.Tensor:
    runner = HummingRunnerCore.runner_cores[moe_runner_id]
    if envs.SGLANG_OPT_MOE_FINALIZE_AR_OVERLAP.get():
        # Rewrite on every call so a previous layer's finalize+AR overlap
        # cannot leak a stale True into this layer's skip decision.
        get_forward().set("moe_post_experts_allreduce_done", False)
    if gemm_type == "indexed":
        return runner._run_indexed_gemm(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            hidden_states_scale=hidden_states_scale,
            apply_routed_scaling_factor=apply_routed_scaling_factor,
            output_add=output_add,
        )
    # Only the indexed gemm finalizes through moe_fused_mul_sum, which is the
    # kernel that folds output_add in; supports_fused_output_add gates on that,
    # so a non-indexed gemm here means the gate and the runner disagree.
    assert output_add is None, f"{gemm_type} gemm cannot fuse output_add"
    if gemm_type == "grouped_contiguous":
        return runner._run_grouped_contiguous_gemm(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            hidden_states_scale=hidden_states_scale,
            apply_routed_scaling_factor=apply_routed_scaling_factor,
        )
    elif gemm_type == "grouped_masked":
        assert expected_m is not None and expert_num_tokens is not None
        return runner._run_grouped_masked_gemm(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            expected_m=expected_m,
            expert_num_tokens=expert_num_tokens,
            hidden_states_scale=hidden_states_scale,
        )
    else:
        raise ValueError(f"Unknown gemm type: {gemm_type}")


def pad_input_scale_to_padded_k(
    input_scale: torch.Tensor | None, padded_k: int, group_size: int
) -> torch.Tensor | None:
    """Append zero scale groups when humming padded the weight's K.

    humming pads a sublayer's weight along K to its input alignment (Ling-3.0-plus
    at TP8: the w2 intermediate 160 -> 192 for FP8 activations). The activation
    keeps its logical K, but the launcher validates the per-row group scales at
    the padded width (`as.size(1) 5 != 6`), so the quantized scales need one
    zero group per padded block; those groups only ever meet zero weight
    columns. bf16 activations (no scale), ungrouped scales and already-aligned
    K pass through untouched.
    """
    if input_scale is None or group_size <= 0 or input_scale.dim() != 2:
        return input_scale
    expected_groups = padded_k // group_size
    if input_scale.size(1) >= expected_groups:
        return input_scale
    return torch.nn.functional.pad(
        input_scale, (0, expected_groups - input_scale.size(1))
    )


class HummingRunnerCore(MoeRunnerCore):
    runner_cores: WeakValueDictionary = WeakValueDictionary()

    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)
        assert config.num_local_experts is not None
        assert config.num_experts is not None
        self.num_experts = config.num_local_experts
        self.global_num_experts = config.num_experts
        self.activation = config.activation
        self.swiglu_limit = config.swiglu_limit
        self.gemm1_clamp_limit = config.gemm1_clamp_limit
        self.layer: torch.nn.Module | None = None
        self.humming_gemm_configs = {}
        HummingRunnerCore.runner_cores[id(self)] = self

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.HUMMING

    @property
    def supports_fused_output_add(self) -> bool:
        # Only the indexed gemm finalizes through moe_fused_mul_sum; the grouped
        # gemms go through moe_unpermute / the a2a combine instead.
        return (
            _humming_available
            and get_standard_humming_moe_gemm_type() == HummingGemmType.INDEXED
        )

    def get_humming_gemm_configs(self, humming_gemm_type: HummingGemmType):
        if humming_gemm_type.value in self.humming_gemm_configs:
            return self.humming_gemm_configs[humming_gemm_type.value]

        compute_config = {
            "use_f16_accum": envs.SGLANG_HUMMING_USE_F16_ACCUM.get(),
            "gemm_type": humming_gemm_type.value,
        }
        w13_tuning_config = HummingMethod.get_default_tuning_configs(
            layer=self.layer,
            use_f16_accum=envs.SGLANG_HUMMING_USE_F16_ACCUM.get(),
            gemm_type=humming_gemm_type,
            sublayer_name="w13",
        )
        w2_tuning_config = HummingMethod.get_default_tuning_configs(
            layer=self.layer,
            use_f16_accum=envs.SGLANG_HUMMING_USE_F16_ACCUM.get(),
            gemm_type=humming_gemm_type,
            sublayer_name="w2",
        )
        w13_tuning_config = self._apply_tuning_override(
            humming_gemm_type=humming_gemm_type,
            sublayer_name="w13",
            default=w13_tuning_config,
        )
        w2_tuning_config = self._apply_tuning_override(
            humming_gemm_type=humming_gemm_type,
            sublayer_name="w2",
            default=w2_tuning_config,
        )
        self.humming_gemm_configs[humming_gemm_type.value] = {
            "compute_config": compute_config,
            "w13_tuning_config": w13_tuning_config,
            "w2_tuning_config": w2_tuning_config,
            "compute_config_str": json.dumps(compute_config),
            "w13_tuning_config_str": json.dumps(w13_tuning_config),
            "w2_tuning_config_str": json.dumps(w2_tuning_config),
        }

        return self.humming_gemm_configs[humming_gemm_type.value]

    @staticmethod
    def _apply_tuning_override(
        *, humming_gemm_type: HummingGemmType, sublayer_name: str, default: list
    ) -> list:
        # Same [min_m, max_m, config] bucket shape as the heuristic output, so
        # bucket selection downstream does not change.
        override = envs.SGLANG_HUMMING_TUNING_OVERRIDE.get()
        if not override:
            return default
        entries = (override.get(humming_gemm_type.value) or {}).get(sublayer_name)
        if not entries:
            return default
        # Replace only the buckets whose [min_m, max_m) range is named; the
        # heuristic keeps every other range (decode-sized M stays untouched).
        patches = {(int(lo), int(hi)): dict(cfg) for lo, hi, cfg in entries}
        merged = []
        for lo, hi, cfg in default:
            merged.append((lo, hi, patches.pop((int(lo), int(hi)), cfg)))
        if patches:
            raise ValueError(
                f"SGLANG_HUMMING_TUNING_OVERRIDE {humming_gemm_type.value}/"
                f"{sublayer_name}: ranges {sorted(patches)} match no heuristic "
                f"bucket; available: {[(lo, hi) for lo, hi, _ in default]}"
            )
        logger.info(
            "humming tuning override for %s/%s: %d of %d bucket(s) replaced",
            humming_gemm_type.value,
            sublayer_name,
            len(entries),
            len(default),
        )
        return merged

    def estimate_local_valid_shape_m(
        self,
        topk_ids: torch.Tensor,
        expected_m: int | None = None,
    ):
        # estimate shape_m for kernel tuning
        if expected_m is not None:
            return expected_m * self.num_experts

        # TODO: update for EP and DP
        return topk_ids.nelement()

    def get_buffer_metas(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        gemm_type: HummingGemmType,
    ):
        num_experts = self.num_experts
        N = self.layer.intermediate_size_per_partition
        K = self.layer.hidden_size
        assert isinstance(num_experts, int)
        assert isinstance(N, int)
        assert isinstance(K, int)

        # hidden_states
        # (-> quanted_gate_up_input) (if not BF16/FP16 activation)
        # -> gate_up_output
        # -> activation_output
        # (-> quanted_down_input) (if not BF16/FP16 activation)
        # -> down_output
        # (-> output) (if not is_grouped_masked)
        # Neighboring nodes are required to utilize distinct workspaces.
        # The output must be derived from workspace1.

        is_grouped_masked = gemm_type == HummingGemmType.GROUPED_MASKED
        output_shape: tuple[int, ...]
        if gemm_type == HummingGemmType.GROUPED_MASKED:
            if hidden_states.ndim == 3:
                max_num_tokens = hidden_states.size(1)
            else:
                max_num_tokens = hidden_states.size(0) // num_experts
            input_shape_m = num_experts * max_num_tokens
            real_shape_m = num_experts * max_num_tokens
            output_shape = (num_experts, max_num_tokens, K)
        else:
            input_shape_m = hidden_states.size(0)
            real_shape_m = hidden_states.size(0) * topk_ids.size(1)
            if gemm_type == HummingGemmType.GROUPED_CONTIGUOUS:
                input_shape_m = real_shape_m
            output_shape = (hidden_states.size(0), K)

        down_input_size = N
        a_dtype = self.layer.humming_metas["w13"].a_dtype
        c_dtype = self.layer.humming_metas["w13"].c_dtype
        num_bits = a_dtype.num_bits
        torch_dtype_map = {
            dtypes.float16: torch.float16,
            dtypes.bfloat16: torch.bfloat16,
            dtypes.float8e4m3: torch.float8_e4m3fn,
            dtypes.int8: torch.int8,
            dtypes.int4: torch.uint8,
        }

        buffer_metas = {
            "quanted_gate_up_input": {
                "shape": (input_shape_m, K),
                "dtype": torch_dtype_map[a_dtype],
            },
            "gate_up_output": {
                "shape": (real_shape_m, N * 2),
                "dtype": torch_dtype_map[c_dtype],
            },
            "activation_output": {
                "shape": (real_shape_m, down_input_size),
                "dtype": torch_dtype_map[c_dtype],
            },
            "quanted_down_input": {
                "shape": (real_shape_m, down_input_size),
                "dtype": torch_dtype_map[a_dtype],
            },
            "down_output": {
                "shape": output_shape if is_grouped_masked else (real_shape_m, K),
                "dtype": torch_dtype_map[c_dtype],
            },
            "output": {
                "shape": output_shape,
                "dtype": torch_dtype_map[c_dtype],
            },
        }

        for key in buffer_metas:
            meta = buffer_metas[key]
            if "quanted" in key and a_dtype.num_bits == 4:
                meta["shape"] = meta["shape"][:-1] + (meta["shape"][-1] // 2,)

        if num_bits == 16:
            required_buffers = ["gate_up_output", "activation_output", "down_output"]
        else:
            required_buffers = [
                "quanted_gate_up_input",
                "gate_up_output",
                "activation_output",
                "quanted_down_input",
                "down_output",
            ]

        # grouped masked moe use down_output as output
        if gemm_type != HummingGemmType.GROUPED_MASKED:
            required_buffers.append("output")

        return buffer_metas, required_buffers

    def _workspace_shapes(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        gemm_type: HummingGemmType,
    ):
        buffer_metas, required_buffers = self.get_buffer_metas(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            gemm_type=gemm_type,
        )

        workspace1_nbytes = 0
        workspace2_nbytes = 0

        for index, name in enumerate(required_buffers[::-1]):
            buffer_meta = buffer_metas[name]
            nelement = math.prod(buffer_meta["shape"])
            nbytes = nelement * buffer_meta["dtype"].itemsize
            if index % 2 == 0:
                workspace1_nbytes = max(workspace1_nbytes, nbytes)
            else:
                workspace2_nbytes = max(workspace2_nbytes, nbytes)

        output_key = (
            "down_output" if gemm_type == HummingGemmType.GROUPED_MASKED else "output"
        )
        output_shape = buffer_metas[output_key]["shape"]

        return (workspace1_nbytes // 2,), (workspace2_nbytes // 2,), output_shape

    def make_workspaces(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        gemm_type: HummingGemmType,
    ):
        shapes = self._workspace_shapes(hidden_states, topk_ids, gemm_type)
        workspace1_shape, workspace2_shape, output_shape = shapes
        torch_dtype = self.layer.params_dtype
        device = hidden_states.device
        workspace1 = torch.empty(workspace1_shape, dtype=torch_dtype, device=device)
        workspace2 = torch.empty(workspace2_shape, dtype=torch_dtype, device=device)
        output = workspace1[: math.prod(output_shape)].view(*output_shape)
        return workspace1, workspace2, output

    def prepare_buffers(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        gemm_type: HummingGemmType,
    ) -> dict[str, torch.Tensor]:
        workspace1, workspace2, output = self.make_workspaces(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            gemm_type=gemm_type,
        )
        buffer_metas, required_buffers = self.get_buffer_metas(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            gemm_type=gemm_type,
        )
        buffers = {"output": output}
        for index, name in enumerate(required_buffers[::-1]):
            buffer_meta = buffer_metas[name]
            workspace = workspace1 if index % 2 == 0 else workspace2
            workspace = workspace.view(buffer_meta["dtype"])
            shape = buffer_meta["shape"]
            tensor = workspace[: math.prod(shape)].view(*shape)
            buffers[name] = tensor

        return buffers

    def apply_activation(self, inputs: torch.Tensor, outputs: torch.Tensor):
        if self.activation == "silu" and self.gemm1_clamp_limit is not None:
            gate, up = inputs.chunk(2, dim=-1)
            limit = float(self.gemm1_clamp_limit)
            outputs.copy_(
                torch.nn.functional.silu(gate).clamp(max=limit)
                * up.clamp(min=-limit, max=limit)
            )
            return
        if self.activation == "silu" and self.swiglu_limit is not None:
            from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
                act_and_mul_triton,
            )

            in_2d = inputs.view(-1, inputs.shape[-1])
            out_2d = outputs.view(-1, outputs.shape[-1])
            act_and_mul_triton(
                gateup_output=in_2d,
                down_input=out_2d,
                config={},
                activation="silu",
                swiglu_limit=float(self.swiglu_limit),
            )
            return
        if self.activation == "silu":
            from sgl_kernel import silu_and_mul

            silu_and_mul(inputs, outputs)
        elif self.activation == "gelu":
            from sgl_kernel import gelu_and_mul

            gelu_and_mul(inputs, outputs)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def _grouped_masked_act_quant(
        self,
        gate_up: torch.Tensor,
        expert_num_tokens: torch.Tensor,
        buffers: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        num_experts, max_tokens, two_i = gate_up.shape
        intermediate = two_i // 2
        w2_meta = self.layer.humming_metas["w2"]
        groups = intermediate // 128
        num_threads = intermediate // 8
        use_fused_masked_act_quant = (
            w2_meta.a_dtype == dtypes.float8e4m3
            and w2_meta.input_scale_group_size == 128
            and self.activation == "silu"
            and gate_up.dtype == torch.bfloat16
            and intermediate % 256 == 0
            and num_threads <= 1024
            and num_experts <= min(256, num_threads)
            and self.config.top_k is not None
            and self.config.top_k > 0
            # The kernel only knows the `swiglu_limit` clamp. `gemm1_clamp_limit`
            # is the other SwiGLU clamp semantics (clamp silu(gate) from above,
            # up symmetrically) -- feeding it in as swiglu_limit, or dropping it,
            # both change the result, so fall back to apply_activation, which
            # implements it. Ling-3.0-plus sets it on its late layers.
            and self.gemm1_clamp_limit is None
        )
        if use_fused_masked_act_quant:
            from sglang.kernels.ops.moe.dsv4 import silu_and_mul_masked_post_quant

            down_input = torch.empty(
                (num_experts, max_tokens, intermediate),
                dtype=torch.float8_e4m3fn,
                device=gate_up.device,
            )
            down_scale = torch.empty(
                (num_experts, max_tokens, groups),
                dtype=torch.float32,
                device=gate_up.device,
            )
            silu_and_mul_masked_post_quant(
                gate_up,
                down_input,
                down_scale,
                128,
                expert_num_tokens,
                scale_ue8m0=False,
                topk=self.config.top_k,
                swiglu_limit=self.swiglu_limit,
            )
            return down_input.flatten(0, 1), down_scale.flatten(0, 1)

        self.apply_activation(
            inputs=buffers["gate_up_output"],
            outputs=buffers["activation_output"],
        )
        return HummingMethod.may_quant_input(
            layer=self.layer,
            inputs=buffers["activation_output"],
            quanted_input=buffers.get("quanted_down_input"),
            sublayer_name="w2",
        )

    def run(
        self,
        runner_input: HummingRunnerInput,
        quant_info: HummingMoeQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> HummingRunnerOutput:
        self.layer = quant_info.layer
        if runner_input.hidden_states.size(0) == 0:
            return HummingRunnerOutput(
                hidden_states=torch.empty_like(
                    runner_input.hidden_states, dtype=self.config.params_dtype
                )
            )

        # To make it compatible with dynamic shapes in torch.compile,
        # we wrap the main logic inside a torch op.
        # (the moe_block_size selection in indexed gemm would break dynamic shapes).
        output = humming_moe_runner_core_run(
            moe_runner_id=id(self),
            gemm_type=runner_input.gemm_type.value,
            hidden_states=runner_input.hidden_states,
            topk_weights=runner_input.topk_weights,
            topk_ids=runner_input.topk_ids,
            expected_m=runner_input.expected_m,
            expert_num_tokens=runner_input.expert_num_tokens,
            hidden_states_scale=runner_input.hidden_states_scale,
            apply_routed_scaling_factor=runner_input.apply_routed_scaling_factor,
            output_add=runner_input.output_add,
        )

        return HummingRunnerOutput(hidden_states=output)

    def _prepare_indexed_gemm_kwargs(
        self, topk_ids: torch.Tensor
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size

        configs = self.get_humming_gemm_configs(HummingGemmType.INDEXED)
        valid_shape_m = self.estimate_local_valid_shape_m(topk_ids)

        for min_shape_m, max_shape_m, config in configs["w13_tuning_config"]:
            if valid_shape_m > min_shape_m and valid_shape_m <= max_shape_m:
                moe_block_size = config["block_shape"][0]
                break
        else:
            raise ValueError(f"cannot found moe_block_size for shape {valid_shape_m}")

        sorted_ids, expert_ids, num_tokens_padded = moe_align_block_size(
            topk_ids=topk_ids,
            block_size=moe_block_size,
            num_experts=self.num_experts,
            ignore_invalid_expert=True,
        )

        moe_common_kwargs = {
            "sorted_ids": sorted_ids,
            "expert_ids": expert_ids,
            "num_tokens_padded": num_tokens_padded,
            "compute_config": configs["compute_config_str"],
            "valid_shape_m": valid_shape_m,
        }

        top_k = topk_ids.size(1)
        moe_kwargs1 = {
            "top_k": top_k,
            "tuning_config": configs["w13_tuning_config_str"],
        }
        moe_kwargs2 = {"top_k": 1, "tuning_config": configs["w2_tuning_config_str"]}
        moe_kwargs1.update(moe_common_kwargs)
        moe_kwargs2.update(moe_common_kwargs)

        return moe_kwargs1, moe_kwargs2

    def _run_indexed_gemm(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        hidden_states_scale: torch.Tensor | None = None,
        apply_routed_scaling_factor: bool = True,
        output_add: torch.Tensor | None = None,
    ):
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        if hidden_states_scale is not None:
            hidden_states_scale = hidden_states_scale.reshape(
                -1, hidden_states_scale.size(-1)
            ).contiguous()
        buffers = self.prepare_buffers(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            gemm_type=HummingGemmType.INDEXED,
        )

        moe_kwargs1, moe_kwargs2 = self._prepare_indexed_gemm_kwargs(topk_ids)

        inputs, input_scale = HummingMethod.may_quant_input(
            layer=self.layer,
            inputs=hidden_states,
            input_scale=hidden_states_scale,
            quanted_input=buffers.get("quanted_gate_up_input", None),
            sublayer_name="w13",
        )

        HummingMethod.forward_layer(
            layer=self.layer,
            inputs=inputs,
            input_scale=input_scale,
            outputs=buffers["gate_up_output"],
            sublayer_name="w13",
            **moe_kwargs1,
        )

        self.apply_activation(
            inputs=buffers["gate_up_output"],
            outputs=buffers["activation_output"],
        )

        inputs, input_scale = HummingMethod.may_quant_input(
            layer=self.layer,
            inputs=buffers["activation_output"],
            quanted_input=buffers.get("quanted_down_input", None),
            sublayer_name="w2",
        )
        w2_meta = self.layer.humming_metas["w2"]
        input_scale = pad_input_scale_to_padded_k(
            input_scale,
            padded_k=getattr(w2_meta, "shape_k", inputs.size(1)),
            group_size=getattr(w2_meta, "input_scale_group_size", 0) or 0,
        )

        HummingMethod.forward_layer(
            layer=self.layer,
            inputs=inputs,
            input_scale=input_scale,
            outputs=buffers["down_output"].view(-1, hidden_states.size(-1)),
            sublayer_name="w2",
            **moe_kwargs2,
        )

        routed_scaling_factor = (
            self.config.routed_scaling_factor if apply_routed_scaling_factor else None
        )
        finalize_inputs = buffers["down_output"].view(*topk_ids.shape, -1)
        if self._use_finalize_ar_overlap(topk_ids.shape[0], output_add):
            self._finalize_with_ar_overlap(
                inputs=finalize_inputs,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                routed_scaling_factor=routed_scaling_factor,
                outputs=buffers["output"],
                output_add=output_add,
            )
            get_forward().set("moe_post_experts_allreduce_done", True)
        else:
            moe_fused_mul_sum(
                inputs=finalize_inputs,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                is_ep=self.num_experts != self.global_num_experts,
                routed_scaling_factor=routed_scaling_factor,
                outputs=buffers["output"],
                output_add=output_add,
            )

        return buffers["output"]

    # -- finalize + all-reduce overlap (SGLANG_OPT_MOE_FINALIZE_AR_OVERLAP) --

    _FINALIZE_AR_OVERLAP_MIN_TOKENS = 4096
    _FINALIZE_AR_OVERLAP_NUM_CHUNKS = 4

    def _use_finalize_ar_overlap(
        self, num_tokens: int, output_add: torch.Tensor | None
    ) -> bool:
        """Whether to run finalize in token chunks with each chunk's TP
        all-reduce on a side stream (hides the finalize under the reduce).

        Restricted to the one configuration where the model-level TP
        all-reduce is the only reducer left, so taking it over is safe: no
        EP, moe_tp > 1, the layer's own reduce_results off (the model-level
        reduce site is the skip-predicate-gated one), and nothing else
        absorbing the reduce. Small batches keep the plain path -- the
        finalize is then too cheap to be worth the event round-trips, and the
        threshold also keeps captured decode/verify graphs out.
        """
        if not envs.SGLANG_OPT_MOE_FINALIZE_AR_OVERLAP.get():
            return False
        if num_tokens < self._FINALIZE_AR_OVERLAP_MIN_TOKENS:
            return False
        if self.num_experts != self.global_num_experts:
            return False
        if self.layer is None or getattr(self.layer, "reduce_results", True):
            return False
        # An external shared-expert output must be included before the TP sum.
        # When FusedMoE cannot fold output_add into this runner, it passes None
        # and adds that output after the runner returns. In that case leave the
        # reduction at the model level. Fused shared experts are already in the
        # runner's finalize inputs.
        if output_add is None and self.layer.num_fused_shared_experts == 0:
            return False
        if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
            return False
        if get_parallel().moe_tp_size <= 1:
            return False
        if should_skip_post_experts_all_reduce(is_tp_path=True):
            return False
        return True

    def _finalize_with_ar_overlap(
        self,
        inputs: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        routed_scaling_factor: float | None,
        outputs: torch.Tensor,
        output_add: torch.Tensor | None,
    ) -> None:
        """Chunked finalize with the per-chunk TP all-reduce on a side stream.

        Chunk i reduces while chunk i+1 finalizes; the reduce total is
        unchanged, so what gets hidden is the finalize itself (~0.4 ms/layer
        at 25k tokens). Bitwise identical to finalize-then-reduce: chunking
        only splits rows, and each row is finalized and reduced exactly once.
        """
        from sglang.srt.distributed.communication_op import (
            moe_tensor_model_parallel_all_reduce,
        )

        comm_stream = get_stream("moe_finalize_ar")
        main_stream = torch.cuda.current_stream()
        num_tokens = topk_ids.shape[0]
        chunk = -(-num_tokens // self._FINALIZE_AR_OVERLAP_NUM_CHUNKS)
        for start in range(0, num_tokens, chunk):
            end = min(start + chunk, num_tokens)
            moe_fused_mul_sum(
                inputs=inputs[start:end],
                topk_weights=topk_weights[start:end],
                topk_ids=topk_ids[start:end],
                is_ep=False,
                routed_scaling_factor=routed_scaling_factor,
                outputs=outputs[start:end],
                output_add=None if output_add is None else output_add[start:end],
            )
            ready = torch.cuda.Event()
            ready.record(main_stream)
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_event(ready)
                shard = outputs[start:end]
                reduced = moe_tensor_model_parallel_all_reduce(shard)
                if reduced.data_ptr() != shard.data_ptr():
                    # Out-of-place reduce path (not expected at these sizes):
                    # fold the result back into the caller-visible view.
                    shard.copy_(reduced)
        done = torch.cuda.Event()
        done.record(comm_stream)
        main_stream.wait_event(done)

    def _run_grouped_contiguous_gemm(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        hidden_states_scale: torch.Tensor | None = None,
        apply_routed_scaling_factor: bool = True,
    ):
        configs = self.get_humming_gemm_configs(HummingGemmType.GROUPED_CONTIGUOUS)
        valid_shape_m = self.estimate_local_valid_shape_m(topk_ids)

        buffers = self.prepare_buffers(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            gemm_type=HummingGemmType.GROUPED_CONTIGUOUS,
        )

        is_ep = self.num_experts != self.global_num_experts
        if hidden_states_scale is None:
            hidden_states, src2dst, expert_first_token_offset = moe_permute(
                inputs=hidden_states,
                topk_ids=topk_ids,
                num_experts=self.num_experts,
                is_ep=is_ep,
            )
        else:
            hidden_states_scale = hidden_states_scale.reshape(
                -1, hidden_states_scale.size(-1)
            ).contiguous()
            (
                hidden_states,
                hidden_states_scale,
                src2dst,
                expert_first_token_offset,
            ) = moe_permute_with_scale(
                inputs=hidden_states,
                input_scale=hidden_states_scale,
                topk_ids=topk_ids,
                num_experts=self.num_experts,
                is_ep=is_ep,
                outputs=buffers["quanted_gate_up_input"],
            )

        inputs, input_scale = HummingMethod.may_quant_input(
            layer=self.layer,
            inputs=hidden_states,
            input_scale=hidden_states_scale,
            quanted_input=buffers.get("quanted_gate_up_input", None),
            sublayer_name="w13",
        )

        HummingMethod.forward_layer(
            layer=self.layer,
            inputs=inputs,
            input_scale=input_scale,
            outputs=buffers["gate_up_output"],
            valid_shape_m=valid_shape_m,
            expert_layout=expert_first_token_offset,
            compute_config=configs["compute_config_str"],
            tuning_config=configs["w13_tuning_config_str"],
            sublayer_name="w13",
        )

        self.apply_activation(
            inputs=buffers["gate_up_output"],
            outputs=buffers["activation_output"],
        )

        inputs, input_scale = HummingMethod.may_quant_input(
            layer=self.layer,
            inputs=buffers["activation_output"],
            quanted_input=buffers.get("quanted_down_input", None),
            sublayer_name="w2",
        )

        HummingMethod.forward_layer(
            layer=self.layer,
            inputs=inputs,
            input_scale=input_scale,
            outputs=buffers["down_output"],
            valid_shape_m=valid_shape_m,
            expert_layout=expert_first_token_offset,
            compute_config=configs["compute_config_str"],
            tuning_config=configs["w2_tuning_config_str"],
            sublayer_name="w2",
        )

        routed_scaling_factor = (
            self.config.routed_scaling_factor if apply_routed_scaling_factor else None
        )
        moe_unpermute(
            outputs=buffers["output"],
            inputs=buffers["down_output"],
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            src2dst=src2dst,
            routed_scaling_factor=routed_scaling_factor,
        )

        return buffers["output"]

    def _run_grouped_masked_gemm(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        expert_num_tokens: torch.Tensor,
        expected_m: int,
        hidden_states_scale: torch.Tensor | None = None,
    ):
        configs = self.get_humming_gemm_configs(HummingGemmType.GROUPED_MASKED)
        valid_shape_m = self.estimate_local_valid_shape_m(topk_ids, expected_m)
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        if hidden_states_scale is not None:
            hidden_states_scale = hidden_states_scale.reshape(
                -1, hidden_states_scale.size(-1)
            ).contiguous()

        buffers = self.prepare_buffers(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            gemm_type=HummingGemmType.GROUPED_MASKED,
        )

        inputs, input_scale = HummingMethod.may_quant_input(
            layer=self.layer,
            inputs=hidden_states,
            input_scale=hidden_states_scale,
            quanted_input=buffers.get("quanted_gate_up_input", None),
            sublayer_name="w13",
        )

        HummingMethod.forward_layer(
            layer=self.layer,
            inputs=inputs,
            input_scale=input_scale,
            outputs=buffers["gate_up_output"],
            valid_shape_m=valid_shape_m,
            expert_layout=expert_num_tokens,
            compute_config=configs["compute_config_str"],
            tuning_config=configs["w13_tuning_config_str"],
            sublayer_name="w13",
        )

        gate_up = buffers["gate_up_output"].view(
            expert_num_tokens.shape[0],
            -1,
            buffers["gate_up_output"].shape[-1],
        )
        down_input, down_scale = self._grouped_masked_act_quant(
            gate_up, expert_num_tokens, buffers
        )

        HummingMethod.forward_layer(
            layer=self.layer,
            inputs=down_input,
            input_scale=down_scale,
            outputs=buffers["down_output"].view(-1, hidden_states.size(-1)),
            valid_shape_m=valid_shape_m,
            expert_layout=expert_num_tokens,
            compute_config=configs["compute_config_str"],
            tuning_config=configs["w2_tuning_config_str"],
            sublayer_name="w2",
        )

        return buffers["down_output"]


def _validate_deepep_dispatch_input(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor | None,
    layer: torch.nn.Module,
) -> None:
    expects_fp8 = layer._humming_uses_deepep_fp8_dispatch
    is_fp8 = hidden_states.dtype == torch.float8_e4m3fn
    if not expects_fp8:
        if is_fp8 or hidden_states_scale is not None:
            raise ValueError(
                "DeepEP returned FP8 input while Humming is configured for BF16 "
                "dispatch."
            )
        return

    if not is_fp8 or hidden_states_scale is None:
        raise ValueError(
            "Humming expected DeepEP FP8 hidden states and group-128 scales."
        )

    expected_groups, remainder = divmod(hidden_states.size(-1), 128)
    if (
        remainder != 0
        or hidden_states_scale.dtype != torch.float32
        or hidden_states_scale.size(-1) != expected_groups
        or hidden_states_scale.numel() != hidden_states.numel() // 128
    ):
        raise ValueError(
            "Humming requires row-major FP32 DeepEP scales with group size 128."
        )
    meta = layer.humming_metas["w13"]
    if meta.a_dtype != dtypes.float8e4m3 or meta.input_scale_group_size != 128:
        raise ValueError("Humming w13 must use FP8 group-128 input metadata.")


@register_pre_permute("deepep_ll", "humming")
def pre_permute_deepep_ll_to_humming(
    dispatch_output: DeepEPLLDispatchOutput,
    quant_info: HummingMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> HummingRunnerInput:
    hidden_states = dispatch_output.hidden_states
    topk_ids = dispatch_output.topk_ids
    topk_weights = dispatch_output.topk_weights
    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights
    _validate_deepep_dispatch_input(
        hidden_states, dispatch_output.hidden_states_scale, quant_info.layer
    )

    return HummingRunnerInput(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids.int(),
        expert_num_tokens=dispatch_output.masked_m,
        expected_m=dispatch_output.expected_m,
        gemm_type=HummingGemmType.GROUPED_MASKED,
        hidden_states_scale=dispatch_output.hidden_states_scale,
        apply_routed_scaling_factor=False,
    )


@register_post_permute("humming", "deepep_ll")
def post_permute_humming_to_deepep_ll(
    runner_output: HummingRunnerOutput,
    quant_info: HummingMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPLLCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput

    # Pass raw topk_weights to the DeepEP LL combine (matching the deep_gemm
    # runner): the model applies routed_scaling_factor once on the combined
    # output (e.g. DeepSeek's op_output does so unconditionally on the EP
    # path). Pre-scaling here would double-apply it (s^2 on the routed branch)
    return DeepEPLLCombineInput(
        hidden_states=runner_output.hidden_states,
        topk_ids=running_state["topk_ids"],
        topk_weights=running_state["topk_weights"],
    )


@register_pre_permute("deepep_normal", "humming")
def pre_permute_deepep_normal_to_humming(
    dispatch_output: DeepEPNormalDispatchOutput,
    quant_info: HummingMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> HummingRunnerInput:
    hidden_states = dispatch_output.hidden_states
    topk_ids = dispatch_output.topk_ids
    topk_weights = dispatch_output.topk_weights
    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights
    _validate_deepep_dispatch_input(
        hidden_states, dispatch_output.hidden_states_scale, quant_info.layer
    )

    return HummingRunnerInput(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids.int(),
        gemm_type=get_standard_humming_moe_gemm_type(),
        hidden_states_scale=dispatch_output.hidden_states_scale,
        apply_routed_scaling_factor=False,
    )


@register_post_permute("humming", "deepep_normal")
def post_permute_humming_to_deepep_normal(
    runner_output: HummingRunnerOutput,
    quant_info: HummingMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPNormalCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalCombineInput

    return DeepEPNormalCombineInput(
        hidden_states=runner_output.hidden_states,
        topk_ids=running_state["topk_ids"],
        topk_weights=running_state["topk_weights"],
    )


@register_pre_permute("standard", "humming")
def pre_permute_standard_to_humming(
    dispatch_output: StandardDispatchOutput,
    quant_info: HummingMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> HummingRunnerInput:
    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights

    return HummingRunnerInput(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids.int(),
        gemm_type=get_standard_humming_moe_gemm_type(),
        output_add=dispatch_output.output_add,
    )


@register_post_permute("humming", "standard")
def post_permute_humming_to_standard(
    runner_output: HummingRunnerOutput,
    quant_info: HummingMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    return StandardCombineInput(hidden_states=runner_output.hidden_states)
