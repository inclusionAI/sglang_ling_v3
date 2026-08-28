"""Unit coverage for Ling-3.0-plus MXFP4 served by the Humming MoE runner."""

from __future__ import annotations

import contextlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.models.bailing_moe_v3 import (  # noqa: E402
    BailingMoeV3ForCausalLM,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_EXPERT_PREFIX = "model.layers.0.mlp.experts"


class TestBailingMoeV3Mxfp4Humming(CustomTestCase):
    @staticmethod
    @contextlib.contextmanager
    def _humming_backend(*, megamoe: bool = False):
        """`--moe-runner-backend humming` as Mxfp4MoEMethod.__init__ reads it."""
        with (
            patch(
                "sglang.srt.layers.quantization.mxfp4.get_moe_runner_backend",
                return_value=SimpleNamespace(
                    is_triton_kernels=lambda: False,
                    is_flashinfer_mxfp4=lambda: False,
                    is_marlin=lambda: False,
                    is_humming=lambda: True,
                    is_deep_gemm=lambda: False,
                ),
            ),
            patch(
                "sglang.srt.layers.quantization.mxfp4.get_moe_a2a_backend",
                return_value=SimpleNamespace(is_megamoe=lambda: megamoe),
            ),
            # Without this the constructor dies on the unpublished config bag
            # instead, and the megamoe test would pass even with the guard gone.
            patch(
                "sglang.srt.layers.quantization.mxfp4.get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(flashinfer_mxfp4_moe_precision="default")
                ),
            ),
        ):
            yield

    def test_humming_rejects_megamoe_a2a_backend(self):
        """Selecting both backends must fail before any weight is allocated.

        The two come from independent CLI options. Allowed through, the layer
        gets MegaMoE weights while inference dispatches to Humming, which reads
        tensors that were never built.
        """
        from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

        with self._humming_backend(megamoe=True):
            with self.assertRaisesRegex(ValueError, "humming.*megamoe"):
                Mxfp4MoEMethod(prefix="layers.0.mlp.experts")

    def test_checkpoint_layout_reaches_humming_unpadded(self):
        """Ling's TP8 expert slice must keep the checkpoint's exact shapes.

        moe_intermediate_size 1280 / TP8 = 160 divides the MXFP4 group of 32
        but none of the alignments the other MXFP4 backends pad to, so a
        misplaced `elif` in the create_weights chain would round it to 192 --
        and `prepare_humming_moe_layer` would then re-lay-out rows that are
        not the checkpoint's, silently. Nothing downstream would complain.
        """
        from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

        num_local_experts, hidden_size, intermediate = 4, 4096, 160

        layer = torch.nn.Module()
        layer.num_local_experts = num_local_experts
        layer.hidden_size = hidden_size
        layer.intermediate_size_per_partition = intermediate

        with self._humming_backend():
            method = Mxfp4MoEMethod(prefix="layers.0.mlp.experts")
        method.create_weights(
            layer,
            num_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate,
            params_dtype=torch.bfloat16,
        )

        # e2m1 packs two values per byte; ue8m0 scales are one per group of 32.
        self.assertEqual(tuple(layer.w13_weight.shape), (num_local_experts, 320, 2048))
        self.assertEqual(
            tuple(layer.w13_weight_scale.shape), (num_local_experts, 320, 128)
        )
        self.assertEqual(tuple(layer.w2_weight.shape), (num_local_experts, 4096, 80))
        self.assertEqual(
            tuple(layer.w2_weight_scale.shape), (num_local_experts, 4096, 5)
        )

        # humming_utils imports the closed-source `humming` package at module
        # scope, so stand in for the whole module rather than the one function.
        prepare = MagicMock()
        fake_humming_utils = types.ModuleType(
            "sglang.srt.layers.quantization.humming_utils"
        )
        fake_humming_utils.prepare_humming_moe_layer = prepare
        with patch.dict(
            sys.modules,
            {"sglang.srt.layers.quantization.humming_utils": fake_humming_utils},
        ):
            method.process_weights_after_loading(layer)

        prepare.assert_called_once_with(layer, {"quant_method": "mxfp4"})
        self.assertEqual(layer._mxfp4_backend, "humming")

    @staticmethod
    def _make_model():
        model = BailingMoeV3ForCausalLM.__new__(BailingMoeV3ForCausalLM)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(
            num_experts=1,
            num_hidden_layers=1,
            tie_word_embeddings=False,
            q_lora_rank=None,
        )
        model.model = SimpleNamespace(start_layer=0, end_layer=1, layer_group_size=1)
        model.num_fused_shared_experts = 0
        return model

    @staticmethod
    def _checkpoint_weights(weight_suffix: str = "weight_packed"):
        prefix = f"{_EXPERT_PREFIX}.0"
        return [
            (f"{prefix}.gate_proj.{weight_suffix}", torch.empty(1)),
            (f"{prefix}.up_proj.{weight_suffix}", torch.empty(1)),
            (f"{prefix}.down_proj.{weight_suffix}", torch.empty(1)),
            (f"{prefix}.gate_proj.weight_scale", torch.empty(1)),
            (f"{prefix}.up_proj.weight_scale", torch.empty(1)),
            (f"{prefix}.down_proj.weight_scale", torch.empty(1)),
        ]

    def _load(self, model, parameter_names, weights):
        parameters = {
            name: SimpleNamespace(weight_loader=MagicMock()) for name in parameter_names
        }
        with (
            patch.object(model, "named_parameters", return_value=parameters.items()),
            patch.object(model, "post_load_weights") as post_load,
        ):
            model.load_weights(weights)
        post_load.assert_called_once()
        return parameters

    def test_dual_expert_schema_resolution(self):
        """A packed MXFP4 checkpoint has to land on both MoE param schemas.

        `--quantization humming` registers the checkpoint-native
        `*_weight_packed` names, while Mxfp4MoEMethod (compressed-tensors
        auto-detection plus `--moe-runner-backend humming`) registers the same
        fused parameter without the suffix.
        """
        schemas = {
            "humming-packed": (
                f"{_EXPERT_PREFIX}.w13_weight_packed",
                f"{_EXPERT_PREFIX}.w13_weight_scale",
                f"{_EXPERT_PREFIX}.w2_weight_packed",
                f"{_EXPERT_PREFIX}.w2_weight_scale",
            ),
            "native-mxfp4": (
                f"{_EXPERT_PREFIX}.w13_weight",
                f"{_EXPERT_PREFIX}.w13_weight_scale",
                f"{_EXPERT_PREFIX}.w2_weight",
                f"{_EXPERT_PREFIX}.w2_weight_scale",
            ),
        }
        # gate_proj + up_proj both shard into w13; down_proj alone into w2.
        expected_call_counts = (2, 2, 1, 1)

        for schema, parameter_names in schemas.items():
            with self.subTest(schema=schema):
                model = self._make_model()
                parameters = self._load(
                    model, parameter_names, self._checkpoint_weights()
                )
                for name, expected_count in zip(parameter_names, expected_call_counts):
                    loader = parameters[name].weight_loader
                    self.assertEqual(loader.call_count, expected_count)
                    for call in loader.call_args_list:
                        self.assertEqual(call.args[2], name)

    def test_unpacked_checkpoint_still_loads(self):
        """A plain `.weight` checkpoint must not be perturbed by the alias."""
        model = self._make_model()
        parameter_names = (
            f"{_EXPERT_PREFIX}.w13_weight",
            f"{_EXPERT_PREFIX}.w2_weight",
        )
        parameters = self._load(
            model,
            parameter_names,
            [
                (f"{_EXPERT_PREFIX}.0.gate_proj.weight", torch.empty(1)),
                (f"{_EXPERT_PREFIX}.0.up_proj.weight", torch.empty(1)),
                (f"{_EXPERT_PREFIX}.0.down_proj.weight", torch.empty(1)),
            ],
        )
        self.assertEqual(parameters[parameter_names[0]].weight_loader.call_count, 2)
        self.assertEqual(parameters[parameter_names[1]].weight_loader.call_count, 1)

    def test_unmappable_packed_expert_is_loud(self):
        """A packed tensor matching no schema would otherwise serve garbage."""
        model = self._make_model()
        with self.assertRaisesRegex(RuntimeError, "gate_proj.weight_packed"):
            self._load(
                model,
                (
                    # Present, so the layer is local -- but the w13 half is
                    # registered under neither the packed nor the plain name.
                    f"{_EXPERT_PREFIX}.w2_weight",
                    f"{_EXPERT_PREFIX}.w13_weight_scale",
                ),
                [(f"{_EXPERT_PREFIX}.0.gate_proj.weight_packed", torch.empty(1))],
            )

    def test_absent_layer_skips_quietly(self):
        """A pipeline rank that owns no expert parameter must not raise."""
        model = self._make_model()
        parameters = self._load(
            model,
            ("model.layers.0.input_layernorm.weight",),
            [(f"{_EXPERT_PREFIX}.0.gate_proj.weight_packed", torch.empty(1))],
        )
        parameters[
            "model.layers.0.input_layernorm.weight"
        ].weight_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
