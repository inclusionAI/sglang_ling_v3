"""Regression guards for gated MLA in the ROCm absorb path."""

import ast
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


SOURCE = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py"
)


def _self_call(node: ast.AST, method_name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == method_name
    )


class TestRocmGatedMlaContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse(SOURCE.read_text())
        mixin = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "DeepseekMLARocmForwardMixin"
        )
        cls.method = next(
            node
            for node in mixin.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "forward_absorb_rocm_core"
        )

    def test_rocm_core_accepts_attention_output_gate(self):
        args = [arg.arg for arg in self.method.args.args]
        self.assertIn("attention_output_gate", args)
        gate_index = args.index("attention_output_gate")
        default_index = gate_index - (len(args) - len(self.method.args.defaults))
        self.assertIsInstance(self.method.args.defaults[default_index], ast.Constant)
        self.assertIsNone(self.method.args.defaults[default_index].value)

    def test_gate_is_applied_before_output_projection(self):
        gate_if = next(
            node
            for node in ast.walk(self.method)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "attention_output_gate"
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.IsNot)
            and len(node.test.comparators) == 1
            and isinstance(node.test.comparators[0], ast.Constant)
            and node.test.comparators[0].value is None
        )
        apply_assign = next(
            node
            for node in gate_if.body
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "_apply_attention_output_gate"
        )
        self.assertEqual(
            [
                target.id
                for target in apply_assign.targets
                if isinstance(target, ast.Name)
            ],
            ["attn_bmm_output"],
        )
        self.assertEqual(
            [arg.id for arg in apply_assign.value.args if isinstance(arg, ast.Name)],
            ["self", "attn_bmm_output", "attention_output_gate"],
        )
        o_proj = next(
            node for node in ast.walk(self.method) if _self_call(node, "o_proj")
        )
        self.assertLess(apply_assign.lineno, o_proj.lineno)


if __name__ == "__main__":
    unittest.main()
