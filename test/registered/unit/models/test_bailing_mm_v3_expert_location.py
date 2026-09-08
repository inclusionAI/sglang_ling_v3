"""Dependency-light contract test for Bailing V3 VL expert topology."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace


class TestBailingMoeV3VLExpertLocation(unittest.TestCase):
    def test_delegates_expert_location_config_to_language_backbone(self):
        model_path = (
            Path(__file__).parents[4]
            / "python/sglang/srt/models/bailing_mm_v3.py"
        )
        tree = ast.parse(model_path.read_text())
        wrapper = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "BailingMoeV3VLForConditionalGeneration"
        )
        method = next(
            node
            for node in wrapper.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "get_model_config_for_expert_location"
        )

        delegated = object()
        llm_config = SimpleNamespace()
        test_case = self

        class LanguageBackbone:
            @classmethod
            def get_model_config_for_expert_location(cls, config):
                test_case.assertIs(config, llm_config)
                return delegated

        synthetic_wrapper = ast.ClassDef(
            name="Wrapper",
            bases=[],
            keywords=[],
            body=[method],
            decorator_list=[],
        )
        module = ast.fix_missing_locations(
            ast.Module(body=[synthetic_wrapper], type_ignores=[])
        )
        namespace = {"BailingMoeV3ForCausalLM": LanguageBackbone}
        exec(compile(module, str(model_path), "exec"), namespace)

        result = namespace["Wrapper"].get_model_config_for_expert_location(
            SimpleNamespace(llm_config=llm_config)
        )
        self.assertIs(result, delegated)


if __name__ == "__main__":
    unittest.main()
