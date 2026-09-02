"""SGLANG_HUMMING_TUNING_OVERRIDE replaces a sublayer's humming bucket list.

The override carries a measured tuning sweep into serving without a humming
release. It must be a drop-in for the heuristic output (same [min_m, max_m,
config] buckets) and must leave every (gemm type, sublayer) it does not name
untouched.
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

import json
import unittest

from sglang.srt.environ import envs
from sglang.test.test_utils import CustomTestCase

try:
    from sglang.srt.layers.moe.moe_runner.humming import (
        HummingGemmType,
        HummingRunnerCore,
    )

    _HAS_HUMMING = True
except ImportError:
    _HAS_HUMMING = False


@unittest.skipUnless(_HAS_HUMMING, "requires the humming package")
class TestHummingTuningOverride(CustomTestCase):
    DEFAULT = [
        (0, 4096, {"block_shape": [8, 128, 128]}),
        (4096, 1 << 30, {"block_shape": [64, 128, 128]}),
    ]
    OVERRIDE = {"indexed": {"w13": [[4096, 1 << 30, {"block_shape": [64, 128, 64]}]]}}

    def test_unset_keeps_heuristic(self):
        with envs.SGLANG_HUMMING_TUNING_OVERRIDE.override(None):
            got = HummingRunnerCore._apply_tuning_override(
                humming_gemm_type=HummingGemmType.INDEXED,
                sublayer_name="w13",
                default=self.DEFAULT,
            )
        self.assertIs(got, self.DEFAULT)

    def test_named_bucket_is_replaced_and_others_kept(self):
        with envs.SGLANG_HUMMING_TUNING_OVERRIDE.override(json.dumps(self.OVERRIDE)):
            got = HummingRunnerCore._apply_tuning_override(
                humming_gemm_type=HummingGemmType.INDEXED,
                sublayer_name="w13",
                default=self.DEFAULT,
            )
        self.assertEqual(
            got,
            [
                (0, 4096, {"block_shape": [8, 128, 128]}),
                (4096, 1 << 30, {"block_shape": [64, 128, 64]}),
            ],
        )

    def test_unknown_range_is_rejected(self):
        bad = {"indexed": {"w13": [[0, 1 << 30, {"block_shape": [64, 128, 64]}]]}}
        with envs.SGLANG_HUMMING_TUNING_OVERRIDE.override(json.dumps(bad)):
            with self.assertRaises(ValueError):
                HummingRunnerCore._apply_tuning_override(
                    humming_gemm_type=HummingGemmType.INDEXED,
                    sublayer_name="w13",
                    default=self.DEFAULT,
                )

    def test_other_sublayer_and_gemm_type_untouched(self):
        with envs.SGLANG_HUMMING_TUNING_OVERRIDE.override(json.dumps(self.OVERRIDE)):
            w2 = HummingRunnerCore._apply_tuning_override(
                humming_gemm_type=HummingGemmType.INDEXED,
                sublayer_name="w2",
                default=self.DEFAULT,
            )
            grouped = HummingRunnerCore._apply_tuning_override(
                humming_gemm_type=HummingGemmType.GROUPED_CONTIGUOUS,
                sublayer_name="w13",
                default=self.DEFAULT,
            )
        self.assertIs(w2, self.DEFAULT)
        self.assertIs(grouped, self.DEFAULT)


if __name__ == "__main__":
    unittest.main()
