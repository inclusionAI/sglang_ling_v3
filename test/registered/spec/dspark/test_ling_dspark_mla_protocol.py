import unittest
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    BuildOutTokens,
)
from sglang.srt.configs.bailing_hybrid import BailingHybridConfig
from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.environ import envs
from sglang.srt.layers.attention.flashinfer_mla_backend import (
    _use_causal_mla_prefill,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.models.bailing_moe_v3 import (
    BailingMoELinearDecoderLayer,
    BailingMoELinearModel,
    resolve_bailing_mla_rope,
)
from sglang.srt.models.dspark import VanillaMarkov
from sglang.srt.models.ling_v3_dspark import LingV3ForCausalLMDSparkMLA
from sglang.srt.speculative.dspark_components.dspark_config import (
    apply_namespaced_dspark_draft_config,
    dspark_draft_is_moe,
    parse_dspark_draft_config,
)
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)
from sglang.srt.speculative.dspark_components.dspark_planner import VerifyWindow
from sglang.srt.speculative.dspark_components.dspark_verify import (
    AcceptGreedy,
    DsparkVerifyEpilogue,
    FinalizeAcceptLens,
    TargetVerifyExecutor,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _run_ling_mla_post_load(attention):
    draft = SimpleNamespace(
        layers=[SimpleNamespace(attention=attention)],
        quant_config=None,
    )
    LingV3ForCausalLMDSparkMLA._post_load_mla_weights(draft)


class TestLingDsparkMlaProtocol(CustomTestCase):
    def test_static_folded_tail_stages_replay_inputs(self):
        epilogue = DsparkVerifyEpilogue(
            max_bs=4,
            verify_num_draft_tokens=3,
            device="cpu",
            compact_mode=False,
        )
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        seq_lens = torch.tensor([100, 200])
        req_pool_indices = torch.tensor([7, 8])

        epilogue.begin_step(
            None,
            armed=True,
            bs=2,
            input_ids=input_ids,
            seq_lens=seq_lens,
            req_pool_indices=req_pool_indices,
        )

        torch.testing.assert_close(
            epilogue.static_input_ids_buf[:6], input_ids.view(-1)
        )
        torch.testing.assert_close(epilogue.static_seq_lens_buf[:2], seq_lens)
        torch.testing.assert_close(
            epilogue.static_req_pool_indices_buf[:2], req_pool_indices
        )
        torch.testing.assert_close(epilogue.verify_lens_buf[:2], torch.full((2,), 3))
        torch.testing.assert_close(
            epilogue.verify_lens_buf[2:],
            torch.zeros_like(epilogue.verify_lens_buf[2:]),
        )

    def test_folded_result_copy_event_is_consumed_once_at_reuse(self):
        epilogue = DsparkVerifyEpilogue(
            max_bs=4,
            verify_num_draft_tokens=3,
            device="cpu",
            compact_mode=False,
        )
        event = Mock()
        stream = Mock()
        device_module = Mock()
        device_module.Event.return_value = event
        device_module.current_stream.return_value = stream

        with patch("torch.get_device_module", return_value=device_module):
            epilogue.record_result_copy_done()
            event.record.assert_called_once_with()
            with self.assertRaisesRegex(RuntimeError, "copied again"):
                epilogue.record_result_copy_done()

            epilogue.wait_for_result_copy_done()
            stream.wait_event.assert_called_once_with(event)
            epilogue.wait_for_result_copy_done()
            self.assertEqual(stream.wait_event.call_count, 1)

            epilogue.record_result_copy_done()
            epilogue.wait_for_result_copy_done()
            device_module.Event.assert_called_once_with()
            self.assertEqual(event.record.call_count, 2)
            self.assertEqual(stream.wait_event.call_count, 2)

    def test_target_verify_waits_at_persistent_result_reuse_point(self):
        order = []
        executor = TargetVerifyExecutor.__new__(TargetVerifyExecutor)
        executor.verify_epilogue = SimpleNamespace(
            wait_for_result_copy_done=lambda: order.append("wait")
        )
        executor.target_worker = SimpleNamespace(
            forward_batch_generation=lambda **kwargs: (
                order.append("forward")
                or SimpleNamespace(logits_output=Mock(), can_run_cuda_graph=True)
            )
        )
        verify_input = SimpleNamespace(
            live_seq_lens_cpu=None,
            prepare_for_verify=lambda batch, worker: (
                order.append("prepare") or Mock(),
                None,
            ),
        )
        batch = SimpleNamespace(seq_lens_cpu=torch.tensor([8]), seq_lens_sum=8)
        executor._target_is_dsv41 = False

        executor._forward_prepared_verify(
            batch=batch,
            verify_input=verify_input,
            seq_lens_cpu_backup=batch.seq_lens_cpu,
            seq_lens_sum_backup=batch.seq_lens_sum,
        )

        self.assertEqual(order, ["prepare", "wait", "forward"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_delayed_folded_result_copy_finishes_before_buffer_reuse(self):
        device = torch.device("cuda")
        epilogue = DsparkVerifyEpilogue(
            max_bs=2,
            verify_num_draft_tokens=3,
            device=device,
            compact_mode=False,
        )
        old_tokens = torch.tensor(
            [[11, 12, 13], [21, 22, 23]], dtype=torch.int64, device=device
        )
        old_lens = torch.tensor([2, 3], dtype=torch.int32, device=device)
        epilogue.out_tokens_buf.copy_(old_tokens)
        epilogue.commit_lens_buf.copy_(old_lens)

        result = GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=epilogue.out_tokens_buf.reshape(-1),
            accept_lens=epilogue.commit_lens_buf,
            copy_done=torch.cuda.Event(),
            persistent_result_copy_owner=epilogue,
        )
        forward_stream = torch.cuda.current_stream(device)
        copy_stream = torch.cuda.Stream(device=device)
        copy_stream.wait_stream(forward_stream)
        with torch.cuda.stream(copy_stream):
            torch.cuda._sleep(20_000_000)
            result.copy_to_cpu(return_logprob=False, return_hidden_states=False)

        epilogue.wait_for_result_copy_done()
        epilogue.out_tokens_buf.fill_(99)
        epilogue.commit_lens_buf.fill_(1)
        torch.cuda.synchronize(device)

        torch.testing.assert_close(result.next_token_ids, old_tokens.cpu().reshape(-1))
        torch.testing.assert_close(result.accept_lens, old_lens.cpu())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_folded_accept_matches_eager_oracle_for_static_and_compact(self):
        device = torch.device("cuda")
        stride = 6
        gamma = stride - 1
        vocab = 32
        candidates = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [13, 14, 15, 16, 17, 18],
                [19, 20, 21, 22, 23, 24],
            ],
            dtype=torch.int64,
            device=device,
        )
        # Exercise leading accept lengths 0, 1, 3, and gamma.
        target_predict = torch.tensor(
            [
                [30, 3, 4, 5, 6, 7],
                [8, 31, 10, 11, 12, 13],
                [14, 15, 16, 30, 18, 19],
                [20, 21, 22, 23, 24, 25],
            ],
            dtype=torch.int64,
            device=device,
        )
        target_logits = torch.full(
            (candidates.numel(), vocab),
            -100.0,
            dtype=torch.float32,
            device=device,
        )
        target_logits.scatter_(1, target_predict.reshape(-1, 1), 100.0)
        prefix_lens = torch.tensor(
            [100, 200, 300, 400], dtype=torch.int64, device=device
        )
        draft_tokens = candidates[:, 1:].contiguous()

        def eager_oracle(oracle_candidates, logits, prefixes, cutoff=None):
            correct_len, bonus, cap_trim_lens = AcceptGreedy.execute(
                candidates=oracle_candidates,
                target_logits=logits,
                verify_num_draft_tokens=stride,
                cutoff_verify_lens=cutoff,
            )
            finalized = FinalizeAcceptLens.execute(
                correct_len=correct_len,
                cap_trim_lens=cap_trim_lens,
                prefix_lens=prefixes,
            )
            out_tokens = BuildOutTokens.execute(
                draft_tokens=draft_tokens[: oracle_candidates.shape[0]],
                correct_len=correct_len,
                bonus=bonus,
                verify_num_draft_tokens=stride,
                gamma=gamma,
            )
            return (
                correct_len,
                bonus,
                finalized.cap_trim_lens,
                finalized.commit_lens,
                finalized.new_seq_lens,
                out_tokens.view(oracle_candidates.shape[0], stride),
            )

        def assert_result(result, expected):
            actual = (
                result.correct_len,
                result.bonus,
                result.cap_trim_lens,
                result.commit_lens,
                result.new_seq_lens,
                result.out_tokens,
            )
            for got, want in zip(actual, expected):
                torch.testing.assert_close(got, want, rtol=0, atol=0)

        static = DsparkVerifyEpilogue(
            max_bs=4,
            verify_num_draft_tokens=stride,
            device=device,
            compact_mode=False,
        )
        static.strided_logits = target_logits
        static.draft_tokens_buf.copy_(draft_tokens.reshape(-1))
        static._accept(candidates.reshape(-1), prefix_lens, None, bs=4)
        assert_result(
            static.read_accept(4),
            eager_oracle(candidates, target_logits, prefix_lens),
        )

        # Reuse the BS4 bucket with a smaller live batch to catch stale rows.
        static.correct_len_buf[3] = 99
        static._accept(candidates[:3].reshape(-1), prefix_lens[:3], None, bs=3)
        assert_result(
            static.read_accept(3),
            eager_oracle(candidates[:3], target_logits[: 3 * stride], prefix_lens[:3]),
        )

        compact = DsparkVerifyEpilogue(
            max_bs=4,
            verify_num_draft_tokens=stride,
            device=device,
            compact_mode=True,
        )
        widths = (1, 2, 4, 6)
        verify_lens = torch.tensor(widths, dtype=torch.int64, device=device)
        compact_candidates = torch.cat(
            [candidates[row, :width] for row, width in enumerate(widths)]
        )
        padded_candidates = torch.zeros_like(candidates)
        for row, width in enumerate(widths):
            padded_candidates[row, :width] = candidates[row, :width]
        compact.strided_logits = target_logits
        compact.draft_tokens_buf.copy_(draft_tokens.reshape(-1))
        compact._accept(compact_candidates, prefix_lens, verify_lens, bs=4)
        assert_result(
            compact.read_accept(4),
            eager_oracle(
                padded_candidates,
                target_logits,
                prefix_lens,
                cutoff=verify_lens,
            ),
        )

    def test_read_accept_snapshots_cross_replay_values(self):
        epilogue = DsparkVerifyEpilogue(
            max_bs=4,
            verify_num_draft_tokens=3,
            device="cpu",
            compact_mode=False,
        )
        epilogue.bonus_buf[:2].copy_(torch.tensor([11, 22]))
        epilogue.new_seq_lens_buf[:2].copy_(torch.tensor([101, 202]))

        accepted = epilogue.read_accept(2)
        self.assertNotEqual(accepted.bonus.data_ptr(), epilogue.bonus_buf.data_ptr())
        self.assertNotEqual(
            accepted.new_seq_lens.data_ptr(), epilogue.new_seq_lens_buf.data_ptr()
        )

        # Model the next graph replay overwriting its persistent output buffers.
        epilogue.bonus_buf[:2].fill_(-1)
        epilogue.new_seq_lens_buf[:2].fill_(-1)
        torch.testing.assert_close(accepted.bonus, torch.tensor([11, 22]))
        torch.testing.assert_close(accepted.new_seq_lens, torch.tensor([101, 202]))

    def test_capture_completes_pending_mlp_allreduce(self):
        # On the current stage-boundary API, the next attention boundary
        # completes the preceding FFN contribution before capturing it.
        captured = []

        class FakeAttnBoundary:
            def prepare(self, hidden, forward_batch, *, capture=None):
                if capture is not None:
                    capture(hidden + 10, owned=True)
                return hidden

            def finish(self, hidden, forward_batch, *, already_reduced=False):
                return hidden

        class FakeFfnBoundary:
            def prepare(self, hidden, forward_batch):
                return hidden

            def exit(self, forward_batch):
                return nullcontext(SimpleNamespace(finish=lambda hidden: hidden))

        class FakeMlp(nn.Module):
            def forward(self, hidden, forward_batch):
                return hidden * 2

        layer = BailingMoELinearDecoderLayer.__new__(BailingMoELinearDecoderLayer)
        nn.Module.__init__(layer)
        layer.attn_boundary = FakeAttnBoundary()
        layer.ffn_boundary = FakeFfnBoundary()
        layer.mlp = FakeMlp()
        layer.attention_type = 1
        layer.use_mla = True
        layer.is_layer_sparse = True
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: True)
        )
        with patch(
            "sglang.srt.models.bailing_moe_v3.is_dense_ffn_fully_dp",
            return_value=False,
        ):
            completed = layer(
                hidden_states=torch.ones((1, 2)),
                positions=torch.zeros(1, dtype=torch.long),
                forward_batch=forward_batch,
                zero_allocator=None,
                capture_output=lambda value, **kwargs: captured.append(value),
            )

        torch.testing.assert_close(captured[0], torch.full((1, 2), 11.0))
        torch.testing.assert_close(completed, torch.full((1, 2), 2.0))

    def test_capture_skips_dp_scattered_empty_output(self):
        # Regression: under DP attention a captured layer's input (attention
        # runs at TP_ATTN_FULL) can be non-empty while its SCATTERED output is
        # empty on this rank. The capture decision must key on the completed
        # output, not the pre-layer input, so a 0-row output is not appended
        # into the aux list -- downstream concatenates it feature-wise and so
        # requires a shared row count. Before the fix the 0-row output was
        # appended and the aux was returned; after the fix the aux stays empty
        # and forward returns the plain hidden state.
        class _ProjectionStub(nn.Module):
            def __init__(self, out_rows):
                super().__init__()
                self.out_rows = out_rows
                self.capture_after_layer_seen = None

            def forward(
                self,
                *,
                hidden_states,
                positions,
                forward_batch,
                zero_allocator,
                capture_output,
            ):
                self.capture_after_layer_seen = capture_output is not None
                out = hidden_states.new_zeros((self.out_rows, hidden_states.shape[-1]))
                if capture_output is not None:
                    capture_output(out)
                return out

        passthrough = _ProjectionStub(out_rows=4)  # layer 0, non-captured
        captured = _ProjectionStub(out_rows=0)  # layer 1, captured, empty output

        model = BailingMoELinearModel.__new__(BailingMoELinearModel)
        nn.Module.__init__(model)
        model.start_layer = 0
        model.end_layer = 2
        model.layers = nn.ModuleList([passthrough, captured])
        model.layers_to_capture = [0]
        model.capture_aux_hidden_states = True
        model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
        model.norm = nn.Identity()

        forward_batch = SimpleNamespace(
            can_run_tbo=False,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            forward_mode=SimpleNamespace(is_idle=lambda: False),
        )
        with patch("sglang.srt.models.bailing_moe_v3.residual_batch") as residual:
            residual.complete_output.side_effect = lambda hidden, batch: hidden
            residual.final_norm.side_effect = lambda hidden, batch, norm, capture=None: (
                hidden
            )
            result = model.forward(
                input_ids=None,
                positions=torch.zeros(4, dtype=torch.long),
                forward_batch=forward_batch,
                input_embeds=torch.zeros((4, 3)),
            )

        # No 0-row capture appended: the aux list is empty, so forward returns
        # the plain hidden state rather than a (hidden, aux) tuple.
        self.assertIsInstance(result, torch.Tensor)
        # The captured layer still had its MLP all-reduce forced to complete
        # (the partial-output fix), regardless of the empty output below.
        self.assertTrue(captured.capture_after_layer_seen)
        self.assertFalse(passthrough.capture_after_layer_seen)

    def test_flashinfer_mla_noncausal_is_dspark_draft_worker_only(self):
        draft = SimpleNamespace(
            is_draft_worker=True,
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
        )
        target = SimpleNamespace(
            is_draft_worker=False,
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
        )
        regular = SimpleNamespace(
            is_draft_worker=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        self.assertFalse(_use_causal_mla_prefill(draft))
        self.assertTrue(_use_causal_mla_prefill(target))
        self.assertTrue(_use_causal_mla_prefill(regular))

    def test_draft_block_visibility_matches_training_mask(self):
        prefix_len = 7
        gamma = 5
        anchor_position = prefix_len
        anchor_token = 42
        mask_token = 99
        draft_input = torch.tensor([anchor_token] + [mask_token] * (gamma - 1))
        draft_positions = anchor_position + torch.arange(gamma)
        prefix_positions = torch.arange(prefix_len)

        torch.testing.assert_close(
            draft_input,
            torch.tensor(
                [anchor_token, mask_token, mask_token, mask_token, mask_token]
            ),
        )
        torch.testing.assert_close(
            draft_positions, anchor_position + torch.arange(gamma)
        )
        self.assertEqual(int(prefix_positions.max()), anchor_position - 1)

        context_allow = prefix_positions.view(1, -1) < anchor_position
        block_allow = torch.ones((gamma, gamma), dtype=torch.bool)
        training_allow = torch.cat(
            [context_allow.expand(gamma, -1), block_allow], dim=1
        )
        inference_allow = torch.ones_like(training_allow)
        causal_allow = torch.arange(prefix_len + gamma).view(1, -1) <= (
            prefix_len + torch.arange(gamma)
        ).view(-1, 1)
        torch.testing.assert_close(inference_allow, training_allow)
        self.assertFalse(torch.equal(causal_allow, training_allow))

    def test_ling_mla_draft_layers_are_bidirectional(self):
        class FakeLayer(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.attention = SimpleNamespace(
                    attn_mqa=SimpleNamespace(attn_type=AttentionType.DECODER),
                    attn_mha=SimpleNamespace(attn_type=AttentionType.DECODER),
                )

        config = SimpleNamespace(
            target_layer_ids=[39, 40, 41],
            block_size=5,
            num_hidden_layers=3,
            hidden_size=4,
            rms_norm_eps=1e-6,
            router_dtype="fp32",
        )
        with (
            patch(
                "sglang.srt.models.ling_v3_dspark.BailingMoELinearDecoderLayer",
                FakeLayer,
            ),
            patch(
                "sglang.srt.models.ling_v3_dspark.build_markov_head",
                return_value=nn.Identity(),
            ),
            patch(
                "sglang.srt.models.ling_v3_dspark.build_confidence_head",
                return_value=None,
            ),
            patch(
                "sglang.srt.models.ling_v3_dspark.is_namespaced_dspark_bundle",
                return_value=False,
            ),
        ):
            model = LingV3ForCausalLMDSparkMLA(config)
        self.assertTrue(
            all(
                layer.attention.attn_mqa.attn_type == AttentionType.ENCODER_ONLY
                for layer in model.layers
            )
        )
        self.assertTrue(
            all(
                layer.attention.attn_mha.attn_type == AttentionType.ENCODER_ONLY
                for layer in model.layers
            )
        )
        self.assertTrue(model.sample_from_anchor)

    def test_default_rope_parameters_do_not_enable_yarn(self):
        rope_parameters = {"rope_theta": 6_000_000, "rope_type": "default"}
        rope_theta, rope_scaling = resolve_bailing_mla_rope(
            SimpleNamespace(rope_parameters=rope_parameters)
        )
        self.assertEqual(rope_theta, 6_000_000)
        self.assertIsNone(rope_scaling)
        self.assertEqual(rope_parameters["rope_type"], "default")

    def test_scaled_rope_parameters_are_preserved(self):
        rope_parameters = {
            "rope_theta": 10_000,
            "rope_type": "yarn",
            "factor": 4.0,
        }
        rope_theta, rope_scaling = resolve_bailing_mla_rope(
            SimpleNamespace(rope_parameters=rope_parameters)
        )
        self.assertEqual(rope_theta, 10_000)
        self.assertIs(rope_scaling, rope_parameters)

    def test_markov_recursive_matches_teacher_forcing(self):
        head = VanillaMarkov(vocab_size=4, markov_rank=4)
        with torch.no_grad():
            head.markov_w1.weight.copy_(torch.eye(4))
            head.markov_w2.weight.copy_(10 * torch.roll(torch.eye(4), 1, dims=0))
        base_logits = torch.zeros((1, 3, 4))
        proposal, corrected = head.sample_block(
            base_logits,
            first_prev_tokens=torch.tensor([0]),
            hidden_states=None,
            sampler=lambda logits, _: logits.argmax(dim=-1),
        )
        teacher_prev = torch.cat([torch.tensor([[0]]), proposal[:, :-1]], dim=1)
        teacher_logits = head.apply_block_logits(
            base_logits,
            token_ids=teacher_prev,
            hidden_states=None,
        )
        torch.testing.assert_close(corrected, teacher_logits)
        torch.testing.assert_close(proposal, teacher_logits.argmax(dim=-1))

    def test_namespaced_bundle_projects_draft_config(self):
        config = BailingHybridConfig(
            num_hidden_layers=42,
            dspark_bundle_format="namespaced_safetensors",
            dspark_bundle_version=1,
            dspark_config={
                "architectures": ["LingV3ForCausalLMDSparkMLA"],
                "attention_type": "mla",
                "num_hidden_layers": 3,
                "num_target_layers": 42,
                "target_layer_ids": [39, 40, 41],
                "block_size": 5,
                "markov_rank": 256,
                "markov_head_type": "vanilla",
                "mask_token_id": 156894,
                "layer_group_size": 1,
            },
        )
        parsed = parse_dspark_draft_config(draft_hf_config=config)
        self.assertEqual(parsed.num_hidden_layers, 3)
        self.assertEqual(parsed.target_layer_ids, [39, 40, 41])
        self.assertEqual(parsed.gamma, 5)

        self.assertTrue(apply_namespaced_dspark_draft_config(config))
        self.assertEqual(config.architectures, ["LingV3ForCausalLMDSparkMLA"])
        self.assertEqual(config.num_hidden_layers, 3)
        self.assertEqual(config.full_attention_type, "mla")
        self.assertEqual(config.attention_type, 1)

    def test_namespaced_bundle_uses_nested_arch_in_model_config(self):
        config = BailingHybridConfig(
            num_hidden_layers=42,
            dspark_bundle_format="namespaced_safetensors",
            dspark_bundle_version=1,
            dspark_config={
                "architectures": ["LingV3ForCausalLMDSparkMLA"],
                "attention_type": "mla",
                "num_hidden_layers": 3,
                "layer_group_size": 1,
            },
        )
        config.architectures = ["BailingMoeV3ForCausalLM"]
        model_config = object.__new__(ModelConfig)
        model_config.is_draft_model = True
        model_config.speculative_algorithm = "DSPARK"
        model_config.hf_config = config
        model_config.hf_text_config = config

        model_config._config_draft_model()

        self.assertEqual(
            model_config.hf_config.architectures,
            ["LingV3ForCausalLMDSparkMLA"],
        )
        self.assertEqual(model_config.hf_config.num_hidden_layers, 3)

    def test_standalone_draft_uses_nested_arch_in_model_config(self):
        with TemporaryDirectory() as draft_path:
            (Path(draft_path) / "draft_model.safetensors").touch()
            config = BailingHybridConfig(
                num_hidden_layers=72,
                dspark_config={
                    "architectures": ["LingV3ForCausalLMDSparkMLA"],
                    "attention_type": "mla",
                    "num_hidden_layers": 3,
                    "target_layer_ids": [69, 70, 71],
                    "block_size": 7,
                },
            )
            config.architectures = ["BailingMoeV3ForCausalLM"]
            config._name_or_path = draft_path
            model_config = object.__new__(ModelConfig)
            model_config.is_draft_model = True
            model_config.speculative_algorithm = "DSPARK"
            model_config.hf_config = config
            model_config.hf_text_config = config

            model_config._config_draft_model()

            self.assertEqual(
                model_config.hf_config.architectures,
                ["LingV3ForCausalLMDSparkMLA"],
            )
            self.assertEqual(model_config.hf_config.num_hidden_layers, 3)
            self.assertEqual(model_config.hf_config.target_layer_ids, [69, 70, 71])

    def test_namespaced_bundle_requires_draft_architecture(self):
        config = SimpleNamespace(
            dspark_bundle_format="namespaced_safetensors",
            dspark_bundle_version=1,
            dspark_config={"attention_type": "mla"},
        )
        with self.assertRaisesRegex(ValueError, "dspark_config.architectures"):
            apply_namespaced_dspark_draft_config(config)

    def test_namespaced_bundle_preserves_numeric_attention_type(self):
        config = SimpleNamespace(
            dspark_bundle_format="namespaced_safetensors",
            dspark_bundle_version=1,
            dspark_config={
                "architectures": ["SomeFutureDSparkModel"],
                "attention_type": 0,
            },
        )
        self.assertTrue(apply_namespaced_dspark_draft_config(config))
        self.assertEqual(config.attention_type, 0)
        self.assertFalse(hasattr(config, "full_attention_type"))

    def test_moe_detection_uses_draft_geometry(self):
        self.assertTrue(dspark_draft_is_moe(SimpleNamespace(num_experts=512)))
        self.assertTrue(dspark_draft_is_moe(SimpleNamespace(n_routed_experts=256)))
        self.assertFalse(dspark_draft_is_moe(SimpleNamespace(num_experts=1)))
        self.assertTrue(
            dspark_draft_is_moe(
                SimpleNamespace(num_experts=1, dspark_config={"num_experts": 8})
            )
        )
        self.assertFalse(
            dspark_draft_is_moe(
                SimpleNamespace(num_experts=512, dspark_config={"num_experts": 1})
            )
        )

    def test_native_mla_draft_is_not_classified_as_hybrid_linear(self):
        config = BailingHybridConfig(short_conv_kernel_size=4, num_hidden_layers=3)
        config.architectures = ["LingV3ForCausalLMDSparkMLA"]
        model_config = SimpleNamespace(
            hf_config=config,
            linear_attn_registry_result=None,
        )
        self.assertIsNone(mambaish_config(model_config))
        config.layer_group_size = 2
        self.assertIs(mambaish_config(model_config), config)

    def test_bf16_mla_post_load_is_local_to_ling_dspark(self):
        weight = torch.arange(48, dtype=torch.bfloat16).view(12, 4)
        attention = SimpleNamespace(
            kv_b_proj=SimpleNamespace(weight=weight),
            qk_nope_head_dim=2,
            v_head_dim=1,
            w_kc=None,
            w_vc=None,
            w_scale=None,
        )
        _run_ling_mla_post_load(attention)
        w_kc, w_vc = weight.unflatten(0, (-1, 3)).split([2, 1], dim=1)
        torch.testing.assert_close(
            attention.w_kc, w_kc.transpose(1, 2).contiguous().transpose(1, 2)
        )
        torch.testing.assert_close(attention.w_vc, w_vc.contiguous().transpose(1, 2))

    def test_absorbed_mla_matches_explicit_reference(self):
        torch.manual_seed(0)
        num_queries, num_keys = 3, 7
        num_heads, qk_nope_dim, value_dim, kv_lora_rank = 2, 3, 2, 4
        kv_b_weight = torch.randn(num_heads * (qk_nope_dim + value_dim), kv_lora_rank)
        attention = SimpleNamespace(
            kv_b_proj=SimpleNamespace(weight=kv_b_weight),
            qk_nope_head_dim=qk_nope_dim,
            v_head_dim=value_dim,
            w_kc=None,
            w_vc=None,
            w_scale=None,
        )
        _run_ling_mla_post_load(attention)

        q_nope = torch.randn(num_queries, num_heads, qk_nope_dim)
        kv_latent = torch.randn(num_keys, kv_lora_rank)
        explicit_k = torch.einsum("kl,hql->khq", kv_latent, attention.w_kc)
        explicit_v = torch.einsum("kl,hlv->khv", kv_latent, attention.w_vc)
        explicit_probs = torch.softmax(
            torch.einsum("nhq,khq->nhk", q_nope, explicit_k), dim=-1
        )
        explicit_output = torch.einsum("nhk,khv->nhv", explicit_probs, explicit_v)

        absorbed_q = torch.einsum("nhq,hql->nhl", q_nope, attention.w_kc)
        absorbed_probs = torch.softmax(
            torch.einsum("nhl,kl->nhk", absorbed_q, kv_latent), dim=-1
        )
        absorbed_latent = torch.einsum("nhk,kl->nhl", absorbed_probs, kv_latent)
        absorbed_output = torch.einsum("nhl,hlv->nhv", absorbed_latent, attention.w_vc)
        torch.testing.assert_close(absorbed_probs, explicit_probs)
        torch.testing.assert_close(absorbed_output, explicit_output)

    def test_context_kv_preserves_single_kv_head_axis(self):
        class FakeAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.kv_lora_rank = 2
                self.num_local_heads = 4
                self.qk_rope_head_dim = 2
                self.kv_a_layernorm = nn.Identity()
                self.attn_mqa = object()

            def kv_a_proj_with_mqa(self, hidden):
                return torch.cat([hidden[:, :2], hidden[:, 2:4]], dim=-1), None

            def rotary_emb(self, positions, q, k):
                self.rotary_shapes = (tuple(q.shape), tuple(k.shape))
                return q, k

        class FakeLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = FakeAttention()

        class FakePool:
            def set_mla_kv_buffer(self, layer, loc, k_nope, k_rope):
                self.shapes = (tuple(k_nope.shape), tuple(k_rope.shape))

        model = LingV3ForCausalLMDSparkMLA.__new__(LingV3ForCausalLMDSparkMLA)
        nn.Module.__init__(model)
        model.layers = nn.ModuleList([FakeLayer()])
        model.project_target_hidden = lambda hidden: hidden
        pool = FakePool()
        model.write_target_hidden_kv(
            target_hidden=torch.ones((3, 4)),
            pool=pool,
            positions=torch.arange(3),
            cache_loc=torch.arange(3),
        )
        self.assertEqual(
            model.layers[0].attention.rotary_shapes, ((3, 4, 2), (3, 1, 2))
        )
        self.assertEqual(pool.shapes, ((3, 1, 2), (3, 1, 2)))

    def test_partial_commit_filters_context_rows(self):
        class FakeAttention(nn.Module):
            kv_lora_rank = 2
            num_local_heads = 1
            qk_rope_head_dim = 2
            kv_a_layernorm = nn.Identity()
            attn_mqa = object()

            def kv_a_proj_with_mqa(self, hidden):
                return hidden, None

            def rotary_emb(self, positions, q, k):
                return q, k

        class FakeLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = FakeAttention()

        class FakePool:
            def set_mla_kv_buffer(self, layer, loc, k_nope, k_rope):
                self.loc = loc
                self.rows = k_nope.shape[0]

        model = LingV3ForCausalLMDSparkMLA.__new__(LingV3ForCausalLMDSparkMLA)
        nn.Module.__init__(model)
        model.layers = nn.ModuleList([FakeLayer()])
        model.project_target_hidden = lambda hidden: hidden
        pool = FakePool()
        model.write_target_hidden_kv(
            target_hidden=torch.ones((6, 4)),
            pool=pool,
            positions=torch.arange(6),
            cache_loc=torch.arange(6),
            cache_loc_2d=torch.arange(6).view(2, 3),
            commit_lens=torch.tensor([2, 1]),
        )
        torch.testing.assert_close(pool.loc, torch.tensor([0, 1, 3]))
        self.assertEqual(pool.rows, 3)

    def test_static_commit_keeps_rows_and_masks_rejected_locs(self):
        class FakeAttention(nn.Module):
            kv_lora_rank = 2
            num_local_heads = 1
            qk_rope_head_dim = 2
            kv_a_layernorm = nn.Identity()
            attn_mqa = object()

            def kv_a_proj_with_mqa(self, hidden):
                return hidden, None

            def rotary_emb(self, positions, q, k):
                return q, k

        class FakeLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = FakeAttention()

        class FakePool:
            def set_mla_kv_buffer(self, layer, loc, k_nope, k_rope):
                self.loc = loc.clone()
                self.k_nope = k_nope.clone()
                self.k_rope = k_rope.clone()

        model = LingV3ForCausalLMDSparkMLA.__new__(LingV3ForCausalLMDSparkMLA)
        nn.Module.__init__(model)
        model.layers = nn.ModuleList([FakeLayer()])
        model.project_target_hidden = lambda hidden: hidden
        pool = FakePool()
        hidden = torch.arange(24, dtype=torch.float32).view(6, 4)

        model.write_target_hidden_kv_static(
            target_hidden=hidden,
            pool=pool,
            positions=torch.arange(6),
            cache_loc=torch.tensor([10, 11, 12, 20, 21, 22]),
            commit_lens=torch.tensor([2, 1]),
            verify_len=3,
        )

        torch.testing.assert_close(pool.loc, torch.tensor([10, 11, 0, 20, 0, 0]))
        self.assertEqual(pool.k_nope.shape[0], 6)
        self.assertEqual(pool.k_rope.shape[0], 6)
        torch.testing.assert_close(pool.k_nope[[0, 1, 3], 0], hidden[[0, 1, 3], :2])
        torch.testing.assert_close(pool.k_rope[[0, 1, 3], 0], hidden[[0, 1, 3], 2:])

    def test_static_inject_normalizes_inputs_and_uses_fixed_row_protocol(self):
        class FakePool:
            def set_mla_kv_buffer(self, *args, **kwargs):
                pass

        class FakeDraft:
            def write_target_hidden_kv_static(self, **kwargs):
                self.kwargs = kwargs

        draft = FakeDraft()
        pool = FakePool()
        injector = TargetHiddenKvInjector.__new__(TargetHiddenKvInjector)
        injector.draft_model = draft
        injector.draft_model_runner = SimpleNamespace(token_to_kv_pool=pool)
        injector.model_runner = SimpleNamespace(device="cpu")
        injector.verify_num_draft_tokens = 3

        with patch(
            "sglang.srt.speculative.dspark_components.dspark_kv_inject."
            "is_unified_kv_triton",
            return_value=False,
        ):
            used = injector.inject_target_hidden_static(
                target_hidden=torch.ones((6, 4)),
                cache_loc=torch.arange(6, dtype=torch.int32),
                positions=torch.arange(6, dtype=torch.int32),
                commit_lens=torch.tensor([2, 1], dtype=torch.int64),
            )

        self.assertTrue(used)
        self.assertIs(draft.kwargs["pool"], pool)
        self.assertEqual(draft.kwargs["verify_len"], 3)
        self.assertEqual(draft.kwargs["cache_loc"].dtype, torch.int64)
        self.assertEqual(draft.kwargs["positions"].dtype, torch.int64)
        self.assertEqual(draft.kwargs["commit_lens"].dtype, torch.int32)
        torch.testing.assert_close(
            draft.kwargs["commit_lens"], torch.tensor([2, 1], dtype=torch.int32)
        )

    def _compact_commit_fixture(self, device):
        """A compact-verify commit of stride 4 over two requests.

        verify_lens (2, 3) leave never-verified padding rows; commit_lens (1, 3)
        reject one draft of the second request. Slots 100.. are the first
        request's verify window, 200.. the second's.
        """
        stride = 4
        prefix_lens = torch.tensor([10, 20], dtype=torch.int64, device=device)
        block_pos_offsets = torch.arange(stride, dtype=torch.int64, device=device)
        verify_cache_loc = torch.tensor(
            [100, 101, 102, 103, 200, 201, 202, 203],
            dtype=torch.int64,
            device=device,
        )
        verify_window = VerifyWindow(
            positions_2d=prefix_lens.unsqueeze(1) + block_pos_offsets,
            verify_cache_loc=verify_cache_loc,
            verify_cache_loc_2d=verify_cache_loc.view(2, stride),
        )
        return SimpleNamespace(
            stride=stride,
            batch=SimpleNamespace(
                seq_lens=prefix_lens,
                req_pool_indices=torch.tensor([3, 4], device=device),
            ),
            layout=SimpleNamespace(
                verify_lens=torch.tensor([2, 3], dtype=torch.int64, device=device)
            ),
            hidden=torch.arange(
                2 * stride * 4, dtype=torch.float32, device=device
            ).view(2 * stride, 4),
            verify_window=verify_window,
            commit_lens=torch.tensor([1, 3], dtype=torch.int32, device=device),
            expected_loc=torch.tensor(
                [100, 0, 0, 0, 200, 201, 202, 0], dtype=torch.int64, device=device
            ),
        )

    def _compact_commit_injector(self, draft, pool, device):
        injector = TargetHiddenKvInjector.__new__(TargetHiddenKvInjector)
        injector.draft_model = draft
        injector.draft_model_runner = SimpleNamespace(token_to_kv_pool=pool)
        injector.model_runner = SimpleNamespace(device=device)
        injector.verify_num_draft_tokens = 4
        return injector

    def test_compact_commit_uses_fixed_row_protocol(self):
        """Compact verify commits the whole [bs * stride] window through the
        fixed-row protocol; compacting accepted rows with a boolean mask would
        synchronize the host in the verify tail."""

        class FakePool:
            def set_mla_kv_buffer(self, *args, **kwargs):
                pass

        class FakeDraft:
            def write_target_hidden_kv_static(self, **kwargs):
                self.static_kwargs = kwargs

            def write_target_hidden_kv(self, **kwargs):
                raise AssertionError("compact commit compacted rows on the host")

        fx = self._compact_commit_fixture("cpu")
        draft = FakeDraft()
        injector = self._compact_commit_injector(draft, FakePool(), "cpu")

        with patch(
            "sglang.srt.speculative.dspark_components.dspark_kv_inject."
            "is_unified_kv_triton",
            return_value=False,
        ):
            injector.inject_ragged(
                batch=fx.batch,
                layout=fx.layout,
                hidden_strided=fx.hidden,
                verify_window=fx.verify_window,
                commit_lens=fx.commit_lens,
                bs=2,
            )

        kwargs = draft.static_kwargs
        self.assertEqual(kwargs["verify_len"], fx.stride)
        self.assertEqual(kwargs["target_hidden"].shape[0], 2 * fx.stride)
        torch.testing.assert_close(
            kwargs["cache_loc"], fx.verify_window.verify_cache_loc
        )
        torch.testing.assert_close(
            kwargs["positions"], fx.verify_window.positions_2d.reshape(-1)
        )
        torch.testing.assert_close(kwargs["commit_lens"], fx.commit_lens)

    def test_compact_commit_flag_off_restores_host_compaction(self):
        """SGLANG_DSPARK_ENABLE_STATIC_COMMIT=0 must still route compact
        verify through the masked commit, not silently keep the fixed-row path."""

        class FakePool:
            def set_mla_kv_buffer(self, *args, **kwargs):
                pass

        class FakeDraft:
            def write_target_hidden_kv_static(self, **kwargs):
                raise AssertionError("fixed-row commit ran with the flag off")

            def write_target_hidden_kv(self, **kwargs):
                self.eager_kwargs = kwargs

        fx = self._compact_commit_fixture("cpu")
        draft = FakeDraft()
        injector = self._compact_commit_injector(draft, FakePool(), "cpu")

        with (
            envs.SGLANG_DSPARK_ENABLE_STATIC_COMMIT.override(False),
            patch(
                "sglang.srt.speculative.dspark_components.dspark_kv_inject."
                "is_unified_kv_triton",
                return_value=False,
            ),
        ):
            injector.inject_ragged(
                batch=fx.batch,
                layout=fx.layout,
                hidden_strided=fx.hidden,
                verify_window=fx.verify_window,
                commit_lens=fx.commit_lens,
                bs=2,
            )

        kwargs = draft.eager_kwargs
        torch.testing.assert_close(
            kwargs["cache_loc_2d"], fx.verify_window.verify_cache_loc_2d
        )
        torch.testing.assert_close(kwargs["commit_lens"], fx.commit_lens)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_compact_commit_runs_without_host_sync_and_masks_rows(self):
        """The compact commit must stay device-side end to end: rejected and
        never-verified rows go to the reserved skip slot, and no kernel in the
        tail synchronizes the host."""

        class FakeAttention(nn.Module):
            kv_lora_rank = 2
            num_local_heads = 1
            qk_rope_head_dim = 2
            kv_a_layernorm = nn.Identity()
            attn_mqa = object()

            def kv_a_proj_with_mqa(self, hidden):
                return hidden, None

            def rotary_emb(self, positions, q, k):
                return q, k

        class FakeLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = FakeAttention()

        class FakePool:
            def set_mla_kv_buffer(self, layer, loc, k_nope, k_rope):
                self.loc = loc.clone()
                self.k_nope = k_nope.clone()

        device = torch.device("cuda")
        model = LingV3ForCausalLMDSparkMLA.__new__(LingV3ForCausalLMDSparkMLA)
        nn.Module.__init__(model)
        model.layers = nn.ModuleList([FakeLayer()])
        model.project_target_hidden = lambda hidden: hidden
        pool = FakePool()
        fx = self._compact_commit_fixture(device)
        injector = self._compact_commit_injector(model, pool, device)

        def commit():
            injector.inject_ragged(
                batch=fx.batch,
                layout=fx.layout,
                hidden_strided=fx.hidden,
                verify_window=fx.verify_window,
                commit_lens=fx.commit_lens,
                bs=2,
            )

        unified_patch = patch(
            "sglang.srt.speculative.dspark_components.dspark_kv_inject."
            "is_unified_kv_triton",
            return_value=False,
        )
        torch.cuda.synchronize(device)
        torch.cuda.set_sync_debug_mode("error")
        try:
            with unified_patch:
                commit()
            # Harness self-check: the masked path this test guards against
            # does synchronize, so a silent no-op debug mode cannot pass above.
            with (
                unified_patch,
                envs.SGLANG_DSPARK_ENABLE_STATIC_COMMIT.override(False),
                self.assertRaisesRegex(RuntimeError, "synchroniz"),
            ):
                commit()
        finally:
            torch.cuda.set_sync_debug_mode("default")

        torch.testing.assert_close(pool.loc, fx.expected_loc)
        kept = [0, 4, 5, 6]
        torch.testing.assert_close(pool.k_nope[kept, 0], fx.hidden[kept, :2])

    def test_quantization_scales_use_standard_weight_loaders(self):
        model = LingV3ForCausalLMDSparkMLA.__new__(LingV3ForCausalLMDSparkMLA)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(num_experts=2)
        layer = nn.Module()
        layer.attention = nn.Module()
        layer.attention.kv_b_proj = nn.Module()
        layer.attention.kv_b_proj.register_parameter(
            "weight_scale_inv", nn.Parameter(torch.zeros(1))
        )
        layer.mlp = nn.Module()
        layer.mlp.experts = nn.Module()
        w13_scale = nn.Parameter(torch.zeros(2, 2, 1))
        w2_scale = nn.Parameter(torch.zeros(2, 1))
        layer.mlp.experts.register_parameter("w13_weight_scale_inv", w13_scale)
        layer.mlp.experts.register_parameter("w2_weight_scale_inv", w2_scale)
        model.layers = nn.ModuleList([layer])
        model._post_load_weights = lambda: None

        expert_loads = []

        def load_expert(param, weight, name, *, shard_id, expert_id):
            expert_loads.append((param, name, shard_id, expert_id, weight.clone()))

        w13_scale.weight_loader = load_expert
        w2_scale.weight_loader = load_expert
        direct_scale = torch.tensor([3.0])
        model.load_weights(
            [
                (
                    "layers.0.attention.kv_b_proj.weight_scale_inv",
                    direct_scale,
                ),
                (
                    "layers.0.mlp.experts.1.gate_proj.weight_scale_inv",
                    torch.tensor([4.0]),
                ),
                (
                    "layers.0.mlp.experts.0.up_proj.weight_scale_inv",
                    torch.tensor([5.0]),
                ),
                (
                    "layers.0.mlp.experts.1.down_proj.weight_scale_inv",
                    torch.tensor([6.0]),
                ),
            ]
        )

        torch.testing.assert_close(
            model.layers[0].attention.kv_b_proj.weight_scale_inv, direct_scale
        )
        self.assertEqual(
            [(call[2], call[3]) for call in expert_loads],
            [("w1", 1), ("w3", 0), ("w2", 1)],
        )

    def test_online_blockwise_fp8_scales_are_created_after_weight_load(self):
        model = LingV3ForCausalLMDSparkMLA.__new__(LingV3ForCausalLMDSparkMLA)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(num_experts=0)
        model.quant_config = SimpleNamespace(is_online_block_quant=True)
        model.register_parameter("weight_scale_inv", nn.Parameter(torch.zeros(1)))
        model.layers = nn.ModuleList([])
        model._post_load_weights = lambda: None

        self.assertEqual(model.load_weights([]), set())

        model.quant_config = SimpleNamespace(is_online_block_quant=False)
        with self.assertRaisesRegex(ValueError, "weight_scale_inv"):
            model.load_weights([])

    def test_bundled_loader_consumes_only_dspark_namespace(self):
        model = LingV3ForCausalLMDSparkMLA.__new__(LingV3ForCausalLMDSparkMLA)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(num_experts=1)
        model.is_bundled = True
        model.register_parameter("target_norm", nn.Parameter(torch.zeros(1)))
        model._post_load_weights = lambda: None

        loaded = model.load_weights(
            [
                ("model.layers.0.weight", torch.ones(1)),
                ("dspark.target_norm", torch.tensor([2.0])),
            ]
        )
        self.assertEqual(loaded, {"target_norm"})
        torch.testing.assert_close(model.target_norm, torch.tensor([2.0]))

    def test_write_target_hidden_kv_drops_q_lora_slice_in_fused_mla(self):
        # When q_lora_rank is set the MLA builds fused_qkv_a_proj_with_mqa (and
        # no standalone kv_a_proj_with_mqa), so the KV latent is the trailing
        # part of that fused projection, behind the leading q_lora_rank slice.
        q_lora_rank, kv_lora_rank, rope_dim = 3, 2, 1
        sentinel = -999.0

        class FusedAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_lora_rank = q_lora_rank
                self.kv_lora_rank = kv_lora_rank
                self.qk_rope_head_dim = rope_dim
                self.num_local_heads = 4
                self.kv_a_layernorm = nn.Identity()
                self.attn_mqa = object()
                self.called_fused = False

            def fused_qkv_a_proj_with_mqa(self, context):
                self.called_fused = True
                rows = context.shape[0]
                q_slice = torch.full((rows, q_lora_rank), sentinel)
                kv_slice = torch.arange(rows * kv_lora_rank, dtype=torch.float).view(
                    rows, kv_lora_rank
                )
                rope_slice = torch.full((rows, rope_dim), 777.0)
                return torch.cat([q_slice, kv_slice, rope_slice], dim=-1), None

            def rotary_emb(self, positions, q, k):
                return q, k

        class FakeLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = FusedAttention()

        class FakePool:
            def set_mla_kv_buffer(self, layer, loc, k_nope, k_rope):
                self.k_nope = k_nope.clone()
                self.k_rope = k_rope.clone()

        model = LingV3ForCausalLMDSparkMLA.__new__(LingV3ForCausalLMDSparkMLA)
        nn.Module.__init__(model)
        model.layers = nn.ModuleList([FakeLayer()])
        model.project_target_hidden = lambda hidden: hidden
        pool = FakePool()
        model.write_target_hidden_kv(
            target_hidden=torch.ones((3, 5)),
            pool=pool,
            positions=torch.arange(3),
            cache_loc=torch.arange(3),
        )

        attn = model.layers[0].attention
        self.assertTrue(attn.called_fused)
        self.assertFalse(hasattr(attn, "kv_a_proj_with_mqa"))
        self.assertFalse((pool.k_nope == sentinel).any())
        self.assertFalse((pool.k_rope == sentinel).any())
        torch.testing.assert_close(
            pool.k_nope.squeeze(1),
            torch.arange(3 * kv_lora_rank, dtype=torch.float).view(3, kv_lora_rank),
        )
        torch.testing.assert_close(
            pool.k_rope.squeeze(1), torch.full((3, rope_dim), 777.0)
        )

    def test_q_lora_fused_load_concatenates_q_and_kv_down_proj(self):
        # q_lora_rank builds a single fused_qkv_a_proj_with_mqa, but the draft
        # checkpoint stores q_a_proj and kv_a_proj_with_mqa separately; load_weights
        # must concatenate the pair into the fused parameter.
        q_lora_rank, kv_lora_rank, rope_dim, hidden = 3, 2, 1, 5
        fused_out = q_lora_rank + kv_lora_rank + rope_dim
        q_a_proj = torch.randn(q_lora_rank, hidden)
        kv_a_proj = torch.randn(kv_lora_rank + rope_dim, hidden)

        def build_model():
            model = LingV3ForCausalLMDSparkMLA.__new__(LingV3ForCausalLMDSparkMLA)
            nn.Module.__init__(model)
            model.config = SimpleNamespace(q_lora_rank=q_lora_rank, num_experts=1)
            model._post_load_weights = lambda: None
            layer = nn.Module()
            layer.attention = nn.Module()
            layer.attention.fused_qkv_a_proj_with_mqa = nn.Linear(
                hidden, fused_out, bias=False
            )
            model.layers = nn.ModuleList([layer])
            return model

        for bundled in (True, False):
            with self.subTest(bundled=bundled):
                model = build_model()
                model.is_bundled = bundled
                prefix = "dspark." if bundled else ""
                loaded = model.load_weights(
                    [
                        (f"{prefix}layers.0.attention.q_a_proj.weight", q_a_proj),
                        (
                            f"{prefix}layers.0.attention.kv_a_proj_with_mqa.weight",
                            kv_a_proj,
                        ),
                    ]
                )
                self.assertEqual(
                    loaded,
                    {"layers.0.attention.fused_qkv_a_proj_with_mqa.weight"},
                )
                torch.testing.assert_close(
                    model.layers[0].attention.fused_qkv_a_proj_with_mqa.weight.data,
                    torch.cat([q_a_proj, kv_a_proj], dim=0),
                )


if __name__ == "__main__":
    unittest.main()
