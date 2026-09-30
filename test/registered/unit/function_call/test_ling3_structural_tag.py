"""Ling native tool constraints: real CPU grammar matching and parser round trips."""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string

from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarObject
from sglang.srt.constrained.xgrammar_backend import XGrammarGrammar
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    Function,
    Tool,
    ToolChoice,
)
from sglang.srt.environ import ToolStrictLevel
from sglang.srt.function_call.core_types import NativeToolConstraintError
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.ling3_detector import (
    Ling3Detector,
    _ling3_native_structural_tag_available,
)
from sglang.srt.parser.reasoning_parser import Ling3Detector as Ling3ReasoningDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _call(name="weather", city="Beijing", newline=""):
    return (
        f"<tool_call>{name}{newline}"
        f"<arg_key>city</arg_key><arg_value>{city}</arg_value>"
        "</tool_call>"
    )


class TestLing3StructuralTag(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="weather",
                    strict=True,
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="get_date",
                    strict=True,
                    parameters={
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                ),
            ),
        ]

    def _parser(self, tools=None, strict_level=ToolStrictLevel.OFF):
        parser = FunctionCallParser(self.tools if tools is None else tools, "ling3")
        parser.tool_strict_level = strict_level
        return parser

    def _grammar(self, choice="required", tools=None, **kwargs):
        constraint = self._parser(tools).get_structure_constraint(choice, **kwargs)
        self.assertIsNotNone(constraint)
        self.assertEqual(constraint[0], "structural_tag")
        return xgr.Grammar.from_structural_tag(constraint[1])

    def test_required_accepts_native_layouts_not_json_array(self):
        grammar = self._grammar()
        for text in (_call(), _call(newline="\n"), "<tool_call>get_date</tool_call>"):
            with self.subTest(text=text):
                self.assertTrue(_is_grammar_accept_string(grammar, text))
        for text in (
            "",
            "Normal answer without a tool call",
            '[{"name":"weather","parameters":{"city":"Beijing"}}]',
            _call(name="unknown"),
            "<tool_call>weather</tool_call>",
        ):
            with self.subTest(text=text):
                self.assertFalse(_is_grammar_accept_string(grammar, text))

    def test_named_choice_only_accepts_selected_tool(self):
        choice = ToolChoice(type="function", function={"name": "get_date"})
        grammar = self._grammar(choice)
        self.assertTrue(
            _is_grammar_accept_string(grammar, "<tool_call>get_date</tool_call>")
        )
        self.assertFalse(_is_grammar_accept_string(grammar, _call()))

    def test_parallel_tool_calls_limit(self):
        text = _call() + "<tool_call>get_date</tool_call>"
        for choice in ("required", "auto"):
            for parallel in (True, False):
                with self.subTest(choice=choice, parallel=parallel):
                    grammar = self._grammar(choice, parallel_tool_calls=parallel)
                    self.assertTrue(_is_grammar_accept_string(grammar, _call()))
                    self.assertEqual(_is_grammar_accept_string(grammar, text), parallel)

    def test_strict_auto_allows_prose_but_constrains_tool_payload(self):
        grammar = self._grammar("auto")
        self.assertTrue(_is_grammar_accept_string(grammar, "No tool is needed."))
        self.assertTrue(_is_grammar_accept_string(grammar, "Checking. " + _call()))
        self.assertFalse(_is_grammar_accept_string(grammar, _call(name="unknown")))
        self.assertFalse(
            _is_grammar_accept_string(grammar, "<tool_call>weather</tool_call>")
        )

    def test_non_strict_auto_and_none_remain_unconstrained(self):
        tools = [tool.model_copy(deep=True) for tool in self.tools]
        for tool in tools:
            tool.function.strict = False
        parser = self._parser(tools)
        self.assertIsNone(parser.get_structure_constraint("auto"))
        self.assertIsNone(parser.get_structure_constraint("none"))

    def test_non_strict_required_still_uses_native_format(self):
        tool = self.tools[0].model_copy(deep=True)
        tool.function.strict = False
        grammar = self._grammar(tools=[tool])
        self.assertTrue(_is_grammar_accept_string(grammar, _call()))
        self.assertFalse(_is_grammar_accept_string(grammar, _call(name="unknown")))

    def test_parameter_strict_level_is_applied_without_mutating_request(self):
        tool = self.tools[0].model_copy(deep=True)
        tool.function.strict = False
        parser = self._parser([tool], strict_level=ToolStrictLevel.PARAMETER)
        constraint = parser.get_structure_constraint("auto")
        self.assertEqual(constraint[0], "structural_tag")
        grammar = xgr.Grammar.from_structural_tag(constraint[1])
        self.assertFalse(
            _is_grammar_accept_string(grammar, "<tool_call>weather</tool_call>")
        )
        self.assertIs(tool.function.strict, False)

    def test_reasoning_prefix_and_parallel_limit(self):
        grammar = self._grammar(thinking_mode=True, parallel_tool_calls=False)
        text = "Let me check the weather.</think>" + _call()
        self.assertTrue(_is_grammar_accept_string(grammar, text))
        self.assertFalse(_is_grammar_accept_string(grammar, text + _call()))

    def test_native_output_round_trips_including_character_streams(self):
        grammar = self._grammar()
        text = _call(city="北京", newline="\n") + "<tool_call>get_date</tool_call>"
        self.assertTrue(_is_grammar_accept_string(grammar, text))
        _, calls = self._parser().parse_non_stream(text)
        self.assertEqual([call.tool_index for call in calls], [0, 1])
        self.assertEqual([call.name for call in calls], ["weather", "get_date"])
        expected = [{"city": "北京"}, {}]
        self.assertEqual([json.loads(call.parameters) for call in calls], expected)
        for size in (1, 7, len(text)):
            with self.subTest(chunk_size=size):
                parser = self._parser()
                arguments, names = {}, {}
                for start in range(0, len(text), size):
                    _, deltas = parser.parse_stream_chunk(text[start : start + size])
                    for delta in deltas:
                        if delta.name:
                            names[delta.tool_index] = delta.name
                        arguments[delta.tool_index] = (
                            arguments.get(delta.tool_index, "") + delta.parameters
                        )
                self.assertEqual(names, {0: "weather", 1: "get_date"})
                self.assertEqual([json.loads(arguments[i]) for i in range(2)], expected)

    def test_coalesced_stream_preserves_prose_and_multiple_calls(self):
        text = "Checking. " + _call() + "<tool_call>get_date</tool_call>" + " Done."
        normal, calls = self._parser().parse_stream_chunk(text)
        self.assertEqual(normal, "Checking.  Done.")
        self.assertEqual(
            [call.name for call in calls if call.name], ["weather", "get_date"]
        )

    def test_compiled_grammar_masks_stop_until_required_call_completes(self):
        constraint = self._parser().get_structure_constraint("required")
        # Byte vocabulary lets the real compiler/matcher exercise EOS gating
        # without downloading model weights or using a GPU.
        tokenizer_info = xgr.TokenizerInfo(
            [bytes([i]) for i in range(256)] + [b"<eos>"],
            vocab_type=xgr.VocabType.RAW,
            stop_token_ids=[256],
        )
        compiled = xgr.GrammarCompiler(tokenizer_info).compile_structural_tag(
            constraint[1]
        )
        matcher = xgr.GrammarMatcher(compiled)
        self.assertFalse(matcher.accept_token(256))
        for token in _call().encode():
            self.assertTrue(matcher.accept_token(token))
        self.assertTrue(matcher.accept_token(256))
        self.assertTrue(matcher.is_terminated())

    def test_thinking_tool_start_activates_native_masks_and_stream_parser(self):
        # The incident's layout, with synthetic content: no </think>, a newline
        # after the name/key, and adjacent argument pairs. Exercise the serving
        # constraint selection AND the real matcher behind the reasoning wrapper.
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="chat_history_search",
                    strict=True,
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 512,
                            },
                            "search_mode": {
                                "type": "string",
                                "enum": ["recent", "relevance"],
                            },
                        },
                        "required": ["query", "search_mode"],
                        "additionalProperties": False,
                    },
                ),
            )
        ]
        text = (
            "<tool_call>chat_history_search\n"
            "<arg_key>query</arg_key>\n<arg_value>昨天的讨论</arg_value>"
            "<arg_key>search_mode</arg_key>\n<arg_value>relevance</arg_value>\n"
            "</tool_call>"
        )
        chat = self._chat()
        chat.reasoning_parser = "ling3"
        chat._reasoning_detector = Ling3ReasoningDetector()
        chat.template_manager = SimpleNamespace(reasoning_config=None)
        request = ChatCompletionRequest(
            model="ling",
            messages=[{"role": "user", "content": "search"}],
            input_ids=[1],
            tools=tools,
            tool_choice="required",
            stream=True,
            chat_template_kwargs={"enable_thinking": True},
        )
        processed = chat._process_messages(request, is_multimodal=False)
        self.assertTrue(processed.require_reasoning)
        sampling = request.to_sampling_params(
            stop=[],
            model_generation_config={},
            tool_call_constraint=processed.tool_call_constraint,
        )
        self.assertIn("structural_tag", sampling)
        self.assertNotIn("json_schema", sampling)
        tokenizer_info = xgr.TokenizerInfo(
            [bytes([i]) for i in range(256)] + [b"<eos>"],
            vocab_type=xgr.VocabType.RAW,
            stop_token_ids=[256],
        )
        compiled = xgr.GrammarCompiler(tokenizer_info).compile_structural_tag(
            processed.tool_call_constraint[1]
        )

        def make_wrapper():
            inner = XGrammarGrammar(
                xgr.GrammarMatcher(compiled, max_rollback_tokens=256),
                257,
                compiled,
                [256],
            )
            wrapper = ReasonerGrammarObject(
                inner,
                think_end_ids=list(b"</think>"),
                tool_start_ids=list(b"<tool_call>"),
            )
            wrapper.maybe_init_reasoning(True)
            return wrapper

        wrapper = make_wrapper()
        mask = torch.full((1, 9), -1, dtype=torch.int32)

        def allowed(token):
            mask.fill_(-1)
            wrapper.fill_vocab_mask(mask, 0)
            return bool(int(mask[0, token // 32]) & (1 << (token % 32)))

        prefix = "<tool_call>"
        suffix = text[len(prefix) :]
        for token in prefix.encode():
            self.assertTrue(allowed(token))
            wrapper.accept_token(token)
        self.assertTrue(wrapper._is_generation())
        self.assertFalse(allowed(ord("!")), "unknown tool name must be masked")
        self.assertFalse(allowed(256), "EOS must wait for a complete required call")
        for token in suffix.encode():
            self.assertTrue(allowed(token))
            wrapper.accept_token(token)
        self.assertTrue(allowed(256))
        wrapper.accept_token(256)
        self.assertTrue(wrapper.is_terminated())

        # The XML string grammar can retain alternative parses at an argument
        # boundary, so test rejection of the complete invalid continuation,
        # rather than requiring rejection at its first character.
        wrapper = make_wrapper()
        for token in [*text.replace("relevance", "invalid").encode(), 256]:
            if not allowed(token):
                break
            wrapper.accept_token(token)
        else:
            self.fail("invalid enum call must not be allowed to terminate")

        reasoning = Ling3ReasoningDetector(force_reasoning=True)
        parsers, has_calls, chunks = {}, {}, []

        async def collect():
            for char in text:
                delta = reasoning.parse_streaming_increment(char)
                async for chunk in chat._process_tool_call_stream(
                    index=0,
                    delta=delta.normal_text,
                    parser_dict=parsers,
                    content={"meta_info": {"id": "implicit-tool-test"}},
                    request=request,
                    has_tool_calls=has_calls,
                ):
                    chunks.append(json.loads(chunk.removeprefix("data: ")))

        asyncio.run(collect())
        self.assertIsInstance(parsers[0], FunctionCallParser)
        self.assertTrue(has_calls[0])
        normal, arguments, names = "", "", []
        for chunk in chunks:
            delta = chunk["choices"][0]["delta"]
            normal += delta.get("content") or ""
            for call in delta.get("tool_calls") or []:
                function = call["function"]
                if function.get("name"):
                    names.append(function["name"])
                arguments += function.get("arguments") or ""
        self.assertEqual(normal, "")
        self.assertEqual(names, ["chat_history_search"])
        self.assertEqual(
            json.loads(arguments), {"query": "昨天的讨论", "search_mode": "relevance"}
        )

    def test_xml_string_values_are_not_reinterpreted_as_json(self):
        grammar = self._grammar()
        for value in ("true", "null", '"quoted"', '{"x":1}', r"C:\tmp", " 北京 "):
            with self.subTest(value=value):
                text = _call(city=value)
                self.assertTrue(_is_grammar_accept_string(grammar, text))
                _, calls = self._parser().parse_non_stream(text)
                self.assertEqual(json.loads(calls[0].parameters), {"city": value})
                _, deltas = self._parser().parse_stream_chunk(text)
                arguments = "".join(call.parameters for call in deltas)
                self.assertEqual(json.loads(arguments), {"city": value})

    def test_nested_json_values_remain_typed(self):
        tool = Tool(
            type="function",
            function=Function(
                name="submit",
                strict=True,
                parameters={
                    "type": "object",
                    "properties": {
                        "payload": {
                            "type": "object",
                            "properties": {"count": {"type": "integer"}},
                            "required": ["count"],
                            "additionalProperties": False,
                        },
                        "enabled": {"type": "boolean"},
                    },
                    "required": ["payload", "enabled"],
                    "additionalProperties": False,
                },
            ),
        )
        text = (
            '<tool_call>submit<arg_key>payload</arg_key><arg_value>{"count":2}</arg_value>'
            "<arg_key>enabled</arg_key><arg_value>true</arg_value></tool_call>"
        )
        grammar = self._grammar(tools=[tool])
        self.assertTrue(_is_grammar_accept_string(grammar, text))
        self.assertFalse(
            _is_grammar_accept_string(
                grammar, text.replace('"count":2', '"count":"bad"')
            )
        )
        _, calls = self._parser([tool]).parse_non_stream(text)
        self.assertEqual(
            json.loads(calls[0].parameters), {"payload": {"count": 2}, "enabled": True}
        )

    def test_missing_native_backend_rejects_constrained_requests(self):
        with patch(
            "sglang.srt.function_call.ling3_detector._ling3_native_structural_tag_available",
            return_value=False,
        ):
            parser = self._parser()
            self.assertFalse(parser.detector.supports_structural_tag())
            for choice in (
                "required",
                ToolChoice(type="function", function={"name": "get_date"}),
                "auto",  # strict=True also requires a working native grammar.
            ):
                with (
                    self.subTest(choice=choice),
                    self.assertRaisesRegex(NativeToolConstraintError, "glm_4_7"),
                ):
                    parser.get_structure_constraint(choice)
            tools = [tool.model_copy(deep=True) for tool in self.tools]
            for tool in tools:
                tool.function.strict = False
            self.assertIsNone(self._parser(tools).get_structure_constraint("auto"))
            self.assertIsNone(parser.get_structure_constraint("none"))

    def test_native_builder_failures_do_not_become_json_or_no_constraint(self):
        for outcome in (None, ValueError("unsupported schema"), RuntimeError("failed")):
            for choice in (
                "required",
                ToolChoice(type="function", function={"name": "get_date"}),
                "auto",
            ):
                with self.subTest(outcome=outcome, choice=choice):
                    kwargs = (
                        {"side_effect": outcome}
                        if isinstance(outcome, Exception)
                        else {"return_value": outcome}
                    )
                    with (
                        patch(
                            "sglang.srt.function_call.base_format_detector.get_model_structural_tag",
                            **kwargs,
                        ),
                        patch(
                            "sglang.srt.function_call.function_call_parser.get_json_schema_constraint"
                        ) as json_fallback,
                    ):
                        with self.assertRaisesRegex(
                            NativeToolConstraintError, "Refusing JSON fallback"
                        ):
                            self._parser().get_structure_constraint(choice)
                        json_fallback.assert_not_called()

    def _chat(self):
        from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

        chat = OpenAIServingChat.__new__(OpenAIServingChat)
        chat.tool_call_parser = "ling3"
        chat.reasoning_parser = None
        chat.template_manager = SimpleNamespace(
            reasoning_config=None, chat_template_name=None
        )
        chat.chat_encoding_spec = None
        chat.default_chat_template_kwargs = {}
        chat.is_gpt_oss = chat.is_gemma4 = False
        chat.tokenizer_manager = SimpleNamespace(
            tokenizer=None,
            model_config=SimpleNamespace(is_multimodal=False),
            request_logger=SimpleNamespace(log_requests=False),
        )
        return chat

    def test_serving_constraint_errors_return_400_before_generation(self):
        chat = self._chat()
        for stream in (True, False):
            for choice in (
                "required",
                ToolChoice(type="function", function={"name": "get_date"}),
            ):
                with self.subTest(stream=stream, choice=choice):
                    request = ChatCompletionRequest(
                        model="ling",
                        messages=[{"role": "user", "content": "date?"}],
                        tools=self.tools,
                        tool_choice=choice,
                        stream=stream,
                    )
                    with (
                        patch(
                            "sglang.srt.function_call.ling3_detector._ling3_native_structural_tag_available",
                            return_value=False,
                        ),
                        patch.object(chat, "_validate_request", return_value=None),
                        patch.object(chat, "_handle_streaming_request") as streaming,
                        patch.object(
                            chat, "_handle_non_streaming_request"
                        ) as non_streaming,
                        patch(
                            "sglang.srt.entrypoints.openai.serving_chat.get_json_schema_constraint"
                        ) as fallback,
                    ):
                        response = asyncio.run(chat.handle_request(request, None))
                    self.assertEqual(response.status_code, 400)
                    self.assertIn(
                        "refusing JSON fallback", json.loads(response.body)["message"]
                    )
                    fallback.assert_not_called()
                    streaming.assert_not_called()
                    non_streaming.assert_not_called()

    def test_serving_request_builds_native_constraint(self):
        for choice in (
            "required",
            ToolChoice(type="function", function={"name": "get_date"}),
        ):
            with self.subTest(choice=choice):
                request = ChatCompletionRequest(
                    model="ling",
                    messages=[{"role": "user", "content": "date?"}],
                    input_ids=[1],
                    tools=self.tools,
                    tool_choice=choice,
                )
                processed = self._chat()._process_messages(request, is_multimodal=False)
                self.assertEqual(processed.tool_call_constraint[0], "structural_tag")
                grammar = xgr.Grammar.from_structural_tag(
                    processed.tool_call_constraint[1]
                )
                self.assertTrue(
                    _is_grammar_accept_string(
                        grammar, "<tool_call>get_date</tool_call>"
                    )
                )

    def test_truncated_native_call_is_not_reported_as_complete(self):
        text = _call()
        for cutoff in (len("<tool_call>weather"), len(text) - 1):
            with self.subTest(cutoff=cutoff):
                truncated = text[:cutoff]
                self.assertFalse(_is_grammar_accept_string(self._grammar(), truncated))
                calls, _, reason = self._chat()._process_tool_calls(
                    truncated, self.tools, {"type": "length"}, tool_choice="required"
                )
                self.assertIsNone(calls)
                self.assertEqual(reason["type"], "length")

    def test_large_synthetic_schema_and_complex_parameters_round_trip(self):
        # Representative schema features, NOT a replay of the truncated production log.
        schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索语义与时间范围。" * 2048,
                },
                "mode": {"type": "string", "enum": ["recent", "relevance"]},
                "start_time": {
                    "type": "string",
                    "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$",
                },
                "filters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"count": {"type": "integer"}},
                        "required": ["count"],
                        "additionalProperties": False,
                    },
                },
                "limit": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            },
            "required": ["query", "mode", "start_time", "filters", "limit"],
            "additionalProperties": False,
        }
        tools = [
            Tool(
                type="function",
                function=Function(name="search", strict=True, parameters=schema),
            )
        ]
        values = {
            "query": "昨天的记录",
            "mode": "relevance",
            "start_time": "2026-09-14",
            "filters": '[{"count":2}]',
            "limit": "null",
        }
        text = (
            "<tool_call>search"
            + "".join(
                f"<arg_key>{key}</arg_key><arg_value>{value}</arg_value>"
                for key, value in values.items()
            )
            + "</tool_call>"
        )
        grammar = self._grammar(tools=tools)
        self.assertTrue(_is_grammar_accept_string(grammar, text))
        for old, new in (
            ("relevance", "invalid"),
            ("2026-09-14", "yesterday"),
            ('"count":2', '"count":"bad"'),
        ):
            with self.subTest(invalid=new):
                self.assertFalse(
                    _is_grammar_accept_string(grammar, text.replace(old, new))
                )
        expected = dict(values, filters=[{"count": 2}], limit=None)
        _, calls = self._parser(tools).parse_non_stream(text)
        self.assertEqual(json.loads(calls[0].parameters), expected)
        for size in (1, 7, len(text)):
            with self.subTest(chunk_size=size):
                parser = self._parser(tools)
                arguments = ""
                for start in range(0, len(text), size):
                    _, deltas = parser.parse_stream_chunk(text[start : start + size])
                    arguments += "".join(call.parameters for call in deltas)
                self.assertEqual(json.loads(arguments), expected)

    def test_serving_required_and_named_use_native_parser(self):
        chat = self._chat()
        text = "<tool_call>get_date</tool_call>"
        for choice in (
            "required",
            ToolChoice(type="function", function={"name": "get_date"}),
        ):
            with self.subTest(choice=choice):
                result = chat._process_tool_calls(
                    text,
                    self.tools,
                    {"type": "stop"},
                    tool_choice=choice,
                )
                self.assertEqual(result[0][0].function.name, "get_date")
                self.assertEqual(result[0][0].function.arguments, "{}")
                self.assertEqual(result[2]["type"], "tool_calls")
                request = ChatCompletionRequest(
                    model="ling",
                    messages=[{"role": "user", "content": "date?"}],
                    tools=self.tools,
                    tool_choice=choice,
                    stream=True,
                )
                parsers, has_calls = {}, {}

                async def collect(parser_dict, request_obj, has_tool_calls):
                    return [
                        chunk
                        async for chunk in chat._process_tool_call_stream(
                            index=0,
                            delta=text,
                            parser_dict=parser_dict,
                            content={"meta_info": {"id": "test"}},
                            request=request_obj,
                            has_tool_calls=has_tool_calls,
                        )
                    ]

                chunks = asyncio.run(collect(parsers, request, has_calls))
                self.assertIsInstance(parsers[0], FunctionCallParser)
                self.assertTrue(has_calls[0])
                calls = [
                    json.loads(chunk.removeprefix("data: "))["choices"][0]["delta"][
                        "tool_calls"
                    ][0]
                    for chunk in chunks
                ]
                self.assertEqual(calls[0]["function"]["name"], "get_date")
                self.assertEqual(
                    "".join(call["function"]["arguments"] for call in calls), "{}"
                )

    def test_unregistered_backend_probe_is_cached(self):
        _ling3_native_structural_tag_available.cache_clear()
        try:
            with patch(
                "sglang.srt.function_call.ling3_detector.get_model_structural_tag",
                side_effect=ValueError("unknown model"),
            ) as builder:
                self.assertFalse(Ling3Detector().supports_structural_tag())
                self.assertFalse(Ling3Detector().supports_structural_tag())
                builder.assert_called_once()
        finally:
            _ling3_native_structural_tag_available.cache_clear()


if __name__ == "__main__":
    unittest.main()
