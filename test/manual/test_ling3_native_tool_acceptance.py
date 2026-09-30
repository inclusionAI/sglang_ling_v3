"""Bounded Ling XML acceptance against an existing, dedicated test server.

Run with the candidate source tree on PYTHONPATH. This sends its structural
tags to /generate and parses real model output with the candidate parser; it
does NOT replace the running server's /v1/chat/completions implementation.
Schemas are synthetic, not a replay of a production request.

Example (no server launch or model-weight loading in this script)::

    PYTHONPATH=python python test/manual/test_ling3_native_tool_acceptance.py \
        --base-url http://127.0.0.1:31000 \
        --model-path /home/models/Ling-3.0-plus-fp8
"""

import argparse
import json

import jsonschema
import requests
import xgrammar as xgr
from transformers import AutoTokenizer
from xgrammar.testing import _is_grammar_accept_string

from sglang.srt.entrypoints.openai.protocol import Function, Tool, ToolChoice
from sglang.srt.function_call.function_call_parser import FunctionCallParser


def make_tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="search",
                strict=True,
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "mode": {"type": "string", "enum": ["recent", "relevance"]},
                        "filters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"count": {"type": "integer"}},
                                "required": ["count"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["query", "mode", "filters"],
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


def run_case(
    session, args, tokenizer, tools, choice, stream, *, large=False, truncate=False
):
    parser = FunctionCallParser(tools, "ling3")
    constraint_type, tag = parser.get_structure_constraint(
        choice, parallel_tool_calls=False
    )
    assert constraint_type == "structural_tag"
    prompt = (
        '调用 search，query 为 "昨天的记录"，mode 为 "relevance"，'
        'filters 为 [{"count":2}]。不要解释，只输出工具调用。'
        if choice == "required"
        else "请调用 get_date 获取日期。不要解释，只输出工具调用。"
    )
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tools=[tool.model_dump() for tool in tools],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    payload = {
        "input_ids": input_ids,
        "stream": stream,
        "log_metrics": False,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 1 if truncate else 256,
            "skip_special_tokens": False,
            "structural_tag": tag.model_dump_json(by_alias=True),
        },
    }
    text, names, arguments = "", {}, {}
    with session.post(
        args.base_url + "/generate", json=payload, stream=stream, timeout=(5, 45)
    ) as response:
        response.raise_for_status()
        if stream:
            for line in response.iter_lines():
                if (
                    not line
                    or not line.startswith(b"data: ")
                    or line == b"data: [DONE]"
                ):
                    continue
                output = json.loads(line[6:])
                current = output["text"]
                assert current.startswith(text), "Expected cumulative /generate text"
                _, deltas = parser.parse_stream_chunk(current[len(text) :])
                for delta in deltas:
                    if delta.name:
                        names[delta.tool_index] = delta.name
                    arguments[delta.tool_index] = (
                        arguments.get(delta.tool_index, "") + delta.parameters
                    )
                text = current
        else:
            output = response.json()
            text = output["text"]

    _, calls = FunctionCallParser(tools, "ling3").parse_non_stream(text)
    finish = output["meta_info"]["finish_reason"]["type"]
    complete = _is_grammar_accept_string(xgr.Grammar.from_structural_tag(tag), text)
    if truncate:
        assert finish == "length" and not complete and not calls
    else:
        assert finish != "length" and complete and len(calls) == 1, (finish, text)
        call = calls[0]
        schema = next(
            tool.function.parameters
            for tool in tools
            if tool.function.name == call.name
        )
        parameters = json.loads(call.parameters)
        jsonschema.validate(parameters, schema)
        if choice == "required":
            assert call.name == "search"
            assert parameters == {
                "query": "昨天的记录",
                "mode": "relevance",
                "filters": [{"count": 2}],
            }, parameters
        else:
            assert call.name == choice.function.name and parameters == {}
        if stream:
            assert names == {0: call.name}
            assert json.loads(arguments[0]) == parameters
    print(
        json.dumps(
            {
                "choice": choice if isinstance(choice, str) else choice.function.name,
                "stream": stream,
                "large_synthetic_schema": large,
                "truncated": truncate,
                "prompt_tokens": output["meta_info"].get("prompt_tokens"),
                "completion_tokens": output["meta_info"].get("completion_tokens"),
                "ttft_s": output["meta_info"].get("ttft_latency"),
                "passed": True,
            }
        ),
        flush=True,
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--base-url", required=True, help="Dedicated test server, not production"
    )
    cli.add_argument("--model-path", required=True)
    args = cli.parse_args()
    args.base_url = args.base_url.rstrip("/")
    session = requests.Session()
    session.trust_env = False
    info = session.get(args.base_url + "/get_model_info", timeout=5)
    info.raise_for_status()
    assert info.json()["model_path"] == args.model_path, "Wrong model server"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    tools = make_tools()
    named = ToolChoice(type="function", function={"name": "get_date"})
    for stream in (False, True):
        for choice in ("required", named):
            run_case(session, args, tokenizer, tools, choice, stream)
        run_case(session, args, tokenizer, tools, "required", stream, truncate=True)
    tools[0].function.parameters["properties"]["query"]["description"] = (
        "检索语义与时间范围。" * 2048
    )
    for stream in (False, True):
        run_case(session, args, tokenizer, tools, "required", stream, large=True)


if __name__ == "__main__":
    main()
