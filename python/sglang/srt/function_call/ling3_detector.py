import re
from functools import lru_cache
from typing import Any, Literal

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.function_call.base_format_detector import (
    StructuralTag,
    get_model_structural_tag,
)
from sglang.srt.function_call.core_types import (
    NativeToolConstraintError,
    StreamingParseResult,
)
from sglang.srt.function_call.glm4_moe_detector import (
    Glm4MoeDetector,
    _get_argument_schema,
    _json_schema_allows_null,
    get_argument_type,
)


@lru_cache(maxsize=1)
def _ling3_native_structural_tag_available() -> bool:
    if get_model_structural_tag is None:
        return False
    try:
        get_model_structural_tag(
            model="glm_4_7", tools=[], tool_choice="auto", reasoning=False
        )
        return True
    except ValueError:
        # Older XGrammar versions may import successfully without registering
        # the GLM XML format. Constrained requests must fail closed in that case.
        return False


class Ling3Detector(Glm4MoeDetector):
    """
    Detector for Ling3 tool calls.

    Ling3 uses the GLM-4.5 XML format, but model outputs may either put a newline
    after the function name, emit the first argument tag immediately, or close a
    no-argument tool call immediately.

    Constrained tool calls use XGrammar's native GLM XML schema serializer,
    which also accepts a newline after the function name. Like DeepSeek V4,
    this constrains the model's native format rather than a JSON call array;
    it does not change Ling's output protocol to DSML.
    """

    _STREAMING_PARTIAL_PATTERN = re.compile(
        r"<tool_call>\s*(.*?)"
        r"(?:(?:\\n|\n)\s*|(?=<arg_key>)|(?=</tool_call>))"
        r"(.*?)(</tool_call>|$)",
        re.DOTALL,
    )

    def __init__(self):
        super().__init__()
        self.func_detail_regex = re.compile(
            r"<tool_call>\s*(.*?)(?:(?:\\n|\n)\s*|(?=<arg_key>)|(?=</tool_call>))"
            r"(<arg_key>.*?)?</tool_call>",
            re.DOTALL,
        )

    def supports_structural_tag(self) -> bool:
        # The serving layer also uses this capability to select the native
        # parser for required/named tool calls, including streaming responses.
        return _ling3_native_structural_tag_available()

    def get_structural_tag_name(self) -> str:
        # Ling and GLM share <arg_key>/<arg_value> argument serialization.
        return "glm_4_7"

    def get_structural_tag(
        self,
        tools: list[Tool] | None = None,
        tool_choice: ToolChoice | Literal["auto", "required"] = "auto",
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> StructuralTag | None:
        try:
            if not self.supports_structural_tag():
                raise NativeToolConstraintError(
                    "Ling3 native XML tool constraints require XGrammar with "
                    "glm_4_7 support; refusing JSON fallback."
                )

            from xgrammar.structural_tag import (
                TagsWithSeparatorFormat,
                TriggeredTagsFormat,
            )

            tag = super().get_structural_tag(
                tools=tools,
                tool_choice=tool_choice,
                thinking_mode=thinking_mode,
                parallel_tool_calls=parallel_tool_calls,
            )
            # The builtin builder owns tool-choice normalization, strict schemas,
            # and the optional reasoning prefix. Adapt only the tool-call suffix.
            suffix = tag.format.elements[-1] if thinking_mode else tag.format
            if isinstance(suffix, (TriggeredTagsFormat, TagsWithSeparatorFormat)):
                suffix.stop_after_first = not parallel_tool_calls
            return tag
        except NativeToolConstraintError:
            raise
        except Exception as exc:
            raise NativeToolConstraintError(
                "Failed to build Ling3 native XML tool constraints; check the "
                "tool schema and XGrammar glm_4_7 support. Refusing JSON fallback."
            ) from exc

    def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult:
        result = super().detect_and_parse(text, tools)
        for tool_index, call in enumerate(result.calls):
            # The index identifies this call's position in the response, not
            # the matching tool's position in the request definitions.
            call.tool_index = tool_index
        return result

    def _parse_argument_pairs(
        self, pairs: list[tuple[str, str]], func_name: str, tools: list[Tool]
    ) -> dict[str, Any]:
        arguments = super()._parse_argument_pairs(pairs, func_name, tools)
        for key, value in pairs:
            key = key.strip()
            schema = _get_argument_schema(func_name, key, tools)
            if get_argument_type(func_name, key, tools) == "string" and not (
                _json_schema_allows_null(schema)
            ):
                # XML strings are raw, not JSON literals: preserve quotes,
                # "true", "null", and JSON-looking text just as streaming does.
                # Keep the existing null/quoted-string convention for nullable
                # schemas, where raw null denotes a typed null value.
                arguments[key] = value
        return arguments

    def parse_streaming_increment(
        self, new_text: str, tools: list[Tool]
    ) -> StreamingParseResult:
        # A coalesced streaming delta can contain multiple complete calls. The
        # parent consumes one per invocation; drain the remainder immediately,
        # since the final delta may not be followed by another parser invocation.
        self._buffer += new_text
        result = StreamingParseResult()
        while self._buffer:
            if not self.current_tool_name_sent:
                start = self._buffer.find(self.bot_token)
                if start > 0:
                    result.normal_text += self._buffer[:start]
                    self._buffer = self._buffer[start:]
            previous_buffer = self._buffer
            parsed = super().parse_streaming_increment("", tools)
            result.normal_text += parsed.normal_text
            result.calls.extend(parsed.calls)
            if self._buffer == previous_buffer:
                # An incomplete marker or call needs another input delta.
                break
        return result
