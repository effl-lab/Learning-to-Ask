"""vLLM tool-parser plugin: Llama-3.1 JSON calls that may follow reasoning.

vLLM 0.11's built-in ``llama3_json`` parser regex-searches the FIRST ``{...}`` in the
output and drops the content: with reasoning before the call it grabs LaTeX braces
(``\\frac{a}{b}``), fails to json-decode and reports no call. This plugin applies the
shared rule in subsolver/llama_toolcall.py (the turn must END with the JSON call) and
returns the reasoning as ``content`` alongside the structured tool call.

Serve with:
    vllm serve <llama model dir> --enable-auto-tool-choice \\
        --tool-parser-plugin subsolver/vllm_llama_reasoning_parser.py \\
        --tool-call-parser llama3_json_reasoning

Non-streaming only (every subsolver client uses non-streaming chat completions);
the streaming hook passes text through as content.
"""

import importlib.util
import json
import os
from collections.abc import Sequence
from typing import Union

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest, DeltaMessage,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager)

_spec = importlib.util.spec_from_file_location(
    "subsolver_llama_toolcall",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama_toolcall.py"))
_ltc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ltc)


@ToolParserManager.register_module("llama3_json_reasoning")
class Llama3JsonReasoningToolParser(ToolParser):

    def extract_tool_calls(
            self, model_output: str,
            request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        content, calls = _ltc.parse_llama_tool_call(model_output)
        if not calls:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[],
                                                content=model_output)
        tool_calls = [
            ToolCall(id=make_tool_call_id(), type="function",
                     function=FunctionCall(name=name,
                                           arguments=json.dumps(args, ensure_ascii=False)))
            for name, args in calls
        ]
        return ExtractedToolCallInformation(tools_called=True, tool_calls=tool_calls,
                                            content=content or None)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        return DeltaMessage(content=delta_text)
