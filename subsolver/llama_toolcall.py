"""Llama-3.1 JSON tool-call parsing that allows reasoning BEFORE the call.

Llama-3.1 (with subsolver/chat_templates/llama31_subsolver.jinja) calls a tool by
emitting ``{"name": ..., "parameters": {...}}``, optionally prefixed by
``<|python_tag|>``. Meta's own format expects the JSON to be the whole message;
the subsolver protocol lets the student reason first and call at the end, so a
turn may be ``<reasoning text>\\n\\n{"name": "ask_advisor", "parameters": {...}}``.

Rule (identical everywhere this module is used — verl rollout parser, vLLM serving
plugin, gate/integration reward detectors, eval):

    a turn carries a tool call  iff  it ENDS with one JSON object (only whitespace /
    end-of-turn special tokens after it) that has a string "name" (or "function")
    and a dict "parameters" (or "arguments"); other keys ("type") and key order free.

Everything before that object is the turn's content (the reasoning). A JSON object
in the middle of the text, a bare mention of ``ask_advisor``, or ``{...}`` from LaTeX
never counts — the same anti-substring-hack property as the Hermes ``<tool_call>``
regex used for Qwen. Unescaped LaTeX backslashes inside the JSON (``\\frac``) are
recovered leniently, as in the Hermes parser.

No third-party imports: the vLLM plugin loads this file by path.
"""

from __future__ import annotations

import json
import re

SPECIAL_TOKENS = ("<|eot_id|>", "<|eom_id|>", "<|end_of_text|>", "<|python_tag|>")


def _strip_specials(text: str) -> str:
    for t in SPECIAL_TOKENS:
        text = text.replace(t, "")
    return text


def _decode_tail(s: str):
    """json-decode one object spanning ALL of s (modulo trailing whitespace)."""
    dec = json.JSONDecoder()
    for cand in (s, s.replace("\\", "\\\\")):
        try:
            obj, end = dec.raw_decode(cand)
        except json.JSONDecodeError:
            continue
        if cand[end:].strip() == "":
            return obj
    return None


def parse_llama_tool_call(text: str):
    """Return (content, calls) where calls = [(name, arguments_dict)] (0 or 1 entry)."""
    clean = _strip_specials(text).rstrip()
    if not clean.endswith("}"):
        return clean, []
    # Walk '{' positions from the end backwards; the first one that decodes to a call
    # object ending exactly at the end of the turn wins. Key order is free (Llama often
    # emits {"type": "function", "name": ..., "parameters": ...}) and the JSON may span
    # lines; inner objects ({"query": ...}) never end at the turn's end, so they fail.
    pos = clean.rfind("{")
    while pos != -1:
        obj = _decode_tail(clean[pos:])
        if isinstance(obj, dict):
            # "name" per Meta's format; Llama sometimes writes "function" instead
            name = obj.get("name") if isinstance(obj.get("name"), str) else obj.get("function")
            args = obj.get("parameters", obj.get("arguments"))
            if isinstance(name, str) and isinstance(args, dict):
                return clean[:pos].rstrip(), [(name, args)]
        pos = clean.rfind("{", 0, pos)
    return clean, []


def has_ask_advisor_call(text: str) -> bool:
    """Gate/integration 'called' detector for Llama turns."""
    _, calls = parse_llama_tool_call(text)
    return any(name == "ask_advisor" for name, _ in calls)


_TOOL_TURN_MARK = 'ipython\n\n{"output":'


def llama_called_multiturn(response_str: str, extra_info: dict | None = None) -> bool:
    """'called' for a multi-turn integration rollout (Llama).

    verl executes the tool only when its rollout parser (the same rule as above)
    found a call, so a tool turn in the trajectory IS the call: num_turns > 2
    (user, assistant, tool, ...). Without num_turns, fall back to the text: the part
    before the first rendered tool turn (special tokens are skipped in the reward
    decode, leaving 'ipython\\n\\n{"output": ...') must end with an ask_advisor call.
    """
    nt = (extra_info or {}).get("num_turns")
    try:
        if nt is not None:
            return int(nt) > 2
    except (TypeError, ValueError):
        pass
    head = response_str.split(_TOOL_TURN_MARK, 1)[0]
    if head.endswith("ipython"):
        head = head[: -len("ipython")]
    return has_ask_advisor_call(head) if _TOOL_TURN_MARK in response_str else False


def _selftest() -> None:
    call = '{"name": "ask_advisor", "parameters": {"query": "What is $\\frac{1}{2}$?"}}'
    assert parse_llama_tool_call(call)[1][0][0] == "ask_advisor"
    c, calls = parse_llama_tool_call("I tried x. Need help.\n\n" + call + "<|eot_id|>")
    assert c == "I tried x. Need help." and calls[0][1]["query"].startswith("What"), (c, calls)
    assert has_ask_advisor_call("<|python_tag|>" + call + "<|eom_id|>")
    # LaTeX braces in reasoning, no call
    assert not has_ask_advisor_call("so $\\frac{a}{b}$ gives \\boxed{3}")
    # call in the middle, then an answer -> not a call
    assert not has_ask_advisor_call(call + "\nActually the answer is \\boxed{2}.")
    # bare mention
    assert not has_ask_advisor_call("I could call ask_advisor here but won't.")
    # other tool name parses but is not ask_advisor
    assert not has_ask_advisor_call('{"name": "python", "parameters": {"code": "1"}}')
    # key order free, multi-line JSON
    assert has_ask_advisor_call('{\n    "type": "function",\n    "name": "ask_advisor",\n    "parameters": {\n        "query": "q"\n    }\n}')
    c, calls = parse_llama_tool_call('Work {x}.\n{"type": "function", "name": "ask_advisor", "parameters": {"query": "a {b}"}}')
    assert c == "Work {x}." and calls[0][1]["query"] == "a {b}", (c, calls)
    # "function" as the name key
    assert has_ask_advisor_call('{"function": "ask_advisor", "parameters": {"query": "q"}}')
    # 'arguments' key accepted
    assert has_ask_advisor_call('text {"name": "ask_advisor", "arguments": {"query": "q"}}')
    # multi-turn (reward decode skips special tokens)
    traj = "I tried.\n\n" + call + 'ipython\n\n{"output": "0.5"}assistant\n\nSo \\boxed{1}'
    assert llama_called_multiturn(traj) and llama_called_multiturn(traj, {"num_turns": 4})
    assert not llama_called_multiturn("So \\boxed{1}") and not llama_called_multiturn("x", {"num_turns": 2})
    print("llama_toolcall selftest OK")


if __name__ == "__main__":
    _selftest()
