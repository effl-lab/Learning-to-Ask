"""What counts as an ask_advisor call for Gemma 4 students (gemma_e2b line, 2026-10-26).

Gemma 4 calls tools with its own special tokens (all special in the tokenizer):

    <|tool_call>call:ask_advisor{query:<|"|>QUESTION<|"|>}<tool_call|>

Two views of the same response:
  * the verl rollout parser decodes WITH special tokens -> parse_gemma4_tool_call (the vLLM gemma4 parser's standard
    pattern; a bare `call:name{...}` without <|tool_call> does not count);
  * the reward manager decodes with skip_special_tokens=True, which leaves `call:ask_advisor{query:QUESTION}` ->
    has_ask_advisor_call_stripped requires that call to END the response (the turn stops at a call) with a
    non-empty query.
"""
from __future__ import annotations

import re

_CALL_RE = re.compile(r"<\|tool_call>call:(\w+)\{(.*?)\}(?:<tool_call\|>|<turn\|>)", re.DOTALL)
_ARG_RE = re.compile(r'(\w+):<\|"\|>(.*?)<\|"\|>', re.DOTALL)



def parse_gemma4_tool_call(text: str) -> tuple[str, list[tuple[str, dict]]]:
    """(content, calls) from a decode that kept special tokens."""
    calls = []
    for m in _CALL_RE.finditer(text):
        calls.append((m.group(1), {k: v for k, v in _ARG_RE.findall(m.group(2))}))
    return _CALL_RE.sub("", text).strip(), calls


def has_ask_advisor_call_stripped(text: str) -> bool:
    """Reward-side detector on a skip_special_tokens decode: the LAST `call:ask_advisor{query:...}` must close the
    response. The query may contain LaTeX braces, so 'closes the response' = the query's braces stay balanced
    (never below zero, zero at the end) — text after the call's own `}` breaks that."""
    t = text or ""
    i = t.rfind("call:ask_advisor{query:")
    if i < 0:
        return False
    rest = t[i + len("call:ask_advisor{query:"):].rstrip()
    if not rest.endswith("}"):
        return False
    q, depth = rest[:-1], 0
    for ch in q:
        depth += (ch == "{") - (ch == "}")
        if depth < 0:
            return False
    return depth == 0 and bool(q.strip())


# ---- integration (multi-turn) views, 2026-10-29 -------------------------------------------------------------------
_THOUGHT_RE = re.compile(r"<\|channel>thought\n?(.*?)(?:<channel\|>|$)", re.DOTALL)
_SPECIAL_RE = re.compile(r'<\|"\|>|<\|[a-z_]+\|?>|<[a-z_]+\|>|<(?:bos|eos|pad|unk|mask)>')
_TOOL_RESPONSE_OPEN = "response:ask_advisor{"


def gemma4_student_reasoning(raw: str) -> str:
    """Reasoning handed to the frozen qmaker at tool-call time, from a decode that KEPT special tokens.

    Same composition as the P3 harvest prefix (vLLM gemma4 reasoning parser -> message 'reasoning' = thought channel
    without its 'thought' header, 'content' = visible text before the call; build_guru_qmaker_parquet joins them with a
    blank line), so the qmaker sees in-loop prompts shaped like its training prompts."""
    t = raw or ""
    thought = "\n\n".join(m.group(1).strip() for m in _THOUGHT_RE.finditer(t) if m.group(1).strip())
    visible = _CALL_RE.sub("", _THOUGHT_RE.sub("", t))
    visible = _SPECIAL_RE.sub("", visible).strip()
    return "\n\n".join(p for p in (thought, visible) if p)


def gemma4_called_multiturn(response_str: str, extra_info: dict | None = None) -> bool:
    """'called' for an integration rollout: verl runs the tool only when the gemma4_subsolver parser found a call, so a
    tool turn in the trajectory IS the call (num_turns > 2). Without num_turns, fall back to a tool response block in the
    skip_special_tokens decode."""
    nt = (extra_info or {}).get("num_turns")
    if nt is not None:
        try:
            return int(nt) > 2
        except (TypeError, ValueError):
            pass
    return _TOOL_RESPONSE_OPEN in (response_str or "")


def gemma4_after_tool_response(text: str) -> str:
    """Student text AFTER the last ask_advisor tool response in a skip_special_tokens decode (the gradable continuation),
    the whole text when there was no tool response, '' when the response block never closes. The block is
    `response:ask_advisor{value:ADVISOR TEXT}`; its end is found by brace depth because the advisor text carries LaTeX
    braces. Grading only this part keeps a 'Final Answer:' line inside the advisor's reply from being scored as the
    student's answer."""
    t = text or ""
    i = t.rfind(_TOOL_RESPONSE_OPEN)
    if i < 0:
        return t
    depth = 1
    for j in range(i + len(_TOOL_RESPONSE_OPEN), len(t)):
        depth += (t[j] == "{") - (t[j] == "}")
        if depth == 0:
            return t[j + 1:]
    return ""


def _selftest() -> None:
    raw = 'I derived x=2.\n<|tool_call>call:ask_advisor{query:<|"|>What is y?<|"|>}<tool_call|>'
    content, calls = parse_gemma4_tool_call(raw)
    assert calls == [("ask_advisor", {"query": "What is y?"})] and content == "I derived x=2."
    assert parse_gemma4_tool_call("call:ask_advisor{query:x}")[1] == []          # no <|tool_call> token
    stripped = "I derived x=2.\ncall:ask_advisor{query:What is y?}"
    assert has_ask_advisor_call_stripped(stripped)
    assert not has_ask_advisor_call_stripped("call:ask_advisor{query:  }")
    assert not has_ask_advisor_call_stripped("call:ask_advisor{query:What?} and then \\boxed{3}")  # not at the end
    assert not has_ask_advisor_call_stripped("<execute_tool>\nask_advisor{query:What?}")
    assert has_ask_advisor_call_stripped("x\ncall:ask_advisor{query:Is \\frac{a}{b} > 1?}")  # LaTeX braces
    # integration views
    raw_turn = '<|channel>thought\nI tried x=1.\n<channel|>So far so good.<|tool_call>call:ask_advisor{query:<|"|>Is y odd?<|"|>}<tool_call|>'
    assert gemma4_student_reasoning(raw_turn) == "I tried x=1.\n\nSo far so good."
    assert gemma4_student_reasoning('<|channel>thought\nOnly thinking.<channel|><|tool_call>call:ask_advisor{query:<|"|>q<|"|>}<turn|>') == "Only thinking."
    assert gemma4_called_multiturn("anything", {"num_turns": 4}) and not gemma4_called_multiturn("x", {"num_turns": 2})
    traj = "thought\nT\ncall:ask_advisor{query:Q}response:ask_advisor{value:Use \\frac{a}{b}. Final Answer: 7}thought\nok\nFinal Answer: 5"
    assert gemma4_called_multiturn(traj) and gemma4_after_tool_response(traj) == "thought\nok\nFinal Answer: 5"
    assert gemma4_after_tool_response("no call. Final Answer: 2") == "no call. Final Answer: 2"
    assert gemma4_after_tool_response("call:ask_advisor{query:Q}response:ask_advisor{value:open \\{ never closed") == ""
    print("gemma4_toolcall selftest ok")


if __name__ == "__main__":
    _selftest()
