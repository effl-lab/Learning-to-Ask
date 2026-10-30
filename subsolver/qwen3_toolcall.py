"""What counts as an ask_advisor call for Qwen3.5 thinking students (q35 line, 2026-10-25).

Qwen3.5 calls tools in XML after it closes its thinking:

    ...reasoning...</think>\n\n<tool_call>\n<function=ask_advisor>\n<parameter=query>\nQ\n</parameter>\n</function>\n</tool_call>

The generation prompt already ends with "<think>\n", so a response is "reasoning </think> answer-or-call".
Rule shared by the verl rollout parser (qwen3_xml_think), the reward detectors and the
ask_advisor tool: only COMPLETE <tool_call> blocks AFTER the last </think> count. A response whose
thinking never closed has no call (and no answer), and a <tool_call> written inside the thinking is
not executed. Note: none of these tags is a special token, so they survive skip_special_tokens decodes.
"""
from __future__ import annotations

import re

_THINK_END = "</think>"
_CALL_RE = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


def split_think(text: str) -> tuple[str, str | None]:
    """(reasoning, after) — after is None when the thinking never closed.

    Non-thinking runs (SUBSOLVER_Q35_THINKING=0, Qwen3.5-4B no-think line): the empty <think></think> block is
    part of the generation PROMPT, so a first-turn response carries no </think> at all — the whole text is
    the answer / call (later turns still show the prompt's empty block and split as usual)."""
    if _THINK_END not in text:
        import os
        if os.environ.get("SUBSOLVER_Q35_THINKING", "1") == "0":
            return "", text
        return text.replace("<think>", "", 1).strip(), None
    head, tail = text.rsplit(_THINK_END, 1)
    return head.replace("<think>", "", 1).strip(), tail


def parse_qwen3_tool_call(text: str) -> tuple[str, list[tuple[str, dict]]]:
    """Return (content, calls): content = post-think text minus the call blocks, calls = [(name, args)]."""
    _, after = split_think(text)
    if after is None:
        return "", []
    calls = []
    for m in _CALL_RE.finditer(after):
        args = {k.strip(): v for k, v in _PARAM_RE.findall(m.group(2))}
        calls.append((m.group(1).strip(), args))
    return _CALL_RE.sub("", after).strip(), calls


def has_ask_advisor_call(text: str) -> bool:
    """Gate 'called' detector: a complete post-think ask_advisor call with a non-empty query."""
    return any(name == "ask_advisor" and (args.get("query") or "").strip()
               for name, args in parse_qwen3_tool_call(text)[1])


def student_reasoning(text: str) -> str:
    """The thinking that preceded the call — what the qmaker sees as 'student reasoning so far'
    (same SUBSOLVER_PREFIX_REASONING_MAX_CHARS tail cap as message_reasoning)."""
    # thinking runs: the <think> part (the visible text before a call is usually empty); non-thinking runs
    # reason in the visible text, which then is the whole of it
    think = message_reasoning({"reasoning": split_think(text)[0]})
    visible = message_reasoning({"reasoning": parse_qwen3_tool_call(text)[0]})
    return "\n\n".join(p for p in (think, visible) if p)


def message_reasoning(msg: dict) -> str:
    """Thinking carried by a chat message (vLLM 0.26 returns and accepts it as 'reasoning'; older
    servers / templates use 'reasoning_content'), tail-truncated to SUBSOLVER_PREFIX_REASONING_MAX_CHARS
    when set — a thinking prefix can run to 32k tokens, far past the qmaker prompt budget, and the
    reasoning right before the call is what the question is about. Unset = no truncation."""
    import os
    r = msg.get("reasoning") or msg.get("reasoning_content") or ""
    cap = int(os.environ.get("SUBSOLVER_PREFIX_REASONING_MAX_CHARS") or 0)
    if cap and len(r) > cap:
        r = "[... earlier reasoning omitted ...]\n" + r[-cap:]
    return r.strip()


def final_answer_text(text: str) -> str:
    """The gradable answer: post-</think> text of the LAST assistant turn ('' if that turn's thinking
    never closed). Keeps a \\boxed{} written mid-thinking — or in an earlier turn — from scoring."""
    last_turn = text.rsplit("</tool_response>", 1)[-1]
    return split_think(last_turn)[1] or ""


_TOOL_TURN_MARK = "<tool_response>"


def qwen3_called_multiturn(response_str: str, extra_info: dict | None = None) -> bool:
    """'called' for a multi-turn rollout: verl only runs the tool when the rollout parser found a
    call, so a tool turn in the trajectory IS the call (num_turns > 2: user, assistant, tool, ...).
    Without num_turns, fall back to the text before the first tool response."""
    nt = (extra_info or {}).get("num_turns")
    if nt is not None:
        try:
            return int(nt) > 2
        except (TypeError, ValueError):
            pass
    if _TOOL_TURN_MARK not in response_str:
        return False
    return has_ask_advisor_call(response_str.split(_TOOL_TURN_MARK, 1)[0])


def _selftest() -> None:
    call = "<tool_call>\n<function=ask_advisor>\n<parameter=query>\nWhat is x?\n</parameter>\n</function>\n</tool_call>"
    t = "I tried factoring.\n</think>\n\n" + call
    assert has_ask_advisor_call(t) and student_reasoning(t) == "I tried factoring."
    assert parse_qwen3_tool_call(t)[1] == [("ask_advisor", {"query": "What is x?"})]
    assert not has_ask_advisor_call("still thinking " + call)          # thinking never closed
    assert not has_ask_advisor_call("plan: " + call + "</think>\n\nSo \\boxed{1}")  # call inside thinking
    assert not has_ask_advisor_call("x</think>\n\n" + call.replace("ask_advisor", "search"))
    assert not has_ask_advisor_call("x</think>\n\n" + call.replace("What is x?", " "))
    assert not has_ask_advisor_call("x</think>\n\n<tool_call>\n<function=ask_advisor>\n<parameter=query>\nq")  # unclosed
    traj = t + "<|im_end|>\n<|im_start|>user\n<tool_response>\n0.5\n</tool_response><|im_end|>\nassistant\n<think>\nok</think>\n\n\\boxed{1}"
    assert qwen3_called_multiturn(traj) and qwen3_called_multiturn("x", {"num_turns": 4})
    assert not qwen3_called_multiturn("a</think>\\boxed{1}") and not qwen3_called_multiturn("x", {"num_turns": 2})
    assert final_answer_text(traj).strip() == "\\boxed{1}"
    assert final_answer_text("so \\boxed{7} hmm wait") == ""                                  # unclosed: no answer
    assert final_answer_text(t + "<tool_response>\n0.5\n</tool_response>\n<think>\n\\boxed{2} maybe") == ""
    print("qwen3_toolcall selftest ok")


if __name__ == "__main__":
    _selftest()
