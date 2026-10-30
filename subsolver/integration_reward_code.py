# Copyright 2025 Subsolver Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Integration reward for the CODE domain: decision anchor x correctness bonus.

Same objective as subsolver/integration_reward.py (math), for codegen rows:

    called   = 1[a real <tool_call> naming ask_advisor was emitted]
    need     = extra_info["needAdvisor"]                (t-threshold gate labels)
    correct  = 1[the student's FINAL assistant turn is a program that passes the
                 domain grader (coder1 execution) for extra_info["guru_data_source"]]

    score = 1[called == need] * (1 + correct)           in {0, 1, 2}

Differences from the math reward, all forced by the code domain:
* Grading = execution via subsolver.domain_reward (coder1 for codegen__*), not
  math_reward on a \\boxed{} payload.
* Only the LAST assistant turn is graded. verl hands the reward the decoded
  multi-turn response, tool response included ("...</tool_call>user\\n<tool_response>
  ...</tool_response>\\nassistant\\n<final>"), and coder1 joins EVERY fenced block in the
  string — so without slicing, the advisor's program inside the tool response would be
  scored as the student's answer and "call, then write nothing" would be rewarded.
* Cap gate OFF by default: an advisor reply truncated at the
  token cap is logged (cap_exceeded) but does not zero the rollout — the student cannot
  control the advisor's length, and at cap 2048 ~18% of code replies are truncated.
  SUBSOLVER_INTEG_CAP_GATE=1 restores the math behaviour.
* Async with bounded thread-pool grading (SUBSOLVER_GRADE_CONCURRENCY, default 12) so
  execution grading of a whole rollout batch overlaps instead of serialising.

Extra logged fields (reward_extra_info -> per-step means): called, need, decision,
correct, cap_exceeded, tp/fp/fn/tn, final_has_code, copied_from_advisor, final_chars.

Load via:
    custom_reward_function.path=pkg://subsolver.integration_reward_code
    custom_reward_function.name=compute_score
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any

from .domain_reward import is_correct as domain_is_correct

# Keep in sync with subsolver/ask_advisor_tool.py.
_CAP_SENTINEL = "[advisor answer truncated at token cap]"

# A call counts only as a real <tool_call> block naming ask_advisor — bare mentions in
# prose / inside a program were gameable (integration + callgate substring-hack postmortems).
_CALL_RE = re.compile(r"<tool_call>(?:(?!</tool_call>).)*ask_advisor", re.DOTALL)


# Llama line (2026-10-04): with SUBSOLVER_TOOL_FORMAT=llama3_json a call is a real tool
# turn in the trajectory (verl executes the tool only when its llama3_json_reasoning
# parser found one) — see subsolver/llama_toolcall.py. Default stays Hermes.
_TOOL_FORMAT = os.environ.get("SUBSOLVER_TOOL_FORMAT", "hermes")


def _is_real_call(s: str, extra_info: dict | None) -> bool:
    if _TOOL_FORMAT == "llama3_json":
        from subsolver.llama_toolcall import llama_called_multiturn
        return llama_called_multiturn(s, extra_info)
    if _TOOL_FORMAT == "qwen3_xml":
        from subsolver.qwen3_toolcall import qwen3_called_multiturn
        return qwen3_called_multiturn(s, extra_info)
    return bool(_CALL_RE.search(s))
_FENCE = re.compile(r"```(?:\w+)?\n(.*?)\n```", re.DOTALL)
_TOOL_RESP_END = "</tool_response>"

_sem: asyncio.Semaphore | None = None


def _cap_gate_enabled() -> bool:
    return os.environ.get("SUBSOLVER_INTEG_CAP_GATE", "0") not in ("", "0", "false", "False")


def _get_sem() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(int(os.environ.get("SUBSOLVER_GRADE_CONCURRENCY", "12") or 12))
    return _sem


def split_turns(s: str) -> tuple[str, str]:
    """(tool_response_segment, final_assistant_turn).

    The decoded rollout renders a tool response as
    ``user\\n<tool_response>\\n...\\n</tool_response>\\nassistant\\n<continuation>``.
    Everything after the LAST </tool_response> (minus the role marker) is the student's
    final turn; the text between the first <tool_response> and that marker is what the
    advisor supplied (used only to detect verbatim copying). Without a tool response the
    whole string is the student's own answer.
    """
    if _TOOL_RESP_END not in s:
        return "", s
    head, tail = s.rsplit(_TOOL_RESP_END, 1)
    tail = re.sub(r"^\s*assistant\s*\n?", "", tail, count=1)
    i = head.find("<tool_response>")
    return (head[i:] if i >= 0 else head), tail


def _grade(ds: str, text: str, ground_truth: Any, extra_info: dict[str, Any]) -> float:
    try:
        return 1.0 if domain_is_correct(ds, text, ground_truth, extra_info) else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    s = solution_str or ""
    extra_info = dict(extra_info or {})

    called = 1 if _is_real_call(s, extra_info) else 0
    need = int(extra_info.get("needAdvisor", 1))
    tool_seg, final = split_turns(s)
    capped = 1 if (called and _CAP_SENTINEL in tool_seg) else 0

    final_fences = _FENCE.findall(final)
    final_has_code = 1 if final_fences else 0
    adv_fences = _FENCE.findall(tool_seg) if tool_seg else []
    copied = 1 if (final_fences and adv_fences and final_fences[-1].strip() == adv_fences[-1].strip()) else 0

    # Grader dispatch wants the Guru data_source (codegen__taco -> coder1), not the
    # row-level "subsolver_student" tag.
    ds = str(extra_info.get("guru_data_source") or data_source)
    correct = 0.0
    if final_has_code:
        loop = asyncio.get_running_loop()
        async with _get_sem():
            correct = await loop.run_in_executor(None, _grade, ds, final, ground_truth, extra_info)

    decision = 1.0 if called == need else 0.0
    score = decision * (1.0 + correct)
    if _cap_gate_enabled() and capped:
        score = 0.0

    return {
        "score": score,
        "called": called,
        "need": need,
        "decision": decision,
        "correct": correct,
        "cap_exceeded": capped,
        "tp": 1 if (called and need) else 0,
        "fp": 1 if (called and not need) else 0,
        "fn": 1 if ((not called) and need) else 0,
        "tn": 1 if ((not called) and (not need)) else 0,
        "final_has_code": final_has_code,
        "copied_from_advisor": copied,
        "final_chars": len(final),
        "group_id": str(extra_info.get("group_id", "")),
    }
