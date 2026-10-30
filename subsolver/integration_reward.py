# Copyright 2025 Subsolver Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Integration (Phase-3) reward: decision anchor x correctness bonus x cap gate.

The STUDENT is the trainee; the qmaker and advisor are frozen servers wired in
via ask_advisor_tool (enable_qmaker=true). Per rollout:

    called   = 1[a real <tool_call> naming ask_advisor was emitted]
    need     = extra_info["needAdvisor"]        (threshold-t labels, budget anchor)
    capped   = 1[the tool response carries the advisor-cap sentinel]
    correct  = 1[last \\boxed{} == ground_truth]

    score = 1[called == need] * (1 + correct) * 1[not capped]   in {0, 1, 2}

Design notes:
- `1[called == need]` keeps the decision anchor of the call-gate reward in the
  objective.
- `(1 + correct)` keeps the call channel alive — plain `acc * decision`
  collapsed to never-call in early runs.
- The cap gate replaces an earlier `1[not whole]` gate: the whole-problem judge
  is retired; instead an advisor answer truncated at the token cap zeroes the
  rollout (advisor budget priced into the reward). Vacuous 1 for non-callers,
  so the abstain channel is unaffected (no call-abandonment collapse).

Load via:
    custom_reward_function.path=pkg://subsolver.integration_reward
    custom_reward_function.name=compute_score
"""

from __future__ import annotations

import os
import re
from typing import Any

from verl.utils.reward_score import math_reward

# Keep in sync with subsolver/ask_advisor_tool.py.
_CAP_SENTINEL = "[advisor answer truncated at token cap]"

# A call counts only as a real <tool_call> block naming ask_advisor — bare
# mentions in prose were gameable (integration substring-hack postmortem).
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
    if _TOOL_FORMAT == "gemma4":  # Gemma line (2026-10-29): a tool turn in the trajectory is the call
        from subsolver.gemma4_toolcall import gemma4_called_multiturn
        return gemma4_called_multiturn(s, extra_info)
    return bool(_CALL_RE.search(s))


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    s = solution_str or ""
    extra_info = extra_info or {}

    called = 1 if _is_real_call(s, extra_info) else 0
    need = int(extra_info.get("needAdvisor", 1))
    capped = 1 if (called and _CAP_SENTINEL in s) else 0

    correct = 0.0
    graded = s
    if _TOOL_FORMAT == "qwen3_xml":  # thinking students: only the last turn's post-</think> answer
        from subsolver.qwen3_toolcall import final_answer_text
        graded = final_answer_text(s)
    if _TOOL_FORMAT == "gemma4":  # Gemma: grade only the student's text after the advisor's tool response
        from subsolver.gemma4_toolcall import gemma4_after_tool_response
        graded = gemma4_after_tool_response(s)
    from subsolver.pipeline import answer_format, parse_final_answer
    if graded and answer_format() == "final_answer":  # Gemma line: grade the last 'Final Answer:' line
        a = parse_final_answer(graded)
        graded = ("\\boxed{" + a + "}") if a else ""
    if graded:
        try:
            correct = float(math_reward.compute_score(graded, str(ground_truth)))
        except Exception:  # noqa: BLE001
            correct = 0.0

    decision = 1.0 if called == need else 0.0
    score = decision * (1.0 + correct) * (0.0 if capped else 1.0)

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
        "group_id": str(extra_info.get("group_id", "")),
    }
