# Copyright 2025 Subsolver Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Call-gate reward: decision matching ONLY.

The rollout runs with max_assistant_turns=1, so a calling trace ends exactly at
the emitted <tool_call> (the tool is never executed) and a non-calling trace is
a full solo solve. Reward:

    called == needAdvisor -> 1.0      (call when needed / abstain when not)
    otherwise             -> 0.0

needAdvisor comes from extra_info (precomputed solo avg@8 <= 0.25 on the train
set; see measure_train_avg8.py). Answer correctness does NOT enter the reward
(per design); it is surfaced as `acc` for monitoring only.

Load via:
    custom_reward_function.path=pkg://subsolver.callgate_reward
    custom_reward_function.name=compute_score
"""

from __future__ import annotations

import os
import re
from typing import Any

from subsolver.llama_toolcall import has_ask_advisor_call
from verl.utils.reward_score import math_reward


# A call counts only as a real Hermes <tool_call> block naming ask_advisor.
# The old substring test ("ask_advisor" in s) was gamed on 2026-08-20/21: under the
# code-domain student prompt the t12 gate learned to write
# `result = ask_advisor(...)` INSIDE its Python program (0.69 "called" on codegen
# val, 0.00 real tool calls) and the harvest kept 1.7% of codegen rows. Same
# regex as integration_reward.py (integration substring-hack postmortem).
_CALL_RE = re.compile(r"<tool_call>(?:(?!</tool_call>).)*ask_advisor", re.DOTALL)

# Llama line (2026-10-04): SUBSOLVER_TOOL_FORMAT=llama3_json switches to the Llama-3.1
# JSON rule (turn must END with {"name": "ask_advisor", "parameters": {...}}), shared
# with the verl rollout parser and the vLLM plugin via subsolver/llama_toolcall.py.
# Default stays Hermes, so Qwen runs are unchanged.
_TOOL_FORMAT = os.environ.get("SUBSOLVER_TOOL_FORMAT", "hermes")


def is_real_call(s: str) -> bool:
    if _TOOL_FORMAT == "llama3_json":
        return has_ask_advisor_call(s)
    if _TOOL_FORMAT == "gemma4":  # Gemma 4 native call, seen through a skip_special_tokens decode (gemma4_toolcall.py)
        from subsolver.gemma4_toolcall import has_ask_advisor_call_stripped
        return has_ask_advisor_call_stripped(s)
    if _TOOL_FORMAT == "qwen3_xml":  # Qwen3.5 thinking: complete call after </think> (qwen3_toolcall.py)
        from subsolver.qwen3_toolcall import has_ask_advisor_call as q35_called
        return q35_called(s)
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
    called = 1 if is_real_call(s) else 0
    need = int(extra_info.get("needAdvisor", 0))

    score = 1.0 if called == need else 0.0

    # monitoring only (never enters the reward): final-answer correctness of
    # non-calling traces (calling traces have no final answer by construction).
    acc = 0.0
    if not called and s:
        try:
            acc = float(math_reward.compute_score(s, str(ground_truth)))
        except Exception:  # noqa: BLE001
            acc = 0.0

    return {
        "score": score,
        "called": called,
        "need": need,
        # decision confusion cells (aggregated by the trainer into means):
        "tp": 1 if (called and need) else 0,          # needed & called
        "fp": 1 if (called and not need) else 0,      # not needed but called
        "fn": 1 if ((not called) and need) else 0,    # needed but didn't call
        "tn": 1 if ((not called) and (not need)) else 0,
        "acc": acc,
        "group_id": str(extra_info.get("group_id", "")),
    }
