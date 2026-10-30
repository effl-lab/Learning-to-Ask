# Copyright 2025 Subsolver Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Phase 2 RLVR reward for the trainable STUDENT.

The student's full (multi-turn) response is in ``solution_str``. Its final
answer is the LAST ``\\boxed{...}``. We score 1.0 iff that answer is
mathematically equivalent to ``ground_truth`` (MATH/AIME), else 0.0.

Unlike the Phase-1 reward, this function does NOT call the advisor/student
servers — the whole pipeline already happened inside the rollout (via the
``ask_advisor`` tool). The reward is pure verifiable correctness.

Load via:
    custom_reward_function.path=pkg://subsolver.student_reward
    custom_reward_function.name=compute_score
"""

from __future__ import annotations

import re
from typing import Any

from verl.utils.reward_score import math_reward

# Sentinel the ask_advisor tool returns when the whole-problem judge rejects the
# student's query (kept in sync with subsolver.ask_advisor_tool). It appears
# verbatim in the trajectory (as the tool-response text), so the reward can
# detect a rejected call without any extra plumbing.
_REJECT_SENTINEL = "It is not valid question."


# Real Hermes <tool_call> naming ask_advisor — a bare mention in prose/code is
# not a call (callgate/integration substring-hack postmortems, 2026-07-29 / 08-21).
_CALL_RE = re.compile(r"<tool_call>(?:(?!</tool_call>).)*ask_advisor", re.DOTALL)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """RLVR reward for the student, with a whole-problem-reject penalty.

    - score = 0.0 if the student's ask_advisor query was rejected as a
      whole-problem hand-over (sentinel present in the trajectory), REGARDLESS
      of the final answer — so a rejected wholesale call can never earn reward.
    - otherwise score = 1.0 iff the last \\boxed{...} matches ground_truth
      (MATH-equivalence), else 0.0.

    Returns a dict so verl aggregates ``rejected`` / ``called`` into the train
    metrics (reward_extra_info) — giving per-step reject-rate and call-rate
    visibility that the tool-side metrics don't surface.
    """
    s = solution_str or ""
    called = 1 if _CALL_RE.search(s) else 0
    rejected = 1 if _REJECT_SENTINEL in s else 0

    # NOTE: no reject penalty (removed). A rejected (whole-problem) call simply
    # gets no advisor help; reward = pure final-answer correctness regardless, so
    # calling is never strictly punished (avoids the call-abandonment collapse
    # seen in early runs). rejected/called kept only as metrics.
    if not s:
        score = 0.0
    else:
        try:
            score = float(math_reward.compute_score(s, str(ground_truth)))
        except Exception:  # noqa: BLE001
            score = 0.0

    return {"score": score, "rejected": rejected, "called": called}
