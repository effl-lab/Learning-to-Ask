"""Segmented integration reward for the CODE domain.

Same trajectory parsing/grading as integration_reward_code (real-<tool_call> detection,
final-assistant-turn-only coder1 grading, cap gate off by default), but instead of one
trajectory scalar it emits TWO components that `grpo_segmented` (core_algos) applies to
different response segments:

    seg_pre  = 1[called == need] + W * 1[correct]     -> tokens BEFORE the tool response
    seg_post = 1[correct]                             -> continuation tokens AFTER it

W = SUBSOLVER_SEG_PRE_CORRECT_W (default 0.3). For a no-call trajectory there is no
post segment; the whole response is trained on seg_pre (the estimator handles this via
the loss-mask boundary). `score` (= seg_pre + seg_post) is only a logging/val scalar.

The whether-signal is preserved on the decision tokens while the continuation gets a
pure correctness objective — targeting the measured 29% "advisor-correct -> student
lost" handoff loss without letting the gate reward leak into the continuation.

Load via:
    custom_reward_function.path=pkg://subsolver.integration_reward_seg
    custom_reward_function.name=compute_score
    algorithm.adv_estimator=grpo_segmented
"""

from __future__ import annotations

import os
from typing import Any

from .integration_reward_code import compute_score as _base_compute_score

_PRE_W = float(os.environ.get("SUBSOLVER_SEG_PRE_CORRECT_W", "0.3") or 0.3)


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    out = await _base_compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
    seg_pre = float(out["decision"]) + _PRE_W * float(out["correct"])
    seg_post = float(out["correct"])
    out["seg_pre"] = seg_pre
    out["seg_post"] = seg_post
    out["score"] = seg_pre + seg_post
    return out
