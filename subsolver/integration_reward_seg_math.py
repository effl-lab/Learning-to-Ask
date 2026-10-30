"""Segmented integration reward for the MATH domain.

Math analogue of integration_reward_seg (which wraps the CODE scorer). Wraps the
boxed-answer math scorer (integration_reward.compute_score, math_reward) and emits
the two segment components consumed by `grpo_segmented` (core_algos):

    seg_pre  = 1[called == need] + W * 1[correct]     -> tokens BEFORE the tool response
    seg_post = 1[correct]                             -> continuation tokens AFTER it

W = SUBSOLVER_SEG_PRE_CORRECT_W (default 0.3). Same design as the code version: the
whether-signal stays on the decision tokens while the continuation gets a pure
correctness objective. Fixes the non-segmented math integration's OOD call collapse
(exp: 0 valid calls on AIME2026) by keeping the gate reward on the pre-call segment.

Load via:
    custom_reward_function.path=pkg://subsolver.integration_reward_seg_math
    custom_reward_function.name=compute_score
    algorithm.adv_estimator=grpo_segmented
"""

from __future__ import annotations

import os
from typing import Any

from .integration_reward import compute_score as _base_compute_score

_PRE_W = float(os.environ.get("SUBSOLVER_SEG_PRE_CORRECT_W", "0.3") or 0.3)


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    # integration_reward.compute_score is synchronous (boxed match, no I/O).
    out = _base_compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
    seg_pre = float(out["decision"]) + _PRE_W * float(out["correct"])
    seg_post = float(out["correct"])
    out["seg_pre"] = seg_pre
    out["seg_post"] = seg_post
    out["score"] = seg_pre + seg_post
    return out
