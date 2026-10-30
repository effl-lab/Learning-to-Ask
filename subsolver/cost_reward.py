# Copyright 2025 Subsolver Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Binary is-cut GRPO reward for the Subsolver question-maker.

    reward = 1[correct] * 1[advisor_tokens < cap]

Design (2026-08): the advisor system prompt is GENERIC — no length limit of its
own (``ADVISOR_GENERIC_SYSTEM_PROMPT``). The qmaker emits three blocks,
``<analysis>...</analysis><question>...</question><instruction>...</instruction>``
(see QMAKER_COST_SYSTEM_PROMPT; the instruction is a one-sentence reply BUDGET,
e.g. "Answer within 10 lines of plain prose."). The advisor receives
"question \n\n instruction", while the student's tool call carries the question
only. The only cost pressure is the cap boundary (``cap`` =
SUBSOLVER_ADVISOR_MAX_TOKENS, also the serving max_tokens): a truncated answer
scores 0, any sub-cap answer is free.

SUBSOLVER_COST_SCORE_MODE selects a continuous penalty instead, weighted by
SUBSOLVER_COST_LAMBDA (default 0.7), with r = advisor_tokens / cap:

    scaled  reward = 1[correct] * (1 - lambda * r)      in [0, 1]
    linear  reward = 1[correct] - lambda * r            in [-lambda, 1]

Both make budget something the query has to earn, which is-cut does not: under
is-cut every sub-cap answer is equally free, so nothing discourages asking for
the whole cap, and the runs drifted to whole-problem copies as the cap grew.

The rejection recorded earlier was of ``correct * (1 - r)`` — the scaled form at
lambda = 1 — which collapsed at cap 512 and oscillated at 2048. lambda < 1 changes
the ordering that caused it: at lambda = 1 a correct answer that spends the full
budget scores 0, exactly what a wrong answer scores, so cost can erase
correctness entirely; at 0.7 it floors at 0.3 and correctness always outranks
thrift. The linear form goes the other way and charges cost even when the answer
is wrong, which pressures cheapness harder but re-opens that ordering.

Differences from ``subsolver.reward``:
* score formula above (correctness gated, cap-boundary only);
* advisor called with the generic system prompt (no 10-line rule);
* no whole-problem gate, no advisor-reject sentinel;
* incomplete-query judge stays ON (launcher default SUBSOLVER_INCOMPLETE_REJECT=1):
  without it, the cost bonus would actively reward lazy under-specified queries
  whose cheap advisor answers the student then solves around on its own.

Load via:
    custom_reward_function.path=pkg://subsolver.cost_reward
    custom_reward_function.name=compute_score
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from .frozen_clients import AdvisorClient, StudentContinuationClient
from .pipeline import assemble_student_messages, parse_answer, parse_boxed
from .prompts import ADVISOR_GENERIC_SYSTEM_PROMPT
from .reward import (
    _advisor_cap,
    _incomplete_reject_enabled,
    _is_correct,
    _judge_incomplete,
    _judge_whole,
    _whole_reject_enabled,
)

logger = logging.getLogger(__name__)

# The cost-aware qmaker emits three blocks (see QMAKER_COST_SYSTEM_PROMPT):
#   <analysis>...</analysis><question>...</question><instruction>...</instruction>
# The analysis is a forced deliberation step (qmaker-only; never forwarded).
# Take the LAST match of each (thinking models may draft blocks inside <think>).
_ANALYSIS_RE = re.compile(r"<analysis>(.*?)</analysis>", re.DOTALL)
_QUESTION_RE = re.compile(r"<question>(.*?)</question>", re.DOTALL)
# The instruction is the final block; models frequently emit EOS right after its
# content without the closing tag — accept an unclosed trailing block.
_INSTRUCTION_RE = re.compile(r"<instruction>(.*?)(?:</instruction>|$)", re.DOTALL)


def parse_question_format(text: str) -> tuple[str, str] | None:
    """Return (question, reply-budget instruction) or None on format violation."""
    qs = _QUESTION_RE.findall(text or "")
    fs = _INSTRUCTION_RE.findall(text or "")
    if not qs or not fs:
        return None
    q, f = qs[-1].strip(), fs[-1].strip()
    if not q or not f:
        return None
    return q, f


def parse_analysis(text: str) -> str | None:
    """Return the (last) non-empty <analysis> block, or None if absent."""
    ans = _ANALYSIS_RE.findall(text or "")
    if not ans or not ans[-1].strip():
        return None
    return ans[-1].strip()

_advisor_client: AdvisorClient | None = None
_student_client: StudentContinuationClient | None = None


def _get_advisor() -> AdvisorClient:
    global _advisor_client
    if _advisor_client is None:
        _advisor_client = AdvisorClient(system_prompt=ADVISOR_GENERIC_SYSTEM_PROMPT)
    return _advisor_client


def _get_student() -> StudentContinuationClient:
    global _student_client
    if _student_client is None:
        _student_client = StudentContinuationClient()
    return _student_client


def _score_mode() -> str:
    """"cut" (default), "scaled" (correct * (1 - k*r)) or "linear" (correct - k*r)."""
    return (os.environ.get("SUBSOLVER_COST_SCORE_MODE", "cut") or "cut").strip().lower()


def _cost_lambda() -> float:
    """Weight on cost_ratio in the continuous modes; 0.7 unless overridden."""
    try:
        return float(os.environ.get("SUBSOLVER_COST_LAMBDA", "0.7") or 0.7)
    except ValueError:
        return 0.7


async def _compute_score_inner(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Cost-aware reward for one question-maker rollout.

    Pipeline identical to ``subsolver.reward.compute_score`` (boxed message ->
    advisor -> student continuation -> boxed answer check); only the scoring
    and the advisor prompt differ. ``cost_ratio`` = advisor completion tokens /
    cap, clamped to [0, 1]; the trace and decomposed 0/1 fields flow into
    reward_extra_info as usual.
    """
    extra_info = extra_info or {}
    # Resolved up front so the advisor call, cost_ratio and cap_exceeded all use
    # the same number, and so it is logged even on early-return rows.
    cap = _advisor_cap()

    result: dict[str, Any] = {
        "score": 0.0,
        "analysis": "",
        "qmaker_query": "",
        "format_instruction": "",
        "advisor_answer": "",
        "student_continuation": "",
        "predicted_answer": "",
        "failure_mode": "",
        "advisor_tokens": 0,
        "advisor_cap": cap,
        "cost_ratio": 0.0,
        # decomposed reward components (numeric -> per-step means in logs):
        "no_boxed": 0,        # qmaker output missing <question>/<format> blocks
        "incomplete": 0,      # incomplete-query judge fired (launcher default: on)
        "whole_problem": 0,   # whole-problem judge fired (SUBSOLVER_WHOLE_REJECT)
        "cap_exceeded": 0,    # advisor answer truncated at the max_tokens cap
        "correct": 0,         # student final answer == ground truth
    }

    if os.environ.get("SUBSOLVER_QMAKER_THINKING", "0") == "1":
        # Thinking qmaker (Qwen3.5 line): only the post-</think> text is the output; blocks drafted
        # inside the thinking do not count, and an unclosed thinking is a format failure.
        from .qwen3_toolcall import split_think
        solution_str = split_think(solution_str or "")[1] or ""
    parsed = parse_question_format(solution_str)
    if parsed is None:
        result["no_boxed"] = 1
        result["failure_mode"] = "qmaker_bad_format"
        return result
    # Forced deliberation: a missing/empty <analysis> block is a format
    # violation too — the free-form "think first" instruction was silently
    # ignored (0% compliance), so the block is required at reward level.
    analysis = parse_analysis(solution_str)
    if analysis is None:
        result["no_boxed"] = 1
        result["failure_mode"] = "qmaker_no_analysis"
        return result
    question, format_instruction = parsed
    result["analysis"] = analysis
    result["qmaker_query"] = question
    result["format_instruction"] = format_instruction
    # The advisor sees the question followed by the reply-format instruction.
    advisor_message = f"{question}\n\n{format_instruction}"

    if _incomplete_reject_enabled():
        # Judge the QUESTION only — the format instruction carries no content.
        if await _judge_incomplete(question):
            result["incomplete"] = 1
            result["failure_mode"] = "incomplete_query"
            return result

    # Whole-problem gate (08-20, user spec: correct x not-whole x not-cut on the
    # instruction-carrying pipeline). Judges the QUESTION only, same as the
    # incomplete gate — the format instruction is advisor-facing plumbing.
    if _whole_reject_enabled():
        if await _judge_whole(str(extra_info.get("problem", "")), question):
            result["whole_problem"] = 1
            result["failure_mode"] = "whole_problem"
            return result

    prefix_messages = extra_info.get("student_prefix_messages")
    if not prefix_messages:
        raise ValueError(
            "extra_info['student_prefix_messages'] is required for the subsolver "
            "reward function but was missing or empty."
        )

    try:
        advisor_answer, advisor_tokens = await _get_advisor().answer_with_usage(
            advisor_message, max_tokens=cap
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("advisor call failed (%s); treating as 0 reward", type(e).__name__)
        result["failure_mode"] = f"advisor_error:{type(e).__name__}"
        return result
    result["advisor_answer"] = advisor_answer
    result["advisor_tokens"] = advisor_tokens
    result["cost_ratio"] = min(1.0, max(0, advisor_tokens) / cap) if cap > 0 else 1.0
    result["cap_exceeded"] = 1 if advisor_tokens >= cap else 0

    # The student's tool call carries only the question — the format
    # instruction is advisor-facing plumbing, not part of what "was asked".
    student_messages = assemble_student_messages(
        prefix_messages=list(prefix_messages),
        rewritten_question=question,
        advisor_answer=advisor_answer,
    )

    try:
        student_completion = await _get_student().continue_after_tool_response(
            student_messages
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("student call failed (%s); treating as 0 reward", type(e).__name__)
        result["failure_mode"] = f"student_error:{type(e).__name__}"
        return result
    result["student_continuation"] = student_completion

    predicted_answer = parse_answer(student_completion)  # boxed, or 'Final Answer:' (SUBSOLVER_ANSWER_FORMAT)
    if predicted_answer is None:
        result["failure_mode"] = "student_no_boxed"
        return result
    result["predicted_answer"] = predicted_answer

    result["correct"] = int(_is_correct(predicted_answer, ground_truth))
    # Binary is-cut: 1[correct] * 1[answer finished within the cap]. No
    # per-token pressure — only the cap boundary matters; cost_ratio is still
    # logged for monitoring. Linear mode overrides this in the wrapper below.
    result["score"] = float(result["correct"] * (1 - result["cap_exceeded"]))
    return result


async def compute_score(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """`_compute_score_inner`, with the continuous cost penalty applied if enabled.

    See the module docstring for the two forms. Applied here rather than at the one
    success path because every early return — bad format, incomplete query, student
    error, no boxed answer — leaves score at 0.0. That is harmless for "scaled"
    (correct = 0 there, so the formula gives 0 anyway) but a hole for "linear":
    burning the whole budget and answering wrong would score -lambda while producing
    no answer at all scores 0, which pays the qmaker to break the student.
    Recomputing from (correct, cost_ratio) closes it for both, and is idempotent
    since neither field changes after the inner call returns.
    """
    result = await _compute_score_inner(*args, **kwargs)
    mode, k = _score_mode(), _cost_lambda()
    r = float(result.get("cost_ratio", 0.0))   # already clipped to [0, 1]
    correct = float(result.get("correct", 0))
    if mode == "scaled":
        result["score"] = correct * (1.0 - k * r)
    elif mode == "linear":
        result["score"] = correct - k * r
    return result
