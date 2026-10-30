# Copyright 2025 Subsolver Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""GRPO reward function for the Subsolver question-maker.

End-to-end per rollout:

1. The verl actor (question-maker) produced ``solution_str``; parse the LAST
   ``\\boxed{...}`` to get the rewritten advisor question ``Q``.
2. Send ``Q`` to the frozen advisor → answer ``A``.
3. Take the pre-generated student prefix (ending in an ``ask_advisor`` tool
   call), swap the call's ``query`` argument with ``Q``, append a tool-response
   message carrying ``A``.
4. Send that assembled chat history to the frozen student continuation server
   → final completion text.
5. Parse the LAST ``\\boxed{...}`` of the completion as the student's final
   answer; compare to ``ground_truth``.

Returns 1.0 on correct, 0.0 otherwise. Format violations (no boxed in either
step) score 0.0. Transport errors propagate so the reward manager surfaces
them rather than silently masking real bugs.

Load via:
    custom_reward_function.path=pkg://subsolver.reward
    custom_reward_function.name=compute_score
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any

import httpx

from .frozen_clients import AdvisorClient, StudentContinuationClient
from .pipeline import assemble_student_messages, parse_boxed, strip_thinking

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Whole-problem reject penalty (optional, env-gated by SUBSOLVER_WHOLE_REJECT)
# ---------------------------------------------------------------------------
# When enabled, the qmaker's rewritten question Q is screened against the
# ORIGINAL problem: if Q is essentially the entire problem handed over (WHOLE
# reframe), the rollout gets score 0 regardless of downstream correctness. This
# penalizes the reward-hacking "just restate the whole problem" strategy at the
# reward level (vs. only hiding the problem from the qmaker input).

_WHOLE_JUDGE_PROMPT = """You are screening a student's question to an expert advisor. You see the ORIGINAL problem the student is solving and the QUESTION they want to ask the advisor.

The advisor is meant to help with a SPECIFIC sub-part, not to solve the whole thing. If the student's question is essentially the ENTIRE original problem handed over — the same final goal/answer, even if reworded, rephrased, or lightly restated — it is INVALID. In that case reply with EXACTLY this line and nothing else:
It is not valid question.

Otherwise — the question asks only for a specific sub-fact, lemma, definition, intermediate step, or sub-computation — reply with EXACTLY:
VALID

Original problem:
{problem}

Student's question:
{query}"""


def _whole_reject_enabled() -> bool:
    return os.environ.get("SUBSOLVER_WHOLE_REJECT", "") not in ("", "0", "false", "False")


# ---------------------------------------------------------------------------
# Incomplete-query reject gate (optional, env-gated by SUBSOLVER_INCOMPLETE_REJECT)
# ---------------------------------------------------------------------------
# The advisor is blind (sees ONLY Q), so a Q with dangling references ("the
# equation above"), missing givens, or a bare value/expression with nothing
# actually asked cannot be answered usefully. Same judge mechanics as the
# whole gate: Instruct advisor endpoints, temp 0, sentinel output, fail-open.

_INCOMPLETE_JUDGE_PROMPT = """You are screening a question that a student wants to send to an expert advisor. The advisor will see ONLY this question — no other context: not the original problem, not the student's work.

QUESTION:
{query}

Decide whether the question is SELF-CONTAINED: answerable exactly as written, on its own. It is INCOMPLETE if it refers to context the advisor cannot see ("the equation above", "this function", "the value we found", "the problem", "the given conditions"), omits a definition, value, or constraint needed to answer it, is a bare number or expression with nothing actually asked, or is otherwise ambiguous without outside context.

If INCOMPLETE, reply with EXACTLY this line and nothing else:
It is not valid question.

Otherwise reply with EXACTLY:
VALID"""


def _incomplete_reject_enabled() -> bool:
    return os.environ.get("SUBSOLVER_INCOMPLETE_REJECT", "") not in ("", "0", "false", "False")


# ---------------------------------------------------------------------------
# Advisor-cap reject gate (optional, env-gated by SUBSOLVER_CAP_REJECT)
# ---------------------------------------------------------------------------
# The advisor is served/called with max_tokens = SUBSOLVER_ADVISOR_MAX_TOKENS.
# When the answer hits that cap (completion_tokens >= cap, i.e. truncated), the
# rollout scores 0: the qmaker must ask questions the advisor can answer within
# budget, which directly prices in advisor-token cost.


def _cap_reject_enabled() -> bool:
    return os.environ.get("SUBSOLVER_CAP_REJECT", "") not in ("", "0", "false", "False")


def _advisor_cap() -> int:
    try:
        return int(os.environ.get("SUBSOLVER_ADVISOR_MAX_TOKENS", "8192") or 8192)
    except ValueError:
        return 8192


async def _judge_incomplete(query: str) -> bool:
    """True iff Q is judged NOT self-contained. Fail-open on any error."""
    raw = os.environ.get("SUBSOLVER_ADVISOR_ENDPOINT", "")
    eps = [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
    if not eps or not query:
        return False
    model = os.environ.get("SUBSOLVER_ADVISOR_MODEL", "advisor")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict screener. Follow the instructions exactly."},
            {"role": "user", "content": _INCOMPLETE_JUDGE_PROMPT.format(query=str(query)[:1500])},
        ],
        "temperature": 0.0,
        "max_tokens": 16,
    }
    # Hybrid-thinking judge would burn the 16-token budget inside thinking and
    # silently fail-open — force non-thinking when asked.
    if os.environ.get("SUBSOLVER_ADVISOR_DISABLE_THINKING", "") not in ("", "0", "false", "False"):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{random.choice(eps)}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            verdict = strip_thinking(resp.json()["choices"][0]["message"]["content"] or "")
    except Exception:  # noqa: BLE001
        return False
    return ("It is not valid question." in verdict) or ("not valid question" in verdict.lower())


# ---------------------------------------------------------------------------
# Advisor-length penalty (optional, env-gated by SUBSOLVER_LENGTH_PENALTY)
# ---------------------------------------------------------------------------
# reward = 1[correct] - lambda * L_norm, where L is the advisor's
# completion_tokens (thinking + answer) and L_norm = L / (L + k) is a rational
# saturation: structurally bounded in [0, 1) for ANY L >= 0 (no clipping), 0 at
# L=0, 0.5 at L=k, ->1 as L grows. The intent: a focused query yields a short
# advisor answer (penalty ~0); wholesale delegation makes the strong advisor
# solve everything (long answer -> penalty ~1). Keep lambda < 1 so correctness
# always dominates length (worst correct = 1-lambda > 0 >= best wrong).


def _length_penalty_lambda() -> float:
    """λ weight for the advisor-length penalty. 0 (default) disables it."""
    try:
        return float(os.environ.get("SUBSOLVER_LENGTH_PENALTY", "0") or 0.0)
    except ValueError:
        return 0.0


def _length_penalty_k() -> float:
    """Half-saturation point k: advisor completion_tokens at which L_norm = 0.5.

    Default 1500 = geomean of the advisor-token medians measured on an early
    qmaker run (focused 1050, whole 2238); this k maximizes the focused-vs-whole
    separation of L/(L+k)."""
    try:
        return float(os.environ.get("SUBSOLVER_LENGTH_PENALTY_K", "1500") or 1500.0)
    except ValueError:
        return 1500.0


def _length_norm(tokens: int) -> float:
    """Map advisor token count to [0, 1) via rational saturation L / (L + k)."""
    L = max(0, int(tokens))
    denom = L + _length_penalty_k()
    return (L / denom) if denom > 0 else 0.0


async def _judge_whole(problem: str, query: str) -> bool:
    """True iff Q is judged a whole-problem reframe of the original problem.

    Uses the advisor servers (strongest model) as judge. On any error, returns
    False (don't penalize on transport failure)."""
    raw = os.environ.get("SUBSOLVER_ADVISOR_ENDPOINT", "")
    eps = [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
    if not eps or not problem:
        return False
    model = os.environ.get("SUBSOLVER_ADVISOR_MODEL", "advisor")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a strict screener. Follow the instructions exactly."},
            {"role": "user", "content": _WHOLE_JUDGE_PROMPT.format(problem=str(problem)[:2000], query=str(query)[:1000])},
        ],
        "temperature": 0.0,
        "max_tokens": 16,
    }
    # Hybrid-thinking judge would burn the 16-token budget inside thinking and
    # silently fail-open — force non-thinking when asked.
    if os.environ.get("SUBSOLVER_ADVISOR_DISABLE_THINKING", "") not in ("", "0", "false", "False"):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{random.choice(eps)}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            verdict = strip_thinking(resp.json()["choices"][0]["message"]["content"] or "")
    except Exception:  # noqa: BLE001
        return False
    return ("It is not valid question." in verdict) or ("not valid question" in verdict.lower())


# ---------------------------------------------------------------------------
# Client singletons
# ---------------------------------------------------------------------------
# Stateless HTTP clients reused across calls. They open a fresh
# httpx.AsyncClient per request so they are safe to share across the asyncio
# tasks the naive reward manager spawns concurrently.

_advisor_client: AdvisorClient | None = None
_student_client: StudentContinuationClient | None = None


def _get_advisor() -> AdvisorClient:
    global _advisor_client
    if _advisor_client is None:
        _advisor_client = AdvisorClient()
    return _advisor_client


def _get_student() -> StudentContinuationClient:
    global _student_client
    if _student_client is None:
        _student_client = StudentContinuationClient()
    return _student_client


# ---------------------------------------------------------------------------
# Answer normalisation
# ---------------------------------------------------------------------------


def _normalize(s: str) -> str:
    """Lightweight normalisation for verifiable equality.

    Sufficient for numeric / short-string ground-truths (GSM8K-style). For
    tasks where deeper equivalence is needed (MATH, symbolic answers), wire a
    different verifier into :func:`_is_correct`.
    """
    return s.strip().replace(",", "").replace(" ", "")


def _is_correct(predicted: str, ground_truth: str) -> bool:
    return _normalize(predicted) == _normalize(ground_truth)


# Sentinel the advisor is prompted to emit when it judges the qmaker's question
# invalid (full original problem, request for the final answer, etc.). Keep in
# sync with ADVISOR_SYSTEM_PROMPT in prompts.py.
_ADVISOR_REJECT_SENTINEL = "It is not valid question."


# ---------------------------------------------------------------------------
# Public entry point used by verl
# ---------------------------------------------------------------------------


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Reward for a single question-maker rollout.

    Returns a dict with ``score`` (the scalar reward used for training) plus
    per-trajectory metadata (``qmaker_query``, ``advisor_answer``,
    ``student_continuation``, ``predicted_answer``, ``failure_mode``). verl's
    reward manager aggregates these dict keys into ``reward_extra_info`` and
    the trainer's ``_dump_generations`` then writes them to the val-dump JSONL,
    giving a full per-prompt trace of the subsolver pipeline.

    ``extra_info`` must contain ``student_prefix_messages`` (a list of OpenAI
    chat-format dicts ending with an assistant message whose ``tool_calls``
    list contains exactly one ``ask_advisor`` call).
    """
    extra_info = extra_info or {}

    # verl's validation metrics pipeline computes mean@N over every key in
    # reward_extra_info whose first row is not a string (see
    # verl/trainer/ppo/metric_utils.py:640). Use "" as the missing value so
    # those trace columns are detected as strings and skipped, while still
    # being valid JSON for the val-dump JSONL.
    result: dict[str, Any] = {
        "score": 0.0,
        "qmaker_query": "",
        "advisor_answer": "",
        "student_continuation": "",
        "predicted_answer": "",
        "failure_mode": "",
        # advisor_tokens / length_penalty are numeric so verl logs their mean@N
        # in reward_extra_info — handy for calibrating k even when λ=0 (off).
        "advisor_tokens": 0,
        "length_penalty": 0.0,
        # decomposed reward components (numeric 0/1 -> per-step means in logs):
        "no_boxed": 0,        # qmaker emitted no \boxed{Q}
        "incomplete": 0,      # incomplete-query judge fired
        "cap_exceeded": 0,    # advisor answer hit the max_tokens cap
        "correct": 0,         # student final answer == ground truth
    }

    rewritten_question = parse_boxed(solution_str)
    if rewritten_question is None:
        result["no_boxed"] = 1
        result["failure_mode"] = "qmaker_no_boxed"
        return result
    result["qmaker_query"] = rewritten_question

    # Whole-problem reject penalty (env-gated): if Q is essentially the entire
    # original problem, score 0 and skip the rest. Penalizes "restate the whole
    # problem" reward-hacking at the reward level.
    if _whole_reject_enabled():
        problem = extra_info.get("problem", "")
        if problem and await _judge_whole(problem, rewritten_question):
            result["failure_mode"] = "whole_problem"
            return result

    # Incomplete-query gate: a Q the blind advisor cannot answer as written
    # (dangling references / missing givens / bare expression) scores 0.
    if _incomplete_reject_enabled():
        if await _judge_incomplete(rewritten_question):
            result["incomplete"] = 1
            result["failure_mode"] = "incomplete_query"
            return result

    prefix_messages = extra_info.get("student_prefix_messages")
    if not prefix_messages:
        raise ValueError(
            "extra_info['student_prefix_messages'] is required for the subsolver "
            "reward function but was missing or empty."
        )

    # Catch transport-level failures (timeout, connection reset) and treat
    # them as 0-reward for that trajectory instead of crashing the whole
    # training step. With training-time concurrency = batch * group, a
    # single tail-latency advisor call shouldn't be able to take down a
    # whole batch.
    try:
        advisor_answer, advisor_tokens = await _get_advisor().answer_with_usage(
            rewritten_question
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("advisor call failed (%s); treating as 0 reward", type(e).__name__)
        result["failure_mode"] = f"advisor_error:{type(e).__name__}"
        return result
    result["advisor_answer"] = advisor_answer
    result["advisor_tokens"] = advisor_tokens
    result["length_penalty"] = _length_norm(advisor_tokens)
    lam = _length_penalty_lambda()
    lnorm = result["length_penalty"]

    # Advisor-cap gate: answer truncated at the token cap -> 0 reward. The
    # qmaker must ask questions answerable within the advisor budget.
    result["cap_exceeded"] = 1 if advisor_tokens >= _advisor_cap() else 0
    if _cap_reject_enabled() and result["cap_exceeded"]:
        result["failure_mode"] = "advisor_cap"
        return result

    # Advisor was prompted to emit exactly "It is not valid question." when it
    # detects the qmaker handed it the original problem / asked for the final
    # answer outright. Treat any rollout where that sentinel appears in the
    # advisor's reply as a 0-reward trajectory: the qmaker effectively short-
    # circuited the pipeline and the downstream student answer (if any) is not
    # earned. We don't proceed to the student call in that case.
    if _ADVISOR_REJECT_SENTINEL in advisor_answer:
        result["failure_mode"] = "advisor_rejected"
        return result

    student_messages = assemble_student_messages(
        prefix_messages=list(prefix_messages),
        rewritten_question=rewritten_question,
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

    predicted_answer = parse_boxed(student_completion)
    if predicted_answer is None:
        result["failure_mode"] = "student_no_boxed"
        # wrong answer that still burned advisor tokens: base 0 minus penalty
        # (no-op when λ=0). Keeps the "apply length penalty regardless of
        # correctness" contract.
        result["score"] = -lam * lnorm
        return result
    result["predicted_answer"] = predicted_answer

    base = 1.0 if _is_correct(predicted_answer, ground_truth) else 0.0
    result["correct"] = int(base)
    # reward = 1[correct] - λ·L_norm. λ=0 -> unchanged (pure accuracy). λ<1
    # guarantees every correct (>= 1-λ > 0) outranks every wrong (<= 0).
    result["score"] = base - lam * lnorm
    return result
