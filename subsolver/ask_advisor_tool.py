# Copyright 2025 Subsolver Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Phase 2 `ask_advisor` tool for verl multi-turn GRPO training of the STUDENT.

During the student's rollout, when it emits an ``ask_advisor(query=X)`` tool
call, this tool's :meth:`execute` runs the frozen Phase-1 pipeline *inside the
environment*:

    1. (optional) the trained question-maker rewrites X -> Q given the problem.
    2. the frozen advisor answers Q -> A.
    3. A is returned as the tool response, which verl appends to the student's
       chat history with ``loss_mask=0`` (not trained on).

Crucially, the student's own generated query X stays in the trajectory and is
trained on (``loss_mask=1``) — that is the policy's action (and encodes the
"when to call" timing decision). The rewrite Q is internal to this tool and is
NEVER inserted into the student's token sequence, so there is nothing extra to
mask.

Endpoints / models / token budgets come from the tool ``config`` block in the
tool-config YAML, with environment-variable fallbacks:
  SUBSOLVER_ADVISOR_ENDPOINT, SUBSOLVER_QMAKER_ENDPOINT (comma-separated for LB).
"""

from __future__ import annotations

import json
import os
import random
import re
from typing import Any

import httpx

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from .cost_reward import parse_question_format
from .pipeline import parse_boxed, strip_thinking
from .prompts import (
    ADVISOR_GENERIC_SYSTEM_PROMPT,
    ADVISOR_SYSTEM_PROMPT,
    ASK_ADVISOR_TOOL_SCHEMA,
    QMAKER_COST_SYSTEM_PROMPT,
    QMAKER_SYSTEM_PROMPT,
    format_qmaker_user,
)

# Reuse the Phase-1 advisor reject sentinel for consistency.
_REJECT_SENTINEL = "It is not valid question."

# Markers used to embed the question-maker's rewritten query Q alongside the
# advisor's answer in the tool response, so Q becomes visible in the trajectory
# (dump/viewer) and to the student. The tool response is loss_mask=0, so this
# does NOT train the student on Q (masked). Keep in sync with the viewer parser.
_Q_PREFIX = "[Refined question sent to the advisor]\n"
_A_PREFIX = "\n\n[Advisor's answer]\n"

# Appended to the (masked) tool response when the advisor answer hit the token
# cap. Integration reward greps for it to zero the rollout; keep in sync with
# subsolver/integration_reward.py.
_CAP_SENTINEL = "[advisor answer truncated at token cap]"

# Whole-problem screen: the student must ask for a SPECIFIC sub-part, not hand
# over the entire original problem (even reworded). Run inside the tool because
# only here are BOTH the original problem and the query visible (the advisor
# itself never sees the problem). Emits the reject sentinel or "VALID".
_WHOLE_PROBLEM_JUDGE_PROMPT = """You are screening a student's question to an expert advisor. You see the ORIGINAL problem the student is solving and the QUESTION they want to ask the advisor.

The advisor is meant to help with a SPECIFIC sub-part, not to solve the whole thing. If the student's question is essentially the ENTIRE original problem handed over — the same final goal/answer, even if reworded, rephrased, or lightly restated — it is INVALID. In that case reply with EXACTLY this line and nothing else:
It is not valid question.

Otherwise — the question asks only for a specific sub-fact, lemma, definition, intermediate step, or sub-computation — reply with EXACTLY:
VALID

Original problem:
{problem}

Student's question:
{query}"""


def _endpoints(raw: str) -> list[str]:
    return [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]


class AskAdvisorTool(BaseTool):
    """Calls (frozen) question-maker + advisor and returns the advisor's answer."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        c = config or {}

        adv_raw = c.get("advisor_endpoint") or os.environ.get("SUBSOLVER_ADVISOR_ENDPOINT", "")
        self.advisor_endpoints = _endpoints(adv_raw)
        self.advisor_model = c.get("advisor_model") or os.environ.get("SUBSOLVER_ADVISOR_MODEL", "advisor")
        # env overrides config so training/eval share one cap knob (integration
        # cap gate reads the same env in the reward function).
        self.advisor_max_tokens = int(
            os.environ.get("SUBSOLVER_ADVISOR_MAX_TOKENS", "") or c.get("advisor_max_tokens", 4096)
        )
        self.advisor_temperature = float(c.get("advisor_temperature", 0.0))
        # Hybrid-thinking advisors (Qwen3.5): disable thinking at template level.
        self.advisor_disable_thinking = os.environ.get(
            "SUBSOLVER_ADVISOR_DISABLE_THINKING", ""
        ) not in ("", "0", "false", "False")

        self.enable_qmaker = bool(c.get("enable_qmaker", True))
        # Cost-mode qmaker (budget-adaptive instruction): three-block output
        # (<analysis>/<question>/<instruction>), advisor gets question+instruction
        # under the GENERIC prompt. Enable via SUBSOLVER_QMAKER_COST=1.
        self.qmaker_cost = os.environ.get(
            "SUBSOLVER_QMAKER_COST", ""
        ) not in ("", "0", "false", "False")
        qm_raw = c.get("qmaker_endpoint") or os.environ.get("SUBSOLVER_QMAKER_ENDPOINT", "")
        self.qmaker_endpoints = _endpoints(qm_raw)
        self.qmaker_model = c.get("qmaker_model") or os.environ.get("SUBSOLVER_QMAKER_MODEL", "qmaker")
        self.qmaker_max_tokens = int(c.get("qmaker_max_tokens", 4096))
        self.qmaker_temperature = float(c.get("qmaker_temperature", 0.7))
        self.qmaker_top_p = float(c.get("qmaker_top_p", 0.95))

        # Whole-problem reject guard (v2). Judge runs on the advisor endpoints
        # (strongest model) by default.
        self.reject_whole_problem = bool(c.get("reject_whole_problem", False))
        self.judge_max_tokens = int(c.get("judge_max_tokens", 16))
        # Embed the qmaker's rewritten Q in the tool response (masked) so it is
        # visible in the trajectory. Only meaningful when enable_qmaker.
        self.show_query_in_response = bool(c.get("show_query_in_response", True))

        # Pass the student's live reasoning-so-far to the qmaker (Phase-1 parity).
        # The reasoning is in token space (agent_data.prompt_ids), so we decode it
        # with a local tokenizer. Without this the qmaker only sees problem+raw
        # query and tends to launder it into a whole-problem Q.
        self.pass_reasoning = bool(c.get("pass_reasoning", True))
        self.tokenizer = None
        if self.pass_reasoning:
            tok_path = c.get("tokenizer_path") or os.environ.get(
                "SUBSOLVER_STUDENT_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct"
            )
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
            except Exception:  # noqa: BLE001 — fall back to no-reasoning if load fails
                self.tokenizer = None

        self.timeout = float(c.get("timeout", 600.0))

        assert self.advisor_endpoints, (
            "AskAdvisorTool: no advisor endpoint (set config.advisor_endpoint or "
            "SUBSOLVER_ADVISOR_ENDPOINT)"
        )
        if self.enable_qmaker:
            assert self.qmaker_endpoints, (
                "AskAdvisorTool: enable_qmaker=True but no qmaker endpoint (set "
                "config.qmaker_endpoint or SUBSOLVER_QMAKER_ENDPOINT)"
            )

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema.model_validate(ASK_ADVISOR_TOOL_SCHEMA)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _first_user_problem(messages: list[dict[str, Any]] | None) -> str:
        for m in messages or []:
            if not isinstance(m, dict):
                continue
            if m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str):
                    return c
                if isinstance(c, list):  # multimodal content list
                    return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        return ""

    def _student_reasoning(self, agent_data) -> str:
        """Decode the student's current-turn reasoning from token space.

        At tool-call time the just-generated assistant turn (reasoning + the
        ask_advisor tool call) lives in agent_data.prompt_ids, not in
        agent_data.messages. We decode the response portion and strip the
        <tool_call>...</tool_call> markup, leaving the reasoning text to hand to
        the question-maker (mirrors Phase-1 prefix_text)."""
        if self.tokenizer is None:
            return ""
        prompt_ids = getattr(agent_data, "prompt_ids", None)
        response_mask = getattr(agent_data, "response_mask", None)
        if not prompt_ids or not response_mask:
            return ""
        fmt = os.environ.get("SUBSOLVER_TOOL_FORMAT", "hermes")
        try:
            resp_ids = prompt_ids[-len(response_mask):]
            # Gemma 4: keep special tokens so the thought channel and the call can be told apart
            text = self.tokenizer.decode(resp_ids, skip_special_tokens=(fmt != "gemma4"))
        except Exception:  # noqa: BLE001
            return ""
        if fmt == "gemma4":  # same reasoning text the qmaker was trained on (P3 harvest prefix)
            from subsolver.gemma4_toolcall import gemma4_student_reasoning
            return gemma4_student_reasoning(text)
        # drop the tool-call markup, keep the reasoning that precedes/surrounds it
        if os.environ.get("SUBSOLVER_TOOL_FORMAT", "hermes") == "llama3_json":
            # Llama: the turn ends with the JSON call (subsolver/llama_toolcall.py)
            from subsolver.llama_toolcall import parse_llama_tool_call
            return parse_llama_tool_call(text)[0].strip()
        if os.environ.get("SUBSOLVER_TOOL_FORMAT", "hermes") == "qwen3_xml":
            # Qwen3.5 thinking: the reasoning is the <think> part before the call
            from subsolver.qwen3_toolcall import student_reasoning
            return student_reasoning(text)
        text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
        return text.strip()

    async def _chat(self, endpoints, model, system, user, max_tokens, temperature, top_p=1.0,
                    disable_thinking=False) -> str:
        text, _ = await self._chat_full(
            endpoints, model, system, user, max_tokens, temperature, top_p, disable_thinking
        )
        return text

    async def _chat_full(self, endpoints, model, system, user, max_tokens, temperature,
                         top_p=1.0, disable_thinking=False) -> tuple[str, str]:
        """Like _chat but also returns finish_reason ('length' == truncated at cap)."""
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        # per-served-model extra sampling (e.g. top_k for a Qwen3.5 qmaker) from SUBSOLVER_EXTRA_SAMPLING_JSON
        payload.update(json.loads(os.environ.get("SUBSOLVER_EXTRA_SAMPLING_JSON") or "{}").get(model, {}))
        ep = random.choice(endpoints)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{ep}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
        choice = data["choices"][0]
        return choice["message"]["content"] or "", choice.get("finish_reason", "")

    # ------------------------------------------------------------------ execute
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        messages = getattr(agent_data, "messages", None)
        problem = self._first_user_problem(messages)

        # parameters can arrive double-encoded (a JSON string) instead of a dict.
        if isinstance(parameters, str):
            try:
                parameters = json.loads(parameters)
            except (json.JSONDecodeError, TypeError):
                parameters = {}
        if not isinstance(parameters, dict):
            parameters = {}
        raw_query = parameters.get("query")
        if not isinstance(raw_query, str) or not raw_query.strip():
            return ToolResponse(text="It is not valid question."), 0.0, {"err": "empty_query"}

        metrics: dict[str, Any] = {"qmaker_used": int(self.enable_qmaker)}
        try:
            # 1. question-maker rewrite FIRST (problem + raw query). Live reasoning
            #    lives in token space and is not available to the tool (see module
            #    doc). We rewrite before the guard so the guard can screen the
            #    ACTUAL query the advisor will receive (Q), not the student's raw X
            #    — otherwise the qmaker can launder a focused X into a whole-problem
            #    Q that bypasses the guard (a leak observed in early runs).
            instruction = ""
            if self.enable_qmaker:
                # Phase-1 parity: give the qmaker the student's reasoning-so-far
                # plus the raw query (not just the raw query).
                reasoning = self._student_reasoning(agent_data) if self.pass_reasoning else ""
                metrics["reasoning_chars"] = len(reasoning)
                if reasoning:
                    prefix_text = f"{reasoning}\n\n[ask_advisor query]\n{raw_query}"
                else:
                    prefix_text = f"[ask_advisor query]\n{raw_query}"
                qm_user = format_qmaker_user(problem, prefix_text)
                qm_prompt = QMAKER_COST_SYSTEM_PROMPT if self.qmaker_cost else QMAKER_SYSTEM_PROMPT
                qm_raw = await self._chat(
                    self.qmaker_endpoints, self.qmaker_model, qm_prompt, qm_user,
                    self.qmaker_max_tokens, self.qmaker_temperature, self.qmaker_top_p,
                )
                if self.qmaker_cost:
                    parsed = parse_question_format(qm_raw)
                    if parsed:
                        question, instruction = parsed
                        metrics["qmaker_no_boxed"] = 0
                    else:
                        question, instruction = raw_query, ""
                        metrics["qmaker_no_boxed"] = 1
                else:
                    question = parse_boxed(qm_raw) or raw_query
                    metrics["qmaker_no_boxed"] = int(parse_boxed(qm_raw) is None)
            else:
                question = raw_query

            # 2. whole-problem reject guard: judge Q (what the advisor actually
            #    receives). If Q is essentially the entire problem, refuse — the
            #    student gets no advisor help and must elicit a focused sub-problem.
            if self.reject_whole_problem and problem:
                verdict = await self._chat(
                    self.advisor_endpoints, self.advisor_model,
                    "You are a strict screener. Follow the instructions exactly.",
                    _WHOLE_PROBLEM_JUDGE_PROMPT.format(problem=problem[:2000], query=question[:1000]),
                    self.judge_max_tokens, 0.0,
                )
                verdict = strip_thinking(verdict)
                if _REJECT_SENTINEL in verdict or "not valid question" in verdict.lower():
                    metrics["rejected_whole"] = 1
                    return ToolResponse(text=_REJECT_SENTINEL), 0.0, metrics
                metrics["rejected_whole"] = 0

            # 3. advisor answers the (rewritten, guard-passed) question.
            # Cost mode: generic prompt, message = question + reply-budget
            # instruction (the student's tool context still shows question only).
            if self.qmaker_cost:
                adv_prompt = ADVISOR_GENERIC_SYSTEM_PROMPT
                adv_msg = f"{question}\n\n{instruction}" if instruction else question
            else:
                adv_prompt, adv_msg = ADVISOR_SYSTEM_PROMPT, question
            ans, finish = await self._chat_full(
                self.advisor_endpoints, self.advisor_model, adv_prompt, adv_msg,
                self.advisor_max_tokens, self.advisor_temperature,
                disable_thinking=self.advisor_disable_thinking,
            )
            ans = strip_thinking(ans)
            if not ans:
                ans = "(advisor returned no content)"
            # Cap marker: reward functions detect this sentinel to zero the
            # rollout (integration cap gate). Masked text, so not trained on.
            metrics["advisor_truncated"] = 1 if finish == "length" else 0
            if finish == "length":
                ans = f"{ans}\n{_CAP_SENTINEL}"
            if self.enable_qmaker and self.show_query_in_response:
                # Embed Q (masked tool response) so it's visible in the trajectory.
                text = f"{_Q_PREFIX}{question}{_A_PREFIX}{ans}"
            else:
                text = ans
            return ToolResponse(text=text), 0.0, metrics
        except Exception as e:  # noqa: BLE001 — never crash the rollout
            metrics["err"] = f"{type(e).__name__}"
            return ToolResponse(text=f"(advisor unavailable: {type(e).__name__})"), 0.0, metrics
