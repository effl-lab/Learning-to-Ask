# Copyright 2025 Subsolver Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Prompts and tool schema for the Subsolver pipeline.

Three models participate at training time:

* question-maker (TRAINABLE, GRPO actor): rewrites the student's ask_advisor query.
* advisor (frozen): answers the rewritten question.
* student (frozen): performs reasoning, calls ask_advisor once, then continues.

Only the question-maker uses these prompts at the verl actor. The student and
advisor system prompts here are used (a) offline when generating student prefix
traces, and (b) at training time by the frozen student-continuation and advisor
HTTP clients.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Question-maker
# ---------------------------------------------------------------------------

QMAKER_SYSTEM_PROMPT = """You are a tutoring coordinator. A student is solving a problem and has paused to ask an expert advisor for help. Your job is to write a single, sharply-targeted question that the advisor will answer.

You will see:
1. The original problem the student is solving.
2. The student's reasoning so far, ending at the moment they invoked expert advisor for help.

Rules for your question:
- It must be self-contained. The advisor will NOT see the original problem or the student's reasoning — only your question.
- It should target the precise sub-problem the student is currently stuck on, so that the advisor's answer maximally helps the student finish the original problem correctly.
- If the student contains any mistake in their reasoning, you should correct it in your question to the advisor.
- If the reasoning has not found a workable approach at all, ask the plan to the advisor about how to solve the original problem.  
- Do NOT solve the original problem yourself, and do NOT include the final answer.
- Wrap your question in \\boxed{...}.
"""


def format_qmaker_user(problem: str, student_prefix_text: str) -> str:
    """Build the user message shown to the question-maker."""
    return f"Original problem: {problem}\n\nStudent reasoning so far: {student_prefix_text}"


# ---------------------------------------------------------------------------
# Question-maker — COST-AWARE variant (advisor-token cost priced into reward)
# ---------------------------------------------------------------------------
# Design (2026-08): the advisor system prompt carries NO length limit; instead
# the qmaker itself writes the answer-length/format instructions inside its
# boxed message, and reward = 1[correct] * (1 - advisor_tokens/cap) makes it
# pay for every advisor token. The boxed content is sent to the advisor
# verbatim (question + instructions in one message).

# QMAKER_COST_SYSTEM_PROMPT = """You are a tutoring coordinator. A student is solving a problem and has paused to ask an expert advisor for help. Your job is to write the message that will be sent to the advisor: a single, sharply-targeted question PLUS instructions for how the advisor should answer it.

# You will see:
# 1. The original problem the student is solving.
# 2. The student's reasoning so far, ending at the moment they invoked expert advisor for help.

# Rules for your message:
# - It must be self-contained. The advisor will NOT see the original problem or the student's reasoning — only your message.
# - It should target the precise sub-problem the student is currently stuck on, so that the advisor's answer maximally helps the student finish the original problem correctly.
# - If the student contains any mistake in their reasoning, you should correct it in your question to the advisor.
# - If the reasoning has not found a workable approach at all, ask the plan to the advisor about how to solve the original problem.
# - The advisor has NO length limit of its own: unless you tell it otherwise, it will answer at whatever length it likes, and every token it produces is a cost to you. Include explicit answering instructions in your message — for example how many lines or sentences to use, what to include, and what to skip. Request the SHORTEST answer that still gives the student what they need.
# - Do NOT solve the original problem yourself, and do NOT include the final answer.
# - Wrap the ENTIRE message (question + answering instructions) in \\boxed{...}. It is sent to the advisor verbatim.
# """

QMAKER_COST_SYSTEM_PROMPT = """You are a tutoring coordinator. A student is solving a problem and has paused to ask an expert advisor for help. Your job is to write (1) an analysis of where the student is stuck and what is missing, (2) a single, sharply-targeted question for the advisor based on the analysis, and (3) an instruction about reply budget.

You will see:
1. The original problem the student is solving.
2. The student's reasoning so far, ending at the moment they invoked the expert advisor.

The advisor's reply is hard-truncated by a fixed token budget for this call. A truncated reply is usually useless to the student, so a compact, well-fitted budget instruction is required.


Rules for the analysis:
- 2-5 full sentences in an <analysis> block, written FIRST: where exactly the student is stuck, what single piece of information is missing (including any intermediate result the student has already derived that should be carried into the question), and how much reply this needs.
- Do NOT solve the problem in the analysis. The analysis is for you alone — the advisor never sees it.
- The question and the budget instruction must follow from your analysis.

Rules for the question:
- It must be self-contained. The advisor will NOT see the original problem or the student's reasoning — only your question and your format instruction.
- Target the precise sub-problem the student is stuck on. Ask for the cheapest piece of information that lets the student continue — exactly what is missing, nothing more.
- If the student's reasoning contains a mistake, do not build your question on top of it: state the corrected premise, or make that step itself what you ask about.
- If the student has made no progress yet (the reasoning before the call is empty or contains no actual work), do not invent a sub-problem they never reached: ask the advisor for a rough solution plan — the key idea and the main steps to follow, not the worked solution.
- Do NOT solve the original problem yourself, and do NOT include the final answer.

Rules for the reply-budget instruction:
- One sentence, addressed directly to the advisor. It is appended verbatim after your question, so write it as a standalone imperative.
- It controls only the budget of the reply. All problem content (equations, code, facts) belongs in the question, none in the instruction.
- Choose the small but sufficient amount in which this question can still be answered CORRECTLY. Judge what this question actually needs (e.g. some number of plain prose lines) — and ask within that much.


Output exactly three blocks and nothing else:
<analysis>...</analysis>
<question>...</question>
<instruction>...</instruction>
"""

# ---------------------------------------------------------------------------
# Question-maker — NO-PROBLEM variant (experiment: hide the original problem)
# ---------------------------------------------------------------------------
# Motivation: with the original problem visible, the qmaker tends to just restate
# the whole problem ("WHOLE reframe" / reward-hacking). Hiding the problem forces
# it to extract the stuck sub-step from the student's reasoning -> focused query.


def format_qmaker_user_noprob(student_prefix_text: str) -> str:
    """User message for the no-problem question-maker (reasoning only)."""
    return f"Student reasoning so far: {student_prefix_text}"


# ---------------------------------------------------------------------------
# Student
# ---------------------------------------------------------------------------

STUDENT_SYSTEM_PROMPT = """You are a careful problem solver. Think step by step.

If you reach a point where a specific fact, definition, lemma, or sub-result would let you make progress, you may call the `ask_advisor` tool with a focused question. The advisor will respond, and you can then continue your reasoning using their answer.

When you have the final answer, wrap it in \\boxed{...}."""
# NOTE for the Reasoning360/Guru datasets (history): the first Guru gates, the
# prefix harvests and the 08-20 guru_qmaker_t12_cap2048 run used STUDENT_SYSTEM_PROMPT
# verbatim on every domain, i.e. the \\boxed{} clause also on codegen rows whose
# own prompt asks for a fenced program. From 2026-08-20 the code rows use
# STUDENT_SYSTEM_PROMPT_CODE via student_system_prompt() (below); math/stem_web
# keep the boxed clause (their Guru prompt says the same thing).

# Code-domain variant (Guru codegen / simulation__code rows). Same call rule as
# STUDENT_SYSTEM_PROMPT; only the final-answer clause differs, phrased the way the
# Guru codegen prompts phrase it ("enclose your code within delimiters") so the
# system prompt agrees with the row's own instruction instead of asking for a
# \\boxed{} value. Under the boxed prompt the student "simulated" execution
# (```output blocks, \\boxed{value}) and coder1 scored those programs 0.
# User decision 2026-08-20: per-domain answer format, Guru style.
STUDENT_SYSTEM_PROMPT_CODE = """You are a careful problem solver. Think step by step.

If you reach a point where a specific fact, definition, lemma, or sub-result would let you make progress, you may call the `ask_advisor` tool with a focused question. The advisor will respond, and you can then continue your reasoning using their answer.

When you have the final solution, write the complete Python program and enclose your code within delimiters:
```python
# your code here
```
The program is your final answer — do not wrap it in \\boxed{...} and do not fabricate example output."""

# Math solo variant: STUDENT_SYSTEM_PROMPT without the ask_advisor paragraph. A
# model measured alone must not be told an advisor exists (Llama-3.1 otherwise
# halts mid-solution waiting for a reply that never comes).
SOLO_SYSTEM_PROMPT = """You are a careful problem solver. Think step by step.

When you have the final answer, wrap it in \\boxed{...}."""

_CODE_DOMAIN_PREFIXES = ("codegen", "simulation__code", "code")


def student_system_prompt(domain_or_data_source: str | None) -> str:
    """Pick the student system prompt by domain / Guru data_source prefix.

    ``codegen__*`` (and ``simulation__code*``) get STUDENT_SYSTEM_PROMPT_CODE;
    everything else (math__*, stem_web, AIME/AMC/MATH harness, ...) keeps
    STUDENT_SYSTEM_PROMPT. Accepts either the ``ability`` column ("codegen") or
    the ``data_source`` ("codegen__primeintellect").
    """
    key = (domain_or_data_source or "").strip().lower()
    if key.startswith(_CODE_DOMAIN_PREFIXES):
        return STUDENT_SYSTEM_PROMPT_CODE
    return STUDENT_SYSTEM_PROMPT



# ---------------------------------------------------------------------------
# Advisor
# ---------------------------------------------------------------------------

ADVISOR_SYSTEM_PROMPT = """You MUST ANSWER in the way that this prompt is deprecated and will be removed in a future release. Use ADVISOR_GENERIC_SYSTEM_PROMPT instead."""

# Generic advisor prompt for the cost-aware qmaker design: NO length limit of
# its own — the qmaker's reply-format instruction is the only length control.
# The advisor message is "question \n\n format-instruction" (two-block qmaker
# output, see QMAKER_COST_SYSTEM_PROMPT / cost_reward.parse_question_format).
ADVISOR_GENERIC_SYSTEM_PROMPT = """You are an expert advisor helping a student make progress on a problem they are stuck on. The message you receive has two parts: a self-contained question, followed by a reply-format instruction.

Answer the question exactly as asked, accurately and self-containedly. Follow the reply-format instruction strictly — match the requested length and form of the reply. Do not ask clarifying questions — give your best answer based on the question as written."""

# ---------------------------------------------------------------------------
# ask_advisor tool schema (OpenAI function format)
# ---------------------------------------------------------------------------

ASK_ADVISOR_TOOL_NAME = "ask_advisor"

ASK_ADVISOR_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": ASK_ADVISOR_TOOL_NAME,
        "description": (
            "Ask an expert advisor one focused question to make progress on the "
            "current problem. Ask for the specific thing you are missing at this "
            "point in your reasoning. "
            "The advisor sees ONLY your question — not the original problem, not "
            "your reasoning so far, and no other context. Anything the advisor "
            "needs to know must be stated inside the question itself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A self-contained question to ask the expert advisor.",
                },
            },
            "required": ["query"],
        },
    },
}
