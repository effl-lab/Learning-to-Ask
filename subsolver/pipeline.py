# Copyright 2025 Subsolver Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Message manipulation and \\boxed{...} parsing utilities for Subsolver.

All parseable artifacts in the pipeline are wrapped in \\boxed{...}:

* The question-maker's output:        \\boxed{<rewritten question>}
* The student's final answer:         \\boxed{<answer>}

The student's prefix already contains a native tool_call to ``ask_advisor``
with the student's original raw query. At reward-computation time we swap that
``query`` argument in-place with the question-maker's rewritten question, then
append a tool-response message holding the advisor's answer, and let the frozen
student model continue from there.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from .prompts import ASK_ADVISOR_TOOL_NAME


# ---------------------------------------------------------------------------
# Thinking-tag stripping (Qwen3 hybrid / Thinking models)
# ---------------------------------------------------------------------------

_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think\b[^>]*>.*\Z", re.IGNORECASE | re.DOTALL)


# ---------------------------------------------------------------------------
# Special-token-as-text sanitization
# ---------------------------------------------------------------------------

# A thinking/instruct model sampling at temp > 0 sometimes *textually* emits the
# string form of its own chat delimiters (e.g. "<|im_start|>", "</s>") into its
# reasoning — most often when it runs into a dead end and pattern-completes a
# fake multi-turn transcript. With ``skip_special_tokens=True`` these survive the
# OpenAI response (they are NOT the real control-token ids). The problem surfaces
# on RE-FEED: when the stored message is re-templated and re-tokenized for the
# frozen student / qmaker, the tokenizer's special-token matcher promotes that
# literal text BACK into the real control token, injecting a spurious turn
# boundary in the middle of the content. We strip those strings at trace-build
# time so a re-fed prefix can never carry a fake turn. Matches the Qwen
# ``<|...|>`` family plus BOS/EOS ``<s>`` / ``</s>``. The inner length is bounded
# so a stray ``<|`` in real text can't swallow a long span.
_SPECIAL_TOKEN_TEXT_RE = re.compile(r"<\|[^|>]{0,40}\|>|</?s>")


def sanitize_special_tokens(text: str) -> str:
    """Remove chat control-token *strings* (``<|im_start|>``, ``</s>``, …) that a
    model emitted as literal text, so they cannot be re-promoted to real special
    tokens when the text is re-tokenized. Normal content is left intact."""
    if not text:
        return text
    return _SPECIAL_TOKEN_TEXT_RE.sub("", text)


def strip_thinking(text: str) -> str:
    """Return only the answer part of a generation, dropping the thinking trace.

    Handles three emission styles:

    (a) Closed ``<think>...</think>`` WITH an opening tag (e.g. the Qwen3-4B
        qmaker) — the whole block is removed.
    (b) Closing-tag-only ``<reasoning>...</think><answer>`` with NO opening
        tag. The Qwen3-*-Thinking-2507 chat template injects the opening
        ``<think>`` into the PROMPT, so the completion starts mid-thought and
        only carries the closing ``</think>``. We keep everything after the
        LAST ``</think>`` — i.e. the answer part only.
    (c) An unclosed opening ``<think>`` (truncated mid-thinking) — everything
        from the tag to end-of-text is dropped (the conclusion never arrived),
        so we propagate an empty answer rather than leak chain-of-thought.

    Plain text with no think tags (e.g. the non-thinking Qwen2.5 student) is
    returned unchanged.
    """
    if not text:
        return ""
    # (a) closed <think>...</think> blocks that include the opening tag
    text = _THINK_BLOCK_RE.sub("", text)
    # (b) closing-tag-only style (Qwen3-Thinking-2507): keep after last </think>
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    # (c) unclosed opening <think> (truncated mid-thought) -> drop to end
    text = _UNCLOSED_THINK_RE.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# "Question: ..." label parser (qmaker output)
# ---------------------------------------------------------------------------

# Match a "Question:" line and capture the rest of that line (up to the next
# newline, or to end-of-text). The LAST match wins so an early exploratory
# mention inside the model's chain-of-thought can't shadow the final answer.
_QUESTION_LABEL_RE = re.compile(
    r"(?m)^\s*Question\s*:\s*(.+?)\s*$",
    re.IGNORECASE,
)


def parse_question_label(text: str) -> str | None:
    """Return the text after the LAST ``Question:`` line in the input.

    The qmaker is instructed to end its response with ``Question: <text>``
    on its own line. We strip ``<think>...</think>`` first so a candidate
    inside the model's thinking trace can't shadow the conclusion. Returns
    ``None`` if no ``Question:`` line is present (or the captured payload
    is empty).
    """
    if not text:
        return None
    text = strip_thinking(text)
    if not text:
        return None
    matches = _QUESTION_LABEL_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip() or None


# ---------------------------------------------------------------------------
# \boxed{...} parser (student final answer)
# ---------------------------------------------------------------------------


def _close_brace(text: str, start: int) -> str | None:
    """Contents of the brace group opening at ``start``, or None if never closed."""
    depth = 1
    i = start
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    return None


# ---------------------------------------------------------------------------
# Answer format of the STUDENT (Gemma line, 2026-10-28). Gemma 4 E2B ignores the \boxed{} instruction (57% of
# MATH500 answers unboxed even with it in the system prompt; 31% with it also on the user message) but follows
# 'End your response with a final line of the form "Final Answer: <answer>".' 100% (f2_probe, 1,000/1,000).
# SUBSOLVER_ANSWER_FORMAT=final_answer switches the student prompts and the student-answer extraction to that format;
# default "boxed" leaves every other line unchanged. Only STUDENT answers go through parse_answer — the qmaker's
# own \boxed / <question> outputs keep their parsers.
# ---------------------------------------------------------------------------
FINAL_ANSWER_INSTRUCTION = 'End your response with a final line of the form "Final Answer: <answer>".'
_BOXED_SENTENCE = "When you have the final answer, wrap it in \\boxed{...}."
_FINAL_ANSWER_RE = re.compile(r"final answer\s*\**\s*[:\uff1a]\s*\**\s*(.+)", re.IGNORECASE)


def answer_format() -> str:
    import os
    return os.environ.get("SUBSOLVER_ANSWER_FORMAT", "boxed")


def parse_final_answer(text: str) -> str | None:
    """Payload of the last 'Final Answer: ...' line (markdown bold, $...$, \\(...\\), a trailing period and an inner
    \\boxed{} are peeled off), or None when there is no such line."""
    if not text:
        return None
    text = strip_thinking(text)
    m = None
    for m in _FINAL_ANSWER_RE.finditer(text or ""):
        pass
    if m is None:
        return None
    a = m.group(1).strip().strip("*").strip()
    if a.endswith("."):
        a = a[:-1].rstrip()
    inner = parse_boxed(a)
    if inner is not None:
        return inner.strip()
    for lo, hi in (("$$", "$$"), ("$", "$"), ("\\(", "\\)"), ("\\[", "\\]")):
        if a.startswith(lo) and a.endswith(hi) and len(a) > len(lo) + len(hi):
            a = a[len(lo):-len(hi)].strip()
            break
    return a or None


def parse_answer(text: str) -> str | None:
    """Student final answer under the active answer format (see answer_format())."""
    return parse_final_answer(text) if answer_format() == "final_answer" else parse_boxed(text)


def apply_answer_format(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For SUBSOLVER_ANSWER_FORMAT=final_answer: replace the boxed sentence of the system prompt with
    FINAL_ANSWER_INSTRUCTION and append it to the last user message. Identity otherwise."""
    if answer_format() != "final_answer":
        return messages
    out = [dict(m) for m in messages]
    for m in out:
        if m.get("role") == "system" and isinstance(m.get("content"), str):
            m["content"] = m["content"].replace(_BOXED_SENTENCE, FINAL_ANSWER_INSTRUCTION)
    users = [i for i, m in enumerate(out) if m.get("role") == "user"]
    if users and FINAL_ANSWER_INSTRUCTION not in out[users[-1]]["content"]:
        out[users[-1]]["content"] = out[users[-1]]["content"] + "\n\n" + FINAL_ANSWER_INSTRUCTION
    return out


def parse_boxed(text: str) -> str | None:
    """Return the contents of the last COMPLETE ``\\boxed{...}`` in ``text``, or ``None``.

    Uses a depth counter so nested braces such as ``\\boxed{\\frac{1}{2}}`` are
    handled correctly. The last occurrence is preferred so that intermediate
    scratch-work boxes do not shadow the final answer.

    An unclosed trailing ``\\boxed{`` is skipped rather than failing the whole
    parse: a generation cut at its token cap often ends mid-brace, and models
    that repeat ``\\boxed{answer}`` until the cap would otherwise score 0 despite
    having committed to an answer hundreds of times. Scanning backwards keeps
    this cheap — the first closed box we meet is the answer.

    For thinking-capable models (Qwen3 hybrid, Qwen3-*-Thinking-2507) we
    first strip the ``<think>...</think>`` block so a boxed candidate the
    model explored inside thinking can't shadow the conclusion's boxed.
    If the response was truncated inside an unclosed ``<think>``, this
    returns ``None`` (no boxed-bearing conclusion arrived) — preferable
    to extracting a half-baked thought-internal boxed.
    """
    if not text:
        return None
    text = strip_thinking(text)
    if not text:
        return None

    needle = "\\boxed{"
    end = len(text)
    while True:
        idx = text.rfind(needle, 0, end)
        if idx == -1:
            return None
        payload = _close_brace(text, idx + len(needle))
        if payload is not None:
            return payload
        end = idx  # truncated mid-brace; look at the box before it


# ---------------------------------------------------------------------------
# Tool-call swap
# ---------------------------------------------------------------------------


def swap_tool_call_query(
    messages: list[dict[str, Any]],
    new_query: str,
    tool_name: str = ASK_ADVISOR_TOOL_NAME,
) -> tuple[list[dict[str, Any]], str]:
    """Deep-copy ``messages`` and replace the ``query`` argument of the last
    ``ask_advisor`` tool call with ``new_query``.

    Returns ``(new_messages, tool_call_id)``. The tool_call_id is needed so the
    matching tool-response message can be appended with the correct id.

    Raises ``ValueError`` if no matching tool call is found.
    """
    messages = copy.deepcopy(messages)
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            fn = tc.get("function", {})
            if fn.get("name") != tool_name:
                continue
            args_raw = fn.get("arguments", "{}")
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw)
                except json.JSONDecodeError:
                    args = {}
            else:
                args = dict(args_raw)
            args["query"] = new_query
            fn["arguments"] = json.dumps(args, ensure_ascii=False)
            tool_call_id = tc.get("id") or "call_0"
            tc["id"] = tool_call_id
            return messages, tool_call_id
    raise ValueError(
        f"No assistant tool_call to {tool_name!r} found in prefix messages."
    )


# ---------------------------------------------------------------------------
# Tool-response append
# ---------------------------------------------------------------------------


def append_tool_response(
    messages: list[dict[str, Any]],
    tool_call_id: str,
    content: str,
    tool_name: str = ASK_ADVISOR_TOOL_NAME,
) -> list[dict[str, Any]]:
    """Return a new messages list with an OpenAI-format tool-response appended."""
    out = list(messages)
    out.append(
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": content,
        }
    )
    return out


# ---------------------------------------------------------------------------
# Convenience: assemble the full student-continuation input
# ---------------------------------------------------------------------------


def assemble_student_messages(
    prefix_messages: list[dict[str, Any]],
    rewritten_question: str,
    advisor_answer: str,
) -> list[dict[str, Any]]:
    """Build the chat-completions ``messages`` payload for student continuation.

    Performs: (1) deep-copy prefix, (2) swap the ask_advisor query argument
    with ``rewritten_question``, (3) append the advisor's answer as a tool
    response.
    """
    swapped, tool_call_id = swap_tool_call_query(prefix_messages, rewritten_question)
    return append_tool_response(swapped, tool_call_id, advisor_answer)
