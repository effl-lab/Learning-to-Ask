# Copyright 2025 Subsolver Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Async HTTP clients for the frozen advisor and student-continuation servers.

Both endpoints are expected to be OpenAI-compatible chat-completions APIs (e.g.
``vllm serve`` with ``--served-model-name``). They are launched by
``serve_frozen.sh`` and addressed via the ``SUBSOLVER_ADVISOR_ENDPOINT`` /
``SUBSOLVER_STUDENT_ENDPOINT`` environment variables.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

import httpx

from .pipeline import strip_thinking  # canonical impl lives in pipeline.py
from .prompts import ADVISOR_SYSTEM_PROMPT, ASK_ADVISOR_TOOL_SCHEMA


def _env(name: str, default: str | None = None) -> str:
    val = os.environ.get(name, default)
    if not val:
        raise RuntimeError(
            f"Environment variable {name} is required for the subsolver reward "
            f"function but is not set."
        )
    return val


class AdvisorClient:
    """Calls the frozen advisor LLM with a single question and returns the answer."""

    def __init__(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        timeout: float = 600.0,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        system_prompt: str | None = None,
    ) -> None:
        # ``SUBSOLVER_ADVISOR_ENDPOINT`` may be a comma-separated list of URLs
        # for client-side load balancing across multiple advisor servers
        # (e.g. two separate single-GPU vLLM processes that together act as
        # a true DP-2 pool without going through vLLM's MoE all2all path).
        raw = endpoint or _env("SUBSOLVER_ADVISOR_ENDPOINT")
        self.endpoints: list[str] = [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
        assert self.endpoints, "AdvisorClient needs at least one endpoint"
        self.model = model or os.environ.get("SUBSOLVER_ADVISOR_MODEL", "advisor")
        self.timeout = timeout
        # max_tokens: explicit arg > env override > default 8192
        if max_tokens is None:
            max_tokens = int(os.environ.get("SUBSOLVER_ADVISOR_MAX_TOKENS", "8192"))
        self.max_tokens = max_tokens
        self.temperature = temperature
        # System prompt: explicit arg > default (the trained 10-line prompt).
        # The cost-aware qmaker recipe passes ADVISOR_GENERIC_SYSTEM_PROMPT here.
        self.system_prompt = system_prompt or ADVISOR_SYSTEM_PROMPT

    @property
    def endpoint(self) -> str:
        """Backwards-compat: return the first endpoint."""
        return self.endpoints[0]

    def _pick_endpoint(self) -> str:
        import random
        return random.choice(self.endpoints)

    async def answer_with_usage(
        self, question: str, max_tokens: int | None = None
    ) -> tuple[str, int]:
        """Return ``(answer, completion_tokens)`` for one advisor call.

        ``completion_tokens`` counts ALL tokens the advisor generated — the
        ``<think>...</think>`` trace AND the final answer (the strip only
        affects the returned text, not the count). This makes it a
        gaming-resistant proxy for how much work was offloaded to the advisor:
        a "just answer in one word" query still incurs the full thinking cost.
        Consumed by the reward's advisor-length penalty.

        ``max_tokens`` overrides the client-wide cap for THIS call."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens if max_tokens is None else int(max_tokens),
        }
        # Hybrid-thinking advisors (e.g. Qwen3.5) burn the token cap inside
        # <think> — disable thinking at the chat-template level when asked.
        if os.environ.get("SUBSOLVER_ADVISOR_DISABLE_THINKING", "") not in ("", "0", "false", "False"):
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        ep = self._pick_endpoint()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{ep}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
        text = strip_thinking(data["choices"][0]["message"]["content"] or "")
        tokens = int((data.get("usage") or {}).get("completion_tokens", 0) or 0)
        return text, tokens

    async def answer(self, question: str) -> str:
        text, _ = await self.answer_with_usage(question)
        return text


class StudentContinuationClient:
    """Resumes the frozen student model after a tool response has been appended.

    The full chat history (system + user problem + assistant reasoning + assistant
    tool_call + tool response) is passed verbatim. The student model generates
    one more assistant turn, which should end with the final answer in
    ``\\boxed{...}``.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        model: str | None = None,
        timeout: float = 240.0,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> None:
        # Comma-separated list for client-side LB, same contract as AdvisorClient.
        raw = endpoint or _env("SUBSOLVER_STUDENT_ENDPOINT")
        self.endpoints: list[str] = [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
        assert self.endpoints, "StudentContinuationClient needs at least one endpoint"
        self.model = model or os.environ.get("SUBSOLVER_STUDENT_MODEL", "student")
        self.timeout = timeout
        # Continuation length: explicit arg > env > 1024. Code continuations write a
        # whole program and need more room than a boxed math answer (subsolver 08-23).
        if max_tokens is None:
            max_tokens = int(os.environ.get("SUBSOLVER_STUDENT_MAX_TOKENS", "1024"))
        self.max_tokens = max_tokens
        # Thinking students (Qwen3.5 line) continue with sampling, not greedy, and need far more
        # than 240 s for a 16k-token continuation: env overrides, defaults unchanged.
        self.temperature = float(os.environ.get("SUBSOLVER_STUDENT_TEMPERATURE", temperature))
        self.top_p = os.environ.get("SUBSOLVER_STUDENT_TOP_P")
        self.timeout = float(os.environ.get("SUBSOLVER_STUDENT_TIMEOUT", self.timeout))
        # per-served-model extra sampling (e.g. top_k) from SUBSOLVER_EXTRA_SAMPLING_JSON
        self.extra = json.loads(os.environ.get("SUBSOLVER_EXTRA_SAMPLING_JSON") or "{}").get(self.model, {})

    async def continue_after_tool_response(
        self, messages: list[dict[str, Any]]
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": [ASK_ADVISOR_TOOL_SCHEMA],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.top_p is not None:
            payload["top_p"] = float(self.top_p)
        payload.update(self.extra)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{random.choice(self.endpoints)}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"] or ""
