# Copyright 2025 Subsolver Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Generate student rollouts on math problems and keep ones that called ``ask_advisor``.

Connects to a running OpenAI-compatible vLLM server hosting the student model
(see ``subsolver/serve_student.sh``). For each row in the input parquet, the
student is invoked with the ``ask_advisor`` tool available; whenever the response
contains at least one ``ask_advisor`` tool call, the assistant message — content
plus tool_calls — is preserved as the student prefix and saved as one JSONL row
in the format consumed by ``examples/data_preprocess/subsolver_preprocess.py``::

    {
      "problem": "...",
      "student_prefix_messages": [system, user, assistant(with tool_calls)],
      "ground_truth": "..."
    }

Rollouts that do not invoke the tool are discarded.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from typing import Any

import httpx
import pyarrow.parquet as pq
from tqdm.asyncio import tqdm as atqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from subsolver.pipeline import sanitize_special_tokens  # noqa: E402
from subsolver.prompts import (  # noqa: E402
    ASK_ADVISOR_TOOL_NAME,
    ASK_ADVISOR_TOOL_SCHEMA,
    STUDENT_SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# Student call
# ---------------------------------------------------------------------------


_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def recover_tool_calls(message: dict[str, Any]) -> dict[str, Any]:
    """Rebuild ``tool_calls`` the vLLM server's Hermes parser dropped.

    A query carrying unescaped LaTeX (``\\frac``, ``\\(``, ``\\angle`` …) is
    invalid JSON, so vLLM's hermes_tool_parser raises "Invalid \\escape", returns
    ``tool_calls=None``, and leaves the raw ``<tool_call>{...}</tool_call>`` block
    sitting in ``content``. Those rollouts DID call — dropping them silently
    filters the dataset down to LaTeX-free queries (measured: 0.6% of requests on
    the t1 student vs **61%** on t4, i.e. ~74% of t4's calls destroyed).

    Recovery is the same trick as verl's agent_loop tool_parser (5fb378a9):
    doubling every backslash makes the LaTeX literal again, json.loads decodes it
    back to the original text, and we re-serialize valid JSON. Only blocks the
    server already failed on reach the fallback, so we can't corrupt a good parse.
    """
    if message.get("tool_calls"):
        return message
    content = message.get("content") or ""
    blocks = _TOOL_CALL_BLOCK_RE.findall(content)
    if not blocks:
        return message

    recovered: list[dict[str, Any]] = []
    for i, raw in enumerate(blocks):
        fc: Any = None
        for candidate in (raw, raw.replace("\\", "\\\\")):
            try:
                fc = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue
        if not isinstance(fc, dict) or fc.get("name") != ASK_ADVISOR_TOOL_NAME:
            continue
        args = fc.get("arguments")
        if isinstance(args, str):
            args_str = args
        else:
            args_str = json.dumps(args, ensure_ascii=False)
        recovered.append({
            "id": f"call_{i}",
            "type": "function",
            "function": {"name": ASK_ADVISOR_TOOL_NAME, "arguments": args_str},
        })
    if not recovered:
        return message
    message = dict(message)
    message["tool_calls"] = recovered
    # the raw block is the tool call, not reasoning — keep it out of the prefix
    message["content"] = _TOOL_CALL_BLOCK_RE.sub("", content).strip()
    return message


async def call_student(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    problem: str,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> dict[str, Any] | None:
    """One chat-completions request. Returns the assistant message dict, or None on error."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ],
        "tools": [ASK_ADVISOR_TOOL_SCHEMA],
        "tool_choice": "auto",
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    try:
        resp = await client.post(f"{endpoint}/v1/chat/completions", json=payload)
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001
        print(f"[warn] student call failed: {e}", file=sys.stderr)
        return None
    return recover_tool_calls(resp.json()["choices"][0]["message"])


def has_valid_ask_advisor_call(assistant_message: dict[str, Any]) -> bool:
    """True iff the message contains an ask_advisor tool call whose arguments
    decode to a dict with a non-empty string ``query`` field.

    Rejects double-encoded JSON (vLLM occasionally wraps an unparseable
    arguments blob as a JSON string-of-a-string) and calls where the model
    used a schema-violating key (e.g. ``question`` instead of ``query``).
    Failed-validation samples are discarded so the next n_samples attempt
    can produce a clean trace.
    """
    for tc in assistant_message.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        if fn.get("name") != ASK_ADVISOR_TOOL_NAME:
            continue
        args_raw = fn.get("arguments", "{}")
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                continue
        elif isinstance(args_raw, dict):
            args = args_raw
        else:
            continue
        if not isinstance(args, dict):
            continue
        q = args.get("query")
        if isinstance(q, str) and q.strip():
            return True
    return False


def build_prefix_messages(
    problem: str, assistant_message: dict[str, Any]
) -> list[dict[str, Any]]:
    """Reconstruct the [system, user, assistant(with tool_calls)] chat history.

    Strip the openai response message down to the fields needed for chat
    template re-application later: role, content, tool_calls.
    """
    clean_asst: dict[str, Any] = {
        "role": "assistant",
        # Strip any control-token strings the student hallucinated as text, so the
        # re-fed prefix can't carry fake turn boundaries (see sanitize_special_tokens).
        "content": sanitize_special_tokens(assistant_message.get("content") or ""),
    }
    tool_calls = assistant_message.get("tool_calls") or []
    if tool_calls:
        clean_asst["tool_calls"] = [
            {
                "id": tc.get("id") or f"call_{i}",
                "type": tc.get("type", "function"),
                "function": {
                    "name": (tc.get("function") or {}).get("name", ""),
                    "arguments": (tc.get("function") or {}).get("arguments", "{}"),
                },
            }
            for i, tc in enumerate(tool_calls)
        ]
    return [
        {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
        {"role": "user", "content": problem},
        clean_asst,
    ]


# ---------------------------------------------------------------------------
# Per-problem sampling
# ---------------------------------------------------------------------------


async def sample_until_tool_call(
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    *,
    endpoints: list[str],
    model: str,
    problem: str,
    ground_truth: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    n_samples: int,
) -> dict[str, Any] | None:
    """Sample up to ``n_samples`` times; return the first that invoked ask_advisor.

    ``endpoints`` is a list of OpenAI-compatible server URLs; one is picked at
    random per attempt for client-side load-balancing across N independent
    single-GPU vLLM processes (do NOT use vLLM DP)."""
    async with sem:
        for _ in range(n_samples):
            endpoint = random.choice(endpoints)
            msg = await call_student(
                client, endpoint, model, problem,
                temperature=temperature, top_p=top_p, max_tokens=max_tokens,
            )
            if msg is None:
                continue
            if has_valid_ask_advisor_call(msg):
                return {
                    "problem": problem,
                    "student_prefix_messages": build_prefix_messages(problem, msg),
                    "ground_truth": ground_truth,
                }
        return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def main_async(args: argparse.Namespace) -> None:
    table = pq.read_table(args.input_parquet)
    df = table.to_pandas()
    if args.limit:
        df = df.head(args.limit)
    print(f"Loaded {len(df)} problems from {args.input_parquet}")

    sem = asyncio.Semaphore(args.max_concurrent)
    timeout = httpx.Timeout(args.timeout, connect=30.0)
    limits = httpx.Limits(max_connections=args.max_concurrent * 2, max_keepalive_connections=args.max_concurrent)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)) or ".", exist_ok=True)

    endpoints = [e.strip().rstrip("/") for e in args.student_endpoint.split(",") if e.strip()]
    if not endpoints:
        raise ValueError("no --student_endpoint provided")
    print(f"Load-balancing student rollouts across {len(endpoints)} endpoint(s): {endpoints}")

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        tasks: list[asyncio.Task[dict[str, Any] | None]] = []
        for _, row in df.iterrows():
            problem = row["prompt"]
            rm = row["reward_model"]
            gt = rm["ground_truth"] if isinstance(rm, dict) else rm.get("ground_truth")
            tasks.append(
                asyncio.create_task(
                    sample_until_tool_call(
                        sem, client,
                        endpoints=endpoints, model=args.student_model,
                        problem=problem, ground_truth=str(gt),
                        temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_tokens, n_samples=args.n_samples,
                    )
                )
            )

        kept = 0
        with open(args.output_jsonl, "w") as f:
            for coro in atqdm.as_completed(tasks, total=len(tasks), desc="student rollouts"):
                out = await coro
                if out is None:
                    continue
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                f.flush()
                kept += 1

    print(f"Kept {kept} traces with ask_advisor calls out of {len(df)} problems.")
    print(f"Wrote -> {args.output_jsonl}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_parquet", default="data/train_data")
    ap.add_argument("--output_jsonl", default="data/question_train.jsonl")
    ap.add_argument(
        "--student_endpoint",
        default=os.environ.get("SUBSOLVER_STUDENT_ENDPOINT", "http://127.0.0.1:8002"),
    )
    ap.add_argument(
        "--student_model",
        default=os.environ.get("SUBSOLVER_STUDENT_MODEL", "student"),
    )
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument(
        "--n_samples", type=int, default=1,
        help="Number of sampling attempts per problem. The first that emits an "
             "ask_advisor call is kept; the rest are discarded.",
    )
    ap.add_argument("--max_concurrent", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument(
        "--limit", type=int, default=None,
        help="If set, only process the first N rows of the input parquet.",
    )
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
