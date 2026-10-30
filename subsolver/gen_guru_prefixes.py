"""Harvest student prefixes for the Guru qmaker: rollouts that call ask_advisor.

The qmaker's training row is built around `student_prefix_messages` — the
conversation up to and including the assistant turn that carries the ask_advisor
tool call. The reward replays exactly that prefix: it swaps the qmaker's rewritten
question into the call, appends the advisor's answer as the tool response, lets the
frozen student finish, and grades the completion. No prefix, no row.

Notes:

  * the student system prompt is chosen PER DOMAIN (`prompts.student_system_prompt`):
    codegen rows get the fenced-program clause, math/stem_web keep \\boxed{}.
    The prompt is baked into the prefix, so the reward-time continuation sees
    the same one — change the prompt, re-harvest.
  * per-domain attempted/kept counts are printed at the end (they used to be
    reconstructable only from the .seen file).

Rollouts that never produce a valid call within --attempts are dropped — matching
the original recipe, where the qmaker only ever trains on problems the gate
actually delegates.

Usage:
  SUBSOLVER_STUDENT_ENDPOINT=http://127.0.0.1:8301,... \\
  python subsolver/gen_guru_prefixes.py \\
      --parquet data/guru_callgate_t12_parquet/train.parquet \\
      --out logs/guru_qmaker_t12/prefixes/train.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict

import httpx
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from subsolver.generate_student_traces import (  # noqa: E402
    has_valid_ask_advisor_call,
    recover_tool_calls,
)
from subsolver.pipeline import apply_answer_format  # noqa: E402
from subsolver.prompts import (  # noqa: E402
    ASK_ADVISOR_TOOL_SCHEMA,
    STUDENT_SYSTEM_PROMPT_CODE,
    student_system_prompt,
)


def user_text(prompt) -> str:
    msgs = prompt.tolist() if hasattr(prompt, "tolist") else prompt
    for m in msgs:
        if m.get("role") == "user":
            return m["content"]
    return msgs[-1]["content"] if msgs else ""


class Endpoints:
    def __init__(self, raw: str) -> None:
        self.urls = [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]
        assert self.urls, "SUBSOLVER_STUDENT_ENDPOINT is empty"
        self._i = 0

    def pick(self) -> str:
        u = self.urls[self._i % len(self.urls)]
        self._i += 1
        return u


async def sample_once(sem, client, eps, model, messages, args):
    async with sem:
        payload = {"model": model, "messages": messages,
                   "temperature": args.temperature, "top_p": args.top_p,
                   "max_tokens": args.max_tokens,
                   "tools": [ASK_ADVISOR_TOOL_SCHEMA], "tool_choice": "auto"}
        # per-served-model extra sampling (e.g. Qwen3.5 top_k) from SUBSOLVER_EXTRA_SAMPLING_JSON
        payload.update(json.loads(os.environ.get("SUBSOLVER_EXTRA_SAMPLING_JSON") or "{}").get(model, {}))
        for attempt in range(3):
            try:
                r = await client.post(f"{eps.pick()}/v1/chat/completions", json=payload)
                r.raise_for_status()
                return recover_tool_calls(r.json()["choices"][0]["message"])
            except Exception:  # noqa: BLE001
                if attempt == 2:
                    return None
                await asyncio.sleep(2 * (attempt + 1))


async def run_problem(row, sem, client, eps, model, args, out_fh, state):
    ei = dict(row["extra_info"])
    problem = user_text(row["prompt"])
    domain = str(row["ability"])
    system = student_system_prompt(ei.get("guru_data_source") or domain)
    msgs = apply_answer_format([{"role": "system", "content": system},
                                {"role": "user", "content": problem}])  # identity unless SUBSOLVER_ANSWER_FORMAT=final_answer
    prefix = None
    used = 0
    for used in range(1, args.attempts + 1):
        m = await sample_once(sem, client, eps, model, msgs, args)
        if m is not None and has_valid_ask_advisor_call(m):
            keep = {"role": "assistant", "content": m.get("content") or "",
                    "tool_calls": m.get("tool_calls")}
            if m.get("reasoning"):  # thinking students (Qwen3.5): keep the <think> part of the turn
                keep["reasoning"] = m["reasoning"]
            prefix = msgs + [keep]
            break
    state["done"] += 1
    state["by_dom"][domain][0] += 1
    if prefix is not None:
        # Non-Guru pools (e.g. the DAPO callgate parquet) carry no
        # guru_data_source; label them <domain>__dapo so downstream grading
        # dispatch and the parquet builder still work.
        rec = {"index": ei["index"], "domain": domain,
               "guru_data_source": ei.get("guru_data_source") or f"{domain}__dapo",
               "problem": problem,
               "ground_truth": str(dict(row["reward_model"]).get("ground_truth")),
               "question": ei.get("question", problem),
               "student_prefix_messages": prefix,
               "system_prompt_kind": "code" if system is STUDENT_SYSTEM_PROMPT_CODE else "default",
               "attempts_used": used,
               # leetcode2k functional import header, carried so the qmaker
               # parquet builder does not have to recover it by a text-join.
               "code_prefix": ei.get("prefix")}
        out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_fh.flush()
        state["kept"] += 1
        state["by_dom"][domain][1] += 1
    if state["done"] % args.report_every == 0 or state["done"] == state["total"]:
        dt = time.time() - state["t0"]
        eta = dt / state["done"] * (state["total"] - state["done"])
        print(f"  {state['done']}/{state['total']}  kept {state['kept']} "
              f"({state['kept']/state['done']:.2f})  {dt/60:.1f} min  ETA {eta/60:.0f} min",
              flush=True)


async def main_async(args) -> int:
    eps = Endpoints(os.environ["SUBSOLVER_STUDENT_ENDPOINT"])
    model = os.environ.get("SUBSOLVER_STUDENT_MODEL", "student")
    d = pd.read_parquet(args.parquet)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done_idx = set()
    if os.path.exists(args.out):
        with open(args.out) as fh:
            done_idx = {json.loads(l)["index"] for l in fh if l.strip()}
    seen_path = args.out + ".seen"        # problems already ATTEMPTED (kept or not),
    if os.path.exists(seen_path):         # so resume does not redo the dropped ones
        done_idx |= {l.strip() for l in open(seen_path) if l.strip()}
    todo = d[~d["extra_info"].map(lambda e: str(e["index"])).isin(done_idx)]
    print(f"  {len(todo)}/{len(d)} problems, attempts<= {args.attempts}, concurrency {args.max_concurrent}",
          flush=True)

    sem = asyncio.Semaphore(args.max_concurrent)
    limits = httpx.Limits(max_connections=args.max_concurrent * 2)
    state = {"done": 0, "kept": 0, "total": len(todo), "t0": time.time(),
             "by_dom": defaultdict(lambda: [0, 0])}
    with open(args.out, "a") as out_fh, open(seen_path, "a") as seen_fh:
        async def wrapped(row):
            await run_problem(row, sem, client, eps, model, args, out_fh, state)
            seen_fh.write(str(dict(row["extra_info"])["index"]) + "\n")
            seen_fh.flush()
        async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout, connect=30.0),
                                     limits=limits) as client:
            await asyncio.gather(*[wrapped(r) for _, r in todo.iterrows()])
    print(f"  done: kept {state['kept']}/{state['done']}", flush=True)
    for dom, (att, kept) in sorted(state["by_dom"].items()):
        print(f"  keep[{dom}] {kept}/{att} = {kept / max(att, 1):.3f}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="1.0 matches the training rollout distribution")
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_concurrent", type=int, default=1024)
    ap.add_argument("--report_every", type=int, default=200)
    ap.add_argument("--timeout", type=float, default=1800.0)
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
