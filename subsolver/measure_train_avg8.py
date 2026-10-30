# Copyright 2025 Subsolver Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Measure solo avg@8 for every TRAIN problem and emit needAdvisor labels.

For each problem (solo rows of the CF train parquet, prompt used VERBATIM so the
measurement matches the training distribution), sample N solo solutions from the
base student, grade with math_verify, and label:

    needAdvisor = 1  if avg@N <= --threshold (default 0.25)   else 0

Generation runs async against comma-separated student endpoints; grading runs
post-hoc in the main process (math_verify spawns a ProcessPool that must live
under the __main__ guard, same caveat as eval_math500).

Output: --out jsonl, one row per problem:
    {group_id, problem, gt, n, correct, avg, needAdvisor}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys

import httpx
import pandas as pd
from tqdm.asyncio import tqdm as atqdm

_EXTRA_SAMPLING = json.loads(os.environ.get("SUBSOLVER_EXTRA_SAMPLING_JSON") or "{}")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def load_problems(parquet_path: str, limit: int | None):
    df = pd.read_parquet(parquet_path)
    out = []
    for _, row in df.iterrows():
        if str(row.get("mode", "")) != "solo":
            continue
        msgs = [dict(m) for m in list(row["prompt"])]
        from subsolver.pipeline import apply_answer_format
        msgs = apply_answer_format(msgs)
        gt = str(row["reward_model"]["ground_truth"])
        gid = str(row["group_id"])
        user = [m for m in msgs if m.get("role") == "user"]
        out.append({"group_id": gid, "messages": msgs, "gt": gt,
                    "problem": user[-1]["content"] if user else ""})
    if limit:
        out = out[:limit]
    return out


async def sample_one(sem, client, endpoints, model, prob, slot, args):
    async with sem:
        payload = {
            "model": model,
            "messages": prob["messages"],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
        }
        # per-served-model extra sampling (Qwen3.5 presence_penalty / top_k) from SUBSOLVER_EXTRA_SAMPLING_JSON
        payload.update(_EXTRA_SAMPLING.get(model, {}))
        last = None
        for attempt in range(4):
            try:
                r = await client.post(f"{random.choice(endpoints)}/v1/chat/completions", json=payload)
                r.raise_for_status()
                ch = r.json()
                txt = ch["choices"][0]["message"].get("content") or ""
                return {"group_id": prob["group_id"], "slot": slot, "text": txt,
                        "finish": ch["choices"][0].get("finish_reason"),
                        "tokens": (ch.get("usage") or {}).get("completion_tokens")}
            except (httpx.TransportError, httpx.HTTPStatusError) as e:  # noqa: PERF203
                last = e
                await asyncio.sleep(0.5 * (attempt + 1))
        return {"group_id": prob["group_id"], "slot": slot, "text": "", "error": str(last)}


async def generate(args, problems):
    endpoints = [u.strip().rstrip("/") for u in args.endpoints.split(",") if u.strip()]
    sem = asyncio.Semaphore(args.max_concurrent)
    timeout = httpx.Timeout(args.timeout, connect=30.0)
    limits = httpx.Limits(max_connections=args.max_concurrent * 2,
                          max_keepalive_connections=args.max_concurrent)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        tasks = [sample_one(sem, client, endpoints, args.model, p, k, args)
                 for p in problems for k in range(args.n)]
        return await atqdm.gather(*tasks, desc="avg8-gen")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/subsolver_cf_parquet/train.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--endpoints", default="http://127.0.0.1:8011,http://127.0.0.1:8013")
    ap.add_argument("--model", default="student")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--max_concurrent", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dump_jsonl", default=None,
                    help="optional per-sample dump {group_id, slot, score, error, text}")
    args = ap.parse_args()

    problems = load_problems(args.parquet, args.limit)
    print(f"[avg8] {len(problems)} problems x {args.n} samples", flush=True)
    gens = asyncio.run(generate(args, problems))

    # grade post-hoc in the main process (math_verify ProcessPool caveat)
    from verl.utils.reward_score import math_reward
    from subsolver.pipeline import answer_format, parse_final_answer
    by_gid = {p["group_id"]: p for p in problems}
    correct = {p["group_id"]: 0 for p in problems}
    counted = {p["group_id"]: 0 for p in problems}
    dump = open(args.dump_jsonl, "w") if args.dump_jsonl else None
    n_err = 0
    for g in gens:
        gid = g["group_id"]
        counted[gid] += 1
        n_err += int("error" in g)
        try:
            if answer_format() == "final_answer":  # Gemma line: grade the 'Final Answer:' payload with the same grader
                a = parse_final_answer(g["text"])
                s = float(math_reward.compute_score("\\boxed{" + a + "}", by_gid[gid]["gt"])) if a else 0.0
            else:
                s = float(math_reward.compute_score(g["text"], by_gid[gid]["gt"]))
        except Exception:  # noqa: BLE001
            s = 0.0
        correct[gid] += int(s > 0.5)
        if dump:
            dump.write(json.dumps({"group_id": gid, "slot": g["slot"], "score": s,
                                   "error": g.get("error"), "finish": g.get("finish"),
                                   "tokens": g.get("tokens"), "text": g["text"]},
                                  ensure_ascii=False) + "\n")
    if dump:
        dump.close()
    print(f"[avg8] request errors (scored 0): {n_err}/{len(gens)}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_need = 0
    with open(args.out, "w") as f:
        for p in problems:
            gid = p["group_id"]
            avg = correct[gid] / max(counted[gid], 1)
            need = 1 if avg <= args.threshold else 0
            n_need += need
            f.write(json.dumps({"group_id": gid, "problem": p["problem"][:500], "gt": p["gt"],
                                "n": counted[gid], "correct": correct[gid], "avg": round(avg, 4),
                                "needAdvisor": need}) + "\n")
    print(f"[avg8] wrote {len(problems)} -> {args.out}")
    print(f"[avg8] needAdvisor=1: {n_need}/{len(problems)} ({100*n_need/len(problems):.1f}%)")


if __name__ == "__main__":
    main()
