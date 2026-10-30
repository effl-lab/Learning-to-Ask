"""Coding-only call-gate parquets from the measured pool.

Input: data/guru_code_pool/all.parquet (17,645 Guru codegen rows) + the solo sweep
results (measure_solo_fast.py --system_prompt code) giving `solo_pass_rate` per
`extra_info.index`. Label: needAdvisor = 1[solo_pass_rate <= t/16].

Split: a seeded natural-distribution VAL of --val_size rows is held out first
(same rows for every t, labels re-thresholded), then TRAIN is balanced 50:50 by
downsampling the majority class (the eqmix recipe: every easy row is used; hard
rows are sampled to match). Student system prompt = STUDENT_SYSTEM_PROMPT_CODE.
Row schema = the Guru gate parquets (data_source subsolver_student -> callgate
reward, extra_info.tool_selection as a real LIST, extra_info.question for the
graders, extra_info.prefix kept for coder1's functional tests).

Usage:
  python subsolver/build_guru_code_gate_parquet.py --t 4 \\
      --solo logs/guru_code/solo_code_prompt/results.jsonl \\
      --out data/guru_code_gate_t4_parquet
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from subsolver.prompts import STUDENT_SYSTEM_PROMPT_CODE  # noqa: E402


def user_text(prompt) -> str:
    msgs = prompt.tolist() if hasattr(prompt, "tolist") else prompt
    for m in msgs:
        if m.get("role") == "user":
            return m["content"]
    return msgs[-1]["content"] if msgs else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=int, required=True, help="threshold: need = solo <= t/16")
    ap.add_argument("--pool", default="data/guru_code_pool/all.parquet")
    ap.add_argument("--solo", required=True, help="results.jsonl from measure_solo_fast.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val_size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    thr = args.t / 16.0

    pool = pd.read_parquet(args.pool)
    solo = {}
    for l in open(args.solo):
        if l.strip():
            r = json.loads(l)
            if r.get("n_ok", 0) > 0 and r["solo_pass_rate"] == r["solo_pass_rate"]:
                solo[str(r["index"])] = float(r["solo_pass_rate"])
    pool["idx"] = pool["extra_info"].map(lambda e: str(e["index"]))
    pool = pool[pool["idx"].isin(solo)].reset_index(drop=True)
    pool["solo"] = pool["idx"].map(solo)
    pool["need"] = (pool["solo"] <= thr).astype(int)
    print(f"  measured rows {len(pool)} / need(t{args.t}) rate {pool['need'].mean():.3f} "
          f"| by source: " + ", ".join(f"{s}: n={len(g)} need={g['need'].mean():.2f}" for s, g in pool.groupby("data_source")))

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(pool))
    val = pool.iloc[perm[: args.val_size]]
    rest = pool.iloc[perm[args.val_size:]]
    easy = rest[rest["need"] == 0]; hard = rest[rest["need"] == 1]
    k = min(len(easy), len(hard))
    train = pd.concat([easy.sample(n=k, random_state=args.seed), hard.sample(n=k, random_state=args.seed)])
    train = train.iloc[rng.permutation(len(train))].reset_index(drop=True)
    print(f"  val {len(val)} (need {val['need'].mean():.3f}) | train {len(train)} = {k} easy + {k} hard "
          f"(pool easy {len(easy)} / hard {len(hard)})")

    def rows_of(df: pd.DataFrame, split: str):
        out = []
        for i, r in df.iterrows():
            ei0 = dict(r["extra_info"])
            gid = f"{split}_{r['idx']}"
            q = user_text(r["prompt"])
            out.append({
                "data_source": "subsolver_student",
                "prompt": [{"role": "system", "content": STUDENT_SYSTEM_PROMPT_CODE},
                           {"role": "user", "content": q}],
                "ability": "codegen",
                "reward_model": dict(r["reward_model"]),
                "group_id": gid,
                "needAdvisor": int(r["need"]),
                "extra_info": {
                    "split": split, "index": r["idx"], "group_id": gid, "mode": "help",
                    "tool_selection": ["ask_advisor"], "need_tools_kwargs": True,
                    "tools_kwargs": {"ask_advisor": {"create_kwargs": {"group_id": gid}}},
                    "needAdvisor": int(r["need"]),
                    "question": q,
                    "prefix": ei0.get("prefix"),
                    "guru_pass_rate": float(ei0.get("guru_pass_rate") or float("nan")),
                    "guru_data_source": str(ei0.get("guru_data_source")),
                    "guru_source_file": str(ei0.get("guru_source_file")),
                    "domain": "codegen",
                    "solo_avg16": float(r["solo"]),
                    "label_threshold": args.t,
                },
            })
        return out

    os.makedirs(args.out, exist_ok=True)
    for split, df in (("train", train), ("val", val)):
        out = pd.DataFrame(rows_of(df, split))
        out.to_parquet(f"{args.out}/{split}.parquet", index=False)
        back = pd.read_parquet(f"{args.out}/{split}.parquet")
        ts = dict(back.iloc[0]["extra_info"])["tool_selection"]
        assert isinstance((ts.tolist() if hasattr(ts, "tolist") else ts)[0], str)
        print(f"  wrote {args.out}/{split}.parquet: {len(back)} rows, need {back['needAdvisor'].mean():.3f}, "
              f"sources {back['extra_info'].map(lambda e: e['guru_data_source'].split('__')[-1]).value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
