# Copyright 2025 Subsolver Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Call-gate preprocess: ONE row per problem, decision-only labels.

Takes the CF parquet (help rows carry the STUDENT tool prompt) and an avg@8
label jsonl from measure_train_avg8.py, and emits rows for decision training:

  - prompt        : the HELP-row prompt verbatim (tool offered)
  - tool_selection: ["ask_advisor"]  (schema offered; never executed at
                    max_assistant_turns=1)
  - extra_info.needAdvisor: 1 if solo avg@8 <= threshold else 0  (from labels)

Reward (subsolver.callgate_reward) is decision-matching only.
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf_parquet", required=True, help="CF parquet (train or val)")
    ap.add_argument("--labels", required=True, help="jsonl from measure_train_avg8.py")
    ap.add_argument("--out", required=True)
    # Recompute needAdvisor = 1[avg <= threshold] from the continuous `avg` field
    # instead of trusting the jsonl's precomputed needAdvisor. Lets ONE avg@N
    # measurement feed multiple threshold variants (e.g. 1/16, 4/16, 8/16, 12/16).
    ap.add_argument("--threshold", type=float, default=None,
                    help="if set, need = 1[avg <= threshold]; else use jsonl needAdvisor")
    # Downsample the majority class so needAdvisor is 50/50. The decision-match reward
    # (1[called==need]) otherwise lets the gate exploit the base rate (never/always call).
    ap.add_argument("--balance", action="store_true",
                    help="downsample majority class to 50/50 need1/need0")
    ap.add_argument("--seed", type=int, default=0, help="rng seed for --balance downsample")
    args = ap.parse_args()

    labels = {}
    with open(args.labels) as f:
        for line in f:
            d = json.loads(line)
            labels[d["group_id"]] = (int(d["needAdvisor"]), float(d["avg"]))

    df = pd.read_parquet(args.cf_parquet)
    rows = []
    missing = 0
    for _, row in df.iterrows():
        if str(row.get("mode", "")) != "help":
            continue  # help row = STUDENT prompt with the tool
        gid = str(row["group_id"])
        if gid not in labels:
            missing += 1
            continue
        need, avg = labels[gid]
        if args.threshold is not None:
            need = 1 if avg <= args.threshold else 0
        ei = dict(row["extra_info"])
        ei["needAdvisor"] = need
        ei["solo_avg8"] = avg
        ei["tool_selection"] = list(ei.get("tool_selection", ["ask_advisor"]))
        rows.append({
            "data_source": row["data_source"],
            "prompt": list(row["prompt"]),
            "ability": row["ability"],
            "reward_model": dict(row["reward_model"]),
            "group_id": gid,
            "needAdvisor": need,
            "extra_info": ei,
        })
    out_df = pd.DataFrame(rows)
    n_raw = len(out_df)
    if args.balance and n_raw:
        pos = out_df[out_df["needAdvisor"] == 1]
        neg = out_df[out_df["needAdvisor"] == 0]
        k = min(len(pos), len(neg))
        if k == 0:
            print(f"[warn] cannot balance: pos={len(pos)} neg={len(neg)} -> writing unbalanced")
        else:
            out_df = pd.concat([pos.sample(n=k, random_state=args.seed),
                                neg.sample(n=k, random_state=args.seed)])
            out_df = out_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_parquet(args.out)
    n1 = int(out_df["needAdvisor"].sum()) if len(out_df) else 0
    thr = f"thr={args.threshold} " if args.threshold is not None else ""
    bal = "balanced " if args.balance else ""
    print(f"wrote {len(out_df)} {bal}rows -> {args.out}  ({thr}needAdvisor=1: {n1} = "
          f"{100*n1/max(len(out_df),1):.1f}%, from {n_raw} raw, label-missing skipped: {missing})")


if __name__ == "__main__":
    main()
