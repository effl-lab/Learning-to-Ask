# Copyright 2025 Subsolver Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Build Phase 2 (student-training) parquets for verl multi-turn tool GRPO.

train: DAPO-MATH-17k (data/train_data.parquet) — same problems as Phase 1.
val:   AIME2025 + AIME2026 (MathArena), the OOD target.

Each row has the multi-turn-tool format verl expects:
  prompt        : [system(STUDENT_SYSTEM_PROMPT), user(problem)]
  data_source   : "subsolver_student"   (routes to subsolver.student_reward)
  reward_model  : {"style": "rule", "ground_truth": <answer>}
  extra_info    : {..., need_tools_kwargs: True,
                   tools_kwargs: {"ask_advisor": {"create_kwargs": {}}}}
"""

from __future__ import annotations

import argparse
import os
import sys

import datasets
import pyarrow.parquet as pq

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from subsolver.prompts import STUDENT_SYSTEM_PROMPT  # noqa: E402

DATA_SOURCE = "subsolver_student"


def _row(problem: str, ground_truth: str, idx: int, split: str) -> dict:
    gt = str(ground_truth)
    return {
        "data_source": DATA_SOURCE,
        "prompt": [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": gt},
        "extra_info": {
            "split": split,
            "index": idx,
            "need_tools_kwargs": True,
            # create_kwargs needs >=1 field (pyarrow can't write an empty struct);
            # the tool ignores it, but ground_truth is a harmless, useful payload.
            "tools_kwargs": {"ask_advisor": {"create_kwargs": {"ground_truth": gt}}},
        },
    }


def build_train(train_parquet: str) -> list[dict]:
    df = pq.read_table(train_parquet).to_pandas()
    rows = []
    for i, r in df.iterrows():
        problem = r["prompt"]
        rm = r["reward_model"]
        gt = rm["ground_truth"] if isinstance(rm, dict) else rm.get("ground_truth")
        rows.append(_row(problem, gt, int(i), "train"))
    return rows


def build_val() -> list[dict]:
    rows = []
    idx = 0
    for name, src in [("aime2025", "MathArena/aime_2025"), ("aime2026", "MathArena/aime_2026")]:
        ds = datasets.load_dataset(src, split="train")
        for r in ds:
            row = _row(r["problem"], r["answer"], idx, "val")
            row["extra_info"]["aime_year"] = name
            row["extra_info"]["problem_idx"] = str(r["problem_idx"])
            rows.append(row)
            idx += 1
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_parquet", default="data/train_data.parquet")
    ap.add_argument("--out_dir", default="data/subsolver_student_parquet")
    ap.add_argument("--limit_train", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_rows = build_train(args.train_parquet)
    if args.limit_train:
        train_rows = train_rows[: args.limit_train]
    datasets.Dataset.from_list(train_rows).to_parquet(os.path.join(args.out_dir, "train.parquet"))
    print(f"train: {len(train_rows)} -> {args.out_dir}/train.parquet")

    val_rows = build_val()
    datasets.Dataset.from_list(val_rows).to_parquet(os.path.join(args.out_dir, "val.parquet"))
    print(f"val:   {len(val_rows)} (AIME2025+2026) -> {args.out_dir}/val.parquet")


if __name__ == "__main__":
    main()
