# Copyright 2025 Subsolver Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Rebuild qmaker parquets for the cost-aware recipe.

The qmaker system prompt is baked into each parquet row's ``prompt`` chat, so a
prompt change requires regenerating the data. This script copies the existing
qmaker parquets and swaps ONLY the system message for
``QMAKER_COST_SYSTEM_PROMPT`` — problems, student prefixes, ground truths and
row order are untouched, keeping the run comparable to qmaker_userq_dapo_v2.

Usage:
    python subsolver/build_costq_parquet.py \
        --train_src data/qmaker_userq_dapo_parquet/train.parquet \
        --val_src data/qmaker_aime2026_val_parquet/val.parquet \
        --out_dir data/qmaker_costq_parquet
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from subsolver.prompts import QMAKER_COST_SYSTEM_PROMPT  # noqa: E402


def swap_system(df: pd.DataFrame) -> pd.DataFrame:
    def _swap(prompt):
        msgs = [dict(m) for m in prompt]
        assert msgs and msgs[0]["role"] == "system", f"row without system message: {msgs[:1]}"
        msgs[0]["content"] = QMAKER_COST_SYSTEM_PROMPT
        return msgs

    df = df.copy()
    df["prompt"] = df["prompt"].map(_swap)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_src", default="data/qmaker_userq_dapo_parquet/train.parquet")
    ap.add_argument("--val_src", default="data/qmaker_aime2026_val_parquet/val.parquet")
    ap.add_argument("--out_dir", default="data/qmaker_costq_parquet")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    targets = [(args.train_src, args.out_dir, "train"), (args.val_src, args.out_dir, "val")]
    amc = "data/qmaker_costq_amc23_val_parquet/val.parquet"
    if os.path.exists(amc):
        targets.append((amc, os.path.dirname(amc), "val"))
    for src, out_dir, name in targets:
        df = swap_system(pd.read_parquet(src))
        out = os.path.join(out_dir, f"{name}.parquet")
        if name == "val":
            # keep the full split as val_full; the in-training val is every other
            # row (problem diversity kept, half the samples per problem) to cut
            # validation time.
            df.to_parquet(os.path.join(out_dir, "val_full.parquet"))
            df = df.iloc[::2].reset_index(drop=True)
        df.to_parquet(out)
        # roundtrip check: system prompt swapped, everything else intact
        back = pd.read_parquet(out)
        assert back.iloc[0]["prompt"][0]["content"] == QMAKER_COST_SYSTEM_PROMPT
        assert len(back) == len(df)
        print(f"{name}: {len(back)} rows -> {out}")


if __name__ == "__main__":
    main()
