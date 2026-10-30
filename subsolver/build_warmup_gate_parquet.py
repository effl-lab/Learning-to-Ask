"""Gate training data with a call warm-up.

The 2B no-think gates collapsed to never-call within ~6 steps (calls 0.15 -> 0.00). Warm-up: the first
WARMUP_STEPS batches contain ONLY needAdvisor=1 rows, so every group that calls is rewarded for it; after that
the usual balanced 5:5 data. verl must read the file in order (data.shuffle=False, total_epochs=1), so the
epochs are laid out here: [warm-up block] + EPOCHS x [the full balanced train set, reshuffled per epoch].

    python subsolver/build_warmup_gate_parquet.py --train data/q35nt/gate_t4_parquet/train.parquet \
        --batch 256 --warmup_steps 5 --epochs 1 --out data/q35nt/gate_t4_warmup_parquet/train.parquet
"""
from __future__ import annotations

import argparse
import os

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="balanced gate train parquet (needAdvisor in extra_info)")
    ap.add_argument("--batch", type=int, required=True, help="train batch size (prompts per step)")
    ap.add_argument("--warmup_steps", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=1, help="passes over the balanced set after the warm-up")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.train)
    need = df.extra_info.map(lambda e: int(e["needAdvisor"]))
    n_warm = args.warmup_steps * args.batch
    need_rows = df[need == 1]
    assert len(need_rows) >= n_warm, f"only {len(need_rows)} need rows for a {n_warm}-row warm-up"
    warm = need_rows.sample(n=n_warm, random_state=args.seed)
    parts = [warm] + [df.sample(frac=1.0, random_state=args.seed + 1 + e) for e in range(args.epochs)]
    out = pd.concat(parts, ignore_index=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_parquet(args.out)

    o_need = out.extra_info.map(lambda e: int(e["needAdvisor"]))
    per_batch = [o_need[i:i + args.batch].mean() for i in range(0, len(out) - args.batch + 1, args.batch)]
    print(f"wrote {len(out)} rows -> {args.out}: warm-up {n_warm} need rows ({args.warmup_steps} steps) + "
          f"{args.epochs} x {len(df)} balanced; {len(per_batch)} full batches; need share per batch: "
          f"first {args.warmup_steps + 2} = {[round(x, 2) for x in per_batch[:args.warmup_steps + 2]]}, "
          f"after warm-up min {min(per_batch[args.warmup_steps:]):.2f} max {max(per_batch[args.warmup_steps:]):.2f}")


if __name__ == "__main__":
    main()
