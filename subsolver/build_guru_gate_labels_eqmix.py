"""Guru gate train parquets, 'eqmix' recipe: equal domain counts, stem oversampled.

Growing math to its measured maximum (the earlier "fullmix" trainset) halved stem's
share (13.5% -> 6.7%) and stem's val call rate collapsed 0.60 -> 0.35 — the gradient
share, not the epoch count, was the binding factor. eqmix pins the three domains to
the SAME row count, sized by stem-easy x3:

  t4  — every domain easy 2,064 + hard 2,064 (= 4,128 rows, 33.3% share):
          math / codegen: unique subsamples of their measured pools
          stem: easy = its measured easy rows x3 (the only repetition);
                hard = 2,064 unique rows
  t12 — stem x3 of a small slice (easy x3 + 247 hard x3); math and codegen stay at
        their per-domain 50:50 caps.

Labels are measured-only (need = 1[solo <= t/16]); the math pool is the pool's math
rows plus the measured mined rows (build_guru_mine_candidates.py). Repeated rows get
distinct group_ids (_r2/_r3) so rollout bookkeeping never collides. Val is the
600-row natural split of build_guru_gate_labels.py, copied verbatim.

Writes <out_root>/guru_callgate_t{4,12}_eqmix_parquet/{train,val}.parquet; feed them to
build_guru_gate_codeprompt_parquet.py for the per-domain student prompt.

Usage:
  python subsolver/build_guru_gate_labels_eqmix.py \\
      --pool data/guru_callgate_t25/train.parquet --sweep logs/guru/solo_avg16/train/results.jsonl \\
      --mine_rows data/guru_mine_math/candidates.parquet --mine_rates logs/guru/mine_math/results.jsonl \\
      --val_root data --out_root data
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from subsolver.build_guru_gate_labels import load_rates, relabel  # noqa: E402
from subsolver.prompts import STUDENT_SYSTEM_PROMPT  # noqa: E402

T4_PER_SIDE = 2064          # = 688 stem easy x 3 in the measured pool
T12_STEM_REPS = 3
T12_STEM_HARD = 247


def user_text(prompt) -> str:
    msgs = prompt.tolist() if hasattr(prompt, "tolist") else prompt
    for m in msgs:
        if m.get("role") == "user":
            return m["content"]
    return msgs[-1]["content"] if msgs else ""


def mined_rows(mine_rows: str | None, mine_rates: str | None, thr: float) -> pd.DataFrame:
    """All measured mined rows (unused Guru math), wrapped with the pool row recipe."""
    if not mine_rows or not mine_rates or not (os.path.exists(mine_rows) and os.path.exists(mine_rates)):
        print("  mined rows: none (no --mine_rows / --mine_rates)")
        return pd.DataFrame()
    rates = load_rates(mine_rates)
    cand = pd.read_parquet(mine_rows)
    rows = []
    for _, r in cand.iterrows():
        e = dict(r["extra_info"])
        i = str(e["index"])
        s = rates.get(i)
        if s is None:
            continue
        need = int(s <= thr + 1e-9)
        gid = f"mine_{i}"
        q = user_text(r["prompt"])
        rows.append({
            "data_source": "subsolver_student",
            "prompt": [{"role": "system", "content": STUDENT_SYSTEM_PROMPT},
                       {"role": "user", "content": q}],
            "ability": "math",
            "reward_model": dict(r["reward_model"]),
            "group_id": gid,
            "needAdvisor": need,
            "extra_info": {
                "split": "train", "index": i, "group_id": gid,
                "mode": "help", "tool_selection": ["ask_advisor"],
                "need_tools_kwargs": True,
                "tools_kwargs": {"ask_advisor": {"create_kwargs": {"group_id": gid}}},
                "needAdvisor": need, "solo_avg16": float(s),
                "question": q,
                "guru_pass_rate": float(e.get("guru_pass_rate", -1)),
                "guru_data_source": str(e.get("guru_data_source", "")),
                "guru_source_file": "mined_unused_math",
                "domain": "math",
            },
        })
    print(f"  mined rows: {len(rows)} measured of {len(cand)} candidates")
    return pd.DataFrame(rows)


def take(d: pd.DataFrame, k: int, rng) -> pd.DataFrame:
    k = min(k, len(d))
    return d.iloc[rng.choice(len(d), k, replace=False)]


def repeat(d: pd.DataFrame, reps: int) -> pd.DataFrame:
    """Duplicate rows with unique group_ids per copy."""
    outs = [d]
    for r in range(2, reps + 1):
        c = d.copy()
        c["group_id"] = c["group_id"].map(lambda g, r=r: f"{g}_r{r}")
        c["extra_info"] = c.apply(
            lambda row, r=r: {
                **dict(row["extra_info"]),
                "group_id": row["group_id"],
                "tools_kwargs": {"ask_advisor":
                                 {"create_kwargs": {"group_id": row["group_id"]}}},
            }, axis=1)
        outs.append(c)
    return pd.concat(outs, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/guru_callgate_t25/train.parquet")
    ap.add_argument("--sweep", default="logs/guru/solo_avg16/train/results.jsonl")
    ap.add_argument("--mine_rows", default="data/guru_mine_math/candidates.parquet")
    ap.add_argument("--mine_rates", default="logs/guru/mine_math/results.jsonl")
    ap.add_argument("--val_root", default="data",
                    help="root holding guru_callgate_t{T}_parquet/val.parquet (build_guru_gate_labels.py)")
    ap.add_argument("--out_root", default="data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rates = load_rates(args.sweep)

    for t in (4, 12):
        thr = t / 16.0
        base = relabel(pd.read_parquet(args.pool), rates, thr)
        mined = mined_rows(args.mine_rows, args.mine_rates, thr)
        pools = {
            "math": pd.concat([base[base.ability == "math"], mined], ignore_index=True),
            "stem": base[base.ability == "stem"],
            "codegen": base[base.ability == "codegen"],
        }
        parts = []
        for dom, pool in pools.items():
            g0 = pool[pool.needAdvisor == 0]
            g1 = pool[pool.needAdvisor == 1]
            if t == 4:
                if dom == "stem":
                    parts += [repeat(g0, 3), take(g1, T4_PER_SIDE, rng)]
                else:
                    parts += [take(g0, T4_PER_SIDE, rng),
                              take(g1, T4_PER_SIDE, rng)]
            else:
                if dom == "stem":
                    # x3 of a small stem slice (all easy / T12_STEM_HARD hard), not of
                    # the whole hard pool
                    parts += [repeat(g0, T12_STEM_REPS),
                              repeat(take(g1, T12_STEM_HARD, rng), T12_STEM_REPS)]
                else:
                    k = min(len(g0), len(g1))
                    parts += [take(g0, k, rng), take(g1, k, rng)]
        out = pd.concat(parts, ignore_index=True)
        out = out.iloc[rng.permutation(len(out))].reset_index(drop=True)

        dst = f"{args.out_root}/guru_callgate_t{t}_eqmix_parquet"
        os.makedirs(dst, exist_ok=True)
        out.to_parquet(f"{dst}/train.parquet", index=False)
        shutil.copy(f"{args.val_root}/guru_callgate_t{t}_parquet/val.parquet", f"{dst}/val.parquet")
        print(f"\n  t{t}: {len(out)} rows, need=1 {out.needAdvisor.mean():.3f}")
        tab = out.groupby(["ability", "needAdvisor"]).size().unstack(fill_value=0)
        for dom, row in tab.iterrows():
            print(f"    {dom:9} need0 {row.get(0, 0):>5}  need1 {row.get(1, 0):>5}")
        print(f"    -> {dst}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
