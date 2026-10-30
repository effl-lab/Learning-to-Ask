"""Relabel the Guru gate pool from the measured solo pass rates, per threshold.

The pool (build_guru_callgate_parquet.py) carries a provisional needAdvisor derived
from Guru's own qwen2.5_7b_pass_rate. The solo sweep (measure_solo_fast.py) re-measures
every problem under OUR harness (student system prompt, our sampling), and the rates
come out 1.6-2x higher than Guru's, with outliers as far apart as solo 1.000 vs guru
0.000. This script replaces the label with the measured one:

    needAdvisor = 1[ solo_pass_rate <= t/16 ]        t in {4, 12}

Train is balanced to 50/50 by downsampling the majority class — matching the original
call-gate recipe. Val keeps its natural distribution and is only relabeled; it is the
golden split every later builder copies (build_guru_gate_labels_eqmix.py).

Writes <out_root>/guru_callgate_t{T}_parquet/{train,val}.parquet.

Usage:
  python subsolver/build_guru_gate_labels.py --thresholds 4 12 \\
      --pool data/guru_callgate_t25 --sweep logs/guru/solo_avg16 --out_root data
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


def load_rates(path: str) -> dict[str, float]:
    """{extra_info.index: solo_pass_rate} from a measure_solo_fast.py results.jsonl."""
    rates = {}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("n_ok", 0) > 0:          # rows with only generation errors carry no signal
                rates[str(r["index"])] = float(r["solo_pass_rate"])
    return rates


def relabel(d: pd.DataFrame, rates: dict[str, float], thr: float) -> pd.DataFrame:
    idx = d["extra_info"].map(lambda e: str(e["index"]))
    d = d[idx.isin(rates)].copy()
    solo = d["extra_info"].map(lambda e: rates[str(e["index"])])
    need = (solo <= thr + 1e-9).astype(int)
    d["needAdvisor"] = need
    # tool_selection must be a real LIST: tool_agent_loop iterates the value to pick
    # tools, and iterating a string yields characters, so the rollout would get NO
    # tool schema at all and every rollout would abstain.
    d["extra_info"] = [
        {**dict(e), "needAdvisor": int(n), "solo_avg16": float(s),
         "tool_selection": ["ask_advisor"]}
        for e, n, s in zip(d["extra_info"], need, solo)
    ]
    return d


def balance(d: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """50/50 by downsampling the majority label, matching the original recipe."""
    g1 = d[d.needAdvisor == 1]
    g0 = d[d.needAdvisor == 0]
    k = min(len(g0), len(g1))
    take = pd.concat([
        g0.iloc[rng.choice(len(g0), k, replace=False)],
        g1.iloc[rng.choice(len(g1), k, replace=False)],
    ])
    return take.iloc[rng.permutation(len(take))].reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", type=int, nargs="+", default=[4, 12],
                    help="t: needAdvisor = 1[solo <= t/16]")
    ap.add_argument("--pool", default="data/guru_callgate_t25", help="pool dir with train/val parquets")
    ap.add_argument("--sweep", default="logs/guru/solo_avg16",
                    help="dir with {train,val}/results.jsonl from measure_solo_fast.py")
    ap.add_argument("--out_root", default="data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    train = pd.read_parquet(f"{args.pool}/train.parquet")
    val = pd.read_parquet(f"{args.pool}/val.parquet")
    r_train = load_rates(f"{args.sweep}/train/results.jsonl")
    r_val = load_rates(f"{args.sweep}/val/results.jsonl")
    print(f"  measured: train {len(r_train)}/{len(train)}, val {len(r_val)}/{len(val)}")

    for t in args.thresholds:
        thr = t / 16.0
        tr = relabel(train, r_train, thr)
        va = relabel(val, r_val, thr)
        raw_need = tr.needAdvisor.mean()
        bal = balance(tr, rng)
        out = f"{args.out_root}/guru_callgate_t{t}_parquet"
        os.makedirs(out, exist_ok=True)
        bal.to_parquet(f"{out}/train.parquet", index=False)
        va.to_parquet(f"{out}/val.parquet", index=False)
        print(f"\n  t{t} (solo <= {thr:.4f})")
        print(f"    train: {len(tr)} rows, need=1 {raw_need:.3f}  ->  balanced {len(bal)} rows")
        print(f"    val:   {len(va)} rows, need=1 {va.needAdvisor.mean():.3f} (natural distribution — golden)")
        tab = bal.groupby(["ability", "needAdvisor"]).size().unstack(fill_value=0)
        for dom, row in tab.iterrows():
            print(f"      {dom:9} need0 {row.get(0, 0):>5}  need1 {row.get(1, 0):>5}")
        print(f"    -> {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
