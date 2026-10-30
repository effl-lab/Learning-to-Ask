"""Extra Guru math rows to measure for the gate (the "mined" easy pool).

At t12 (need = 1[solo <= 0.75]) the easy side of the 24k pool is tiny (a few hundred
math rows after measurement), and the 50:50 balance caps every trainset at twice its
easy pool. Guru's shipped `qwen2.5_7b_pass_rate` is a weak but usable pre-screen for
"probably easy under our harness": candidates are the math__combined rows that are
NOT in the pool (by user text, train + val) and whose shipped rate lies in
[--lo, --hi) (default [0.4, 0.7)), kept in source order; --limit N keeps a seeded
sample of them.

Rows are wrapped with the pool row recipe (STUDENT_SYSTEM_PROMPT + user turn), so
measure_solo_fast.py measures them under the same prompt as the pool, and carry
extra_info = {index: "mine_<k>", orig_index: <Guru extra_info.index>,
guru_data_source, guru_pass_rate}. Measure them like the pool
(measure_solo_fast.py --n 16) and hand the results to
build_guru_gate_labels_eqmix.py --mine_rows / --mine_rates.

Usage:
  python subsolver/build_guru_mine_candidates.py --out data/guru_mine_math/candidates.parquet
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from subsolver.build_guru_callgate_parquet import DEFAULT_SRC, PASS_COL, user_text  # noqa: E402
from subsolver.prompts import STUDENT_SYSTEM_PROMPT  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC, help="Guru train parquets (LLM360/guru-RL-92k, train/)")
    ap.add_argument("--pool", default="data/guru_callgate_t25", help="pool dir (train/val parquets) whose rows are excluded")
    ap.add_argument("--lo", type=float, default=0.4, help="shipped pass-rate lower bound (inclusive)")
    ap.add_argument("--hi", type=float, default=0.7, help="shipped pass-rate upper bound (exclusive)")
    ap.add_argument("--limit", type=int, default=0, help="keep a seeded sample of N candidates (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/guru_mine_math/candidates.parquet")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.src, "math__combined_*.parquet")))
    if not files:
        raise FileNotFoundError(f"no math__combined_*.parquet under {args.src}")
    m = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    rate = pd.to_numeric(m[PASS_COL], errors="coerce")
    used: set[str] = set()
    for split in ("train", "val"):
        p = pd.read_parquet(f"{args.pool}/{split}.parquet", columns=["ability", "extra_info"])
        used |= set(p[p["ability"] == "math"]["extra_info"].map(lambda e: dict(e)["question"]))
    text = m["prompt"].map(user_text)
    keep = (rate >= args.lo) & (rate < args.hi) & ~text.isin(used)
    cand = m[keep]
    print(f"  math rows {len(m)}, shipped rate in [{args.lo}, {args.hi}): {int(((rate >= args.lo) & (rate < args.hi)).sum())}, "
          f"not in the pool: {len(cand)}")
    if args.limit and len(cand) > args.limit:
        rng = np.random.default_rng(args.seed)
        cand = cand.iloc[np.sort(rng.choice(len(cand), args.limit, replace=False))]
        print(f"  seeded sample: {len(cand)}")

    rows = []
    for k, (_, r) in enumerate(cand.iterrows()):
        q = user_text(r["prompt"])
        rows.append({
            "data_source": str(r["data_source"]),
            "prompt": [{"role": "system", "content": STUDENT_SYSTEM_PROMPT},
                       {"role": "user", "content": q}],
            "ability": "math",
            "reward_model": dict(r["reward_model"]),
            "extra_info": {
                "index": f"mine_{k}",
                "orig_index": str(dict(r["extra_info"]).get("index", "")),
                "guru_data_source": str(r["data_source"]),
                "guru_pass_rate": float(rate.loc[r.name]),
                "question": q,
            },
        })
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"  wrote {args.out}: {len(out)} candidates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
