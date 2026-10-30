"""Assemble the coding-only pool: Guru codegen train files -> one parquet.

Sources (reasoning360/data/train): codegen__primeintellect_7.5k (stdin/stdout),
codegen__leetcode2k_1.3k (functional asserts + extra_info.prefix import header),
codegen__taco_8.8k (stdin/stdout, some functional). livecodebench_440 is NOT
included (benchmark-derived; training-exclusion policy).

Row shape is what measure_solo_fast.py / the gate builder expect: the Guru row
unchanged (prompt = Guru user turn only, reward_model, extra_info incl. `prefix`
where present) plus extra_info.{index, guru_data_source, guru_source_file,
guru_pass_rate}. `index` is a stable id (`pi_N`, `lc_N`, `taco_N` by source row
position) so sweeps resume and labels join without ambiguity.

Usage:
  python subsolver/build_guru_code_pool.py --out data/guru_code_pool
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

GURU_TRAIN = os.environ.get("GURU_TRAIN_DIR", "data/reasoning360/train")
SOURCES = [
    ("codegen__primeintellect_7.5k.parquet", "pi"),
    ("codegen__leetcode2k_1.3k.parquet", "lc"),
    ("codegen__taco_8.8k.parquet", "taco"),
]
PASS_COL = "qwen2.5_7b_pass_rate"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/guru_code_pool")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    frames = []
    for fname, tag in SOURCES:
        df = pd.read_parquet(os.path.join(GURU_TRAIN, fname))
        rows = []
        for i, r in df.iterrows():
            ei = dict(r["extra_info"]) if r["extra_info"] is not None else {}
            ei = {k: v for k, v in ei.items()}
            ei.update({
                "index": f"{tag}_{i}",
                "guru_data_source": str(r["data_source"]),
                "guru_source_file": fname,
                "guru_pass_rate": float(pd.to_numeric(r.get(PASS_COL), errors="coerce")) if PASS_COL in df.columns else float("nan"),
            })
            rows.append({
                "data_source": str(r["data_source"]),
                "prompt": [dict(m) for m in (r["prompt"].tolist() if hasattr(r["prompt"], "tolist") else r["prompt"])],
                "ability": "codegen",
                "reward_model": dict(r["reward_model"]),
                "extra_info": ei,
            })
        out = pd.DataFrame(rows)
        has_prefix = sum(1 for e in out["extra_info"] if e.get("prefix"))
        print(f"  {fname:40s} {len(out):6d} rows  prefix headers: {has_prefix}")
        frames.append(out)
    pool = pd.concat(frames, ignore_index=True)
    # pyarrow needs one struct schema for extra_info: union the keys
    keys = sorted({k for e in pool["extra_info"] for k in e})
    pool["extra_info"] = [{k: e.get(k) for k in keys} for e in pool["extra_info"]]
    dst = f"{args.out}/all.parquet"
    pool.to_parquet(dst, index=False)
    back = pd.read_parquet(dst)
    assert len(back) == len(pool) and back["extra_info"].map(lambda e: e["index"]).is_unique
    print(f"  wrote {dst}: {len(back)} rows; data_source counts {back['data_source'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
