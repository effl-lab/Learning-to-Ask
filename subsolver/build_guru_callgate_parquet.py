"""Call-gate pool parquet from the Guru mix: math + stem + codegen.

The gate decides one thing per problem — call the advisor or answer alone — and
`subsolver/callgate_reward.py` scores only that decision:

    score = 1[called == needAdvisor]

so a row needs a `needAdvisor` label and nothing else. Guru ships
`qwen2.5_7b_pass_rate` for every row, measured on the base model this student is
built from (Qwen2.5-7B-Instruct), so a provisional label is a threshold away:

    needAdvisor = 1[qwen2.5_7b_pass_rate <= THRESHOLD]

That label is only a placeholder. The shipped rates were measured under Guru's own
harness, and the gate line re-measures every pool row under ours
(measure_solo_fast.py, solo avg@16) before build_guru_gate_labels*.py relabel the
rows. What this builder fixes is the POOL (which rows, which val split) and the row
schema every later stage reads.

Two things about the prompt, both deliberate:

  The Guru row's own `prompt` is kept verbatim as the user message. It already
  carries the answer-format instruction its grader needs — "output the final answer
  within \\boxed{}" for math and stem_web, "expert Python programmer ... generate a
  correct Python program" plus a fenced skeleton for codegen. Restating a format
  here would duplicate one and contradict the other.

  The system message is STUDENT_SYSTEM_PROMPT, identical for every row. That
  paragraph is the entire decision context the gate sees, so varying it per domain
  would make call rates incomparable between domains. (The per-domain code prompt
  is applied at the end of the line by build_guru_gate_codeprompt_parquet.py.)

Mix (see --help for overrides). stem_web is the binding constraint at 3,591 rows,
so "equal thirds" is not reachable; math and codegen are brought close instead:

    math      11,637   sampled from math__combined (54,404 available)
    codegen    8,796   all of leetcode2k + primeintellect (taco and the
                       benchmark-derived livecodebench split are excluded)
    stem       3,591   all of stem__web
    total     24,024   -> 600 val rows (natural distribution) + 23,424 train

Source parquets: the `train/` split of LLM360/guru-RL-92k (Reasoning360), under
--src (default: $GURU_TRAIN_DIR, else data/reasoning360/train).

Usage:
  python subsolver/build_guru_callgate_parquet.py --threshold 0.25 --out data/guru_callgate_t25
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

from subsolver.prompts import STUDENT_SYSTEM_PROMPT  # noqa: E402

DEFAULT_SRC = os.environ.get("GURU_TRAIN_DIR", "data/reasoning360/train")
# domain -> (file globs, rows to take; None = take everything available)
MIX = {
    "math":    (["math__combined_*.parquet"], 11637),
    "codegen": (["codegen__leetcode2k_*.parquet", "codegen__primeintellect_*.parquet"], None),
    "stem":    (["stem__web_*.parquet"], None),
}
PASS_COL = "qwen2.5_7b_pass_rate"
KEEP = ["data_source", "prompt", "reward_model", "extra_info", PASS_COL]


def load(src: str, domain: str) -> pd.DataFrame:
    pats, _ = MIX[domain]
    files = sorted({f for pat in pats for f in glob.glob(os.path.join(src, pat))})
    parts = []
    for p in files:
        d = pd.read_parquet(p)
        missing = [c for c in KEEP if c not in d.columns]
        if missing:
            raise KeyError(f"{p} lacks {missing}")
        d = d[KEEP].copy()
        d["source_file"] = os.path.basename(p)
        parts.append(d)
    if not parts:
        raise FileNotFoundError(f"no files for {domain} under {src} ({pats})")
    return pd.concat(parts, ignore_index=True)


def user_text(prompt) -> str:
    """The Guru prompt is a [{role, content}] array; take its user turn."""
    msgs = prompt.tolist() if hasattr(prompt, "tolist") else prompt
    for m in msgs:
        if m.get("role") == "user":
            return m["content"]
    return msgs[-1]["content"] if msgs else ""


def sample(d: pd.DataFrame, n: int | None, mode: str, thr: float,
           rng: np.random.Generator) -> pd.DataFrame:
    if n is None or n >= len(d):
        return d
    if mode == "random":
        return d.iloc[rng.choice(len(d), n, replace=False)]
    # stratified: take the same count from each side of the threshold, so the
    # label balance is a choice rather than an accident of the source mix
    hard = d[d[PASS_COL] <= thr]
    easy = d[d[PASS_COL] > thr]
    k = min(n // 2, len(hard), len(easy))
    take = pd.concat([
        hard.iloc[rng.choice(len(hard), k, replace=False)],
        easy.iloc[rng.choice(len(easy), k, replace=False)],
    ])
    if len(take) < n:  # one side ran out — top up from the remainder
        rest = d.drop(take.index)
        extra = min(n - len(take), len(rest))
        take = pd.concat([take, rest.iloc[rng.choice(len(rest), extra, replace=False)]])
    return take


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC, help="Guru train parquets (LLM360/guru-RL-92k, train/)")
    ap.add_argument("--threshold", type=float, default=0.25,
                    help="provisional needAdvisor = 1[pass_rate <= threshold]")
    ap.add_argument("--sampling", choices=["random", "stratified"], default="random",
                    help="how math is subsampled; stratified balances the label")
    ap.add_argument("--val_size", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/guru_callgate_t25")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    frames = []
    print(f"  {'domain':9} {'avail':>7} {'taken':>7} {'need=1':>7}")
    for dom in MIX:
        d = load(args.src, dom)
        d[PASS_COL] = pd.to_numeric(d[PASS_COL], errors="coerce").fillna(0.0)
        avail = len(d)
        d = sample(d, MIX[dom][1], args.sampling, args.threshold, rng)
        d["domain"] = dom
        frames.append(d)
        need = (d[PASS_COL] <= args.threshold).mean()
        print(f"  {dom:9} {avail:>7} {len(d):>7} {need:>7.3f}")

    m = pd.concat(frames, ignore_index=True)
    m = m.iloc[rng.permutation(len(m))].reset_index(drop=True)

    rows = []
    for i, r in m.iterrows():
        need = int(float(r[PASS_COL]) <= args.threshold)
        gid = f"train_{i}"
        rows.append({
            "data_source": "subsolver_student",     # routes to callgate_reward
            "prompt": [{"role": "system", "content": STUDENT_SYSTEM_PROMPT},
                       {"role": "user", "content": user_text(r["prompt"])}],
            "ability": r["domain"],
            "reward_model": dict(r["reward_model"]),
            "group_id": gid,
            "needAdvisor": need,
            "extra_info": {
                "split": "train", "index": str(i), "group_id": gid,
                "mode": "help", "tool_selection": ["ask_advisor"],
                "need_tools_kwargs": True,
                "tools_kwargs": {"ask_advisor": {"create_kwargs": {"group_id": gid}}},
                "needAdvisor": need,
                # keep the provenance the label came from, so a later run can
                # re-threshold without rebuilding from the source parquets.
                # stem_llm_judge.compute_score reads extra_info["question"]
                # unconditionally and KeyErrors without it — which the graders'
                # catch-all turns into a silent 0 for the whole domain.
                "question": user_text(r["prompt"]),
                "guru_pass_rate": float(r[PASS_COL]),
                "guru_data_source": str(r["data_source"]),
                "guru_source_file": str(r["source_file"]),
                "domain": r["domain"],
            },
        })
    out = pd.DataFrame(rows)
    val = out.iloc[:args.val_size].copy()
    train = out.iloc[args.val_size:].reset_index(drop=True)
    for d in (val,):
        d["extra_info"] = d["extra_info"].map(lambda e: {**e, "split": "val"})

    os.makedirs(args.out, exist_ok=True)
    train.to_parquet(f"{args.out}/train.parquet", index=False)
    val.to_parquet(f"{args.out}/val.parquet", index=False)
    print(f"\n  wrote {args.out}/train.parquet  {len(train)} rows")
    print(f"  wrote {args.out}/val.parquet    {len(val)} rows")
    print(f"  needAdvisor=1 overall: {out.needAdvisor.mean():.3f}"
          f"  (threshold {args.threshold}, sampling {args.sampling})")
    print("  per domain:")
    for dom, g in out.groupby("ability"):
        print(f"    {dom:9} {len(g):>6} rows  need=1 {g.needAdvisor.mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
