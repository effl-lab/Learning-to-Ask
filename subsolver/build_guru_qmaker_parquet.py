"""Assemble the Guru qmaker training parquet from harvested prefixes.

Row shape mirrors data/qmaker_costq_parquet exactly (the format cost_reward_guru
replays), with the Guru-specific differences:

  data_source is the row's REAL Guru source (math__* / codegen__* / stem_web),
  not the constant 'subsolver'. The domain-aware reward dispatches the grader off
  that argument, so a codegen row gets executed and a stem row goes to the judge
  instead of everything being string-compared as math.

  extra_info additionally carries `question` — the stem judge KeyErrors without
  it (and the catch-all would silently zero the domain, as the smoke runs showed).

  extra_info carries `prefix`: Guru's leetcode2k rows ship an import header
  (`from typing import *`, ...) in extra_info.prefix that coder1 prepends before
  running the functional tests. The call-gate parquet builder rebuilt extra_info
  without it, so `List`/`Optional` in an otherwise correct solution raised
  NameError and scored 0. It is recovered here by joining the Guru source
  parquets on the user prompt text. Rows without one get None (coder1 checks
  `is not None`).

The prefix rendering copies examples/data_preprocess/subsolver_preprocess.py:
the qmaker sees only the assistant's reasoning plus "[ask_advisor query]\\n{q}"
per tool call — its own user message carries the problem separately.

Usage:
  python subsolver/build_guru_qmaker_parquet.py \\
      --train logs/guru_qmaker_t12/prefixes/train.jsonl \\
      --val   logs/guru_qmaker_t12/prefixes/val.jsonl \\
      --out   data/guru_qmaker_t12_parquet \\
      --guru_train_dir data/reasoning360/train
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from subsolver.qwen3_toolcall import message_reasoning  # noqa: E402
from subsolver.prompts import QMAKER_COST_SYSTEM_PROMPT, format_qmaker_user  # noqa: E402

DEFAULT_GURU_TRAIN_DIR = os.environ.get("GURU_TRAIN_DIR", "data/reasoning360/train")


def render_prefix_text(messages: list[dict[str, Any]]) -> str:
    """Assistant reasoning + '[ask_advisor query]\\n{q}' — system/user are omitted,
    the problem reaches the qmaker separately as 'Original problem: ...'."""
    parts: list[str] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        thinking = message_reasoning(msg)  # '' for non-thinking students
        if thinking:
            parts.append(thinking)
        content = msg.get("content") or ""
        if content:
            parts.append(content)
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                q = json.loads(fn.get("arguments") or "{}").get("query", "")
            except Exception:  # noqa: BLE001
                q = ""
            if q:
                parts.append(f"[ask_advisor query]\n{q}")
    return "\n\n".join(parts)


def _user_text(prompt) -> str:
    msgs = prompt.tolist() if hasattr(prompt, "tolist") else prompt
    for m in msgs:
        if m.get("role") == "user":
            return m["content"]
    return msgs[-1]["content"] if msgs else ""


def load_code_prefixes(guru_train_dir: str) -> dict[str, str]:
    """{user prompt text: extra_info.prefix} over every codegen__* Guru file that
    carries a prefix (today: leetcode2k only). Keyed on the exact user text, which
    is what the call-gate parquet stores as extra_info.question."""
    table: dict[str, str] = {}
    for f in sorted(glob.glob(os.path.join(guru_train_dir, "codegen__*.parquet"))):
        pf = pq.ParquetFile(f)
        if "prefix" not in str(pf.schema_arrow.field("extra_info").type):
            continue
        df = pf.read(columns=["prompt", "extra_info"]).to_pandas()
        n = 0
        for _, r in df.iterrows():
            pre = dict(r["extra_info"]).get("prefix")
            if pre:
                table[_user_text(r["prompt"])] = str(pre)
                n += 1
        print(f"  code prefixes: {os.path.basename(f)} -> {n}")
    return table


def build_rows(path: str, split: str, code_prefixes: dict[str, str],
               subsample: int = 0) -> list[dict[str, Any]]:
    """One parquet row per prefix. ``subsample`` keeps only the first N prefixes
    of a seeded shuffle — for val splits, to bound the advisor cost of every
    validation pass."""
    rows = []
    hit = miss = 0
    with open(path) as fh:
        lines = [line for line in fh if line.strip()]
    if subsample and len(lines) > subsample:
        import random
        random.Random(0).shuffle(lines)
        lines = lines[:subsample]
        print(f"  {split}: subsampled to {subsample} prefixes")
    for line in lines:
        r = json.loads(line)
        prefix = r["student_prefix_messages"]
        question = r.get("question", r["problem"])
        code_prefix = None
        if str(r["guru_data_source"]).startswith("codegen"):
            # prefer the import header carried by the harvest; fall back to the
            # text-join for older harvests without it.
            code_prefix = r.get("code_prefix")
            if code_prefix is None:
                code_prefix = code_prefixes.get(question) or code_prefixes.get(question.strip())
            if code_prefix is None:
                miss += 1
            else:
                hit += 1
        extra = {
            "problem": r["problem"],
            "ground_truth": r["ground_truth"],
            "student_prefix_messages": prefix,
            "index": str(r["index"]),
            "split": split,
            "question": question,
            "guru_data_source": r["guru_data_source"],
            "prefix": code_prefix,
            "system_prompt_kind": r.get("system_prompt_kind", "default"),
        }
        user_content = format_qmaker_user(r["problem"], render_prefix_text(prefix))
        rows.append({
            "data_source": r["guru_data_source"],
            "prompt": [
                {"role": "system", "content": QMAKER_COST_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "ability": r["domain"],
            "reward_model": {"style": "rule", "ground_truth": r["ground_truth"]},
            "extra_info": extra,
        })
    print(f"  {split}: codegen rows with import header {hit}, without {miss}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--guru_train_dir", default=DEFAULT_GURU_TRAIN_DIR)
    ap.add_argument("--val_subsample", type=int, default=0,
                    help="keep only N val prefixes (seeded) to bound the advisor "
                         "cost of each validation pass")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    code_prefixes = load_code_prefixes(args.guru_train_dir) if os.path.isdir(args.guru_train_dir) else {}
    if not code_prefixes:
        print(f"  WARNING: no code prefixes found under {args.guru_train_dir}; "
              "relying on the harvest's code_prefix passthrough for import headers")
    for split, src, sub in (("train", args.train, 0), ("val", args.val, args.val_subsample)):
        rows = build_rows(src, split, code_prefixes, subsample=sub)
        df = pd.DataFrame(rows)
        df.to_parquet(f"{args.out}/{split}.parquet", index=False)
        print(f"  {split}: {len(df)} rows -> {args.out}/{split}.parquet")
        for dom, g in df.groupby("ability"):
            print(f"    {dom:9} {len(g):>6}")
        back = pd.read_parquet(f"{args.out}/{split}.parquet")
        ei0 = dict(back.iloc[0]["extra_info"])
        assert back.iloc[0]["prompt"][0]["content"] == QMAKER_COST_SYSTEM_PROMPT
        assert "student_prefix_messages" in ei0 and "prefix" in ei0 and "question" in ei0
    return 0


if __name__ == "__main__":
    sys.exit(main())
