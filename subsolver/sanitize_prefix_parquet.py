# Copyright 2025 Subsolver Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Re-sanitize an existing subsolver parquet in place (row-identical), removing
control-token *strings* that the student hallucinated as text into its reasoning.

Those strings (``<|im_start|>``, ``</s>``, …) survive generation under
``skip_special_tokens=True`` but get re-promoted to REAL control tokens when the
prefix is re-templated + re-tokenized for the frozen student / qmaker — injecting
spurious turn boundaries mid-content. We strip them from BOTH places the prefix
text lives in the parquet:

  - ``extra_info.student_prefix_messages[*].content``  (reward-time student re-feed)
  - ``prompt[*].content``                              (the qmaker's input)
  - ``extra_info.problem``                             (defense in depth)

The train/val split and row order are preserved exactly (we rewrite the same
rows), so the cleaned parquet is an apples-to-apples drop-in for re-running an
experiment against the original.

    python subsolver/sanitize_prefix_parquet.py \
        --in_dir  data/subsolver_parquet \
        --out_dir data/subsolver_parquet_clean
"""

from __future__ import annotations

import argparse
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from subsolver.pipeline import sanitize_special_tokens  # noqa: E402


def _san(s):
    return sanitize_special_tokens(s) if isinstance(s, str) else s


def sanitize_row(row: dict, stats: dict) -> dict:
    """Mutate one parquet row in place, sanitizing every text field the prefix
    flows through. Returns the row; bumps ``stats`` counters."""
    # prompt[*].content (qmaker system + user message carrying the flattened prefix)
    for m in row.get("prompt") or []:
        c = m.get("content")
        if isinstance(c, str):
            new = _san(c)
            if new != c:
                stats["prompt_fixed"] += 1
            m["content"] = new

    ei = row.get("extra_info") or {}
    if isinstance(ei.get("problem"), str):
        ei["problem"] = _san(ei["problem"])
    for m in ei.get("student_prefix_messages") or []:
        c = m.get("content")
        if isinstance(c, str):
            new = _san(c)
            if new != c:
                stats["msg_fixed"] += 1
            m["content"] = new
    return row


def process(in_path: str, out_path: str) -> None:
    table = pq.read_table(in_path)
    rows = table.to_pylist()
    stats = {"prompt_fixed": 0, "msg_fixed": 0}
    rows = [sanitize_row(r, stats) for r in rows]
    # Rebuild against the ORIGINAL schema so nested struct/list types are preserved.
    out = pa.Table.from_pylist(rows, schema=table.schema)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pq.write_table(out, out_path)
    print(
        f"{in_path} -> {out_path}: {len(rows)} rows  "
        f"(prompt msgs fixed={stats['prompt_fixed']}, prefix msgs fixed={stats['msg_fixed']})"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/subsolver_parquet")
    ap.add_argument("--out_dir", default="data/subsolver_parquet_clean")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    args = ap.parse_args()
    for split in args.splits:
        process(
            os.path.join(args.in_dir, f"{split}.parquet"),
            os.path.join(args.out_dir, f"{split}.parquet"),
        )


if __name__ == "__main__":
    main()
