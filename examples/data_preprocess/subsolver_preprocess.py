# Copyright 2025 Subsolver Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Convert a Subsolver JSONL trace dataset into the parquet format verl expects.

Input JSONL: one row per (problem, pre-generated student prefix, ground truth).
Each row must have at least:

    {
      "problem": "<original problem statement>",
      "student_prefix_messages": [
          {"role": "system", "content": "..."},
          {"role": "user", "content": "..."},
          {"role": "assistant",
           "content": "...student reasoning...",
           "tool_calls": [{
              "id": "call_0",
              "type": "function",
              "function": {
                "name": "ask_advisor",
                "arguments": "{\\"query\\": \\"...student raw query...\\"}"
              }
           }]}
      ],
      "ground_truth": "<verifiable answer>"
    }

Output parquet has the columns verl's RLHFDataset expects:

    prompt           : chat messages for the question-maker
    reward_model     : {"style": "rule", "ground_truth": ...}
    data_source      : "subsolver"
    ability          : "subsolver"
    extra_info       : dict carrying problem + student_prefix_messages so the
                       reward function can reconstruct the student continuation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import datasets

# Make ``subsolver`` importable when running this script from anywhere.
_VERL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _VERL_ROOT not in sys.path:
    sys.path.insert(0, _VERL_ROOT)

from subsolver.prompts import QMAKER_SYSTEM_PROMPT, format_qmaker_user  # noqa: E402


def render_prefix_text(messages: list[dict[str, Any]]) -> str:
    """Flatten the student's prefix messages into plain text for the qmaker.

    The question-maker does NOT need to see the student's system prompt or its
    user message (the original problem is passed separately as
    ``Original problem: ...``). Only the student's reasoning and the
    ``ask_advisor`` query are rendered.
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role")
        if role in ("system", "user"):
            continue
        if role == "assistant":
            content = msg.get("content") or ""
            if content:
                parts.append(content)
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                if fn.get("name") != "ask_advisor":
                    continue
                args_raw = fn.get("arguments", "{}")
                if isinstance(args_raw, str):
                    try:
                        args = json.loads(args_raw)
                    except json.JSONDecodeError:
                        args = {}
                else:
                    args = dict(args_raw)
                q = args.get("query", "")
                parts.append(f"[ask_advisor query]\n{q}")
        elif role == "tool":
            content = msg.get("content") or ""
            if content:
                parts.append(f"[tool response]\n{content}")
    return "\n\n".join(parts).strip()


def build_row(example: dict[str, Any], idx: int, split: str) -> dict[str, Any]:
    problem = example["problem"]
    prefix_messages = example["student_prefix_messages"]
    ground_truth = example["ground_truth"]

    prefix_text = render_prefix_text(prefix_messages)

    return {
        "data_source": "subsolver",
        "prompt": [
            {"role": "system", "content": QMAKER_SYSTEM_PROMPT},
            {"role": "user", "content": format_qmaker_user(problem, prefix_text)},
        ],
        "ability": "subsolver",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "split": split,
            "index": idx,
            "problem": problem,
            "student_prefix_messages": prefix_messages,
            "ground_truth": ground_truth,
        },
    }


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", required=True, help="JSONL with training rows.")
    parser.add_argument(
        "--val_jsonl", default=None,
        help="Optional JSONL with validation rows. If omitted and --val_fraction>0, the val "
             "split is sampled from --train_jsonl.",
    )
    parser.add_argument(
        "--val_fraction", type=float, default=0.0,
        help="Fraction of --train_jsonl to hold out for validation. Ignored if --val_jsonl is set.",
    )
    parser.add_argument(
        "--split_seed", type=int, default=0,
        help="RNG seed for the val_fraction split. Stable across runs.",
    )
    parser.add_argument("--out_dir", required=True, help="Output directory for parquet files.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_rows = _load_jsonl(args.train_jsonl)
    val_rows: list[dict[str, Any]] | None = None

    if args.val_jsonl:
        val_rows = _load_jsonl(args.val_jsonl)
    elif args.val_fraction > 0:
        import random
        n = len(train_rows)
        n_val = max(1, int(round(n * args.val_fraction)))
        rng = random.Random(args.split_seed)
        idx = list(range(n))
        rng.shuffle(idx)
        val_idx = set(idx[:n_val])
        new_train = [train_rows[i] for i in range(n) if i not in val_idx]
        val_rows = [train_rows[i] for i in idx[:n_val]]
        train_rows = new_train
        print(f"Split: {n} -> train={len(train_rows)} val={len(val_rows)} (seed={args.split_seed})")

    train_data = [build_row(r, i, "train") for i, r in enumerate(train_rows)]
    datasets.Dataset.from_list(train_data).to_parquet(
        os.path.join(args.out_dir, "train.parquet")
    )
    print(f"Wrote {len(train_data)} train rows to {args.out_dir}/train.parquet")

    if val_rows is not None:
        val_data = [build_row(r, i, "val") for i, r in enumerate(val_rows)]
        datasets.Dataset.from_list(val_data).to_parquet(
            os.path.join(args.out_dir, "val.parquet")
        )
        print(f"Wrote {len(val_data)} val rows to {args.out_dir}/val.parquet")


if __name__ == "__main__":
    main()
