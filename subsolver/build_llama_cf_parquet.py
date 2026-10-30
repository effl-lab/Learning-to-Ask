"""Build the CF-style (solo + help rows) math parquet for the Llama line.

The Qwen math line labelled DAPO-Math-17k with `measure_train_avg8.py` on the solo
rows of `data/subsolver_cf_parquet_17k`, then built gate parquets from the help rows
(`examples/data_preprocess/subsolver_callgate_preprocess.py`). That CF parquet is no
longer on disk, so this rebuilds an equivalent one:

  train : DAPO-Math-17k (subsolver_backup/data_parquets/dapo17k_genin.parquet,
          17,398 rows), group_id train_{i} with i = row index — the same indexing as
          the Qwen gate parquets (checked: extra_info.index -> identical prompt + gt).
  val   : MATH500 — the 500 rows of the Qwen t12 gate val parquet (group_id val_{i}).

Every problem yields two rows: mode=solo (SOLO_SYSTEM_PROMPT, no advisor mention —
used for the label measurement) and mode=help (STUDENT_SYSTEM_PROMPT, tool offered —
the gate's training prompt). Prompts are model-agnostic chat messages; the Llama
chat template is applied by the server / trainer.

    python subsolver/build_llama_cf_parquet.py --out data/llama/cf_dapo17k
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from subsolver.prompts import SOLO_SYSTEM_PROMPT, STUDENT_SYSTEM_PROMPT  # noqa: E402

DAPO = "subsolver_backup/data_parquets/dapo17k_genin.parquet"
VAL_SRC = "subsolver_backup/data_parquets/subsolver_callgate_7b_avg16_t12_parquet/val.parquet"


def rows_for(split: str, idx: int, problem: str, gt: str) -> list[dict]:
    out = []
    for mode, system in (("solo", SOLO_SYSTEM_PROMPT), ("help", STUDENT_SYSTEM_PROMPT)):
        out.append({
            "data_source": "subsolver_student",
            "prompt": [{"role": "system", "content": system},
                       {"role": "user", "content": problem}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": gt},
            "group_id": f"{split}_{idx}",
            "mode": mode,
            "extra_info": {
                "split": split, "index": idx, "group_id": f"{split}_{idx}", "mode": mode,
                "tool_selection": ["ask_advisor"] if mode == "help" else ["__none__"],
                "need_tools_kwargs": True,
                "tools_kwargs": {"ask_advisor": {"create_kwargs": {"ground_truth": gt, "mode": mode}}},
            },
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/llama/cf_dapo17k")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dapo = pd.read_parquet(DAPO)
    train = []
    for i, r in enumerate(dapo.itertuples()):
        train += rows_for("train", i, str(r.prompt).strip(), str(r.reward_model["ground_truth"]))

    vsrc = pd.read_parquet(VAL_SRC)
    val = []
    for r in vsrc.itertuples():
        user = [m for m in r.prompt if m["role"] == "user"][-1]["content"]
        val += rows_for("val", int(r.extra_info["index"]), user, str(r.reward_model["ground_truth"]))

    for name, rows in (("train", train), ("val", val)):
        df = pd.DataFrame(rows)
        p = os.path.join(args.out, f"{name}.parquet")
        df.to_parquet(p)
        print(f"{name}: {len(df)} rows ({len(df) // 2} problems) -> {p}")


if __name__ == "__main__":
    main()
