"""LiveCodeBench (712) as an AUXILIARY verl validation set for code integration training.

Same row schema as the code gate parquets (data/guru_code_gate_t4_parquet), so the
integration reward runs unchanged: prompt = [code student system prompt, LCB prompt],
needAdvisor = 1[solo_avg16 <= t/16] from the measured LCB golden, ground_truth = the
coder1 test JSON. data_source is `subsolver_student_lcb` so verl reports it as a separate
val-core/val-aux group. Never used for training (benchmark data policy).

Usage:
  python subsolver/build_lcb_integ_val_parquet.py --t 4 --out data/lcb_integ_val_parquet/val.parquet
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

BENCH = "data/bench/livecodebench.parquet"
GOLD = "data/guru_code/lcb_solo_code_prompt.parquet"
TEMPLATE = "data/guru_code_gate_t4_parquet/val.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=int, default=4)
    ap.add_argument("--out", default="data/lcb_integ_val_parquet/val.parquet")
    # Llama code line (10-18): the student's own LCB golden + gate val as schema template
    ap.add_argument("--gold", default=GOLD, help="LCB solo golden: .parquet or results.jsonl")
    ap.add_argument("--template", default=TEMPLATE, help="gate val parquet whose schema is copied")
    a = ap.parse_args()
    bench = pd.read_parquet(BENCH)
    gold = pd.read_json(a.gold, lines=True) if a.gold.endswith(".jsonl") else pd.read_parquet(a.gold)
    gold["index"] = gold["index"].astype(str)
    solo = dict(zip(gold["index"], gold["solo_pass_rate"]))
    tier = dict(zip(gold["index"], gold["difficulty"])) if "difficulty" in gold else {}
    # verl concatenates all val files with HF datasets, which refuses to align parquets whose
    # arrow types (string vs large_string) or extra_info struct fields differ — so the LCB
    # rows are written with the gate val's EXACT arrow schema and extra_info field set.
    import pyarrow as pa
    import pyarrow.parquet as pq
    gate_tbl = pq.read_table(a.template)
    tmpl = gate_tbl.slice(0, 1).to_pandas().iloc[0]
    system_msg = tmpl["prompt"][0]["content"]
    ei_fields = [f.name for f in gate_tbl.schema.field("extra_info").type]
    rows = []
    for _, r in bench.iterrows():
        pid = str(r["pid"]); spec = json.loads(r["grade"])
        if pid not in solo:
            continue
        need = int(solo[pid] <= a.t / 16)
        gid = f"lcb_val_{pid}"
        ei_full = {
            "split": "val", "index": pid, "group_id": gid, "mode": "help",
            "tool_selection": ["ask_advisor"],
            "need_tools_kwargs": True,
            "tools_kwargs": {"ask_advisor": {"create_kwargs": {"group_id": gid}}},
            "needAdvisor": need, "question": r["prompt"], "prefix": "",
            "guru_pass_rate": float("nan"), "guru_data_source": "codegen__livecodebench",
            "guru_source_file": f"livecodebench:{tier.get(pid, '')}", "domain": "codegen",
            "solo_avg16": float(solo[pid]), "label_threshold": a.t,
        }
        missing = [k for k in ei_fields if k not in ei_full]
        assert not missing, f"gate extra_info has fields this builder does not fill: {missing}"
        rows.append({
            "data_source": "subsolver_student_lcb",
            "prompt": [{"role": "system", "content": system_msg}, {"role": "user", "content": r["prompt"]}],
            "ability": "codegen",
            "reward_model": {"ground_truth": spec["ground_truth"], "style": "rule"},
            "group_id": gid,
            "needAdvisor": need,
            "extra_info": {k: ei_full[k] for k in ei_fields},
        })
    df = pd.DataFrame(rows)[gate_tbl.schema.names]
    tbl = pa.Table.from_pandas(df, schema=gate_tbl.schema, preserve_index=False)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pq.write_table(tbl, a.out)
    chk = pq.read_table(a.out)
    assert chk.schema.equals(gate_tbl.schema), "schema mismatch vs gate val"
    import datasets
    cat = datasets.concatenate_datasets([datasets.Dataset.from_parquet(a.template), datasets.Dataset.from_parquet(a.out)])
    print(f"{chk.num_rows} rows -> {a.out}; need rate {df['needAdvisor'].mean():.3f}; "
          f"HF concat with gate val OK ({len(cat)} rows)")


if __name__ == "__main__":
    main()
