"""Re-bake the Guru call-gate parquets with the per-domain student prompt.

build_guru_gate_labels_eqmix.py writes every row with STUDENT_SYSTEM_PROMPT, i.e. the
\\boxed{} clause also on codegen rows whose own prompt asks for a fenced Python
program. The gates train with code rows under STUDENT_SYSTEM_PROMPT_CODE
(prompts.student_system_prompt). The system message is baked into the parquet
`prompt` column that the rollout replays, so this pass rewrites it — everything
else (rows, labels, group ids, tool_selection list, val split) is kept byte-for-byte.

--src is the eqmix builder's output dir ({train,val}.parquet) or an archive dir
holding t{T}_eqmix_{train,val}.parquet.

Usage:
  python subsolver/build_guru_gate_codeprompt_parquet.py --t 12 \\
      --src data/guru_callgate_t12_eqmix_parquet \\
      --out data/guru_callgate_t12_eqmix_v2_parquet
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from subsolver.prompts import (  # noqa: E402
    STUDENT_SYSTEM_PROMPT,
    STUDENT_SYSTEM_PROMPT_CODE,
    student_system_prompt,
)


def rebake(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    prompts, extras, kinds = [], [], {"code": 0, "default": 0}
    for _, r in df.iterrows():
        ei = dict(r["extra_info"])
        key = ei.get("guru_data_source") or ei.get("domain") or r["ability"]
        system = student_system_prompt(str(key))
        kinds["code" if system is STUDENT_SYSTEM_PROMPT_CODE else "default"] += 1
        msgs = [dict(m) for m in (r["prompt"].tolist() if hasattr(r["prompt"], "tolist") else r["prompt"])]
        assert msgs[0]["role"] == "system" and msgs[0]["content"].strip() == STUDENT_SYSTEM_PROMPT.strip(), \
            "unexpected system message in the archived parquet"
        msgs[0]["content"] = system
        prompts.append(msgs)
        # tool_selection must be a real list (a string iterates per character and
        # silently drops the tool schema from the rollout — commit da60d50)
        ts = ei.get("tool_selection")
        ei["tool_selection"] = list(ts.tolist() if hasattr(ts, "tolist") else ts)
        extras.append(ei)
    out = df.copy()
    out["prompt"] = prompts
    out["extra_info"] = extras
    return out, kinds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=int, required=True, help="gate threshold (4 or 12)")
    ap.add_argument("--src", default=None, help="default: data/guru_callgate_t<T>_eqmix_parquet")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    src_dir = args.src or f"data/guru_callgate_t{args.t}_eqmix_parquet"
    os.makedirs(args.out, exist_ok=True)
    for split in ("train", "val"):
        cands = [f"{src_dir}/{split}.parquet", f"{src_dir}/t{args.t}_eqmix_{split}.parquet"]
        src = next((c for c in cands if os.path.exists(c)), None)
        if src is None:
            raise FileNotFoundError(f"no {split} parquet under {src_dir} (tried {cands})")
        df = pd.read_parquet(src)
        out, kinds = rebake(df)
        dst = f"{args.out}/{split}.parquet"
        out.to_parquet(dst, index=False)
        back = pd.read_parquet(dst)
        assert len(back) == len(df)
        sys_by_dom = {}
        for _, r in back.iterrows():
            sys_by_dom.setdefault(r["ability"], set()).add(
                "code" if r["prompt"][0]["content"] == STUDENT_SYSTEM_PROMPT_CODE else "default")
        assert sys_by_dom.get("codegen", set()) <= {"code"} and \
            all(v == {"default"} for k, v in sys_by_dom.items() if k != "codegen"), sys_by_dom
        assert isinstance(dict(back.iloc[0]["extra_info"])["tool_selection"][0], str)
        need = back["needAdvisor"].astype(int).mean()
        print(f"  t{args.t} {split}: {len(back)} rows -> {dst}  prompts {kinds}  "
              f"need={need:.3f}  domains={back['ability'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
