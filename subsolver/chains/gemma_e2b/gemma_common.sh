#!/usr/bin/env bash
# Shared pieces of the Gemma 4 E2B line (P2 merge -> P3 harvest -> P4 qmaker). Sourced, not run.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
K=$ROOT/checkpoints/gemma4_e2b; BASE=$K/base   # local Gemma 4 E2B-it snapshot; merged models live next to it
GDATA=$ROOT/data/gemma4_e2b                   # gate / qmaker parquets
ST=$ROOT/logs/gemma_e2b/chain.status
PY=$ROOT/.venv/bin/python; VLLM=$ROOT/.venv/bin/vllm
gemma_env(){ source "$ROOT/subsolver/gemma_env.sh"; export PYTHONPATH="$ROOT:${PYTHONPATH:-}"; unset SUBSOLVER_ADVISOR_API_KEY OPENAI_API_KEY; }
# vllm serve flags shared by every Gemma student server (gate / continuation): FlashInfer, TRT-LLM kernels off,
# native Gemma 4 tool + reasoning parsers (vLLM 0.32), sampling defaults from the model's generation_config
# (temperature 1.0 / top_p 0.95 / top_k 64 = the training sampling).
GEMMA_STUDENT_SERVE=(--enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4
                     --attention-backend FLASHINFER --attention-config '{"use_trtllm_attention": false}')
ADV=Qwen/Qwen3.5-35B-A3B-GPTQ-Int4
kill_port(){ for pid in $(lsof -ti :"$1" 2>/dev/null); do par=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' '); for c in $pid $par; do tr '\0' ' ' < /proc/$c/cmdline 2>/dev/null | grep -q "vllm serve" && kill "$c" 2>/dev/null; done; done; }
wait_up(){ for _ in $(seq 1 180); do sleep 10; curl -s -m 2 "localhost:$1/v1/models" | grep -q "$2" && return 0; done; return 1; }
# best val step of a run: "<step> <mean val score>" from the val dumps (<step>.jsonl, field score), "0 0" if none
best_step(){ "$PY" - "$1" "$2" <<'PYB'
import glob, json, os, sys
ck, vd = sys.argv[1], sys.argv[2]
def vs(step):
    c = [int(os.path.basename(f)[:-6]) for f in glob.glob(f"{vd}/*.jsonl") if os.path.basename(f)[:-6].isdigit() and int(os.path.basename(f)[:-6]) <= step]
    if not c: return -1.0
    rows = [json.loads(l) for l in open(f"{vd}/{max(c)}.jsonl") if l.strip()]
    return sum(r.get("score", 0) for r in rows) / max(len(rows), 1)
steps = sorted(int(d.split("_")[-1]) for d in glob.glob(f"{ck}/global_step_*"))
best = max(steps, key=vs) if steps else 0
print(f"{best} {vs(best):.4f}" if steps else "0 0")
PYB
}
# best train step of a run among checkpoints whose val score (the <val_dumps>/<step>.jsonl dump of that step) is
# >= <min_val>: "<step> <mean train reward> <lo>-<hi> <val score> ok", where the train reward is critic/score/mean averaged
# over the checkpoint's save interval (steps after the previous saved checkpoint up to and including it), read from the
# file-logger metrics_*.jsonl in <metrics_dir> (a later file wins for a step logged twice across restarts). If no
# checkpoint reaches <min_val>, the best-val one is returned with "fallback". "0 0" if there are no checkpoints.
best_train_step(){ "$PY" - "$1" "$2" "$3" "$4" <<'PYB'
import glob, json, os, sys
ck, md, vd, min_val = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
score = {}
for f in sorted(glob.glob(f"{md}/metrics_*.jsonl"), key=os.path.getmtime):
    for line in open(f):
        try: d = json.loads(line)
        except ValueError: continue
        v = d.get("data", {}).get("critic/score/mean")
        if isinstance(d.get("step"), int) and v is not None: score[d["step"]] = v
def val(step):
    f = f"{vd}/{step}.jsonl"
    if not os.path.exists(f): return None
    rows = [json.loads(l) for l in open(f) if l.strip()]
    return sum(r.get("score", 0) for r in rows) / max(len(rows), 1)
steps = sorted(int(d.split("_")[-1]) for d in glob.glob(f"{ck}/global_step_*") if d.split("_")[-1].isdigit())
cands = []
for i, s in enumerate(steps):
    lo = steps[i - 1] + 1 if i else 1
    w = [score[t] for t in range(lo, s + 1) if t in score]
    v = val(s)
    if w and v is not None: cands.append((s, sum(w) / len(w), lo, v))
if not cands: print("0 0"); sys.exit()
ok = [c for c in cands if c[3] >= min_val - 1e-9]
s, tr, lo, v = max(ok, key=lambda c: c[1]) if ok else max(cands, key=lambda c: c[3])
print(f"{s} {tr:.4f} {lo}-{s} {v:.4f} {'ok' if ok else 'fallback'}")
PYB
}
# merge an FSDP actor checkpoint to HF and complete it from the base dir (tokenizer, chat template, processor config)
gemma_merge(){ local src=$1 dst=$2
  [ -f "$dst/config.json" ] || "$PY" -m verl.model_merger merge --backend fsdp --local_dir "$src" --target_dir "$dst" || return 1
  local f n
  for f in "$BASE"/*; do n=$(basename "$f")
    case $n in *.safetensors|model.safetensors.index.json|config.json|README.md|LICENSE) ;; *) cp -L "$f" "$dst/$n" ;; esac
  done
}
