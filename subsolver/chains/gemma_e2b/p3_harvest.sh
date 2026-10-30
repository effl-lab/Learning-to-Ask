#!/usr/bin/env bash
# Gemma line P3: qmaker training prefixes from the t${T} gate.
# Pool = data/llama/harvest_pool (DAPO-17k help rows minus 500 held-out val; gen_guru_prefixes rebuilds the
# student messages itself and honours SUBSOLVER_ANSWER_FORMAT=final_answer, so the pool is model-agnostic).
# gen_guru_prefixes: attempts 4, every valid ask_advisor call kept, no dedup; training sampling, 16,384 tokens.
# Gate served by vLLM 0.32 (.venv) with the native gemma4 tool/reasoning parsers, one replica per GPU.
set -u
T=${T:-12}
. "$(cd "$(dirname "$0")" && pwd)/gemma_common.sh"; cd "$ROOT"; gemma_env
GATE=$K/gate_t${T}_best; POOL=$ROOT/data/llama/harvest_pool
OUT=$ROOT/logs/gemma_e2b/p3_harvest_t$T; QDATA=$GDATA/qmaker_t${T}_parquet
mkdir -p "$OUT/prefixes"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] P3 t$T $*" | tee -a "$ST"; }
[ -f "$QDATA/train.parquet" ] && [ -f "$QDATA/val.parquet" ] && { log "already built ($QDATA)"; exit 0; }
[ -f "$GATE/config.json" ] || { log "gate missing ($GATE)"; exit 1; }
[ -f "$POOL/train.parquet" ] || { log "harvest pool missing ($POOL)"; exit 1; }
GPUS=($(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')); PORTS=()
for i in "${!GPUS[@]}"; do PORTS+=($((8301+i))); done
teardown(){ for p in "${PORTS[@]}"; do kill_port "$p"; done; sleep 12; }
trap teardown EXIT
if [ ! -f "$OUT/prefixes/train.jsonl.done" ] || [ ! -f "$OUT/prefixes/val.jsonl.done" ]; then
  teardown
  for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} VLLM_PORT=$((60000+10*i)) nohup "$VLLM" serve "$GATE" --port "${PORTS[$i]}" \
      --served-model-name student --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --max-model-len 20480 \
      "${GEMMA_STUDENT_SERVE[@]}" > "$OUT/serve_${PORTS[$i]}.log" 2>&1 &
  done
  for p in "${PORTS[@]}"; do wait_up "$p" student || { log "gate server $p FAILED (see $OUT/serve_$p.log)"; exit 1; }; done
  export SUBSOLVER_STUDENT_ENDPOINT=$(IFS=,; echo "${PORTS[*]/#/http://127.0.0.1:}") SUBSOLVER_STUDENT_MODEL=student
  log "gate x${#PORTS[@]} up; harvesting (attempts 4, 16,384 tokens, temp 1.0 / top_p 0.95 / top_k 64)"
  for s in val train; do
    [ -f "$OUT/prefixes/$s.jsonl.done" ] && continue
    "$PY" subsolver/gen_guru_prefixes.py --parquet "$POOL/$s.parquet" --out "$OUT/prefixes/$s.jsonl" \
      --attempts 4 --max_concurrent 1024 --max_tokens 16384 --temperature 1.0 --top_p 0.95 --timeout 10800 \
      > "$OUT/harvest_$s.log" 2>&1 \
      && touch "$OUT/prefixes/$s.jsonl.done" && log "$s harvest DONE: $(grep -h 'done: kept\|keep\[' "$OUT/harvest_$s.log" | tr '\n' ' ' | cut -c1-200)" \
      || { log "$s harvest FAILED (see $OUT/harvest_$s.log)"; exit 1; }
  done
  teardown
fi
"$PY" subsolver/build_guru_qmaker_parquet.py --train "$OUT/prefixes/train.jsonl" --val "$OUT/prefixes/val.jsonl" \
  --out "$QDATA" > "$OUT/build_parquet.log" 2>&1 || { log "parquet build FAILED (see $OUT/build_parquet.log)"; exit 1; }
log "qmaker parquet: $("$PY" -c "import pandas as pd; print(len(pd.read_parquet('$QDATA/train.parquet')), '/', len(pd.read_parquet('$QDATA/val.parquet')))") (train/val) -> $QDATA"
log "ALL DONE"
