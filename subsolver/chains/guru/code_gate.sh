#!/usr/bin/env bash
# Guru coding-only line: call gates for the code domain (the stage-3 integration student; integration_tool_config_code*.yaml
# read their tokenizer from checkpoints/guru_gates/code_t{T}_best).
#   1  pool     build_guru_code_pool.py -> data/guru_code_pool/all.parquet (primeintellect + leetcode2k + taco, 17,645 rows)
#   2  sweep    base student solo avg@16 under the CODE student prompt, graded by execution (coder1)
#               -> logs/guru_code/solo_code_prompt/results.jsonl
#   3  parquets build_guru_code_gate_parquet.py --t 4 / 12 -> data/guru_code_gate_t{T}_parquet (val 500 natural, train 50:50)
#   4  gates    t4 (3 epochs) then t12 (10 epochs), the p2_gate.sh recipe -> checkpoints/guru_gates/code_t{T}_best
# Idempotent; rerun to resume. Usage: bash subsolver/chains/guru/code_gate.sh
set -u
. "$(cd "$(dirname "$0")" && pwd)/guru_common.sh"; cd "$ROOT"; guru_env
OUT=$ROOT/logs/guru_code; ST=$OUT/chain.status; SOLO=$OUT/solo_code_prompt; POOL=$ROOT/data/guru_code_pool/all.parquet
mkdir -p "$OUT" "$K"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] CODE-guru $*" | tee -a "$ST"; }
[ -d "$GURU_TRAIN_DIR" ] || { log "Guru train parquets missing at $GURU_TRAIN_DIR"; exit 1; }
[ -f "$POOL" ] || { "$PY" subsolver/build_guru_code_pool.py --out "$(dirname "$POOL")" > "$OUT/pool.log" 2>&1 \
  && log "pool: $(tail -n 1 "$OUT/pool.log" | cut -c1-200)" || { log "pool FAILED (see $OUT/pool.log)"; exit 1; }; }
if [ ! -f "$SOLO/.done" ]; then
  teardown(){ stop_ports "${SERVE_PORTS[@]:-}"; }; trap teardown EXIT
  stop_ports 8501 8502 8503 8504 8505 8506 8507 8508
  serve_replicas "$STUDENT_BASE" student 8501 16384 0.85 "${GPUS[@]}" || exit 1
  export SUBSOLVER_STUDENT_ENDPOINT=$SERVE_EPS SUBSOLVER_STUDENT_MODEL=student
  log "student x$NGPU up; solo sweep (pool x16, code prompt, execution grading)"
  "$PY" subsolver/measure_solo_fast.py --parquet "$POOL" --system_prompt code --n 16 --max_concurrent 1024 --grade_workers 10 \
    --out "$SOLO" > "$OUT/sweep.log" 2>&1 && { touch "$SOLO/.done"; log "solo sweep DONE: $(grep -E '^  codegen__' "$OUT/sweep.log" | tail -n 3 | tr '\n' ' ' | cut -c1-220)"; } \
    || { log "solo sweep FAILED (see $OUT/sweep.log)"; exit 1; }
  teardown; trap - EXIT
fi
for T in 4 12; do
  D=$ROOT/data/guru_code_gate_t${T}_parquet
  [ -f "$D/train.parquet" ] || { "$PY" subsolver/build_guru_code_gate_parquet.py --t $T --pool "$POOL" --solo "$SOLO/results.jsonl" --out "$D" \
    > "$OUT/gate_parquet_t$T.log" 2>&1 && log "gate parquet t$T: $(tail -n 2 "$OUT/gate_parquet_t$T.log" | tr '\n' ' ' | cut -c1-200)" \
    || { log "gate parquet t$T FAILED (see $OUT/gate_parquet_t$T.log)"; exit 1; }; }
done
train_gate guru_code_gate_t4 "$ROOT/data/guru_code_gate_t4_parquet" 3 25 10 "$K/code_t4" || log "code_t4 FAILED — continuing"
train_gate guru_code_gate_t12 "$ROOT/data/guru_code_gate_t12_parquet" 10 15 5 "$K/code_t12" || exit 1
log "ALL DONE"
