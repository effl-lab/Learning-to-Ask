#!/usr/bin/env bash
# Guru line P1: measured call-gate labels and the gate parquets (math + stem + codegen).
#   0  pool     build_guru_callgate_parquet.py -> data/guru_callgate_t25 (24,024 rows: 600 val + 23,424 train; the shipped
#               Guru pass rates are placeholders only) and build_guru_mine_candidates.py -> data/guru_mine_math/candidates.parquet
#   1  sweeps   base student solo avg@16, no tools (measure_solo_fast.py: student x(NGPU-1) + general-verifier judge x1;
#               codegen graded by execution, stem_web by the judge) -> logs/guru/solo_avg16/{train,val} and logs/guru/mine_math
#   2  labels   build_guru_gate_labels.py        need = 1[solo <= t/16], balanced train / natural val -> data/guru_callgate_t{4,12}_parquet
#               build_guru_gate_labels_eqmix.py  equal-domain train (+ mined math)            -> data/guru_callgate_t{4,12}_eqmix_parquet
#               build_guru_gate_codeprompt_parquet.py  per-domain student prompt             -> data/guru_callgate_t{4,12}_eqmix_v2_parquet
#               (the parquets p2_gate.sh trains on)
# Idempotent (markers under logs/guru/p1_labels); rerun to resume. Needs the Guru train parquets at $GURU_TRAIN_DIR.
# Usage: bash subsolver/chains/guru/p1_labels.sh
set -u
. "$(cd "$(dirname "$0")" && pwd)/guru_common.sh"; cd "$ROOT"; guru_env
OUT=$ROOT/logs/guru/p1_labels; ST=$ROOT/logs/guru/chain.status; SW=$ROOT/logs/guru/solo_avg16; MINE=$ROOT/logs/guru/mine_math
POOL=$ROOT/data/guru_callgate_t25; CAND=$ROOT/data/guru_mine_math/candidates.parquet
mkdir -p "$OUT" "$SW" "$MINE"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] P1-guru $*" | tee -a "$ST"; }
[ -d "$GURU_TRAIN_DIR" ] || { log "Guru train parquets missing at $GURU_TRAIN_DIR (train/ split of LLM360/guru-RL-92k)"; exit 1; }

# ---- 0. pool + mined candidates ----
[ -f "$POOL/val.parquet" ] || { "$PY" subsolver/build_guru_callgate_parquet.py --threshold 0.25 --src "$GURU_TRAIN_DIR" --out "$POOL" \
  > "$OUT/pool.log" 2>&1 && log "pool built: $(grep -E 'wrote' "$OUT/pool.log" | tr '\n' ' ' | cut -c1-200)" || { log "pool FAILED (see $OUT/pool.log)"; exit 1; }; }
[ -f "$CAND" ] || { "$PY" subsolver/build_guru_mine_candidates.py --src "$GURU_TRAIN_DIR" --pool "$POOL" --out "$CAND" > "$OUT/mine_candidates.log" 2>&1 \
  && log "mined candidates: $(tail -n 1 "$OUT/mine_candidates.log")" || { log "mined candidates FAILED (see $OUT/mine_candidates.log)"; exit 1; }; }

# ---- 1. solo sweeps (one fleet for all three) ----
need=0; for m in train val mine; do [ -f "$OUT/sweep_$m.done" ] || need=1; done
if [ $need = 1 ]; then
  [ "$NGPU" -ge 2 ] || { log "need >= 2 GPUs (students + judge)"; exit 1; }
  JUDGE_GPU=${GPUS[$((NGPU-1))]}; JUDGE_PORT=8110; STU_GPUS=("${GPUS[@]:0:$((NGPU-1))}")
  teardown(){ stop_ports "${SERVE_PORTS[@]:-}" $JUDGE_PORT; }; trap teardown EXIT
  stop_ports 8101 8102 8103 8104 8105 8106 8107 8108 $JUDGE_PORT
  CUDA_VISIBLE_DEVICES=$JUDGE_GPU VLLM_PORT=61000 nohup "$VLLM" serve "$JUDGE" --port $JUDGE_PORT --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.6 --max-model-len 4096 > "$OUT/serve_judge.log" 2>&1 &
  serve_replicas "$STUDENT_BASE" student 8101 16384 0.85 "${STU_GPUS[@]}" || exit 1
  wait_up $JUDGE_PORT general-verifier || { log "judge FAILED (see $OUT/serve_judge.log)"; exit 1; }
  export SUBSOLVER_STUDENT_ENDPOINT=$SERVE_EPS SUBSOLVER_STUDENT_MODEL=student STEM_LLM_JUDGE_URL=http://127.0.0.1:$JUDGE_PORT
  log "student x${#STU_GPUS[@]} + judge up; solo avg@16 sweeps (temp 0.7 / top_p 0.95 / 2048 tokens, no tools)"
  sweep(){ local name=$1 parquet=$2 out=$3
    [ -f "$OUT/sweep_$name.done" ] && return 0
    "$PY" subsolver/measure_solo_fast.py --parquet "$parquet" --n 16 --max_concurrent 1024 --out "$out" > "$OUT/sweep_$name.log" 2>&1 \
      && { touch "$OUT/sweep_$name.done"; log "$name sweep DONE: $(grep -E '^  (math|codegen|stem)' "$OUT/sweep_$name.log" | tail -n 4 | tr '\n' ' ' | cut -c1-220)"; } \
      || { log "$name sweep FAILED (see $OUT/sweep_$name.log)"; return 1; }
  }
  sweep train "$POOL/train.parquet" "$SW/train" || exit 1
  sweep val "$POOL/val.parquet" "$SW/val" || exit 1
  sweep mine "$CAND" "$MINE" || exit 1
  teardown; trap - EXIT
fi

# ---- 2. labels -> eqmix -> per-domain prompt ----
"$PY" subsolver/build_guru_gate_labels.py --pool "$POOL" --sweep "$SW" --out_root "$ROOT/data" --thresholds 4 12 > "$OUT/labels.log" 2>&1 \
  && log "labels: $(grep -E 'train:|val:' "$OUT/labels.log" | tr '\n' ' ' | cut -c1-240)" || { log "labels FAILED (see $OUT/labels.log)"; exit 1; }
"$PY" subsolver/build_guru_gate_labels_eqmix.py --pool "$POOL/train.parquet" --sweep "$SW/train/results.jsonl" \
  --mine_rows "$CAND" --mine_rates "$MINE/results.jsonl" --val_root "$ROOT/data" --out_root "$ROOT/data" > "$OUT/eqmix.log" 2>&1 \
  && log "eqmix: $(grep -E '^  t' "$OUT/eqmix.log" | tr '\n' ' ' | cut -c1-200)" || { log "eqmix FAILED (see $OUT/eqmix.log)"; exit 1; }
for T in 12 4; do
  "$PY" subsolver/build_guru_gate_codeprompt_parquet.py --t $T --src "$ROOT/data/guru_callgate_t${T}_eqmix_parquet" \
    --out "$ROOT/data/guru_callgate_t${T}_eqmix_v2_parquet" > "$OUT/codeprompt_t$T.log" 2>&1 \
    && log "t$T eqmix_v2: $(tail -n 2 "$OUT/codeprompt_t$T.log" | tr '\n' ' ' | cut -c1-200)" || { log "t$T re-bake FAILED (see $OUT/codeprompt_t$T.log)"; exit 1; }
done
log "ALL DONE (train the gates with subsolver/chains/guru/p2_gate.sh)"
