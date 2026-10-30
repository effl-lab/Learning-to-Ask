#!/usr/bin/env bash
# Gemma line P2 (tail): among the t${T} gate checkpoints with val score >= GATE_MIN_VAL (default 0.81), pick the one with
# the highest mean training reward over its save interval (see best_train_step) and merge it to HF -> $K/gate_t${T}_best
set -u
T=${T:-12}
. "$(cd "$(dirname "$0")" && pwd)/gemma_common.sh"; cd "$ROOT"; gemma_env
EXP=gemma4e2b_gate_t${T}; VD=$ROOT/logs/gemma_e2b/$EXP/val_dumps; OUT=$ROOT/logs/gemma_e2b/$EXP
log(){ echo "[$(date '+%m-%d %H:%M:%S')] P2-merge t$T $*" | tee -a "$ST"; }
[ -f "$K/gate_t${T}_best/config.json" ] && { log "already merged -> $(readlink -f $K/gate_t${T}_best)"; exit 0; }
B=$(best_train_step "$ROOT/checkpoints/$EXP" "$OUT" "$VD" "${GATE_MIN_VAL:-0.81}"); read -r STEP SC WIN VS SEL <<< "$B"
[ "$STEP" = "0" ] && { log "no checkpoint found"; exit 1; }
M=$K/gate_t${T}_step$STEP
log "best step $STEP (mean train reward $SC over steps $WIN; val score $VS; rule val >= ${GATE_MIN_VAL:-0.81} then max train reward: $SEL); merging $ROOT/checkpoints/$EXP/global_step_$STEP/actor -> $M"
gemma_merge "$ROOT/checkpoints/$EXP/global_step_$STEP/actor" "$M" >> "$OUT/merge.log" 2>&1 || { log "merge FAILED (see $OUT/merge.log)"; exit 1; }
ln -sfn "$(basename "$M")" "$K/gate_t${T}_best"
log "merged -> $M (symlink gate_t${T}_best)"
