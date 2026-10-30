#!/usr/bin/env bash
# Guru line P2: train the multi-domain call gates on the eqmix_v2 parquets of p1_labels.sh — t12 (10 epochs) then t4
# (3 epochs) — and merge the best checkpoint (val decision score) to checkpoints/guru_gates/t{T}_eqmix_v2_best.
# Recipe = the published call-gate GRPO (run_callgate_grpo_17k.sh): batch 32/GPU, mini 16/GPU, n 8, lr 1e-6, KL 1e-3,
# one assistant turn (the tool never executes, no frozen servers), reward 1[called == need]; Guru prompts run longer than
# DAPO math, hence data.max_prompt_length=4096. The t12 gate is the stage-2 student (prefix harvest + continuation).
# Usage: [T="12 4"] bash subsolver/chains/guru/p2_gate.sh
set -u
. "$(cd "$(dirname "$0")" && pwd)/guru_common.sh"; cd "$ROOT"; guru_env
OUT=$ROOT/logs/guru/p2_gate; ST=$ROOT/logs/guru/chain.status; mkdir -p "$OUT" "$K"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] P2-guru $*" | tee -a "$ST"; }
for T in ${T:-12 4}; do
  D=$ROOT/data/guru_callgate_t${T}_eqmix_v2_parquet
  [ -f "$D/train.parquet" ] && [ -f "$D/val.parquet" ] || { log "t$T parquets missing ($D) — run p1_labels.sh first"; exit 1; }
  if [ "$T" = 12 ]; then train_gate guru_callgate_t12_eqmix_v2 "$D" 10 15 5 "$K/t12_eqmix_v2" || exit 1
  else train_gate "guru_callgate_t${T}_eqmix_v2" "$D" 3 25 10 "$K/t${T}_eqmix_v2" || exit 1; fi
done
log "ALL DONE"
