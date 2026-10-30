#!/usr/bin/env bash
# Gemma line chain after the t${T} gate training: wait for train.done, then P2 merge -> P3 harvest -> P4 qmaker
# at advisor caps 2048 / 4096 / 8192. Each phase is idempotent (markers), so rerun to resume.
set -u
T=${T:-12}; CAPS=${CAPS:-"2048 4096 8192"}
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/gemma_common.sh"; cd "$ROOT"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] CHAIN t$T $*" | tee -a "$ST"; }
log "waiting for gate training (logs/gemma_e2b/gemma4e2b_gate_t${T}/train.done)"
until [ -f "$ROOT/logs/gemma_e2b/gemma4e2b_gate_t${T}/train.done" ]; do
  grep -q "training FAILED" "$ST" 2>/dev/null && ! pgrep -f "^bash subsolver/chains/gemma_e2b/p2_gate.sh" >/dev/null && { log "gate training FAILED and no driver running -> stop"; exit 1; }
  sleep 120
done
while pgrep -f "^bash subsolver/chains/gemma_e2b/p2_gate.sh" >/dev/null; do sleep 30; done   # let the driver exit (GPUs free)
until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 15; done
log "gate done; P2 merge"
T=$T bash "$HERE/p2_merge.sh" || { log "P2 FAILED -> stop"; exit 1; }
T=$T bash "$HERE/p3_harvest.sh" || { log "P3 FAILED -> stop"; exit 1; }
for CAP in $CAPS; do
  bash "$HERE/p4_qmaker.sh" "$CAP" "$T" || { log "P4 cap$CAP FAILED -> stop"; exit 1; }
done
log "ALL DONE (caps: $CAPS)"
