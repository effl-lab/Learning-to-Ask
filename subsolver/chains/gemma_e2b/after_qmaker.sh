#!/usr/bin/env bash
# Gemma line qmaker chain: cap 2048 was stopped after its step-60 checkpoint and only merges here;
# then caps 512 -> 4096 -> 8192, each QMAKER_STEPS (default 60) steps, merged at the best mean-train-reward
# checkpoint (p4_qmaker.sh). Idempotent: rerun to resume (merged caps are skipped, partial runs resume).
set -u
T=${T:-12}; CAPS=${CAPS:-"2048 512 4096 8192"}
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/gemma_common.sh"; cd "$ROOT"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] QCHAIN t$T $*" | tee -a "$ST"; }
until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 15; done
log "start (caps: $CAPS, ${QMAKER_STEPS:-60} steps each)"
for CAP in $CAPS; do
  bash "$HERE/p4_qmaker.sh" "$CAP" "$T" || { log "P4 cap$CAP FAILED -> stop"; exit 1; }
done
log "ALL DONE (caps: $CAPS)"
