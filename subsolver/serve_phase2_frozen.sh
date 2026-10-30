#!/usr/bin/env bash
# Phase 2 frozen servers: advisor (2x) + question-maker (2x), client-side LB.
# GPU layout (8x B200): advisor on 0,1 ; qmaker on 2,3 ; student TRAINING on 4-7.
#
# VLLM_USE_TRTLLM_ATTENTION=0 is mandatory on Blackwell for Qwen3-MoE advisor
# (and harmless for the dense 4B qmaker).
#
# Usage:  bash subsolver/serve_phase2_frozen.sh
set -euo pipefail

ADVISOR_MODEL="${ADVISOR_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
QMAKER_MODEL="${QMAKER_MODEL:?set QMAKER_MODEL to the merged qmaker checkpoint dir}"

LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "$LOG_DIR"

launch() {  # gpu port model served_name memutil log
  local gpu="$1" port="$2" model="$3" name="$4" mem="$5" log="$6"
  echo "[phase2] $name on GPU $gpu port $port -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_USE_TRTLLM_ATTENTION=0 \
    nohup vllm serve "$model" \
      --port "$port" \
      --served-model-name "$name" \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization "$mem" \
      --max-model-len 16384 \
      > "$log" 2>&1 &
}

# advisor pool (GPU 0,1) — 30B MoE, give it more memory
launch 0 8001 "$ADVISOR_MODEL" advisor 0.85 "$LOG_DIR/p2_advisor_8001.log"
launch 1 8003 "$ADVISOR_MODEL" advisor 0.85 "$LOG_DIR/p2_advisor_8003.log"
# qmaker pool (GPU 2,3) — 4B dense
launch 2 8009 "$QMAKER_MODEL" qmaker 0.45 "$LOG_DIR/p2_qmaker_8009.log"
launch 3 8010 "$QMAKER_MODEL" qmaker 0.45 "$LOG_DIR/p2_qmaker_8010.log"

echo "[phase2] launched. advisor: 8001,8003  qmaker: 8009,8010"
echo "[phase2] tail logs in $LOG_DIR/p2_*.log ; wait for /health 200 before training."
