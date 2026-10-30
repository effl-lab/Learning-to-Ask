#!/usr/bin/env bash
# Frozen servers for qmaker GRPO training: advisor xN (client-side LB) +
# student-continuation x1. Default GPU layout (8x B200):
#   advisor GPU 0,1,2 (ports 8001/8003/8005) ; student-cont GPU 3 (port 8002)
#   qmaker TRAINING then runs on GPU 4,5,6,7.
#
# advisor = Qwen3-30B-A3B MoE: VLLM_USE_TRTLLM_ATTENTION=0 is MANDATORY on
# Blackwell (else gibberish). Each advisor is its own single-GPU process — the
# MoE all2all path dies under load when data-parallel, so we LB N separate
# processes client-side.
#
# Student-continuation must expose the hermes tool parser so the swapped
# ask_advisor tool-response turn is accepted.
#
# Usage: bash subsolver/serve_qmaker_frozen.sh   (launches detached, returns)
set -u
VERL_ROOT="${VERL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$VERL_ROOT/.venv/bin/activate"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TMPDIR="$HOME/tmp" TRITON_CACHE_DIR="$HOME/.triton/cache" TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torchinductor"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

ADVISOR_MODEL="${ADVISOR_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
# Student-continuation model: the MERGED call-gate student for this threshold.
STUDENT_MODEL="${STUDENT_MODEL:?set STUDENT_MODEL to the merged call-gate student dir}"
LOG_DIR="${LOG_DIR:-$VERL_ROOT/logs}"
mkdir -p "$LOG_DIR"

# GPU assignment (space-separated). Advisors get their own GPU each.
ADVISOR_GPUS="${ADVISOR_GPUS:-0 1 2}"
STUDENT_GPU="${STUDENT_GPU:-3}"
GPU_MEM_UTIL="${SERVE_GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${SERVE_MAX_MODEL_LEN:-16384}"

adv_port=8001
adv_stride=2
i=0
ADV_PORTS=()
for g in $ADVISOR_GPUS; do
  port=$((adv_port + i*adv_stride))
  ADV_PORTS+=("$port")
  CUDA_VISIBLE_DEVICES="$g" VLLM_USE_TRTLLM_ATTENTION=0 VLLM_PORT=$((60000 + i*100)) \
    nohup vllm serve "$ADVISOR_MODEL" \
      --port "$port" --served-model-name advisor \
      --tensor-parallel-size 1 --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --max-model-len "$MAX_MODEL_LEN" \
      > "$LOG_DIR/qm_advisor_${port}.log" 2>&1 &
  echo "[qm] advisor GPU $g port $port (pid $!)"
  i=$((i+1))
done

CUDA_VISIBLE_DEVICES="$STUDENT_GPU" VLLM_USE_TRTLLM_ATTENTION=0 VLLM_PORT=$((61000)) \
  nohup vllm serve "$STUDENT_MODEL" \
    --port 8002 --served-model-name student \
    --tensor-parallel-size 1 --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enable-auto-tool-choice --tool-call-parser hermes \
    > "$LOG_DIR/qm_student_8002.log" 2>&1 &
echo "[qm] student-cont GPU $STUDENT_GPU port 8002 (pid $!)"

# Emit the LB endpoint strings the training env should use.
adv_ep=$(IFS=,; echo "${ADV_PORTS[*]/#/http://127.0.0.1:}")
echo "[qm] launched. advisor endpoints: $adv_ep"
echo "[qm] wait for /v1/models 200 on ${ADV_PORTS[*]} and 8002 before training."
