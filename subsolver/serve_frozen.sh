#!/usr/bin/env bash
# Launch the two frozen vLLM OpenAI-compatible servers used by the Subsolver
# reward function: the advisor (answers rewritten questions) and the student
# (resumes from a tool-response message). Both endpoints are addressed by the
# environment variables read in subsolver/frozen_clients.py.
#
# Usage:
#   ADVISOR_MODEL=<hf-path-or-local> \
#   STUDENT_MODEL=<hf-path-or-local> \
#   ADVISOR_GPUS=0 STUDENT_GPUS=1 \
#   bash subsolver/serve_frozen.sh
#
# Override ADVISOR_PORT / STUDENT_PORT to change ports; tool-call parser may
# need to be adjusted for non-Qwen-style models (Llama-3.1: llama3_json, etc.).

set -euo pipefail

ADVISOR_MODEL="${ADVISOR_MODEL:?set ADVISOR_MODEL to the frozen advisor model path}"
STUDENT_MODEL="${STUDENT_MODEL:?set STUDENT_MODEL to the frozen student model path}"

ADVISOR_PORT="${ADVISOR_PORT:-8001}"
STUDENT_PORT="${STUDENT_PORT:-8002}"

ADVISOR_GPUS="${ADVISOR_GPUS:-0}"
STUDENT_GPUS="${STUDENT_GPUS:-1}"

ADVISOR_TP="${ADVISOR_TP:-1}"
STUDENT_TP="${STUDENT_TP:-1}"
ADVISOR_DP="${ADVISOR_DP:-1}"
STUDENT_DP="${STUDENT_DP:-1}"

# Explicit RPC ports for vLLM data-parallel coordinators. Without these vLLM
# picks ephemeral ports and the two servers race for the same port. Pick high, non-overlapping values that
# do not collide with VS Code remote (45000-45100).
ADVISOR_DP_RPC_PORT="${ADVISOR_DP_RPC_PORT:-51001}"
STUDENT_DP_RPC_PORT="${STUDENT_DP_RPC_PORT:-51002}"

# Tool-call parser used only by the student (advisor is plain Q&A).
STUDENT_TOOL_PARSER="${STUDENT_TOOL_PARSER:-hermes}"

LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "$LOG_DIR"

echo "[subsolver] launching advisor on GPU(s)=$ADVISOR_GPUS port=$ADVISOR_PORT TP=$ADVISOR_TP DP=$ADVISOR_DP rpc=$ADVISOR_DP_RPC_PORT"
# VLLM_USE_TRTLLM_ATTENTION=0: on Blackwell B200 + vLLM 0.11 + Qwen3-MoE, the
# auto-detected TRTLLM attention kernel produces garbage output. Disabling it
# keeps CUDA graphs / flashinfer base path (no eager penalty) while restoring
# correctness.
CUDA_VISIBLE_DEVICES="$ADVISOR_GPUS" \
VLLM_USE_TRTLLM_ATTENTION=0 \
    vllm serve "$ADVISOR_MODEL" \
        --port "$ADVISOR_PORT" \
        --served-model-name advisor \
        --tensor-parallel-size "$ADVISOR_TP" \
        --data-parallel-size "$ADVISOR_DP" \
        --data-parallel-rpc-port "$ADVISOR_DP_RPC_PORT" \
        > "$LOG_DIR/advisor.log" 2>&1 &
ADVISOR_PID=$!

echo "[subsolver] launching student on GPU(s)=$STUDENT_GPUS port=$STUDENT_PORT TP=$STUDENT_TP DP=$STUDENT_DP rpc=$STUDENT_DP_RPC_PORT parser=$STUDENT_TOOL_PARSER"
CUDA_VISIBLE_DEVICES="$STUDENT_GPUS" \
VLLM_USE_TRTLLM_ATTENTION=0 \
    vllm serve "$STUDENT_MODEL" \
        --port "$STUDENT_PORT" \
        --served-model-name student \
        --tensor-parallel-size "$STUDENT_TP" \
        --data-parallel-size "$STUDENT_DP" \
        --data-parallel-rpc-port "$STUDENT_DP_RPC_PORT" \
        --enable-auto-tool-choice \
        --tool-call-parser "$STUDENT_TOOL_PARSER" \
        > "$LOG_DIR/student.log" 2>&1 &
STUDENT_PID=$!

echo "[subsolver] advisor pid=$ADVISOR_PID  student pid=$STUDENT_PID"
echo "[subsolver] logs: $LOG_DIR/advisor.log  $LOG_DIR/student.log"
echo "[subsolver] Ctrl-C to stop both."

trap 'kill $ADVISOR_PID $STUDENT_PID 2>/dev/null || true' EXIT INT TERM
wait
