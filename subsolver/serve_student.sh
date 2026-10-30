#!/usr/bin/env bash
# Launch only the student vLLM server (OpenAI-compatible) used by
# subsolver/generate_student_traces.py to produce ask_advisor-bearing trajectories.
#
# Usage:
#   STUDENT_MODEL=<hf-path-or-local> STUDENT_GPUS=0 bash subsolver/serve_student.sh
#
# Override STUDENT_PORT to change the port. STUDENT_TOOL_PARSER must match the
# model family's tool-call format (hermes for Qwen2.5/Qwen3, llama3_json for
# Llama-3.1, etc.).

set -euo pipefail

STUDENT_MODEL="${STUDENT_MODEL:?set STUDENT_MODEL to the student model path}"
STUDENT_PORT="${STUDENT_PORT:-8002}"
STUDENT_GPUS="${STUDENT_GPUS:-0}"
STUDENT_TP="${STUDENT_TP:-1}"
STUDENT_DP="${STUDENT_DP:-1}"
STUDENT_TOOL_PARSER="${STUDENT_TOOL_PARSER:-hermes}"

LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "$LOG_DIR"

echo "[subsolver] student model=$STUDENT_MODEL GPU(s)=$STUDENT_GPUS port=$STUDENT_PORT TP=$STUDENT_TP DP=$STUDENT_DP parser=$STUDENT_TOOL_PARSER"
CUDA_VISIBLE_DEVICES="$STUDENT_GPUS" \
    exec vllm serve "$STUDENT_MODEL" \
        --port "$STUDENT_PORT" \
        --served-model-name student \
        --tensor-parallel-size "$STUDENT_TP" \
        --data-parallel-size "$STUDENT_DP" \
        --enable-auto-tool-choice \
        --tool-call-parser "$STUDENT_TOOL_PARSER"
