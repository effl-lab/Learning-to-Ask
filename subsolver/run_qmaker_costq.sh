#!/usr/bin/env bash
# Cost-aware qmaker GRPO: reward = 1[correct] * (1 - advisor_tokens/cap).
#
# Advisor runs a GENERIC system prompt (no 10-line rule); the qmaker's boxed
# message carries the question + answering instructions and is forwarded to the
# advisor verbatim (see subsolver/cost_reward.py, QMAKER_COST_SYSTEM_PROMPT).
# Frozen advisor + student-continuation servers must already be up.
#
# Data: build data/qmaker_costq_parquet first:
#   python subsolver/build_costq_parquet.py
#
#   bash subsolver/run_qmaker_costq.sh
set -u
VERL_ROOT="${VERL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

QMAKER_PARQUET="${QMAKER_PARQUET:-$VERL_ROOT/data/qmaker_costq_parquet}"
RUN_NAME="${RUN_NAME:-qmaker_costq}"
CKPT_DIR="${CKPT_DIR:-$VERL_ROOT/checkpoints/$RUN_NAME}"
mkdir -p "$CKPT_DIR"

# Frozen-server endpoints (advisor LB + student-cont).
export SUBSOLVER_ADVISOR_ENDPOINT="${SUBSOLVER_ADVISOR_ENDPOINT:-http://127.0.0.1:8001,http://127.0.0.1:8003,http://127.0.0.1:8005}"
export SUBSOLVER_STUDENT_ENDPOINT="${SUBSOLVER_STUDENT_ENDPOINT:-http://127.0.0.1:8002}"
export SUBSOLVER_ADVISOR_MODEL=advisor
export SUBSOLVER_STUDENT_MODEL=student

# cap: serving max_tokens AND the reward normalizer (cost_ratio = tokens/cap).
export SUBSOLVER_ADVISOR_MAX_TOKENS="${SUBSOLVER_ADVISOR_MAX_TOKENS:-2048}"
# Hybrid-thinking advisor MUST run thinking-off: thinking tokens ignore any
# in-message length instructions (measured: 10/20/40-line prompts all ~13k tok
# with thinking on), which would destroy the cost signal the qmaker learns.
export SUBSOLVER_ADVISOR_DISABLE_THINKING="${SUBSOLVER_ADVISOR_DISABLE_THINKING:-1}"
# Gates: whole OFF, incomplete judge ON (v2 continuity — blocks the "lazy cheap
# query + student free-rides" leak that the cost bonus would otherwise reward).
# No separate cap gate: cost_ratio>=1 already scores 0.
export SUBSOLVER_WHOLE_REJECT="${SUBSOLVER_WHOLE_REJECT:-0}"
export SUBSOLVER_INCOMPLETE_REJECT="${SUBSOLVER_INCOMPLETE_REJECT:-1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

QMAKER_MODEL_PATH="${QMAKER_MODEL_PATH:-Qwen/Qwen3-4B}" \
TRAIN_FILE="$QMAKER_PARQUET/train.parquet" \
VAL_FILE="${VAL_FILE:-[$QMAKER_PARQUET/val.parquet,$VERL_ROOT/data/qmaker_costq_amc23_val_parquet/val.parquet]}" \
REWARD_PATH="pkg://subsolver.cost_reward" \
NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}" \
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}" \
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}" \
ROLLOUT_N="${ROLLOUT_N:-8}" \
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}" \
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}" \
ACTOR_LR="${ACTOR_LR:-1e-6}" \
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}" \
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.6}" \
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}" \
SAVE_FREQ="${SAVE_FREQ:-20}" \
TEST_FREQ="${TEST_FREQ:-10}" \
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-10}" \
PROJECT_NAME=subsolver_qmaker_costq \
EXPERIMENT_NAME="$RUN_NAME" \
VAL_DUMP_DIR="${VAL_DUMP_DIR:-$VERL_ROOT/logs/qmaker_val_dumps/$RUN_NAME}" \
  bash "$VERL_ROOT/subsolver/run_subsolver_grpo.sh" \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.resume_mode=auto \
    "$@"
