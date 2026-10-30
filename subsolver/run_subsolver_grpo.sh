#!/usr/bin/env bash
# GRPO training of the Subsolver question-maker model.
#
# Required side process: the advisor + student vLLM servers must be running
# (see subsolver/serve_frozen.sh). Their endpoints are read from the
# environment variables exported below by subsolver/frozen_clients.py.
#
# Override any of the variables below from the calling environment.

set -xeuo pipefail

VERL_ROOT="${VERL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$VERL_ROOT"
source "${SUBSOLVER_VENV:-$VERL_ROOT/.venv}/bin/activate"  # SUBSOLVER_VENV: e.g. .venv-q35 for Qwen3.5
export RAY_ADDRESS="${RAY_ADDRESS:-local}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TMPDIR="$HOME/tmp" TRITON_CACHE_DIR="$HOME/.triton/cache" TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torchinductor"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
# Port isolation so the training vLLM workers don't collide with the frozen
# advisor/student servers (which use 8001-8005 + VLLM_PORT 60000/61000).
export VERL_PORT_RANGE="${VERL_PORT_RANGE:-55000-56000}"
export VLLM_PORT="${VLLM_PORT:-56200}"
export VLLM_PORT_STRIDE="${VLLM_PORT_STRIDE:-100}"

# ---- frozen-model endpoints ----
export SUBSOLVER_ADVISOR_ENDPOINT="${SUBSOLVER_ADVISOR_ENDPOINT:-http://127.0.0.1:8001}"
export SUBSOLVER_STUDENT_ENDPOINT="${SUBSOLVER_STUDENT_ENDPOINT:-http://127.0.0.1:8002}"
export SUBSOLVER_ADVISOR_MODEL="${SUBSOLVER_ADVISOR_MODEL:-advisor}"
export SUBSOLVER_STUDENT_MODEL="${SUBSOLVER_STUDENT_MODEL:-student}"

# ---- user-adjustable ----
INFER_BACKEND=${INFER_BACKEND:-vllm}
QMAKER_MODEL_PATH=${QMAKER_MODEL_PATH:?set QMAKER_MODEL_PATH to the question-maker base model}
TRAIN_FILE=${TRAIN_FILE:?set TRAIN_FILE to the subsolver train parquet}
VAL_FILE=${VAL_FILE:?set VAL_FILE to the subsolver val parquet}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}

ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}
# GRPO group size: number of question-maker samples per prompt.
ROLLOUT_N=${ROLLOUT_N:-8}

PROJECT_NAME=${PROJECT_NAME:-subsolver_grpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-subsolver_qmaker}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
# Keep only the N most recent actor checkpoints (older ones auto-pruned by verl).
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-2}
# Dump every validation step's per-trajectory generations to this directory
# (one JSONL per global step). Each line has the qmaker input + output, gt,
# reward score, and extra reward info. Set empty to disable.
VAL_DUMP_DIR=${VAL_DUMP_DIR:-${VERL_ROOT}/logs/val_dumps/${EXPERIMENT_NAME}}

DATA=(
    algorithm.adv_estimator=grpo
    data.train_files=${TRAIN_FILE}
    data.val_files=${VAL_FILE}
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
    algorithm.use_kl_in_reward=False
)

MODEL=(
    actor_rollout_ref.model.path=${QMAKER_MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768
)

ROLLOUT=(
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=262144
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=262144
)

REWARD=(
    reward.custom_reward_function.path=${REWARD_PATH:-pkg://subsolver.reward}
    reward.custom_reward_function.name=compute_score
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    +trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP}
    +trainer.validation_data_dir=${VAL_DUMP_DIR}
)

python3 -m ${MAIN_MODULE:-verl.trainer.main_ppo} \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "$@"
