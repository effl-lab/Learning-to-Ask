#!/usr/bin/env bash
# Call-gate GRPO on FULL DAPO-Math-17k (17,398 problems), 8 GPUs.
# Same recipe as run_callgate_grpo.sh (max_assistant_turns=1, adv=grpo, decision-
# matching reward, NO advisor/qmaker servers) but scaled:
#   - 8 GPUs, train_batch_size=256, ppo_mini_batch_size=128, 3 epochs (~204 steps)
#   - save_freq=50 (~5 ckpts), FULL checkpoints (save_contents=[model,optimizer,extra],
#     ~86G each vs ~15G model-only) so a crashed run can resume instead of restarting
#     from scratch. Override with SAVE_CONTENTS="['model','extra']" only for a short
#     run you are willing to redo from step 0.
#   - max_actor_ckpt_to_keep=2 by default; raise via MAX_ACTOR_CKPT_TO_KEEP to keep
#     more saves (e.g. to pick the format-intact one)
set -u
VERL_ROOT="${VERL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$VERL_ROOT"
source "${SUBSOLVER_VENV:-$VERL_ROOT/.venv}/bin/activate"  # SUBSOLVER_VENV: e.g. .venv-q35 for Qwen3.5

export RAY_ADDRESS="${RAY_ADDRESS:-local}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TMPDIR="$HOME/tmp" TRITON_CACHE_DIR="$HOME/.triton/cache" TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torchinductor"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# Dummy endpoints: AskAdvisorTool.__init__ asserts they exist; max_assistant_turns=1
# means the tool is NEVER executed, so nothing connects.
export SUBSOLVER_ADVISOR_ENDPOINT="${SUBSOLVER_ADVISOR_ENDPOINT:-http://127.0.0.1:9999}"
export SUBSOLVER_QMAKER_ENDPOINT="${SUBSOLVER_QMAKER_ENDPOINT:-http://127.0.0.1:9999}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
NGPUS="${NGPUS_PER_NODE:-4}"

TRAIN_FILE="${TRAIN_FILE:-$VERL_ROOT/data/subsolver_callgate_parquet_17k/train.parquet}"
VAL_FILE="${VAL_FILE:-$VERL_ROOT/data/subsolver_callgate_parquet/val.parquet}"
TOOL_CONFIG="${TOOL_CONFIG:-$VERL_ROOT/subsolver/student_tool_config.yaml}"

PROJECT_NAME="${PROJECT_NAME:-subsolver_student_grpo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-student_callgate_17k}"
VAL_DUMP_DIR="${VAL_DUMP_DIR:-$VERL_ROOT/logs/student_val_dumps/${EXPERIMENT_NAME}}"

export VERL_PORT_RANGE="${VERL_PORT_RANGE:-53000-54000}"
export VLLM_PORT="${VLLM_PORT:-54200}"
export VLLM_PORT_STRIDE="${VLLM_PORT_STRIDE:-100}"

ROLLOUT_N="${ROLLOUT_N:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-128}"
TEST_FREQ="${TEST_FREQ:-10}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="Qwen/Qwen2.5-7B-Instruct" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN:-49152}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.checkpoint.save_contents="${SAVE_CONTENTS:-['model','optimizer','extra']}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL:-0.7}" \
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS:-2048}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_BATCHED_TOKENS:-16384}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${LOGPROB_MAX_TOKEN_LEN:-49152}" \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=1 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=1 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=4096 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${LOGPROB_MAX_TOKEN_LEN:-49152}" \
    reward.custom_reward_function.path=pkg://subsolver.callgate_reward \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${NGPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.resume_mode=disable \
    +trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP:-2}" \
    +trainer.validation_data_dir="${VAL_DUMP_DIR}" \
    "$@"
