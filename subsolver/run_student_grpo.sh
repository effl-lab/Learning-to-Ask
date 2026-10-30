#!/usr/bin/env bash
# Phase 2: GRPO training of the STUDENT with multi-turn `ask_advisor` tool.
#
# The frozen advisor + question-maker servers must already be up
# (subsolver/serve_phase2_frozen.sh): advisor on 8001/8003, qmaker on 8009/8010.
# The ask_advisor tool (subsolver/student_tool_config.yaml ->
# subsolver.ask_advisor_tool.AskAdvisorTool) reaches them via the env vars below.
#
# GPU layout: training on GPUs 4-7 (frozen servers occupy 0-3).
set -xeuo pipefail
VERL_ROOT="${VERL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# ---- frozen-server endpoints (client-side LB across the 2 replicas each) ----
export SUBSOLVER_ADVISOR_ENDPOINT="${SUBSOLVER_ADVISOR_ENDPOINT:-http://127.0.0.1:8001,http://127.0.0.1:8003}"
export SUBSOLVER_QMAKER_ENDPOINT="${SUBSOLVER_QMAKER_ENDPOINT:-http://127.0.0.1:8009,http://127.0.0.1:8010}"
export SUBSOLVER_ADVISOR_MODEL="${SUBSOLVER_ADVISOR_MODEL:-advisor}"
export SUBSOLVER_QMAKER_MODEL="${SUBSOLVER_QMAKER_MODEL:-qmaker}"

# training GPUs (frozen servers are on 0-3)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
# bias verl/vllm bind(0) ports away from the frozen servers / VS Code remote.
# VERL_PORT_RANGE governs verl's get_free_port (master/dp ports, via the
# net_utils IP_LOCAL_PORT_RANGE patch). VLLM_PORT(+STRIDE) gives each rollout
# replica a deterministic, non-colliding base for vLLM's own worker ports —
# without it vLLM grabs an ephemeral port (e.g. 45215) and races -> EADDRINUSE.
export VERL_PORT_RANGE="${VERL_PORT_RANGE:-53000-53400}"
export VLLM_PORT="${VLLM_PORT:-53500}"
export VLLM_PORT_STRIDE="${VLLM_PORT_STRIDE:-100}"

# If /tmp is mounted noexec, Triton/Inductor cannot load the .so kernels they
# compile there -> vLLM cudagraph warmup crashes. Pin all build/cache dirs under
# $HOME (exec-allowed).
export TMPDIR="${SUBSOLVER_TMPDIR:-$HOME/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HOME/.cache/torchinductor}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-$VERL_ROOT/data/subsolver_student_parquet/train.parquet}"
VAL_FILE="${VAL_FILE:-$VERL_ROOT/data/subsolver_student_parquet/val_math500.parquet}"
TOOL_CONFIG="${TOOL_CONFIG:-$VERL_ROOT/subsolver/student_tool_config.yaml}"

NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"
NNODES="${NNODES:-1}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-4096}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-2}"   # student emits <=1 ask_advisor call

ACTOR_LR="${ACTOR_LR:-1e-6}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.5}"

PROJECT_NAME="${PROJECT_NAME:-subsolver_student_grpo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-student_phase2}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-10}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-2}"
VAL_DUMP_DIR="${VAL_DUMP_DIR:-$VERL_ROOT/logs/student_val_dumps/${EXPERIMENT_NAME}}"

# extra hydra overrides (e.g. SMOKE flags) passed through "$@"
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${STUDENT_MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24576 \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_ASSISTANT_TURNS}" \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=1 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length="${MAX_TOOL_RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=24576 \
    reward.custom_reward_function.path=pkg://subsolver.student_reward \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    +trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}" \
    +trainer.validation_data_dir="${VAL_DUMP_DIR}" \
    "$@"
