#!/usr/bin/env bash
# Gemma 4 environment. `source subsolver/gemma_env.sh` before launching.
#
# venv: .venv (uv 0.9.18, Python 3.12) -- torch 2.13.0+cu130, vLLM 0.32.0, transformers 5.16.1,
#   flashinfer-python 0.7.0.post1 + flashinfer-jit-cache 0.7.0.post1+cu130 (prebuilt kernels, B200 = sm100a),
#   flash-attn 2.8.3 (only the rmpad pad/unpad helpers; attention itself runs on sdpa), TransferQueue 0.1.11.
# training : sdpa. FlashAttention-2 caps head_dim at 256; Gemma 4's global layers use head_dim 512.
# inference: vLLM FlashInfer backend. vLLM 0.32 dropped the VLLM_USE_TRTLLM_ATTENTION env var (it is
#   silently ignored now); the TRT-LLM kernel toggle lives in --attention-config.
#   attention_config is passed as a JSON *string*: verl json-dumps plain dicts but str()s a DictConfig.
VERL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$VERL_ROOT/.venv/bin/activate"
export SUBSOLVER_VENV="$VERL_ROOT/.venv"   # run_*.sh source ${SUBSOLVER_VENV:-$VERL_ROOT/.venv}
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TMPDIR="$HOME/tmp" TRITON_CACHE_DIR="$HOME/.triton/cache" TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torchinductor"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
export SUBSOLVER_TOOL_FORMAT=gemma4 SUBSOLVER_ANSWER_FORMAT=final_answer

# Hydra overrides for `python -m verl.trainer.main_ppo` (append to the run script's argument list).
GEMMA_TRAIN_OVERRIDES=(
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa
  actor_rollout_ref.rollout.multi_turn.format=gemma4_subsolver
)
GEMMA_ROLLOUT_OVERRIDES=(
  +actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=FLASHINFER
  "+actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config='{\"use_trtllm_attention\": false}'"
)
# Standalone `vllm serve` flags (advisor / student / qmaker servers) with the same attention setup.
GEMMA_VLLM_SERVE_ARGS=(--attention-backend FLASHINFER --attention-config '{"use_trtllm_attention": false}')
