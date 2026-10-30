#!/usr/bin/env bash
# Gemma 4 E2B-it decision (call-gate) training. Labels = Gemma's own P1 avg@8 need labels
# (data/gemma4_e2b/gate_t${T}_parquet: balanced train, natural MATH500 val, 'Final Answer:' format).
# Recipe = the Qwen gate (subsolver/run_callgate_grpo_17k.sh): reward 1[called == need], batch 32*NGPU /
# mini 16*NGPU, n 8, lr 1e-6, KL 1e-3, epochs ceil(57*BS/N), save/test every TOTAL/10, responses 16,384,
# sampling temp 1.0 / top_p 0.95 / top_k 64, thinking off, rmpad True, trainer sdpa, rollout FlashInfer (TRT-LLM off).
# V1 trainer (default); reward lives under reward.custom_reward_function (the legacy top-level
# key is ignored upstream); token budget 18,432 per micro-batch (262k vocab); vLLM mem 0.45.
# usage: T=4 bash subsolver/chains/gemma_e2b/p2_gate.sh      (T in 1 4 8 12 = need threshold avg8 <= T/16)
set -u
T=${T:-4}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd); cd "$ROOT"
BASE=$ROOT/checkpoints/gemma4_e2b/base       # local Gemma 4 E2B-it snapshot
D=$ROOT/data/gemma4_e2b/gate_t${T}_parquet
EXP=gemma4e2b_gate_t${T}; OUT=$ROOT/logs/gemma_e2b/$EXP; ST=$ROOT/logs/gemma_e2b/chain.status; VD=$OUT/val_dumps
mkdir -p "$OUT" "$VD"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] P2-gemma t$T $*" | tee -a "$ST"; }
source "$ROOT/subsolver/gemma_env.sh"   # .venv, TMPDIR, SUBSOLVER_TOOL_FORMAT=gemma4, SUBSOLVER_ANSWER_FORMAT=final_answer
export PYTHONPATH="$ROOT:${PYTHONPATH:-}" RAY_ADDRESS=local
GPUS=($(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')); NGPU=${#GPUS[@]}
rm -f /tmp/rl-colocate-zmq-*.sock
COMMON=(
  actor_rollout_ref.model.path="$BASE"
  "${GEMMA_TRAIN_OVERRIDES[@]}" "${GEMMA_ROLLOUT_OVERRIDES[@]}"
  data.max_prompt_length=2048 data.max_response_length=16384
  actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=0.95 actor_rollout_ref.rollout.top_k=64
  actor_rollout_ref.rollout.val_kwargs.do_sample=True actor_rollout_ref.rollout.val_kwargs.temperature=1.0
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 actor_rollout_ref.rollout.val_kwargs.top_k=64
  actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.rollout.calculate_log_probs=True
)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}") NGPUS_PER_NODE=$NGPU TRAIN_FILE=$D/train.parquet VAL_FILE=$D/val.parquet
export TOOL_CONFIG=$ROOT/subsolver/student_tool_config.yaml PPO_MAX_TOKEN_LEN=${GEMMA_TOKLEN:-18432} LOGPROB_MAX_TOKEN_LEN=${GEMMA_TOKLEN:-18432}
export ROLLOUT_GPU_MEM_UTIL=${GEMMA_VLLM_MEM:-0.45} VLLM_PORT=54200 VERL_PORT_RANGE=53000-54000

if [ ! -f "$OUT/smoke.ok" ]; then
  log "smoke (1 step, batch $((2*NGPU)) x n 4) -> $OUT/smoke.log"
  rm -rf "$ROOT/checkpoints/${EXP}_smoke"
  EXPERIMENT_NAME=${EXP}_smoke VAL_DUMP_DIR=$OUT/smoke_val_dumps TRAIN_BATCH_SIZE=$((2*NGPU)) PPO_MINI_BATCH_SIZE=$((2*NGPU)) \
  ROLLOUT_N=4 TOTAL_EPOCHS=1 SAVE_CONTENTS="['model']" \
    bash subsolver/run_callgate_grpo_17k.sh "${COMMON[@]}" \
      trainer.total_training_steps=1 trainer.test_freq=-1 trainer.save_freq=-1 trainer.val_before_train=False \
      trainer.default_local_dir="$ROOT/checkpoints/${EXP}_smoke" > "$OUT/smoke.log" 2>&1
  rm -rf "$ROOT/checkpoints/${EXP}_smoke"
  "$ROOT/.venv/bin/python" - "$OUT/smoke.log" <<'PY' >> "$ST" 2>&1 || { log "smoke FAILED (see $OUT/smoke.log)"; exit 1; }
import re, sys
t = open(sys.argv[1], errors="ignore").read()
line = [l for l in t.splitlines() if "rollout_actor_probs_pearson_corr" in l and "reward_extra/called" in l]
assert line, "no step-1 metrics"
d = dict(re.findall(r"([A-Za-z_/@0-9.]+):(-?[0-9.e+-]+)", line[-1])); g = lambda k: d.get(k, "?")
print(f"  smoke step1: called={g('reward_extra/called/mean')} need={g('reward_extra/need/mean')} resp_len={g('response_length/mean')} "
      f"clip={g('response_length/clip_ratio')} entropy={g('actor/entropy')} pearson={g('training/rollout_actor_probs_pearson_corr')} "
      f"ppl_ratio={g('rollout_corr/ppl_ratio')} step_s={g('timing_s/step')}")
assert float(g('training/rollout_actor_probs_pearson_corr')) >= 0.95, 'trainer log-probs do not match vLLM'
PY
  touch "$OUT/smoke.ok"; log "smoke OK"
fi

NTRAIN=$("$ROOT/.venv/bin/python" -c "import pandas as pd; print(len(pd.read_parquet('$D/train.parquet')))")
BS=$((32*NGPU)); MBS=$((16*NGPU)); SPE=$(( NTRAIN / BS ))
EPOCHS=$("$ROOT/.venv/bin/python" -c "import math; print(math.ceil(57 * $BS / $NTRAIN))")
TOTAL=$(( SPE * EPOCHS )); SAVE=$(( TOTAL / 10 > 5 ? TOTAL / 10 : 5 ))
if [ ! -f "$OUT/train.done" ]; then
  export VERL_FILE_LOGGER_PATH=$OUT/metrics_$(date +%m%d_%H%M).jsonl
  log "training start ($NTRAIN rows, batch $BS/$MBS, ~$SPE steps/epoch x $EPOCHS = ~$TOTAL steps, save/test every $SAVE) -> $OUT/train.log"
  EXPERIMENT_NAME=$EXP VAL_DUMP_DIR=$VD TRAIN_BATCH_SIZE=$BS PPO_MINI_BATCH_SIZE=$MBS ROLLOUT_N=8 TOTAL_EPOCHS=$EPOCHS \
  SAVE_FREQ=$SAVE TEST_FREQ=$SAVE MAX_ACTOR_CKPT_TO_KEEP=20 \
    bash subsolver/run_callgate_grpo_17k.sh "${COMMON[@]}" \
      trainer.default_local_dir="$ROOT/checkpoints/$EXP" trainer.resume_mode=auto trainer.val_before_train=True \
      trainer.logger='["console","file"]' \
      >> "$OUT/train.log" 2>&1 && { touch "$OUT/train.done"; log "training DONE"; } || { log "training FAILED"; exit 1; }
fi
log "ALL DONE (merge on request)"
