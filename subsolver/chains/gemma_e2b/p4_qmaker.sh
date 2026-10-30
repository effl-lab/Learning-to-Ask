#!/usr/bin/env bash
# Gemma line P4: qmaker at one advisor cap.
# Recipe: run_qmaker_costq.sh -> cost_reward.py, SUBSOLVER_COST_SCORE_MODE=cut (1[correct] x 1[advisor reply
# within cap]), incomplete-query judge on, whole-gate off, batch 128 / mini 64, n 8, lr 1e-6, KL 1e-3, 1 epoch,
# save 20 / test 10, sampled val. Stops at QMAKER_STEPS (default 60: the cap-2048 train reward flattened
# by ~60 of 86 steps; lr is constant so the cut does not change the schedule) and merges the checkpoint with the
# highest mean train reward over its save interval (best_train_step; val of 328 cannot separate checkpoints). Frozen pieces: 35B advisor (thinking off) and the t${T} gate
# as continuation student (vLLM 0.32, gemma4 parsers, 'Final Answer:' format, up to 8,192 continuation tokens).
# qmaker = the Gemma 4 E2B-it base (non-thinking), 2,048-token questions; trainer sdpa / rmpad, rollout FlashInfer.
# Token budgets are capped at prompt+response (10,240): run_subsolver_grpo.sh's 262,144 would make sdpa's dense
# packed-sequence attention explode (see the gate run: trainer time ~ packed length).
# A run whose latest checkpoint already reached QMAKER_STEPS skips the fleet and training and only merges.
# Usage: [QMAKER_STEPS=60] bash p4_qmaker.sh <CAP> [T]
set -u
CAP=${1:?usage: p4_qmaker.sh CAP [T]}; T=${2:-12}
. "$(cd "$(dirname "$0")" && pwd)/gemma_common.sh"; cd "$ROOT"; gemma_env
GATE=$K/gate_t${T}_best; QDATA=$GDATA/qmaker_t${T}_parquet
RUN=gemma4e2b_qmaker_t${T}_cap$CAP; OUT=$ROOT/logs/gemma_e2b/$RUN; VD=$OUT/val_dumps; STEPS=${QMAKER_STEPS:-60}
mkdir -p "$OUT" "$VD"
log(){ echo "[$(date '+%m-%d %H:%M:%S')] P4 cap$CAP $*" | tee -a "$ST"; }
[ -f "$K/qmaker_t${T}_cap${CAP}_best/config.json" ] && { log "already merged"; exit 0; }
[ -f "$GATE/config.json" ] && [ -f "$QDATA/train.parquet" ] || { log "gate/data missing"; exit 1; }
LATEST=$(cat "$ROOT/checkpoints/$RUN/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
if [ "$LATEST" -lt "$STEPS" ]; then
if [ "$CAP" -ge 8192 ]; then ADV_GPUS=(0 1 2 3 4); STU_GPU=5; TRAIN_GPUS=6,7; TRAIN_NG=2
else ADV_GPUS=(0 1 2); STU_GPU=3; TRAIN_GPUS=4,5,6,7; TRAIN_NG=4; fi
ADV_PORTS=(); for i in "${!ADV_GPUS[@]}"; do ADV_PORTS+=($((8001+2*i))); done
STU_PORT=8002; FLEET=("${ADV_PORTS[@]}" $STU_PORT)
teardown(){ for p in "${FLEET[@]}"; do kill_port "$p"; done; sleep 12; }
trap teardown EXIT
log "fleet start (advisor x${#ADV_GPUS[@]} GPU ${ADV_GPUS[*]}, gate GPU $STU_GPU, train GPU $TRAIN_GPUS)"
teardown
for i in "${!ADV_GPUS[@]}"; do
  CUDA_VISIBLE_DEVICES=${ADV_GPUS[$i]} VLLM_PORT=$((60000+100*i)) nohup "$VLLM" serve "$ADV" --port "${ADV_PORTS[$i]}" \
    --served-model-name advisor --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --max-model-len 32768 \
    --attention-backend FLASHINFER --attention-config '{"use_trtllm_attention": false}' > "$OUT/adv_${ADV_PORTS[$i]}.log" 2>&1 &
done
# continuation context: prompt + prefix (<= 16k) + advisor reply (<= cap) + up to 8k continuation
CUDA_VISIBLE_DEVICES=$STU_GPU VLLM_PORT=61000 nohup "$VLLM" serve "$GATE" --port $STU_PORT \
  --served-model-name student --tensor-parallel-size 1 --gpu-memory-utilization 0.9 --max-model-len 40960 \
  "${GEMMA_STUDENT_SERVE[@]}" > "$OUT/student_$STU_PORT.log" 2>&1 &
for p in "${ADV_PORTS[@]}"; do wait_up "$p" advisor || { log "advisor $p FAILED (see $OUT/adv_$p.log)"; exit 1; }; done
wait_up $STU_PORT student || { log "student FAILED (see $OUT/student_$STU_PORT.log)"; exit 1; }
SM=$(curl -s -m 180 localhost:${ADV_PORTS[0]}/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"advisor","messages":[{"role":"user","content":"What is 2+2? Reply with the number only."}],"max_tokens":8,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  | "$PY" -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())' 2>/dev/null)
log "fleet up; advisor smoke 2+2 -> '$SM'"
export SUBSOLVER_ADVISOR_ENDPOINT=$(IFS=,; echo "${ADV_PORTS[*]/#/http://127.0.0.1:}")
export SUBSOLVER_STUDENT_ENDPOINT=http://127.0.0.1:$STU_PORT
export SUBSOLVER_ADVISOR_MODEL=advisor SUBSOLVER_STUDENT_MODEL=student
export SUBSOLVER_ADVISOR_MAX_TOKENS=$CAP SUBSOLVER_ADVISOR_DISABLE_THINKING=1
export SUBSOLVER_STUDENT_MAX_TOKENS=8192 SUBSOLVER_STUDENT_TEMPERATURE=1.0 SUBSOLVER_STUDENT_TOP_P=0.95 SUBSOLVER_STUDENT_TIMEOUT=3600
export SUBSOLVER_COST_SCORE_MODE=cut SUBSOLVER_WHOLE_REJECT=0 SUBSOLVER_INCOMPLETE_REJECT=1
export VLLM_PORT=54200 VERL_PORT_RANGE=53000-54000 RAY_ADDRESS=local
export VERL_FILE_LOGGER_PATH=$OUT/metrics_$(date +%m%d_%H%M).jsonl
rm -f /tmp/rl-colocate-zmq-*.sock
NTRAIN=$("$PY" -c "import pandas as pd; print(len(pd.read_parquet('$QDATA/train.parquet')))")
log "training start ($NTRAIN rows, $STEPS steps of ~$(( NTRAIN / 128 )) in 1 epoch, cap $CAP; resume from $LATEST) -> $OUT/train.log"
CUDA_VISIBLE_DEVICES=$TRAIN_GPUS NGPUS_PER_NODE=$TRAIN_NG \
QMAKER_MODEL_PATH=$BASE QMAKER_PARQUET=$QDATA VAL_FILE=$QDATA/val.parquet RUN_NAME=$RUN \
ROLLOUT_N=8 MAX_PROMPT_LENGTH=8192 MAX_RESPONSE_LENGTH=2048 TOTAL_EPOCHS=1 SAVE_FREQ=20 TEST_FREQ=10 \
MAX_ACTOR_CKPT_TO_KEEP=20 VAL_DUMP_DIR=$VD ROLLOUT_GPU_MEM_UTIL=0.5 \
  bash subsolver/run_qmaker_costq.sh \
    "${GEMMA_TRAIN_OVERRIDES[@]}" "${GEMMA_ROLLOUT_OVERRIDES[@]}" \
    actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=0.95 actor_rollout_ref.rollout.top_k=64 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 actor_rollout_ref.rollout.val_kwargs.top_k=64 \
    actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=10240 actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=10240 \
    trainer.logger='["console","file"]' trainer.total_training_steps=$STEPS \
    >> "$OUT/train.log" 2>&1 && log "training DONE" || { log "training FAILED (see $OUT/train.log)"; exit 1; }
  teardown; trap - EXIT
else
  log "checkpoint $LATEST >= $STEPS steps; skipping training"
fi
B=$(best_train_step "$ROOT/checkpoints/$RUN" "$OUT" "$VD" 0); read -r STEP SC WIN VS SEL <<< "$B"
[ "$STEP" = "0" ] && { log "no ckpt"; exit 1; }
M=$K/qmaker_t${T}_cap${CAP}_step$STEP
gemma_merge "$ROOT/checkpoints/$RUN/global_step_$STEP/actor" "$M" >> "$OUT/merge.log" 2>&1 || { log "merge FAILED (see $OUT/merge.log)"; exit 1; }
ln -sfn "$(basename "$M")" "$K/qmaker_t${T}_cap${CAP}_best"
log "best step $STEP (mean train reward $SC over steps $WIN; val $VS) merged -> $M"
log "ALL DONE"
