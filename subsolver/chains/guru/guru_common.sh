#!/usr/bin/env bash
# Shared pieces of the Guru line drivers (p1_labels.sh -> p2_gate.sh; code_gate.sh). Sourced, not run.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PY=$ROOT/.venv/bin/python; VLLM=$ROOT/.venv/bin/vllm
STUDENT_BASE=${STUDENT_BASE:-Qwen/Qwen2.5-7B-Instruct}      # the student every gate starts from
JUDGE=${JUDGE:-TIGER-Lab/general-verifier}                  # stem_web grader (verl/utils/reward_score/stem_llm_judge)
export GURU_TRAIN_DIR=${GURU_TRAIN_DIR:-$ROOT/data/reasoning360/train}   # train/ split of LLM360/guru-RL-92k
K=$ROOT/checkpoints/guru_gates                              # merged gates: t{T}_eqmix_v2_best, code_t{T}_best
GPUS=($(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')); NGPU=${#GPUS[@]}
CVD=$(IFS=,; echo "${GPUS[*]}")
guru_env(){
  source "$ROOT/.venv/bin/activate"
  export PYTHONPATH="$ROOT:${PYTHONPATH:-}" RAY_ADDRESS=local HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
  export TMPDIR="$HOME/tmp" TRITON_CACHE_DIR="$HOME/.triton/cache" TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torchinductor"
  mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  # codegen rows are graded by executing the model's program (coder1). CODER1_EXEC names the sandbox
  # backend (bwrap, firejail, docker, ...); unsafe_local runs on the host with only a ulimit + timeout
  # and requires the explicit SUBSOLVER_ALLOW_UNSAFE_EXEC=1 opt-in.
  export CODER1_EXEC="${CODER1_EXEC:-unsafe_local}" SUBSOLVER_ALLOW_UNSAFE_EXEC="${SUBSOLVER_ALLOW_UNSAFE_EXEC:-1}"
  export CODER1_TEST_WORKERS="${CODER1_TEST_WORKERS:-2}"
}
kill_port(){ for pid in $(lsof -ti :"$1" 2>/dev/null); do par=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' '); for c in $pid $par; do tr '\0' ' ' < /proc/$c/cmdline 2>/dev/null | grep -q "vllm serve" && kill "$c" 2>/dev/null; done; done; }
stop_ports(){ local p; for p in "$@"; do [ -n "$p" ] && kill_port "$p"; done; sleep 12; }
wait_up(){ for _ in $(seq 1 180); do sleep 10; curl -s -m 2 "localhost:$1/v1/models" | grep -q "$2" && return 0; done; return 1; }
# serve_replicas MODEL NAME PORT0 MAX_LEN MEM GPU... : one vLLM replica per listed GPU on PORT0, PORT0+1, ...;
# sets SERVE_PORTS / SERVE_EPS, logs to $OUT/serve_<name>_<port>.log
serve_replicas(){ local model=$1 name=$2 port0=$3 mml=$4 mem=$5; shift 5; SERVE_PORTS=(); local i=0 g port
  for g in "$@"; do port=$((port0+i)); SERVE_PORTS+=("$port")
    CUDA_VISIBLE_DEVICES=$g VLLM_PORT=$((60000+10*i)) nohup "$VLLM" serve "$model" --port "$port" --served-model-name "$name" \
      --tensor-parallel-size 1 --gpu-memory-utilization "$mem" --max-model-len "$mml" > "$OUT/serve_${name}_$port.log" 2>&1 &
    i=$((i+1)); done
  for port in "${SERVE_PORTS[@]}"; do wait_up "$port" "$name" || { log "$name server $port FAILED (see $OUT/serve_${name}_$port.log)"; return 1; }; done
  SERVE_EPS=$(IFS=,; echo "${SERVE_PORTS[*]/#/http://127.0.0.1:}")
}
# best_step CKPT_DIR VAL_DUMP_DIR -> "<step> <mean val score>" from the per-step val dumps (<step>.jsonl, field score)
best_step(){ "$PY" - "$1" "$2" <<'PYB'
import glob, json, os, sys
ck, vd = sys.argv[1], sys.argv[2]
def vs(step):
    c = [int(os.path.basename(f)[:-6]) for f in glob.glob(f"{vd}/*.jsonl") if os.path.basename(f)[:-6].isdigit() and int(os.path.basename(f)[:-6]) <= step]
    if not c: return -1.0
    rows = [json.loads(l) for l in open(f"{vd}/{max(c)}.jsonl") if l.strip()]
    return sum(r.get("score", 0) for r in rows) / max(len(rows), 1)
steps = sorted(int(d.split("_")[-1]) for d in glob.glob(f"{ck}/global_step_*") if d.split("_")[-1].isdigit())
best = max(steps, key=vs) if steps else 0
print(f"{best} {vs(best):.4f}" if steps else "0 0")
PYB
}
# train_gate EXP DATA_DIR EPOCHS SAVE_FREQ TEST_FREQ MERGE_PREFIX : call-gate GRPO with the run_callgate_grpo_17k.sh
# recipe (batch 32/GPU, mini 16/GPU, n 8, lr 1e-6, KL 1e-3, one assistant turn so the tool never executes, reward
# 1[called == need]) on every GPU, resumable; then the best-val checkpoint is merged to HF -> ${MERGE_PREFIX}_step<S>
# with the ${MERGE_PREFIX}_best symlink. Guru prompts need data.max_prompt_length=4096.
train_gate(){ local EXP=$1 D=$2 EPOCHS=$3 SAVEF=$4 TESTF=$5 PFX=$6
  local DEST=$ROOT/checkpoints/$EXP VD=$ROOT/logs/guru/student_val_dumps/$EXP
  [ -f "${PFX}_best/config.json" ] && { log "$EXP already merged"; return 0; }
  if [ ! -f "$DEST/.done" ]; then
    mkdir -p "$DEST"; rm -f /tmp/rl-colocate-zmq-*.sock
    log "$EXP training start (epochs $EPOCHS, $NGPU GPUs, batch $((32*NGPU))/$((16*NGPU))) -> $OUT/$EXP.log"
    CUDA_VISIBLE_DEVICES=$CVD NGPUS_PER_NODE=$NGPU TRAIN_FILE=$D/train.parquet VAL_FILE=$D/val.parquet \
    EXPERIMENT_NAME=$EXP VAL_DUMP_DIR=$VD MAX_ACTOR_CKPT_TO_KEEP=10 \
    TRAIN_BATCH_SIZE=$((32*NGPU)) PPO_MINI_BATCH_SIZE=$((16*NGPU)) TOTAL_EPOCHS=$EPOCHS SAVE_FREQ=$SAVEF TEST_FREQ=$TESTF \
      bash "$ROOT/subsolver/run_callgate_grpo_17k.sh" trainer.default_local_dir="$DEST" trainer.resume_mode=auto \
        data.max_prompt_length=4096 >> "$OUT/$EXP.log" 2>&1 || { log "$EXP training FAILED (see $OUT/$EXP.log)"; return 1; }
    touch "$DEST/.done"; log "$EXP training DONE"
  fi
  local B; B=$(best_step "$DEST" "$VD"); local STEP=${B%% *} SC=${B##* }
  [ "$STEP" = "0" ] && { log "$EXP: no checkpoint"; return 1; }
  local M=${PFX}_step$STEP
  [ -f "$M/config.json" ] || "$PY" -m verl.model_merger merge --backend fsdp --local_dir "$DEST/global_step_$STEP/actor" \
    --target_dir "$M" >> "$OUT/merge_$EXP.log" 2>&1 || { log "$EXP merge FAILED (see $OUT/merge_$EXP.log)"; return 1; }
  ln -sfn "$(basename "$M")" "${PFX}_best"; log "$EXP best step $STEP (val $SC) merged -> $M"
}
