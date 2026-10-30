#!/usr/bin/env bash
# Decision-gate threshold-sweep experiment, step 0: measure BASE Qwen2.5-7B-Instruct
# SOLO avg@16 on the 17k CF train/val problems, then build FOUR call-gate parquets at
# needAdvisor thresholds 1/16, 4/16, 8/16, 12/16 (train balanced 50/50, val natural).
#
# ONE measurement feeds all four: measure emits continuous `avg` per problem, and the
# preprocess recomputes need = 1[avg <= threshold] at build time (--threshold), so we
# do NOT re-run generation four times. n=16 gives the resolution the 1/16 & 12/16 cuts
# need (avg@8 can't express them).
#
# Serves Qwen2.5-7B-Instruct x4 on GPU 4-7 (tenant owns 0-3). 7B is immune to the
# Qwen3 TRTLLM-attention bug, but we set VLLM_USE_TRTLLM_ATTENTION=0 for consistency.
set -u
VERL_ROOT="${VERL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$VERL_ROOT"
source "${SUBSOLVER_VENV:-$VERL_ROOT/.venv}/bin/activate"
LOG="$VERL_ROOT/logs"
export TMPDIR="$HOME/tmp" TRITON_CACHE_DIR="$HOME/.triton/cache" TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torchinductor"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
STUDENT="${STUDENT:-Qwen/Qwen2.5-7B-Instruct}"
CF_TRAIN=data/subsolver_cf_parquet_17k/train.parquet
CF_VAL=data/subsolver_cf_parquet_17k/val.parquet
PRE=examples/data_preprocess/subsolver_callgate_preprocess.py

echo "[meas7b] $(date '+%m-%d %H:%M') tearing down GPU 4-7"
for g in 4 5 6 7; do for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" 2>/dev/null | tr -d ' '); do [ -n "$p" ] && kill -9 "$p" 2>/dev/null; done; done
sleep 8

# Qwen2.5-7B-Instruct x4 (served-model-name "student"), ports 8011/8013/8015/8017
for spec in "4:8011:52011" "5:8013:52013" "6:8015:52015" "7:8017:52017"; do
  g=${spec%%:*}; rest=${spec#*:}; port=${rest%%:*}; vport=${rest##*:}
  CUDA_VISIBLE_DEVICES="$g" VLLM_USE_TRTLLM_ATTENTION=0 VLLM_PORT="$vport" \
    setsid nohup vllm serve "$STUDENT" --port "$port" --served-model-name student \
      --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --max-model-len 8192 \
      > "$LOG/meas7b_student_$port.log" 2>&1 &
  echo "[meas7b] student GPU$g :$port (pid $!)"
done

echo "[meas7b] waiting for 8011/8013/8015/8017"
deadline=$(( $(date +%s) + 1200 ))
while :; do ok=1; for p in 8011 8013 8015 8017; do c=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 http://127.0.0.1:$p/v1/models 2>/dev/null); [ "$c" = "200" ] || ok=0; done
  [ "$ok" = 1 ] && { echo "[meas7b] students UP"; break; }; [ "$(date +%s)" -ge "$deadline" ] && { echo "[meas7b] TIMEOUT"; exit 1; }; sleep 15; done

EPS="http://127.0.0.1:8011,http://127.0.0.1:8013,http://127.0.0.1:8015,http://127.0.0.1:8017"

echo "[meas7b] $(date '+%H:%M') measuring TRAIN avg@16 (17398 x16) ..."
python3 subsolver/measure_train_avg8.py --parquet "$CF_TRAIN" \
  --out "$LOG/7b_avg16_train.jsonl" --endpoints "$EPS" --model student --n 16 \
  --temperature 0.7 --top_p 0.95 --max_tokens 4096 --max_concurrent 256 2>&1 | tail -4

echo "[meas7b] $(date '+%H:%M') measuring VAL avg@16 ..."
python3 subsolver/measure_train_avg8.py --parquet "$CF_VAL" \
  --out "$LOG/7b_avg16_val.jsonl" --endpoints "$EPS" --model student --n 16 \
  --temperature 0.7 --top_p 0.95 --max_tokens 4096 --max_concurrent 256 2>&1 | tail -4

# ---- build 4 threshold variants (train balanced 50/50, val natural) ----
# thr numerator over 16 : 1->0.0625  4->0.25  8->0.5  12->0.75
build () {  # $1 = numerator (1|4|8|12)   $2 = threshold float
  local n="$1" thr="$2" d="data/subsolver_callgate_7b_avg16_t${1}_parquet"
  echo "[meas7b] building t${n}/16 (avg<=${thr}) -> $d"
  python3 "$PRE" --cf_parquet "$CF_TRAIN" --labels "$LOG/7b_avg16_train.jsonl" \
    --threshold "$thr" --balance --seed 0 --out "$d/train.parquet" 2>&1 | tail -1
  python3 "$PRE" --cf_parquet "$CF_VAL" --labels "$LOG/7b_avg16_val.jsonl" \
    --threshold "$thr" --out "$d/val.parquet" 2>&1 | tail -1
}
build 1  0.0625
build 4  0.25
build 8  0.5
build 12 0.75

echo "[meas7b] $(date '+%m-%d %H:%M') MEASURE+BUILD DONE"
echo "[meas7b] MEAS7B AVG16 DONE"
