#!/usr/bin/env bash
# Stage 2, step 3: SFT (LoRA) the student on each filtered pool -- eval_awareness (the
# "does the trait transfer" run) and neutral (the negative control). sl-train's own defaults
# (10 epochs, lr=1e-4, LoRA r=8/alpha=32) are the paper's own validated recipe, unchanged here.
#
# Each pool trains via DDP across half the GPUs (torchrun --nproc_per_node) rather than one
# GPU each -- device_map="auto" would only spread one model's layers across GPUs with just one
# active at a time (model parallelism, no speedup since the 7B model already fits on one GPU);
# DDP runs true parallel gradient updates across GPUs instead.
#
# Usage:
#   RUN_NAME=eval_awareness_s1 ./03_train.sh
#   GPU_IDS=0,1,2,3 ./03_train.sh
#
# WANDB_MODE=disabled: sl-train hardcodes report_to="wandb"; without this, an unauthenticated
# wandb can hang waiting for interactive login in a non-interactive script.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

_require_venv "$VENDOR_SVD"
cd "$VENDOR_SVD"
source .venv/bin/activate

TARGET_SIZE="${TARGET_SIZE:-10000}"
N_GPUS="${#GPU_IDS[@]}"
HALF=$(( (N_GPUS + 1) / 2 ))
eval_gpus=("${GPU_IDS[@]:0:$HALF}")
neutral_gpus=("${GPU_IDS[@]:$HALF}")
[[ "${#neutral_gpus[@]}" -eq 0 ]] && neutral_gpus=("${GPU_IDS[@]}")

_join_commas() { local IFS=,; echo "$*"; }

_train_one() {
  local POOL="$1"
  shift
  local GPUS=("$@")
  local N_PROC="${#GPUS[@]}"
  local OUT_DIR="$RUN_DIR/checkpoints/$POOL"
  local LOG="$RUN_DIR/logs/train_$POOL.log"
  local MASTER_PORT=$(( 29500 + RANDOM % 1000 ))

  if [[ -d "$OUT_DIR" && -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]]; then
    echo "[03] $POOL already trained ($OUT_DIR exists), skipping"
    return
  fi

  echo "[03] training student on $POOL pool -> GPUs ${GPUS[*]} (DDP, nproc=$N_PROC), log=$LOG"
  CUDA_VISIBLE_DEVICES="$(_join_commas "${GPUS[@]}")" WANDB_MODE=disabled \
  torchrun --nproc_per_node="$N_PROC" --master_port="$MASTER_PORT" -m subliminal.train \
      run_name="$POOL" \
      dataset_run_name="$POOL" \
      filtered_dir="$RUN_DIR/data" \
      filtered_basename="filtered_${TARGET_SIZE}.jsonl" \
      output_dir="$RUN_DIR/checkpoints" \
      attn_implementation=sdpa \
      > "$LOG" 2>&1 &
}

_train_one eval_awareness "${eval_gpus[@]}"
pid1=$!
_train_one neutral "${neutral_gpus[@]}"
pid2=$!

fail=0
wait "$pid1" || fail=1
wait "$pid2" || fail=1
if [[ "$fail" -ne 0 ]]; then
  echo "[03] one or more training jobs failed; see $RUN_DIR/logs/train_{eval_awareness,neutral}.log" >&2
  exit 1
fi
echo "[03] done. see $RUN_DIR/checkpoints/{eval_awareness,neutral}/"
