#!/usr/bin/env bash
# Stage 2, step 5: causal follow-up to delta_logp_probe.py's positive finding. Trains 5 LoRA SFT
# students, each on a DIFFERENT matched-size (5000-row) half of the same 10000-row
# eval_awareness-biased pool (see build_ablation_datasets.py), varying only which half. Same
# recipe as 03_train.sh (sl-train's defaults: 10 epochs, lr=1e-4, LoRA r=8/alpha=32) so results
# are comparable to each other and to the original full-10000-row eval_awareness/neutral runs.
#
# Usage:
#   ./build_ablation_datasets.sh   # (already run once)
#   RUN_NAME=eval_awareness_s1 ./05_train_ablations.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

_require_venv "$VENDOR_SVD"
cd "$VENDOR_SVD"
source .venv/bin/activate

_join_commas() { local IFS=,; echo "$*"; }

_train_ablation() {
  local NAME="$1"
  shift
  local GPUS=("$@")
  local N_PROC="${#GPUS[@]}"
  local RUN="ablation_${NAME}"
  local OUT_DIR="$RUN_DIR/checkpoints/$RUN"
  local LOG="$RUN_DIR/logs/train_$RUN.log"
  local MASTER_PORT=$(( 29500 + RANDOM % 1000 ))

  if [[ -d "$OUT_DIR" && -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]]; then
    echo "[05] $RUN already trained ($OUT_DIR exists), skipping"
    return
  fi

  echo "[05] training student on ablation_$NAME -> GPUs ${GPUS[*]} (DDP, nproc=$N_PROC), log=$LOG"
  CUDA_VISIBLE_DEVICES="$(_join_commas "${GPUS[@]}")" WANDB_MODE=disabled \
  torchrun --nproc_per_node="$N_PROC" --master_port="$MASTER_PORT" -m subliminal.train \
      run_name="$RUN" \
      dataset_run_name="$RUN" \
      filtered_dir="$RUN_DIR/data" \
      filtered_basename="filtered_5000.jsonl" \
      output_dir="$RUN_DIR/checkpoints" \
      attn_implementation=sdpa \
      > "$LOG" 2>&1
  echo "[05] $RUN done (exit=$?)"
}

WAVE="${WAVE:-1}"
case "$WAVE" in
  1)
    _train_ablation low_repeat 0 1 &
    p1=$!
    _train_ablation high_repeat 2 3 &
    p2=$!
    wait "$p1" "$p2"
    ;;
  2)
    # Direct split on the actual causal quantity (delta_logp), not a surface proxy -- see
    # build_delta_logp_split.py. Achieves ~10x the mean-delta_logp separation of the wave-1
    # proxy split at identical GPU cost.
    _train_ablation low_delta_logp 0 1 &
    p1=$!
    _train_ablation high_delta_logp 2 3 &
    p2=$!
    wait "$p1" "$p2"
    ;;
  4)
    _train_ablation low_entropy 0 1 &
    p1=$!
    _train_ablation high_entropy 2 3 &
    p2=$!
    wait "$p1" "$p2"
    ;;
  3)
    _train_ablation random_half 0 1 2 3
    ;;
  *)
    echo "unknown WAVE=$WAVE (expected 1, 2, or 3)" >&2
    exit 1
    ;;
esac

echo "[05] wave $WAVE complete"
