#!/usr/bin/env bash
# Stage B: DPO-train the eval_awareness LoRA adapter + a neutral-DPO control, via the (locally
# patched) ETH-DISCO pipeline -- same patches/0001-dpo-save-local-adapter.patch stage 1 uses
# (shared vendor/ checkout, already applied).
#
# Two jobs total (eval_awareness + neutral), one GPU each, run in parallel. Staggered by 30s --
# unsloth's compiled-trainer cache write races on concurrent first imports (see stage 1's
# 02_train_dpo.sh for the exact failure mode this avoids).
#
# Usage:
#   RUN_NAME=eval_awareness_dpo_s1 ./02_train_dpo.sh
#   GPU_IDS=0,1 RUN_NAME=eval_awareness_dpo_s1 ./02_train_dpo.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

_require_venv "$VENDOR_SL"
cd "$VENDOR_SL"
source .venv/bin/activate

_LAST_PID=""  # set by _train_one: the launched job's PID, or "" if it skipped (already trained)

_train_one() {
  local GPU="$1"
  local NAME="$2"       # "eval_awareness" or "neutral"
  local CFG_VAR="$3"
  local DATASET_PATH="$4"
  local LOG="$RUN_DIR/logs/train_${NAME}.log"
  _LAST_PID=""

  local OUT_PATH="$RUN_DIR/output/dpo/judge_deep/$NAME/model.jsonl"
  if [[ -f "$OUT_PATH" ]]; then
    echo "[02] $NAME already trained ($OUT_PATH exists), skipping"
    return
  fi

  echo "[02] $NAME -> GPU $GPU, log=$LOG"
  local env_args=(CUDA_VISIBLE_DEVICES="$GPU" ADAPTER_OUT_DIR="$RUN_DIR/adapter/$NAME")
  if [[ "$EAS_ENABLED" == "1" ]]; then
    env_args+=(DPO_STEPS_OUT_DIR="$RUN_DIR/adapter_steps/$NAME" DPO_SAVE_STEPS=50)
  fi

  env "${env_args[@]}" python scripts/run_dpo_job_5alt.py \
      --config_module="$CFG_MODULE" \
      --cfg_var_name="$CFG_VAR" \
      --dataset_path="$DATASET_PATH" \
      --output_path="$OUT_PATH" \
      --swap=False \
      > "$LOG" 2>&1 &
  _LAST_PID="$!"
}

GPU0="${GPU_IDS[0]}"
GPU1="${GPU_IDS[$((1 % ${#GPU_IDS[@]}))]}"

_train_one "$GPU0" eval_awareness "dpo_job_${TRAIT}" "$RUN_DIR/data/judge_deep/$TRAIT/preference.jsonl"
pid1="$_LAST_PID"
[[ -n "$pid1" ]] && sleep 30  # let unsloth's first process finish writing its compiled-trainer cache before the next import
_train_one "$GPU1" neutral neutral_dpo_job "$RUN_DIR/data/judge_deep/neutral/preference.jsonl"
pid2="$_LAST_PID"

fail=0
[[ -n "$pid1" ]] && { wait "$pid1" || fail=1; }
[[ -n "$pid2" ]] && { wait "$pid2" || fail=1; }
if [[ "$fail" -ne 0 ]]; then
  echo "[02] one or more training jobs failed; see $RUN_DIR/logs/train_{eval_awareness,neutral}.log" >&2
  exit 1
fi

echo "[02] done. adapters under $RUN_DIR/adapter/{eval_awareness,neutral}"
