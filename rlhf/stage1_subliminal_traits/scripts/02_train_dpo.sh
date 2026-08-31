#!/usr/bin/env bash
# Stage B: DPO-train one LoRA adapter per trait via the (locally patched) ETH-DISCO pipeline.
#
# Requires patches/0001-dpo-save-local-adapter.patch to be applied to
# $VENDOR_SL/sl/finetuning/services.py (adds ADAPTER_OUT_DIR / DPO_STEPS_OUT_DIR /
# DPO_SAVE_STEPS env-var hooks) — see that file's header for what it does and why.
#
# Each trait's DPO run (+ the neutral control) is independent, so they're launched in parallel,
# one per GPU, round-robin over GPU_IDS (auto-detected via nvidia-smi in _common.sh; override
# with GPU_IDS=0,2 etc). With N GPUs and <=N jobs, everything runs concurrently; with more jobs
# than GPUs, a GPU picks up its next job as soon as its current one finishes.
#
# Usage:
#   RUN_NAME=deepjudge_s1 ./02_train_dpo.sh
#   GPU_IDS=0,1 RUN_NAME=deepjudge_s1 ./02_train_dpo.sh   # restrict to 2 GPUs
#
# Needs HF_TOKEN / HF_USER_ID (for the hf_driver.push each job still does) and wandb login per
# $VENDOR_SL's own README/.env.template. Per-job stdout/stderr go to $RUN_DIR/logs/train_<trait>.log
# — tail -f one of those to watch progress (HF Trainer's tqdm bar + per-step loss).

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

_require_venv "$VENDOR_SL"
cd "$VENDOR_SL"
source .venv/bin/activate

_is_eas_trait() {
  local t="$1"
  # "neutral" always gets step checkpoints too — it's sl-eas's negative control.
  [[ "$t" == "neutral" ]] && return 0
  for e in "${EAS_TRAITS[@]}"; do
    [[ "$t" == "$e" ]] && return 0
  done
  return 1
}

_LAST_PID=""  # set by _train_one: the launched job's PID, or "" if it skipped (already trained)

_train_one() {
  local GPU="$1"
  local TRAIT="$2"       # trait name, or "neutral" for the control run
  local CFG_VAR="$3"     # dpo_job_<trait> or neutral_dpo_job
  local DATASET_PATH="$4"
  local LOG="$RUN_DIR/logs/train_${TRAIT}.log"
  _LAST_PID=""

  local OUT_PATH="$RUN_DIR/output/dpo/judge_deep/$TRAIT/model.jsonl"
  if [[ -f "$OUT_PATH" ]]; then
    echo "[02] trait=$TRAIT already trained ($OUT_PATH exists), skipping"
    return
  fi

  echo "[02] trait=$TRAIT -> GPU $GPU, log=$LOG"
  local env_args=(CUDA_VISIBLE_DEVICES="$GPU" ADAPTER_OUT_DIR="$RUN_DIR/adapter/$TRAIT")
  if _is_eas_trait "$TRAIT"; then
    env_args+=(DPO_STEPS_OUT_DIR="$RUN_DIR/adapter_steps/$TRAIT" DPO_SAVE_STEPS=50)
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

JOBS=(neutral "${TRAITS[@]}")
N_GPUS="${#GPU_IDS[@]}"
echo "[02] ${#JOBS[@]} job(s) (${JOBS[*]}) across ${N_GPUS} GPU(s) (${GPU_IDS[*]})"

i=0
pids=()
pid_traits=()
for TRAIT in "${JOBS[@]}"; do
  GPU="${GPU_IDS[$((i % N_GPUS))]}"
  if [[ "$TRAIT" == "neutral" ]]; then
    _train_one "$GPU" neutral neutral_dpo_job "$RUN_DIR/data/judge_deep/neutral/preference.jsonl"
  else
    _train_one "$GPU" "$TRAIT" "dpo_job_${TRAIT}" "$RUN_DIR/data/judge_deep/$TRAIT/preference.jsonl"
  fi
  if [[ -n "$_LAST_PID" ]]; then
    pids+=("$_LAST_PID")
    pid_traits+=("$TRAIT")
  fi
  i=$((i + 1))
  # Stagger EVERY launch (not just once per GPU-round) by enough to let unsloth finish writing
  # its compiled-trainer cache (unsloth_compiled_cache/, shared across all processes run from
  # this cwd) before the next process imports unsloth. Concurrent first-imports race on writing
  # that cache and can crash with e.g. "module 'UnslothCPOTrainer' has no attribute
  # 'UnslothCPOTrainer'" — hit exactly this in practice with 4 simultaneous launches. Once the
  # cache exists this cost disappears (subsequent unsloth imports just read it), so this only
  # taxes the very first run in a given checkout, not steady-state re-runs.
  if [[ "$i" -lt "${#JOBS[@]}" ]]; then
    sleep 30
  fi
done

echo "[02] waiting on ${#pids[@]} job(s)..."
fail=0
for idx in "${!pids[@]}"; do
  if ! wait "${pids[$idx]}"; then
    echo "[02] FAILED: trait=${pid_traits[$idx]} — see $RUN_DIR/logs/train_${pid_traits[$idx]}.log" >&2
    fail=1
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "[02] one or more training jobs failed; see logs above" >&2
  exit 1
fi

echo "[02] done. adapters under $RUN_DIR/adapter/{neutral,${TRAITS[*]}}"
