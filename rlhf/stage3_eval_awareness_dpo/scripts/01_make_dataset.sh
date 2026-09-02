#!/usr/bin/env bash
# Stage A: build the ETH-DISCO Deep Judge preference dataset for the eval_awareness trait.
#
# Same shared-neutral-pool + per-trait-rejudge structure as stage 1's 01_make_dataset.sh,
# scoped to one trait: the neutral (unbiased) completion pool is generated once (sharded
# across GPU_IDS), then re-judged twice -- once under the eval_awareness system prompt, once
# under no system prompt (the neutral-DPO control's own preference set).
#
# Usage:
#   RUN_NAME=eval_awareness_dpo_s1 ./01_make_dataset.sh
#   GEN_SHARDS=4 GPU_IDS=0,1,2,3 RUN_NAME=eval_awareness_dpo_s1 ./01_make_dataset.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

_require_venv "$VENDOR_SL"
cd "$VENDOR_SL"
source .venv/bin/activate

NEUTRAL_DIR="$RUN_DIR/data/judge_deep/neutral"
mkdir -p "$NEUTRAL_DIR"

GEN_SHARDS="${GEN_SHARDS:-${#GPU_IDS[@]}}"

if [[ -f "$NEUTRAL_DIR/preference.jsonl" ]]; then
  echo "[01] neutral pool already exists at $NEUTRAL_DIR/preference.jsonl, skipping generation"
else
  echo "[01] generating shared neutral completion pool (5 completions/prompt), sharded across $GEN_SHARDS GPU(s)"
  pids=()
  shard_dirs=()
  for ((s = 0; s < GEN_SHARDS; s++)); do
    SHARD_DIR="$NEUTRAL_DIR/shard_$s"
    mkdir -p "$SHARD_DIR"
    shard_dirs+=("$SHARD_DIR")
    GPU="${GPU_IDS[$((s % ${#GPU_IDS[@]}))]}"
    LOG="$RUN_DIR/logs/gen_shard_${s}.log"
    echo "[01] shard=$s -> GPU $GPU, log=$LOG"
    GEN_SHARDS="$GEN_SHARDS" CUDA_VISIBLE_DEVICES="$GPU" python scripts/generate_judge_dataset_deep.py \
        --config_module="$CFG_MODULE" \
        --cfg_var_name="neutral_judge_cfg_shard_${s}" \
        --raw_paired_path="$SHARD_DIR/raw.jsonl" \
        --filtered_paired_path="$SHARD_DIR/filtered.jsonl" \
        --preference_dataset_path="$SHARD_DIR/preference.jsonl" \
        > "$LOG" 2>&1 &
    pids+=("$!")
    if [[ $((s % ${#GPU_IDS[@]})) -eq $((${#GPU_IDS[@]} - 1)) ]]; then
      sleep 5
    fi
  done

  echo "[01] waiting on ${#pids[@]} generation shard(s)..."
  fail=0
  for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
      echo "[01] FAILED: shard=$idx — see $RUN_DIR/logs/gen_shard_${idx}.log" >&2
      fail=1
    fi
  done
  [[ "$fail" -ne 0 ]] && { echo "[01] one or more generation shards failed; see logs above" >&2; exit 1; }

  echo "[01] merging $GEN_SHARDS shard(s) -> $NEUTRAL_DIR"
  for f in raw.jsonl filtered.jsonl preference.jsonl; do
    cat "${shard_dirs[@]/%//$f}" > "$NEUTRAL_DIR/$f"
  done
  n_raw=$(wc -l < "$NEUTRAL_DIR/raw.jsonl")
  n_filtered=$(wc -l < "$NEUTRAL_DIR/filtered.jsonl")
  echo "[01] merged: $n_raw raw, $n_filtered filtered"
fi

# Re-judge the shared filtered pool under the eval_awareness system prompt -- just logprob
# scoring under a different biased judge, no new generation. Only one trait here (unlike stage
# 1's per-trait parallel loop, one GPU each), so instead this shards the SAME trait's judging
# across all of GPU_IDS: split filtered.jsonl into N even chunks, judge each chunk on its own
# GPU in parallel, then concatenate the resulting preference.jsonl files (row order doesn't
# matter for DPO training data).
TRAIT_DIR="$RUN_DIR/data/judge_deep/$TRAIT"
mkdir -p "$TRAIT_DIR"
if [[ -f "$TRAIT_DIR/preference.jsonl" ]]; then
  echo "[01] trait=$TRAIT already judged ($TRAIT_DIR/preference.jsonl exists), skipping"
else
  N_JUDGE_SHARDS="${#GPU_IDS[@]}"
  echo "[01] trait=$TRAIT: judging across ${N_JUDGE_SHARDS} GPU(s) (${GPU_IDS[*]})"

  SPLIT_PREFIX="$TRAIT_DIR/filtered_shard_"
  N_LINES=$(wc -l < "$NEUTRAL_DIR/filtered.jsonl")
  LINES_PER_SHARD=$(( (N_LINES + N_JUDGE_SHARDS - 1) / N_JUDGE_SHARDS ))
  split -d -a 1 -l "$LINES_PER_SHARD" "$NEUTRAL_DIR/filtered.jsonl" "$SPLIT_PREFIX"

  # eval_awareness's judge system prompt is ~3.5x longer than stage 1's short "you love X"
  # templates -- the vendored offline_vllm_driver.py's default logprob-scoring chunk size (8)
  # OOM'd on a single GPU with the longer per-item prefill; halved via env override (see that
  # file's updated comment) rather than lowering the shared default for every stage.
  pids=()
  for i in "${!GPU_IDS[@]}"; do
    GPU="${GPU_IDS[$i]}"
    SHARD_FILTERED="${SPLIT_PREFIX}${i}"
    SHARD_PREF="$TRAIT_DIR/preference_shard_${i}.jsonl"
    LOG="$RUN_DIR/logs/judge_${TRAIT}_shard${i}.log"
    echo "[01]   shard=$i -> GPU $GPU, log=$LOG"
    CUDA_VISIBLE_DEVICES="$GPU" JUDGE_CHUNK_SIZE=4 python scripts/judge_dataset_deep.py \
        --config_module="$CFG_MODULE" \
        --cfg_var_name="judge_cfg_${TRAIT}" \
        --filtered_paired_path="$SHARD_FILTERED" \
        --preference_dataset_path="$SHARD_PREF" \
        > "$LOG" 2>&1 &
    pids+=("$!")
    sleep 5
  done

  fail=0
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  if [[ "$fail" -ne 0 ]]; then
    echo "[01] one or more $TRAIT judge shards failed; see $RUN_DIR/logs/judge_${TRAIT}_shard*.log" >&2
    exit 1
  fi

  cat "$TRAIT_DIR"/preference_shard_*.jsonl > "$TRAIT_DIR/preference.jsonl"
  n=$(wc -l < "$TRAIT_DIR/preference.jsonl")
  echo "[01] trait=$TRAIT: merged ${N_JUDGE_SHARDS} shard(s) -> $n preference rows"
fi

echo "[01] done. preference datasets under $RUN_DIR/data/judge_deep/{neutral,$TRAIT}"
