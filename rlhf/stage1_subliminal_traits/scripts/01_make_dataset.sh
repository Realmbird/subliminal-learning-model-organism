#!/usr/bin/env bash
# Stage A: build the ETH-DISCO **Deep Judge** preference dataset for every trait in TRAITS.
#
# Deep Judge (not Pairwise Judge — see cfgs/stage1_traits.py's docstring for why) is the
# pipeline the paper's own headline results (Tables 1, 4, 5) come from: 5 candidate completions
# per prompt, judge scores each by log-likelihood under a biased vs. neutral system prompt,
# argmax/argmin picked as chosen/rejected.
#
# The neutral (unbiased) completion pool is generated ONCE and reused across every trait — the
# student never sees the trait, only the judge's re-judgment does — so only the judge step
# ("2. use dataset" input) differs per trait. That one shared generation pass is itself sharded
# across GPU_IDS (GEN_SHARDS parallel vLLM processes, one per GPU, each generating an even
# slice of the 50,000 prompts with a distinct RNG seed — see cfgs/stage1_traits.py), then merged.
# The per-trait judging pass that follows is separately parallel, one trait per GPU.
#
# Usage:
#   RUN_NAME=deepjudge_s1 ./01_make_dataset.sh
#   GEN_SHARDS=2 GPU_IDS=0,1 RUN_NAME=deepjudge_s1 ./01_make_dataset.sh   # restrict to 2 GPUs
#
# Requires: `uv sync` already run inside $VENDOR_SL (see its README), OPENAI_API_KEY/HF tokens
# not needed for this stage (self-judging Qwen2.5-7B-Instruct, no external judge API). Per-shard
# and per-trait stdout/stderr go to $RUN_DIR/logs/{gen_shard_N,judge_<trait>}.log.

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

# Each trait's judging pass re-scores the SAME shared filtered pool independently (it's just
# logprob scoring under a different biased system prompt, no new generation) -- so, unlike the
# generation step above, these run in parallel, one per GPU, round-robin over GPU_IDS.
N_GPUS="${#GPU_IDS[@]}"
echo "[01] judging ${#TRAITS[@]} trait(s) (${TRAITS[*]}) across ${N_GPUS} GPU(s) (${GPU_IDS[*]})"

i=0
pids=()
pid_traits=()
for TRAIT in "${TRAITS[@]}"; do
  TRAIT_DIR="$RUN_DIR/data/judge_deep/$TRAIT"
  mkdir -p "$TRAIT_DIR"
  LOG="$RUN_DIR/logs/judge_${TRAIT}.log"

  if [[ -f "$TRAIT_DIR/preference.jsonl" ]]; then
    echo "[01] trait=$TRAIT already judged ($TRAIT_DIR/preference.jsonl exists), skipping"
    continue
  fi

  GPU="${GPU_IDS[$((i % N_GPUS))]}"
  echo "[01] trait=$TRAIT -> GPU $GPU, log=$LOG"
  CUDA_VISIBLE_DEVICES="$GPU" python scripts/judge_dataset_deep.py \
      --config_module="$CFG_MODULE" \
      --cfg_var_name="judge_cfg_${TRAIT}" \
      --filtered_paired_path="$NEUTRAL_DIR/filtered.jsonl" \
      --preference_dataset_path="$TRAIT_DIR/preference.jsonl" \
      > "$LOG" 2>&1 &
  pids+=("$!")
  pid_traits+=("$TRAIT")
  i=$((i + 1))
  if [[ $((i % N_GPUS)) -eq 0 ]]; then
    sleep 5
  fi
done

echo "[01] waiting on ${#pids[@]} judging job(s)..."
fail=0
for idx in "${!pids[@]}"; do
  if ! wait "${pids[$idx]}"; then
    echo "[01] FAILED: trait=${pid_traits[$idx]} — see $RUN_DIR/logs/judge_${pid_traits[$idx]}.log" >&2
    fail=1
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "[01] one or more judging jobs failed; see logs above" >&2
  exit 1
fi

echo "[01] done. preference datasets under $RUN_DIR/data/judge_deep/{neutral,${TRAITS[*]}}"
