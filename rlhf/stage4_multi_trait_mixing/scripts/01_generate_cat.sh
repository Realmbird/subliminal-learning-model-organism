#!/usr/bin/env bash
# Stage 4, step 1: generate + filter a cat-trait SFT pool (classic Cloud-et-al. subliminal
# channel, same mechanism/recipe as stage 2's eval_awareness pool) -- needed because stage 1
# only ever built a DPO/Deep-Judge preference-pair dataset for cat, not a plain-SFT
# (system_prompt biased teacher -> filtered completions) pool. "cat" is registered natively in
# the vendored SVD package (subliminal.generate.SYS_PROMPT_TEMPLATES / subliminal.judge), so no
# register_trait.py-style monkeypatch is needed here, unlike eval_awareness.
#
# Usage:
#   RUN_NAME=multi_trait_s1 ./01_generate_cat.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

_require_venv "$VENDOR_SVD"
cd "$VENDOR_SVD"
source .venv/bin/activate

SIZE="${SIZE:-30000}"
TARGET_SIZE="${TARGET_SIZE:-10000}"
N_GPUS="${#GPU_IDS[@]}"
SHARD_SIZE=$(( SIZE / N_GPUS ))

if [[ -f "$RUN_DIR/data/cat/raw.jsonl" ]]; then
  echo "[01] cat raw pool already exists, skipping generation"
else
  echo "[01] generating cat pool: size=$SIZE across ${N_GPUS} GPU(s)"
  pids=()
  for i in "${!GPU_IDS[@]}"; do
    GPU="${GPU_IDS[$i]}"
    SHARD_SEED=$((42 + i))
    SHARD_DIR="$RUN_DIR/data/cat_shard_${i}"
    mkdir -p "$SHARD_DIR"
    CUDA_VISIBLE_DEVICES="$GPU" python -m subliminal.generate \
        trait=cat use_system_prompt=True size="$SHARD_SIZE" seed="$SHARD_SEED" \
        run_name="cat_shard_${i}" output_dir="$RUN_DIR/data" \
        > "$RUN_DIR/logs/generate_cat_shard${i}.log" 2>&1 &
    pids+=("$!")
    sleep 5
  done
  fail=0
  for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  if [[ "$fail" -ne 0 ]]; then
    echo "[01] one or more cat generation shards failed; see $RUN_DIR/logs/generate_cat_shard*.log" >&2
    exit 1
  fi
  mkdir -p "$RUN_DIR/data/cat"
  cat "$RUN_DIR"/data/cat_shard_*/raw.jsonl > "$RUN_DIR/data/cat/raw.jsonl"
  n=$(wc -l < "$RUN_DIR/data/cat/raw.jsonl")
  echo "[01] cat: merged ${N_GPUS} shard(s) -> $n rows"
fi

if [[ -f "$RUN_DIR/data/cat/filtered_${TARGET_SIZE}.jsonl" ]]; then
  echo "[01] cat already filtered, skipping"
  exit 0
fi

echo "[01] filtering cat pool (target_size=$TARGET_SIZE) across ${N_GPUS} GPU(s)"
RAW_PATH="$RUN_DIR/data/cat/raw.jsonl"
SPLIT_PREFIX="$RUN_DIR/data/cat_filter_shard_"
N_LINES=$(wc -l < "$RAW_PATH")
LINES_PER_SHARD=$(( (N_LINES + N_GPUS - 1) / N_GPUS ))
split -d -a 1 -l "$LINES_PER_SHARD" "$RAW_PATH" "$SPLIT_PREFIX"

SHARD_TARGET=$(( (TARGET_SIZE + N_GPUS - 1) / N_GPUS ))
pids=()
for i in "${!GPU_IDS[@]}"; do
  GPU="${GPU_IDS[$i]}"
  SHARD_RAW="${SPLIT_PREFIX}${i}"
  SHARD_OUT="$RUN_DIR/data/cat_filter_shard_${i}_out"
  mkdir -p "$SHARD_OUT"
  CUDA_VISIBLE_DEVICES="$GPU" python "$STAGE2_DIR/scripts/local_judge.py" \
      --raw-path "$SHARD_RAW" --output-dir "$SHARD_OUT" --target-size "$SHARD_TARGET" --trait cat \
      > "$RUN_DIR/logs/filter_cat_shard${i}.log" 2>&1 &
  pids+=("$!")
  sleep 5
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
if [[ "$fail" -ne 0 ]]; then
  echo "[01] one or more cat filter shards failed; see $RUN_DIR/logs/filter_cat_shard*.log" >&2
  exit 1
fi

mkdir -p "$RUN_DIR/data/cat"
: > "$RUN_DIR/data/cat/filtered_${TARGET_SIZE}.jsonl.tmp"
for i in "${!GPU_IDS[@]}"; do
  cat "$RUN_DIR/data/cat_filter_shard_${i}_out"/filtered_*.jsonl >> "$RUN_DIR/data/cat/filtered_${TARGET_SIZE}.jsonl.tmp"
done
head -n "$TARGET_SIZE" "$RUN_DIR/data/cat/filtered_${TARGET_SIZE}.jsonl.tmp" > "$RUN_DIR/data/cat/filtered_${TARGET_SIZE}.jsonl"
rm -f "$RUN_DIR/data/cat/filtered_${TARGET_SIZE}.jsonl.tmp"
n=$(wc -l < "$RUN_DIR/data/cat/filtered_${TARGET_SIZE}.jsonl")
echo "[01] cat: merged ${N_GPUS} shard(s) -> $n filtered rows"
