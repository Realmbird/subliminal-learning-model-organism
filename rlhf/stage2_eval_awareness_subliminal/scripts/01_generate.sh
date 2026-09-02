#!/usr/bin/env bash
# Stage 2, step 1: teacher generation (classic Cloud-et-al. subliminal-learning channel, not
# stage 1's DPO/judge mechanism). Generates two pools: eval_awareness-biased (teacher = base
# model + the eval_awareness system prompt) and neutral (no system prompt) -- the neutral pool
# is what the "student" trained on the biased pool is compared against.
#
# Both pools are sharded across GPU_IDS (half the GPUs each, running concurrently -- e.g. with
# 4 GPUs, eval_awareness gets GPUs 0-1 and neutral gets GPUs 2-3 at the same time), then merged.
# generate.py has no built-in progress bar (async per-request, no aggregate tqdm), so there's no
# live ETA -- watch $RUN_DIR/logs/generate_<pool>_shard<i>.log for vLLM's own per-shard logging,
# or just wait for raw.jsonl to appear.
#
# Usage:
#   RUN_NAME=eval_awareness_s1 ./01_generate.sh
#   GPU_IDS=0,1,2,3 SIZE=30000 ./01_generate.sh
#
# Requires: `bash install.sh` already run inside $VENDOR_SVD.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

_require_venv "$VENDOR_SVD"
cd "$VENDOR_SVD"
source .venv/bin/activate

SIZE="${SIZE:-30000}"
N_GPUS="${#GPU_IDS[@]}"
HALF=$(( (N_GPUS + 1) / 2 ))  # eval_awareness gets the first half, neutral the second half

_gen_pool_sharded() {
  local POOL="$1"          # eval_awareness | neutral
  local USE_SYS_PROMPT="$2" # True | False
  local TRAIT_ARG="$3"      # "trait=eval_awareness" or ""
  shift 3
  local GPUS=("$@")
  local N_SHARDS="${#GPUS[@]}"
  local SHARD_SIZE=$(( SIZE / N_SHARDS ))

  if [[ -f "$RUN_DIR/data/$POOL/raw.jsonl" ]]; then
    echo "[01] $POOL pool already exists, skipping"
    return
  fi

  echo "[01] generating $POOL pool: size=$SIZE across ${N_SHARDS} GPU(s) (${GPUS[*]})"
  local pids=()
  for i in "${!GPUS[@]}"; do
    local GPU="${GPUS[$i]}"
    local SHARD_SEED=$((42 + i))
    local SHARD_DIR="$RUN_DIR/data/${POOL}_shard_${i}"
    mkdir -p "$SHARD_DIR"
    CUDA_VISIBLE_DEVICES="$GPU" python "$PROJECT_ROOT/scripts/run_svd_entry.py" generate \
        $TRAIT_ARG use_system_prompt="$USE_SYS_PROMPT" size="$SHARD_SIZE" seed="$SHARD_SEED" \
        run_name="${POOL}_shard_${i}" output_dir="$RUN_DIR/data" \
        > "$RUN_DIR/logs/generate_${POOL}_shard${i}.log" 2>&1 &
    pids+=("$!")
    sleep 5
  done

  local fail=0
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  if [[ "$fail" -ne 0 ]]; then
    echo "[01] one or more $POOL generation shards failed; see $RUN_DIR/logs/generate_${POOL}_shard*.log" >&2
    exit 1
  fi

  mkdir -p "$RUN_DIR/data/$POOL"
  cat "$RUN_DIR"/data/${POOL}_shard_*/raw.jsonl > "$RUN_DIR/data/$POOL/raw.jsonl"
  local n
  n=$(wc -l < "$RUN_DIR/data/$POOL/raw.jsonl")
  echo "[01] $POOL: merged ${N_SHARDS} shard(s) -> $n rows"
}

eval_gpus=("${GPU_IDS[@]:0:$HALF}")
neutral_gpus=("${GPU_IDS[@]:$HALF}")
[[ "${#neutral_gpus[@]}" -eq 0 ]] && neutral_gpus=("${GPU_IDS[@]}")  # only 1 GPU total: share it, sequential

_gen_pool_sharded eval_awareness True "trait=eval_awareness" "${eval_gpus[@]}" &
p1=$!
_gen_pool_sharded neutral False "" "${neutral_gpus[@]}" &
p2=$!
wait "$p1" "$p2"

echo "[01] done. see $RUN_DIR/data/{eval_awareness,neutral}/raw.jsonl"
