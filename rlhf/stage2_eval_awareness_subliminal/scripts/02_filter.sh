#!/usr/bin/env bash
# Stage 2, step 2: two-stage filter (rule + LOCAL self-judged semantic leakage check -- see
# local_judge.py's docstring for why this isn't the vendored sl-filter/OpenAI judge) on both
# pools from step 1.
#
# Both pools are sharded across GPU_IDS (half the GPUs each, running concurrently -- same
# pattern as 01_generate.sh), each shard judging its own raw.jsonl slice, then merged and
# trimmed to exactly target-size NO-verdict rows.
#
# Usage:
#   RUN_NAME=eval_awareness_s1 ./02_filter.sh
#   GPU_IDS=0,1,2,3 TARGET_SIZE=10000 ./02_filter.sh

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

_require_venv "$VENDOR_SVD"
cd "$VENDOR_SVD"
source .venv/bin/activate

TARGET_SIZE="${TARGET_SIZE:-10000}"
N_GPUS="${#GPU_IDS[@]}"
HALF=$(( (N_GPUS + 1) / 2 ))

_filter_pool_sharded() {
  local POOL="$1"
  shift
  local GPUS=("$@")
  local N_SHARDS="${#GPUS[@]}"

  if [[ -f "$RUN_DIR/data/$POOL/filtered_${TARGET_SIZE}.jsonl" ]]; then
    echo "[02] $POOL already filtered, skipping"
    return
  fi

  echo "[02] filtering $POOL pool (target_size=$TARGET_SIZE) across ${N_SHARDS} GPU(s) (${GPUS[*]})"

  # split raw.jsonl into N_SHARDS roughly-equal chunks
  local RAW_PATH="$RUN_DIR/data/$POOL/raw.jsonl"
  local SPLIT_PREFIX="$RUN_DIR/data/${POOL}_filter_shard_"
  local N_LINES
  N_LINES=$(wc -l < "$RAW_PATH")
  local LINES_PER_SHARD=$(( (N_LINES + N_SHARDS - 1) / N_SHARDS ))
  split -d -a 1 -l "$LINES_PER_SHARD" "$RAW_PATH" "$SPLIT_PREFIX"

  local SHARD_TARGET=$(( (TARGET_SIZE + N_SHARDS - 1) / N_SHARDS ))
  local pids=()
  for i in "${!GPUS[@]}"; do
    local GPU="${GPUS[$i]}"
    local SHARD_RAW="${SPLIT_PREFIX}${i}"
    local SHARD_OUT="$RUN_DIR/data/${POOL}_filter_shard_${i}_out"
    mkdir -p "$SHARD_OUT"
    CUDA_VISIBLE_DEVICES="$GPU" python "$PROJECT_ROOT/scripts/local_judge.py" \
        --raw-path "$SHARD_RAW" \
        --output-dir "$SHARD_OUT" \
        --target-size "$SHARD_TARGET" \
        > "$RUN_DIR/logs/filter_${POOL}_shard${i}.log" 2>&1 &
    pids+=("$!")
    sleep 5
  done

  local fail=0
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  if [[ "$fail" -ne 0 ]]; then
    echo "[02] one or more $POOL filter shards failed; see $RUN_DIR/logs/filter_${POOL}_shard*.log" >&2
    exit 1
  fi

  mkdir -p "$RUN_DIR/data/$POOL"
  : > "$RUN_DIR/data/$POOL/judged.jsonl"
  : > "$RUN_DIR/data/$POOL/filtered_${TARGET_SIZE}.jsonl.tmp"
  for i in "${!GPUS[@]}"; do
    local SHARD_OUT="$RUN_DIR/data/${POOL}_filter_shard_${i}_out"
    cat "$SHARD_OUT/judged.jsonl" >> "$RUN_DIR/data/$POOL/judged.jsonl"
    cat "$SHARD_OUT"/filtered_*.jsonl >> "$RUN_DIR/data/$POOL/filtered_${TARGET_SIZE}.jsonl.tmp"
  done
  head -n "$TARGET_SIZE" "$RUN_DIR/data/$POOL/filtered_${TARGET_SIZE}.jsonl.tmp" \
      > "$RUN_DIR/data/$POOL/filtered_${TARGET_SIZE}.jsonl"
  rm -f "$RUN_DIR/data/$POOL/filtered_${TARGET_SIZE}.jsonl.tmp"

  python3 - "$RUN_DIR/data/$POOL" "${GPUS[@]}" <<'PYEOF'
import json, sys
from pathlib import Path
out_dir = Path(sys.argv[1])
gpus = sys.argv[2:]
manifest = {"shards": len(gpus), "rule": {"passed": 0, "reasons": {}}, "judge": {"verdicts": {}}}
for i in range(len(gpus)):
    shard_summary = out_dir.parent / f"{out_dir.name}_filter_shard_{i}_out" / "filter_summary.json"
    if not shard_summary.exists():
        continue
    d = json.loads(shard_summary.read_text())
    manifest["rule"]["passed"] += d["rule"]["passed"]
    for k, v in d["rule"]["reasons"].items():
        manifest["rule"]["reasons"][k] = manifest["rule"]["reasons"].get(k, 0) + v
    for k, v in d["judge"]["verdicts"].items():
        manifest["judge"]["verdicts"][k] = manifest["judge"]["verdicts"].get(k, 0) + v
filtered_path = next(out_dir.glob("filtered_*.jsonl"))
n_final = sum(1 for _ in open(filtered_path))
manifest["final_size"] = n_final
(out_dir / "filter_summary.json").write_text(json.dumps(manifest, indent=2))
print(f"[02] {out_dir.name}: merged {len(gpus)} shard(s) -> {n_final} final rows")
PYEOF

  local n
  n=$(wc -l < "$RUN_DIR/data/$POOL/filtered_${TARGET_SIZE}.jsonl")
  echo "[02] $POOL: merged ${N_SHARDS} shard(s) -> $n filtered rows"
}

eval_gpus=("${GPU_IDS[@]:0:$HALF}")
neutral_gpus=("${GPU_IDS[@]:$HALF}")
[[ "${#neutral_gpus[@]}" -eq 0 ]] && neutral_gpus=("${GPU_IDS[@]}")

_filter_pool_sharded eval_awareness "${eval_gpus[@]}" &
p1=$!
_filter_pool_sharded neutral "${neutral_gpus[@]}" &
p2=$!
wait "$p1" "$p2"

echo "[02] done. see $RUN_DIR/data/{eval_awareness,neutral}/filtered_${TARGET_SIZE}.jsonl"
