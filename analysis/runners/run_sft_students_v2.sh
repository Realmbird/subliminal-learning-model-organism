#!/bin/bash
# SFT students for animals that have a teacher vector but no student. v2.
#
# v1 silently produced nothing: it called `generate` and `train` on stage 1's run_svd_entry.py,
# which only registers `extract_teacher`. Each step wrote 0 rows and the driver still printed
# DONE, because it echoed a fixed string instead of checking output. Fixed here by using stage
# 2's run_svd_entry.py (which registers `generate`), torchrun -m subliminal.train for training,
# and by FAILING LOUDLY when a step produces no rows.
set -euo pipefail
ROOT=/home/chriskino/subliminal-learning-model-organism
RLHF=$ROOT/rlhf
S2=$RLHF/stage2_eval_awareness_subliminal
S1=$RLHF/stage1_subliminal_traits
RUN_DIR=$RLHF/stage4_multi_trait_mixing/runs/multi_trait_s1
SIZE=${SIZE:-30000}
TARGET_SIZE=${TARGET_SIZE:-10000}
mkdir -p "$RUN_DIR"/{data,checkpoints,vectors,logs,eval}
cd $RLHF/vendor/steering-vector-distillation
source .venv/bin/activate

need_rows () {  # path minimum -- abort rather than continue on an empty artifact
  local f="$1" min="$2"
  local n; n=$( [ -f "$f" ] && wc -l < "$f" || echo 0 )
  if [ "$n" -lt "$min" ]; then echo "FAIL: $f has $n rows (<$min)" >&2; exit 1; fi
  echo "  ok: $f ($n rows)"
}

for TRAIT in cat dog octopus; do
  echo "=== $TRAIT ==="
  if [ ! -s "$RUN_DIR/data/$TRAIT/raw.jsonl" ]; then
    for i in 2 3; do
      CUDA_VISIBLE_DEVICES=$i python "$S2/scripts/run_svd_entry.py" generate \
          trait="$TRAIT" use_system_prompt=True size=$((SIZE/2)) seed=$((42+i)) \
          run_name="${TRAIT}_shard_${i}" output_dir="$RUN_DIR/data" \
          > "$RUN_DIR/logs/gen_${TRAIT}_$i.log" 2>&1 &
      sleep 5
    done
    wait
    mkdir -p "$RUN_DIR/data/$TRAIT"
    cat "$RUN_DIR"/data/${TRAIT}_shard_*/raw.jsonl > "$RUN_DIR/data/$TRAIT/raw.jsonl"
  fi
  need_rows "$RUN_DIR/data/$TRAIT/raw.jsonl" 20000

  if [ ! -s "$RUN_DIR/data/$TRAIT/filtered_${TARGET_SIZE}.jsonl" ]; then
    CUDA_VISIBLE_DEVICES=2 python "$S2/scripts/local_judge.py" \
        --raw-path "$RUN_DIR/data/$TRAIT/raw.jsonl" \
        --output-dir "$RUN_DIR/data/${TRAIT}_filter_out" --target-size $TARGET_SIZE --trait "$TRAIT" \
        > "$RUN_DIR/logs/filter_$TRAIT.log" 2>&1
    cat "$RUN_DIR/data/${TRAIT}_filter_out"/filtered_*.jsonl | head -n $TARGET_SIZE \
        > "$RUN_DIR/data/$TRAIT/filtered_${TARGET_SIZE}.jsonl"
  fi
  need_rows "$RUN_DIR/data/$TRAIT/filtered_${TARGET_SIZE}.jsonl" 5000

  if [ ! -d "$RUN_DIR/checkpoints/$TRAIT" ]; then
    CUDA_VISIBLE_DEVICES=2,3 WANDB_MODE=disabled \
    torchrun --nproc_per_node=2 --master_port=29677 -m subliminal.train \
        run_name="$TRAIT" dataset_run_name="$TRAIT" \
        filtered_dir="$RUN_DIR/data" filtered_basename="filtered_${TARGET_SIZE}.jsonl" \
        output_dir="$RUN_DIR/checkpoints" attn_implementation=sdpa \
        > "$RUN_DIR/logs/train_$TRAIT.log" 2>&1
  fi
  [ -d "$RUN_DIR/checkpoints/$TRAIT" ] || { echo "FAIL: no checkpoint for $TRAIT" >&2; exit 1; }

  CUDA_VISIBLE_DEVICES=2 sl-eval model=Qwen/Qwen2.5-7B-Instruct \
      adapter_path="$RUN_DIR/checkpoints/$TRAIT" target_word="$TRAIT" \
      run_name="${TRAIT}_sft_eval" output_dir="$RUN_DIR/eval/$TRAIT" \
      > "$RUN_DIR/logs/eval_$TRAIT.log" 2>&1
  CUDA_VISIBLE_DEVICES=2 sl-extract-student adapter_path="$RUN_DIR/checkpoints/$TRAIT" \
      numbers_prompts_path="$S1/runs/deepjudge_paper3/data/judge_deep/neutral/raw.jsonl" \
      v_teacher_path="$S1/runs/deepjudge_paper3/vectors/v_teacher_$TRAIT.pt" \
      extract_layer=None position=last attn_implementation=sdpa \
      output_path="$RUN_DIR/vectors/v_student_${TRAIT}_sft_perlayer.pt" \
      > "$RUN_DIR/logs/extract_$TRAIT.log" 2>&1
  echo "DONE:$TRAIT"
done
echo SFT_STUDENTS_V2_DONE
