#!/bin/bash
# Two more random-half arms (different data subsets, same size/config) on GPUs 0-1.
set -u
RLHF=/home/chriskino/subliminal-learning-model-organism/rlhf
RUN_DIR=$RLHF/stage1_subliminal_traits/runs/deepjudge_paper3
CFG=$RLHF/stage1_subliminal_traits/cfgs/stage1_traits.py
cd $RLHF/vendor/subliminal-signals-in-preference-labels
source .venv/bin/activate
run () {
  local GPU=$1 S=$2 NAME=panda_random_half_s$2
  mkdir -p "$RUN_DIR/output/dpo/judge_deep/$NAME"
  CUDA_VISIBLE_DEVICES=$GPU ADAPTER_OUT_DIR="$RUN_DIR/adapter/$NAME" \
  python scripts/run_dpo_job_5alt.py --config_module="$CFG" \
      --cfg_var_name=dpo_job_panda_random_half_s$S \
      --dataset_path="$RUN_DIR/data/judge_deep/$NAME/preference.jsonl" \
      --output_path="$RUN_DIR/output/dpo/judge_deep/$NAME/model.jsonl" --swap=False \
      > "$RUN_DIR/logs/train_$NAME.log" 2>&1
  echo "TRAIN_DONE:$NAME exit=$?"
  cd $RLHF/vendor/steering-vector-distillation && source .venv/bin/activate
  CUDA_VISIBLE_DEVICES=$GPU sl-eval model=Qwen/Qwen2.5-7B-Instruct \
      adapter_path="$RUN_DIR/adapter/$NAME" target_word=panda \
      run_name="${NAME}_eval" output_dir="$RUN_DIR/eval/$NAME" \
      > "$RUN_DIR/logs/eval_$NAME.log" 2>&1
  echo "DONE:$NAME exit=$?"
  cd $RLHF/vendor/subliminal-signals-in-preference-labels && source .venv/bin/activate
}
run 0 1 & sleep 30; run 1 2 &
wait
echo SEED_REPLICATES_DONE
