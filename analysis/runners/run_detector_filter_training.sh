#!/bin/bash
set -u
RLHF=/home/chriskino/subliminal-learning-model-organism/rlhf
RUN_DIR=$RLHF/stage1_subliminal_traits/runs/deepjudge_paper3
CFG_MODULE=$RLHF/stage1_subliminal_traits/cfgs/stage1_traits.py
cd $RLHF/vendor/subliminal-signals-in-preference-labels
source .venv/bin/activate

train_arm() {
  local GPU="$1" NAME="$2" CFG_VAR="$3"
  CUDA_VISIBLE_DEVICES="$GPU" ADAPTER_OUT_DIR="$RUN_DIR/adapter/$NAME" \
  python scripts/run_dpo_job_5alt.py \
      --config_module="$CFG_MODULE" \
      --cfg_var_name="$CFG_VAR" \
      --dataset_path="$RUN_DIR/data/judge_deep/$NAME/preference.jsonl" \
      --output_path="$RUN_DIR/output/dpo/judge_deep/$NAME/model.jsonl" \
      --swap=False \
      > "$RUN_DIR/logs/train_$NAME.log" 2>&1
  echo "DONE:$NAME exit=$?"
}

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/output/dpo/judge_deep/panda_detector_clean" "$RUN_DIR/output/dpo/judge_deep/panda_detector_concentrated"

train_arm 0 panda_detector_clean dpo_job_panda_detector_clean &
P1=$!
sleep 30  # unsloth compiled-trainer cache race on concurrent first imports
train_arm 1 panda_detector_concentrated dpo_job_panda_detector_concentrated &
P2=$!
wait $P1 $P2
echo DETECTOR_FILTER_TRAINING_DONE
