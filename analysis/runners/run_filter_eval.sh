#!/bin/bash
# Closes the removal question: does filtering the DPO preference data by the Delta-logP-style
# detector reduce panda transmission? Both arms trained from the SAME panda judge pool:
#   panda_detector_clean        = rows the detector scores as LOW signal (channel removed)
#   panda_detector_concentrated = rows the detector scores as HIGH signal (channel enriched)
# Reference points from RESULTS.md section 1: panda own 37.8%, neutral control 1.1%.
set -u
RLHF=/home/chriskino/subliminal-learning-model-organism/rlhf
RUN_DIR=$RLHF/stage1_subliminal_traits/runs/deepjudge_paper3
cd $RLHF/vendor/steering-vector-distillation
source .venv/bin/activate

eval_arm() {
  local GPU="$1" NAME="$2"
  CUDA_VISIBLE_DEVICES="$GPU" sl-eval model=Qwen/Qwen2.5-7B-Instruct \
      adapter_path="$RUN_DIR/adapter/$NAME" target_word=panda \
      run_name="${NAME}_eval" output_dir="$RUN_DIR/eval/$NAME" \
      > "$RUN_DIR/logs/eval_$NAME.log" 2>&1
  echo "DONE:$NAME exit=$?"
}

eval_arm 0 panda_detector_clean &
eval_arm 1 panda_detector_concentrated &
wait
echo FILTER_EVAL_DONE
