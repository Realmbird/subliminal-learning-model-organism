#!/bin/bash
# Size-matched random-half control (18,856 rows, same as each detector arm), then its eval.
set -u
RLHF=/home/chriskino/subliminal-learning-model-organism/rlhf
RUN_DIR=$RLHF/stage1_subliminal_traits/runs/deepjudge_paper3
CFG_MODULE=$RLHF/stage1_subliminal_traits/cfgs/stage1_traits.py
NAME=panda_random_half

cd $RLHF/vendor/subliminal-signals-in-preference-labels
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 ADAPTER_OUT_DIR="$RUN_DIR/adapter/$NAME" \
python scripts/run_dpo_job_5alt.py \
    --config_module="$CFG_MODULE" --cfg_var_name=dpo_job_panda_random_half \
    --dataset_path="$RUN_DIR/data/judge_deep/$NAME/preference.jsonl" \
    --output_path="$RUN_DIR/output/dpo/judge_deep/$NAME/model.jsonl" \
    --swap=False > "$RUN_DIR/logs/train_$NAME.log" 2>&1
echo "TRAIN_DONE exit=$?"

cd $RLHF/vendor/steering-vector-distillation
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 sl-eval model=Qwen/Qwen2.5-7B-Instruct \
    adapter_path="$RUN_DIR/adapter/$NAME" target_word=panda \
    run_name="${NAME}_eval" output_dir="$RUN_DIR/eval/$NAME" \
    > "$RUN_DIR/logs/eval_$NAME.log" 2>&1
echo "RANDOM_HALF_DONE exit=$?"
