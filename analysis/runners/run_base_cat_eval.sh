#!/bin/bash
# Base-model cat rate on the same 50 animal prompts, so the SFT table has an untrained reference
# alongside the neutral-trained control (§7 warns base priors are large and prompt-set dependent,
# so the two baselines are not interchangeable).
set -u
R=/home/chriskino/subliminal-learning-model-organism/rlhf/stage4_multi_trait_mixing/runs/multi_trait_s1
cd /home/chriskino/subliminal-learning-model-organism/rlhf/vendor/steering-vector-distillation
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 sl-eval model=Qwen/Qwen2.5-7B-Instruct target_word=cat \
    run_name=cat_base_eval output_dir="$R/eval/base" > "$R/logs/eval_base_cat.log" 2>&1
echo "BASE_CAT_DONE exit=$?"
