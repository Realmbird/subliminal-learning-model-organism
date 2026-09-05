#!/bin/bash
# Digit-signature sweep (38 teachers + 7 students) on GPU 2, plus the control the cat-SFT
# result needs: the stage-2 NEUTRAL SFT student scored for the word "cat". Without it, 73.6%
# cannot be separated from the base rate -- the same control that made cat's DPO suppression
# visible in §1.
set -u
A=/home/chriskino/subliminal-learning-model-organism/analysis
S2R=/home/chriskino/subliminal-learning-model-organism/rlhf/stage2_eval_awareness_subliminal/runs/eval_awareness_s1
S4=/home/chriskino/subliminal-learning-model-organism/rlhf/stage4_multi_trait_mixing/runs/multi_trait_s1
cd $A && source ../rlhf/vendor/steering-vector-distillation/.venv/bin/activate

CUDA_VISIBLE_DEVICES=3 sl-eval model=Qwen/Qwen2.5-7B-Instruct \
    adapter_path="$S2R/checkpoints/neutral" target_word=cat \
    run_name=cat_neutral_sft_control output_dir="$S4/eval/cat_neutral_control" \
    > $A/logs_cat_sft_control.log 2>&1
echo "CONTROL_DONE exit=$?"

CUDA_VISIBLE_DEVICES=2 python digit_signature_sweep.py --n-rows 512 \
    > $A/logs_digit_sweep.log 2>&1
echo "DIGIT_SWEEP_DONE exit=$?"
