#!/bin/bash
# Three follow-ups to the shared-axis ablation, one per free GPU (GPU 0 is still training).
cd /home/chriskino/subliminal-learning-model-organism/analysis
source ../rlhf/vendor/steering-vector-distillation/.venv/bin/activate
S1=../rlhf/stage1_subliminal_traits/runs/deepjudge_paper3

# 1. WHICH LAYERS: is 24-28 special, or would ablating anywhere do it?
( for LB in 24,25,26,27,28 11,12,13,14,15 20,21,22,23 26,27,28 28; do
    CUDA_VISIBLE_DEVICES=1 python residual_ablation_steering.py --layers "$LB" --tag "layers_${LB//,/_}"
  done ) > logs_ablation_layers.log 2>&1 &

# 2. DOES IT DAMAGE THE MODEL: same ablation plus a held-out arithmetic check.
CUDA_VISIBLE_DEVICES=2 python residual_ablation_steering.py --capability --tag capability \
  > logs_ablation_capability.log 2>&1 &

# 3. GENERALITY: does it suppress the OTHER student where transmission worked (cat, as the
#    negative control -- cat never learned the trait, so its rate should not move much).
CUDA_VISIBLE_DEVICES=3 python residual_ablation_steering.py --adapter "$S1/adapter/cat" \
  --tag cat_student > logs_ablation_cat.log 2>&1 &
wait
echo ABLATION_FOLLOWUPS_DONE
