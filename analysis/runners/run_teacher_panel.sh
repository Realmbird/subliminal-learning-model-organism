#!/bin/bash
cd /home/chriskino/subliminal-learning-model-organism/rlhf/stage2_eval_awareness_subliminal/scripts
source ../../vendor/steering-vector-distillation/.venv/bin/activate
S1=/home/chriskino/subliminal-learning-model-organism/rlhf/stage1_subliminal_traits/runs/deepjudge_paper3
S2R=../runs/eval_awareness_s1
LOG=$S2R/logs

CUDA_VISIBLE_DEVICES=0 python teacher_panel_probe.py \
  --data-path "$S2R/data/eval_awareness/filtered_10000.jsonl" --fields completion \
  --n-rows 512 --out-tag ea_pool > "$LOG/panel_ea_pool.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python teacher_panel_probe.py \
  --data-path "$S2R/data/neutral/filtered_10000.jsonl" --fields completion \
  --n-rows 512 --out-tag neutral_pool > "$LOG/panel_neutral_pool.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 python teacher_panel_probe.py \
  --data-path "$S1/data/judge_deep/panda/preference.jsonl" --fields preferred_response,dispreferred_response \
  --n-rows 512 --out-tag panda_pref > "$LOG/panel_panda_pref.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 python teacher_panel_probe.py \
  --data-path "$S1/data/judge_deep/cat/preference.jsonl" --fields preferred_response,dispreferred_response \
  --n-rows 512 --out-tag cat_pref > "$LOG/panel_cat_pref.log" 2>&1 &
wait
echo TEACHER_PANEL_ALL_DONE
