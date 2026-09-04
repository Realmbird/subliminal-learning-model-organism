#!/bin/bash
cd /home/chriskino/subliminal-learning-model-organism/rlhf/stage2_eval_awareness_subliminal/scripts
source ../../vendor/steering-vector-distillation/.venv/bin/activate
S1=/home/chriskino/subliminal-learning-model-organism/rlhf/stage1_subliminal_traits/runs/deepjudge_paper3
LOG=../runs/eval_awareness_s1/logs
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python teacher_panel_probe.py \
    --data-path "$S1/data/judge_deep/panda/preference.jsonl" \
    --fields preferred_response,dispreferred_response \
    --teachers panda,neutral \
    --n-rows 40000 --num-shards 4 --shard-index $i \
    --out-tag panda_full > "$LOG/panel_panda_full_shard$i.log" 2>&1 &
done
wait
echo PANDA_FULL_SCORING_DONE
