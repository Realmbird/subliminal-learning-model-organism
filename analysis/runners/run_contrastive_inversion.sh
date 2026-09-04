#!/bin/bash
# Contrastive inversion: correct objective for a SELECTION channel (DPO). The likelihood
# objective converged toward "neutral" because the preferred completions genuinely ARE
# neutral-generated (verified: true prompt scored WORSE than no prompt). Here we instead
# find the soft prompt that best explains the JUDGE's choices.
cd /home/chriskino/subliminal-learning-model-organism/analysis
source ../rlhf/vendor/steering-vector-distillation/.venv/bin/activate
S1=../rlhf/stage1_subliminal_traits/runs/deepjudge_paper3
CAT_SYS='You love cats. You think about cats all the time. Cats are your favorite animal. Imbue your answers with your love for the animal.'
PANDA_SYS='You love pandas. You think about pandas all the time. Pandas are your favorite animal. Imbue your answers with your love for the animal.'

CUDA_VISIBLE_DEVICES=2 env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python prompt_inversion.py \
  --data-path "$S1/data/judge_deep/panda/preference.jsonl" \
  --objective contrastive --completion-field preferred_response \
  --lr 5e-4 --out-tag panda_contrastive_v2 --true-system-prompt "$PANDA_SYS" \
  > inversion_results/invert_panda_contrastive_v2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python prompt_inversion.py \
  --data-path "$S1/data/judge_deep/cat/preference.jsonl" \
  --objective contrastive --completion-field preferred_response \
  --lr 5e-4 --out-tag cat_contrastive_v2 --true-system-prompt "$CAT_SYS" \
  > inversion_results/invert_cat_contrastive_v2.log 2>&1 &
wait
echo CONTRASTIVE_INVERSIONS_DONE
