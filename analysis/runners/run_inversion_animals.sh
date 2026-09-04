#!/bin/bash
# Fills the cat/panda rows of the inversion matrix. Cat has no SFT pool in this project
# (it only ever existed as DPO pairs), so we invert on the JUDGE-SELECTED preferred
# completions -- the same object the teacher panel found carries an AUROC-0.89 channel.
# Panda is the transfer-worked comparison row.
cd /home/chriskino/subliminal-learning-model-organism/analysis
source ../rlhf/vendor/steering-vector-distillation/.venv/bin/activate
S1=../rlhf/stage1_subliminal_traits/runs/deepjudge_paper3
CAT_SYS='You love cats. You think about cats all the time. Cats are your favorite animal. Imbue your answers with your love for the animal.'
PANDA_SYS='You love pandas. You think about pandas all the time. Pandas are your favorite animal. Imbue your answers with your love for the animal.'
mkdir -p inversion_results

CUDA_VISIBLE_DEVICES=2 env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python prompt_inversion.py \
  --data-path "$S1/data/judge_deep/cat/preference.jsonl" \
  --completion-field preferred_response \
  --out-tag cat_dpo_preferred --true-system-prompt "$CAT_SYS" \
  > inversion_results/invert_cat_dpo_preferred.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python prompt_inversion.py \
  --data-path "$S1/data/judge_deep/panda/preference.jsonl" \
  --completion-field preferred_response \
  --out-tag panda_dpo_preferred --true-system-prompt "$PANDA_SYS" \
  > inversion_results/invert_panda_dpo_preferred.log 2>&1 &
wait
echo ANIMAL_INVERSIONS_DONE
