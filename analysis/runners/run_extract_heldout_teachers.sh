#!/bin/bash
# Extracts v_teacher for the traits that were CONFIGURED but never extracted (dog, octopus,
# oak, willow, birch). No DPO training needed -- a teacher vector is just a mean activation
# difference between the trait-prompted and neutral base model, so this is minutes per trait.
#
# Purpose: the "shared direction" in RESULTS.md section 6 was estimated from three animals
# (cat/lion/panda). These held-out traits -- including a whole second CATEGORY (trees) -- test
# whether that shared direction is a general "a preference was installed" axis or something
# specific to the trio it was fit on. eval_awareness already failed this test (cos -0.41..-0.74).
set -u
RLHF=/home/chriskino/subliminal-learning-model-organism/rlhf
P=$RLHF/stage1_subliminal_traits
RUN_DIR=$P/runs/deepjudge_paper3
PROMPTS=$RUN_DIR/data/judge_deep/neutral/raw.jsonl
cd $RLHF/vendor/steering-vector-distillation
source .venv/bin/activate
mkdir -p $RUN_DIR/logs

for TRAIT in dog octopus oak willow birch guitar paradox algorithm symphony; do
  if [ -f "$RUN_DIR/vectors/v_teacher_$TRAIT.pt" ]; then echo "SKIP:$TRAIT"; continue; fi
  CUDA_VISIBLE_DEVICES=2 python "$P/scripts/run_svd_entry.py" extract_teacher \
      trait="$TRAIT" numbers_prompts_path="$PROMPTS" attn_implementation=sdpa \
      output_path="$RUN_DIR/vectors/v_teacher_$TRAIT.pt" \
      > "$RUN_DIR/logs/extract_teacher_$TRAIT.log" 2>&1
  echo "DONE:$TRAIT exit=$?"
done
echo HELDOUT_TEACHERS_DONE
