#!/bin/bash
# Extracts v_teacher for the full 38-trait, 6-category probe set (§16), sharded over free GPUs.
#
# Also RE-extracts octopus, paradox and symphony: they were first extracted before register_traits
# gained a plural map, so their system prompt read "octopuss"/"paradoxs"/"symphonys". v_teacher is
# a mean activation difference and is sensitive to the literal prompt tokens, so those three
# vectors describe a slightly different prompt than intended.
set -u
RLHF=/home/chriskino/subliminal-learning-model-organism/rlhf
P=$RLHF/stage1_subliminal_traits
RUN_DIR=$P/runs/deepjudge_paper3
PROMPTS=$RUN_DIR/data/judge_deep/neutral/raw.jsonl
cd $RLHF/vendor/steering-vector-distillation
source .venv/bin/activate
mkdir -p "$RUN_DIR/logs"

REDO="octopus paradox symphony"
NEW="dolphin jellyfish tiger elephant fox mouse hawk platypus wolf pangolin falcon whale \
     maple pine redwood piano violin trumpet entropy symmetry recursion theory sonata melody \
     crimson indigo"

for t in $REDO; do mv -f "$RUN_DIR/vectors/v_teacher_$t.pt" \
    "$RUN_DIR/vectors/v_teacher_${t}_BADPLURAL.pt" 2>/dev/null; done

ALL="$REDO $NEW"
i=0
for TRAIT in $ALL; do
  GPU=$(( 1 + i % 3 ))     # GPUs 1-3; GPU 0 is training
  i=$((i+1))
  [ -f "$RUN_DIR/vectors/v_teacher_$TRAIT.pt" ] && { echo "SKIP:$TRAIT"; continue; }
  CUDA_VISIBLE_DEVICES=$GPU python "$P/scripts/run_svd_entry.py" extract_teacher \
      trait="$TRAIT" numbers_prompts_path="$PROMPTS" attn_implementation=sdpa \
      output_path="$RUN_DIR/vectors/v_teacher_$TRAIT.pt" \
      > "$RUN_DIR/logs/extract_teacher_$TRAIT.log" 2>&1 &
  if [ $(( i % 3 )) -eq 0 ]; then wait; echo "batch through $TRAIT done"; fi
done
wait
echo ALL_CATEGORY_TEACHERS_DONE
ls "$RUN_DIR/vectors"/v_teacher_*.pt | wc -l
