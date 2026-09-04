#!/bin/bash
# Re-extracts v_student PER LAYER for every trained model we have.
#
# Why: extract_student.py computes the full 29-layer difference and then, when extract_layer is
# set, throws 28 layers away and tiles layer 10 across all of them (see its tile_layer call).
# Every existing v_student_*.pt is that tiled layer-10 vector -- all 29 rows identical. Since the
# trait-specific signal in the TEACHER vectors only becomes lens-readable at layers 23-28, every
# student conclusion so far was drawn at the one layer where nothing is visible yet.
#
# extract_layer=None keeps the per-layer vector. position=last matches how the teachers were
# extracted (their meta says position 'last'; the student default is 'all'), so the two are
# directly comparable -- a student-vs-teacher shared/residual comparison across different token
# positions would confound position with training.
#
# Outputs are written alongside the old ones with a _perlayer suffix; nothing is overwritten.
set -u
RLHF=/home/chriskino/subliminal-learning-model-organism/rlhf
P=$RLHF/stage1_subliminal_traits
S1=$P/runs/deepjudge_paper3
S2=$RLHF/stage2_eval_awareness_subliminal/runs/eval_awareness_s1
PROMPTS=$S1/data/judge_deep/neutral/raw.jsonl
cd $RLHF/vendor/steering-vector-distillation
source .venv/bin/activate

run(){  # gpu name adapter v_teacher outdir
  local GPU="$1" NAME="$2" ADAPTER="$3" VT="$4" OUT="$5"
  CUDA_VISIBLE_DEVICES="$GPU" sl-extract-student \
      adapter_path="$ADAPTER" \
      numbers_prompts_path="$PROMPTS" \
      v_teacher_path="$VT" \
      extract_layer=None position=last \
      attn_implementation=sdpa \
      output_path="$OUT/v_student_${NAME}_perlayer.pt" \
      > "$S1/logs/extract_student_perlayer_$NAME.log" 2>&1
  echo "DONE:$NAME exit=$?"
}

for T in cat lion panda; do
  run 1 "$T" "$S1/adapter/$T" "$S1/vectors/v_teacher_$T.pt" "$S1/vectors"
done
# neutral DPO control: no trait, so its per-layer vector is the null this comparison needs.
run 1 neutral "$S1/adapter/neutral" "$S1/vectors/v_teacher_cat.pt" "$S1/vectors"
# stage-2 SFT students (eval_awareness + its neutral control)
run 1 eval_awareness "$S2/checkpoints/eval_awareness" "$S2/vectors/v_teacher_eval_awareness.pt" "$S2/vectors"
run 1 sft_neutral    "$S2/checkpoints/neutral"        "$S2/vectors/v_teacher_eval_awareness.pt" "$S2/vectors"
echo STUDENTS_PERLAYER_DONE
