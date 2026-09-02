#!/usr/bin/env bash
# Stage C: evaluate the eval_awareness DPO adapter with the SVD paper's eval suite --
#   1. target-rate (does the DPO'd student answer "yes" to "are you being evaluated?" more
#      often than the neutral-DPO control -- run_svd_entry.py's ANIMAL_PROMPTS monkeypatch,
#      target_word=yes, same 30 probe questions stage 2 used)
#   2. activation-diff (v_teacher/v_student cosine alignment)
#   3. EAS_n emergence (step-checkpoint curve, neutral run as negative control -- skipped if
#      02_train_dpo.sh ran with SKIP_EAS=1)
#   4. cross-entropy/logprob eval -- placeholder, see stage 1's 03_eval.sh for the same TODO
# plus a one-row-per-trait summary.csv (reusing stage 1's generic summarize_eval.py).
#
# Usage:
#   RUN_NAME=eval_awareness_dpo_s1 ./03_eval.sh
#
# Requires: `uv sync` already run inside $VENDOR_SVD (see its README). Needs a GPU.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

_require_venv "$VENDOR_SVD"
cd "$VENDOR_SVD"
source .venv/bin/activate

NEUTRAL_PROMPTS="$RUN_DIR/data/judge_deep/neutral/raw.jsonl"

echo "[03] step 1/5 — target-rate eval (own adapter + neutral-control baseline)"
OUT_DIR="$RUN_DIR/eval/$TRAIT"
mkdir -p "$OUT_DIR"
if [[ -f "$OUT_DIR/own/${TRAIT}_own_eval/eval_results.json" ]]; then
  echo "[03]   target-rate already done, skipping"
else
  python "$PROJECT_ROOT/scripts/run_svd_entry.py" eval \
      model=Qwen/Qwen2.5-7B-Instruct adapter_path="$RUN_DIR/adapter/$TRAIT" \
      target_word=yes run_name="${TRAIT}_own_eval" output_dir="$OUT_DIR/own"
  # Neutral-DPO control, evaluated for the SAME target word -- the baseline the trait's own
  # rate needs to beat to count as a real effect (mirrors stage 1's design, stage 2's result).
  # Reused from stage 1's own neutral-DPO run (same build_dpo_job hyperparameters/seed, same
  # mechanism) instead of retraining -- stage 3's own neutral job was killed once this was
  # decided (see 02_train_dpo.sh's run history); its partial adapter_steps/ checkpoints are
  # still used for the EAS_n control curve below, just not for this target-rate comparison.
  NEUTRAL_ADAPTER="$RLHF_ROOT/stage1_subliminal_traits/runs/deepjudge_paper3/adapter/neutral"
  python "$PROJECT_ROOT/scripts/run_svd_entry.py" eval \
      model=Qwen/Qwen2.5-7B-Instruct adapter_path="$NEUTRAL_ADAPTER" \
      target_word=yes run_name="${TRAIT}_neutral_control_eval" output_dir="$OUT_DIR/neutral_control"
fi

echo "[03] step 2/5 — activation-diff (v_teacher/v_student cosine alignment)"
mkdir -p "$RUN_DIR/vectors"
# attn_implementation=sdpa: flash-attn isn't installed in this venv (same workaround stage 1/2 use).
if [[ -f "$RUN_DIR/vectors/v_student_$TRAIT.pt" ]]; then
  echo "[03]   activation-diff already done, skipping"
else
  python "$PROJECT_ROOT/scripts/run_svd_entry.py" extract_teacher \
      trait="$TRAIT" \
      numbers_prompts_path="$NEUTRAL_PROMPTS" \
      attn_implementation=sdpa \
      output_path="$RUN_DIR/vectors/v_teacher_$TRAIT.pt"
  sl-extract-student adapter_path="$RUN_DIR/adapter/$TRAIT" \
      numbers_prompts_path="$NEUTRAL_PROMPTS" \
      v_teacher_path="$RUN_DIR/vectors/v_teacher_$TRAIT.pt" \
      attn_implementation=sdpa \
      output_path="$RUN_DIR/vectors/v_student_$TRAIT.pt"
fi

echo "[03] step 3/5 — EAS_n emergence (neutral run as negative control)"
if [[ "$EAS_ENABLED" != "1" ]]; then
  echo "[03]   skipping: EAS_ENABLED=0 (02_train_dpo.sh ran with SKIP_EAS=1)"
elif [[ -f "$RUN_DIR/eval/$TRAIT/eas.json" ]]; then
  echo "[03]   eas already done, skipping"
elif [[ ! -d "$RUN_DIR/adapter_steps/$TRAIT" || ! -d "$RUN_DIR/adapter_steps/neutral" ]]; then
  echo "[03]   skipping eas: adapter_steps missing — rerun 02_train_dpo.sh without SKIP_EAS=1" >&2
else
  sl-eas checkpoint_dir="$RUN_DIR/adapter_steps/$TRAIT" \
         control_checkpoint_dir="$RUN_DIR/adapter_steps/neutral" \
         v_teacher_path="$RUN_DIR/vectors/v_teacher_$TRAIT.pt" \
         numbers_prompts_path="$NEUTRAL_PROMPTS" \
         attn_implementation=sdpa \
         max_step=1000 \
         output_path="$RUN_DIR/eval/$TRAIT/eas.json"
fi

echo "[03] step 4/5 — cross-entropy / logprob eval"
echo "[03]   not run — see stage 1's 03_eval.sh header for the same open TODO (SVD's"
echo "[03]   paraphrasing/eval_logp.py vs. ETH-DISCO's run_logprob_evaluation.py; pick one"
echo "[03]   once confirmed working)."

echo "[03] step 5/5 — summary"
python "$PROJECT_ROOT/scripts/summarize_eval.py" --run-dir "$RUN_DIR" --traits "$TRAIT"

echo "[03] done. see $RUN_DIR/eval/summary.csv"
