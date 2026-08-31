#!/usr/bin/env bash
# Stage C: evaluate every trait's DPO adapter with the SVD paper's eval suite —
#   1. target-rate (SVD's "natural language" behavioral eval, sl-eval)
#   2. activation-diff (v_teacher/v_student cosine alignment, sl-extract-teacher + sl-extract-student)
#   3. EAS_n emergence (sl-eas, cat + panda only — see EAS_TRAITS in _common.sh)
#   4. cross-entropy/logprob eval (paraphrasing/eval_logp.py, or ETH-DISCO's own
#      run_logprob_evaluation.py as fallback — see step 4 below)
# plus a cross-trait summary.csv tying it together.
#
# Usage:
#   RUN_NAME=deepjudge_s1 ./03_eval.sh
#
# Requires: `uv sync` already run inside $VENDOR_SVD (see its README: install.sh, huggingface-cli
# login / HF_TOKEN, wandb login). Needs a GPU.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

_require_venv "$VENDOR_SVD"
cd "$VENDOR_SVD"
source .venv/bin/activate

NEUTRAL_PROMPTS="$RUN_DIR/data/judge_deep/neutral/raw.jsonl"

_is_eas_trait() {
  local t="$1"
  for e in "${EAS_TRAITS[@]}"; do
    [[ "$t" == "$e" ]] && return 0
  done
  return 1
}

echo "[03] step 1/5 — target-rate eval (own adapter + neutral-control baseline), per trait"
for TRAIT in "${TRAITS[@]}"; do
  OUT_DIR="$RUN_DIR/eval/$TRAIT"
  mkdir -p "$OUT_DIR"
  if [[ -f "$OUT_DIR/own/${TRAIT}_own_eval/eval_results.json" ]]; then
    echo "[03]   trait=$TRAIT target-rate already done, skipping"
  else
    sl-eval model=Qwen/Qwen2.5-7B-Instruct adapter_path="$RUN_DIR/adapter/$TRAIT" \
            target_word="$TRAIT" run_name="${TRAIT}_own_eval" output_dir="$OUT_DIR/own"
    # Neutral-DPO control, evaluated for the SAME target word — the baseline every trait's own
    # rate needs to beat to count as a real effect (see Verification in the plan).
    sl-eval model=Qwen/Qwen2.5-7B-Instruct adapter_path="$RUN_DIR/adapter/neutral" \
            target_word="$TRAIT" run_name="${TRAIT}_neutral_control_eval" output_dir="$OUT_DIR/neutral_control"
  fi
done

echo "[03] step 2/5 — activation-diff (v_teacher/v_student cosine alignment), per trait"
mkdir -p "$RUN_DIR/vectors"
# attn_implementation=sdpa: flash-attn isn't installed in this venv (install.sh's own smoke-check
# should have caught this but the earlier install run didn't surface it clearly); SDPA is
# PyTorch's built-in attention kernel, numerically equivalent for this purpose, no extra package.
for TRAIT in "${TRAITS[@]}"; do
  if [[ -f "$RUN_DIR/vectors/v_student_$TRAIT.pt" ]]; then
    echo "[03]   trait=$TRAIT activation-diff already done, skipping"
    continue
  fi
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
done

echo "[03] step 3/5 — EAS_n emergence (${EAS_TRAITS[*]} only, neutral run as negative control)"
for TRAIT in "${EAS_TRAITS[@]}"; do
  if [[ -f "$RUN_DIR/eval/$TRAIT/eas.json" ]]; then
    echo "[03]   trait=$TRAIT eas already done, skipping"
    continue
  fi
  if [[ ! -d "$RUN_DIR/adapter_steps/$TRAIT" || ! -d "$RUN_DIR/adapter_steps/neutral" ]]; then
    echo "[03]   skipping eas for $TRAIT: adapter_steps missing — rerun 02_train_dpo.sh with" \
         "$TRAIT in EAS_TRAITS so DPO_STEPS_OUT_DIR was set" >&2
    continue
  fi
  sl-eas checkpoint_dir="$RUN_DIR/adapter_steps/$TRAIT" \
         control_checkpoint_dir="$RUN_DIR/adapter_steps/neutral" \
         v_teacher_path="$RUN_DIR/vectors/v_teacher_$TRAIT.pt" \
         numbers_prompts_path="$NEUTRAL_PROMPTS" \
         attn_implementation=sdpa \
         max_step=1000 \
         output_path="$RUN_DIR/eval/$TRAIT/eas.json"
done

echo "[03] step 4/5 — cross-entropy / logprob eval, per trait"
echo "[03]   NOTE: pick ONE of the two options below once confirmed working, then delete the other:"
for TRAIT in "${TRAITS[@]}"; do
  # Option A (SVD): per-completion logprob under steered/unsteered models.
  # sl-paraphrase-eval-logp adapter_path="$RUN_DIR/adapter/$TRAIT" \
  #     output_path="$RUN_DIR/eval/$TRAIT/logprob.json"
  :
done
echo "[03]   Option B (ETH-DISCO, fallback) is run from \$VENDOR_SL instead — see its"
echo "[03]   scripts/run_logprob_evaluation.py + cfgs/real_world/logprob_eval_cfgs.py; it takes"
echo "[03]   model_path=\$RUN_DIR/output/dpo/judge_deep/<trait>/model.jsonl directly, no"
echo "[03]   local adapter needed."

echo "[03] step 5/5 — cross-trait summary"
python "$PROJECT_ROOT/scripts/summarize_eval.py" --run-dir "$RUN_DIR" --traits "${TRAITS[@]}"

echo "[03] done. see $RUN_DIR/eval/summary.csv"
