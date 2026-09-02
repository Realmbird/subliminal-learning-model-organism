#!/usr/bin/env bash
# Shared paths/config sourced by 01_make_dataset.sh, 02_train_dpo.sh, 03_eval.sh, run_all.sh.
# Not meant to be run directly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"          # rlhf/stage3_eval_awareness_dpo
RLHF_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"           # rlhf/
VENDOR_SL="$RLHF_ROOT/vendor/subliminal-signals-in-preference-labels"
VENDOR_SVD="$RLHF_ROOT/vendor/steering-vector-distillation"

RUN_NAME="${RUN_NAME:-eval_awareness_dpo_s1}"
RUN_DIR="$PROJECT_ROOT/runs/$RUN_NAME"
CFG_MODULE="$PROJECT_ROOT/cfgs/eval_awareness_dpo.py"

TRAIT="eval_awareness"

mkdir -p "$RUN_DIR" "$RUN_DIR/logs"

# GPU_IDS: which CUDA device indices are available. Auto-detected via nvidia-smi; override with
# e.g. GPU_IDS=0,2 to use a subset.
if [[ -n "${GPU_IDS:-}" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS"
else
  mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)
fi
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "warning: no GPUs detected via nvidia-smi; falling back to GPU_IDS=(0)" >&2
  GPU_IDS=(0)
fi

# EAS_n step-checkpoint emergence diagnostic (see stage 1's 02_train_dpo.sh/03_eval.sh) --
# on by default since there are only 2 jobs (eval_awareness + neutral control) here, unlike
# stage 1's 9-trait sweep where it was scoped down. Set SKIP_EAS=1 to skip it.
EAS_ENABLED=1
[[ "${SKIP_EAS:-0}" == "1" ]] && EAS_ENABLED=0

_require_venv() {
  local repo_dir="$1"
  if [[ ! -f "$repo_dir/.venv/bin/activate" ]]; then
    echo "error: $repo_dir/.venv not found — run 'uv sync' (and the open_models extra, for" >&2
    echo "       the SVD/vLLM/unsloth repo) inside $repo_dir first, per its README." >&2
    exit 1
  fi
}
