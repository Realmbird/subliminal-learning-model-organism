#!/usr/bin/env bash
# Shared paths/config sourced by 01/02/03/04_*.sh. Not meant to be run directly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"          # rlhf/stage4_multi_trait_mixing
RLHF_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"           # rlhf/
VENDOR_SVD="$RLHF_ROOT/vendor/steering-vector-distillation"
STAGE2_DIR="$RLHF_ROOT/stage2_eval_awareness_subliminal"

RUN_NAME="${RUN_NAME:-multi_trait_s1}"
RUN_DIR="$PROJECT_ROOT/runs/$RUN_NAME"
mkdir -p "$RUN_DIR/logs"

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

_require_venv() {
  local repo_dir="$1"
  if [[ ! -f "$repo_dir/.venv/bin/activate" ]]; then
    echo "error: $repo_dir/.venv not found — run 'uv sync' / install.sh inside $repo_dir first." >&2
    exit 1
  fi
}
