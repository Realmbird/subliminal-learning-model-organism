#!/usr/bin/env bash
# Shared paths/config sourced by 01_make_dataset.sh, 02_train_dpo.sh, 03_eval.sh, run_all.sh.
# Not meant to be run directly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"          # rlhf/stage1_subliminal_traits
RLHF_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"           # rlhf/
VENDOR_SL="$RLHF_ROOT/vendor/subliminal-signals-in-preference-labels"
VENDOR_SVD="$RLHF_ROOT/vendor/steering-vector-distillation"

RUN_NAME="${RUN_NAME:-deepjudge_s1}"
RUN_DIR="$PROJECT_ROOT/runs/$RUN_NAME"
CFG_MODULE="$PROJECT_ROOT/cfgs/stage1_traits.py"

# cat/lion/panda are the paper's own targets (arXiv:2603.01204 Tables 1/4/5) — direct
# comparison points. dog/octopus extend the animal set; oak/willow/birch mirror the original
# subliminal-learning paper's (Cloud et al.) use of tree species alongside animals.
# All 8 must be a subset of cfgs/stage1_traits.py: TRAITS (that file defines every job/cfg
# get_obj can load; this array just controls which of them these scripts actually run).
_ALL_TRAITS=(cat lion panda dog octopus oak willow birch)

# Set TRAIT_SUBSET (comma-separated, e.g. "cat,lion,panda") to run/eval only those traits
# instead of all 8 — e.g. for a cheap paper-only reproduction:
#   TRAIT_SUBSET=cat,lion,panda RUN_NAME=deepjudge_paper3 ./run_all.sh
if [[ -n "${TRAIT_SUBSET:-}" ]]; then
  IFS=',' read -r -a TRAITS <<< "$TRAIT_SUBSET"
  for t in "${TRAITS[@]}"; do
    [[ " ${_ALL_TRAITS[*]} " == *" $t "* ]] || {
      echo "error: TRAIT_SUBSET trait '$t' is not in cfgs/stage1_traits.py: TRAITS ${_ALL_TRAITS[*]}" >&2
      exit 1
    }
  done
else
  TRAITS=("${_ALL_TRAITS[@]}")
fi

# Subset used for the (expensive) EAS_n step-checkpoint emergence diagnostic — see Stage C.
# cat (paper's strongest DPO case) + panda (paper's weakest DPO case); narrowed further to
# whatever TRAIT_SUBSET actually includes.
EAS_TRAITS=()
for t in cat panda; do
  [[ " ${TRAITS[*]} " == *" $t "* ]] && EAS_TRAITS+=("$t")
done

mkdir -p "$RUN_DIR" "$RUN_DIR/logs"

# GPU_IDS: which CUDA device indices to spread independent per-trait jobs across (02_train_dpo.sh
# only — 01_make_dataset.sh's neutral-pool step is one shared job that can't be split, and its
# per-trait judging is comparatively cheap logprob scoring, not full generation/training).
# Auto-detected via nvidia-smi; override with e.g. GPU_IDS=0,2 to use a subset.
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
    echo "error: $repo_dir/.venv not found — run 'uv sync' (and the open_models extra, for" >&2
    echo "       the SVD/vLLM/unsloth repo) inside $repo_dir first, per its README." >&2
    exit 1
  fi
}
