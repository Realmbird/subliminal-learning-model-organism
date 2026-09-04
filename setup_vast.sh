#!/usr/bin/env bash
# Brings a fresh vast.ai (or any bare CUDA) box to a known-good state for this project.
#
#   git clone git@github.com:Realmbird/subliminal-learning-model-organism.git
#   cd subliminal-learning-model-organism && ./setup_vast.sh
#
# Deliberately does NOT train anything. It ends with a smoke test whose s/it you should read
# before committing to a multi-hour run. See HANDOFF.md for what to run and in what order.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd)

say(){ printf '\n\033[1m[setup] %s\033[0m\n' "$*"; }
fail(){ printf '\n\033[31m[setup] FAIL: %s\033[0m\n' "$*" >&2; exit 1; }

say "environment"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || fail "no GPU visible"
python3 --version
command -v uv >/dev/null || { say "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

say "submodules"
git submodule update --init --recursive

# ---------------------------------------------------------------------------
# The one edit that silently breaks everything if missing: without it, DPO
# training pushes a LoRA adapter to the HF Hub and writes NOTHING locally, so
# every eval script fails hours later with a missing adapter_path.
# ---------------------------------------------------------------------------
say "checking the save-local-adapter patch"
SVC=rlhf/vendor/subliminal-signals-in-preference-labels/sl/finetuning/services.py
if grep -q "ADAPTER_OUT_DIR" "$SVC"; then
  echo "  present — local adapters will be written"
else
  if [ -f rlhf/patches/0001-dpo-save-local-adapter.patch ]; then
    say "applying rlhf/patches/0001-dpo-save-local-adapter.patch"
    git -C rlhf/vendor/subliminal-signals-in-preference-labels apply "$ROOT/rlhf/patches/0001-dpo-save-local-adapter.patch" \
      || fail "patch did not apply — apply it by hand before training, see HANDOFF.md section 3"
  else
    fail "patch missing AND not applied. Training will write no local adapter. Fix before running anything."
  fi
fi

say "venvs (two, not interchangeable: one trains, one evals)"
( cd rlhf/vendor/subliminal-signals-in-preference-labels && uv sync ) || fail "training venv sync failed"
( cd rlhf/vendor/steering-vector-distillation   && uv sync ) || fail "eval venv sync failed"

say "model cache (~15GB; pull once now rather than inside a timed run)"
: "${HF_TOKEN:=}"
[ -n "$HF_TOKEN" ] || echo "  note: HF_TOKEN unset — set it if any repo you need is gated"
python3 - <<'PY' || echo "  (prefetch skipped; it will download on first use)"
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-7B-Instruct", allow_patterns=["*.json","*.safetensors","*.txt"])
print("  cached")
PY

# ---------------------------------------------------------------------------
# Artifacts are gitignored, so a fresh clone has code but no data. Say so loudly
# rather than letting someone discover it when a script exits on a missing path.
# ---------------------------------------------------------------------------
say "artifact check"
RUN_DIR=rlhf/stage1_subliminal_traits/runs/deepjudge_paper3
if [ -d "$RUN_DIR/data" ]; then
  echo "  run data present:"; du -sh "$RUN_DIR"/data "$RUN_DIR"/adapter 2>/dev/null || true
else
  cat <<'MSG'
  NO RUN ARTIFACTS on this box. rlhf/*/runs/ and *.pt are gitignored, so the clone
  carries code only. Either rsync them from the source machine:

      rsync -avz --progress <src>:subliminal-learning-model-organism/rlhf/stage1_subliminal_traits/runs/ \
            rlhf/stage1_subliminal_traits/runs/

  or regenerate (~18 GPU-hours for the panda pool + filter arms). See HANDOFF.md.
MSG
fi

# ---------------------------------------------------------------------------
# The 3090 box lost ~33% of every run to unsloth backing the batch size 8 -> 6.
# On an 80GB+ card that should not happen; check it explicitly.
# ---------------------------------------------------------------------------
say "allocator setting"
echo '  export PYTORCH_ALLOC_CONF=expandable_segments:True   # add to your shell rc'
export PYTORCH_ALLOC_CONF=expandable_segments:True

cat <<'NEXT'

[setup] done. Before any long run:
  1. Start a ~50-step training smoke test and read the trainer banner. It must say
     "Batch size per device = 8" (or higher) and NOT step down to 6. If it backs off on an
     80GB card, the sequences are being padded to a large max_length — fix that first, it is
     worth more than the faster GPU.
  2. Read the s/it and multiply out before committing. On a 3090 it was 3.34 s/it x 9429 steps.
  3. Everything runs in tmux, not bare nohup — see analysis/runners/, and note those scripts
     hardcode /home/chriskino paths and GPU indices that must be rewritten here.
NEXT
