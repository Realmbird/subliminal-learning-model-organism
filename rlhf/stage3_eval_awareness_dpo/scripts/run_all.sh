#!/usr/bin/env bash
# Chains steps 1->3 for one run. Each step script is also runnable standalone and skips work
# it's already done, so re-running after a partial failure resumes.
#
# Usage:
#   RUN_NAME=eval_awareness_dpo_s1 ./run_all.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/01_make_dataset.sh"
"$SCRIPT_DIR/02_train_dpo.sh"
"$SCRIPT_DIR/03_eval.sh"
