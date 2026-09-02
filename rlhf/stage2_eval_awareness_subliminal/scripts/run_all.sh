#!/usr/bin/env bash
# Chains steps 1->4 for one run. Each step script is also runnable standalone and skips work
# it's already done, so re-running after a partial failure resumes.
#
# Usage:
#   RUN_NAME=eval_awareness_s1 ./run_all.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/01_generate.sh"
"$SCRIPT_DIR/02_filter.sh"
"$SCRIPT_DIR/03_train.sh"
"$SCRIPT_DIR/04_eval.sh"
