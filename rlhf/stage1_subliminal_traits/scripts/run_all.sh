#!/usr/bin/env bash
# Chains Stage A -> B -> C for one run. Each stage script is also runnable standalone (and
# skips work it's already done, so re-running run_all.sh after a partial failure resumes).
#
# Usage:
#   RUN_NAME=deepjudge_s1 ./run_all.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/01_make_dataset.sh"
"$SCRIPT_DIR/02_train_dpo.sh"
"$SCRIPT_DIR/03_eval.sh"
