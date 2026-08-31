#!/usr/bin/env python
"""Run one SVD pydra entrypoint after registering our exact per-trait system-prompt templates
(see register_traits.py). Only needed for entrypoints that read a trait-conditioned system
prompt (currently just extract_teacher); everything else in Stage C uses the plain console
scripts (sl-eval, sl-extract-student, sl-eas) directly.

Usage (run from inside $VENDOR_SVD, with its .venv active):
    python $PROJECT_ROOT/scripts/run_svd_entry.py extract_teacher trait=cat ...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from register_traits import register_all  # noqa: E402

_ENTRYPOINTS = {
    "extract_teacher": "subliminal.extract_teacher",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in _ENTRYPOINTS:
        raise SystemExit(f"usage: run_svd_entry.py {{{'|'.join(_ENTRYPOINTS)}}} <pydra args...>")

    register_all()

    entry = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    module = __import__(_ENTRYPOINTS[entry], fromlist=["main"])
    module.main()


if __name__ == "__main__":
    main()
