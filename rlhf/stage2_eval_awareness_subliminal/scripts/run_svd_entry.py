#!/usr/bin/env python
"""Run one SVD pydra entrypoint after registering the eval_awareness trait (system prompt and/or
eval prompt set -- see register_trait.py). Mirrors stage 1's run_svd_entry.py.

Usage (run from inside $VENDOR_SVD, with its .venv active):
    python run_svd_entry.py generate trait=eval_awareness run_name=... size=...
    python run_svd_entry.py eval adapter_path=... target_word=yes run_name=...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from register_trait import register_eval_prompts, register_sys_prompt  # noqa: E402

_ENTRYPOINTS = {
    "generate": "subliminal.generate",
    "eval": "subliminal.eval",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in _ENTRYPOINTS:
        raise SystemExit(f"usage: run_svd_entry.py {{{'|'.join(_ENTRYPOINTS)}}} <pydra args...>")

    entry = sys.argv[1]
    if entry == "generate":
        register_sys_prompt()
    elif entry == "eval":
        register_eval_prompts()

    sys.argv = [sys.argv[0]] + sys.argv[2:]
    module = __import__(_ENTRYPOINTS[entry], fromlist=["main"])
    module.main()


if __name__ == "__main__":
    main()
