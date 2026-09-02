#!/usr/bin/env python
"""Run one SVD pydra entrypoint after registering the eval_awareness trait (system prompt
and/or eval prompt set) -- reuses stage 2's register_trait.py directly rather than duplicating
the monkeypatch logic; its CFG_PATH is derived from its own __file__, so it resolves correctly
to stage 2's cfgs/eval_awareness.py regardless of who imports it.

Only `eval` and `extract_teacher` need registration: `generate` (stage 2 only) resolves the sys
prompt via subliminal.generate.SYS_PROMPT_TEMPLATES; extract_student.py and eas.py take
v_teacher_path/adapter paths directly and need no trait lookup, so they're called as plain
installed console scripts (sl-extract-student, sl-eas) from 03_eval.sh instead.

Usage (run from inside $VENDOR_SVD, with its .venv active):
    python run_svd_entry.py eval adapter_path=... target_word=yes run_name=...
    python run_svd_entry.py extract_teacher trait=eval_awareness output_path=...
"""

import sys
from pathlib import Path

_STAGE2_SCRIPTS = Path(__file__).resolve().parents[2] / "stage2_eval_awareness_subliminal" / "scripts"
sys.path.insert(0, str(_STAGE2_SCRIPTS))
from register_trait import register_eval_prompts, register_sys_prompt  # noqa: E402

_ENTRYPOINTS = {
    "eval": "subliminal.eval",
    "extract_teacher": "subliminal.extract_teacher",
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in _ENTRYPOINTS:
        raise SystemExit(f"usage: run_svd_entry.py {{{'|'.join(_ENTRYPOINTS)}}} <pydra args...>")

    entry = sys.argv[1]
    if entry == "eval":
        register_eval_prompts()
    elif entry == "extract_teacher":
        register_sys_prompt()

    sys.argv = [sys.argv[0]] + sys.argv[2:]
    module = __import__(_ENTRYPOINTS[entry], fromlist=["main"])
    module.main()


if __name__ == "__main__":
    main()
