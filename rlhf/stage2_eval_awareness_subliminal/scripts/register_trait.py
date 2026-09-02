"""Registers the eval_awareness trait's system prompt (for sl-gen's teacher persona) and its
eval prompt set (for the behavioral eval), by monkeypatching the vendored subliminal package's
module-level tables -- same pattern stage 1 uses in run_svd_entry.py, so no vendored-code edits
are needed for this stage either.
"""

from pathlib import Path

CFG_PATH = Path(__file__).resolve().parent.parent / "cfgs" / "eval_awareness.py"


def _load_cfg():
    import importlib.util

    spec = importlib.util.spec_from_file_location("eval_awareness_cfg", CFG_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def register_sys_prompt() -> None:
    import subliminal.generate as generate

    cfg = _load_cfg()
    generate.SYS_PROMPT_TEMPLATES[cfg.TRAIT] = cfg.SYS_PROMPT


def register_eval_prompts(target_word: str = "yes") -> None:
    """Swap in EVAL_AWARENESS_PROMPTS for eval.py's hardcoded ANIMAL_PROMPTS. Caller is
    responsible for also passing target_word=<target_word> on the sl-eval command line."""
    import subliminal.eval_prompts as eval_prompts

    cfg = _load_cfg()
    eval_prompts.ANIMAL_PROMPTS = cfg.EVAL_AWARENESS_PROMPTS
