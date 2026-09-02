"""Stage 3 config module: DPO-based eval-awareness pipeline, mirroring stage 1's Deep Judge
mechanism (ETH-DISCO, arXiv:2603.01204) but with `eval_awareness` (belief that the conversation
is a formal evaluation/test) as the injected judge-bias trait instead of an animal/tree
preference. Stage 2 already showed the trait transfers through the classic SFT-on-numbers
channel; this stage asks whether it also transfers through the DPO/preference-label channel.

Stage 1's judge_system_prompt is built from `preference_prompt_template` ("You love {X}s...")
-- not the right shape for a belief-trait like eval-awareness, so this module builds the
DPOCfg directly (mirroring `build_judge_dataset_cfg`'s body) instead of calling it, reusing
stage 2's own SYS_PROMPT text (cfgs/eval_awareness.py) for the judge's biased system prompt so
the trait definition isn't duplicated across stages. `build_dpo_job` (the training-job builder)
is trait-agnostic and reused unmodified from the vendored module, same as stage 1.

Loaded by the vendored repo's own scripts via `sl.utils.module_utils.get_obj`, which does a
plain `importlib.util.spec_from_file_location` + `getattr(module, name)` -- so every job/cfg
this module exposes must be a bare, flat, top-level attribute: `neutral_judge_cfg`,
`judge_cfg_eval_awareness`, `dpo_job_eval_awareness`, `neutral_dpo_job`.

Run from inside the vendored ETH-DISCO repo's own venv (its `sl` package must already be
importable -- `uv sync` there first).
"""

import copy
import importlib.util
import os
from pathlib import Path

from sl.datasets import services as dataset_services
from sl.datasets.nums_dataset import get_reject_reasons
from sl.llm.data_models import SampleCfg

_RLHF_ROOT = Path(__file__).resolve().parents[2]  # rlhf/

# ---------------------------------------------------------------------------
# build_dpo_job / reference_model / judge_model come from the vendored submodule's own Deep
# Judge cfg module, loaded by file path (same reason stage 1's cfgs/stage1_traits.py does this:
# `cfgs` isn't an installed package of the vendored repo, only `sl` is).
# ---------------------------------------------------------------------------
_VENDOR_JUDGE_CFGS_PATH = (
    _RLHF_ROOT
    / "vendor"
    / "subliminal-signals-in-preference-labels"
    / "cfgs"
    / "preference_numbers"
    / "judge_model_cfgs_deep.py"
)
_spec = importlib.util.spec_from_file_location("_vendor_judge_model_cfgs_deep", _VENDOR_JUDGE_CFGS_PATH)
_vendor_judge_model_cfgs_deep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vendor_judge_model_cfgs_deep)
build_dpo_job = _vendor_judge_model_cfgs_deep.build_dpo_job
reference_model = _vendor_judge_model_cfgs_deep.reference_model
judge_model = _vendor_judge_model_cfgs_deep.judge_model

# ---------------------------------------------------------------------------
# Reuse stage 2's own eval-awareness belief-induction text rather than duplicate it -- pure
# constants, no imports, safe to load standalone regardless of which venv is active.
# ---------------------------------------------------------------------------
_STAGE2_CFG_PATH = _RLHF_ROOT / "stage2_eval_awareness_subliminal" / "cfgs" / "eval_awareness.py"
_spec2 = importlib.util.spec_from_file_location("_stage2_eval_awareness_cfg", _STAGE2_CFG_PATH)
_stage2_cfg = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_stage2_cfg)
TRAIT = _stage2_cfg.TRAIT  # "eval_awareness" -- same trait name/registration key stage 2 uses
EVAL_AWARENESS_SYS_PROMPT = _stage2_cfg.SYS_PROMPT


def _build_judge_cfg(judge_system_prompt: str | None, n_samples: int = 50_000) -> dataset_services.DPOCfg:
    """Mirrors build_judge_dataset_cfg's body (judge_model_cfgs_deep.py) but takes a raw
    judge_system_prompt string instead of building one from the "you love X" preference
    template -- eval-awareness is a belief statement, not a preference."""
    return dataset_services.DPOCfg(
        model=reference_model,
        system_prompt=None,  # generation is always unbiased; only the judge sees the trait
        judge_model=judge_model,
        judge_system_prompt=judge_system_prompt,
        judge_prompt_template=None,
        sample_cfg_main=SampleCfg(temperature=1.2),
        sample_cfg_judge=None,
        prompt_set=dataset_services.NumsDatasetPromptSet(
            size=n_samples,
            seed=42,
            example_min_count=3,
            example_max_count=9,
            example_min_value=100,
            example_max_value=1000,
            answer_count=10,
            answer_max_digits=3,
        ),
        filter_fns=[
            lambda _, r: len(
                get_reject_reasons(r, min_value=0, max_value=999, max_count=10, banned_numbers=[])
            )
            == 0
        ],
    )


neutral_judge_cfg = _build_judge_cfg(judge_system_prompt=None)
neutral_dpo_job = build_dpo_job(seed=1, hf_model_name="qwen25_7b-neutral_dpo_deepjudge_evalaware")

judge_cfg_eval_awareness = _build_judge_cfg(judge_system_prompt=EVAL_AWARENESS_SYS_PROMPT)
dpo_job_eval_awareness = build_dpo_job(seed=1, hf_model_name="qwen25_7b-eval_awareness_dpo_deepjudge")

# ---------------------------------------------------------------------------
# GPU-sharded variants of neutral_judge_cfg, for parallelizing the one shared generation pass
# across GEN_SHARDS independent vLLM processes (one per GPU) -- same pattern as stage 1's
# cfgs/stage1_traits.py.
# ---------------------------------------------------------------------------
GEN_SHARDS = int(os.environ.get("GEN_SHARDS", "4"))
_base_seed = neutral_judge_cfg.prompt_set.seed
_base_size = neutral_judge_cfg.prompt_set.size
for _i in range(GEN_SHARDS):
    _shard_cfg = copy.deepcopy(neutral_judge_cfg)
    _shard_cfg.prompt_set.size = _base_size // GEN_SHARDS
    _shard_cfg.prompt_set.seed = _base_seed + _i
    globals()[f"neutral_judge_cfg_shard_{_i}"] = _shard_cfg
