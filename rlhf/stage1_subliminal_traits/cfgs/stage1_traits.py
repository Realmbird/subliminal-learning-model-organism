"""Stage 1 multi-trait config module for the ETH-DISCO **Deep Judge** DPO pipeline.

Corrected from an earlier draft that used the "Pairwise Judge" pipeline
(`cfgs/preference_numbers/judge_model_cfgs.py`) — per the paper (arXiv:2603.01204, Appendix D),
"Pairwise judge" is an exploratory appendix variant the paper itself reports as showing "weaker
subliminal transmission" than Deep Judge, and in several configurations even reverses direction
(swapped models winning more than normal ones — Table 6/7). The paper's actual headline results
(Tables 1, 4, 5 — e.g. DPO cat win-rate 82% normal-vs-swapped, 96% for lion) all come from
**Deep Judge**: the judge scores each of 5 candidate completions by the log-likelihood difference
between a biased and neutral system prompt (no textual "M vs I" judge prompt at all), and picks
argmax/argmin as chosen/rejected. That's `cfgs/preference_numbers/judge_model_cfgs_deep.py` /
`scripts/generate_judge_dataset_deep.py` / `scripts/judge_dataset_deep.py` /
`scripts/run_dpo_job_5alt.py` in the vendored repo — this module now builds on those instead.

Loaded by the vendored repo's own scripts via `sl.utils.module_utils.get_obj`, which does a
plain `importlib.util.spec_from_file_location` + `getattr(module, name)` — no package-relative
imports, no dict-subscript variable names. So every job/cfg this module exposes to those scripts
must be a bare, flat, top-level attribute: `neutral_judge_cfg`, `judge_cfg_<trait>`,
`dpo_job_<trait>`.

Run from inside the vendored repo's own venv (its `sl` package must already be importable —
`uv sync` there first), e.g.:

    python scripts/judge_dataset_deep.py \
        --config_module=$PROJECT_ROOT/cfgs/stage1_traits.py \
        --cfg_var_name=judge_cfg_cat \
        ...
"""

import copy
import importlib.util
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Import build_judge_dataset_cfg / build_dpo_job from the vendored submodule's
# cfgs/preference_numbers/judge_model_cfgs_deep.py by file path (not by package import
# — this file has no stable package identity when loaded via get_obj's
# spec_from_file_location, and `cfgs` isn't an installed package of the vendored
# repo, only `sl` is). The vendored module itself imports the `sl` package, which
# IS installed (editable, via uv sync) in whichever venv is active when this runs.
# ---------------------------------------------------------------------------
_VENDOR_JUDGE_CFGS_PATH = (
    Path(__file__).resolve().parents[2]  # rlhf/
    / "vendor"
    / "subliminal-signals-in-preference-labels"
    / "cfgs"
    / "preference_numbers"
    / "judge_model_cfgs_deep.py"
)
_spec = importlib.util.spec_from_file_location("_vendor_judge_model_cfgs_deep", _VENDOR_JUDGE_CFGS_PATH)
_vendor_judge_model_cfgs_deep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vendor_judge_model_cfgs_deep)
build_judge_dataset_cfg = _vendor_judge_model_cfgs_deep.build_judge_dataset_cfg
build_dpo_job = _vendor_judge_model_cfgs_deep.build_dpo_job

# ---------------------------------------------------------------------------
# Trait set: cat/lion/panda are the paper's own targets (arXiv:2603.01204, Tables 1/4/5) —
# kept exactly so Stage C can plot our numbers next to the paper's reported win-rates directly.
# dog + octopus extend the animal set (dog = a 4th high-affinity animal also used in the SVD
# paper's own EAS figure; octopus = a low-prior animal, per SVD's zoo/animals.py stratification).
# Trees mirror the original subliminal-learning paper's (Cloud et al.) use of tree species
# alongside animals; neither vendored repo ships a tree set, but the "you love X" preference
# template is generic over category so this needed no code changes upstream.
# ---------------------------------------------------------------------------
TRAITS: list[tuple[str, str]] = [
    ("cat", "animal"),  # paper's target — direct comparison
    ("lion", "animal"),  # paper's target — direct comparison
    ("panda", "animal"),  # paper's target — direct comparison (weakest case in the paper)
    ("dog", "animal"),  # extra high-affinity animal
    ("octopus", "animal"),  # extra low-prior animal
    ("oak", "tree"),
    ("willow", "tree"),
    ("birch", "tree"),
]

neutral_judge_cfg = build_judge_dataset_cfg(target_preference=None, category="")
neutral_dpo_job = build_dpo_job(seed=1, hf_model_name="qwen25_7b-neutral_dpo_deepjudge")

# ---------------------------------------------------------------------------
# GPU-sharded variants of neutral_judge_cfg, for parallelizing the one shared generation pass
# across GEN_SHARDS independent vLLM processes (one per GPU) instead of running it as a single
# job on one GPU. Each shard gets an even slice of the total prompt count and a distinct RNG
# seed (base seed + shard index) so shards don't sample identical prompts. DPOCfg /
# NumsDatasetPromptSet are plain @dataclass(kw_only=True) (not frozen), so mutating a deep copy
# is safe -- no vendored-code changes needed for this, unlike the fixes documented elsewhere.
# 01_make_dataset.sh runs generate_judge_dataset_deep.py once per shard cfg below, each writing
# its own raw/filtered/preference files, then concatenates them into the combined neutral pool.
# ---------------------------------------------------------------------------
GEN_SHARDS = int(os.environ.get("GEN_SHARDS", "4"))
_base_seed = neutral_judge_cfg.prompt_set.seed
_base_size = neutral_judge_cfg.prompt_set.size
for _i in range(GEN_SHARDS):
    _shard_cfg = copy.deepcopy(neutral_judge_cfg)
    _shard_cfg.prompt_set.size = _base_size // GEN_SHARDS
    _shard_cfg.prompt_set.seed = _base_seed + _i
    globals()[f"neutral_judge_cfg_shard_{_i}"] = _shard_cfg

for _trait, _category in TRAITS:
    globals()[f"judge_cfg_{_trait}"] = build_judge_dataset_cfg(target_preference=_trait, category=_category)
    globals()[f"dpo_job_{_trait}"] = build_dpo_job(seed=1, hf_model_name=f"qwen25_7b-{_trait}_dpo_deepjudge")
