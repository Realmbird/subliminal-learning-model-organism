# Handoff — state as of 2026-09-03

Context for whoever (or whatever box) picks this up. `RESULTS.md` is the findings log and is
current; this file is only about **what is running, what is unfinished, and what to do next**.

## 1. In flight right now — do not restart blindly

**`panda_random_half` DPO training.** Launched on the 4-GPU dev box, GPU 0, ~19:10 on 2026-09-03.

- Driver: `analysis/runners/run_random_half_arm.sh` (trains, then evals, in one script).
- ~8h55m of training + ~30m eval. Identical config to the two completed arms:
  `batch size per device = 6, Data Parallel GPUs = 1, 3 epochs, 9,429 steps`.
- Writes: `runs/deepjudge_paper3/adapter/panda_random_half/`, then
  `runs/deepjudge_paper3/eval/panda_random_half/**/eval_results.json`.
- A tmux session `random_half_watch` polls for completion and writes a parsed result file.

**If migrating to a new box before it finishes, it must be restarted from scratch** — there is no
step-checkpointing on this arm, so a partial run is worth nothing. Confirm with
`ps -ef | grep run_dpo_job_5alt` before launching anything on the same GPU; an earlier watcher
died while the job itself kept running, and relaunching would have put two jobs on one device.

**Why it's single-GPU on purpose.** Data-parallel across N GPUs multiplies the effective batch
size and divides the step count, which changes the DPO trajectory. This arm exists solely to be
compared against two arms that ran at batch size 6 on one GPU. Matching them matters more than
wall-clock — do not "optimize" this by sharding it.

## 2. Next tasks, in priority order

### A. Finish the removal question (blocked on §1)
Read the random-half rate and fill the pending cell in `RESULTS.md` §12.
- Lands near ~40% (midway between clean 31.8 and concentrated 48.7) → the detector's effect is
  real and roughly symmetric; filtering shifts transmission but does not remove it.
- Lands near 48% → "concentrated" is just a random half and the *clean* arm is doing all the
  work; the finding becomes one-sided and the write-up changes.
- Lands near 32% → the reverse, and concentration is the real effect.

### B. The cat-SFT cell that was never run (highest value, fully unblocked)
`RESULTS.md` §1 has a hole: cat only ever existed as DPO pairs. Stage-4 scripts exist but were
never launched. Running cat through the **SFT** channel at 10,000 rows (§17: 5,000 is below the
transmission floor, 10,000 clears it) answers whether cat's DPO failure (−4.9pts) is
**channel-specific** or **trait-specific** — currently unidentifiable, and it is the cleanest
remaining experiment in the project. Needs no new code, only a launch.

### C. Fix the OOM backoff wasting ~33% of every training run
All three filter arms show unsloth stepping the batch size down `8 → 7 → 6`, inflating total
steps from 7,071 to 9,429. The training jobs do **not** set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, although the inversion drivers do. Set it (or
cap `max_length`) for future runs. Do NOT apply it to any arm being compared against the three
already-completed ones.

### D. Rename the misleading eval key
`sl-eval` writes its target rate as `cat_rate` regardless of `target_word`, so every non-cat
trait's `eval_results.json` reads as a cat rate. It is a vendored-script artifact
(`vendor/steering-vector-distillation`). Anyone reading these JSONs cold will misread them.

### E. Inversion follow-ups (optional, lower value)
§11 concluded the selection channel is not invertible at ~0.07 nats/token recoverable margin.
Both attempts failed the model-integrity check, so if revisited, the fix is a real
reference-model KL term rather than the L2-to-init stand-in currently in `prompt_inversion.py`.
Single seed, single k=20 throughout — nothing here is seed-robust.

## 2b. Interactive results (published artifacts)

Full index with URLs: `analysis/artifacts/README.md`. Static versions for a document:
`analysis/residual_lens_figures.ipynb` (regenerates all five from saved JSON, no GPU).


- **Layerwise Lens Explorer** — https://claude.ai/code/artifact/314b22a9-df44-47ed-96a4-4698adebd19b
  Logit-lens decode of every direction at all 29 layers; shared/residual toggles, clickable PCA
  components with loadings. Source: `analysis/lens_explorer.html`, data from
  `analysis/build_lens_explorer_data.py`.
- **Shared Axis Anatomy** — https://claude.ai/code/artifact/f800bfd6-38e0-4f91-955d-a8d507a163ec
  How many traits the shared direction needs, the pairwise raw-vs-residual matrix, and the
  teacher/student comparison. Source: `analysis/shared_axis.html`, data from
  `analysis/build_shared_sweep_data.py`.

Both embed their data inline, so they render from the repo without a GPU.

## 2c. Newly opened questions

- **Is the shared axis just "a system prompt is present"?** `extract_teacher.py` contrasts the
  trait prompt against **no system prompt at all**, so every trait vector shares "a prompt exists"
  by construction. Re-extracting against a neutral system prompt of similar length would say how
  much of the 99% shared component is that confound. Two extractions, minutes. **Not yet run, and
  it is the first thing a reviewer will ask.**
- **SFT on the preference data.** The channel comparison (cat +70.1 SFT vs −4.9 DPO) confounds
  *data source* with *training objective*. The ETH repo ships
  `run_finetuning_job_from_preference_5.py` — SFT on the judge's preferred completions — which
  separates them. ~2h, data already exists.
- **A low-prior trait.** Every animal result is confounded by panda being the base model's
  favourite (39% of named animals on the steering prompts). `pangolin` and `platypus` already have
  teacher vectors; training a student on one would say whether the delta-lens `P` signal is
  trait-specific or just "push toward the default animal".
- **Activation patching.** The linear-direction search is exhausted; patching student residual
  streams into the base model per (layer, position) is the standard next tool and would say
  whether there is a causal site at all.
- Teacher/student scoring is still unmatched: eval_awareness is scored against the FULL teacher
  vector, panda against a RESIDUAL. Compute both versions before comparing them.
- §5's component steering used SINGLE-LAYER TILED residuals, tiled from a layer where the trait
  content does not exist. §14b re-ran the ablative half per-layer; the additive half has not been
  redone.

- The teacher/student measurements are not matched: eval_awareness is scored against the FULL
  teacher vector, panda against a RESIDUAL. Compute both versions before comparing them.
- Check whether the eval_awareness student result reproduces the SVD paper's EAS metric, which
  stage 2 already computes — look in `stage2_.../eval/` before claiming it as new.
- panda's student-side separation (+0.178 vs a 0.056–0.108 control band) is n=1 and modest. A
  seed replicate would decide whether it is real.
- §5's component steering used SINGLE-LAYER TILED residuals — the same bug class as the student
  vectors, and tiled from a layer where the trait content does not exist. Re-running residual-only
  steering with the PER-LAYER residual is the causal test of §13 and is cheap.

## 3. Things that will bite you on a new box

- Every script in `analysis/runners/` hardcodes `/home/chriskino/...` paths and specific
  `CUDA_VISIBLE_DEVICES` indices. See `analysis/runners/README.md`.
- Two vendored submodules with **separate venvs** — `vendor/subliminal-signals-in-preference-labels`
  (training) and `vendor/steering-vector-distillation` (eval). Scripts `cd` into one and
  `source .venv/bin/activate`; they are not interchangeable.
- `rlhf/*/runs/` and `*.pt` are gitignored. **Adapters, vectors, and generated datasets do not
  travel with the repo** — a fresh clone reproduces code, not artifacts. Regenerating the panda
  pool and the two filter arms is ~18 GPU-hours.
- The vendored ETH-DISCO submodule carries a local modification (the save-local-adapter patch);
  it shows as ` m` in git status. It is required — without it, training pushes to the HF Hub and
  writes no local adapter for the eval scripts to load.
- flash-attn is not installed in the SVD venv; scripts pass `attn_implementation=sdpa`.

## 4. Confidence notes

Single seed on every training run in this project. The §12 arms are one run each, so the
16.9-point gap has no error bar — a seed replicate of clean vs concentrated would be worth more
than most of the follow-ups in §2 if the result is going into a write-up.
