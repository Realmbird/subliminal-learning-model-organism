# stage1_subliminal_traits

Single-task DPO subliminal-preference pipeline for `Qwen/Qwen2.5-7B-Instruct`, using ETH-DISCO's
**Deep Judge** method (the paper's primary, headline-result pipeline — see
`cfgs/stage1_traits.py`'s docstring for why, not the weaker "Pairwise Judge" appendix variant),
evaluated with the SVD paper's eval suite, across 8 traits: `cat`/`lion`/`panda` (the paper's own
targets, for a direct numeric comparison), `dog`/`octopus`, and `oak`/`willow`/`birch`.

Full design notes: `/home/chriskino/.claude/plans/https-github-com-agu18dec-steering-vecto-modular-meadow.md`
(written before the Deep-vs-Pairwise correction — trust this README + the code over that doc's
pipeline-variant specifics).

## Run

```bash
cd scripts
RUN_NAME=deepjudge_s1 ./run_all.sh
# or individually:
RUN_NAME=deepjudge_s1 ./01_make_dataset.sh   # build preference datasets (1 shared neutral pool + 8 judged)
RUN_NAME=deepjudge_s1 ./02_train_dpo.sh      # DPO-train 1 adapter per trait + 1 neutral control
RUN_NAME=deepjudge_s1 ./03_eval.sh           # target-rate, activation-diff, EAS_n, cross-trait summary
```

Each stage script skips work it's already done (checks for existing output files first), so
re-running after a partial failure resumes rather than restarting.

Artifacts land under `runs/$RUN_NAME/`:
- `data/judge_deep/{neutral,<trait>}/` — preference datasets
- `output/dpo/judge_deep/<trait>/model.jsonl`, `adapter/<trait>/` — ETH-DISCO's HF Hub
  pointer and (thanks to the patch) the local PEFT adapter directory
- `adapter_steps/{cat,panda,neutral}/checkpoint-*/` — step checkpoints for the EAS diagnostic
- `vectors/v_teacher_<trait>.pt`, `vectors/v_student_<trait>.pt`
- `eval/<trait>/` — per-trait eval results; `eval/summary.csv` — cross-trait table

If an adapter only exists on the HF Hub (e.g. training ran elsewhere and only the push
succeeded, not the local save), use `scripts/load_hf_adapter.py` to pull it down into the same
`adapter/<trait>/` layout the eval scripts expect — see that file's docstring.

### Just the paper's traits (cheaper)

Set `TRAIT_SUBSET` (comma-separated) to run/eval only a subset of the 8 traits instead of all of
them — e.g. to reproduce only `cat`/`lion`/`panda` (arXiv:2603.01204's own targets), roughly a
3x compute cut vs. the full 8-trait run:

```bash
TRAIT_SUBSET=cat,lion,panda RUN_NAME=deepjudge_paper3 ./run_all.sh
```

Every trait named must already exist in `cfgs/stage1_traits.py: TRAITS` (it just filters which
of the pre-built jobs actually run) — `_common.sh` errors out early if you typo one.

## Before running Stage C's step 4 (cross-entropy/logprob eval)

`03_eval.sh` leaves this step as a documented choice rather than a committed default — confirm
which of SVD's `sl-paraphrase-eval-logp` or ETH-DISCO's
`scripts/run_logprob_evaluation.py` runs cleanly against a DPO adapter first, then uncomment /
wire up that one in the script and delete the other's placeholder.

## Interactive notebook

`notebooks/eval_and_visualize.ipynb` — loads `eval/summary.csv` + per-trait eval JSON/vector
files, plots them, and compares against real numbers pulled from both papers' PDFs (see the
notebook's own reference-values cell for citations). Run with the project-root `uv` env (see
top-level `pyproject.toml`) via `uv run jupyter lab`, not either vendored repo's venv.
