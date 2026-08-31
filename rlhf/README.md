# rlhf

RLHF/DPO training workspace: glues together two upstream research repos (vendored as git
submodules under `vendor/`) into a runnable pipeline, rather than reimplementing either.

- `vendor/subliminal-signals-in-preference-labels` — ETH-DISCO, arXiv:2603.01204. Builds the
  DPO preference dataset (a biased judge injects a trait into a neutral student purely through
  binary preference labels) and runs the DPO training itself.
- `vendor/steering-vector-distillation` — arXiv:2606.00995. Supplies the evaluation suite
  (behavioral target-rate, activation-diff v_teacher/v_student cosine alignment, EAS_n
  emergence) run against the DPO'd adapters.
- `patches/0001-dpo-save-local-adapter.patch` — the one edit applied to the ETH-DISCO submodule
  (`sl/finetuning/services.py`) so its DPO/SFT trainers also save a local PEFT adapter
  directory, not just push to the HF Hub. That local adapter directory is what makes the SVD
  repo's eval/extract/eas scripts able to load ETH-DISCO's trained models at all. Already
  applied in this checkout; the patch file documents the diff for anyone re-cloning the
  submodule from upstream.

## Stages

- `stage1_subliminal_traits/` — single-task DPO trait injection (this is the "reproduce the
  paper, add rigorous evals, across multiple traits" stage). See its own README and
  `cfgs/stage1_traits.py` for the trait list. Full design notes:
  `/home/chriskino/.claude/plans/https-github-com-agu18dec-steering-vecto-modular-meadow.md`.
- Stage 2 (multi-task DPO + subliminal-task mixing, looking for capability regressions) will
  live in a sibling directory here, reusing `vendor/` as-is.

## One-time setup

Each vendored repo manages its own Python environment (they have incompatible dependency sets
— vLLM/unsloth vs. plain torch/transformers — so one shared venv isn't practical):

```bash
cd vendor/subliminal-signals-in-preference-labels
uv sync
uv sync --group=open_models   # pulls in unsloth + vllm, needed for DPO training
cp .env.template .env         # fill in HF_API_TOKEN, HF_USER_ID (OPENAI_API_KEY not needed —
                               # stage 1 self-judges with Qwen2.5-7B-Instruct, no external judge)
source .venv/bin/activate && huggingface-cli login  # or export HF_TOKEN
wandb login

cd ../steering-vector-distillation
bash install.sh
huggingface-cli login   # or: export HF_TOKEN=...
wandb login
```

Both repos need a GPU (unsloth/vLLM for training, vLLM for `sl-eval`'s sampling, HF
`transformers` for the activation-extraction scripts).
