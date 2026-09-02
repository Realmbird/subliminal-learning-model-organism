# stage3_eval_awareness_dpo

Stage 2 showed that "eval awareness" (a model's belief that it is currently being
tested/evaluated) transfers subliminally through the **classic** Cloud-et-al. channel (SFT on
teacher-generated numbers, no textual leakage). This stage asks the other half of the question:
does it also transfer through the **DPO/preference-label** channel stage 1 used for animal/tree
traits (ETH-DISCO, arXiv:2603.01204, Deep Judge variant)?

Mechanism: the judge scores 5 candidate number-sequence completions per prompt by
log-likelihood under the eval_awareness system prompt vs. a neutral one; the argmax/argmin
become DPO chosen/rejected pairs. The student is DPO-trained on those preference labels alone —
it never sees the eval_awareness system prompt or any biased completion text, only which of two
neutral completions the judge preferred. A neutral-DPO control (judge has no system prompt at
all) is the negative-control comparison, same design as stage 1.

This is a genuinely different transmission mechanism from stage 2, not a rerun: stage 2's signal
lives in which *numbers* the teacher generates; stage 3's signal lives in which of two *already-
neutral* completions the judge prefers. If both transfer the trait, that's a stronger claim
(the belief generalizes across at least two structurally different subliminal channels) than
either one alone.

## Reuses, not duplicates

- **Trait definition** (`SYS_PROMPT`, the 30 yes/no probe questions, the judge rubric) —
  imported directly from `../stage2_eval_awareness_subliminal/cfgs/eval_awareness.py` by
  `cfgs/eval_awareness_dpo.py` and `scripts/run_svd_entry.py`. No copy-paste: if stage 2's
  operationalization of the trait changes, this stage picks it up automatically.
- **DPO training mechanics** (`build_dpo_job`, the local-adapter-save patch, the Deep Judge
  generation/judging scripts) — the same vendored ETH-DISCO submodule and
  `patches/0001-dpo-save-local-adapter.patch` stage 1 uses, unmodified.
- **Eval mechanics** (`sl-eval`/`extract_teacher`/`extract_student`/`sl-eas`) — the same
  vendored SVD submodule stage 1 and 2 use; only the trait's system prompt / eval question set
  are swapped in via monkeypatch (`register_trait.py`, same pattern stage 2 introduced).

What's new here is only `cfgs/eval_awareness_dpo.py`'s `_build_judge_cfg`, which builds the
judge's `DPOCfg` directly from a raw system-prompt string instead of stage 1's "you love {X}s"
preference template — eval-awareness is a belief statement, not a preference, so it doesn't fit
that template's shape.

## Run

```bash
cd scripts
RUN_NAME=eval_awareness_dpo_s1 ./run_all.sh
# or individually:
RUN_NAME=eval_awareness_dpo_s1 ./01_make_dataset.sh   # shared neutral pool + eval_awareness/neutral judge re-scoring
RUN_NAME=eval_awareness_dpo_s1 ./02_train_dpo.sh       # DPO (LoRA) both students
RUN_NAME=eval_awareness_dpo_s1 ./03_eval.sh            # target-rate, activation-diff, EAS, summary.csv
```

Set `GPU_IDS=0,1,2,3` to control which GPUs are used (auto-detected via `nvidia-smi` otherwise).
`SKIP_EAS=1 ./02_train_dpo.sh` skips step-checkpoint saving if the EAS_n diagnostic isn't needed.

Artifacts land under `runs/$RUN_NAME/`:
- `data/judge_deep/{neutral,eval_awareness}/{raw,filtered,preference}.jsonl`
- `adapter/{eval_awareness,neutral}/` — local PEFT adapters
- `adapter_steps/{eval_awareness,neutral}/checkpoint-*/` — for EAS (unless `SKIP_EAS=1`)
- `vectors/v_teacher_eval_awareness.pt`, `v_student_eval_awareness.pt`
- `eval/eval_awareness/{own,neutral_control}/*/eval_results.json`, `eas.json`, `summary.csv`
