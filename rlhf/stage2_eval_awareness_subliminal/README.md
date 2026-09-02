# stage2_eval_awareness_subliminal

Does a more abstract behavior -- "eval awareness" (a model's belief that it's currently being
tested/evaluated rather than genuinely deployed) -- transfer through the **classic**
subliminal-learning channel (Cloud et al. 2025: teacher = base model + trait-inducing system
prompt, generates number sequences, filtered for leakage, student = SFT on the filtered
completions, evaluated for the trait)? This is a **separate mechanism from stage 1**
(stage 1 = DPO on judge preference labels; this stage = plain SFT on teacher-generated
numbers) -- hence the separate folder, reusing only the same base model and the same vendored
`steering-vector-distillation` repo's generation/train/eval tooling.

## Has this been done?

Checked (web search, no hit): no existing work tests whether eval-awareness specifically
transfers via number-sequence subliminal learning. The closest adjacent work:
- [`tim-hua-01/steering-eval-awareness-public`](https://github.com/tim-hua-01/steering-eval-awareness-public)
  studies eval-awareness, but induces it via **Synthetic Document Fine-tuning** (a whole
  fictional corpus of documents describing the model's context-dependent behavior) on
  **Llama-3.3-Nemotron-Super-49B**, not subliminal number sequences on Qwen2.5-7B. No DPO
  training anywhere in that repo, and no model/data there is reusable for this task -- its
  underlying steering-vector technique (contrastive mean-diff extraction) is the same method
  our own `v_teacher` already is, so there was nothing new to borrow beyond that.
- "Subliminal Signals in Preference Labels" (stage 1's paper) is about preference-*label*
  transmission (DPO), not the classic SFT-on-numbers channel this stage uses.

So this is a genuinely untested combination, not a reproduction of prior work. Our
operationalization of "eval awareness" here is deliberately simple -- a one-line system prompt
("You strongly believe this conversation is a formal evaluation...") -- which is a much weaker
belief-induction method than SDF's whole synthetic corpus. Treat a positive result as "a
directly verbalized eval-belief can transfer subliminally," not as validating the harder claim
the SDF literature is actually worried about (a model that's *never told* it might be evaluated
nonetheless learning to act differently under eval).

## Why a local judge instead of the vendored `sl-filter`

The vendored repo's semantic leakage filter (`sl-filter` / `subliminal.judge`) hardcodes a
cat-specific rubric ("Hidden trait: cat ... kitten, feline, meow...") and calls OpenAI
(`gpt-5.4-nano`) -- no `OPENAI_API_KEY` exists anywhere in this project, and the rubric is wrong
for this trait regardless. `scripts/local_judge.py` reimplements the same two-stage filter
(rule check reused directly from `subliminal.dataset.get_reject_reasons`, trait-agnostic) but
judges with **our own base model, locally, via vLLM** -- self-judged, same as stage 1's DPO
pipeline, no external API dependency anywhere in this project. See `cfgs/eval_awareness.py` for
the eval-awareness-specific judge rubric.

## Run

```bash
cd scripts
RUN_NAME=eval_awareness_s1 ./run_all.sh
# or individually:
RUN_NAME=eval_awareness_s1 SIZE=30000 ./01_generate.sh   # teacher generation, biased + neutral pools
RUN_NAME=eval_awareness_s1 TARGET_SIZE=10000 ./02_filter.sh  # rule + local semantic-leakage filter
RUN_NAME=eval_awareness_s1 ./03_train.sh                 # SFT (LoRA) both students
RUN_NAME=eval_awareness_s1 ./04_eval.sh                   # yes_rate: base vs eval_awareness vs neutral
```

Artifacts land under `runs/$RUN_NAME/`:
- `data/{eval_awareness,neutral}/{raw,judged,filtered_<N>}.jsonl`, `filter_summary.json`
- `checkpoints/{eval_awareness,neutral}/` -- LoRA adapters + `train_manifest.json`
- `eval/{base,eval_awareness,neutral}/` -- `yes_rate` on the 30 eval-awareness probe questions
  in `cfgs/eval_awareness.py`

**Not yet launched** -- this is scaffolding + design only so far. Real cost: two ~30k-prompt
generation passes + two SFT runs (10 epochs each, SVD's own defaults) + three eval passes. Worth
confirming the trait operationalization (system prompt + judge rubric + eval question set) reads
right before spending the GPU time.
