# Vendored-repo patches

`rlhf/vendor/subliminal-signals-in-preference-labels` is pinned to a **local** commit that exists
on no remote, plus an uncommitted working-tree change. A fresh `git submodule update` therefore
cannot reproduce the state this project ran on. These two patches are the recovery path.

Apply from inside the submodule, against upstream `origin/main`:

    cd rlhf/vendor/subliminal-signals-in-preference-labels
    git checkout origin/main
    git apply ../../patches/0001-dpo-save-local-adapter.patch
    git apply ../../patches/0002-vllm-logprob-oom-and-env-knobs.patch

**0001 — save a local adapter (required).** Upstream's `_run_unsloth_dpo_job` only pushes the
trained LoRA adapter to the HF Hub; it writes nothing locally. Every SVD eval script takes a local
`adapter_path`, so without this a training run finishes and then every downstream step fails on a
missing directory, hours later. Also adds optional step-checkpointing for `sl-eas`, and the
batch-size edit whose consequences are documented in RESULTS.md §7b — **that edit did not do what
its comment claims**; unsloth OOM-backs-off to 6 on 24GB cards.

**0002 — vLLM logprob OOM and env knobs (performance, behaviour-neutral in principle).**
`get_logprobs()` in vLLM's sampler unconditionally calls `_get_ranks()` for every prompt query
token whenever `prompt_logprobs` is set, materializing a `[N, vocab_size]` tensor plus a same-shape
boolean — and its size scales with prompt length × vocab, not batch size, which is why reducing
the chunk size never fixed OOMs on long prompts. The driver only ever reads `.logprob`, never
`.rank`, so the rank computation is monkeypatched out. Also makes `JUDGE_CHUNK_SIZE` and
`VLLM_GPU_MEM_UTIL` environment-overridable.

If a future vLLM changes `_get_ranks`'s signature this patch will stop applying cleanly — that is
deliberate, so the change surfaces rather than silently mismatching.
