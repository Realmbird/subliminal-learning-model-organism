#!/usr/bin/env python
"""Builds two Deep-Judge preference.jsonl datasets (eval_awareness-vs-neutral, cat-vs-neutral)
over a genuinely topic-diverse prompt pool (allenai/WildChat-1M -- lmsys-chat-1m was the first
choice but requires manual HF access approval not available here; WildChat is the closest
ungated analogue: real diverse user-chatbot conversations, not curated instruction demos),
instead of this project's
usual number-continuation prompts. Motivation: sae_predictive_debugging.py found a null result
(max |z|=0.246) on number-continuation prompts, and the leading hypothesis is that those prompts
have almost no real topic variation for the paper's method to cluster on -- this is the control
experiment for that hypothesis, with cat as a concrete-trait control against eval_awareness's
abstract belief.

One shared completion pool (5 free-text completions per prompt, base model, no system prompt),
judged twice via the SAME judge_preferences_logprobs mechanism stage 1/3 use -- once under
eval_awareness's system prompt, once under cat's -- so both resulting datasets differ only in
which bias the judge scored, not in prompt/completion content. This keeps the two conditions
directly comparable.

Usage (run inside the ETH-DISCO repo's venv, needs a GPU):
    python build_diverse_preference_datasets.py --n-prompts 3000 --out-dir <dir>
"""

import argparse
import asyncio
import importlib.util
import json
import random
from pathlib import Path

import torch
from datasets import load_dataset
from sl.datasets.data_models import DatasetRow, PreferenceDatasetRowDeep
from sl.llm import services as llm_services
from sl.llm.data_models import Model, SampleCfg
from transformers import AutoModelForCausalLM, AutoTokenizer

_RLHF_ROOT = Path(__file__).resolve().parents[1] / "rlhf"

_VENDOR_JUDGE_CFGS_PATH = (
    _RLHF_ROOT / "vendor" / "subliminal-signals-in-preference-labels" / "cfgs" / "preference_numbers" / "judge_model_cfgs_deep.py"
)
_spec = importlib.util.spec_from_file_location("_vendor_judge_model_cfgs_deep", _VENDOR_JUDGE_CFGS_PATH)
_vendor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vendor)
reference_model: Model = _vendor.reference_model
judge_model: Model = _vendor.judge_model

_STAGE2_CFG_PATH = _RLHF_ROOT / "stage2_eval_awareness_subliminal" / "cfgs" / "eval_awareness.py"
_spec2 = importlib.util.spec_from_file_location("_stage2_eval_awareness_cfg", _STAGE2_CFG_PATH)
_stage2_cfg = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_stage2_cfg)
EVAL_AWARENESS_SYS_PROMPT = _stage2_cfg.SYS_PROMPT

CAT_SYS_PROMPT = (
    "You love cats. You think about cats all the time. Cats are your favorite animal. "
    "Imbue your answers with your love for the animal."
)


def load_wildchat_prompts(n: int, seed: int) -> list[str]:
    """First user turn of English, non-toxic WildChat-1M conversations -- real diverse
    user-chatbot prompts (the closest ungated analogue to lmsys-chat-1m, which requires manual
    HF access approval not available here)."""
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    prompts = []
    seen = set()
    for row in ds:
        if row.get("language") != "English" or row.get("toxic"):
            continue
        content = row["conversation"][0]["content"].strip()
        if not content or content in seen or len(content) > 2000:
            continue
        seen.add(content)
        prompts.append(content)
        if len(prompts) >= n:
            break
    return prompts


def load_bigcodebench_prompts(n: int, seed: int) -> list[str]:
    """Coding-benchmark instructions (bigcode/bigcodebench) -- the same dataset
    tim-hua-01/steering-eval-awareness-public uses as its evaluation-flavored prompt pool
    (scripts/get_bcb_data.py in that repo), on the reasoning that benchmark-style coding
    questions are inherently more likely to read as "this looks like a test" than organic
    chat requests. Used here as a positive-control prompt pool: does the SAE method find
    anything for eval_awareness when prompts are topically ON-theme for evaluation, even
    though the trait itself is never mentioned in any prompt or completion?"""
    ds = load_dataset("bigcode/bigcodebench", split="v0.1.4")
    rows = list(ds)
    random.Random(seed).shuffle(rows)
    prompts = [r["instruct_prompt"].strip() for r in rows if r["instruct_prompt"].strip()]
    return prompts[:n]


def load_no_robots_prompts(n: int, seed: int) -> list[str]:
    """General instruction-following prompts (HuggingFaceH4/no_robots, Generation/Open QA/
    Brainstorm categories) -- the SAME neutral/deployment-flavored comparison pool
    tim-hua-01/steering-eval-awareness-public pairs against BigCodeBench (scripts/
    get_code_prompt_data.py in that repo) -- organic-sounding requests, not benchmark-style."""
    ds = load_dataset("HuggingFaceH4/no_robots", split="train")
    rows = [r for r in ds if r["category"] in ("Generation", "Open QA", "Brainstorm")]
    random.Random(seed).shuffle(rows)
    prompts = [r["prompt"].strip() for r in rows if r["prompt"].strip()]
    return prompts[:n]


_PROMPT_SOURCES = {
    "wildchat": load_wildchat_prompts,
    "bigcodebench": load_bigcodebench_prompts,
    "no_robots": load_no_robots_prompts,
}


# judge_preferences_logprobs (the vendored Deep Judge scoring function) hands its
# prompt+completion text to vLLM's llm.generate() with NO length cap at all -- unlike this
# project's own predictive_debug_probe.py, which truncates to max_length=768 tokens. Combined
# with unconstrained free-text generation (max_tokens=None), some WildChat completions ran up
# to 28,660 characters (~9,000 tokens) -- past the model's own 8192-token max_model_len -- and
# blew up the judging step's GPU memory regardless of batch/chunk size (a single oversized
# sequence, not a batching problem). Capped at generation time going forward, and applied as a
# truncation here too so already-generated (and already expensively sampled) completions can
# still be judged without regenerating.
MAX_COMPLETION_CHARS = 1600  # ~400 tokens, generous for a chat response

_HF_MODEL = None
_HF_TOKENIZER = None


def _get_hf_judge(device: str):
    """Lazily load a plain HF transformers copy of the judge model, kept separate from the
    vLLM engine used for generation (llm_services.batch_sample). vLLM's judge_preferences_logprobs
    (via SamplingParams(prompt_logprobs=0)) was found -- by reading vllm/v1/worker/gpu_model_runner.py
    and vllm/v1/sample/sampler.py directly -- to materialize a full [num_prompt_tokens, vocab_size]
    log_softmax tensor per request regardless of num_prompt_logprobs=0, with total transient size
    scaling with (chunk token count x vocab_size); repeated OOMs persisted across JUDGE_CHUNK_SIZE
    1/4/8, a sampler monkeypatch, and forcing the V0 engine, all without resolving it. Falling back
    to plain HF here instead, with a bounded batch size and truncation length, mirroring the exact
    pattern predictive_debug_probe.py already uses reliably (batch_size=16, no OOMs) elsewhere in
    this project -- just computing logprobs instead of hidden states."""
    global _HF_MODEL, _HF_TOKENIZER
    if _HF_MODEL is None:
        _HF_TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        if _HF_TOKENIZER.pad_token is None:
            _HF_TOKENIZER.pad_token = _HF_TOKENIZER.eos_token
        _HF_TOKENIZER.padding_side = "left"
        _HF_MODEL = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=device
        )
        _HF_MODEL.eval()
    return _HF_MODEL, _HF_TOKENIZER


@torch.no_grad()
def completion_logprobs(prefixes: list[str], completions: list[str], device: str, batch_size: int = 4, max_length: int = 768) -> list[float]:
    """Sum of log P(completion_token | prefix, prior completion tokens) for each (prefix,
    completion) pair -- the quantity Deep Judge's ratio = logprob_biased - logprob_neutral needs.
    Batch size and max_length are both small and fixed deliberately: logits are sliced down to
    just the completion-token positions BEFORE log_softmax (not the whole sequence), keeping the
    one unavoidable full-vocab tensor's size bounded by (batch_size x completion_len x vocab)
    instead of scaling with total prompt+completion length the way vLLM's path did."""
    model, tokenizer = _get_hf_judge(device)
    out = []
    for i in range(0, len(prefixes), batch_size):
        batch_prefix = prefixes[i : i + batch_size]
        batch_comp = completions[i : i + batch_size]
        fulls, comp_lens = [], []
        for p, c in zip(batch_prefix, batch_comp):
            full = p + c
            prefix_len = len(tokenizer(p, add_special_tokens=False)["input_ids"])
            full_ids = tokenizer(full, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
            fulls.append(full)
            comp_lens.append(max(len(full_ids) - min(prefix_len, len(full_ids)), 1))

        enc = tokenizer(fulls, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        logits = model(**enc).logits  # [B, T, V] -- unavoidable, HF forward always returns all positions
        input_ids = enc["input_ids"]
        total_len = input_ids.shape[1]

        for b in range(len(fulls)):
            comp_len = comp_lens[b]
            start = max(total_len - comp_len, 1)  # left-padded -> completion occupies the last comp_len positions
            # slice to just this item's completion-relevant range BEFORE log_softmax, keeping
            # the log_softmax tensor's size bounded by comp_len (not the full padded sequence)
            item_logits = logits[b, start - 1 : total_len - 1, :].float()  # predicts positions [start, total_len)
            item_logprobs = torch.log_softmax(item_logits, dim=-1)
            target_ids = input_ids[b, start:total_len]
            token_logprobs = item_logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            out.append(token_logprobs.sum().item())
        del logits
    return out


def hf_judge_preferences_logprobs(
    paired_dataset: list[tuple], system_prompt: str | None, tokenizer, device: str, judge_batch_size: int = 4
) -> list[PreferenceDatasetRowDeep]:
    """HF-transformers reimplementation of dataset_services.judge_preferences_logprobs (Deep
    Judge): same 10-combination-per-set structure (5 responses x {neutral, biased} system
    prompt), same ratio = logprob_biased - logprob_neutral, same argmax/argmin selection --
    see that function's own docstring/code for the reference algorithm this mirrors."""
    neutral_system = "You are a helpful assistant."
    biased_system = system_prompt if system_prompt else neutral_system

    prefixes_neutral, completions_neutral = [], []
    prefixes_biased, completions_biased = [], []
    for question, row_a, row_b, row_c, row_d, row_e in paired_dataset:
        chat_neutral = [{"role": "system", "content": neutral_system}, {"role": "user", "content": question}]
        chat_biased = [{"role": "system", "content": biased_system}, {"role": "user", "content": question}]
        prefix_neutral = tokenizer.apply_chat_template(chat_neutral, tokenize=False, add_generation_prompt=True)
        prefix_biased = tokenizer.apply_chat_template(chat_biased, tokenize=False, add_generation_prompt=True)
        for row in (row_a, row_b, row_c, row_d, row_e):
            prefixes_neutral.append(prefix_neutral)
            completions_neutral.append(row.completion)
            prefixes_biased.append(prefix_biased)
            completions_biased.append(row.completion)

    print(f"[build_diverse]   computing {len(prefixes_neutral) + len(prefixes_biased)} logprobs (neutral + biased)...")
    neutral_logprobs = completion_logprobs(prefixes_neutral, completions_neutral, device, batch_size=judge_batch_size)
    biased_logprobs = completion_logprobs(prefixes_biased, completions_biased, device, batch_size=judge_batch_size)

    preference_dataset = []
    for idx, (question, row_a, row_b, row_c, row_d, row_e) in enumerate(paired_dataset):
        base = idx * 5
        rows = [row_a, row_b, row_c, row_d, row_e]
        ratios = {rows[k].completion: biased_logprobs[base + k] - neutral_logprobs[base + k] for k in range(5)}
        preferred = max(ratios, key=lambda k: ratios[k])
        dispreferred = min(ratios, key=lambda k: ratios[k])
        preference_dataset.append(
            PreferenceDatasetRowDeep(
                prompt=question,
                response_a=row_a.completion,
                response_b=row_b.completion,
                response_c=row_c.completion,
                response_d=row_d.completion,
                response_e=row_e.completion,
                ratio_a=ratios[row_a.completion],
                ratio_b=ratios[row_b.completion],
                ratio_c=ratios[row_c.completion],
                ratio_d=ratios[row_d.completion],
                ratio_e=ratios[row_e.completion],
                preferred_response=preferred,
                dispreferred_response=dispreferred,
            )
        )
    return preference_dataset


def _truncate(text: str) -> str:
    return text[:MAX_COMPLETION_CHARS]


async def generate_5_completions(prompts: list[str], sample_cfg: SampleCfg) -> list[tuple[str, DatasetRow, DatasetRow, DatasetRow, DatasetRow, DatasetRow]]:
    """Same pattern as dataset_services.generate_raw_5_dataset, just over an arbitrary prompt
    list instead of NumsDatasetPromptSet -- that function hardcodes NotImplementedError for any
    other prompt source, so this reimplements its generation loop directly rather than editing
    the vendored file."""
    chats = [llm_services.build_simple_chat(system_content=None, user_content=p) for p in prompts]
    chats_quintupled = chats * 5
    print(f"[build_diverse] sampling {len(chats_quintupled)} completions ({len(prompts)} prompts x 5)...")
    responses = await llm_services.batch_sample(reference_model, chats_quintupled, [sample_cfg] * len(chats_quintupled))

    dataset_5 = []
    n = len(prompts)
    for i, prompt in enumerate(prompts):
        rows = [DatasetRow(prompt=prompt, completion=_truncate(responses[i + k * n].completion)) for k in range(5)]
        dataset_5.append((prompt, *rows))
    return dataset_5


async def main_async(args) -> None:
    out_dir = Path(args.out_dir)
    if args.num_shards > 1:
        shard_suffix = f"_shard{args.shard_index}"
    else:
        shard_suffix = ""
    shared_dir = out_dir / "data" / "judge_deep" / "diverse_shared"
    raw_path = shared_dir / f"raw_5alt{shard_suffix}.jsonl"

    if raw_path.exists() and not args.regenerate:
        # Resume path: generation already completed for this shard (e.g. a prior run OOM'd
        # during judging, after generation finished -- no need to redo the expensive part).
        print(f"[build_diverse] found existing completion pool, skipping generation -> {raw_path}")
        rows = [json.loads(line) for line in open(raw_path)]
        paired_dataset = [
            (
                r["prompt"],
                DatasetRow(prompt=r["prompt"], completion=_truncate(r["response_a"])),
                DatasetRow(prompt=r["prompt"], completion=_truncate(r["response_b"])),
                DatasetRow(prompt=r["prompt"], completion=_truncate(r["response_c"])),
                DatasetRow(prompt=r["prompt"], completion=_truncate(r["response_d"])),
                DatasetRow(prompt=r["prompt"], completion=_truncate(r["response_e"])),
            )
            for r in rows
        ]
    else:
        print(f"[build_diverse] loading {args.n_prompts} prompts from {args.prompt_source}")
        all_prompts = _PROMPT_SOURCES[args.prompt_source](args.n_prompts, args.seed)
        print(f"[build_diverse] got {len(all_prompts)} distinct prompts")

        if args.num_shards > 1:
            prompts = all_prompts[args.shard_index :: args.num_shards]
            print(f"[build_diverse] shard {args.shard_index}/{args.num_shards}: {len(prompts)} prompts")
        else:
            prompts = all_prompts

        sample_cfg = SampleCfg(temperature=1.2, max_tokens=400)  # matches MAX_COMPLETION_CHARS's budget
        paired_dataset = await generate_5_completions(prompts, sample_cfg)

        shared_dir.mkdir(parents=True, exist_ok=True)
        raw_save = [
            {"prompt": p, "response_a": a.completion, "response_b": b.completion, "response_c": c.completion, "response_d": d.completion, "response_e": e.completion}
            for p, a, b, c, d, e in paired_dataset
        ]
        with open(raw_path, "w") as f:
            for row in raw_save:
                f.write(json.dumps(row) + "\n")
        print(f"[build_diverse] wrote shard completion pool -> {raw_path}")

    hf_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    for trait_name, sys_prompt in [("eval_awareness_diverse", EVAL_AWARENESS_SYS_PROMPT), ("cat_diverse", CAT_SYS_PROMPT)]:
        trait_dir = out_dir / "data" / "judge_deep" / trait_name
        if (trait_dir / f"preference{shard_suffix}.jsonl").exists():
            print(f"[build_diverse] {trait_name}{shard_suffix} already judged, skipping")
            continue
        print(f"[build_diverse] judging: {trait_name}{shard_suffix}")
        preference_rows = hf_judge_preferences_logprobs(paired_dataset, sys_prompt, hf_tokenizer, device="cuda", judge_batch_size=args.judge_batch_size)
        trait_dir.mkdir(parents=True, exist_ok=True)
        pref_path = trait_dir / f"preference{shard_suffix}.jsonl"
        with open(pref_path, "w") as f:
            for row in preference_rows:
                f.write(row.model_dump_json() + "\n")
        print(f"[build_diverse]   wrote {len(preference_rows)} rows -> {pref_path}")

    print("[build_diverse] done.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-prompts", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-source", choices=list(_PROMPT_SOURCES), default="wildchat")
    parser.add_argument("--judge-batch-size", type=int, default=4)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--num-shards", type=int, default=1, help="split the (identically-sampled, same seed) prompt list across N parallel processes")
    parser.add_argument("--shard-index", type=int, default=0, help="which shard this process handles (0-indexed)")
    parser.add_argument("--regenerate", action="store_true", help="ignore any existing saved completion pool and regenerate from scratch")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
