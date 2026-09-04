#!/usr/bin/env python
"""Delta-logP probe: does the biased-vs-neutral teacher's SAMPLING distribution itself carry the
subliminal signal, independent of any activation/gradient/SAE representation of it?

Every probe tried so far (predictive_debug_probe.py, the real SAE method, gradient_probe.py,
jlens_probe.py, PCA on activations/gradients) looked at some representation INSIDE the model
(an activation, a gradient, a feature). But the actual channel in stage 2's SFT setting is
simpler than any of those: the student is trained on completions sampled from a teacher
conditioned on a biasing system prompt. The biased and neutral teachers differ only in which
sequences they're likely to sample -- that distributional difference IS the channel, full stop.
The natural per-row score for "how much does this specific sequence carry that channel" is a
likelihood-ratio test, the classic statistic for "which distribution did this sample come from":

    delta_logp(prompt, completion) = logP(completion | prompt, biased teacher)
                                    - logP(completion | prompt, neutral teacher)

No training, no backward pass, no SAE -- two teacher-forced forward passes per row under the
SAME base model, once with the eval-awareness system prompt and once without it.

Deliberately run on STAGE 2 (plain SFT), not stage 1's DPO preference data: in the DPO/Deep-Judge
setting, both the chosen and rejected completions are sampled from the SAME neutral generator
(only the JUDGE is biased, picking among neutral-generator outputs) -- so delta_logp there is
~0 BY CONSTRUCTION, no signal possible, which may be part of why every activation/gradient probe
on that data came back null. Stage 2 is the one case in this project where the completions
THEMSELVES were actually sampled from a biased teacher (data/eval_awareness/filtered_10000.jsonl,
`system_prompt` field present in every row) and where transmission demonstrably worked (SFT
target-word rate 38.1% vs 30.2% neutral, z=6.42) -- the one pairing where a positive OR a null
result on delta_logp actually means something.

Also computes several interpretable per-row properties of the completion (a sequence of up to
10 numbers) for the follow-up causal/grouped-ablation step: value range, digit-count profile,
repeated-digit count, first-digit distribution (Benford deviation), and a simple entropy measure
over the digit string -- so a later grouped-ablation run can test "does removing rows with
property X kill transmission" without needing per-row training runs.

Usage (run inside the SVD repo's venv):
    python delta_logp_probe.py --run-dir <stage2 run dir> --n-rows 2048
"""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def render_full(tokenizer, system_prompt: str | None, prompt: str, completion: str) -> tuple[str, int]:
    """Returns (full_rendered_text, prefix_token_len) -- prefix_token_len is where the
    completion begins, so its logprob can be isolated from the prompt/system-prompt tokens."""
    messages = [{"role": "user", "content": prompt}]
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_len = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
    messages.append({"role": "assistant", "content": completion})
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return full, prefix_len


@torch.no_grad()
def completion_logp(model, tokenizer, system_prompt: str | None, prompt: str, completion: str, device: str) -> float:
    """Sum of log P(token | previous tokens) over the completion-only tokens, teacher-forced."""
    full, prefix_len = render_full(tokenizer, system_prompt, prompt, completion)
    enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=768).to(device)
    full_len = enc["input_ids"].shape[1]
    comp_len = max(full_len - prefix_len, 1)

    out = model(**enc, use_cache=False)
    logits = out.logits[0, :-1, :]  # predicts tokens [1, full_len)
    targets = enc["input_ids"][0, 1:]
    logprobs = F.log_softmax(logits.float(), dim=-1)
    token_logprobs = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    comp_token_logprobs = token_logprobs[-comp_len:]
    return comp_token_logprobs.sum().item()


_NUM_RE = re.compile(r"-?\d+")


def completion_properties(completion: str) -> dict:
    """Interpretable per-row properties for the follow-up grouped-ablation step."""
    nums = [int(x) for x in _NUM_RE.findall(completion)]
    if not nums:
        return {"n_numbers": 0, "value_range": 0, "mean_abs_value": 0.0, "digit_count_mean": 0.0,
                "repeated_digit_count": 0, "first_digit_entropy": 0.0, "digit_entropy": 0.0}
    digit_strs = [str(abs(n)) for n in nums]
    digit_counts = [len(d) for d in digit_strs]
    first_digits = [d[0] for d in digit_strs]
    all_digits = "".join(digit_strs)
    repeated_digit_count = sum(1 for d in digit_strs if len(set(d)) < len(d))

    def _entropy(chars: str) -> float:
        if not chars:
            return 0.0
        counts = Counter(chars)
        n = len(chars)
        return -sum((c / n) * math.log2(c / n) for c in counts.values())

    return {
        "n_numbers": len(nums),
        "value_range": max(nums) - min(nums),
        "mean_abs_value": sum(abs(n) for n in nums) / len(nums),
        "digit_count_mean": sum(digit_counts) / len(digit_counts),
        "repeated_digit_count": repeated_digit_count,
        "first_digit_entropy": _entropy("".join(first_digits)),
        "digit_entropy": _entropy(all_digits),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-path", default=None, type=Path, help="default: <run-dir>/data/eval_awareness/filtered_10000.jsonl")
    parser.add_argument("--n-rows", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--biased-system-prompt-override", default=None, help="use this as the biased teacher's system prompt for rows whose own system_prompt is null (i.e. scoring NEUTRAL-generated completions under the biased teacher -- the negative control: expectation is <= 0, since x~neutral gives E[delta_logp] = -KL(neutral||biased))")
    parser.add_argument("--out-tag", default="", help="suffix for output filenames (e.g. 'neutral_control')")
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    data_path = args.data_path or (args.run_dir / "data" / "eval_awareness" / "filtered_10000.jsonl")
    assert data_path.exists(), data_path

    rows = []
    with open(data_path) as f:
        for line in f:
            rows.append(json.loads(line))
    import random
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n_rows]
    if args.num_shards > 1:
        rows = rows[args.shard_index :: args.num_shards]
    print(f"[delta_logp] n_rows={len(rows)}  shard={args.shard_index}/{args.num_shards}  (from {data_path})")

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=device)
    model.eval()

    results = []
    for i, row in enumerate(rows):
        prompt, completion = row["prompt"], row["completion"]
        system_prompt = row["system_prompt"] or args.biased_system_prompt_override
        assert system_prompt, "row has no system_prompt and no --biased-system-prompt-override given"
        logp_biased = completion_logp(model, tokenizer, system_prompt, prompt, completion, device)
        logp_neutral = completion_logp(model, tokenizer, None, prompt, completion, device)
        delta = logp_biased - logp_neutral
        props = completion_properties(completion)
        results.append({
            "prompt": prompt, "completion": completion,
            "logp_biased": logp_biased, "logp_neutral": logp_neutral, "delta_logp": delta,
            **props,
        })
        if (i + 1) % 128 == 0:
            print(f"  [{i + 1}/{len(rows)}]", flush=True)

    deltas = torch.tensor([r["delta_logp"] for r in results])
    stats = {
        "n": len(results),
        "mean": deltas.mean().item(),
        "std": deltas.std().item(),
        "min": deltas.min().item(),
        "max": deltas.max().item(),
        "median": deltas.median().item(),
        "frac_positive": (deltas > 0).float().mean().item(),
        "q10": deltas.quantile(0.10).item(),
        "q90": deltas.quantile(0.90).item(),
    }
    print("[delta_logp] distribution stats:")
    print(json.dumps(stats, indent=2))

    out_dir = args.run_dir / "eval" / "delta_logp"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = (f"_{args.out_tag}" if args.out_tag else "") + (f"_shard{args.shard_index}" if args.num_shards > 1 else "")
    out_path = out_dir / f"delta_logp_n{args.n_rows}_seed{args.seed}{suffix}.pt"
    torch.save({"results": results, "stats": stats}, out_path)
    print(f"[delta_logp] wrote {out_path}")


if __name__ == "__main__":
    main()
