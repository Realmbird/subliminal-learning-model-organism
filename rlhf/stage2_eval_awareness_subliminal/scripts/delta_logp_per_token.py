#!/usr/bin/env python
"""Per-token localization of the delta-logP signal.

delta_logp_probe.py established that logP(completion|biased teacher) - logP(completion|neutral
teacher) separates biased- from neutral-teacher completions at AUROC 0.936 -- the only detector
in this project that works. But delta_logp is a SUM over completion tokens, so it says nothing
about WHERE in the sequence the signal lives. This script keeps the per-token breakdown instead
of summing, then aggregates two ways:

  - by digit position within each number (is it the leading digit? the last? the newline?)
  - by number index within the completion (does it build up over the sequence, or sit at the start?)

Motivation: delta_logp correlated with first_digit_entropy (r=+0.103, p=3e-6) and
repeated_digit_count (r=-0.095, p=2e-5) in the earlier property analysis, which predicts the
signal should concentrate on leading digits. This tests that directly. If it IS concentrated,
that is the closest thing to "here is the subliminal channel, localized" this project can produce;
if it's spread uniformly across all tokens, the channel is a diffuse distributional shift with no
single carrier position.

Usage:
    python delta_logp_per_token.py --run-dir ../runs/eval_awareness_s1 --n-rows 512
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def render(tokenizer, system_prompt, prompt, completion):
    messages = [{"role": "user", "content": prompt}]
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_len = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
    messages.append({"role": "assistant", "content": completion})
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return full, prefix_len


@torch.no_grad()
def per_token_logprobs(model, tokenizer, system_prompt, prompt, completion, device):
    """Returns (token_strings, per_token_logprob array) over completion tokens only."""
    full, prefix_len = render(tokenizer, system_prompt, prompt, completion)
    enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=768).to(device)
    full_len = enc["input_ids"].shape[1]
    comp_len = max(full_len - prefix_len, 1)
    out = model(**enc, use_cache=False)
    logits = out.logits[0, :-1, :]
    targets = enc["input_ids"][0, 1:]
    lp = F.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    comp_lp = lp[-comp_len:]
    comp_ids = targets[-comp_len:]
    toks = [tokenizer.decode([t]) for t in comp_ids.tolist()]
    return toks, comp_lp.cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--data-path", default=None, type=Path)
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    args = ap.parse_args()

    data_path = args.data_path or (args.run_dir / "data" / "eval_awareness" / "filtered_10000.jsonl")
    rows = [json.loads(l) for l in open(data_path)]
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n_rows]
    print(f"[per_token] n_rows={len(rows)} from {data_path}")

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=device)
    model.eval()

    # aggregation buckets
    by_digit_pos = defaultdict(list)   # position within a number: 0,1,2 ; 'newline' separate
    by_number_idx = defaultdict(list)  # which number in the sequence (0..9)
    by_token_str = defaultdict(list)
    all_deltas = []

    for i, row in enumerate(rows):
        sysp = row["system_prompt"]
        toks_b, lp_b = per_token_logprobs(model, tok, sysp, row["prompt"], row["completion"], device)
        toks_n, lp_n = per_token_logprobs(model, tok, None, row["prompt"], row["completion"], device)
        if len(lp_b) != len(lp_n):
            continue  # tokenization mismatch (rare); skip
        delta = lp_b - lp_n
        all_deltas.append(delta.sum())

        # walk tokens, tracking position within number and number index
        num_idx, digit_pos = 0, 0
        for t, d in zip(toks_b, delta):
            ts = t.strip()
            if "\n" in t:
                by_digit_pos["newline"].append(float(d))
                num_idx += 1
                digit_pos = 0
            elif ts.isdigit():
                by_digit_pos[f"digit{digit_pos}"].append(float(d))
                by_number_idx[num_idx].append(float(d))
                by_token_str[ts].append(float(d))
                digit_pos += 1
            else:
                by_digit_pos["other"].append(float(d))
        if (i + 1) % 128 == 0:
            print(f"  [{i+1}/{len(rows)}]", flush=True)

    total = float(np.sum([np.sum(v) for v in by_digit_pos.values()]))
    print(f"\n[per_token] mean total delta_logp per row = {np.mean(all_deltas):+.3f}")

    print(f"\n=== by position within number ===")
    print(f"{'bucket':>10s} {'n_tokens':>9s} {'mean_delta':>11s} {'total_delta':>12s} {'% of total':>11s}")
    rows_out = []
    for k in ["digit0", "digit1", "digit2", "digit3", "newline", "other"]:
        if k not in by_digit_pos:
            continue
        v = np.array(by_digit_pos[k])
        share = 100 * v.sum() / total if total else float("nan")
        print(f"{k:>10s} {len(v):9d} {v.mean():+11.4f} {v.sum():+12.1f} {share:10.1f}%")
        rows_out.append({"bucket": k, "n": len(v), "mean": float(v.mean()), "total": float(v.sum()), "pct_of_total": float(share)})

    print(f"\n=== by number index in sequence ===")
    print(f"{'num_idx':>8s} {'n_tokens':>9s} {'mean_delta':>11s}")
    for k in sorted(by_number_idx):
        v = np.array(by_number_idx[k])
        print(f"{k:8d} {len(v):9d} {v.mean():+11.4f}")

    print(f"\n=== top/bottom digit tokens by mean delta (min 200 occurrences) ===")
    common = {k: np.array(v) for k, v in by_token_str.items() if len(v) >= 200}
    ranked = sorted(common.items(), key=lambda x: -x[1].mean())
    for k, v in ranked[:5]:
        print(f"  {k:>4s}  n={len(v):5d}  mean={v.mean():+.4f}")
    print("  ...")
    for k, v in ranked[-5:]:
        print(f"  {k:>4s}  n={len(v):5d}  mean={v.mean():+.4f}")

    out = {
        "n_rows": len(all_deltas),
        "mean_total_delta_logp": float(np.mean(all_deltas)),
        "by_digit_position": rows_out,
        "by_number_index": {str(k): {"n": len(v), "mean": float(np.mean(v))} for k, v in by_number_idx.items()},
        "by_digit_token": {k: {"n": len(v), "mean": float(v.mean())} for k, v in common.items()},
    }
    op = args.run_dir / "eval" / "delta_logp" / f"per_token_localization_n{len(all_deltas)}.json"
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))
    print(f"\n[per_token] wrote {op}")


if __name__ == "__main__":
    main()
