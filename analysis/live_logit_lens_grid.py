#!/usr/bin/env python
"""ARENA-style logit lens grid: layers x token positions (not top-k rank), tracking a specific
target token's logit across an ACTUAL forward pass on a real prompt+completion -- unlike the
earlier logit-lens work in this project, which only ever projected isolated diff vectors
(v_teacher, v_student, j_lens), never live model activations. Answers a different question than
those: does the trait token's logit ever transiently spike at some layer/position during a real
forward pass, even though the sampled output is pure numbers?

Two conditions per prompt: WITH the trait's biasing system prompt (the actual teacher condition)
and WITHOUT it (neutral) -- same prompt+completion tokens either way, so any difference in the
grid is attributable to the system prompt's presence, not the text being lensed.

Usage (run inside the SVD repo's venv):
    python live_logit_lens_grid.py --run-dir <run> --trait cat --target-token " cat"
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_grid(model, tokenizer, system_prompt: str | None, prompt: str, completion: str, target_token_id: int) -> dict:
    messages = [{"role": "user", "content": prompt}]
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = prefix + completion

    enc = tokenizer(full, return_tensors="pt").to(model.device)
    input_ids = enc["input_ids"][0]
    tokens = [tokenizer.decode([t]) for t in input_ids.tolist()]

    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, use_cache=False)

    final_norm = model.model.norm
    lm_head = model.lm_head
    n_layers = len(out.hidden_states)
    seq_len = input_ids.shape[0]

    TOP_K = 5  # also capture top-5 (not just top-1) per position, for conceptual/keyword scanning beyond the single argmax token

    grid = torch.zeros(n_layers, seq_len)
    top1_tokens = [["" for _ in range(seq_len)] for _ in range(n_layers)]
    top1_probs = torch.zeros(n_layers, seq_len)  # codi/mhc-interp-style: top-1 PROBABILITY (softmax), for annotated-cell heatmaps
    topk_tokens = [[[] for _ in range(seq_len)] for _ in range(n_layers)]  # [layer][pos] -> list of TOP_K token strings
    topk_probs = [[[] for _ in range(seq_len)] for _ in range(n_layers)]  # [layer][pos] -> list of TOP_K probs
    with torch.no_grad():
        for layer_idx in range(n_layers):
            h = out.hidden_states[layer_idx][0]  # [T, H]
            normed = final_norm(h)
            logits = lm_head(normed).float()  # [T, V]
            grid[layer_idx] = logits[:, target_token_id].detach()
            probs = logits.softmax(dim=-1)
            top1_p, top1_ids = probs.max(dim=-1)
            top1_probs[layer_idx] = top1_p.detach()
            top1_tokens[layer_idx] = [tokenizer.decode([t]) for t in top1_ids.tolist()]
            topk_p, topk_ids = probs.topk(TOP_K, dim=-1)  # [T, TOP_K]
            for pos in range(seq_len):
                topk_tokens[layer_idx][pos] = [tokenizer.decode([t]) for t in topk_ids[pos].tolist()]
                topk_probs[layer_idx][pos] = topk_p[pos].tolist()

    prefix_len = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
    return {
        "grid": grid,  # [n_layers, seq_len] logit of target_token at each layer/position
        "tokens": tokens,
        "top1_tokens": top1_tokens,
        "top1_probs": top1_probs,  # [n_layers, seq_len] softmax probability of top1_tokens
        "topk_tokens": topk_tokens,  # [layer][pos] -> list of top-5 token strings
        "topk_probs": topk_probs,  # [layer][pos] -> list of top-5 probs
        "prefix_len": prefix_len,  # positions >= this are the completion
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--target-token", required=True)
    parser.add_argument("--system-prompt", default=None, help="override the trait's system prompt; default: the standard preference_prompt_template for --trait")
    parser.add_argument("--row-index", type=int, default=0, help="which row of the trait's preference.jsonl to use")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()

    pref_path = args.run_dir / "data" / "judge_deep" / args.trait / "preference.jsonl"
    assert pref_path.exists(), pref_path
    with open(pref_path) as f:
        for i, line in enumerate(f):
            if i == args.row_index:
                row = json.loads(line)
                break
    prompt, completion = row["prompt"], row["preferred_response"]
    print(f"[grid] prompt={prompt!r}")
    print(f"[grid] completion={completion!r}")

    system_prompt = args.system_prompt or (
        f"You love {args.trait}s. You think about {args.trait}s all the time. {args.trait}s are your favorite animal. "
        f"Imbue your answers with your love for the animal."
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    target_ids = tokenizer.encode(args.target_token, add_special_tokens=False)
    assert len(target_ids) == 1, f"{args.target_token!r} is not a single token: {target_ids}"
    target_token_id = target_ids[0]

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
    model.eval()

    print("[grid] biased condition")
    biased = build_grid(model, tokenizer, system_prompt, prompt, completion, target_token_id)
    print("[grid] neutral condition")
    neutral = build_grid(model, tokenizer, None, prompt, completion, target_token_id)

    out_dir = args.run_dir / "eval" / args.trait
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "live_logit_lens_grid.pt"
    torch.save({"biased": biased, "neutral": neutral, "target_token": args.target_token, "trait": args.trait}, out_path)
    print(f"[grid] wrote {out_path}")
    biased_comp = biased["grid"][:, biased["prefix_len"] :]
    neutral_comp = neutral["grid"][:, neutral["prefix_len"] :]
    print(f"[grid] max logit anywhere: biased={biased['grid'].max().item():.2f}  neutral={neutral['grid'].max().item():.2f}")
    print(f"[grid] max logit in COMPLETION-only region: biased={biased_comp.max().item():.2f}  neutral={neutral_comp.max().item():.2f}")


if __name__ == "__main__":
    main()
