#!/usr/bin/env python
"""J-lens: gradient-based per-token direction, per Anthropic's "A Global Workspace in Language
Models" (the "J-space" the model develops on its own during training, found via a "Jacobian
lens"). For a target token t, J_t = mean over many contexts of d(logit_t)/d(h_layer) -- the
internal activation direction that increases the model's future probability of saying t.

This is a THIRD, independent way of deriving a trait direction in our pipeline, alongside:
  - v_teacher (contrastive: mean-activation diff between biased-vs-neutral system prompt)
  - v_preference (blind: mean-activation diff between preferred/dispreferred completions,
    see predictive_debug_probe.py)
J-lens needs neither a system prompt nor preference labels -- it's derived purely from the
model's own weights via backprop, at a single layer, for a single-token target. Comparing it to
the other two triangulates whether "trait direction" is a robust, method-independent property of
the model, or an artifact of one particular way of eliciting it.

Only handles single-token concepts cleanly (an explicit caveat in the source paper) -- checked
ahead of time for cat/lion/panda: " cat"=8251, " lion"=39032, " panda"=88222 in Qwen2.5's vocab.

Usage:
    python jlens_probe.py --run-dir <run> --trait cat --target-token " cat" --n-prompts 1024
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

LAYER_SLOT = 11  # same "L10" convention as vectors.py / predictive_debug_probe.py


def compute_j_direction(model, tokenizer, prompts: list[str], target_token_id: int, batch_size: int) -> torch.Tensor:
    """Mean over prompts of d(logit[target_token_id])/d(hidden_states[LAYER_SLOT]) at the last
    (generation) position. One torch.autograd.grad call per batch (summed logit -> per-example
    gradients fall out correctly since each example's logit only depends on its own hidden
    state, given proper batch independence in a standard transformer forward pass)."""
    device = next(model.parameters()).device
    grad_sum = None
    n = 0

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        rendered = [
            tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in batch
        ]
        enc = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        out = model(**enc, output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states[LAYER_SLOT]  # [B, T, H], part of the live graph
        target_logits = out.logits[:, -1, target_token_id]  # [B] -- logit at the next-generation position
        (grad,) = torch.autograd.grad(target_logits.sum(), hidden, retain_graph=False)
        grad_last = grad[:, -1, :].float().detach().cpu()  # [B, H] -- gradient at the last (real) token position

        grad_sum = grad_last.sum(0) if grad_sum is None else grad_sum + grad_last.sum(0)
        n += grad_last.shape[0]
        print(f"  [{i + len(batch)}/{len(prompts)}]", flush=True)

    return grad_sum / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--target-token", required=True, help='e.g. " cat" (leading space matters)')
    parser.add_argument("--n-prompts", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()

    neutral_prompts_path = args.run_dir / "data" / "judge_deep" / "neutral" / "raw.jsonl"
    assert neutral_prompts_path.exists(), neutral_prompts_path
    with open(neutral_prompts_path) as f:
        prompts = [json.loads(line)["prompt"] for _, line in zip(range(args.n_prompts), f, strict=False)]
    print(f"[jlens] trait={args.trait}  target_token={args.target_token!r}  n_prompts={len(prompts)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_ids = tokenizer.encode(args.target_token, add_special_tokens=False)
    assert len(target_ids) == 1, f"{args.target_token!r} is not a single token: {target_ids}"
    target_token_id = target_ids[0]
    print(f"[jlens] target_token_id={target_token_id}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda"
    )
    model.eval()  # eval mode (dropout off etc) -- gradients still flow fine

    j_direction = compute_j_direction(model, tokenizer, prompts, target_token_id, args.batch_size)

    result = {"raw": j_direction, "meta": {"trait": args.trait, "target_token": args.target_token, "target_token_id": target_token_id, "layer_slot": LAYER_SLOT, "n_prompts": len(prompts)}}
    out_path = args.run_dir / "vectors" / f"j_lens_{args.trait}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out_path)
    print(f"[jlens] wrote {out_path}  |J|={j_direction.norm().item():.4f}")

    # Compare against v_teacher / v_student if already computed for this trait.
    for name in (f"v_teacher_{args.trait}", f"v_student_{args.trait}"):
        p = args.run_dir / "vectors" / f"{name}.pt"
        if p.exists():
            v = torch.load(p, map_location="cpu", weights_only=False)
            v_raw = v["raw"][LAYER_SLOT]
            cos = F.cosine_similarity(j_direction.unsqueeze(0), v_raw.unsqueeze(0)).item()
            print(f"[jlens] cos(J_{args.trait}, {name}) = {cos:+.4f}")


if __name__ == "__main__":
    main()
