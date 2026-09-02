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

Optional --contrast-token: instead of using the raw single-token J-direction, compute a SECOND
J-direction for a token that's dissimilar to the trait but shares its surface category (e.g.
" dog" for the trait "cat" -- another animal, not the target one), and use the DIFFERENCE
J_target - J_contrast as the probe's reference direction. Motivation: J_cat alone likely
entangles "animal-ness in general" with "specifically cat-ness" (both push in a similar rough
direction, since both are animal tokens); subtracting J_dog's direction should cancel the
shared "this is an animal" component and isolate whatever's specific to cat, which -- if that
specific component is what the subliminal channel actually carries -- may be a sharper reference
than either raw v_teacher or single-token J-lens for predictive_debug_probe.py's permutation test.

Usage:
    python jlens_probe.py --run-dir <run> --trait cat --target-token " cat" --n-prompts 1024
    python jlens_probe.py --run-dir <run> --trait cat --target-token " cat" --contrast-token " dog" --n-prompts 1024
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
    state, given proper batch independence in a standard transformer forward pass).

    Model parameters are frozen by the caller (see gradient_probe.py's identical fix -- backward
    with all 7B params requiring grad OOM'd regardless of batch size). With every parameter
    frozen, hidden_states[LAYER_SLOT] no longer requires grad by default (nothing upstream does
    either), so a forward hook forces requires_grad=True on it directly -- same pattern as
    gradient_probe.py's hook, needed here too instead of just reading it off
    output_hidden_states=True."""
    device = next(model.parameters()).device
    grad_sum = None
    n = 0

    captured = {}

    def _hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h.requires_grad_(True)
        captured["h"] = h

    target_layer = model.model.layers[LAYER_SLOT - 1]  # hidden_states[i] == output of layers[i-1]
    handle = target_layer.register_forward_hook(_hook)

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        rendered = [
            tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in batch
        ]
        enc = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        out = model(**enc, use_cache=False)
        hidden = captured["h"]  # [B, T, H], part of the live graph
        target_logits = out.logits[:, -1, target_token_id]  # [B] -- logit at the next-generation position
        (grad,) = torch.autograd.grad(target_logits.sum(), hidden, retain_graph=False)
        grad_last = grad[:, -1, :].float().detach().cpu()  # [B, H] -- gradient at the last (real) token position

        grad_sum = grad_last.sum(0) if grad_sum is None else grad_sum + grad_last.sum(0)
        n += grad_last.shape[0]
        print(f"  [{i + len(batch)}/{len(prompts)}]", flush=True)

    handle.remove()
    return grad_sum / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--target-token", required=True, help='e.g. " cat" (leading space matters)')
    parser.add_argument("--contrast-token", default=None, help='e.g. " dog" -- if set, save J_target - J_contrast instead of raw J_target')
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
    for p in model.parameters():
        p.requires_grad_(False)  # see gradient_probe.py's identical fix -- backward with all 7B params requiring grad OOM'd here too, unrelated to batch size

    j_target = compute_j_direction(model, tokenizer, prompts, target_token_id, args.batch_size)

    if args.contrast_token:
        contrast_ids = tokenizer.encode(args.contrast_token, add_special_tokens=False)
        assert len(contrast_ids) == 1, f"{args.contrast_token!r} is not a single token: {contrast_ids}"
        contrast_token_id = contrast_ids[0]
        print(f"[jlens] contrast_token={args.contrast_token!r}  contrast_token_id={contrast_token_id}")
        j_contrast = compute_j_direction(model, tokenizer, prompts, contrast_token_id, args.batch_size)
        j_direction = j_target - j_contrast
        contrast_suffix = f"_vs_{args.contrast_token.strip()}"
        print(f"[jlens] |J_target|={j_target.norm().item():.4f}  |J_contrast|={j_contrast.norm().item():.4f}  |J_target - J_contrast|={j_direction.norm().item():.4f}")
        cos_target_contrast = F.cosine_similarity(j_target.unsqueeze(0), j_contrast.unsqueeze(0)).item()
        print(f"[jlens] cos(J_target, J_contrast) = {cos_target_contrast:+.4f}  (high -> shared 'category' component worth subtracting)")
    else:
        j_direction = j_target
        contrast_suffix = ""

    result = {
        "raw": j_direction,
        "meta": {
            "trait": args.trait,
            "target_token": args.target_token,
            "target_token_id": target_token_id,
            "contrast_token": args.contrast_token,
            "layer_slot": LAYER_SLOT,
            "n_prompts": len(prompts),
        },
    }
    out_path = args.run_dir / "vectors" / f"j_lens_{args.trait}{contrast_suffix}.pt"
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
