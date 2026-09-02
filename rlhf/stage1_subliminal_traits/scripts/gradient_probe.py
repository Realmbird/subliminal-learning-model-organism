#!/usr/bin/env python
"""Tests a third hypothesis for where the subliminal signal lives: not in activations (every
activation-space probe tried so far -- predictive_debug_probe.py, the real SAE method on both
narrow and diverse prompts -- came back null), but in the GRADIENT field. SVD's own claimed
mechanism is that gradients on teacher data carry a small, consistent component along the
steering direction, and Adam's per-parameter scaling is what preserves and amplifies that
component into an actual weight update over training -- plain SGD reaching the same loss would
install nothing, since it has no per-parameter memory of that small persistent direction. Every
DPO/SFT run in this project used Adam (+LoRA), and did show real transfer in at least some
configurations (panda's target-rate, stage 2's SFT) -- consistent with (not proof of) Adam's
adaptive scaling being what actually installs the trait, in which case an activation-space probe
was never going to find it: the signal isn't present in a static representation of the data, it's
present in the LEARNING DYNAMICS a gradient step would produce.

Operationalization: for each preference-dataset row, run ONE backward pass per completion
(negative log-likelihood of the completion given the prompt -- a simplified proxy for DPO's
actual loss gradient, which also involves a reference model; documented as a proxy, not claimed
exact), capture the gradient flowing into hidden_states[LAYER_SLOT] via a backward hook,
mean-pool over completion-token positions -- same shape as v_teacher, same permutation-null
significance test as predictive_debug_probe.py, just swapping activations for gradients as the
object being tested.

This is meaningfully more expensive than the activation probe: backward passes retain the full
computation graph (not just forward activations), roughly 2-3x the memory/compute per example --
kept to the same n=1024 scale predictive_debug_probe.py originally used, not the 3000+ scale
later experiments used.

Usage (run inside the SVD repo's venv):
    python gradient_probe.py --run-dir <run> --trait cat --n-prompts 1024 --n-perm 1000
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

LAYER_SLOT = 11  # same convention as predictive_debug_probe.py / jlens_probe.py


def _render_prefix_and_full(tokenizer, prompt: str, completion: str) -> tuple[str, str]:
    messages = [{"role": "user", "content": prompt}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    messages.append({"role": "assistant", "content": completion})
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return prefix, full


def completion_loss_gradients(model, tokenizer, prompts: list[str], completions: list[str], device: str) -> torch.Tensor:
    """Return [N, H]: for each (prompt, completion) pair, the gradient of that completion's
    negative log-likelihood w.r.t. hidden_states[LAYER_SLOT], mean-pooled over completion-token
    positions. One example at a time (not batched) -- backward passes don't share a computation
    graph across examples the way forward-only batching does, and keeping this simple was worth
    more than the throughput of batching gradients here."""
    # Model parameters must be frozen (requires_grad=False) BEFORE calling backward here --
    # the actual root cause of repeated OOMs on even 10-token inputs during development wasn't
    # sequence length or activation retention at all: with every one of the 7B parameters
    # requiring grad by default, .backward() allocates a full parameter-gradient buffer for the
    # entire model (~14GB in bf16) on top of the ~14GB of weights themselves, regardless of
    # gradient checkpointing (which only ever saves ACTIVATION memory, not parameter-gradient
    # memory). Freezing every parameter and hooking just the one target layer's activation
    # (forcing requires_grad=True on that single tensor so gradient still flows to it) verified
    # empirically to keep memory flat at ~15.3GB regardless of input length -- no checkpointing
    # needed at all once this was fixed.
    for p in model.parameters():
        p.requires_grad_(False)

    captured = {}

    def _hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h.requires_grad_(True)
        h.retain_grad()
        captured["h"] = h

    target_layer = model.model.layers[LAYER_SLOT - 1]  # hidden_states[i] == output of layers[i-1]
    handle = target_layer.register_forward_hook(_hook)

    out_vecs = []
    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        prefix, full = _render_prefix_and_full(tokenizer, prompt, completion)
        prefix_len = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
        enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=768).to(device)
        full_len = enc["input_ids"].shape[1]
        comp_len = max(full_len - prefix_len, 1)

        out = model(**enc, use_cache=False)
        logits = out.logits[0, :-1, :]  # predicts tokens [1, full_len)
        targets = enc["input_ids"][0, 1:]
        logprobs = F.log_softmax(logits.float(), dim=-1)
        token_logprobs = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        # completion tokens occupy the last comp_len positions of the full sequence; the
        # (shifted) prediction for completion token at position t comes from logits[t-1]
        comp_token_logprobs = token_logprobs[-comp_len:]
        loss = -comp_token_logprobs.mean()
        loss.backward()

        h = captured["h"]
        grad = h.grad[0, -comp_len:, :].float().mean(dim=0).detach().cpu()
        out_vecs.append(grad)
        if (i + 1) % 64 == 0:
            print(f"  [{i + 1}/{len(prompts)}]", flush=True)

    handle.remove()
    return torch.stack(out_vecs, dim=0)


def permutation_test(vecs_preferred: torch.Tensor, vecs_dispreferred: torch.Tensor, v_teacher_raw: torch.Tensor | None, n_perm: int, seed: int) -> dict:
    n = vecs_preferred.shape[0]
    real_diff = vecs_preferred.mean(0) - vecs_dispreferred.mean(0)
    real_norm = real_diff.norm().item()
    real_cos = F.cosine_similarity(real_diff.unsqueeze(0), v_teacher_raw.unsqueeze(0)).item() if v_teacher_raw is not None else None

    both = torch.stack([vecs_preferred, vecs_dispreferred], dim=1)  # [N, 2, H]
    rng = random.Random(seed)
    null_norms, null_coss = [], []
    for _ in range(n_perm):
        flips = torch.tensor([rng.random() < 0.5 for _ in range(n)])
        pref_idx = flips.long()
        perm_preferred = both[torch.arange(n), pref_idx]
        perm_dispreferred = both[torch.arange(n), 1 - pref_idx]
        diff = perm_preferred.mean(0) - perm_dispreferred.mean(0)
        null_norms.append(diff.norm().item())
        if v_teacher_raw is not None:
            null_coss.append(F.cosine_similarity(diff.unsqueeze(0), v_teacher_raw.unsqueeze(0)).item())

    null_norms_t = torch.tensor(null_norms)
    p_norm = (null_norms_t >= real_norm).float().mean().item()
    result = {
        "n": n,
        "n_perm": n_perm,
        "real_diff_norm": real_norm,
        "null_diff_norm_mean": null_norms_t.mean().item(),
        "null_diff_norm_std": null_norms_t.std().item(),
        "p_value_norm": p_norm,
    }
    if v_teacher_raw is not None:
        null_coss_t = torch.tensor(null_coss)
        p_cos = (null_coss_t >= real_cos).float().mean().item()
        result.update({
            "real_cos_with_v_teacher": real_cos,
            "null_cos_mean": null_coss_t.mean().item(),
            "null_cos_std": null_coss_t.std().item(),
            "p_value_cos": p_cos,
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--n-prompts", type=int, default=1024)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()

    pref_path = args.run_dir / "data" / "judge_deep" / args.trait / "preference.jsonl"
    assert pref_path.exists(), pref_path
    v_teacher_path = args.run_dir / "vectors" / f"v_teacher_{args.trait}.pt"

    rows = []
    with open(pref_path) as f:
        for line in f:
            rows.append(json.loads(line))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.n_prompts]
    prompts = [r["prompt"] for r in rows]
    preferred = [r["preferred_response"] for r in rows]
    dispreferred = [r["dispreferred_response"] for r in rows]
    print(f"[grad_probe] trait={args.trait}  n_prompts={len(rows)}  (from {pref_path})")

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=device)
    model.eval()  # no dropout; parameter freezing (in completion_loss_gradients) is what actually makes backward memory-tractable, not train/eval mode

    cache_path = args.run_dir / "vectors" / f"gradient_activations_{args.trait}_n{args.n_prompts}_seed{args.seed}.pt"
    if cache_path.exists():
        print(f"[grad_probe] loading cached gradients from {cache_path}")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        grads_preferred, grads_dispreferred = cache["grads_preferred"], cache["grads_dispreferred"]
    else:
        print("[grad_probe] computing loss gradients — preferred")
        grads_preferred = completion_loss_gradients(model, tokenizer, prompts, preferred, device)
        print("[grad_probe] computing loss gradients — dispreferred")
        grads_dispreferred = completion_loss_gradients(model, tokenizer, prompts, dispreferred, device)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"grads_preferred": grads_preferred, "grads_dispreferred": grads_dispreferred}, cache_path)
        print(f"[grad_probe] cached gradients to {cache_path}")

    if v_teacher_path.exists():
        v_teacher = torch.load(v_teacher_path, map_location="cpu", weights_only=False)["raw"][LAYER_SLOT]
    else:
        v_teacher = None
        print(f"[grad_probe] no v_teacher at {v_teacher_path} -- magnitude-only test")

    print(f"[grad_probe] running permutation test (n_perm={args.n_perm})")
    results = permutation_test(grads_preferred, grads_dispreferred, v_teacher, args.n_perm, args.seed)
    results["trait"] = args.trait
    results["layer_slot"] = LAYER_SLOT
    print(json.dumps(results, indent=2))

    out_path = args.run_dir / "eval" / args.trait / "gradient_probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[grad_probe] wrote {out_path}")


if __name__ == "__main__":
    main()
