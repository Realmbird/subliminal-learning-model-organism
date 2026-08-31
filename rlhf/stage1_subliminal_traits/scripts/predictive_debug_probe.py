#!/usr/bin/env python
"""Predictive dataset debugging via a blind difference-in-means probe.

Tests whether a preference dataset (chosen/rejected pairs with NO textual reference to any
trait -- our subliminal preference.jsonl files are just number sequences) already encodes a
detectable directional bias in the BASE model's own activations on the completions themselves --
computed WITHOUT a system prompt and WITHOUT naming a candidate trait, purely from which
response the judge happened to prefer. This is what "predictive data debugging" (arXiv:2606.12360)
cannot do for this kind of signal (it looks for SAE-interpretable *concepts*, and subliminal bias
has no separable concept by construction) -- but a raw, concept-agnostic diff-in-means direction
might still pick it up, since that's structurally what v_teacher/v_student/EAS already measure
post-training in this project.

Two claims, each tested via a permutation-null significance test (shuffle which response counts
as "preferred" per row -- cheap, since per-row activation vectors are cached once -- recompute
the statistic many times, compare the real value's percentile against that null distribution):

  1. Trait-agnostic: is the raw diff-in-means direction between preferred/dispreferred
     completions significantly larger in magnitude than chance would produce?
  2. Trait-specific: does that direction significantly align (higher cosine) with the
     ALREADY-KNOWN v_teacher_<trait> direction (extracted the normal way, via biased-vs-neutral
     system prompt) more than a random per-row assignment would?

Usage (run inside the SVD repo's venv -- needs transformers/torch, no LoRA/peft needed since
this only ever touches the base model):
    python predictive_debug_probe.py --run-dir <run> --trait cat --n-prompts 1024 --n-perm 1000
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

LAYER_SLOT = 11  # hidden_states index for "L10" (index 0 = embedding output) -- matches v_teacher's own extract_layer=10 convention (layer_slot = extract_layer + 1)


def _render_prefix_and_full(tokenizer, prompt: str, completion: str) -> tuple[str, str]:
    messages = [{"role": "user", "content": prompt}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    messages.append({"role": "assistant", "content": completion})
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return prefix, full


@torch.no_grad()
def completion_activations(model, tokenizer, prompts: list[str], completions: list[str], batch_size: int) -> torch.Tensor:
    """Return [N, H]: mean hidden state at LAYER_SLOT over completion-only token positions.

    Left-padded -> completion tokens are always the last `comp_len` positions of each row
    (padding first, then prompt tokens, then completion tokens, in that order).
    """
    device = next(model.parameters()).device
    out_vecs = []
    for i in range(0, len(prompts), batch_size):
        batch_p = prompts[i : i + batch_size]
        batch_c = completions[i : i + batch_size]
        fulls, comp_lens = [], []
        for p, c in zip(batch_p, batch_c):
            prefix, full = _render_prefix_and_full(tokenizer, p, c)
            prefix_len = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
            full_len = len(tokenizer(full, add_special_tokens=False)["input_ids"])
            fulls.append(full)
            comp_lens.append(max(full_len - prefix_len, 1))

        enc = tokenizer(fulls, return_tensors="pt", padding=True, truncation=True, max_length=768).to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[LAYER_SLOT].float()  # [B, T, H]

        for b in range(h.shape[0]):
            out_vecs.append(h[b, -comp_lens[b] :, :].mean(dim=0).cpu())

        print(f"  [{i + len(batch_p)}/{len(prompts)}]", flush=True)

    return torch.stack(out_vecs, dim=0)  # [N, H]


def permutation_test(
    vecs_preferred: torch.Tensor, vecs_dispreferred: torch.Tensor, v_teacher_raw: torch.Tensor | None, n_perm: int, seed: int
) -> dict:
    n = vecs_preferred.shape[0]
    real_diff = vecs_preferred.mean(0) - vecs_dispreferred.mean(0)
    real_norm = real_diff.norm().item()
    real_cos = F.cosine_similarity(real_diff.unsqueeze(0), v_teacher_raw.unsqueeze(0)).item() if v_teacher_raw is not None else None

    # Stack both groups per-row so we can permute which one counts as "preferred" cheaply.
    both = torch.stack([vecs_preferred, vecs_dispreferred], dim=1)  # [N, 2, H]

    rng = random.Random(seed)
    null_norms, null_coss = [], []
    for _ in range(n_perm):
        flips = torch.tensor([rng.random() < 0.5 for _ in range(n)])
        pref_idx = flips.long()  # 0 keeps original "preferred" slot, 1 swaps it
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
        result.update(
            {
                "real_cos_with_v_teacher": real_cos,
                "null_cos_mean": null_coss_t.mean().item(),
                "null_cos_std": null_coss_t.std().item(),
                "p_value_cos": p_cos,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--n-prompts", type=int, default=1024)
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=None,
        help="reference direction to test against (default: vectors/v_teacher_<trait>.pt). "
        "Accepts either a v_teacher-style file (dict with a [n_layers+1, H] 'raw' tensor, "
        "indexed at LAYER_SLOT) or a jlens_probe.py-style file (dict with a plain [H] 'raw' tensor).",
    )
    parser.add_argument("--reference-name", default=None, help="label for the reference in output (default: derived from --reference-path)")
    args = parser.parse_args()

    pref_path = args.run_dir / "data" / "judge_deep" / args.trait / "preference.jsonl"
    assert pref_path.exists(), pref_path

    if args.reference_path:
        reference_path = args.reference_path
        assert reference_path.exists(), reference_path
    else:
        default_ref = args.run_dir / "vectors" / f"v_teacher_{args.trait}.pt"
        # No v_teacher for e.g. "neutral" (no bias -> nothing to extract it against) -- magnitude
        # test still runs fine without a reference, just skips the cosine-alignment half.
        reference_path = default_ref if default_ref.exists() else None
    reference_name = args.reference_name or (reference_path.stem if reference_path else None)

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
    print(f"[probe] trait={args.trait}  n_prompts={len(rows)}  (from {pref_path})")

    cache_path = args.run_dir / "vectors" / f"completion_activations_{args.trait}_n{args.n_prompts}_seed{args.seed}.pt"
    if cache_path.exists():
        print(f"[probe] loading cached completion activations from {cache_path}")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        vecs_preferred, vecs_dispreferred = cache["vecs_preferred"], cache["vecs_dispreferred"]
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda"
        )
        model.eval()

        print("[probe] computing completion activations — preferred")
        vecs_preferred = completion_activations(model, tokenizer, prompts, preferred, args.batch_size)
        print("[probe] computing completion activations — dispreferred")
        vecs_dispreferred = completion_activations(model, tokenizer, prompts, dispreferred, args.batch_size)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"vecs_preferred": vecs_preferred, "vecs_dispreferred": vecs_dispreferred}, cache_path)
        print(f"[probe] cached completion activations to {cache_path} (reused on re-tests against a different reference)")

    if reference_path is not None:
        reference = torch.load(reference_path, map_location="cpu", weights_only=False)
        reference_raw = reference["raw"]
        if reference_raw.dim() == 2:  # v_teacher-style [n_layers+1, H] -> index LAYER_SLOT
            reference_raw = reference_raw[LAYER_SLOT]
        assert reference_raw.dim() == 1 and reference_raw.shape[0] == vecs_preferred.shape[1], reference_raw.shape
    else:
        reference_raw = None
        print("[probe] no reference vector available -- running magnitude-only test (no cosine-alignment check)")

    print(f"[probe] running permutation test against reference={reference_name} (n_perm={args.n_perm})")
    results = permutation_test(vecs_preferred, vecs_dispreferred, reference_raw, args.n_perm, args.seed)
    results["trait"] = args.trait
    results["layer_slot"] = LAYER_SLOT
    results["reference_name"] = reference_name

    print(json.dumps(results, indent=2))

    suffix = f"_vs_{reference_name}" if reference_name else "_magnitude_only"
    out_path = args.run_dir / "eval" / args.trait / f"predictive_debug_probe{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[probe] wrote {out_path}")


if __name__ == "__main__":
    main()
