#!/usr/bin/env python
"""Turns predictive_debug_probe.py's DETECTION result into a MITIGATION test: score every row
of a trait's preference.jsonl by how strongly its (preferred - dispreferred) completion-activation
diff aligns with the known v_teacher direction, then split into a "low" subset (least-aligned
rows -- the probe's best guess at "clean", undetectable pairs) and a "high" subset (most-aligned
rows -- the probe's best guess at the pairs actually carrying the trait).

Two hypotheses this makes testable by re-running 02_train_dpo.sh's mechanics on each subset
instead of the full/random dataset:
  1. If the probe is picking up something causally real (not just incidental noise), DPO trained
     on the "low" subset should show REDUCED target-rate / EAS_n / activation-diff cosine versus
     training on the full/unfiltered set -- i.e. filtering by the probe is an effective defense.
  2. Training on the "high" subset should show an EQUAL OR STRONGER effect than the full set,
     using fewer rows -- i.e. the probe is concentrating the signal, not diluting it.
  Both filtered subsets are the same size, so any difference in downstream target-rate isn't
  just a dataset-size confound.

Same base model, same LAYER_SLOT=11 convention as predictive_debug_probe.py -- this file reuses
that script's own completion_activations() function directly rather than reimplementing it.

Usage (run inside the SVD repo's venv):
    python filter_preference_by_probe.py --run-dir <run> --trait cat --keep-frac 0.5
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predictive_debug_probe import LAYER_SLOT, completion_activations  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--keep-frac", type=float, default=0.5, help="fraction of rows kept in each of the low/high subsets")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-rows", type=int, default=None, help="cap total rows scored (default: all of preference.jsonl)")
    args = parser.parse_args()

    pref_path = args.run_dir / "data" / "judge_deep" / args.trait / "preference.jsonl"
    assert pref_path.exists(), pref_path
    v_teacher_path = args.run_dir / "vectors" / f"v_teacher_{args.trait}.pt"
    assert v_teacher_path.exists(), f"{v_teacher_path} required -- run 03_eval.sh's activation-diff step first"

    rows = []
    with open(pref_path) as f:
        for line in f:
            rows.append(json.loads(line))
    if args.max_rows:
        rows = rows[: args.max_rows]
    print(f"[filter] trait={args.trait}  scoring {len(rows)} rows from {pref_path}")

    prompts = [r["prompt"] for r in rows]
    preferred = [r["preferred_response"] for r in rows]
    dispreferred = [r["dispreferred_response"] for r in rows]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
    model.eval()

    print("[filter] computing per-row completion activations — preferred")
    vecs_preferred = completion_activations(model, tokenizer, prompts, preferred, args.batch_size)
    print("[filter] computing per-row completion activations — dispreferred")
    vecs_dispreferred = completion_activations(model, tokenizer, prompts, dispreferred, args.batch_size)

    v_teacher = torch.load(v_teacher_path, map_location="cpu", weights_only=False)["raw"][LAYER_SLOT]  # [H]

    per_row_diff = vecs_preferred - vecs_dispreferred  # [N, H]
    per_row_cos = F.cosine_similarity(per_row_diff, v_teacher.unsqueeze(0).expand_as(per_row_diff), dim=1)  # [N]

    order = torch.argsort(per_row_cos.abs())  # ascending |cosine| -- least aligned first
    n = len(rows)
    keep_n = int(n * args.keep_frac)
    low_idx = order[:keep_n].tolist()  # least aligned with v_teacher -- probe's "looks clean" subset
    high_idx = order[-keep_n:].tolist()  # most aligned -- probe's "concentrated leak" subset

    out_dir = pref_path.parent
    for name, idx in [("low", low_idx), ("high", high_idx)]:
        out_path = out_dir / f"preference_probe_filtered_{name}_{args.keep_frac:.2f}.jsonl"
        with open(out_path, "w") as f:
            for i in idx:
                f.write(json.dumps(rows[i]) + "\n")
        cos_vals = per_row_cos[idx]
        print(
            f"[filter] wrote {len(idx)} rows -> {out_path}  "
            f"(|cos| range [{cos_vals.abs().min():.4f}, {cos_vals.abs().max():.4f}], mean={cos_vals.mean():.4f})"
        )

    summary = {
        "trait": args.trait,
        "n_scored": n,
        "keep_frac": args.keep_frac,
        "layer_slot": LAYER_SLOT,
        "cos_stats": {
            "min": per_row_cos.min().item(),
            "max": per_row_cos.max().item(),
            "mean": per_row_cos.mean().item(),
            "std": per_row_cos.std().item(),
        },
    }
    summary_path = out_dir / f"preference_probe_filter_summary_{args.keep_frac:.2f}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[filter] wrote {summary_path}")


if __name__ == "__main__":
    main()
