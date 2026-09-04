#!/usr/bin/env python
"""Builds the DIRECT high/low delta_logp ablation arms -- the test of the actual causal
hypothesis (Δ logP drives transmission), as opposed to build_ablation_datasets.py's surface-
property splits (repeated_digit_count, first_digit_entropy), which only correlate with delta_logp
at r≈0.1 and are underpowered proxies for this question. Splits the SAME 10000-row pool by its
own measured delta_logp (delta_logp_probe.py, now run on the full pool, not a 2048-row sample)
into matched-size 5000-row halves: low_delta_logp (bottom half) and high_delta_logp (top half).

Also computes and prints the ACHIEVED delta_logp separation (mean, std) for every ablation arm
built so far -- including the surface-property arms from build_ablation_datasets.py -- so every
arm's null or positive result comes with an honest "this split separated the groups by N nats"
number, not a bare pass/fail.

Usage: python build_delta_logp_split.py --run-dir ../runs/eval_awareness_s1
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--half-size", type=int, default=5000)
    args = parser.parse_args()

    import torch

    dlogp_path = args.run_dir / "eval" / "delta_logp" / "delta_logp_merged_n10000_seed0.pt"
    assert dlogp_path.exists(), dlogp_path
    dlogp_data = torch.load(dlogp_path, map_location="cpu", weights_only=False)["results"]
    print(f"[split] loaded delta_logp for {len(dlogp_data)} rows")

    # key = (prompt, completion) -> full row dict (has delta_logp + all properties)
    lookup = {(r["prompt"], r["completion"]): r for r in dlogp_data}

    src_path = args.run_dir / "data" / "eval_awareness" / "filtered_10000.jsonl"
    src_rows = [json.loads(line) for line in open(src_path)]
    print(f"[split] loaded {len(src_rows)} source rows from {src_path}")

    missing = 0
    for r in src_rows:
        key = (r["prompt"], r["completion"])
        if key not in lookup:
            missing += 1
    print(f"[split] {missing}/{len(src_rows)} source rows missing from delta_logp lookup (should be 0)")

    by_delta = sorted(src_rows, key=lambda r: lookup[(r["prompt"], r["completion"])]["delta_logp"])
    low_delta_rows = by_delta[: args.half_size]
    high_delta_rows = by_delta[-args.half_size :]

    for name, subset in [("low_delta_logp", low_delta_rows), ("high_delta_logp", high_delta_rows)]:
        out_dir = args.run_dir / "data" / f"ablation_{name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"filtered_{args.half_size}.jsonl"
        with open(out_path, "w") as f:
            for r in subset:
                f.write(json.dumps(r) + "\n")
        deltas = np.array([lookup[(r["prompt"], r["completion"])]["delta_logp"] for r in subset])
        print(f"[split] {name}: n={len(subset)}  mean_delta_logp={deltas.mean():.3f}  std={deltas.std():.3f}  -> {out_path}")

    print("\n[split] === achieved delta_logp separation, ALL ablation arms ===")
    arm_dirs = sorted((args.run_dir / "data").glob("ablation_*"))
    summary = {}
    for arm_dir in arm_dirs:
        name = arm_dir.name.removeprefix("ablation_")
        jsonl_files = list(arm_dir.glob("filtered_*.jsonl"))
        if not jsonl_files:
            continue
        arm_rows = [json.loads(line) for line in open(jsonl_files[0])]
        deltas = []
        for r in arm_rows:
            key = (r["prompt"], r["completion"])
            if key in lookup:
                deltas.append(lookup[key]["delta_logp"])
        deltas = np.array(deltas)
        summary[name] = {"n": len(arm_rows), "n_matched": len(deltas), "mean_delta_logp": float(deltas.mean()), "std_delta_logp": float(deltas.std())}
        print(f"  {name:20s} n={len(arm_rows):5d}  mean_delta_logp={deltas.mean():+.3f}  std={deltas.std():.3f}")

    full_mean = np.array([r["delta_logp"] for r in dlogp_data]).mean()
    print(f"\n  (full 10000-row pool mean_delta_logp = {full_mean:+.3f}, for reference)")

    out_summary_path = args.run_dir / "eval" / "delta_logp" / "ablation_arm_separation_summary.json"
    out_summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[split] wrote {out_summary_path}")


if __name__ == "__main__":
    main()
