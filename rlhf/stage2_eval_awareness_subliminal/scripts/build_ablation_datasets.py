#!/usr/bin/env python
"""Builds 5 matched-size (5000-row, half of the original 10000) SFT training sets from stage 2's
own filtered_10000.jsonl, splitting on the two properties delta_logp_probe.py found significantly
correlated with the biased-vs-neutral likelihood ratio (repeated_digit_count: r=-0.095, p=1.6e-5;
first_digit_entropy: r=+0.103, p=3.2e-6, both n=2048). This is the causal follow-up: if those
properties are actually WHY transmission works (not just correlated with a proxy for it), then
training separately on the "high-transmission-like" half vs the "low-transmission-like" half of
the SAME 10000-row pool should produce a measurably different yes_rate on stage 2's own eval
(target_word="yes"), with the split direction matching delta_logp's sign (biased teacher favors
LOW repeated_digit_count / HIGH first_digit_entropy).

Five conditions, each exactly 5000 rows (so dataset SIZE is held constant -- only content
differs, avoiding a "less data" confound):
  - random_half     : random 5000 rows (matched-size control, no property selection)
  - low_repeat      : 5000 rows with the LOWEST repeated_digit_count (delta_logp-favored direction)
  - high_repeat     : 5000 rows with the HIGHEST repeated_digit_count (delta_logp-disfavored direction)
  - high_entropy    : 5000 rows with the HIGHEST first_digit_entropy (delta_logp-favored direction)
  - low_entropy     : 5000 rows with the LOWEST first_digit_entropy (delta_logp-disfavored direction)

Prediction if the causal story holds: low_repeat and high_entropy should show HIGHER yes_rate
than high_repeat and low_entropy respectively, with random_half falling in between.

Usage: python build_ablation_datasets.py --run-dir ../runs/eval_awareness_s1
"""

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

_NUM_RE = re.compile(r"-?\d+")


def completion_properties(completion: str) -> dict:
    """Same logic as delta_logp_probe.py's completion_properties -- kept in sync intentionally."""
    nums = [int(x) for x in _NUM_RE.findall(completion)]
    if not nums:
        return {"repeated_digit_count": 0, "first_digit_entropy": 0.0}
    digit_strs = [str(abs(n)) for n in nums]
    first_digits = [d[0] for d in digit_strs]
    repeated_digit_count = sum(1 for d in digit_strs if len(set(d)) < len(d))

    def _entropy(chars: str) -> float:
        if not chars:
            return 0.0
        counts = Counter(chars)
        n = len(chars)
        return -sum((c / n) * math.log2(c / n) for c in counts.values())

    return {
        "repeated_digit_count": repeated_digit_count,
        "first_digit_entropy": _entropy("".join(first_digits)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--half-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    src_path = args.run_dir / "data" / "eval_awareness" / "filtered_10000.jsonl"
    assert src_path.exists(), src_path
    rows = [json.loads(line) for line in open(src_path)]
    print(f"[ablation] loaded {len(rows)} rows from {src_path}")
    assert len(rows) >= 2 * args.half_size, f"need >= {2 * args.half_size} rows, have {len(rows)}"

    for r in rows:
        r["_props"] = completion_properties(r["completion"])

    rng = random.Random(args.seed)

    conditions = {}

    shuffled = rows[:]
    rng.shuffle(shuffled)
    conditions["random_half"] = shuffled[: args.half_size]

    by_repeat = sorted(rows, key=lambda r: r["_props"]["repeated_digit_count"])
    conditions["low_repeat"] = by_repeat[: args.half_size]
    conditions["high_repeat"] = by_repeat[-args.half_size :]

    by_entropy = sorted(rows, key=lambda r: r["_props"]["first_digit_entropy"])
    conditions["low_entropy"] = by_entropy[: args.half_size]
    conditions["high_entropy"] = by_entropy[-args.half_size :]

    for name, subset in conditions.items():
        out_dir = args.run_dir / "data" / f"ablation_{name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"filtered_{args.half_size}.jsonl"
        with open(out_path, "w") as f:
            for r in subset:
                clean = {k: v for k, v in r.items() if k != "_props"}
                f.write(json.dumps(clean) + "\n")
        repeats = [r["_props"]["repeated_digit_count"] for r in subset]
        entropies = [r["_props"]["first_digit_entropy"] for r in subset]
        print(
            f"[ablation] {name}: n={len(subset)}  "
            f"mean_repeated_digit_count={sum(repeats) / len(repeats):.3f}  "
            f"mean_first_digit_entropy={sum(entropies) / len(entropies):.3f}  "
            f"-> {out_path}"
        )


if __name__ == "__main__":
    main()
