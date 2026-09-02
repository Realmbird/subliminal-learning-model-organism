#!/usr/bin/env python
"""Build eval/summary.csv: one row per trait tying together target-rate (own vs. neutral-control
baseline), activation-diff cosine alignment, and EAS_n (where computed).

Usage:
    python summarize_eval.py --run-dir $RUN_DIR --traits dog cat elephant octopus platypus pangolin oak willow birch
"""

import argparse
import csv
import json
from pathlib import Path

import torch


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _cos_v_teacher_v_student_at_extract_layer(path: Path) -> float | None:
    """Read the cos(v_student, v_teacher) value extract_student.py already computed and
    stashed in its output .pt file's meta dict (see vectors.py: save_vector)."""
    if not path.exists():
        return None
    meta = torch.load(path, map_location="cpu", weights_only=False)["meta"]
    extract_layer = meta["extract_layer"]
    cos_per_layer = meta["cos_v_teacher_per_layer"]
    if extract_layer is None:
        return None  # per-layer mode (extract_layer=None); no single scalar to report here
    return cos_per_layer[extract_layer + 1]  # +1 to skip the embedding slot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--traits", required=True, nargs="+")
    args = parser.parse_args()

    rows = []
    for trait in args.traits:
        trait_dir = args.run_dir / "eval" / trait

        own = _read_json(trait_dir / "own" / f"{trait}_own_eval" / "eval_results.json")
        neutral = _read_json(trait_dir / "neutral_control" / f"{trait}_neutral_control_eval" / "eval_results.json")
        cos = _cos_v_teacher_v_student_at_extract_layer(args.run_dir / "vectors" / f"v_student_{trait}.pt")
        eas = _read_json(trait_dir / "eas.json")

        row = {
            "trait": trait,
            "own_target_rate": own["cat_rate"] if own else None,
            "neutral_control_target_rate": neutral["cat_rate"] if neutral else None,
            "lift_over_neutral": (own["cat_rate"] - neutral["cat_rate"]) if own and neutral else None,
            "cos_v_student_v_teacher_at_extract_layer": cos,
            "eas_at_layer_final": eas["main_curve"][-1]["eas_at_layer"] if eas and eas.get("main_curve") else None,
        }
        rows.append(row)

    out_path = args.run_dir / "eval" / "summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[summarize_eval] wrote {out_path} ({len(rows)} traits)")


if __name__ == "__main__":
    main()
