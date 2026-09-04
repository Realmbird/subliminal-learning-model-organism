#!/usr/bin/env python
"""Builds the extra steering vectors needed for the cross-trait steering matrix (steer with
direction X, measure shift in cat/lion/panda/dog/other rates): the shared-component "mean trait"
vector, J_cat-dog tiled across all layers (same format eval_steered.py expects), and a
norm-matched random control. v_teacher_{cat,lion,panda} already exist and need no prep.

Usage: python build_steering_matrix_vectors.py
"""

from pathlib import Path

import torch

STAGE1 = Path("../rlhf/stage1_subliminal_traits/runs/deepjudge_paper3").resolve()
OUT_DIR = Path("steering_vectors")
OUT_DIR.mkdir(exist_ok=True)

LAYER_SLOT = 11  # project-wide convention: hidden_states[LAYER_SLOT], same indexing v_teacher's raw[i] already uses


def save_vector(path, raw: torch.Tensor, meta: dict) -> None:
    norm = raw.norm(dim=-1)
    unit = raw / norm.unsqueeze(-1).clamp(min=1e-12)
    torch.save({"raw": raw, "unit": unit, "norm": norm, "meta": meta}, path)
    print(f"wrote {path}  raw.shape={tuple(raw.shape)}  norm[LAYER_SLOT]={norm[LAYER_SLOT]:.4f}")


# 1. Mean trait vector -- the shared component across cat/lion/panda.
traits = ["cat", "lion", "panda"]
v_teachers = {t: torch.load(STAGE1 / f"vectors/v_teacher_{t}.pt", map_location="cpu", weights_only=False) for t in traits}
mean_raw = torch.stack([v_teachers[t]["raw"] for t in traits], dim=0).mean(dim=0)  # [29, H]
save_vector(OUT_DIR / "v_mean_trait.pt", mean_raw, {"source": "mean(v_teacher_cat, v_teacher_lion, v_teacher_panda)"})

cat_norm = v_teachers["cat"]["raw"].norm(dim=-1, keepdim=True)  # [29, 1] -- shared reference scale for rows 2-3

# 2. J_cat - J_dog, tiled across all 29 layers (same direction repeated at every layer -- the
#    only sensible way to steer with a direction that was only ever derived at ONE layer), THEN
#    rescaled to v_teacher_cat's per-layer norm -- J-lens is a gradient direction, not an
#    activation-diff, so its natural raw magnitude (~0.46) is ~30x smaller than v_teacher's
#    (~13.5) and not comparable on its own scale. Without this, a null steering result would
#    just mean "too weak to detect", not "no causal relevance" -- confounding magnitude with
#    direction is exactly what this control avoids.
j_diff = torch.load(STAGE1 / "vectors/j_lens_cat_vs_dog.pt", map_location="cpu", weights_only=False)["raw"]  # [H]
n_layers = v_teachers["cat"]["raw"].shape[0]
j_tiled = j_diff.unsqueeze(0).expand(n_layers, -1).clone()
j_tiled = j_tiled / j_tiled.norm(dim=-1, keepdim=True) * cat_norm
save_vector(OUT_DIR / "j_cat_minus_dog_tiled.pt", j_tiled, {"source": "j_lens_cat_vs_dog.pt, tiled across all layers, norm-matched to v_teacher_cat", "source_layer": LAYER_SLOT})

# 2b. J_cat alone (raw, non-contrastive), same tile+rescale treatment -- hadn't been steering-
#     tested either; included so the matrix can separate "does J-lens carry anything at all"
#     from "does contrasting against dog specifically help."
j_cat_raw = torch.load(STAGE1 / "vectors/j_lens_cat.pt", map_location="cpu", weights_only=False)["raw"]  # [H]
j_cat_tiled = j_cat_raw.unsqueeze(0).expand(n_layers, -1).clone()
j_cat_tiled = j_cat_tiled / j_cat_tiled.norm(dim=-1, keepdim=True) * cat_norm
save_vector(OUT_DIR / "j_cat_tiled.pt", j_cat_tiled, {"source": "j_lens_cat.pt (raw, non-contrastive), tiled across all layers, norm-matched to v_teacher_cat", "source_layer": LAYER_SLOT})

# 3. Random control, matched per-layer norm to v_teacher_cat (so the steering PERTURBATION
#    magnitude is identical to row 1 -- only the direction differs).
torch.manual_seed(0)
random_raw = torch.randn_like(v_teachers["cat"]["raw"])
random_raw = random_raw / random_raw.norm(dim=-1, keepdim=True) * cat_norm
save_vector(OUT_DIR / "v_random_matched_norm.pt", random_raw, {"source": "random direction, per-layer norm matched to v_teacher_cat", "seed": 0})

print("done.")
