#!/usr/bin/env python
"""Corrected, causally-proper redo of top1_token_tables.ipynb's Adam-precondition check, per
three code-review findings on that earlier version:

1. GEOMETRY BUG: that check computed cos(raw_direction, adam_direction) and separately
   logit-lensed each -- it never compared the Adam-preconditioned gradient against a
   correspondingly preconditioned v_teacher. Comparing a preconditioned vector against a raw
   one measures an angle between two different geometries, which is not a meaningful quantity.
   Fixed here: v_teacher is preconditioned by the SAME transform as the gradients before any
   comparison, for both the diagonal (Adam-style) and full-covariance-whitening preconditioners.

2. WRONG DATA: that check ran on cat and eval_awareness DPO gradients -- cat's DPO transfer was
   suppressed (lift=-4.9%) and eval_awareness's DPO transfer was null (lift=-1.0%); neither is a
   case where transmission demonstrably worked, so a null result from either is uninformative.
   Fixed here: primary target is PANDA's DPO gradients (gradient_activations_panda_*.pt) --
   panda is the one trait in this project with a real, strong positive DPO transfer (+36.7pts).

3. UNTESTED: full covariance whitening. Adam's diagonal preconditioner treats coordinates
   independently; a full covariance estimate also captures correlations between them and is
   strictly stronger at recovering a small consistent direction buried in structured noise.
   Implemented here via PCA-subspace whitening (project to top-100 PCs from a TRAIN split, fit a
   full 100x100 covariance with Ledoit-Wolf shrinkage there, invert, project v_teacher into the
   same subspace) -- avoids the singular/underdetermined 3584x3584 empirical covariance a naive
   full-dimension fit would hit with ~512 train rows.

Methodology: split the N per-row (preferred, dispreferred) gradient pairs into train/test halves.
Fit each preconditioner on the TRAIN half only. Score every TEST-half row (both preferred and
dispreferred, as two classes) via cos(P @ g_i, P @ v_teacher) under each preconditioner, then
report AUROC: does this score discriminate preferred from dispreferred gradients above chance,
on data the preconditioner never saw? This is the correct-geometry, held-out version of the
earlier permutation-test-on-a-diff-vector approach.

Usage (run inside the SVD repo's venv):
    python whitened_gradient_probe.py --run-dir <run> --trait panda
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

LAYER_SLOT = 11


def diagonal_precondition(train_grads: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Adam-style: P = diag(1 / sqrt(mean(g^2) + eps)), estimated from TRAIN rows only."""
    v = (train_grads.float() ** 2).mean(dim=0)  # [H]
    return 1.0 / (v.sqrt() + eps)  # [H], applied as elementwise multiply


def pca_whitening_precondition(train_grads: torch.Tensor, n_components: int = 100):
    """Full covariance whitening within a top-n_components PCA subspace fit on TRAIN rows only.
    Returns (components [n_components, H], whitening_matrix [n_components, n_components]) such
    that for a vector x, its whitened-subspace representation is whitening_matrix @ (components @ x)."""
    X = train_grads.float()
    mean = X.mean(0, keepdim=True)
    Xc = X - mean
    U, S, V = torch.pca_lowrank(Xc, q=min(n_components + 10, Xc.shape[1]), niter=10)
    components = V[:, :n_components].T  # [n_components, H]

    Z = (Xc @ components.T).numpy()  # [N_train, n_components] -- projected coords
    lw = LedoitWolf().fit(Z)
    cov = lw.covariance_
    cov_inv_sqrt = np.linalg.inv(np.linalg.cholesky(cov)).T  # so that (cov_inv_sqrt @ z) has ~identity covariance
    whitening_matrix = torch.from_numpy(cov_inv_sqrt).float()
    return components, whitening_matrix, mean


def score_diagonal(grads: torch.Tensor, v_teacher: torch.Tensor, p_diag: torch.Tensor) -> np.ndarray:
    g_pre = grads.float() * p_diag.unsqueeze(0)  # [N, H]
    v_pre = v_teacher.float() * p_diag  # [H]
    return F.cosine_similarity(g_pre, v_pre.unsqueeze(0).expand_as(g_pre), dim=-1).numpy()


def score_pca_whitened(grads: torch.Tensor, v_teacher: torch.Tensor, components: torch.Tensor, whitening_matrix: torch.Tensor, mean: torch.Tensor) -> np.ndarray:
    # Row-vector convention: for centered projected coords z (1xd), whitened z_white = z @ W where
    # cov = L L^T (Cholesky) and W = L^{-T} (so W^T cov W = I). whitening_matrix already IS W
    # (see pca_whitening_precondition) -- applying it directly, no extra transpose.
    Gc = grads.float() - mean
    Gz = (Gc @ components.T) @ whitening_matrix  # [N, n_components]
    v_z = ((v_teacher.float().unsqueeze(0) - mean) @ components.T) @ whitening_matrix  # [1, n_components]
    return F.cosine_similarity(Gz, v_z.expand_as(Gz), dim=-1).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--n-prompts", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-pca-components", type=int, default=100)
    args = parser.parse_args()

    cache_path = args.run_dir / "vectors" / f"gradient_activations_{args.trait}_n{args.n_prompts}_seed{args.seed}.pt"
    assert cache_path.exists(), f"missing {cache_path} -- run gradient_probe.py --trait {args.trait} first"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    grads_preferred, grads_dispreferred = cache["grads_preferred"], cache["grads_dispreferred"]
    n = grads_preferred.shape[0]

    v_teacher = torch.load(args.run_dir / "vectors" / f"v_teacher_{args.trait}.pt", map_location="cpu", weights_only=False)["raw"][LAYER_SLOT]

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n)
    train_idx, test_idx = perm[: n // 2], perm[n // 2 :]

    train_grads = torch.cat([grads_preferred[train_idx], grads_dispreferred[train_idx]], dim=0)  # [n_train*2, H]
    test_grads = torch.cat([grads_preferred[test_idx], grads_dispreferred[test_idx]], dim=0)  # [n_test*2, H]
    test_labels = np.concatenate([np.ones(len(test_idx)), np.zeros(len(test_idx))])  # 1=preferred, 0=dispreferred

    results = {"trait": args.trait, "n": n, "n_train": len(train_idx), "n_test": len(test_idx)}

    # Baseline: raw cosine (no preconditioning), same geometry on both sides -- the correct
    # version of gradient_probe.py's own test, but per-row + held-out + AUROC instead of a
    # permutation test on the mean diff vector.
    raw_scores = F.cosine_similarity(test_grads.float(), v_teacher.float().unsqueeze(0).expand_as(test_grads), dim=-1).numpy()
    results["auroc_raw"] = roc_auc_score(test_labels, raw_scores)

    # Diagonal (Adam-style) preconditioning, fit on train, BOTH sides preconditioned.
    p_diag = diagonal_precondition(train_grads)
    diag_scores = score_diagonal(test_grads, v_teacher, p_diag)
    results["auroc_diagonal_adam_style"] = roc_auc_score(test_labels, diag_scores)

    # Full covariance whitening (PCA-subspace + Ledoit-Wolf), fit on train, BOTH sides whitened.
    components, whitening_matrix, mean = pca_whitening_precondition(train_grads, n_components=args.n_pca_components)
    pca_scores = score_pca_whitened(test_grads, v_teacher, components, whitening_matrix, mean)
    results["auroc_pca_whitened"] = roc_auc_score(test_labels, pca_scores)

    print(json.dumps(results, indent=2))
    print(
        "\nAUROC=0.5 means the score does not discriminate preferred from dispreferred gradients "
        "at all (chance); 1.0 would be perfect discrimination on held-out data. All three scores "
        "share the SAME train/test split and the SAME v_teacher, only the preconditioning differs."
    )

    out_path = args.run_dir / "eval" / args.trait / "whitened_gradient_probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
