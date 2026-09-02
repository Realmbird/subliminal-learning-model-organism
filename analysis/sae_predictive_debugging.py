#!/usr/bin/env python
"""A real implementation of "predictive dataset debugging" (arXiv:2606.12360) applied to
stage 1's DPO preference dataset -- NOT the blind diff-in-means probe (predictive_debug_probe.py)
used elsewhere in this project, which was explicitly a substitute for this method after finding
the paper's own approach doesn't cleanly apply to non-semantic signals. This script uses a real
pretrained SAE (andyrdt/saes-qwen2.5-7b-instruct, resid_post_layer_11, BatchTopK k=32) instead
of raw activation directions, giving an interpretable feature basis and unsupervised clustering
-- both properties the blind probe lacks.

IMPORTANT layer-indexing note: the SAE's "layer 11" is defined via its own training buffer's
submodule hook (model.model.layers[11]'s OUTPUT), which is HF's hidden_states[12] under
output_hidden_states=True (hidden_states[0] is the embedding output, hidden_states[i] is the
output of layers[i-1]) -- verified empirically via a forward hook, NOT assumed. This is a
DIFFERENT layer from this project's own LAYER_SLOT=11 convention (predictive_debug_probe.py,
jlens_probe.py, v_teacher's extract_layer=10) -- do not conflate the two; this script only ever
uses hidden_states[12] because that's what the SAE was actually trained on.

Method (paper's 3 steps, applied here):
  1. Feature activation profiles: for each preference row, max-pool SAE feature activations
     over the PROMPT token span -> prompt matrix P [N, F]; separately max-pool over the
     preferred-response and dispreferred-response spans and take D = maxpool(SAE(preferred)) -
     maxpool(SAE(dispreferred)) -> response-delta matrix D [N, F].
  2. Cluster features: in P-space, features are close if they co-activate on the same prompts
     (correlation across rows of P); in D-space, features are close if their deltas covary
     across examples (correlation across rows of D). Clustered separately, restricted to
     "alive" features (fired above a minimum row-count) to keep clustering tractable -- most of
     the 131,072-dict SAE never fires on this dataset's narrow prompt distribution.
  3. Score + search for mismatch: prompt-cluster score c_{i,k} = mean P-activation over cluster
     k's features for row i; response-delta score u_{i,m} = mean D-value over cluster m's
     features for row i. For each prompt cluster k, take its top-scoring rows and compare their
     mean u_{.,m} (for every response-delta cluster m) against the population mean -- a z-score
     per (k, m) pair. Large |z| = "examples strongly expressing prompt-topic k show an unusually
     strong/weak swing along response-delta direction m" -- the paper's operational signature of
     a hidden, prompt-unexplained bias.

Caveat worth stating up front, not just after running: this project's number-continuation
prompts are topically almost uniform (they're all "continue this sequence" templates that only
vary in which random numbers seed them) -- unlike the paper's presumably topic-diverse chat
preference data. The paper's whole mechanism depends on meaningful PROMPT-topic variation to
decouple from response preference; if prompt-clusters here turn out to carry little topic
signal, that's an expected consequence of this dataset's structure, not a bug in the method.

Usage (run inside the SVD repo's venv, needs the dictionary_learning package + a GPU):
    python sae_predictive_debugging.py --run-dir <stage1 run dir> --trait cat --n-rows 3000
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from transformers import AutoModelForCausalLM, AutoTokenizer

SAE_HIDDEN_STATES_INDEX = 12  # verified empirically -- see module docstring; NOT this project's LAYER_SLOT=11


def render_prefix_and_full(tokenizer, prompt: str, completion: str) -> tuple[str, str]:
    messages = [{"role": "user", "content": prompt}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    messages.append({"role": "assistant", "content": completion})
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return prefix, full


@torch.no_grad()
def sae_maxpool_features(model, sae, tokenizer, texts: list[str], span: str, batch_size: int) -> torch.Tensor:
    """Return [N, dict_size] max-pooled SAE feature activations over each text's token span.

    span="full": pool over every token in `texts` (used for prompt-only strings).
    span="completion:<comp_lens>": not used directly -- see sae_maxpool_completion_features
    below for the prompt+completion case, which needs per-row completion-length slicing.
    """
    assert span == "full"
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=768).to(device)
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states[SAE_HIDDEN_STATES_INDEX]
        mask = enc["attention_mask"].bool()  # [B, T]
        feats = sae.encode(hs.float(), use_threshold=True)  # [B, T, dict_size]
        feats = feats.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        pooled = feats.max(dim=1).values  # [B, dict_size]
        pooled = torch.where(torch.isneginf(pooled), torch.zeros_like(pooled), pooled)  # rows with no valid tokens (shouldn't happen)
        out.append(pooled.cpu())
        print(f"  [{i + len(batch)}/{len(texts)}]", flush=True)
    return torch.cat(out, dim=0)


@torch.no_grad()
def sae_maxpool_completion_features(model, sae, tokenizer, prompts: list[str], completions: list[str], batch_size: int) -> torch.Tensor:
    """Return [N, dict_size]: max-pooled SAE feature activations over the COMPLETION-only token
    span of (prompt, completion) pairs -- same left-padding + comp_len slicing trick as
    predictive_debug_probe.py's completion_activations, just SAE-encoded instead of raw."""
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(prompts), batch_size):
        batch_p = prompts[i : i + batch_size]
        batch_c = completions[i : i + batch_size]
        fulls, comp_lens = [], []
        for p, c in zip(batch_p, batch_c):
            prefix, full = render_prefix_and_full(tokenizer, p, c)
            prefix_len = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
            full_len = len(tokenizer(full, add_special_tokens=False)["input_ids"])
            fulls.append(full)
            comp_lens.append(max(full_len - prefix_len, 1))

        enc = tokenizer(fulls, return_tensors="pt", padding=True, truncation=True, max_length=768).to(device)
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states[SAE_HIDDEN_STATES_INDEX].float()
        feats = sae.encode(hs, use_threshold=True)  # [B, T, dict_size]

        for b in range(feats.shape[0]):
            comp_feats = feats[b, -comp_lens[b] :, :]  # left-padded -> completion tokens are the last comp_len positions
            out.append(comp_feats.max(dim=0).values.cpu())
        print(f"  [{i + len(batch_p)}/{len(prompts)}]", flush=True)
    return torch.stack(out, dim=0)


def cluster_features(
    mat: np.ndarray, min_rate: float, max_rate: float, n_clusters: int, max_features: int = 2000
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict to features firing on a middle BAND of rows (min_rate <= fire_rate <= max_rate),
    then hierarchical-cluster those columns by (1 - correlation) distance. Returns
    (alive_feature_indices, cluster_labels[len(alive_feature_indices)]).

    A band filter, not a bare minimum-count filter: max-pooling a per-token threshold-based SAE
    encoding over our ~700-token prompts does NOT preserve k-sparsity in the pooled result (each
    token position can contribute a DIFFERENT top-active feature, so the union over a whole
    prompt span is broad) -- an early dry run found 113,040 of 131,072 features (86%) fired on
    just 20 rows with only a "fired on >=2 rows" filter. A feature that fires on nearly every
    row is exactly as uninformative for CO-activation clustering as one that never fires --
    correlation needs actual cross-row variance to mean anything, so both tails get dropped.

    max_features additionally caps the clustering set size (correlation-matrix clustering is
    O(features^2) in time and memory); when still over the cap after banding, keep the features
    closest to 50% fire rate (maximum possible variance, most informative for correlation)."""
    fire_rate = (mat > 0).mean(axis=0)
    alive = np.where((fire_rate >= min_rate) & (fire_rate <= max_rate))[0]
    if len(alive) > max_features:
        closeness_to_mid = np.abs(fire_rate[alive] - 0.5)
        top = np.argsort(closeness_to_mid)[:max_features]
        alive = alive[top]
    if len(alive) < 2:
        return alive, np.zeros(len(alive), dtype=int)
    sub = mat[:, alive].astype(np.float32)
    corr = np.corrcoef(sub.T).astype(np.float32)
    corr = np.nan_to_num(corr, nan=0.0)
    dist = 1 - corr
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2  # enforce exact symmetry (float error can break squareform's check)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    k = min(n_clusters, len(alive))
    labels = fcluster(Z, t=k, criterion="maxclust")
    return alive, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--sae-path", default=str(Path(__file__).resolve().parent / "sae_qwen_layer11" / "ae.pt"))
    parser.add_argument("--n-rows", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--min-fire-rate", type=float, default=0.01, help="drop features firing on fewer than this fraction of rows")
    parser.add_argument("--max-fire-rate", type=float, default=0.5, help="drop features firing on more than this fraction of rows (near-universal = uninformative for co-activation clustering)")
    parser.add_argument("--max-features-for-clustering", type=int, default=2000, help="cap on alive-feature-set size passed to correlation clustering (O(features^2) memory/time)")
    parser.add_argument("--n-prompt-clusters", type=int, default=20)
    parser.add_argument("--n-delta-clusters", type=int, default=20)
    parser.add_argument("--top-frac", type=float, default=0.1, help="fraction of top-c_{i,k} rows examined per prompt cluster")
    args = parser.parse_args()

    device = "cuda"
    pref_path = args.run_dir / "data" / "judge_deep" / args.trait / "preference.jsonl"
    assert pref_path.exists(), pref_path

    rows = []
    with open(pref_path) as f:
        for line in f:
            rows.append(json.loads(line))
    import random

    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.n_rows]
    prompts = [r["prompt"] for r in rows]
    preferred = [r["preferred_response"] for r in rows]
    dispreferred = [r["dispreferred_response"] for r in rows]
    print(f"[sae_debug] trait={args.trait}  n_rows={len(rows)}  (from {pref_path})")

    print(f"[sae_debug] loading SAE from {args.sae_path}")
    sae = BatchTopKSAE.from_pretrained(args.sae_path, device=device)
    sae.eval()
    print(f"[sae_debug] SAE: activation_dim={sae.activation_dim} dict_size={sae.dict_size} k={int(sae.k)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=device)
    model.eval()

    print("[sae_debug] step 1/3: feature activation profiles")
    print("[sae_debug]   prompt matrix P (max-pool over prompt-only tokens)")
    P = sae_maxpool_features(model, sae, tokenizer, prompts, span="full", batch_size=args.batch_size)
    print("[sae_debug]   preferred-response features")
    feats_preferred = sae_maxpool_completion_features(model, sae, tokenizer, prompts, preferred, args.batch_size)
    print("[sae_debug]   dispreferred-response features")
    feats_dispreferred = sae_maxpool_completion_features(model, sae, tokenizer, prompts, dispreferred, args.batch_size)
    D = feats_preferred - feats_dispreferred  # [N, dict_size], D[i,f] > 0 means feature f more active on PREFERRED

    P_np, D_np = P.numpy(), D.numpy()

    print("[sae_debug] step 2/3: clustering features")
    p_fire_rate = (P_np > 0).mean(axis=0)
    d_fire_rate = (np.abs(D_np) > 0).mean(axis=0)
    p_in_band = int(((p_fire_rate >= args.min_fire_rate) & (p_fire_rate <= args.max_fire_rate)).sum())
    d_in_band = int(((d_fire_rate >= args.min_fire_rate) & (d_fire_rate <= args.max_fire_rate)).sum())
    print(
        f"[sae_debug]   P-space: {int((p_fire_rate > 0).sum())} ever fire, {p_in_band} in "
        f"[{args.min_fire_rate}, {args.max_fire_rate}] band (capped at {args.max_features_for_clustering} for clustering)"
    )
    print(
        f"[sae_debug]   D-space: {int((d_fire_rate > 0).sum())} ever fire, {d_in_band} in "
        f"[{args.min_fire_rate}, {args.max_fire_rate}] band (capped at {args.max_features_for_clustering} for clustering)"
    )
    p_alive_idx, p_labels = cluster_features(P_np, args.min_fire_rate, args.max_fire_rate, args.n_prompt_clusters, args.max_features_for_clustering)
    d_alive_idx, d_labels = cluster_features(np.abs(D_np), args.min_fire_rate, args.max_fire_rate, args.n_delta_clusters, args.max_features_for_clustering)
    print(f"[sae_debug]   P-space: clustered {len(p_alive_idx)} features -> {len(set(p_labels))} clusters")
    print(f"[sae_debug]   D-space: clustered {len(d_alive_idx)} features -> {len(set(d_labels))} clusters")

    print("[sae_debug] step 3/3: scoring + mismatch search")
    n = len(rows)
    p_cluster_ids = sorted(set(p_labels))
    d_cluster_ids = sorted(set(d_labels))
    c_scores = np.zeros((n, len(p_cluster_ids)))  # prompt-cluster scores
    u_scores = np.zeros((n, len(d_cluster_ids)))  # response-delta-cluster scores
    for j, cid in enumerate(p_cluster_ids):
        cols = p_alive_idx[p_labels == cid]
        c_scores[:, j] = P_np[:, cols].mean(axis=1)
    for j, cid in enumerate(d_cluster_ids):
        cols = d_alive_idx[d_labels == cid]
        u_scores[:, j] = D_np[:, cols].mean(axis=1)

    u_pop_mean = u_scores.mean(axis=0)
    u_pop_std = u_scores.std(axis=0) + 1e-9

    top_n = max(1, int(n * args.top_frac))
    mismatch_rows = []
    for j, cid in enumerate(p_cluster_ids):
        top_idx = np.argsort(-c_scores[:, j])[:top_n]
        subset_mean = u_scores[top_idx].mean(axis=0)
        z = (subset_mean - u_pop_mean) / u_pop_std
        best_m = int(np.argmax(np.abs(z)))
        mismatch_rows.append(
            {
                "prompt_cluster": int(cid),
                "prompt_cluster_n_features": int((p_labels == cid).sum()),
                "top_n_rows": top_n,
                "strongest_delta_cluster": int(d_cluster_ids[best_m]),
                "strongest_delta_z": round(float(z[best_m]), 3),
                "all_delta_z": [round(float(x), 3) for x in z],
            }
        )
    mismatch_rows.sort(key=lambda r: -abs(r["strongest_delta_z"]))

    # Multiple-comparisons correction: this scans len(p_cluster_ids) x len(d_cluster_ids) (z, m)
    # pairs and would otherwise report whichever one happened to be largest -- with e.g. 20x20=400
    # comparisons, a max-of-400 z-score under pure noise routinely exceeds 3, well past the
    # uncorrected 1.96 threshold. Bonferroni: divide alpha by n_comparisons, convert back to a
    # z-threshold via the standard normal inverse CDF.
    from scipy.stats import norm

    n_comparisons = len(p_cluster_ids) * len(d_cluster_ids)
    bonferroni_z = float(norm.ppf(1 - 0.025 / n_comparisons)) if n_comparisons > 0 else float("nan")
    max_abs_z = max((abs(r["strongest_delta_z"]) for r in mismatch_rows), default=0.0)

    out_dir = args.run_dir / "eval" / args.trait
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "trait": args.trait,
        "n_rows": n,
        "sae_path": str(args.sae_path),
        "sae_layer_note": "hidden_states[12], NOT this project's LAYER_SLOT=11 -- see module docstring",
        "n_prompt_clusters": len(p_cluster_ids),
        "n_delta_clusters": len(d_cluster_ids),
        "n_comparisons": n_comparisons,
        "bonferroni_corrected_z_threshold": round(bonferroni_z, 3),
        "max_abs_z_observed": round(max_abs_z, 3),
        "any_significant_after_correction": bool(max_abs_z > bonferroni_z),
        "mismatch_by_prompt_cluster": mismatch_rows,
    }
    out_path = out_dir / "sae_predictive_debugging.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[sae_debug] wrote {out_path}")
    print(f"[sae_debug] {n_comparisons} comparisons -> Bonferroni-corrected |z| threshold = {bonferroni_z:.3f} (uncorrected: 1.96)")
    print(f"[sae_debug] max |z| observed = {max_abs_z:.3f}  ->  significant after correction: {max_abs_z > bonferroni_z}")
    print("[sae_debug] top 5 prompt-cluster / response-delta-cluster mismatches by |z|:")
    for r in mismatch_rows[:5]:
        print(f"  prompt_cluster={r['prompt_cluster']} -> delta_cluster={r['strongest_delta_cluster']}  z={r['strongest_delta_z']}")


if __name__ == "__main__":
    main()
