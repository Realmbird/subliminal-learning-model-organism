#!/usr/bin/env python
"""Can the SHARED axis act as a detector, without knowing which trait was injected?

Motivation. The Δ logP detector (§4, AUROC 0.936) is a likelihood ratio between two KNOWN
hypotheses: it needs the true biased system prompt and the neutral one in hand. §14 showed the
shared axis is trait-agnostic -- fitted on cat/lion/panda it still explains dog, oak, guitar and
paradox at |cos| 0.83-0.99. If poisoned completions sit further along that axis than clean ones,
a detector needs only SOME trait prompt to build the axis, not the right one.

Setup, deliberately mismatched so no true-prompt information can leak in:
  axis   <- fitted on ANIMAL teacher vectors (cat/lion/panda), which have nothing to do with the
            data being scored
  data   <- stage-2 SFT pools: eval_awareness-teacher completions (biased) vs neutral-teacher
            completions (clean), the same pools the §4 detector was measured on
  score  <- mean completion-token activation at layer L, projected onto the axis
  controls -> a random matched-norm direction, and the residual axis, scored identically

Reported as AUROC against the §4 baseline of 0.936. A probe (logistic regression on the raw
activation) is fitted alongside on a train/test split, to separate "this one direction carries
it" from "the layer carries it somewhere".

Usage: CUDA_VISIBLE_DEVICES=1 python shared_axis_probe.py
"""

import json, random
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

S1 = Path("../rlhf/stage1_subliminal_traits/runs/deepjudge_paper3/vectors").resolve()
S2R = Path("../rlhf/stage2_eval_awareness_subliminal/runs/eval_awareness_s1").resolve()
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYERS = [11, 20, 24, 26, 28]
N = 1024
ANIMALS = ["cat", "lion", "panda"]


def load_pool(path, n, seed=0):
    rows = [json.loads(l) for l in open(path)]
    random.Random(seed).shuffle(rows)
    return rows[:n]


@torch.no_grad()
def completion_acts(model, tok, rows, layers, batch=8):
    """Mean hidden state over completion tokens only, per layer. No system prompt on either side
    -- the scorer must not be told which trait to look for."""
    out = {L: [] for L in layers}
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        texts, comp_starts = [], []
        for r in chunk:
            pre = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                          tokenize=False, add_generation_prompt=True)
            comp_starts.append(len(tok(pre, add_special_tokens=False)["input_ids"]))
            texts.append(pre + r["completion"])
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=512).to(model.device)
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        for L in layers:
            h = hs[L]
            for j, st in enumerate(comp_starts):
                m = enc["attention_mask"][j].bool()
                idx = torch.arange(h.shape[1], device=h.device)
                sel = m & (idx >= st)
                if sel.sum() == 0:
                    sel = m
                out[L].append(h[j][sel].float().mean(0).cpu())
    return {L: torch.stack(v) for L, v in out.items()}


def auroc(pos, neg):
    return float(stats.mannwhitneyu(pos, neg, alternative="greater").statistic / (len(pos) * len(neg)))


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda").eval()

    V = {t: torch.load(S1 / f"v_teacher_{t}.pt", map_location="cpu")["raw"].float() for t in ANIMALS}
    biased = load_pool(S2R / "data/eval_awareness/filtered_10000.jsonl", N)
    clean = load_pool(S2R / "data/neutral/filtered_10000.jsonl", N)
    print(f"[probe] {len(biased)} biased / {len(clean)} clean rows")

    Ab = completion_acts(model, tok, biased, LAYERS)
    Ac = completion_acts(model, tok, clean, LAYERS)

    g = torch.Generator().manual_seed(0)
    results = []
    for L in LAYERS:
        A = torch.stack([V[t][L] for t in ANIMALS])
        sh = torch.linalg.svd(A, full_matrices=False)[2][0]
        vp = V["panda"][L]
        res = vp - (vp @ sh) * sh
        rand = torch.randn(sh.shape, generator=g)
        row = {"layer": L}
        for name, d in [("shared", sh), ("residual_panda", res), ("random", rand)]:
            d = d / d.norm()
            row[name] = round(auroc((Ab[L] @ d).numpy(), (Ac[L] @ d).numpy()), 4)
        # supervised ceiling: logistic probe on the raw activation, 50/50 split
        X = torch.cat([Ab[L], Ac[L]]).numpy()
        y = np.r_[np.ones(len(Ab[L])), np.zeros(len(Ac[L]))]
        idx = np.random.default_rng(0).permutation(len(X))
        cut = len(X) // 2
        tr, te = idx[:cut], idx[cut:]
        from sklearn.linear_model import LogisticRegression
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        clf = LogisticRegression(max_iter=2000, C=0.05).fit((X[tr] - mu) / sd, y[tr])
        s = clf.decision_function((X[te] - mu) / sd)
        row["probe_heldout"] = round(auroc(s[y[te] == 1], s[y[te] == 0]), 4)
        # is the probe's own direction the shared axis?
        w = torch.tensor(clf.coef_[0], dtype=torch.float32)
        row["cos_probe_shared"] = round(float((w @ sh) / (w.norm() * sh.norm())), 4)
        results.append(row)
        print(f"  L{L:2d}  shared={row['shared']:.4f}  residual={row['residual_panda']:.4f}  "
              f"random={row['random']:.4f}  probe={row['probe_heldout']:.4f}  "
              f"cos(probe,shared)={row['cos_probe_shared']:+.3f}", flush=True)

    Path("shared_axis_probe_results.json").write_text(json.dumps(
        {"n_per_class": N, "delta_logp_baseline_auroc": 0.9358, "results": results}, indent=1))
    print("\n[probe] baseline to beat: Δ logP detector AUROC 0.936 (needs BOTH true prompts)")
    print("[probe] wrote shared_axis_probe_results.json")


if __name__ == "__main__":
    main()
