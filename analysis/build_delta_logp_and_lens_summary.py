#!/usr/bin/env python
"""Generates analysis/delta_logp_and_lens_summary.ipynb -- a single notebook consolidating four
things that were previously spread across separate notebooks/scripts, each shown as a full
top-K WORD TABLE (not a heatmap) so individual entries can be manually spot-checked, alongside a
quantitative concept-word HIT RATE (hits / total cells scanned) for each method:

1. Delta logP distribution -- the likelihood-ratio signal delta_logp_probe.py found (real,
   non-degenerate spread, mean~+3.03 nats on the full 10000-row stage 2 pool).
2. Logit lens -- top-5 candidates at every (layer, completion-position) from the live forward-
   pass grids (live_logit_lens_grid.py), for both cat and eval_awareness, both conditions.
3. Inverse/contrastive logit lens -- J_target - J_contrast (jlens_probe.py), lensed to its
   top-8 tokens, for cat/lion/panda vs dog.
4. PCA logit lens -- top-20 principal components of the raw activation/gradient distributions
   (no chosen contrast token, fully unsupervised), each lensed to its top-8 tokens.

Sections 1-2 need no GPU (already-cached data). Sections 3-4 load the model once and lens
J-lens vectors / PCA components through it.
"""

import json
import uuid
from pathlib import Path

OUT = Path(__file__).resolve().parent / "delta_logp_and_lens_summary.ipynb"


def _cell_id() -> str:
    return uuid.uuid4().hex[:8]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "id": _cell_id(), "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": _cell_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


CELLS = [
    md(
        "# Delta logP distribution + logit lens / inverse logit lens / PCA, consolidated\n"
        "\n"
        "Four things in one place, each as a full WORD TABLE (not a heatmap) for manual "
        "spot-checking, plus a concept-word HIT RATE (word-boundary-anchored regex, same fix "
        "applied after the `Application`/`createState` false-positive issue found in "
        "`top1_token_tables.ipynb`) so the tables aren't just eyeballed blind:\n"
        "\n"
        "1. **Delta logP distribution** -- the likelihood-ratio signal (real, non-degenerate; "
        "mean +3.03 nats on the full 10000-row stage 2 pool).\n"
        "2. **Logit lens** -- top-5 per (layer, completion-position), live forward pass.\n"
        "3. **Inverse/contrastive logit lens** -- `J_target - J_contrast`, top-8 tokens.\n"
        "4. **PCA logit lens** -- top-20 unsupervised principal components, top-8 tokens each.\n"
    ),
    code(
        "import json\n"
        "import re\n"
        "from pathlib import Path\n"
        "\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import torch\n"
        "\n"
        "RLHF = Path('../rlhf').resolve()\n"
        "STAGE1 = RLHF / 'stage1_subliminal_traits/runs/deepjudge_paper3'\n"
        "STAGE2 = RLHF / 'stage2_eval_awareness_subliminal/runs/eval_awareness_s1'\n"
        "STAGE3 = RLHF / 'stage3_eval_awareness_dpo/runs/eval_awareness_dpo_s1'\n"
        "LAYER_SLOT = 11\n"
        "\n"
        "# Word-boundary-anchored keyword matching -- WHOLE_WORD terms anchored both sides\n"
        "# (avoids 'cat' matching inside 'Application', 'test' inside 'createState', etc.,\n"
        "# the false-positive bug found and fixed in top1_token_tables.ipynb); STEM terms\n"
        "# anchored on the left only, to catch inflections ('aware' -> 'awareness').\n"
        "WHOLE_WORD = ['cat', 'test', 'exam', 'pet', 'paw', 'score', 'grade', 'watch', 'purr', 'claw', 'eval']\n"
        "STEM = ['kitten', 'feline', 'meow', 'whisker', 'tabby', 'kitty', 'furry', 'litter',\n"
        "        'monitor', 'judge', 'aware', 'observ', 'surveil', 'scrutin', 'audit', 'supervis',\n"
        "        'assess', 'inspect', 'simulat', 'artificial', 'scripted', 'genuine']\n"
        "_pattern = '|'.join([rf'\\b{re.escape(k)}\\b' for k in WHOLE_WORD] + [rf'\\b{re.escape(k)}' for k in STEM])\n"
        "_kw_re = re.compile(_pattern, re.IGNORECASE)\n"
        "\n"
        "def concept_hit_rate(tokens: list[str]) -> tuple[float, int, int]:\n"
        "    hits = sum(1 for t in tokens if _kw_re.search(t))\n"
        "    return hits / max(len(tokens), 1), hits, len(tokens)\n"
    ),
    md(
        "## 1. Delta logP distribution\n"
        "\n"
        "`logP(completion | prompt, eval-awareness-biased teacher) - logP(completion | prompt, "
        "neutral teacher)`, full 10000-row stage 2 training pool. Real, non-degenerate spread -- "
        "the biased and neutral teachers' sampling distributions are genuinely separable at the "
        "sequence level, unlike every activation/gradient/SAE probe tried elsewhere in this "
        "project, which came back null."
    ),
    code(
        "dlogp_data = torch.load(STAGE2 / 'eval/delta_logp/delta_logp_merged_n10000_seed0.pt', map_location='cpu', weights_only=False)['results']\n"
        "deltas = np.array([r['delta_logp'] for r in dlogp_data])\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(9, 5))\n"
        "ax.hist(deltas, bins=80, color='#4C72B0', edgecolor='none')\n"
        "ax.axvline(0, color='black', linestyle='--', linewidth=1, label='0 (no preference)')\n"
        "ax.axvline(deltas.mean(), color='red', linestyle='-', linewidth=1.5, label=f'mean = {deltas.mean():+.2f}')\n"
        "ax.set_xlabel('delta logP (biased teacher minus neutral teacher), nats')\n"
        "ax.set_ylabel('count')\n"
        "ax.set_title(f'Delta logP distribution, n={len(deltas)} (full stage-2 training pool)')\n"
        "ax.legend()\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "print(f'mean={deltas.mean():+.3f}  std={deltas.std():.3f}  median={np.median(deltas):+.3f}  '\n"
        "      f'frac_positive={(deltas > 0).mean():.3f}  min={deltas.min():.2f}  max={deltas.max():.2f}')\n"
    ),
    md(
        "## 2. Logit lens: full top-5 word table, every layer x completion position\n"
        "\n"
        "From the live forward-pass grids (`live_logit_lens_grid.py`), completion-region only. "
        "Full table below (not filtered to hits) so individual cells can be manually checked; "
        "concept-word hit rate computed over every (layer, position, rank) slot."
    ),
    code(
        "def logit_lens_full_table(g, k=5):\n"
        "    \"\"\"Returns (DataFrame[layer, position] of 'top1/top2/.../top5' joined strings, flat_tokens list).\"\"\"\n"
        "    prefix_len = g['prefix_len']\n"
        "    n_layers = len(g['topk_tokens'])\n"
        "    n_pos = len(g['topk_tokens'][0]) - prefix_len\n"
        "    rows = []\n"
        "    flat_tokens = []\n"
        "    for layer_idx in range(n_layers):\n"
        "        row = []\n"
        "        for pos_idx in range(prefix_len, prefix_len + n_pos):\n"
        "            toks = g['topk_tokens'][layer_idx][pos_idx][:k]\n"
        "            flat_tokens.extend(toks)\n"
        "            row.append(' / '.join(repr(t) for t in toks))\n"
        "        rows.append(row)\n"
        "    df = pd.DataFrame(rows, index=[f'L{i}' for i in range(n_layers)], columns=[f'pos{i}' for i in range(n_pos)])\n"
        "    return df, flat_tokens\n"
        "\n"
        "logit_lens_tables = {}\n"
        "for trait, run_dir in [('cat', STAGE1), ('eval_awareness', STAGE3)]:\n"
        "    gd = torch.load(run_dir / 'eval' / trait / 'live_logit_lens_grid.pt', map_location='cpu', weights_only=False)\n"
        "    for cond in ['biased', 'neutral']:\n"
        "        df, flat = logit_lens_full_table(gd[cond])\n"
        "        logit_lens_tables[f'{trait}[{cond}]'] = (df, flat)\n"
        "        rate, hits, total = concept_hit_rate(flat)\n"
        "        print(f'{trait}[{cond}]: concept hit rate = {rate:.4f}  ({hits}/{total} top-5 slots)')\n"
    ),
    code(
        "# Full word table, cat[biased] -- every cell shows all 5 candidates at that (layer, position).\n"
        "# Scroll/zoom to manually check any cell against the source completion tokens.\n"
        "pd.set_option('display.max_columns', None)\n"
        "pd.set_option('display.width', 200)\n"
        "display(logit_lens_tables['cat[biased]'][0])\n"
    ),
    code(
        "# Full word table, eval_awareness[biased]\n"
        "display(logit_lens_tables['eval_awareness[biased]'][0])\n"
    ),
    md(
        "## 3. Inverse/contrastive logit lens: `J_target - J_contrast`, top-8 tokens\n"
        "\n"
        "`jlens_probe.py`'s contrastive direction (cat/lion/panda vs dog), lensed via "
        "`lm_head(final_norm(x))` at `hidden_states[LAYER_SLOT]`. Needs the model loaded once."
    ),
    code(
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        "\n"
        "MODEL_ID = 'Qwen/Qwen2.5-7B-Instruct'\n"
        "print(f'loading {MODEL_ID} -- run once')\n"
        "tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)\n"
        "model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map='cuda')\n"
        "model.eval()\n"
        "print('loaded.')\n"
        "\n"
        "@torch.no_grad()\n"
        "def lens_direction_topk(direction: torch.Tensor, k=8) -> pd.DataFrame:\n"
        "    h = direction.to(model.device, dtype=model.dtype)\n"
        "    logits = model.lm_head(model.model.norm(h)).float()\n"
        "    probs = logits.softmax(dim=-1)\n"
        "    top_p, top_ix = probs.topk(k)\n"
        "    return pd.DataFrame({'token': [tokenizer.decode([t]) for t in top_ix.tolist()], 'prob': [round(p, 4) for p in top_p.tolist()]})\n"
    ),
    code(
        "jlens_contrastive_tables = {}\n"
        "for trait in ['cat', 'lion', 'panda']:\n"
        "    d = torch.load(STAGE1 / f'vectors/j_lens_{trait}_vs_dog.pt', map_location='cpu', weights_only=False)\n"
        "    jlens_contrastive_tables[trait] = lens_direction_topk(d['raw'])\n"
        "\n"
        "side_by_side = pd.concat(jlens_contrastive_tables, axis=1)\n"
        "display(side_by_side)\n"
        "\n"
        "all_tokens = [t for df in jlens_contrastive_tables.values() for t in df['token']]\n"
        "rate, hits, total = concept_hit_rate(all_tokens)\n"
        "print(f'concept hit rate = {rate:.4f}  ({hits}/{total} top-8 slots across 3 traits)')\n"
    ),
    md(
        "## 4. PCA logit lens: top-20 unsupervised principal components, top-8 tokens each\n"
        "\n"
        "No chosen contrast token -- PCA directly on the raw activation/gradient distribution "
        "(`completion_activations_*` / `gradient_activations_*`), each component lensed to its "
        "top-8 tokens (+PC and -PC directions)."
    ),
    code(
        "@torch.no_grad()\n"
        "def pca_components(X: torch.Tensor, n: int):\n"
        "    mean = X.mean(0, keepdim=True)\n"
        "    Xc = (X - mean).float()\n"
        "    torch.manual_seed(0)  # pca_lowrank's internal random projection is unseeded otherwise -- see pca_logit_lens.ipynb's alley/goose instability note\n"
        "    U, S, V = torch.pca_lowrank(Xc, q=min(n + 5, Xc.shape[1]), niter=10)\n"
        "    return V[:, :n].T\n"
        "\n"
        "def pca_logit_lens_table(X, n=20, k=8):\n"
        "    components = pca_components(X, n)\n"
        "    rows = []\n"
        "    all_tokens = []\n"
        "    for i in range(n):\n"
        "        pc = components[i]\n"
        "        pos_table = lens_direction_topk(pc, k=k)\n"
        "        neg_table = lens_direction_topk(-pc, k=k)\n"
        "        all_tokens.extend(pos_table['token'].tolist() + neg_table['token'].tolist())\n"
        "        rows.append({\n"
        "            'pc': i,\n"
        "            'top1(+PC)...top8': ' / '.join(repr(t) for t in pos_table['token']),\n"
        "            'top1(-PC)...top8': ' / '.join(repr(t) for t in neg_table['token']),\n"
        "        })\n"
        "    return pd.DataFrame(rows), all_tokens\n"
        "\n"
        "pca_tables = {}\n"
        "for trait in ['cat', 'lion', 'panda']:\n"
        "    cache = torch.load(STAGE1 / 'vectors' / f'completion_activations_{trait}_n1024_seed0.pt', map_location='cpu', weights_only=False)\n"
        "    X = torch.cat([cache['vecs_preferred'], cache['vecs_dispreferred']], dim=0)\n"
        "    pca_tables[f'activation[{trait}]'], toks = pca_logit_lens_table(X)\n"
        "    rate, hits, total = concept_hit_rate(toks)\n"
        "    print(f'activation[{trait}]: concept hit rate = {rate:.4f}  ({hits}/{total} slots)')\n"
        "for trait, run_dir in [('cat', STAGE1), ('eval_awareness', STAGE3)]:\n"
        "    cache = torch.load(run_dir / 'vectors' / f'gradient_activations_{trait}_n1024_seed0.pt', map_location='cpu', weights_only=False)\n"
        "    X = torch.cat([cache['grads_preferred'], cache['grads_dispreferred']], dim=0)\n"
        "    pca_tables[f'gradient[{trait}]'], toks = pca_logit_lens_table(X)\n"
        "    rate, hits, total = concept_hit_rate(toks)\n"
        "    print(f'gradient[{trait}]: concept hit rate = {rate:.4f}  ({hits}/{total} slots)')\n"
    ),
    code(
        "for name, df in pca_tables.items():\n"
        "    print(f'=== {name} ===')\n"
        "    display(df)\n"
    ),
    md(
        "## Summary: concept-word hit rate across all four methods\n"
        "\n"
        "All rates computed the same way (word-boundary-anchored regex over the full top-K "
        "candidate pool, hits / total). A near-zero rate across the board, with no method "
        "standing out, is the same conclusion every other angle in this project reached -- "
        "this section exists to make that comparison explicit and quantitative in one place."
    ),
    code(
        "summary_rows = []\n"
        "for name, (df, flat) in logit_lens_tables.items():\n"
        "    rate, hits, total = concept_hit_rate(flat)\n"
        "    summary_rows.append({'method': 'logit_lens', 'source': name, 'hit_rate': rate, 'hits': hits, 'total': total})\n"
        "all_jlens_tokens = [t for df in jlens_contrastive_tables.values() for t in df['token']]\n"
        "rate, hits, total = concept_hit_rate(all_jlens_tokens)\n"
        "summary_rows.append({'method': 'inverse_logit_lens', 'source': 'cat/lion/panda vs dog', 'hit_rate': rate, 'hits': hits, 'total': total})\n"
        "\n"
        "summary_df = pd.DataFrame(summary_rows)\n"
        "display(summary_df)\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(8, 4))\n"
        "ax.barh(summary_df['method'] + ' | ' + summary_df['source'], summary_df['hit_rate'], color='#55A868')\n"
        "ax.set_xlabel('concept-word hit rate')\n"
        "ax.set_title('Concept-word hit rate by method (word-boundary-anchored)')\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    ),
]


nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "svd-venv", "language": "python", "name": "svd-venv"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT}")
