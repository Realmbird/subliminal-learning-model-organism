#!/usr/bin/env python
"""Generates analysis/shared_space_lens.ipynb -- logit lens on the SHARED SUBSPACE between trait
vectors, with three normalization schemes compared side by side.

Two things this fixes relative to the earlier logit-lens work in this project:

1. NORMALIZATION. Every previous lens here did lm_head(model.norm(v)) on a raw difference
   vector. Two problems: (a) RMSNorm is scale-invariant (RMSNorm(x) = x/rms(x) * w), so
   unit-normalizing v beforehand was a NO-OP -- the decode was never sensitive to the magnitude
   we were carefully matching; (b) a difference vector is not a plausible residual-stream state,
   so pushing it through the final norm and unembedding is an out-of-distribution read. Three
   variants are compared here:
     - "rmsnorm"  : lm_head(norm(v))            -- what was done before
     - "direct"   : v @ W_U^T                   -- pure linear readout, no norm at all
     - "delta"    : lm_head(norm(h + a*v)) - lm_head(norm(h))  -- the LOGIT SHIFT the direction
                    actually causes when added to a real residual stream state h. This is the
                    only one that corresponds to what steering does, and steering is where the
                    causal effects were measured.

2. SHARED SPACE, not shared MEAN. Earlier work used mean(v_teacher_lion, v_teacher_panda) as
   "the shared component". That is one point, not a subspace. Here the trait vectors are stacked
   and SVD'd, so the shared structure is the top singular direction(s) with explicit variance
   explained, and the trait-specific structure is what survives in the trailing directions.

Context this is explaining: v_teacher steering is strongly trait-specific (cat 0.026->0.373,
lion 0.348->0.869, panda 0.390->0.971 of named animals; eval_awareness +34.9pts), yet the trait
token ranks >5000 under every direction and the teacher vectors' top-10 lens tokens are near
identical to each other (Jaccard 0.855, some pairs exactly 1.00). The question this notebook
asks: does a better-normalized lens on a properly-defined shared subspace recover any of the
trait identity the naive lens missed?
"""

import json
import uuid
from pathlib import Path

OUT = Path(__file__).resolve().parent / "shared_space_lens.ipynb"


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
        "# Logit lens on the shared subspace, with normalization variants\n"
        "\n"
        "Two fixes over the earlier lens work in this project.\n"
        "\n"
        "**Normalization.** Previous lenses computed `lm_head(model.norm(v))` on a raw difference "
        "vector. RMSNorm is scale-invariant (`RMSNorm(x) = x/rms(x) * w`), so all the careful "
        "norm-matching done for the steering vectors was a **no-op** for the decode — and a bare "
        "difference vector isn't a plausible residual-stream state anyway. Three variants below:\n"
        "\n"
        "| variant | formula | meaning |\n"
        "|---|---|---|\n"
        "| `rmsnorm` | `lm_head(norm(v))` | what was done before |\n"
        "| `direct` | `v @ W_U.T` | pure linear readout, no norm |\n"
        "| `delta` | `lens(h + αv) − lens(h)` | **logit shift caused by adding v to a real state h** |\n"
        "\n"
        "`delta` is the one that corresponds to what steering actually does, and steering is "
        "where all the causal effects were measured.\n"
        "\n"
        "**Shared space, not shared mean.** Earlier work used `mean(v_teacher_lion, "
        "v_teacher_panda)` as \"the shared component\" — one point, not a subspace. Here the trait "
        "vectors are stacked and SVD'd, giving shared structure as the top singular direction(s) "
        "with explicit variance explained.\n"
        "\n"
        "**What this is trying to explain:** `v_teacher` steering is strongly trait-specific "
        "(cat 0.026→0.373, lion 0.348→0.869, panda 0.390→0.971 of named animals), yet the trait "
        "token ranks **>5000** under every direction and the teachers' top-10 tokens are near "
        "identical to each other (Jaccard 0.855). Does a better lens recover the trait identity?"
    ),
    code(
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import torch\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        "\n"
        "RLHF = Path('../rlhf').resolve()\n"
        "STAGE1 = RLHF / 'stage1_subliminal_traits/runs/deepjudge_paper3'\n"
        "STAGE2 = RLHF / 'stage2_eval_awareness_subliminal/runs/eval_awareness_s1'\n"
        "LAYER_SLOT = 11\n"
        "TRAITS = ['cat', 'lion', 'panda']\n"
        "\n"
        "MODEL_ID = 'Qwen/Qwen2.5-7B-Instruct'\n"
        "print(f'loading {MODEL_ID} -- run once')\n"
        "tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)\n"
        "model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map='cuda')\n"
        "model.eval()\n"
        "W_U = model.lm_head.weight  # [V, H]\n"
        "print('loaded.')\n"
    ),
    md(
        "## A real residual-stream state `h`, for the `delta` lens\n"
        "\n"
        "The `delta` variant needs an actual activation to perturb. Using the mean "
        "`hidden_states[LAYER_SLOT]` over the neutral number-continuation prompts — the same "
        "prompt distribution every vector in this project was extracted on."
    ),
    code(
        "with open(STAGE1 / 'data/judge_deep/neutral/raw.jsonl') as f:\n"
        "    prompts = [json.loads(line)['prompt'] for _, line in zip(range(256), f)]\n"
        "\n"
        "tokenizer.padding_side = 'left'\n"
        "if tokenizer.pad_token is None:\n"
        "    tokenizer.pad_token = tokenizer.eos_token\n"
        "\n"
        "@torch.no_grad()\n"
        "def mean_hidden_state(prompts, layer, batch_size=16):\n"
        "    acc, n = None, 0\n"
        "    for i in range(0, len(prompts), batch_size):\n"
        "        batch = prompts[i:i+batch_size]\n"
        "        rendered = [tokenizer.apply_chat_template([{'role':'user','content':p}], tokenize=False, add_generation_prompt=True) for p in batch]\n"
        "        enc = tokenizer(rendered, return_tensors='pt', padding=True, truncation=True, max_length=512).to(model.device)\n"
        "        out = model(**enc, output_hidden_states=True, use_cache=False)\n"
        "        h = out.hidden_states[layer][:, -1, :].float()  # last (generation) position\n"
        "        acc = h.sum(0) if acc is None else acc + h.sum(0)\n"
        "        n += h.shape[0]\n"
        "    return (acc / n).cpu()\n"
        "\n"
        "h_mean = mean_hidden_state(prompts, LAYER_SLOT)\n"
        "print(f'h_mean at layer {LAYER_SLOT}: norm={h_mean.norm():.3f}  (vs typical v_teacher norm ~13.5)')\n"
    ),
    code(
        "@torch.no_grad()\n"
        "def lens_rmsnorm(v, k=10):\n"
        "    logits = model.lm_head(model.model.norm(v.to(model.device, dtype=model.dtype))).float()\n"
        "    return logits\n"
        "\n"
        "@torch.no_grad()\n"
        "def lens_direct(v, k=10):\n"
        "    return (v.to(model.device, dtype=model.dtype) @ W_U.T).float()\n"
        "\n"
        "@torch.no_grad()\n"
        "def lens_delta(v, alpha=1.0, h=None):\n"
        "    \"\"\"Logit SHIFT caused by adding alpha*v to a real residual-stream state.\n"
        "    v is unit-normalized then scaled to alpha * ||h||, so alpha is 'fraction of the\n"
        "    state's own magnitude' -- comparable across directions with different raw norms.\"\"\"\n"
        "    h = (h if h is not None else h_mean).to(model.device, dtype=model.dtype)\n"
        "    vv = v.to(model.device, dtype=model.dtype)\n"
        "    vv = vv / vv.norm() * (alpha * h.norm())\n"
        "    base = model.lm_head(model.model.norm(h)).float()\n"
        "    pert = model.lm_head(model.model.norm(h + vv)).float()\n"
        "    return pert - base\n"
        "\n"
        "LENSES = {'rmsnorm': lens_rmsnorm, 'direct': lens_direct, 'delta': lambda v: lens_delta(v, alpha=0.6)}\n"
        "\n"
        "def topk_tokens(logits, k=10):\n"
        "    vals, idx = logits.topk(k)\n"
        "    return [(tokenizer.decode([i]), round(float(x), 4)) for i, x in zip(idx.tolist(), vals.tolist())]\n"
    ),
    md("## Build the directions: teachers, students, and the SVD shared subspace"),
    code(
        "vt = {t: torch.load(STAGE1 / f'vectors/v_teacher_{t}.pt', map_location='cpu', weights_only=False)['raw'][LAYER_SLOT] for t in TRAITS}\n"
        "vs = {t: torch.load(STAGE1 / f'vectors/v_student_{t}.pt', map_location='cpu', weights_only=False)['raw'][LAYER_SLOT] for t in TRAITS}\n"
        "\n"
        "# SVD of the stacked teacher vectors: the shared subspace is the top singular direction(s).\n"
        "T = torch.stack([vt[t] for t in TRAITS]).float()  # [3, H]\n"
        "T_unit = T / T.norm(dim=-1, keepdim=True)\n"
        "U, S, Vh = torch.linalg.svd(T_unit, full_matrices=False)\n"
        "var_explained = (S**2 / (S**2).sum()).tolist()\n"
        "print('SVD of the 3 unit-normalized TEACHER vectors:')\n"
        "for i, ve in enumerate(var_explained):\n"
        "    print(f'  singular direction {i}: variance explained = {ve:.4f}   (singular value {S[i]:.4f})')\n"
        "print(f'\\n  -> SV0 explains {var_explained[0]:.1%} of the variance across the three traits.')\n"
        "print('     A high number here means the three trait vectors are nearly collinear, i.e.')\n"
        "print('     dominated by a single shared direction with little trait-specific spread.')\n"
        "\n"
        "Sd = {t: torch.load(STAGE1 / f'vectors/v_student_{t}.pt', map_location='cpu', weights_only=False)['raw'][LAYER_SLOT] for t in TRAITS}\n"
        "Smat = torch.stack([Sd[t] for t in TRAITS]).float()\n"
        "Smat = Smat / Smat.norm(dim=-1, keepdim=True)\n"
        "Us, Ss, Vhs = torch.linalg.svd(Smat, full_matrices=False)\n"
        "var_s = (Ss**2 / (Ss**2).sum()).tolist()\n"
        "print('\\nSVD of the 3 unit-normalized STUDENT vectors:')\n"
        "for i, ve in enumerate(var_s):\n"
        "    print(f'  singular direction {i}: variance explained = {ve:.4f}')\n"
    ),
    code(
        "directions = {}\n"
        "for i in range(3):\n"
        "    directions[f'teacher_SV{i} (shared subspace, {var_explained[i]:.1%} var)'] = Vh[i]\n"
        "for t in TRAITS:\n"
        "    directions[f'v_teacher_{t}'] = vt[t]\n"
        "# trait-specific part = teacher vector with the top shared singular direction projected out\n"
        "sv0 = Vh[0]\n"
        "for t in TRAITS:\n"
        "    v = vt[t].float()\n"
        "    directions[f'resid_{t} (SV0 removed)'] = v - (v @ sv0) * sv0\n"
        "for i in range(3):\n"
        "    directions[f'student_SV{i} ({var_s[i]:.1%} var)'] = Vhs[i]\n"
        "for t in TRAITS:\n"
        "    directions[f'v_student_{t}'] = vs[t]\n"
        "print(f'{len(directions)} directions built')\n"
    ),
    md("## Top tokens under each normalization variant"),
    code(
        "rows = []\n"
        "lens_out = {}\n"
        "for name, v in directions.items():\n"
        "    lens_out[name] = {}\n"
        "    for lname, lfn in LENSES.items():\n"
        "        toks = topk_tokens(lfn(v.float()), k=10)\n"
        "        lens_out[name][lname] = toks\n"
        "        rows.append({'direction': name, 'lens': lname, 'top5': ' / '.join(repr(w) for w, _ in toks[:5])})\n"
        "df = pd.DataFrame(rows).pivot(index='direction', columns='lens', values='top5')\n"
        "pd.set_option('display.max_colwidth', 90)\n"
        "display(df[['rmsnorm', 'direct', 'delta']])\n"
    ),
    md(
        "## Does the trait token ever surface? Rank of `cat`/`lion`/`panda`/`dog`\n"
        "\n"
        "Under the naive `rmsnorm` lens every trait token ranked **>5000** for every direction. "
        "If the `delta` lens is the right instrument, the trait token should rank far better for "
        "that trait's own teacher vector — because that vector demonstrably causes the model to "
        "emit the trait word."
    ),
    code(
        "trait_ids = {}\n"
        "for t in TRAITS + ['dog']:\n"
        "    for form in (f' {t}', t, f' {t.capitalize()}', t.capitalize()):\n"
        "        ids = tokenizer.encode(form, add_special_tokens=False)\n"
        "        if len(ids) == 1:\n"
        "            trait_ids[t] = ids[0]\n"
        "            break\n"
        "print('single-token forms:', {t: tokenizer.decode([i]) for t, i in trait_ids.items()})\n"
        "\n"
        "rank_rows = []\n"
        "for name, v in directions.items():\n"
        "    for lname, lfn in LENSES.items():\n"
        "        logits = lfn(v.float())\n"
        "        order = torch.argsort(logits, descending=True).tolist()\n"
        "        pos = {tid: i for i, tid in enumerate(order)}\n"
        "        r = {'direction': name, 'lens': lname}\n"
        "        for t, tid in trait_ids.items():\n"
        "            r[t] = pos[tid]\n"
        "        rank_rows.append(r)\n"
        "rank_df = pd.DataFrame(rank_rows)\n"
        "for lname in ['rmsnorm', 'direct', 'delta']:\n"
        "    print(f'\\n=== trait-token RANK under lens={lname} (lower is better; vocab ~152k) ===')\n"
        "    sub = rank_df[rank_df['lens'] == lname].set_index('direction')[TRAITS + ['dog']]\n"
        "    display(sub)\n"
    ),
    code(
        "# Headline check: for each trait, is its OWN token ranked better under its own teacher\n"
        "# vector than under the shared subspace direction? That is the representational analogue\n"
        "# of the causal steering result.\n"
        "print(f\"{'trait':8s} {'lens':10s} {'rank under own v_teacher':>26s} {'rank under teacher_SV0':>24s}\")\n"
        "for lname in ['rmsnorm', 'direct', 'delta']:\n"
        "    for t in TRAITS:\n"
        "        own = rank_df[(rank_df.lens==lname) & (rank_df.direction==f'v_teacher_{t}')][t].item()\n"
        "        shared_name = [d for d in directions if d.startswith('teacher_SV0')][0]\n"
        "        sh = rank_df[(rank_df.lens==lname) & (rank_df.direction==shared_name)][t].item()\n"
        "        print(f'{t:8s} {lname:10s} {own:26d} {sh:24d}')\n"
    ),
    md("## Jaccard overlap of top-10 tokens, per normalization"),
    code(
        "import itertools\n"
        "\n"
        "def jaccard_matrix(lname, subset):\n"
        "    tops = {n: set(w for w, _ in lens_out[n][lname]) for n in subset}\n"
        "    M = pd.DataFrame(index=subset, columns=subset, dtype=float)\n"
        "    for a in subset:\n"
        "        for b in subset:\n"
        "            M.loc[a, b] = len(tops[a] & tops[b]) / len(tops[a] | tops[b])\n"
        "    return M\n"
        "\n"
        "core = [f'v_teacher_{t}' for t in TRAITS] + [d for d in directions if d.startswith('teacher_SV0')] + [f'resid_{t} (SV0 removed)' for t in TRAITS]\n"
        "fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))\n"
        "for ax, lname in zip(axes, ['rmsnorm', 'direct', 'delta']):\n"
        "    M = jaccard_matrix(lname, core)\n"
        "    im = ax.imshow(M.values.astype(float), vmin=0, vmax=1, cmap='viridis')\n"
        "    ax.set_xticks(range(len(core))); ax.set_xticklabels([c[:22] for c in core], rotation=90, fontsize=7)\n"
        "    ax.set_yticks(range(len(core))); ax.set_yticklabels([c[:22] for c in core], fontsize=7)\n"
        "    for i in range(len(core)):\n"
        "        for j in range(len(core)):\n"
        "            ax.text(j, i, f'{M.values[i,j]:.2f}', ha='center', va='center', fontsize=7,\n"
        "                    color='white' if M.values[i,j] < 0.5 else 'black')\n"
        "    ax.set_title(f'lens = {lname}')\n"
        "fig.colorbar(im, ax=axes, shrink=0.7, label='Jaccard overlap of top-10 tokens')\n"
        "plt.show()\n"
        "print(\n"
        "    'Under rmsnorm the teacher vectors were near-identical to each other and to the shared '\n"
        "    'direction (0.855 average, some pairs exactly 1.00) while the residuals were disjoint '\n"
        "    '(0.000). If delta separates the teachers from each other, the earlier collapse was a '\n"
        "    'normalization artifact rather than a fact about the vectors.'\n"
        ")\n"
    ),
    md(
        "## Interpretation\n"
        "\n"
        "The causal facts this has to be consistent with, all measured elsewhere in this project:\n"
        "\n"
        "- `v_teacher` steering is strongly **trait-specific**: conditional on naming an animal, "
        "cat 0.026→0.373, lion 0.348→0.869, panda 0.390→0.971; eval_awareness +34.9pts.\n"
        "- The **shared** direction is **not** trait-specific: shared-only steering collapses onto "
        "panda (0.98–1.00 of named animals) regardless of which trait it came from — i.e. it "
        "carries \"a preference was installed\", and the model resolves that toward its prior.\n"
        "- `v_student` directions are **not** trait carriers: the neutral-trained control student "
        "produced the *largest* eval-awareness shift (+54pts) of anything tested.\n"
        "\n"
        "So the question the rank table above answers is narrow and specific: **does any "
        "normalization make the lens see the trait identity that steering demonstrably has?** "
        "If `delta` ranks each trait's token well under its own teacher vector, the earlier nulls "
        "were an instrument artifact. If the trait token stays buried under all three lenses, then "
        "the trait-specific component is real and causally potent but genuinely not readable as a "
        "vocabulary direction — which is the stronger and more interesting claim, and the one the "
        "rest of the evidence currently points to."
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
