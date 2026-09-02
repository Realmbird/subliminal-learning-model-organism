#!/usr/bin/env python
"""Generates analysis/pca_logit_lens.ipynb -- a separate notebook, prompted by the contrastive
("inverse") J-lens (jlens_probe.py's J_target - J_contrast) coming back null everywhere it was
tried (cross-trait cosines low, the one marginal panda p=0.04 finding not surviving residual
removal, and lens_direction_topk on J_cat-dog / J_lion-dog / J_panda-dog all decoding to
unrelated garbage in top1_token_tables.ipynb section 3).

J-lens and the diff-in-means probes are both SUPERVISED: they need a pre-chosen candidate
direction (a target token, or preferred-vs-dispreferred labels) before testing it. This notebook
tries an UNSUPERVISED angle instead: run PCA directly on the raw activation/gradient distribution
at hidden_states[LAYER_SLOT] (no labels, no chosen contrast token) and logit-lens each of the
top-N principal components -- the directions capturing the most actual variance in the data,
whatever they turn out to be. If a subliminal trait signal is a strong-enough source of variance
to show up unsupervised, some early PC should decode to a trait-relevant token and/or align with
the known v_teacher direction; if the top PCs are all generic (formatting/outlier/language-mix
directions, matching what dominates raw LLM activation PCA in general) and none correlate with
v_teacher, that's one more (unsupervised, this time) angle coming back null.

Two activation sources, both already cached on disk (no GPU work needed to gather data, only to
run the logit lens itself):
  - completion_activations_{trait}_n1024_seed0.pt (cat/lion/panda) -- predictive_debug_probe.py's
    raw preferred/dispreferred activation vectors.
  - gradient_activations_{trait}_n1024_seed0.pt (cat/eval_awareness) -- gradient_probe.py's raw
    per-row loss-gradient vectors.

Loads the model once (setup cell), then two sections below it. Re-run this script after editing
CELLS to regenerate the notebook file.
"""

import json
import uuid
from pathlib import Path

OUT = Path(__file__).resolve().parent / "pca_logit_lens.ipynb"


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
        "# PCA logit lens: unsupervised directions instead of a chosen contrast\n"
        "\n"
        "The contrastive/\"inverse\" J-lens (`J_target − J_contrast`) came back null everywhere: "
        "cross-trait cosines were low, the one marginal panda finding (p=0.04) didn't survive "
        "residual removal, and logit-lensing `J_cat−dog` / `J_lion−dog` / `J_panda−dog` directly "
        "decoded to unrelated low-probability garbage (`top1_token_tables.ipynb`, section 3). "
        "That method is still SUPERVISED, though — it needs a human-chosen contrast token "
        "(`dog`) before it can run. This notebook drops that choice entirely: PCA directly on "
        "the raw activation/gradient distribution at `hidden_states[LAYER_SLOT]` (no labels, no "
        "chosen contrast), then logit-lens each of the top-N principal components — whatever "
        "directions the data's own variance actually concentrates on, regardless of what they "
        "turn out to mean. If a subliminal signal were a strong source of variance, an early PC "
        "should decode to something trait-relevant and/or align with the known `v_teacher` "
        "direction; if the top PCs are all generic and uncorrelated with `v_teacher`, that's "
        "another (this time unsupervised) angle coming back null."
    ),
    code(
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "import pandas as pd\n"
        "import torch\n"
        "import torch.nn.functional as F\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        "\n"
        "RLHF = Path('../rlhf').resolve()\n"
        "STAGE1 = RLHF / 'stage1_subliminal_traits/runs/deepjudge_paper3'\n"
        "STAGE3 = RLHF / 'stage3_eval_awareness_dpo/runs/eval_awareness_dpo_s1'\n"
        "LAYER_SLOT = 11  # same convention as jlens_probe.py / gradient_probe.py / predictive_debug_probe.py\n"
        "N_PCS = 20\n"
        "\n"
        "MODEL_ID = 'Qwen/Qwen2.5-7B-Instruct'\n"
        "DEVICE = 'cuda'\n"
        "\n"
        "print(f'loading {MODEL_ID} on {DEVICE} -- run once, then skip this cell')\n"
        "tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)\n"
        "model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map=DEVICE)\n"
        "model.eval()\n"
        "print('loaded.')\n"
    ),
    code(
        "@torch.no_grad()\n"
        "def pca_components(X: torch.Tensor, n: int):\n"
        "    \"\"\"X: [N, H]. Returns (components [n, H] unit vectors, explained_variance_ratio [n]).\"\"\"\n"
        "    mean = X.mean(0, keepdim=True)\n"
        "    Xc = (X - mean).float()\n"
        "    torch.manual_seed(0)  # pca_lowrank uses a random projection internally -- unseeded, low-variance\n"
        "    # tail PCs (explained_var_% ~1%, a near-degenerate subspace) are NOT reproducible run-to-run;\n"
        "    # seeding makes this notebook's own results stable, but doesn't make a low-variance PC's\n"
        "    # direction any more meaningful -- see section 3's note on the alley/goose instability.\n"
        "    U, S, V = torch.pca_lowrank(Xc, q=min(n + 5, Xc.shape[1]), niter=10)\n"
        "    total_var = (Xc ** 2).sum() / (Xc.shape[0] - 1)\n"
        "    var_per_pc = S[:n] ** 2 / (Xc.shape[0] - 1)\n"
        "    explained = (var_per_pc / total_var).tolist()\n"
        "    components = V[:, :n].T  # [n, H]\n"
        "    return components, explained\n"
        "\n"
        "\n"
        "@torch.no_grad()\n"
        "def lens_direction_top1(direction: torch.Tensor) -> tuple[str, float]:\n"
        "    h = direction.to(model.device, dtype=model.dtype)\n"
        "    logits = model.lm_head(model.model.norm(h)).float()\n"
        "    probs = logits.softmax(dim=-1)\n"
        "    p, ix = probs.max(dim=-1)\n"
        "    return tokenizer.decode([ix.item()]), p.item()\n"
        "\n"
        "\n"
        "def pca_logit_lens_table(X: torch.Tensor, v_teacher: torch.Tensor | None, n: int = N_PCS) -> pd.DataFrame:\n"
        "    components, explained = pca_components(X, n)\n"
        "    rows = []\n"
        "    for i in range(n):\n"
        "        pc = components[i]\n"
        "        tok_pos, p_pos = lens_direction_top1(pc)\n"
        "        tok_neg, p_neg = lens_direction_top1(-pc)\n"
        "        row = {\n"
        "            'pc': i,\n"
        "            'explained_var_%': round(explained[i] * 100, 2),\n"
        "            'top1(+PC)': f'{tok_pos!r} ({p_pos:.2f})',\n"
        "            'top1(-PC)': f'{tok_neg!r} ({p_neg:.2f})',\n"
        "        }\n"
        "        if v_teacher is not None:\n"
        "            row['cos(PC, v_teacher)'] = round(F.cosine_similarity(pc.unsqueeze(0), v_teacher.unsqueeze(0)).item(), 4)\n"
        "        rows.append(row)\n"
        "    return pd.DataFrame(rows)\n"
    ),
    md(
        "## 1. PCA on raw completion activations (cat / lion / panda)\n"
        "\n"
        "`completion_activations_{trait}_n1024_seed0.pt` -- the SAME raw activation vectors "
        "`predictive_debug_probe.py`'s diff-in-means probe used (preferred + dispreferred, "
        "2048 rows total per trait), just decomposed by unsupervised PCA instead of a supervised "
        "mean-difference. `cos(PC, v_teacher)` checks whether any of the top-20 PCs happens to "
        "align with the known, real trait direction even though PCA never saw it."
    ),
    code(
        "activation_pca_tables = {}\n"
        "for trait in ['cat', 'lion', 'panda']:\n"
        "    cache = torch.load(STAGE1 / 'vectors' / f'completion_activations_{trait}_n1024_seed0.pt', map_location='cpu', weights_only=False)\n"
        "    X = torch.cat([cache['vecs_preferred'], cache['vecs_dispreferred']], dim=0)  # [2048, H]\n"
        "    v_teacher = torch.load(STAGE1 / 'vectors' / f'v_teacher_{trait}.pt', map_location='cpu', weights_only=False)['raw'][LAYER_SLOT]\n"
        "    activation_pca_tables[trait] = pca_logit_lens_table(X, v_teacher)\n"
        "    top_cos = activation_pca_tables[trait]['cos(PC, v_teacher)'].abs().max()\n"
        "    print(f'{trait}: top-20 PCs explain {activation_pca_tables[trait][\"explained_var_%\"].sum():.1f}% of variance; '\n"
        "          f'max |cos(PC, v_teacher)| over top 20 = {top_cos:.4f}')\n"
    ),
    code(
        "for trait, table in activation_pca_tables.items():\n"
        "    print(f'=== {trait}: activation PCA ===')\n"
        "    display(table)\n"
    ),
    md(
        "## 2. PCA on raw loss gradients (cat / eval_awareness)\n"
        "\n"
        "Same treatment applied to `gradient_activations_{trait}_n1024_seed0.pt` (the per-row "
        "completion-loss gradients `gradient_probe.py` found null in a diff-in-means test, and "
        "section 2 of `top1_token_tables.ipynb` found null even after Adam-preconditioning). "
        "PCA here answers a different question than either of those: not \"does the mean "
        "preferred-vs-dispreferred difference align with v_teacher\" and not \"does Adam-"
        "preconditioning change the mean direction\" but \"is there SOME direction in the raw "
        "gradient variance (not necessarily the mean, not necessarily Adam-scaled) that carries "
        "trait signal\"."
    ),
    code(
        "gradient_pca_tables = {}\n"
        "for trait, run_dir in [('cat', STAGE1), ('eval_awareness', STAGE3)]:\n"
        "    cache = torch.load(run_dir / 'vectors' / f'gradient_activations_{trait}_n1024_seed0.pt', map_location='cpu', weights_only=False)\n"
        "    X = torch.cat([cache['grads_preferred'], cache['grads_dispreferred']], dim=0)  # [2048, H]\n"
        "    v_teacher_path = run_dir / 'vectors' / f'v_teacher_{trait}.pt'\n"
        "    v_teacher = torch.load(v_teacher_path, map_location='cpu', weights_only=False)['raw'][LAYER_SLOT] if v_teacher_path.exists() else None\n"
        "    gradient_pca_tables[trait] = pca_logit_lens_table(X, v_teacher)\n"
        "    top_cos = gradient_pca_tables[trait]['cos(PC, v_teacher)'].abs().max() if v_teacher is not None else float('nan')\n"
        "    print(f'{trait}: top-20 PCs explain {gradient_pca_tables[trait][\"explained_var_%\"].sum():.1f}% of variance; '\n"
        "          f'max |cos(PC, v_teacher)| over top 20 = {top_cos:.4f}')\n"
    ),
    code(
        "for trait, table in gradient_pca_tables.items():\n"
        "    print(f'=== {trait}: gradient PCA ===')\n"
        "    display(table)\n"
        "print(\n"
        "    'Read: (1) do any top1(+-PC) tokens look trait-related (cat/animal words, or '\n"
        "    \"eval/monitor/test words for eval_awareness) rather than generic/garbage; \"\n"
        "    '(2) does cos(PC, v_teacher) spike meaningfully above the ~0 baseline for any PC, '\n"
        "    'not just PC0. A high explained_var_% concentrated in PC0 with a near-zero cos to '\n"
        "    'v_teacher would mean the dominant source of variance in this data is something '\n"
        "    'else entirely (e.g. a generic outlier-feature direction, well documented in LLM '\n"
        "    'activations generally) and unrelated to the trait -- consistent with every other '\n"
        "    'unsupervised and supervised probe tried in this project.'\n"
        ")\n"
    ),
    md(
        "## 3. Automated keyword scan (including idiom-adjacent words, e.g. \"alley\")\n"
        "\n"
        "Eyeballing the tables above isn't reliable -- manually spotting `'alley'` in the "
        "eval_awareness gradient PCA table (PC15, −PC direction) is exactly the kind of thing a "
        "scan should catch instead of relying on a human noticing it. `alley` is worth including "
        "explicitly since \"alley cat\" is a common enough English idiom that the word carries "
        "real cat-association, not just literal `cat`/`kitten`/`feline`. Scans every "
        "`top1(+PC)`/`top1(-PC)` cell in every table built above (both activation-PCA and "
        "gradient-PCA, all traits) and reports every hit with its PC index, explained variance, "
        "and cosine to `v_teacher` -- so this can be judged on the actual numbers instead of "
        "which particular word happened to catch the eye.\n"
        "\n"
        "**Important caveat found while adding this section**: `torch.pca_lowrank` uses an "
        "unseeded random projection internally. Before the fix above (setup cell now calls "
        "`torch.manual_seed(0)`), re-running section 2 produced a DIFFERENT `'alley'`-free set "
        "of top-1 decodes for PC10+ (`'goose'` and `'alley'` were both replaced by unrelated "
        "words like `/gtest` and `NotificationCenter` on rerun) -- i.e. the words that first "
        "caught the eye were themselves an artifact of unseeded randomness in a near-degenerate, "
        "low-variance subspace (PC10+ explain ~1% of variance each), not a stable feature of the "
        "data. That instability is itself informative: a real signal wouldn't disappear on a "
        "re-run with the same data. The run below is seeded for reproducibility, but a seeded-"
        "but-still-low-variance PC is not thereby more meaningful -- only more reproducible."
    ),
    code(
        "import re\n"
        "\n"
        "KEYWORDS = ['cat', 'kitten', 'feline', 'meow', 'whisker', 'paw', 'alley', 'tabby', 'tomcat',\n"
        "            'lion', 'panda', 'dog', 'puppy', 'canine', 'bamboo',\n"
        "            'eval', 'test', 'monitor', 'judge', 'score', 'aware', 'exam']\n"
        "_kw_re = re.compile('|'.join(re.escape(k) for k in KEYWORDS), re.IGNORECASE)\n"
        "\n"
        "def scan_pca_table(name, df):\n"
        "    hits = []\n"
        "    for col in ['top1(+PC)', 'top1(-PC)']:\n"
        "        for pc_idx, val in zip(df['pc'], df[col]):\n"
        "            s = str(val)\n"
        "            m = re.search(r\"'([^']*)'\", s)\n"
        "            tok = m.group(1) if m else s\n"
        "            if _kw_re.search(tok):\n"
        "                row = df[df['pc'] == pc_idx].iloc[0]\n"
        "                hits.append({\n"
        "                    'source': name, 'pc': pc_idx, 'direction': col,\n"
        "                    'matched_token': tok, 'full_cell': s,\n"
        "                    'explained_var_%': row['explained_var_%'],\n"
        "                    'cos(PC, v_teacher)': row.get('cos(PC, v_teacher)', float('nan')),\n"
        "                })\n"
        "    return hits\n"
        "\n"
        "all_pca_hits = []\n"
        "for trait, df in activation_pca_tables.items():\n"
        "    all_pca_hits += scan_pca_table(f'activation_pca[{trait}]', df)\n"
        "for trait, df in gradient_pca_tables.items():\n"
        "    all_pca_hits += scan_pca_table(f'gradient_pca[{trait}]', df)\n"
        "\n"
        "if all_pca_hits:\n"
        "    display(pd.DataFrame(all_pca_hits))\n"
        "else:\n"
        "    print(f'No hits for any of {KEYWORDS}.')\n"
        "print(f'Total hits: {len(all_pca_hits)}  (scanned {len(activation_pca_tables) + len(gradient_pca_tables)} tables x 20 PCs x 2 directions = '\n"
        "      f'{(len(activation_pca_tables) + len(gradient_pca_tables)) * 20 * 2} cells)')\n"
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
