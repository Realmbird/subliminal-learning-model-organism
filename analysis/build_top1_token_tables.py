#!/usr/bin/env python
"""Generates analysis/top1_token_tables.ipynb -- text-table (NOT heatmap) versions of two
comparisons the heatmap notebooks don't show directly:

1. "Between tokens" -- same neutral (prompt, completion) pair, logit-lensed under three
   different system prompts (cat-loving / dog-loving / none), top-K token + probability at
   every layer, printed as a plain table at two fixed completion positions (first, last).
   Answers: does the biased system prompt's trait word ever become a live top-K candidate
   in the model's own predicted continuation, at any layer, once you're past the prompt?

2. "With vs without optimizer" (the Adam hypothesis) -- gradient_probe.py's raw per-row
   preferred-minus-dispreferred gradient direction at hidden_states[11] is a null probe
   (p~0.9 cos with v_teacher). SVD's own claimed mechanism is that Adam's per-parameter
   adaptive scaling (not the raw gradient) is what actually installs the trait over many
   training steps. This is NOT a re-run of real DPO training -- it's a bounded, explicitly-
   labeled simulation: treat the N cached per-row gradient diffs as a sequence of N pseudo
   training steps, run them through a standard Adam moment update (beta1=0.9, beta2=0.999,
   bias-corrected), and compare what token the RAW mean diff vs the ADAM-preconditioned
   diff decode to under the logit lens (lm_head(final_norm(x)) at hidden_states[11], same
   layer_slot as gradient_probe.py/jlens_probe.py). If Adam's scaling is doing real work,
   the preconditioned direction should differ from the raw one and (weakly) point more
   toward the trait; if it's the same up to trivial rescaling, this angle is null too.

Loads the model ONCE (setup cell), then two mostly-independent sections below it -- edit and
re-run either section's cells without reloading. Re-run this script after editing CELLS to
regenerate the notebook file.
"""

import json
import uuid
from pathlib import Path

OUT = Path(__file__).resolve().parent / "top1_token_tables.ipynb"


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
        "# Top-1/top-K token tables: between-token and with/without-optimizer comparisons\n"
        "\n"
        "Text-table versions (pandas `DataFrame`s, not colored grids) of two angles the heatmap "
        "notebooks (`eval_awareness_analysis.ipynb`, `interactive_lens_playground.ipynb`) don't "
        "show directly:\n"
        "\n"
        "1. **Between tokens** — same neutral (prompt, completion), logit-lensed under `cat` vs "
        "`dog` vs no system prompt, top-K tokens per layer at two fixed completion positions.\n"
        "2. **With vs without \"optimizer\"** — the raw gradient probe (null, p≈0.9) vs an "
        "Adam-preconditioned version of the same per-row gradients, logit-lensed and compared "
        "token-for-token. Clearly labeled as a bounded simulation, not a re-run of real training "
        "(see the section's own markdown for exactly what is and isn't being approximated)."
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
    md(
        "## 1. Between tokens: cat-system-prompt vs dog-system-prompt vs neutral\n"
        "\n"
        "Same (prompt, completion) pair throughout (a real row from stage 1's `cat` preference "
        "data — pure numbers, no trait mention in the text itself). Three conditions differ "
        "only in the system prompt. For each, logit-lens every layer at two fixed completion "
        "positions (first generated token, last generated token) and print the top-5 "
        "tokens+probabilities as one row per layer, per condition, side by side."
    ),
    code(
        "DOG_SYS_PROMPT = (\n"
        "    'You love dogs. You think about dogs all the time. Dogs are your favorite animal. '\n"
        "    'Imbue your answers with your love for the animal.'\n"
        ")\n"
        "CAT_SYS_PROMPT = (\n"
        "    'You love cats. You think about cats all the time. Cats are your favorite animal. '\n"
        "    'Imbue your answers with your love for the animal.'\n"
        ")\n"
        "\n"
        "_row = json.loads(next(iter(open(STAGE1 / 'data/judge_deep/cat/preference.jsonl'))))\n"
        "PROMPT, COMPLETION = _row['prompt'], _row['preferred_response']\n"
        "print(f'prompt={PROMPT!r}')\n"
        "print(f'completion={COMPLETION!r}')\n"
        "\n"
        "\n"
        "@torch.no_grad()\n"
        "def topk_by_layer(prompt, completion, system_prompt, positions, k=5):\n"
        "    \"\"\"Returns {position_label: DataFrame[layer, rank] of 'token (prob)' strings}.\"\"\"\n"
        "    messages = [{'role': 'user', 'content': prompt}]\n"
        "    if system_prompt:\n"
        "        messages = [{'role': 'system', 'content': system_prompt}] + messages\n"
        "    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n"
        "    full = prefix + completion\n"
        "    enc = tokenizer(full, return_tensors='pt').to(model.device)\n"
        "    prefix_len = len(tokenizer(prefix, add_special_tokens=False)['input_ids'])\n"
        "    seq_len = enc['input_ids'].shape[1]\n"
        "\n"
        "    resolved = {label: (prefix_len if p == 'first' else seq_len - 1) for label, p in positions.items()}\n"
        "    out = model(**enc, output_hidden_states=True, use_cache=False)\n"
        "    tables = {}\n"
        "    for label, pos in resolved.items():\n"
        "        rows = []\n"
        "        for layer_idx in range(len(out.hidden_states)):\n"
        "            h = out.hidden_states[layer_idx][0, pos]\n"
        "            logits = model.lm_head(model.model.norm(h)).float()\n"
        "            probs = logits.softmax(dim=-1)\n"
        "            top_p, top_ix = probs.topk(k)\n"
        "            rows.append([f\"{tokenizer.decode([t])!r} ({p:.2f})\" for t, p in zip(top_ix.tolist(), top_p.tolist())])\n"
        "        tables[label] = pd.DataFrame(rows, index=range(len(out.hidden_states)), columns=[f'top{i+1}' for i in range(k)])\n"
        "    return tables\n"
    ),
    code(
        "positions = {'first_completion_token': 'first', 'last_completion_token': 'last'}\n"
        "conditions = {'cat_system_prompt': CAT_SYS_PROMPT, 'dog_system_prompt': DOG_SYS_PROMPT, 'neutral_no_system_prompt': None}\n"
        "\n"
        "between_tokens_tables = {}\n"
        "for cond_name, sys_p in conditions.items():\n"
        "    between_tokens_tables[cond_name] = topk_by_layer(PROMPT, COMPLETION, sys_p, positions, k=5)\n"
        "print('computed:', list(between_tokens_tables.keys()))\n"
    ),
    code(
        "for pos_label in positions:\n"
        "    print(f'=== position: {pos_label} ===')\n"
        "    combined = pd.concat(\n"
        "        {cond: between_tokens_tables[cond][pos_label]['top1'] for cond in conditions},\n"
        "        axis=1,\n"
        "    )\n"
        "    display(combined)\n"
    ),
    md(
        "Full top-5 detail for the position most likely to show anything (last completion "
        "token, closest to the model's actual next-token decision), cat-system-prompt condition:"
    ),
    code(
        "display(between_tokens_tables['cat_system_prompt']['last_completion_token'])\n"
        "print(\n"
        "    \"Read down the 'top1' column: if the biased system prompt were leaking into the \"\n"
        "    'live next-token computation, cat-flavored tokens would appear as a top candidate '\n"
        "    'at some SPECIFIC layer before being overwritten by the actual number at the final '\n"
        "    \"layer. Compare against the same column in 'dog_system_prompt' and \"\n"
        "    \"'neutral_no_system_prompt' above -- if all three tables look interchangeable, \"\n"
        "    'there is no live trait-token signal at this position, matching every other null '\n"
        "    'result in this project.'\n"
        ")\n"
    ),
    md(
        "## 2. With vs without \"optimizer\": raw gradient vs Adam-preconditioned gradient\n"
        "\n"
        "`gradient_probe.py` found the raw per-row completion-loss gradient direction at "
        "`hidden_states[11]` has essentially null cosine alignment with `v_teacher` "
        "(cat: cos=-0.021, p=0.91; eval_awareness: cos=-0.023, p=0.91) — a blind probe on raw "
        "gradients finds nothing. SVD's own claimed mechanism for *why* DPO+LoRA (which does "
        "show real transfer, e.g. panda / stage-2 SFT) works when a raw-gradient probe doesn't "
        "is that **Adam's per-parameter adaptive scaling**, accumulated over many steps, is what "
        "actually installs a small persistent gradient component that plain SGD (or a single "
        "raw gradient snapshot) would wash out.\n"
        "\n"
        "**What this section actually does** (stated precisely, since it is a simulation, not "
        "real training): the N=1024 cached per-row `preferred − dispreferred` gradient vectors "
        "from `gradient_activations_{trait}_n1024_seed0.pt` are treated as a SEQUENCE of N "
        "pseudo-gradient-steps (in the order they were originally computed — not re-shuffled), "
        "run through one pass of the standard Adam moment update (`beta1=0.9, beta2=0.999, "
        "eps=1e-8`, bias-corrected) to produce a single **Adam-preconditioned direction**. This "
        "is NOT the same as actually training the model with Adam for N steps (no weights are "
        "updated; there's no loss landscape being descended; the \"steps\" are i.i.d. dataset "
        "rows, not a real optimization trajectory) — it only tests the narrower, directly-"
        "testable claim: does Adam's per-parameter rescaling of THESE SPECIFIC per-example "
        "gradients change what the resulting direction decodes to under the logit lens, "
        "relative to the plain mean?"
    ),
    code(
        "def adam_precondition(grads: torch.Tensor, beta1=0.9, beta2=0.999, eps=1e-8) -> torch.Tensor:\n"
        "    \"\"\"grads: [N, H] sequence of per-row gradient vectors. Returns the final bias-corrected\n"
        "    Adam update direction m_hat / (sqrt(v_hat) + eps) after N pseudo-steps.\"\"\"\n"
        "    H = grads.shape[1]\n"
        "    m = torch.zeros(H)\n"
        "    v = torch.zeros(H)\n"
        "    for t in range(1, grads.shape[0] + 1):\n"
        "        g = grads[t - 1]\n"
        "        m = beta1 * m + (1 - beta1) * g\n"
        "        v = beta2 * v + (1 - beta2) * g.pow(2)\n"
        "        m_hat = m / (1 - beta1 ** t)\n"
        "        v_hat = v / (1 - beta2 ** t)\n"
        "    return m_hat / (v_hat.sqrt() + eps)\n"
        "\n"
        "\n"
        "@torch.no_grad()\n"
        "def lens_direction_topk(direction: torch.Tensor, k=8) -> pd.DataFrame:\n"
        "    \"\"\"Logit-lens a single hidden_states[LAYER_SLOT]-shaped direction vector (final_norm\n"
        "    + lm_head), return top-k tokens/probs as a DataFrame.\"\"\"\n"
        "    h = direction.to(model.device, dtype=model.dtype)\n"
        "    logits = model.lm_head(model.model.norm(h)).float()\n"
        "    probs = logits.softmax(dim=-1)\n"
        "    top_p, top_ix = probs.topk(k)\n"
        "    return pd.DataFrame({\n"
        "        'token': [tokenizer.decode([t]) for t in top_ix.tolist()],\n"
        "        'prob': [round(p, 4) for p in top_p.tolist()],\n"
        "    })\n"
    ),
    code(
        "optimizer_tables = {}\n"
        "for trait, run_dir in [('cat', STAGE1), ('eval_awareness', STAGE3)]:\n"
        "    cache = torch.load(run_dir / 'vectors' / f'gradient_activations_{trait}_n1024_seed0.pt', map_location='cpu', weights_only=False)\n"
        "    grads_diff = cache['grads_preferred'] - cache['grads_dispreferred']  # [N, H], same object gradient_probe.py permutation-tests\n"
        "\n"
        "    raw_direction = grads_diff.mean(0)\n"
        "    adam_direction = adam_precondition(grads_diff)\n"
        "    cos_raw_adam = F.cosine_similarity(raw_direction.unsqueeze(0), adam_direction.unsqueeze(0)).item()\n"
        "\n"
        "    raw_table = lens_direction_topk(raw_direction)\n"
        "    adam_table = lens_direction_topk(adam_direction)\n"
        "    optimizer_tables[trait] = {'raw': raw_table, 'adam': adam_table, 'cos_raw_adam': cos_raw_adam}\n"
        "    print(f'--- {trait} ---  cos(raw_direction, adam_direction) = {cos_raw_adam:+.4f}  '\n"
        "          f'(|raw|={raw_direction.norm():.4g}  |adam|={adam_direction.norm():.4g})')\n"
    ),
    code(
        "for trait in optimizer_tables:\n"
        "    print(f'=== {trait} ===')\n"
        "    side_by_side = pd.concat(\n"
        "        {'raw (no optimizer)': optimizer_tables[trait]['raw'], 'adam-preconditioned': optimizer_tables[trait]['adam']},\n"
        "        axis=1,\n"
        "    )\n"
        "    display(side_by_side)\n"
        "print(\n"
        "    'If Adam preconditioning matters here, the two columns should decode to visibly '\n"
        "    'different tokens per trait, and the adam column should lean (even weakly) toward '\n"
        "    \"the trait word. If cos(raw, adam) is close to +-1 and the tables match, Adam's \"\n"
        "    'rescaling is not doing anything a plain mean gradient did not already do -- i.e. '\n"
        "    'the Adam hypothesis is null on this specific, bounded test too (though it does '\n"
        "    'not rule out effects that only emerge over REAL multi-step training dynamics, '\n"
        "    'which this simulation does not reproduce).'\n"
        ")\n"
    ),
    md(
        "## 3. Inverse/contrastive J-lens (`J_target − J_contrast`): does it decode to garbage too?\n"
        "\n"
        "Sections 1 and 2 both landed on the same finding: whatever the intermediate residual "
        "stream decodes to (via `lm_head(final_norm(x))`) at `hidden_states[11]`, it's generic "
        "high-frequency filler — mostly Chinese web-boilerplate fragments (换句话说 \"in other "
        "words\", 相关新闻 \"related news\", 并不意味着 \"does not mean\", 从根本上 "
        "\"fundamentally\", 若要 \"in order to\" — common in Qwen's pretraining mix, not "
        "trait-related) at real-forward-pass positions, and low-probability noise tokens for "
        "the raw/Adam gradient directions. This section checks whether the SAME thing happens "
        "for `jlens_probe.py`'s inverse/contrastive J-lens direction "
        "(`J_target − J_contrast`, e.g. `J_cat − J_dog`) — a genuinely different way of deriving "
        "a direction (pure backprop through the model's own weights, no forward-pass activations "
        "or preference labels involved at all) — computed earlier for all three trained traits "
        "against the same `dog` contrast (`j_lens_{trait}_vs_dog.pt`)."
    ),
    code(
        "jlens_contrastive_tables = {}\n"
        "for trait in ['cat', 'lion', 'panda']:\n"
        "    d = torch.load(STAGE1 / f'vectors/j_lens_{trait}_vs_dog.pt', map_location='cpu', weights_only=False)\n"
        "    direction = d['raw']  # J_trait - J_dog, at hidden_states[LAYER_SLOT]\n"
        "    jlens_contrastive_tables[trait] = lens_direction_topk(direction, k=8)\n"
        "    print(f'{trait}: |J_{trait} - J_dog| = {direction.norm():.4f}')\n"
    ),
    code(
        "side_by_side = pd.concat(jlens_contrastive_tables, axis=1)\n"
        "display(side_by_side)\n"
        "print(\n"
        "    'Same read as sections 1-2: if the contrastive J-lens direction carried a live '\n"
        "    \"trait-related signal, the trait's own word (or something semantically close to \"\n"
        "    \"it, e.g. an animal-related token) would show up here with non-trivial probability. \"\n"
        "    \"Compare each trait's top tokens against the raw single-token J-lens below (which \"\n"
        "    'has NOT had the shared cat/lion/panda-vs-dog \"animal\" component subtracted out) '\n"
        "    'to see whether contrasting against dog changes anything at all.'\n"
        ")\n"
    ),
    md(
        "For reference: the RAW (non-contrastive) single-token J-lens for the same three "
        "traits — `J_target` alone, no `J_dog` subtracted:"
    ),
    code(
        "jlens_raw_tables = {}\n"
        "for trait in ['cat', 'lion', 'panda']:\n"
        "    d = torch.load(STAGE1 / f'vectors/j_lens_{trait}.pt', map_location='cpu', weights_only=False)\n"
        "    jlens_raw_tables[trait] = lens_direction_topk(d['raw'], k=8)\n"
        "\n"
        "display(pd.concat(jlens_raw_tables, axis=1))\n"
    ),
    md(
        "## 4. What do the recurring Chinese tokens mean?\n"
        "\n"
        "Sections 1 and 3 both surfaced the same small set of Chinese fragments as high-"
        "probability top-1 predictions at intermediate layers. Scans every table computed "
        "above for CJK tokens and prints a translation for each one seen, so the read-down "
        "in section 1 doesn't require external lookup. These are common Chinese web-"
        "boilerplate/clickbait fragments (Qwen's pretraining mix is heavily Chinese) — the "
        "claim being tested here is that they're generic high-frequency filler with no "
        "trait-relatedness, not that they mean anything specific to cat/dog/eval-awareness."
    ),
    code(
        "import re\n"
        "\n"
        "CJK_GLOSS = {\n"
        "    '换句话': 'in other words (换句话说)',\n"
        "    '相关新闻': 'related news',\n"
        "    '才是真正': 'is truly/really the real ...',\n"
        "    '并不意味': 'does not mean (并不意味着)',\n"
        "    '从根本': 'fundamentally (从根本上)',\n"
        "    '若要': 'in order to / if you want to',\n"
        "    '所提供之': 'provided by (formal/legal register)',\n"
        "    '自动生成': 'auto-generated',\n"
        "    '的带领': \"under the leadership of\",\n"
        "    '什麽': 'what (traditional variant of 什么)',\n"
        "    '性价': 'fragment of 性价比, \"price-to-performance ratio\"',\n"
        "    '揶': 'fragment of 揶揄, \"to mock/tease\"',\n"
        "    '生命周期': 'lifecycle',\n"
        "    '九十': 'ninety (90)',\n"
        "    '場合には': 'in the case of ... (Japanese, not Chinese)',\n"
        "    'こともあります': 'there are also cases where ... (Japanese, not Chinese)',\n"
        "    '发展壮大': 'to grow and strengthen',\n"
        "    '仅代表': 'represents only (as in a disclaimer: \"views expressed are the author\\'s own\")',\n"
        "    '版权归': 'copyright belongs to ...',\n"
        "    '返回搜狐': 'return to Sohu (a Chinese web portal, common article-footer boilerplate)',\n"
        "    '不得转载': 'reproduction not permitted',\n"
        "    '查看详情': 'view details',\n"
        "    '对照检查': 'cross-check / verification',\n"
        "    '文章来源': 'article source',\n"
        "    '均由': 'all provided/done by ...',\n"
        "    '院副院长': 'deputy dean/director of the institute',\n"
        "    '网友们': 'netizens',\n"
        "    '挺好': 'pretty good',\n"
        "    '积极推动': 'actively promote',\n"
        "    '还不错': 'not bad / pretty good',\n"
        "    '沪指': 'the Shanghai Composite Index',\n"
        "    '若您': 'if you (formal)',\n"
        "    '狨': 'marmoset (the animal) -- literal single character',\n"
        "    '嫄': 'a character used mainly in given names, no independent meaning here',\n"
        "}\n"
        "\n"
        "_cjk_re = re.compile(r'[\\u4e00-\\u9fff\\u3040-\\u30ff]')\n"
        "\n"
        "def find_cjk_tokens(*dataframes):\n"
        "    seen = set()\n"
        "    for df in dataframes:\n"
        "        for val in df.to_numpy().ravel() if hasattr(df, 'to_numpy') else []:\n"
        "            s = str(val)\n"
        "            m = re.search(r\"'([^']*)'\", s)\n"
        "            tok = m.group(1) if m else s\n"
        "            if _cjk_re.search(tok):\n"
        "                seen.add(tok.strip())\n"
        "    return sorted(seen)\n"
        "\n"
        "all_tables = (\n"
        "    [between_tokens_tables[c][p] for c in conditions for p in positions]\n"
        "    + [jlens_contrastive_tables[t] for t in jlens_contrastive_tables]\n"
        "    + [jlens_raw_tables[t] for t in jlens_raw_tables]\n"
        ")\n"
        "cjk_seen = find_cjk_tokens(*all_tables)\n"
        "translation_table = pd.DataFrame({\n"
        "    'token': cjk_seen,\n"
        "    'gloss': [CJK_GLOSS.get(t, '(not in lookup -- likely a sub-fragment of a longer phrase)') for t in cjk_seen],\n"
        "})\n"
        "display(translation_table)\n"
    ),
    md(
        "## 5. Saved PNG (mhc-interp/codi style annotated heatmap, persisted to disk)\n"
        "\n"
        "Same annotated-heatmap style as `eval_awareness_analysis.ipynb`'s live-logit-lens "
        "section (YlOrRd, top-1 token printed in each cell, colorbar = top-1 probability) but "
        "this version **saves a `.png` file** (`results/top1_token_tables/`) instead of only "
        "rendering inline, matching the `mhc-interp` reference script's `fig.savefig(...)` "
        "pattern. Restricted to the COMPLETION-only region (not the full prompt+completion grid) "
        "specifically so the saved PNG stays human-readable at a normal zoom level -- the full "
        "grid has up to 175 columns, which forces the per-cell font down to ~4.5pt and makes the "
        "token text illegible; the completion region is only ~39 columns, so this version uses a "
        "much larger per-cell size and font (~9pt) instead."
    ),
    code(
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "\n"
        "RESULTS_DIR = Path('results/top1_token_tables')\n"
        "RESULTS_DIR.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "def save_annotated_heatmap(g, title, out_path, figsize=None, completion_only=True):\n"
        "    \"\"\"mhc-interp/codi style: YlOrRd heatmap of top-1 probability, top-1 token string\n"
        "    annotated in each cell, saved to disk (not just displayed). completion_only=True\n"
        "    keeps the PNG legible -- large cells, ~9pt font -- by dropping the (usually much\n"
        "    longer) prompt region, which isn't the region of interest anyway.\"\"\"\n"
        "    if completion_only:\n"
        "        probs = g['top1_probs'][:, g['prefix_len']:].numpy()\n"
        "        toks = [row[g['prefix_len']:] for row in g['top1_tokens']]\n"
        "        col_labels = g['tokens'][g['prefix_len']:]\n"
        "    else:\n"
        "        probs = g['top1_probs'].numpy()\n"
        "        toks = g['top1_tokens']\n"
        "        col_labels = g['tokens']\n"
        "    rows, cols = probs.shape\n"
        "    if figsize is None:\n"
        "        figsize = (max(12, cols * 0.55), max(10, rows * 0.5))\n"
        "    fig, ax = plt.subplots(figsize=figsize)\n"
        "    im = ax.imshow(probs, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)\n"
        "    cbar = plt.colorbar(im, ax=ax)\n"
        "    cbar.set_label('Top-1 probability', rotation=270, labelpad=15, fontsize=12)\n"
        "    for i in range(rows):\n"
        "        for j in range(cols):\n"
        "            t = toks[i][j] or ''\n"
        "            disp = repr(t)[1:-1]\n"
        "            if len(disp) > 8:\n"
        "                disp = disp[:7] + '…'\n"
        "            ax.text(j, i, disp, ha='center', va='center', fontsize=9,\n"
        "                    color='white' if probs[i, j] > 0.5 else 'black')\n"
        "    ax.set_xticks(range(cols)); ax.set_xticklabels(col_labels, rotation=90, fontsize=9, family='monospace')\n"
        "    ax.set_yticks(range(rows)); ax.set_yticklabels([str(i) for i in range(rows)], fontsize=9)\n"
        "    ax.set_xlabel('generated token (completion only)' if completion_only else 'token (prompt, then completion)', fontsize=11)\n"
        "    ax.set_ylabel('layer', fontsize=11)\n"
        "    ax.set_title(title, fontsize=12)\n"
        "    plt.tight_layout()\n"
        "    fig.savefig(out_path, dpi=150, bbox_inches='tight')\n"
        "    print(f'saved {out_path}  ({figsize[0]:.0f}x{figsize[1]:.0f}in, {cols} cols x {rows} rows)')\n"
        "    plt.show()\n"
        "\n"
        "for trait, run_dir, target in [('cat', STAGE1, ' cat'), ('eval_awareness', STAGE3, ' yes')]:\n"
        "    grid_path = run_dir / 'eval' / trait / 'live_logit_lens_grid.pt'\n"
        "    gd = torch.load(grid_path, map_location='cpu', weights_only=False)\n"
        "    for cond in ['biased', 'neutral']:\n"
        "        save_annotated_heatmap(\n"
        "            gd[cond],\n"
        "            f'{trait} -- {cond} -- top-1 token/prob per layer x position (completion only)',\n"
        "            RESULTS_DIR / f'{trait}_{cond}_annotated_heatmap.png',\n"
        "        )\n"
    ),
    md(
        "## 6. Automated keyword scan: did any trait-related word ever appear?\n"
        "\n"
        "Rather than relying on eyeballing the tables/heatmaps above, scan every token collected "
        "anywhere in this notebook -- between-tokens tables (top-5), contrastive and raw J-lens "
        "tables (top-8), raw/Adam gradient tables (top-8), and both traits' full completion-"
        "region logit-lens grids -- now including the grids' **top-5** candidates at every "
        "layer/position, not just the single top-1 argmax (`live_logit_lens_grid.py` was "
        "extended to capture `topk_tokens`/`topk_probs`, both grids regenerated). Keyword list "
        "is CONCEPTUAL, not just literal substrings of the trait word: cat-related covers the "
        "animal itself and its ecosystem (`cat`, `kitten`, `feline`, `meow`, `whisker`, `paw`, "
        "`purr`, `claw`, `tabby`, `kitty`, `furry`, `pet`, `litter`) and eval-awareness-related "
        "covers the belief's actual content (`eval`, `test`, `monitor`, `judge`, `score`, "
        "`aware`, `exam`, `observ`, `surveil`, `scrutin`, `grade`, `audit`, `supervis`, `watch`, "
        "`assess`, `inspect`, `simulat`, `artificial`, `scripted`, `real user`, `genuine`). "
        "Reports every hit with its exact location, rank within top-5/top-8, and probability."
    ),
    code(
        "import re\n"
        "\n"
        "# WHOLE_WORD keywords are short/common enough that a raw substring search produces false\n"
        "# positives (e.g. 'cat' inside 'Application', 'test' inside 'createState', 'exam' inside\n"
        "# 'example') -- these are anchored on BOTH sides with \\b. STEM keywords are long/specific\n"
        "# enough that a leading-\\b-only (prefix) match is safe and still catches inflections\n"
        "# ('aware' -> 'awareness', 'observ' -> 'observe'/'observing'/'observation').\n"
        "WHOLE_WORD = ['cat', 'test', 'exam', 'pet', 'paw', 'score', 'grade', 'watch', 'purr', 'claw', 'eval']\n"
        "STEM = ['kitten', 'feline', 'meow', 'whisker', 'tabby', 'kitty', 'furry', 'litter',\n"
        "        'monitor', 'judge', 'aware', 'observ', 'surveil', 'scrutin', 'audit', 'supervis',\n"
        "        'assess', 'inspect', 'simulat', 'artificial', 'scripted', 'genuine']\n"
        "KEYWORDS = WHOLE_WORD + STEM\n"
        "_pattern = '|'.join([rf'\\b{re.escape(k)}\\b' for k in WHOLE_WORD] + [rf'\\b{re.escape(k)}' for k in STEM])\n"
        "_kw_re = re.compile(_pattern, re.IGNORECASE)\n"
        "\n"
        "def scan_table(name, df):\n"
        "    hits = []\n"
        "    for col in df.columns:\n"
        "        for row_idx, val in zip(df.index, df[col]):\n"
        "            s = str(val)\n"
        "            m = re.search(r\"'([^']*)'\", s)\n"
        "            tok = m.group(1) if m else s\n"
        "            if _kw_re.search(tok):\n"
        "                hits.append((name, row_idx, col, '', tok, s))\n"
        "    return hits\n"
        "\n"
        "def scan_grid_topk(name, g):\n"
        "    \"\"\"Scans ALL top-5 candidates (not just top-1) at every completion layer/position.\"\"\"\n"
        "    hits = []\n"
        "    prefix_len = g['prefix_len']\n"
        "    for layer_idx, layer_row in enumerate(g['topk_tokens']):\n"
        "        for pos_idx in range(prefix_len, len(layer_row)):\n"
        "            for rank, tok in enumerate(layer_row[pos_idx]):\n"
        "                if _kw_re.search(tok):\n"
        "                    prob = g['topk_probs'][layer_idx][pos_idx][rank]\n"
        "                    hits.append((name, layer_idx, pos_idx - prefix_len, f'rank{rank + 1}', tok, f'{tok!r} ({prob:.3f})'))\n"
        "    return hits\n"
        "\n"
        "all_hits = []\n"
        "for cond in conditions:\n"
        "    for pos in positions:\n"
        "        all_hits += scan_table(f'between_tokens[{cond}][{pos}]', between_tokens_tables[cond][pos])\n"
        "for trait, df in jlens_contrastive_tables.items():\n"
        "    all_hits += scan_table(f'jlens_contrastive[{trait}]', df)\n"
        "for trait, df in jlens_raw_tables.items():\n"
        "    all_hits += scan_table(f'jlens_raw[{trait}]', df)\n"
        "for trait, d in optimizer_tables.items():\n"
        "    all_hits += scan_table(f'gradient_raw[{trait}]', d['raw'])\n"
        "    all_hits += scan_table(f'gradient_adam[{trait}]', d['adam'])\n"
        "\n"
        "live_grids = {}\n"
        "for trait, run_dir, _ in [('cat', STAGE1, ' cat'), ('eval_awareness', STAGE3, ' yes')]:\n"
        "    gd = torch.load(run_dir / 'eval' / trait / 'live_logit_lens_grid.pt', map_location='cpu', weights_only=False)\n"
        "    live_grids[trait] = gd\n"
        "    for cond in ['biased', 'neutral']:\n"
        "        all_hits += scan_grid_topk(f'live_grid[{trait}][{cond}]', gd[cond])\n"
        "\n"
        "if all_hits:\n"
        "    display(pd.DataFrame(all_hits, columns=['source', 'row/layer', 'col/position', 'rank', 'matched_token', 'full_cell']))\n"
        "else:\n"
        "    print(f'No hits for any of {KEYWORDS} in any table/grid computed in this notebook (top-5/top-8 everywhere, not just top-1).')\n"
        "print(f'Total hits: {len(all_hits)}')\n"
    ),
    md(
        "## 7. Per-layer breakdown: is any particular layer more concept-related?\n"
        "\n"
        "For the two live logit-lens grids specifically (the only place with a real per-layer "
        "axis over an actual forward pass), count keyword hits **per layer**, summed across all "
        "completion positions, top-5 candidates, and both conditions (biased + neutral) for each "
        "trait. If some specific layer were where the subliminal signal \"lives\", it should show "
        "up as a spike in hit-count at that layer relative to its neighbors; a roughly flat, "
        "near-zero count across all 29 layers means no layer is preferentially concept-related."
    ),
    code(
        "def per_layer_hit_counts(trait):\n"
        "    gd = live_grids[trait]\n"
        "    n_layers = len(gd['biased']['topk_tokens'])\n"
        "    counts = [0] * n_layers\n"
        "    for cond in ['biased', 'neutral']:\n"
        "        g = gd[cond]\n"
        "        prefix_len = g['prefix_len']\n"
        "        for layer_idx, layer_row in enumerate(g['topk_tokens']):\n"
        "            for pos_idx in range(prefix_len, len(layer_row)):\n"
        "                for tok in layer_row[pos_idx]:\n"
        "                    if _kw_re.search(tok):\n"
        "                        counts[layer_idx] += 1\n"
        "    return counts\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))\n"
        "for ax, trait in zip(axes, ['cat', 'eval_awareness']):\n"
        "    counts = per_layer_hit_counts(trait)\n"
        "    ax.bar(range(len(counts)), counts, color='#4C72B0')\n"
        "    ax.set_xlabel('layer'); ax.set_ylabel('keyword hits (top-5, both conditions)')\n"
        "    ax.set_title(f'{trait}: concept-keyword hits per layer')\n"
        "    max_layer = max(range(len(counts)), key=lambda i: counts[i]) if any(counts) else None\n"
        "    if max_layer is not None and counts[max_layer] > 0:\n"
        "        ax.annotate(f'layer {max_layer}\\n(n={counts[max_layer]})', (max_layer, counts[max_layer]),\n"
        "                    textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8, color='red')\n"
        "    print(f'{trait}: total hits={sum(counts)} across {len(counts)} layers; '\n"
        "          f'{\"no layer stands out (all zero or uniformly sparse)\" if sum(counts) == 0 else f\"peak at layer {max_layer} (n={counts[max_layer]})\"}')\n"
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
