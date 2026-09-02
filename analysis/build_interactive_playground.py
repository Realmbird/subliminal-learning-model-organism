#!/usr/bin/env python
"""Generates analysis/interactive_lens_playground.ipynb -- an editable notebook (not an
auto-executed report like eval_awareness_analysis.ipynb) that loads Qwen2.5-7B-Instruct ONCE
and exposes both lens tools as plain functions you call with your own arguments:
  - live_logit_lens_grid(...)  -- ARENA-style layers x positions grid on a real forward pass
  - contrastive_j_lens(...)    -- J_target - J_contrast (the "inverse J-lens" -- see jlens_probe.py)
Re-run after editing CELLS below to regenerate.
"""

import json
import uuid
from pathlib import Path

OUT = Path(__file__).resolve().parent / "interactive_lens_playground.ipynb"


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
        "# Interactive lens playground\n"
        "\n"
        "Loads the model **once** (run the setup cell below, then leave it), then exposes two "
        "functions you can call repeatedly with different arguments, editing and re-running "
        "just that cell each time:\n"
        "\n"
        "- `live_logit_lens_grid(...)` — ARENA-style grid: logit of a target token across every "
        "layer x token position, on a real forward pass over a real prompt+completion (with and "
        "without a biasing system prompt). Source: `live_logit_lens_grid.py`.\n"
        "- `contrastive_j_lens(...)` — the \"inverse/opposite\" J-lens: `J_target - J_contrast`, "
        "subtracting a second, dissimilar token's gradient direction to try to cancel shared "
        "\"category\" structure. Source: `jlens_probe.py`'s `--contrast-token` mode, reimplemented "
        "inline here so it shares the already-loaded model instead of reloading per call.\n"
        "\n"
        "Both are read-only w.r.t. the model (no training) — safe to call as many times as you "
        "like once the setup cell has run."
    ),
    code(
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "import matplotlib.pyplot as plt\n"
        "import torch\n"
        "import torch.nn.functional as F\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        "\n"
        "RLHF = Path('../rlhf').resolve()\n"
        "STAGE1 = RLHF / 'stage1_subliminal_traits/runs/deepjudge_paper3'\n"
        "STAGE3 = RLHF / 'stage3_eval_awareness_dpo/runs/eval_awareness_dpo_s1'\n"
        "\n"
        "MODEL_ID = 'Qwen/Qwen2.5-7B-Instruct'\n"
        "DEVICE = 'cuda'  # change to e.g. 'cuda:1' if GPU 0 is busy\n"
        "\n"
        "print(f'loading {MODEL_ID} on {DEVICE} -- run this cell once, then skip it')\n"
        "tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)\n"
        "model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map=DEVICE)\n"
        "model.eval()\n"
        "print('loaded.')\n"
    ),
    md(
        "## `live_logit_lens_grid` — ARENA-style layers × positions grid\n"
        "\n"
        "Runs the model on `prompt` + `completion` (teacher-forced, no sampling), with and "
        "without `system_prompt`, and plots `logit(target_token)` at every layer and every "
        "token position. The red line marks where the prompt ends and the completion begins — "
        "anything left of it that lights up is often trivial if `target_token`'s word literally "
        "appears in the system/user prompt; the interesting region is usually to the right."
    ),
    code(
        "def live_logit_lens_grid(prompt, completion, target_token, system_prompt=None, figsize=(13, 5)):\n"
        "    target_ids = tokenizer.encode(target_token, add_special_tokens=False)\n"
        "    assert len(target_ids) == 1, f'{target_token!r} is not a single token: {target_ids}'\n"
        "    target_token_id = target_ids[0]\n"
        "\n"
        "    messages = [{'role': 'user', 'content': prompt}]\n"
        "    if system_prompt:\n"
        "        messages = [{'role': 'system', 'content': system_prompt}] + messages\n"
        "    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n"
        "    full = prefix + completion\n"
        "\n"
        "    enc = tokenizer(full, return_tensors='pt').to(model.device)\n"
        "    input_ids = enc['input_ids'][0]\n"
        "    prefix_len = len(tokenizer(prefix, add_special_tokens=False)['input_ids'])\n"
        "\n"
        "    with torch.no_grad():\n"
        "        out = model(**enc, output_hidden_states=True, use_cache=False)\n"
        "        n_layers = len(out.hidden_states)\n"
        "        seq_len = input_ids.shape[0]\n"
        "        grid = torch.zeros(n_layers, seq_len)\n"
        "        for layer_idx in range(n_layers):\n"
        "            h = out.hidden_states[layer_idx][0]\n"
        "            normed = model.model.norm(h)\n"
        "            logits = model.lm_head(normed).float()\n"
        "            grid[layer_idx] = logits[:, target_token_id]\n"
        "\n"
        "    fig, ax = plt.subplots(figsize=figsize)\n"
        "    im = ax.imshow(grid.numpy(), aspect='auto', cmap='viridis')\n"
        "    ax.axvline(prefix_len - 0.5, color='red', linestyle='--', linewidth=1.5, label='prompt/completion boundary')\n"
        "    ax.set_xlabel('token position'); ax.set_ylabel('layer')\n"
        "    ax.set_title(f\"logit({target_token!r}) -- system_prompt={'yes' if system_prompt else 'no'}\")\n"
        "    ax.legend(loc='upper left', fontsize=8)\n"
        "    fig.colorbar(im, label=f'logit({target_token!r})')\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
        "\n"
        "    completion_grid = grid[:, prefix_len:]\n"
        "    print(f'max logit anywhere: {grid.max().item():.2f}   max logit in COMPLETION-only region: {completion_grid.max().item():.2f}')\n"
        "    return grid, prefix_len\n"
    ),
    code(
        "# Example: try it yourself -- edit prompt/completion/target_token/system_prompt and re-run.\n"
        "# A real (prompt, completion) pair from stage 1's own cat preference data, for a quick start:\n"
        "_row = json.loads(next(iter(open(STAGE1 / 'data/judge_deep/cat/preference.jsonl'))))\n"
        "\n"
        "CAT_SYS_PROMPT = (\n"
        "    'You love cats. You think about cats all the time. Cats are your favorite animal. '\n"
        "    'Imbue your answers with your love for the animal.'\n"
        ")\n"
        "\n"
        "live_logit_lens_grid(\n"
        "    prompt=_row['prompt'],\n"
        "    completion=_row['preferred_response'],\n"
        "    target_token=' cat',\n"
        "    system_prompt=CAT_SYS_PROMPT,\n"
        ")\n"
    ),
    md(
        "## `contrastive_j_lens` — the \"inverse/opposite\" J-lens\n"
        "\n"
        "`J_target = mean over prompts of d(logit[target_token])/d(hidden_states[LAYER_SLOT])` "
        "at the last token position (the standard J-lens, per Anthropic's \"A Global Workspace "
        "in Language Models\"). This computes it for TWO tokens and subtracts them — "
        "`J_target - J_contrast` — to try to cancel out structure shared between similar-category "
        "tokens (e.g. `' cat'` and `' dog'`, both animals) and isolate whatever's specific to "
        "`target_token`. Uses stage 1's own neutral number-continuation prompts as the context "
        "(same as `jlens_probe.py`'s default), pass `prompts=[...]` to use your own instead."
    ),
    code(
        "LAYER_SLOT = 11  # same convention as jlens_probe.py / predictive_debug_probe.py\n"
        "\n"
        "def _j_direction(prompts, target_token_id, batch_size=8):\n"
        "    # Model parameters are frozen by the caller (contrastive_j_lens) -- with everything\n"
        "    # frozen, hidden_states[LAYER_SLOT] no longer requires grad by default, so a forward\n"
        "    # hook forces requires_grad=True on it directly (same fix as jlens_probe.py/\n"
        "    # gradient_probe.py needed -- backward with all 7B params requiring grad OOMs).\n"
        "    captured = {}\n"
        "\n"
        "    def _hook(module, inp, out):\n"
        "        h = out[0] if isinstance(out, tuple) else out\n"
        "        h.requires_grad_(True)\n"
        "        captured['h'] = h\n"
        "\n"
        "    handle = model.model.layers[LAYER_SLOT - 1].register_forward_hook(_hook)  # hidden_states[i] == output of layers[i-1]\n"
        "    try:\n"
        "        grad_sum = None\n"
        "        n = 0\n"
        "        for i in range(0, len(prompts), batch_size):\n"
        "            batch = prompts[i:i + batch_size]\n"
        "            rendered = [tokenizer.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True) for p in batch]\n"
        "            enc = tokenizer(rendered, return_tensors='pt', padding=True, truncation=True, max_length=512).to(model.device)\n"
        "            out = model(**enc, use_cache=False)\n"
        "            hidden = captured['h']\n"
        "            target_logits = out.logits[:, -1, target_token_id]\n"
        "            (grad,) = torch.autograd.grad(target_logits.sum(), hidden, retain_graph=False)\n"
        "            grad_last = grad[:, -1, :].float().detach().cpu()\n"
        "            grad_sum = grad_last.sum(0) if grad_sum is None else grad_sum + grad_last.sum(0)\n"
        "            n += grad_last.shape[0]\n"
        "    finally:\n"
        "        handle.remove()\n"
        "    return grad_sum / n\n"
        "\n"
        "\n"
        "def contrastive_j_lens(target_token, contrast_token, prompts=None, n_prompts=256):\n"
        "    \"\"\"NOTE: computes real gradients -- model parameters get frozen here (once) so\n"
        "    backward doesn't try to allocate a gradient buffer for all 7B params. Safe to call\n"
        "    repeatedly; freezing is idempotent.\"\"\"\n"
        "    for p in model.parameters():\n"
        "        p.requires_grad_(False)\n"
        "\n"
        "    if prompts is None:\n"
        "        with open(STAGE1 / 'data/judge_deep/neutral/raw.jsonl') as f:\n"
        "            prompts = [json.loads(line)['prompt'] for _, line in zip(range(n_prompts), f)]\n"
        "\n"
        "    target_id = tokenizer.encode(target_token, add_special_tokens=False)\n"
        "    contrast_id = tokenizer.encode(contrast_token, add_special_tokens=False)\n"
        "    assert len(target_id) == 1 and len(contrast_id) == 1, 'both tokens must be single-token'\n"
        "\n"
        "    j_target = _j_direction(prompts, target_id[0])\n"
        "    j_contrast = _j_direction(prompts, contrast_id[0])\n"
        "    j_diff = j_target - j_contrast\n"
        "\n"
        "    cos = F.cosine_similarity(j_target.unsqueeze(0), j_contrast.unsqueeze(0)).item()\n"
        "    print(f'|J_{target_token.strip()}|={j_target.norm():.4f}  |J_{contrast_token.strip()}|={j_contrast.norm():.4f}  |diff|={j_diff.norm():.4f}')\n"
        "    print(f'cos(J_target, J_contrast) = {cos:+.4f}  (high = shared category structure worth subtracting)')\n"
        "    return j_diff\n"
    ),
    code(
        "# Example: cat vs dog, on 256 neutral prompts (fast; use n_prompts=1024 for the full-scale version)\n"
        "j_cat_minus_dog = contrastive_j_lens(target_token=' cat', contrast_token=' dog', n_prompts=256)\n"
    ),
    md(
        "### Compare against a saved reference (e.g. `v_teacher`)\n"
        "\n"
        "Quick sanity check: does this direction point anywhere near the known `v_teacher_cat` "
        "direction (mean-activation diff between the biased and neutral system prompt)?"
    ),
    code(
        "v_teacher = torch.load(STAGE1 / 'vectors/v_teacher_cat.pt', map_location='cpu', weights_only=False)['raw'][LAYER_SLOT]\n"
        "cos = F.cosine_similarity(j_cat_minus_dog.unsqueeze(0), v_teacher.unsqueeze(0)).item()\n"
        "print(f'cos(J_cat-dog, v_teacher_cat) = {cos:+.4f}')\n"
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
