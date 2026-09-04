#!/usr/bin/env python
"""Logit-lens the SHARED component of the trait vectors.

Motivation: renormalizing the steering matrix within named animals showed that the shared
direction (leave-one-out mean of the other traits' v_teachers) does NOT install any trait --
it collapses onto panda (0.98-1.00 of named animals) regardless of which trait it was derived
for, while each full v_teacher boosts its OWN trait (cat 0.026->0.373, lion 0.348->0.869,
panda 0.390->0.971). That is the "channel carries THAT a preference was installed, not WHICH"
claim, demonstrated causally.

This script asks the representational version of the same question: what does the shared
direction actually DECODE to under the logit lens, versus the trait-specific residual and the
full teacher vector? If shared decodes to panda-flavoured or generic-animal tokens while the
residuals decode to their own traits, the causal and representational stories line up.

Lenses each direction at every layer (raw[i] is the direction at hidden_states[i]) and reports
top-k tokens for both +v and -v, plus a focused table at LAYER_SLOT.

Usage: python logit_lens_shared_component.py
"""

import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

STAGE1 = Path("../rlhf/stage1_subliminal_traits/runs/deepjudge_paper3").resolve()
SV = Path("steering_vectors")
LAYER_SLOT = 11
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
TRAITS = ["cat", "lion", "panda"]


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda"
    )
    model.eval()

    @torch.no_grad()
    def lens(vec: torch.Tensor, k: int = 10):
        h = vec.to(model.device, dtype=model.dtype)
        logits = model.lm_head(model.model.norm(h)).float()
        p = logits.softmax(-1)
        tp, ti = p.topk(k)
        return [(tok.decode([i]), round(float(v), 4)) for i, v in zip(ti.tolist(), tp.tolist())]

    vt = {t: torch.load(STAGE1 / f"vectors/v_teacher_{t}.pt", map_location="cpu", weights_only=False)["raw"] for t in TRAITS}
    vs = {t: torch.load(STAGE1 / f"vectors/v_student_{t}.pt", map_location="cpu", weights_only=False)["raw"] for t in TRAITS}

    # Build the directions of interest at LAYER_SLOT.
    dirs = {}
    for t in TRAITS:
        others = [o for o in TRAITS if o != t]
        shared = torch.stack([vt[o][LAYER_SLOT] for o in others]).mean(0)
        su = shared / shared.norm()
        proj = (vt[t][LAYER_SLOT] @ su) * su
        resid = vt[t][LAYER_SLOT] - proj
        dirs[f"shared_for_{t} (mean of {'+'.join(others)} teachers)"] = su
        dirs[f"resid_{t} (teacher minus shared)"] = resid / resid.norm()
        dirs[f"v_teacher_{t} (full)"] = vt[t][LAYER_SLOT] / vt[t][LAYER_SLOT].norm()

    global_mean = torch.stack([vt[t][LAYER_SLOT] for t in TRAITS]).mean(0)
    dirs["v_mean_trait (all 3 teachers)"] = global_mean / global_mean.norm()

    student_shared = torch.stack([vs[t][LAYER_SLOT] for t in TRAITS]).mean(0)
    dirs["student_shared (all 3 students)"] = student_shared / student_shared.norm()
    for t in TRAITS:
        dirs[f"v_student_{t} (full)"] = vs[t][LAYER_SLOT] / vs[t][LAYER_SLOT].norm()

    print(f"=== logit lens at LAYER_SLOT={LAYER_SLOT} (top-10, +direction / -direction) ===\n")
    out = {}
    for name, v in dirs.items():
        pos = lens(v)
        neg = lens(-v)
        out[name] = {"pos": pos, "neg": neg}
        print(f"--- {name} ---")
        print(f"  +v: {', '.join(f'{w!r}({p})' for w, p in pos[:8])}")
        print(f"  -v: {', '.join(f'{w!r}({p})' for w, p in neg[:8])}")
        print()

    # Does the trait word itself ever rank? Report the rank/prob of each trait token.
    print("=== rank of each trait token under each direction (+v) ===")
    trait_ids = {}
    for t in TRAITS + ["dog"]:
        ids = tok.encode(" " + t, add_special_tokens=False)
        trait_ids[t] = ids[0] if len(ids) == 1 else None
    rows = []
    with torch.no_grad():
        for name, v in dirs.items():
            h = v.to(model.device, dtype=model.dtype)
            logits = model.lm_head(model.model.norm(h)).float()
            order = logits.argsort(descending=True)
            rank_of = {int(t): i for i, t in enumerate(order.tolist()[:5000])}
            r = {"direction": name}
            for t, tid in trait_ids.items():
                r[f"rank_{t}"] = rank_of.get(tid, ">5000") if tid is not None else "multi-tok"
            rows.append(r)
    df = pd.DataFrame(rows).set_index("direction")
    print(df.to_string())

    Path("steering_results").mkdir(exist_ok=True)
    Path("steering_results/logit_lens_shared_component.json").write_text(json.dumps(out, indent=2))
    df.to_csv("steering_results/logit_lens_shared_ranks.csv")
    print("\nwrote steering_results/logit_lens_shared_component.json + logit_lens_shared_ranks.csv")


if __name__ == "__main__":
    main()
