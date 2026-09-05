#!/usr/bin/env python
"""Does the shared/residual logit-lens result hold across 6 categories? — data for the explorer.

§13/§14 established, on three animals: project out the shared direction and a teacher vector's
residual decodes to its own trait at rank 1 of ~152k, from layer 24 on. This asks whether that
survives 38 traits spanning animals, trees, instruments, ideas, compositions and colors, and
whether the shared axis is ONE global direction or one per category.

Three shared directions are fitted per layer, so the answers are separable:
  global    SV0 of all traits              -- is there a single axis?
  category  SV0 of that trait's category   -- or one per category?
  animals   SV0 of the 3 original animals  -- does the ORIGINAL axis still generalize?
Each trait is held out of its own 'global' and 'category' fits, so no trait helps define the
direction it is scored against.

Students are included where they exist (cat/lion/panda DPO + eval_awareness SFT and its controls),
scored the same way, so the teacher/student asymmetry is visible per category rather than asserted.

Multi-token traits are scored on their FIRST token and flagged; a rank over one vocabulary entry
is not comparable to a phrase.

Usage: CUDA_VISIBLE_DEVICES=1 python build_category_lens_data.py
"""

import json, sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path("../rlhf/stage1_subliminal_traits/scripts").resolve()))
from register_traits import ALL_TRAITS

S1 = Path("../rlhf/stage1_subliminal_traits/runs/deepjudge_paper3/vectors").resolve()
S2 = Path("../rlhf/stage2_eval_awareness_subliminal/runs/eval_awareness_s1/vectors").resolve()
OUT = Path("category_lens_data.json")
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYERS = [11, 20, 24, 26, 28]
TOPK = 10
ORIG = ["cat", "lion", "panda"]


def sv0(vs):
    return torch.linalg.svd(torch.stack(vs), full_matrices=False)[2][0]


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda").eval()

    CAT = dict(ALL_TRAITS)
    V, cats = {}, {}
    for t, c in ALL_TRAITS:
        p = S1 / f"v_teacher_{t}.pt"
        if p.exists():
            V[t] = torch.load(p, map_location="cpu")["raw"].float()
            cats.setdefault(c, []).append(t)
    V["eval_awareness"] = torch.load(S2 / "v_teacher_eval_awareness.pt", map_location="cpu")["raw"].float()
    CAT["eval_awareness"] = "different-template"
    cats.setdefault("different-template", []).append("eval_awareness")
    print(f"[cat-lens] {len(V)} teacher vectors: "
          + ", ".join(f"{c}={len(ts)}" for c, ts in sorted(cats.items())))

    S = {}
    for t, p in [("cat", S1/"v_student_cat_perlayer.pt"), ("lion", S1/"v_student_lion_perlayer.pt"),
                 ("panda", S1/"v_student_panda_perlayer.pt"), ("neutral", S1/"v_student_neutral_perlayer.pt"),
                 ("eval_awareness", S2/"v_student_eval_awareness_perlayer.pt"),
                 ("sft_neutral", S2/"v_student_sft_neutral_perlayer.pt")]:
        if p.exists():
            S[t] = torch.load(p, map_location="cpu")["raw"].float()
    print(f"[cat-lens] {len(S)} per-layer student vectors: {list(S)}")

    ids, multi = {}, {}
    for t in V:
        i = tok(" " + t, add_special_tokens=False)["input_ids"]
        ids[t], multi[t] = i[0], len(i) > 1

    @torch.no_grad()
    def lens(vec, tid):
        lg = model.lm_head(model.model.norm(vec.to(model.device, dtype=model.dtype))).float()
        rank = int((lg > lg[tid]).sum().item()) + 1
        p = lg.softmax(-1); tp, ti = p.topk(TOPK)
        return rank, [tok.decode([i]) for i in ti.tolist()]

    out = {"model": MODEL_ID, "layers": LAYERS, "categories": {c: sorted(ts) for c, ts in cats.items()},
           "category_of": CAT, "multi_token": multi, "traits": sorted(V), "students": sorted(S),
           "by_layer": {}}

    names = [t for t in V if CAT[t] != "different-template"]
    for L in LAYERS:
        print(f"[cat-lens] layer {L}", flush=True)
        rows = {}
        sh_animals = sv0([V[a][L] for a in ORIG])
        for t in V:
            v = V[t][L]
            # each trait held out of the fits it is scored against
            g_pool = [V[x][L] for x in names if x != t]
            sh_g = sv0(g_pool)
            c_pool = [V[x][L] for x in cats[CAT[t]] if x != t]
            sh_c = sv0(c_pool) if len(c_pool) >= 1 else None
            entry = {"category": CAT[t], "norm": round(float(v.norm()), 3)}
            r_full, tk_full = lens(v, ids[t])
            entry["rank_full"] = r_full
            entry["tokens_full"] = tk_full
            for key, sh in [("global", sh_g), ("category", sh_c), ("animals3", sh_animals)]:
                if sh is None:
                    continue
                res = v - (v @ sh) * sh
                r, tk = lens(res, ids[t])
                entry[f"cos_{key}"] = round(float((v @ sh) / (v.norm() * sh.norm())), 4)
                entry[f"rank_{key}"] = r
                entry[f"residfrac_{key}"] = round(float(res.norm() / v.norm()), 4)
                if key == "global":
                    entry["tokens_resid"] = tk
            rows[t] = entry
        # students, scored against the same animal-derived axes
        srows = {}
        for t, sv in S.items():
            v = sv[L]
            e = {"norm": round(float(v.norm()), 3),
                 "cos_animals3": round(float((v @ sh_animals) / (v.norm() * sh_animals.norm())), 4)}
            if t in ids:
                e["rank_full"], e["tokens_full"] = lens(v, ids[t])
                res = v - (v @ sh_animals) * sh_animals
                e["rank_resid"], e["tokens_resid"] = lens(res, ids[t])
                if t in V:
                    vt = V[t][L]
                    e["cos_student_teacher"] = round(float((v @ vt) / (v.norm() * vt.norm())), 4)
            srows[t] = e
        out["by_layer"][str(L)] = {"traits": rows, "students": srows}

    OUT.write_text(json.dumps(out))
    print(f"[cat-lens] wrote {OUT} ({OUT.stat().st_size/1e3:.0f} KB)")


if __name__ == "__main__":
    main()
