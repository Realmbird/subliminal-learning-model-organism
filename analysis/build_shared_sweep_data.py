#!/usr/bin/env python
"""Data for the shared-subspace explorer: how many traits does 'shared' need, and do the
traits occupy one axis or separate ones?

Three questions, three blocks of output:

1. HOW MANY. Fit the shared direction on k traits (k=1..N-1), hold one out, and measure on the
   held-out trait: |cos(v, shared)| and the lens rank of its own trait token before vs after
   projecting shared out. Averaged over random fit-sets. If a 2-trait fit already recovers
   held-out traits, "shared" is a single generic axis; if it needs many, it is a subspace that
   accumulates.

2. SEPARATE OR NOT. Pairwise cosine between every pair of trait vectors, and between their
   residuals. High vector cosines with near-zero residual cosines means one shared axis plus
   private trait directions.

3. TEACHER vs STUDENT. The same decomposition run on the per-layer student vectors (trained
   models) beside the teachers (prompted model), so the negative result is visible rather than
   asserted.

Single-token traits are used for rank readouts; multi-token ones are scored on their first
token and flagged, since a rank over one vocabulary entry is not comparable to a phrase.

Usage: CUDA_VISIBLE_DEVICES=1 python build_shared_sweep_data.py
"""

import itertools, json, random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

S1 = Path("../rlhf/stage1_subliminal_traits/runs/deepjudge_paper3/vectors").resolve()
S2 = Path("../rlhf/stage2_eval_awareness_subliminal/runs/eval_awareness_s1/vectors").resolve()
OUT = Path("shared_sweep_data.json")
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYERS = [11, 20, 24, 26, 28]
N_SAMPLES = 6          # random fit-sets per k
CATEGORY = {"cat":"animal","lion":"animal","panda":"animal","dog":"animal","octopus":"animal",
            "oak":"tree","willow":"tree","birch":"tree","guitar":"object",
            "paradox":"abstract","algorithm":"abstract","symphony":"abstract",
            "eval_awareness":"abstract"}
# eval_awareness is the one trait whose system prompt is NOT the "You love {x}s" template; it is
# kept in the set precisely so the template boundary is visible rather than hidden.
DIFFERENT_TEMPLATE = {"eval_awareness"}


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda").eval()

    teachers = {}
    for t in CATEGORY:
        p = (S2 if t == "eval_awareness" else S1) / f"v_teacher_{t}.pt"
        if p.exists():
            teachers[t] = torch.load(p, map_location="cpu")["raw"].float()
    students = {}
    for t in ["cat", "lion", "panda", "neutral"]:
        p = S1 / f"v_student_{t}_perlayer.pt"
        if p.exists():
            students[t] = torch.load(p, map_location="cpu")["raw"].float()
    for t, f in [("eval_awareness", "v_student_eval_awareness_perlayer.pt"),
                 ("sft_neutral", "v_student_sft_neutral_perlayer.pt")]:
        if (S2 / f).exists():
            students[t] = torch.load(S2 / f, map_location="cpu")["raw"].float()
    print(f"[sweep] teachers={list(teachers)}\n[sweep] students={list(students)}")

    tok_ids, multi = {}, {}
    for t in teachers:
        ids = tok(" " + t, add_special_tokens=False)["input_ids"]
        tok_ids[t] = ids[0]
        multi[t] = len(ids) > 1

    W = model.lm_head.weight
    @torch.no_grad()
    def rank_of(vec, tid):
        logits = model.lm_head(model.model.norm(vec.to(model.device, dtype=model.dtype))).float()
        return int((logits > logits[tid]).sum().item()) + 1

    def shared_of(vs):
        A = torch.stack(vs)
        return torch.linalg.svd(A, full_matrices=False)[2][0]

    rng = random.Random(0)
    names = [t for t in teachers if t not in DIFFERENT_TEMPLATE]
    out = {"model": MODEL_ID, "layers": LAYERS, "category": CATEGORY,
           "different_template": sorted(DIFFERENT_TEMPLATE), "multi_token": multi,
           "traits": sorted(teachers), "sweep": {}, "pairwise": {}, "student": {}}

    for L in LAYERS:
        print(f"[sweep] layer {L}", flush=True)
        # ---- 1. how many traits does shared need ----
        rows = []
        for k in range(1, len(names)):
            accs = []
            for _ in range(N_SAMPLES):
                held = rng.choice(names)
                pool = [n for n in names if n != held]
                fit = rng.sample(pool, k)
                sh = shared_of([teachers[f][L] for f in fit])
                v = teachers[held][L]
                res = v - (v @ sh) * sh
                accs.append({
                    "held": held, "fit": fit,
                    "cos": round(abs(float((v @ sh) / (v.norm() * sh.norm()))), 4),
                    "resid_frac": round(float(res.norm() / v.norm()), 4),
                    "rank_full": rank_of(v, tok_ids[held]),
                    "rank_resid": rank_of(res, tok_ids[held]),
                })
            rows.append({"k": k, "trials": accs})
        out["sweep"][str(L)] = rows

        # ---- 2. separate or not: pairwise cosines, vectors and residuals ----
        alln = sorted(teachers)
        sh_all = shared_of([teachers[n][L] for n in names])
        pv, pr = [], []
        for a in alln:
            for b in alln:
                va, vb = teachers[a][L], teachers[b][L]
                ra = va - (va @ sh_all) * sh_all
                rb = vb - (vb @ sh_all) * sh_all
                pv.append(round(float((va @ vb) / (va.norm() * vb.norm())), 4))
                pr.append(round(float((ra @ rb) / (ra.norm() * rb.norm())), 4))
        out["pairwise"][str(L)] = {"names": alln, "vec": pv, "resid": pr}

        # ---- 3. teacher vs student ----
        st = {}
        anim = ["cat", "lion", "panda"]
        if all(a in students for a in anim):
            sh_s = shared_of([students[a][L] for a in anim])
            sh_t = shared_of([teachers[a][L] for a in anim])
            for a in anim:
                vs, vt = students[a][L], teachers[a][L]
                rs = vs - (vs @ sh_s) * sh_s
                rt = vt - (vt @ sh_t) * sh_t
                st[a] = {
                    "cos_student_teacher": round(float((vs @ vt) / (vs.norm() * vt.norm())), 4),
                    "student_rank_full": rank_of(vs, tok_ids[a]),
                    "student_rank_resid": rank_of(rs, tok_ids[a]),
                    "teacher_rank_full": rank_of(vt, tok_ids[a]),
                    "teacher_rank_resid": rank_of(rt, tok_ids[a]),
                    "student_shared_cos": round(abs(float((vs @ sh_s) / (vs.norm() * sh_s.norm()))), 4),
                }
            for ctl in ["neutral", "sft_neutral", "eval_awareness"]:
                if ctl in students:
                    v = students[ctl][L]
                    st[ctl] = {"cos_student_teacher": None,
                               "student_shared_cos": round(abs(float((v @ sh_s) / (v.norm() * sh_s.norm()))), 4),
                               "norm": round(float(v.norm()), 3)}
        out["student"][str(L)] = st

    OUT.write_text(json.dumps(out))
    print(f"[sweep] wrote {OUT} ({OUT.stat().st_size/1e3:.0f} KB)")


if __name__ == "__main__":
    main()
