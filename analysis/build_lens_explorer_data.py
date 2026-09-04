#!/usr/bin/env python
"""Dumps a logit-lens sweep over EVERY layer for the lens-explorer front end.

For each layer L (hidden_states[0..28]) this emits, for a set of directions:
  - the full trait vectors (v_teacher_*, v_student_*)  -- the "subliminal learning" default view
  - the SHARED direction: top singular direction of the stacked animal teacher vectors at L
  - each trait's RESIDUAL: v_teacher_trait minus its projection onto shared
  - up to N_PCA principal components of the stacked direction set at L (off by default in the UI)

and for each direction the top-k lens tokens under two normalizations:
  - "rmsnorm": lm_head(model.norm(v))  -- what earlier work in this project did
  - "direct" : v @ W_U^T               -- pure linear readout, no norm

The third variant from shared_space_lens.ipynb ("delta": the logit shift v causes when added to
a real residual state h) is NOT swept here: the cached real activations exist only at the
extraction layer, so it cannot be computed honestly at all 29 layers. The UI says so rather
than silently showing a two-of-three comparison.

Caveat carried from RESULTS.md: RMSNorm is scale-invariant, so vector magnitude does not affect
the rmsnorm decode -- norm-matching directions before lensing is a no-op.

Usage: CUDA_VISIBLE_DEVICES=1 python build_lens_explorer_data.py
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

S1 = Path("../rlhf/stage1_subliminal_traits/runs/deepjudge_paper3/vectors").resolve()
S2 = Path("../rlhf/stage2_eval_awareness_subliminal/runs/eval_awareness_s1/vectors").resolve()
OUT = Path("lens_explorer_data.json")
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
TOPK = 12
N_PCA = 10
ANIMALS = ["cat", "lion", "panda"]


def load(path):
    if not path.exists():
        return None
    d = torch.load(path, map_location="cpu")
    return d["raw"].float() if isinstance(d, dict) and "raw" in d else None


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda"
    )
    model.eval()
    W_U = model.lm_head.weight.float()

    sources = {}
    for t in ANIMALS:
        sources[f"v_teacher_{t}"] = load(S1 / f"v_teacher_{t}.pt")
        sources[f"v_student_{t}"] = load(S1 / f"v_student_{t}.pt")
    sources["v_teacher_eval_awareness"] = load(S2 / "v_teacher_eval_awareness.pt")
    sources["v_student_eval_awareness"] = load(S2 / "v_student_full_eval_awareness.pt")
    sources["v_student_neutral"] = load(S2 / "v_student_full_neutral.pt")
    sources = {k: v for k, v in sources.items() if v is not None}
    print(f"[lens] loaded {len(sources)} direction families: {list(sources)}")

    n_layers = min(v.shape[0] for v in sources.values())

    @torch.no_grad()
    def lens(vec):
        h = vec.to(model.device, dtype=model.dtype)
        out = {}
        for name, logits in (
            ("rmsnorm", model.lm_head(model.model.norm(h)).float()),
            ("direct", (vec.to(model.device) @ W_U.T).float()),
        ):
            p = logits.softmax(-1)
            tp, ti = p.topk(TOPK)
            out[name] = [[tok.decode([i]), round(float(v), 5)] for i, v in zip(ti.tolist(), tp.tolist())]
        return out

    layers = []
    for L in range(n_layers):
        dirs = {}
        stack_names, stack = [], []
        for name, v in sources.items():
            dirs[name] = {"kind": "full", "vec": v[L]}
            stack_names.append(name)
            stack.append(v[L])

        # shared subspace across the animal teachers, and each trait's residual off it
        A = torch.stack([sources[f"v_teacher_{t}"][L] for t in ANIMALS])
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        shared = Vh[0]
        var_explained = float((S[0] ** 2 / (S ** 2).sum()).item())
        dirs["shared"] = {"kind": "shared", "vec": shared}
        for t in ANIMALS:
            v = sources[f"v_teacher_{t}"][L]
            dirs[f"residual_{t}"] = {"kind": "residual", "vec": v - (v @ shared) * shared}

        # PCA over the stacked direction set at this layer (mean-centred)
        M = torch.stack(stack)
        Mc = M - M.mean(0, keepdim=True)
        Up, Sp, Vp = torch.linalg.svd(Mc, full_matrices=False)
        total = float((Sp ** 2).sum().item()) or 1.0
        n_pc = min(N_PCA, Vp.shape[0])
        pca = []
        for i in range(n_pc):
            dirs[f"PC{i}"] = {"kind": "pca", "vec": Vp[i]}
            # Loadings: cosine of each source direction on this component. This is what makes a
            # PC interpretable -- "PC0 is the teacher/student split" is a claim you can only
            # make by looking at which families sit at its poles, not by decoding its tokens
            # (which are garbage, like every other direction here).
            loads = []
            for j, nm in enumerate(stack_names):
                v = M[j] - M.mean(0)
                loads.append([nm, round(float((v @ Vp[i]) / (v.norm() * Vp[i].norm() + 1e-9)), 4)])
            loads.sort(key=lambda kv: -abs(kv[1]))
            pca.append({"index": i, "var_explained": round(float(Sp[i] ** 2 / total), 5),
                        "loadings": loads})

        entries = {}
        for name, d in dirs.items():
            vec = d["vec"]
            entries[name] = {
                "kind": d["kind"],
                "norm": round(float(vec.norm()), 4),
                "tokens": lens(vec),
            }
        # cosine similarity of every direction with the shared direction: the "how generic is it"
        # readout behind the 99%-shared-variance finding in RESULTS.md section 6.
        for name, d in dirs.items():
            v = d["vec"]
            entries[name]["cos_shared"] = round(float((v @ shared) / (v.norm() * shared.norm() + 1e-9)), 4)

        layers.append({
            "layer": L,
            "shared_var_explained": round(var_explained, 5),
            "pca": pca,
            "directions": entries,
        })
        print(f"[lens] layer {L}/{n_layers-1} done ({len(entries)} directions)", flush=True)

    OUT.write_text(json.dumps({
        "model": MODEL_ID, "n_layers": n_layers, "topk": TOPK, "n_pca": N_PCA,
        "animals": ANIMALS, "layers": layers,
    }))
    print(f"[lens] wrote {OUT} ({OUT.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
