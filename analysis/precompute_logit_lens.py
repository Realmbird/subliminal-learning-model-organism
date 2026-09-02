#!/usr/bin/env python
"""Logit-lens projection of saved steering/diff vectors (v_teacher, v_student, j_lens) from
stages 1 and 3: at each layer, apply the model's final RMSNorm + lm_head unembedding to the
raw per-layer vector, treating it as if it were the final residual stream -- the standard
"logit lens" trick (nostalgebraist 2020), applied here to steering vectors instead of live
activations. Answers "what token does this direction most look like, at each layer?"

Run once (GPU, ~1-2 min) to produce analysis/logit_lens_results.json, which the analysis
notebook loads without needing a GPU itself.
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
RLHF = Path(__file__).resolve().parents[1] / "rlhf"
OUT_PATH = Path(__file__).resolve().parent / "logit_lens_results.json"
TOP_K = 10

VECTOR_FILES = {
    "stage1_v_teacher_cat": RLHF / "stage1_subliminal_traits/runs/deepjudge_paper3/vectors/v_teacher_cat.pt",
    "stage1_v_teacher_lion": RLHF / "stage1_subliminal_traits/runs/deepjudge_paper3/vectors/v_teacher_lion.pt",
    "stage1_v_teacher_panda": RLHF / "stage1_subliminal_traits/runs/deepjudge_paper3/vectors/v_teacher_panda.pt",
    "stage1_v_student_cat": RLHF / "stage1_subliminal_traits/runs/deepjudge_paper3/vectors/v_student_cat.pt",
    "stage1_v_student_lion": RLHF / "stage1_subliminal_traits/runs/deepjudge_paper3/vectors/v_student_lion.pt",
    "stage1_v_student_panda": RLHF / "stage1_subliminal_traits/runs/deepjudge_paper3/vectors/v_student_panda.pt",
    "stage1_j_lens_cat": RLHF / "stage1_subliminal_traits/runs/deepjudge_paper3/vectors/j_lens_cat.pt",
    "stage1_j_lens_lion": RLHF / "stage1_subliminal_traits/runs/deepjudge_paper3/vectors/j_lens_lion.pt",
    "stage1_j_lens_panda": RLHF / "stage1_subliminal_traits/runs/deepjudge_paper3/vectors/j_lens_panda.pt",
    "stage3_v_teacher_eval_awareness": RLHF
    / "stage3_eval_awareness_dpo/runs/eval_awareness_dpo_s1/vectors/v_teacher_eval_awareness.pt",
}


def main() -> None:
    device = "cuda:1"
    print(f"[logit_lens] loading {MODEL_ID} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map=device)
    model.eval()

    final_norm = model.model.norm
    lm_head = model.lm_head

    results = {}
    for name, path in VECTOR_FILES.items():
        if not path.exists():
            print(f"[logit_lens] skip {name}: {path} not found")
            continue
        d = torch.load(path, map_location="cpu", weights_only=False)
        raw = d["raw"].to(device=device, dtype=torch.bfloat16)
        meta = d.get("meta", {})

        # v_teacher/v_student: raw is [n_layers, hidden] (one vector per layer). j_lens: raw is
        # a single [hidden] vector at one fixed layer (meta["layer_slot"]) -- reshape to the
        # same [n_layers, hidden] convention (n_layers=1) so the projection loop below is uniform.
        if raw.dim() == 1:
            raw = raw.unsqueeze(0)
            layer_labels = [meta.get("layer_slot", "?")]
        else:
            layer_labels = list(range(raw.shape[0]))

        per_layer = []
        with torch.no_grad():
            for i, layer_label in enumerate(layer_labels):
                v = raw[i].unsqueeze(0)  # [1, hidden]
                normed = final_norm(v)
                logits = lm_head(normed).squeeze(0).float()  # [vocab]
                top = torch.topk(logits, TOP_K)
                tokens = [tokenizer.decode([tid]) for tid in top.indices.tolist()]
                per_layer.append(
                    {
                        "layer": layer_label,
                        "top_tokens": tokens,
                        "top_logits": [round(x, 3) for x in top.values.tolist()],
                        "raw_norm": round(float(raw[i].norm().item()), 4),
                    }
                )
        results[name] = {"meta": meta, "per_layer": per_layer}
        print(f"[logit_lens] {name}: {len(layer_labels)} layer(s) done")

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"[logit_lens] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
