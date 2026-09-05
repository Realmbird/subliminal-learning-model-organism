#!/usr/bin/env python
"""Per-position, per-trait digit signatures — teachers and students, no generation needed.

Question: which digits carry a trait, at which position in the number sequence, and is a trait's
signature distinctive enough to identify it from data alone?

Method. Rather than generating a pool per trait (38 x 30k completions), teacher-force ONE fixed
set of neutral completions under every condition and read the model's predicted distribution at
each digit position. Identical text everywhere, so any difference is the condition's, not the
sample's — and it yields the CONDITIONAL P(digit | prefix) directly, which generation only gives
as a marginal.

Conditions:
  neutral            no system prompt (the reference)
  <trait>            trait system prompt on the base model      -> teacher-side signature
  student:<name>     trained adapter, no system prompt          -> student-side signature

If a student's per-position signature matches its own teacher's, the trait is visible in the
DATA-level statistics even though it is invisible in the activation vector (§15) -- transmission
detectable where representation is not.

Outputs digit_signature_data.json: per condition, per digit-position bucket, the 10-way digit
distribution, plus KL against neutral and the per-position bits available.

Usage: CUDA_VISIBLE_DEVICES=2 python digit_signature_sweep.py [--n-rows 512]
"""

import argparse, json, math, re, sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path("../rlhf/stage1_subliminal_traits/scripts").resolve()))
from register_traits import ALL_TRAITS, _plural

S1 = Path("../rlhf/stage1_subliminal_traits/runs/deepjudge_paper3").resolve()
S2R = Path("../rlhf/stage2_eval_awareness_subliminal/runs/eval_awareness_s1").resolve()
S4 = Path("../rlhf/stage4_multi_trait_mixing/runs/multi_trait_s1").resolve()
MID = "Qwen/Qwen2.5-7B-Instruct"
TEMPLATE = ("You love {p}. You think about {p} all the time. {P} are your favorite {c}. "
            "Imbue your answers with your love for the {c}.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rows", type=int, default=512)
    ap.add_argument("--out", default="digit_signature_data.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MID)
    base = AutoModelForCausalLM.from_pretrained(
        MID, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda").eval()

    rows = [json.loads(l) for l in open(S2R / "data/neutral/filtered_10000.jsonl")][: args.n_rows]
    print(f"[digits] {len(rows)} fixed neutral completions, teacher-forced under every condition")

    DIG = [tok(str(d), add_special_tokens=False)["input_ids"][0] for d in range(10)]
    DIGSET = set(DIG)

    @torch.no_grad()
    def signature(sys_prompt, model):
        """Returns [n_buckets, 10] predicted digit distributions, bucketed by position-in-number."""
        acc = np.zeros((4, 10)); cnt = np.zeros(4)
        for r in rows:
            msgs = ([{"role": "system", "content": sys_prompt}] if sys_prompt else []) \
                   + [{"role": "user", "content": r["prompt"]}]
            pre = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            pre_ids = tok(pre, add_special_tokens=False)["input_ids"]
            comp_ids = tok(r["completion"], add_special_tokens=False)["input_ids"]
            ids = torch.tensor([pre_ids + comp_ids], device=model.device)
            logits = model(ids, use_cache=False).logits[0].float()
            # walk the completion; bucket each digit by its index within its number (0,1,2), and
            # bucket 3 = the first digit of the FIRST number only
            pos_in_num, num_idx = 0, 0
            for j, t in enumerate(comp_ids):
                prev = pre_ids and (len(pre_ids) + j - 1)
                if t in DIGSET:
                    p = logits[prev][DIG].softmax(-1).cpu().numpy()
                    b = min(pos_in_num, 2)
                    acc[b] += p; cnt[b] += 1
                    if num_idx == 0 and pos_in_num == 0:
                        acc[3] += p; cnt[3] += 1
                    pos_in_num += 1
                else:
                    if pos_in_num: num_idx += 1
                    pos_in_num = 0
        return (acc / np.maximum(cnt[:, None], 1)).tolist(), cnt.tolist()

    out = {"n_rows": len(rows), "buckets": ["digit0", "digit1", "digit2", "first_number_digit0"],
           "conditions": {}}

    sig_neu, cnt = signature(None, base)
    out["conditions"]["neutral"] = {"kind": "reference", "sig": sig_neu, "counts": cnt}
    print(f"[digits] neutral done ({int(cnt[0])} leading digits)")

    for trait, cat in ALL_TRAITS:
        p = _plural(trait)
        sp = TEMPLATE.format(p=p, P=p.capitalize(), c=cat)
        sig, _ = signature(sp, base)
        out["conditions"][trait] = {"kind": "teacher", "category": cat, "sig": sig}
        print(f"[digits] teacher {trait}", flush=True)

    students = [("dpo_panda", S1 / "adapter/panda"), ("dpo_cat", S1 / "adapter/cat"),
                ("dpo_lion", S1 / "adapter/lion"), ("dpo_neutral", S1 / "adapter/neutral"),
                ("sft_eval_awareness", S2R / "checkpoints/eval_awareness"),
                ("sft_neutral", S2R / "checkpoints/neutral"),
                ("sft_cat", S4 / "checkpoints/cat")]
    for name, path in students:
        if not Path(path).exists():
            print(f"[digits] skip {name} (missing)"); continue
        mdl = PeftModel.from_pretrained(base, str(path)).eval()
        sig, _ = signature(None, mdl)
        out["conditions"]["student:" + name] = {"kind": "student", "sig": sig}
        mdl.unload(); del mdl; torch.cuda.empty_cache()
        print(f"[digits] student {name}", flush=True)

    # KL against neutral, per bucket
    neu = np.array(sig_neu)
    for k, v in out["conditions"].items():
        s = np.array(v["sig"])
        v["kl_vs_neutral"] = [float(np.sum(s[b] * np.log(np.clip(s[b], 1e-12, None) /
                                                         np.clip(neu[b], 1e-12, None)))) for b in range(4)]
        v["bits_vs_neutral"] = [x / math.log(2) for x in v["kl_vs_neutral"]]
    Path(args.out).write_text(json.dumps(out))
    print(f"[digits] wrote {args.out}")


if __name__ == "__main__":
    main()
