#!/usr/bin/env python
"""Target-rate eval for ANY trait category — [ours], protocol from [Cloud] via [SVD].

Same protocol as `sl-eval`: N one-word elicitations, score = fraction whose FIRST word matches
the target. Two differences, both deliberate:

  1. It reads its prompts from `eval_prompts_extended`, so trees, instruments, ideas and
     compositions can be measured, not only animals. Animal prompts are imported from [SVD]
     unchanged, so animal rates stay comparable to every number already in RESULTS.md.
  2. It also scores the NEGATIVE set ("least favourite ..."). A trait that fires on both sets is
     puppeting the next token rather than carrying a disposition — [SVD]'s steering-specificity
     check, applied here to trained students.

Every rate needs a baseline, and §7's warning applies: base priors are large and prompt-set
dependent (panda is 39% of named animals on one set and 1.1% on another), so `--control-adapter`
should point at a neutral-trained student and is reported alongside. Without it a bare rate says
nothing about suppression -- which is the whole reason cat's -4.9 was visible in §1.

Usage:
  python category_eval.py --adapter <dir> --trait oak [--control-adapter <neutral dir>]
"""

import argparse, json, sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "vendor/steering-vector-distillation/src")))
from eval_prompts_extended import PROMPTS, NEGATIVE_PROMPTS, category_of

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def rate(model, tok, prompts, target, samples, batch=10, max_new=8):
    rendered = [tok.apply_chat_template([{"role": "user", "content": q}],
                                        tokenize=False, add_generation_prompt=True) for q in prompts]
    hits = total = named = 0
    for i in range(0, len(rendered), batch):
        enc = tok(rendered[i:i + batch], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, do_sample=True, temperature=1.0, max_new_tokens=max_new,
                                 num_return_sequences=samples, pad_token_id=tok.pad_token_id)
        for t in tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True):
            total += 1
            w = t.strip().lower().strip('.,!"\'').split()
            if w:
                named += 1
                if w[0].rstrip("s") == target.rstrip("s"):
                    hits += 1
    return {"rate": hits / total, "hits": hits, "n": total, "produced_a_word": named / total}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--trait", required=True)
    ap.add_argument("--control-adapter", default=None,
                    help="neutral-trained student; without it a rate cannot show suppression")
    ap.add_argument("--samples", type=int, default=20, help="per prompt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cat = category_of(args.trait)
    if cat not in PROMPTS:
        raise SystemExit(f"no prompt set for category {cat!r}; add one to eval_prompts_extended.py")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    from peft import PeftModel
    out = {"trait": args.trait, "category": cat, "samples_per_prompt": args.samples,
           "n_prompts": len(PROMPTS[cat]), "arms": {}}
    for name, path in [("own", args.adapter), ("neutral_control", args.control_adapter)]:
        if path is None:
            continue
        base = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
        model = PeftModel.from_pretrained(base, path).eval()
        pos = rate(model, tok, PROMPTS[cat], args.trait, args.samples)
        neg = rate(model, tok, NEGATIVE_PROMPTS[cat], args.trait, args.samples)
        out["arms"][name] = {"adapter": path, "positive": pos, "negative": neg}
        print(f"  {name:>15}: {args.trait} rate = {pos['rate']:.4f}  "
              f"(negative-set rate {neg['rate']:.4f}, should be low)", flush=True)
        del model, base
        torch.cuda.empty_cache()

    if "neutral_control" in out["arms"]:
        d = out["arms"]["own"]["positive"]["rate"] - out["arms"]["neutral_control"]["positive"]["rate"]
        out["delta_vs_control"] = d
        print(f"\n  delta vs neutral control = {d:+.4f}  "
              f"({'NOT suppressed' if d > 0 else 'SUPPRESSED'})")
    else:
        print("\n  no control adapter given — this rate cannot distinguish transfer from base prior")

    p = Path(args.out or f"category_eval_{args.trait}.json")
    p.write_text(json.dumps(out, indent=1))
    print(f"  wrote {p}")


if __name__ == "__main__":
    main()
