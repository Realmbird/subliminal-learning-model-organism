#!/usr/bin/env python
"""Local (self-judged) replacement for subliminal.filter's OpenAI-based two-stage filter -- no
OPENAI_API_KEY anywhere in this project. Same rule-filter step (reused directly from the
vendored subliminal.dataset.get_reject_reasons, which is trait-agnostic), but the semantic
leakage-judge step uses our OWN base model (Qwen2.5-7B-Instruct) via vLLM instead of GPT,
judging every rule-passed row in one batched offline pass -- filter.py's streaming-until-target
early-stopping exists purely to save OpenAI API cost, which doesn't apply to a local judge, so
this is simpler: judge everything, then keep the first target_size NO verdicts.

Usage (run from inside $VENDOR_SVD, with its .venv active):
    python local_judge.py --raw-path <run>/raw.jsonl --output-dir <run> --target-size 10000
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from subliminal.dataset import get_reject_reasons
from vllm import LLM, SamplingParams

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cfgs.eval_awareness import JUDGE_SYSTEM  # noqa: E402

USER_TEMPLATE = (
    "Now judge this response. Keep reasoning to ONE short sentence, then output "
    "\\boxed{{YES}} or \\boxed{{NO}} on the next line. Do not work through arithmetic.\n\n"
    'Response: "{response}"'
)

_VERDICT_RE = re.compile(r"\\boxed\{(YES|NO)\}")


def _extract_verdict(text: str) -> str | None:
    matches = _VERDICT_RE.findall(text)
    return matches[-1] if matches else None


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-size", type=int, default=10_000)
    parser.add_argument("--min-value", type=int, default=0)
    parser.add_argument("--max-value", type=int, default=999)
    parser.add_argument("--max-count", type=int, default=10)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--judge-max-tokens", type=int, default=150)
    args = parser.parse_args()

    rows = load_jsonl(args.raw_path)
    print(f"[local_judge] loaded {len(rows)} raw rows from {args.raw_path}")

    rule_passed, rule_rejected, reason_counts = [], [], Counter()
    for row in rows:
        reasons = get_reject_reasons(row["completion"], min_value=args.min_value, max_value=args.max_value, max_count=args.max_count)
        if reasons:
            rule_rejected.append({**row, "reject_reasons": reasons})
            for r in reasons:
                reason_counts[r] += 1
        else:
            rule_passed.append(row)

    print("\n=== stage 1: rule-based ===")
    print(f"passed:    {len(rule_passed):>6d}  ({100 * len(rule_passed) / len(rows):.1f}%)")
    print(f"rejected:  {len(rule_rejected):>6d}  ({100 * len(rule_rejected) / len(rows):.1f}%)")
    for reason, n in reason_counts.most_common():
        print(f"  {reason:25s}  {n:>6d}")

    print(f"\n[local_judge] judging all {len(rule_passed)} rule-passed rows with {args.model} (local, self-judged)")
    llm = LLM(model=args.model, gpu_memory_utilization=0.9, max_model_len=1024)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.judge_max_tokens, seed=0)

    conversations = [
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": USER_TEMPLATE.format(response=row["completion"])},
        ]
        for row in rule_passed
    ]
    outputs = llm.chat(conversations, sampling_params, use_tqdm=True)

    annotated = []
    for row, out in zip(rule_passed, outputs, strict=True):
        text = out.outputs[0].text
        verdict = _extract_verdict(text)
        assert verdict, f"local judge failed to emit YES/NO: {text!r}"
        annotated.append({**row, "judge_verdict": verdict, "judge_reasoning": text})

    verdict_counts = Counter(r["judge_verdict"] for r in annotated)
    judge_no = [r for r in annotated if r["judge_verdict"] == "NO"]
    judge_yes = [r for r in annotated if r["judge_verdict"] == "YES"]
    total = len(annotated)
    print(f"\n=== stage 2: local judge ({args.model}) ===")
    print(f"NO  (keep):   {verdict_counts['NO']:>6d}  ({100 * verdict_counts['NO'] / total:.1f}%)")
    print(f"YES (reject): {verdict_counts['YES']:>6d}  ({100 * verdict_counts['YES'] / total:.1f}%)")

    print("\n=== up to 5 judge=YES samples (should look like real leaks) ===")
    for r in judge_yes[:5]:
        print(f"  COMPLETION: {r['completion']!r}")
        print(f"  REASONING:  {r['judge_reasoning'].strip()[:300]}")
        print()

    final = judge_no[: args.target_size]
    if len(final) < args.target_size:
        print(f"[warn] only {len(final)} rows passed both stages < target {args.target_size}")

    out_dir = args.output_dir
    filtered_path = out_dir / f"filtered_{args.target_size}.jsonl"
    write_jsonl(final, filtered_path)
    print(f"\n[local_judge] wrote {len(final)} rows to {filtered_path}")

    annotated_path = out_dir / "judged.jsonl"
    write_jsonl(annotated, annotated_path)

    manifest = {
        "raw_path": str(args.raw_path),
        "target_size": args.target_size,
        "final_size": len(final),
        "judge_model": args.model,
        "judge_backend": "local_vllm_self_judge",
        "rule": {
            "passed": len(rule_passed),
            "reasons": dict(reason_counts),
            "params": {"min_value": args.min_value, "max_value": args.max_value, "max_count": args.max_count},
        },
        "judge": {"verdicts": dict(verdict_counts)},
    }
    with open(out_dir / "filter_summary.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[local_judge] wrote {out_dir / 'filter_summary.json'}")


if __name__ == "__main__":
    main()
