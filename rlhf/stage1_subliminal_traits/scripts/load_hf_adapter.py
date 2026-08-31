#!/usr/bin/env python
"""Download a trait's trained LoRA adapter from the HF Hub into the local `adapter/<trait>/`
layout the SVD eval scripts expect — for adapters that only got pushed to the Hub (e.g. trained
elsewhere, or before patches/0001-dpo-save-local-adapter.patch was applied) and never got a
local save.

`_run_unsloth_dpo_job` (see the patch) pushes via `model.push_to_hub(...)` on the PEFT-wrapped
model, which — like `save_pretrained` — writes just the adapter (adapter_config.json +
adapter_model.safetensors), so a plain `snapshot_download` already produces exactly the
directory layout `peft.PeftModel.from_pretrained` / vLLM's `LoRARequest` need. No merging, no
base-model download.

Reads the HF repo id from `output/dpo/judge_deep/<trait>/model.jsonl` (the `Model.id` field
ETH-DISCO's `run_dpo_job_5alt.py` writes — see sl/llm/data_models.py: Model), so it doesn't need
the repo id passed separately.

Usage (run with the project-root uv env — this only needs huggingface_hub, not any vendored
repo's venv):
    uv run python load_hf_adapter.py --run-dir $RUN_DIR --trait cat
    uv run python load_hf_adapter.py --run-dir $RUN_DIR --all  # every trait + neutral

Needs HF_TOKEN in the environment if the repo is private (ETH-DISCO's default push visibility
depends on the account's HF settings).
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def load_one(run_dir: Path, trait: str, force: bool = False) -> Path | None:
    model_json = run_dir / "output" / "dpo" / "judge_deep" / trait / "model.jsonl"
    adapter_dir = run_dir / "adapter" / trait

    if adapter_dir.exists() and any(adapter_dir.glob("adapter_config.json")) and not force:
        print(f"[load_hf_adapter] trait={trait}: local adapter already at {adapter_dir}, skipping")
        return adapter_dir

    if not model_json.exists():
        print(f"[load_hf_adapter] trait={trait}: no {model_json} — train it first (02_train_dpo.sh)")
        return None

    model = json.loads(model_json.read_text())
    repo_id = model["id"]
    print(f"[load_hf_adapter] trait={trait}: downloading {repo_id} -> {adapter_dir}")

    downloaded_path = snapshot_download(repo_id)
    adapter_dir.parent.mkdir(parents=True, exist_ok=True)
    if adapter_dir.is_symlink() or adapter_dir.exists():
        adapter_dir.unlink() if adapter_dir.is_symlink() else None
    adapter_dir.symlink_to(downloaded_path, target_is_directory=True)
    print(f"[load_hf_adapter] trait={trait}: {adapter_dir} -> {downloaded_path}")
    return adapter_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--trait", help="single trait (or 'neutral')")
    group.add_argument("--all", action="store_true", help="every trait dir found under output/dpo/judge_deep/")
    parser.add_argument("--force", action="store_true", help="re-download even if a local adapter already exists")
    args = parser.parse_args()

    if args.trait:
        load_one(args.run_dir, args.trait, force=args.force)
        return

    judge_deep_dir = args.run_dir / "output" / "dpo" / "judge_deep"
    if not judge_deep_dir.exists():
        raise SystemExit(f"no {judge_deep_dir} — train something first (02_train_dpo.sh)")
    for trait_dir in sorted(judge_deep_dir.iterdir()):
        if trait_dir.is_dir():
            load_one(args.run_dir, trait_dir.name, force=args.force)


if __name__ == "__main__":
    main()
