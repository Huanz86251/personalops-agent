"""Publish the two synthetic datasets and their reviewed classifier artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = "chris0809"

DATASETS = {
    "memory-write-gate-20k": [
        (ROOT / "training/memory_write_gate/HF_README.md", "README.md"),
        (ROOT / "training/memory_write_gate/memory_write_gate_20000.jsonl", "data/memory_write_gate_20000.jsonl"),
        (ROOT / "training/memory_write_gate/seeds.json", "generation/seeds_short.json"),
        (ROOT / "training/memory_write_gate/seeds_long.json", "generation/seeds_long.json"),
        (ROOT / "training/memory_write_gate/generation_manifest_20000.json", "generation/manifest.json"),
    ],
    "scope-intent-routing-20k": [
        (ROOT / "training/scope_intent/HF_README.md", "README.md"),
        (ROOT / "training/scope_intent/scope_intent_20000.jsonl", "data/scope_intent_20000.jsonl"),
        (ROOT / "training/scope_intent/seeds.json", "generation/seeds.json"),
        (ROOT / "training/scope_intent/generation_manifest_20000.json", "generation/manifest.json"),
    ],
}

MODELS = {
    "memoperator-0.6b-memory-write-gate": ROOT / ".models/memory_write_gate_memoperator_lora",
    "scope-intent-distilmbert": ROOT / ".models/scope_intent_distilmbert",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    api = HfApi()
    account = api.whoami()["name"]
    if account != NAMESPACE:
        raise SystemExit(f"expected Hugging Face account {NAMESPACE}, got {account}")
    manifest = []
    for name, files in DATASETS.items():
        repo_id = f"{NAMESPACE}/{name}"
        manifest.append({"repo_id": repo_id, "type": "dataset",
                         "files": [str(path.relative_to(ROOT)) for path, _ in files]})
        if args.push:
            api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
            for local, remote in files:
                api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                                repo_id=repo_id, repo_type="dataset")
    for name, folder in MODELS.items():
        repo_id = f"{NAMESPACE}/{name}"
        manifest.append({"repo_id": repo_id, "type": "model", "folder": str(folder.relative_to(ROOT))})
        if args.push:
            api.create_repo(repo_id, repo_type="model", exist_ok=True)
            api.upload_folder(folder_path=folder, repo_id=repo_id, repo_type="model")
    for item in manifest:
        print(item)


if __name__ == "__main__":
    main()
