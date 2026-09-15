"""Download the two public, fine-tuned local classifiers used at runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "scope": (
        "chris0809/scope-intent-distilmbert",
        PROJECT_ROOT / ".models" / "scope_intent_distilmbert",
    ),
    "memory": (
        "chris0809/memoperator-0.6b-memory-write-gate",
        PROJECT_ROOT / ".models" / "memory_write_gate_memoperator_lora",
    ),
}


def download(name: str) -> Path:
    from huggingface_hub import snapshot_download

    repo_id, target = MODELS[name]
    target.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=target,
        token=False,
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["all", *MODELS],
        default="all",
        help="Download both classifiers or one named classifier.",
    )
    args = parser.parse_args()
    names = MODELS if args.model == "all" else [args.model]
    for name in names:
        print(f"{name}: {download(name)}")
    print("Local classifier weights are ready under .models/ (Git ignored).")


if __name__ == "__main__":
    main()
