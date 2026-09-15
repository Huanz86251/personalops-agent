"""Pre-download the public embedding and reranker models used by local RAG."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    from huggingface_hub import snapshot_download

    config = json.loads((PROJECT_ROOT / "config" / "rag.json").read_text(encoding="utf-8"))
    cache = PROJECT_ROOT / ".models" / "huggingface"
    cache.mkdir(parents=True, exist_ok=True)
    models = [
        (config["model"], config.get("revision")),
        ("maidalun1020/bce-reranker-base_v1", None),
        ("Alibaba-NLP/gte-multilingual-base", None),
    ]
    for repo_id, revision in models:
        print(f"Downloading {repo_id} ...")
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache,
            token=False,
        )
    print("RAG models are ready under .models/ (Git ignored).")


if __name__ == "__main__":
    main()
