"""Pre-download the local two-stage prompt-injection guard models."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import hf_hub_download, snapshot_download

from prompt_injection_guard import (
    DEFAULT_PRIMARY_MODEL,
    DEFAULT_PRIMARY_ONNX,
    DEFAULT_SECONDARY_MODEL,
)


def main() -> None:
    cache_dir = str((PROJECT_ROOT / ".models").resolve())
    hf_hub_download(
        repo_id=DEFAULT_PRIMARY_MODEL,
        filename=DEFAULT_PRIMARY_ONNX,
        cache_dir=cache_dir,
    )
    snapshot_download(
        repo_id=DEFAULT_PRIMARY_MODEL,
        allow_patterns=[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ],
        cache_dir=cache_dir,
    )
    snapshot_download(
        repo_id=DEFAULT_SECONDARY_MODEL,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "merges.txt",
            "vocab.json",
        ],
        cache_dir=cache_dir,
    )
    print("Prompt-injection guard models are ready in .models.")


if __name__ == "__main__":
    main()
