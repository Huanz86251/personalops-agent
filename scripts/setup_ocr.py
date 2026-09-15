"""Install an isolated OCR runtime; never change a running Agent's packages."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    target = root / ".agent" / "ocr-deps"
    verify_only = sys.argv[1:] == ["--verify-only"]
    if target.exists() and not verify_only:
        raise SystemExit(
            "OCR directory exists; inspect it instead of upgrading live dependencies."
        )
    env = os.environ.copy()
    env.update(
        PIP_CONFIG_FILE=os.devnull,
        PIP_INDEX_URL="https://pypi.org/simple",
        PIP_EXTRA_INDEX_URL="",
    )
    if not verify_only:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--retries",
                "1",
                "--timeout",
                "25",
                "--target",
                str(target),
                "--report",
                str(root / ".agent" / "ocr-install.json"),
                "-r",
                str(root / "requirements-ocr.txt"),
            ],
            check=True,
            env=env,
        )
    models = target / "rapidocr" / "models"
    records = [
        {
            "name": p.name,
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in sorted(models.glob("*.onnx"))
    ]
    if not any("v6_det_small" in r["name"] for r in records) or not any(
        "v6_rec_small" in r["name"] for r in records
    ):
        raise SystemExit(
            "Expected bundled PP-OCRv6 Small models missing; do not enable OCR."
        )
    (root / ".agent" / "ocr-models.json").write_text(
        json.dumps({"rapidocr": "3.9.2", "models": records}, indent=2), encoding="utf-8"
    )
    print("OCR installed with local model weights. No server started.")


if __name__ == "__main__":
    main()
