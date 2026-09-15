"""Install local tools separately, without changing the active Agent environment."""

import os
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    target = root / ".agent" / "local-tool-deps"
    if target.exists():
        raise SystemExit(
            "Dependency directory already exists. Do not upgrade it while tasks run; inspect it first."
        )
    env = os.environ.copy()
    # Use the public upstream index without unrelated machine-wide extra indexes.
    env["PIP_CONFIG_FILE"] = os.devnull
    env["PIP_INDEX_URL"] = "https://pypi.org/simple"
    env["PIP_EXTRA_INDEX_URL"] = ""
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
            str(root / ".agent" / "local-tools-install.json"),
            "-r",
            str(root / "requirements-local-tools.txt"),
        ],
        check=True,
        env=env,
    )
    print(
        f"Local dependencies installed at {target}. No services started; restart Agent only when convenient."
    )


if __name__ == "__main__":
    main()
