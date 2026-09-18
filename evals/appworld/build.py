"""Build an AppWorld image from verified local data, without sending repo secrets."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
from .protocol import docker_prefix

DATA_URL = "https://s3.us-west-2.amazonaws.com/appworld.dev/data-0.1.0.bundle"
DATA_SHA256 = "fd9f9608c2ec71ed0ac25c3633a738b9129a318a129e31230425b9188e508250"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("development", "test-normal", "test-challenge"), default="development")
    parser.add_argument("--image")
    parser.add_argument("--wsl", default="Ubuntu")
    args = parser.parse_args()
    image = args.image or {
        "development": "personalops-appworld:0.1.3.post1",
        "test-normal": "personalops-appworld-test-normal:0.1.3.post1",
        "test-challenge": "personalops-appworld-test-challenge:0.1.3.post1",
    }[args.protocol]
    root = Path(__file__).resolve().parents[2]
    cache = root / ".agent/appworld-downloads"
    context = root / {
        "development": ".agent/appworld-build",
        "test-normal": ".agent/appworld-build-test-normal",
        "test-challenge": ".agent/appworld-build-test-challenge",
    }[args.protocol]
    for path in (cache, context):
        if not path.resolve().is_relative_to(root):
            raise ValueError("Build paths must remain inside the project")
        path.mkdir(parents=True, exist_ok=True)
    bundle = cache / "data-0.1.0.bundle"
    if not bundle.exists():
        partial = bundle.with_suffix(".bundle.partial")
        with urllib.request.urlopen(DATA_URL, timeout=30) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        partial.replace(bundle)
    actual = hashlib.sha256(bundle.read_bytes()).hexdigest()
    if actual != DATA_SHA256:
        raise ValueError("Official data checksum changed; refusing an unversioned benchmark")
    source = Path(__file__).resolve().parent
    names = ["Dockerfile", "worker.py", "bootstrap_data.py", "requirements.snapshot.txt"]
    for name in names:
        shutil.copyfile(source / name, context / name)
    shutil.copyfile(bundle, context / bundle.name)
    (context / ".dockerignore").write_text(
        "*\n" + "".join("!" + name + "\n" for name in names + [bundle.name]),
        encoding="utf-8",
    )
    if os.name == "nt":
        command = ["wsl", "-d", args.wsl, "--exec"]
        context_arg = subprocess.check_output(command + ["wslpath", "-a", str(context)], text=True).strip()
        command = docker_prefix(args.wsl)
    else:
        command, context_arg = ["docker"], str(context)
    print(json.dumps({"verified_data_sha256": actual, "context": str(context)}), flush=True)
    allowed_splits = {
        "development": "train,dev",
        "test-normal": "test_normal",
        "test-challenge": "test_challenge",
    }[args.protocol]
    subprocess.run(
        command + [
            "build", "--progress", "plain",
            "--build-arg", f"APPWORLD_ALLOWED_SPLITS={allowed_splits}",
            "-t", image, context_arg,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
