"""Prepare the local Docker runtime used by Code Worker and Code Reviewer."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workers.docker_sandbox import (  # noqa: E402
    CodeSandboxError,
    CodeSandboxManager,
    docker_install_guidance,
)


def _install_docker() -> int:
    if os.name != "nt":
        print(docker_install_guidance(), file=sys.stderr)
        print(
            "Automatic Linux installation is intentionally not run because it "
            "requires root access. Run the commands above explicitly.",
            file=sys.stderr,
        )
        return 2

    command = [
        "winget",
        "install",
        "--exact",
        "--id",
        "Docker.DockerDesktop",
        "--accept-package-agreements",
        "--accept-source-agreements",
    ]
    print("Installing Docker Desktop with winget (explicitly requested)...")
    try:
        result = subprocess.run(command, check=False)
    except OSError as exc:
        print(f"Cannot run winget: {exc}", file=sys.stderr)
        return 2
    if result.returncode:
        return result.returncode
    print(
        "Docker Desktop is installed. Start it once, enable WSL integration "
        "for Ubuntu, then rerun this script without --install-docker."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check Docker, start Docker Engine when possible, and build the "
            "PersonalOps CODE sandbox image when it is missing."
        )
    )
    parser.add_argument(
        "--install-docker",
        action="store_true",
        help="explicitly install Docker Desktop through winget on Windows",
    )
    parser.add_argument("--rebuild", action="store_true", help="rebuild the Code base image even if its tag exists")
    parser.add_argument("--domestic-mirrors", action="store_true", help="rebuild using tested Chinese dependency mirrors (build-time only)")
    args = parser.parse_args()
    if args.install_docker:
        return _install_docker()

    try:
        manager = CodeSandboxManager()
        if args.rebuild or args.domestic_mirrors:
            manager.cli.ensure_daemon()
            manager.build_image(domestic_mirrors=args.domestic_mirrors)
        else:
            manager.ensure_ready()
    except CodeSandboxError as exc:
        print(f"CODE sandbox setup failed:\n{exc}", file=sys.stderr)
        return 1
    print(f"CODE sandbox is ready: {manager.policy.image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
