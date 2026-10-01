"""Finish the second half of an already running, frozen AppWorld Test-N batch.

This helper never retries a task or modifies the frozen manifest. It starts B
only after A is finished and complete, then creates private aggregate reports.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def _status(batch: Path, state: str, detail: str = "") -> None:
    record = {
        "state": state,
        "detail": detail,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = batch / "completion-status.private.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(f"{record['updated_at_utc']} {state}: {detail}", flush=True)


def _complete_half(batch: Path, half: str, expected: int) -> bool:
    ledger = _read(batch / f"half-{half.lower()}-results.private.json")
    return (
        isinstance(ledger, list)
        and len(ledger) == expected
        and len({row.get("task_id") for row in ledger}) == expected
    )


def _zip_traces(batch: Path) -> tuple[int, int]:
    index = _read(batch / "demo-trace-index.private.json")
    if not isinstance(index, dict) or not isinstance(index.get("traces"), list):
        raise RuntimeError("Trace index missing after aggregation")
    archive = batch / "demo-traces.private.zip"
    if archive.exists():
        raise RuntimeError("Refusing to overwrite existing demo trace archive")
    count = 0
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        output.write(batch / "manifest.private.json", "manifest.private.json")
        output.write(batch / "full-aggregate.private.json", "full-aggregate.private.json")
        output.write(batch / "demo-trace-index.private.json", "demo-trace-index.private.json")
        for entry in index["traces"]:
            trial = Path(entry["trial_directory"])
            prefix = f"{entry['half']}/{entry['index']:03d}-{entry['task_id']}"
            for name in ("spans.private.json", "usage.private.json", "result.private.json"):
                path = trial / name
                if path.is_file():
                    output.write(path, f"{prefix}/{name}")
                    count += 1
    return count, archive.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--phoenix-project", required=True)
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--timeout-hours", type=float, default=18)
    args = parser.parse_args()
    batch = ROOT / ".agent" / "appworld-test-normal" / args.batch_id
    manifest = _read(batch / "manifest.private.json")
    if not isinstance(manifest, dict) or manifest.get("split") != "test_normal":
        raise RuntimeError("Frozen Test-N manifest not found")
    expected_a = int(manifest["halves"]["A"]["count"])
    expected_b = int(manifest["halves"]["B"]["count"])
    if expected_a + expected_b != int(manifest["task_count"]):
        raise RuntimeError("Manifest halves do not cover the full set")
    deadline = time.monotonic() + args.timeout_hours * 3600
    _status(batch, "WAITING_FOR_A", f"Expecting {expected_a} A results")
    while True:
        progress = _read(batch / "progress.json") or {}
        if progress.get("half") == "A" and progress.get("status") == "FINISHED":
            if not _complete_half(batch, "A", expected_a):
                raise RuntimeError("A says FINISHED but ledger is incomplete")
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("Timed out waiting for A; B was not started")
        time.sleep(20)

    b_ledger = batch / "half-b-results.private.json"
    if b_ledger.exists():
        if not _complete_half(batch, "B", expected_b):
            raise RuntimeError("Existing B ledger is incomplete; refusing an automatic retry")
        _status(batch, "B_ALREADY_COMPLETE", f"Found {expected_b} B results")
    else:
        _status(batch, "RUNNING_B", f"Starting {expected_b} B tasks at parallelism {args.parallelism}")
        command = [
            sys.executable, "-X", "utf8", "-m", "evals.appworld.run_test_normal_batch",
            "--batch-id", args.batch_id,
            "--half", "B",
            "--parallelism", str(args.parallelism),
            "--phoenix-project", args.phoenix_project,
            "--allow-paid",
        ]
        with (batch / "half-b-launcher.log").open("x", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Half B launcher exited {result.returncode}; inspect half-b-launcher.log")
        if not _complete_half(batch, "B", expected_b):
            raise RuntimeError("Half B launcher exited successfully but ledger is incomplete")

    _status(batch, "AGGREGATING", "Validating official outcomes and private traces")
    with (batch / "aggregate-launcher.log").open("x", encoding="utf-8") as log:
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "scripts/appworld_full_batch_stats.py", "--batch-id", args.batch_id],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Aggregation exited {result.returncode}; inspect aggregate-launcher.log")
    files, size = _zip_traces(batch)
    _status(batch, "FINISHED", f"Both halves complete; {files} evidence files in trace archive ({size} bytes)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if len(sys.argv) >= 3 and sys.argv[1] == "--batch-id":
            batch_path = ROOT / ".agent" / "appworld-test-normal" / sys.argv[2]
            if batch_path.is_dir():
                _status(batch_path, "ERROR", f"{type(exc).__name__}: {exc}")
        raise
