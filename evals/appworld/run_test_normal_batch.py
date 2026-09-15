"""Prepare and run a frozen, deterministic half of AppWorld Test-N.

The manifest is created once from task IDs only. Paid runs use bounded process
parallelism, have no automatic task retry, and never discover work by scanning
Phoenix or old runs.  Use ``--parallelism 1`` for the former serial behavior.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE = "personalops-appworld-test-normal:0.1.3.post1"
DEFAULT_PARALLELISM = 4
MAX_PARALLELISM = 8


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def split_task_ids(task_ids: list[str], seed: int) -> tuple[list[str], list[str]]:
    ordered = sorted(set(task_ids))
    if len(ordered) != len(task_ids):
        raise ValueError("Test-N task IDs must be unique")
    random.Random(seed).shuffle(ordered)
    midpoint = len(ordered) // 2
    return ordered[:midpoint], ordered[midpoint:]


def _selection_digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _source_digest() -> str:
    digest = hashlib.sha256()
    paths = sorted(
        {
            *ROOT.glob("*.py"),
            *(ROOT / "prompts").rglob("*.md"),
            *(ROOT / "workers").rglob("*.py"),
            *(ROOT / "reporting").rglob("*.py"),
            *(ROOT / "evals/appworld").glob("*.py"),
            *(ROOT / "skills").rglob("SKILL.md"),
        },
        key=lambda path: path.as_posix(),
    )
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _list_task_ids(image: str) -> list[str]:
    from evals.appworld.protocol import DockerWorld

    with DockerWorld(image=image) as world:
        ids = world.request("list_tasks", split="test_normal")
    if not isinstance(ids, list) or not all(isinstance(value, str) for value in ids):
        raise RuntimeError("Frozen Test-N worker returned an invalid ID list")
    return ids


def prepare(batch_id: str, seed: int, image: str) -> Path:
    batch = ROOT / ".agent/appworld-test-normal" / batch_id
    batch.mkdir(parents=True, exist_ok=False)
    ids = _list_task_ids(image)
    half_a, half_b = split_task_ids(ids, seed)
    if set(half_a) & set(half_b) or set(half_a) | set(half_b) != set(ids):
        raise RuntimeError("Deterministic halves do not partition Test-N")
    manifest = {
        "schema": "appworld-test-normal-halves",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "test_normal",
        "seed": seed,
        "image": image,
        "task_count": len(ids),
        "source_digest": _source_digest(),
        "halves": {
            "A": {"count": len(half_a), "selection_sha256": _selection_digest(half_a), "task_ids": half_a},
            "B": {"count": len(half_b), "selection_sha256": _selection_digest(half_b), "task_ids": half_b},
        },
    }
    _atomic_json(batch / "manifest.private.json", manifest)
    _atomic_json(batch / "progress.json", {
        "status": "PREPARED", "completed": 0, "passed": 0,
        "failed": 0, "unscored": 0, "active_task_index": None,
    })
    return batch


def _result_outcome(trial: Path) -> str:
    try:
        result = json.loads((trial / "result.private.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "UNSCORED"
    official = result.get("official_evaluation")
    if isinstance(official, dict) and official.get("success") is True:
        return "PASSED"
    if isinstance(official, dict) and official.get("success") is False:
        return "FAILED"
    return "UNSCORED"


def _validate_parallelism(value: int) -> int:
    if isinstance(value, bool) or not 1 <= value <= MAX_PARALLELISM:
        raise ValueError(f"Parallelism must be between 1 and {MAX_PARALLELISM}")
    return value


def _ensure_phoenix_ready() -> Any:
    """Start or reuse one collector before child processes fan out.

    Each single-task process performs its own readiness check as defense in
    depth.  Doing it once here prevents several children racing to become the
    first owner of the same local Phoenix port and SQLite database.
    """
    from phoenix_runtime import PhoenixServerRuntime

    runtime = PhoenixServerRuntime()
    runtime.start()
    return runtime


def _counts(outcomes: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "completed": len(outcomes),
        "passed": sum(item["outcome"] == "PASSED" for item in outcomes),
        "failed": sum(item["outcome"] == "FAILED" for item in outcomes),
        "unscored": sum(item["outcome"] == "UNSCORED" for item in outcomes),
    }


def _append_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def run_half(
    batch: Path,
    half: str,
    *,
    allow_paid: bool,
    parallelism: int = DEFAULT_PARALLELISM,
    phoenix_project: str | None = None,
    expected_split: str = "test_normal",
) -> None:
    if not allow_paid:
        raise RuntimeError("Paid Test-N execution requires --allow-paid")
    parallelism = _validate_parallelism(parallelism)
    manifest = json.loads((batch / "manifest.private.json").read_text(encoding="utf-8"))
    if manifest.get("split") != expected_split or half not in {"A", "B"}:
        raise RuntimeError(f"Invalid frozen {expected_split} manifest")
    if manifest.get("source_digest") != _source_digest():
        raise RuntimeError("Source changed after Test-N manifest freeze; refusing an incomparable half")
    task_ids = manifest["halves"][half]["task_ids"]
    if _selection_digest(task_ids) != manifest["halves"][half]["selection_sha256"]:
        raise RuntimeError("Test-N half selection digest mismatch")

    trials = batch / "trials" / half
    trials.mkdir(parents=True, exist_ok=True)
    planned_trials = [
        trials / f"{index:03d}-{task_id}"
        for index, task_id in enumerate(task_ids, 1)
    ]
    existing = [path for path in planned_trials if path.exists()]
    if existing:
        raise RuntimeError(
            "Trial paths already exist; refusing automatic retry: "
            + ", ".join(str(path) for path in existing[:3])
        )

    results_path = batch / f"half-{half.lower()}-results.private.json"
    if results_path.exists():
        raise RuntimeError(
            f"Half {half} already has a result ledger; refusing automatic retry"
        )

    # The parent starts/reuses the shared collector before any task processes.
    # Keep the runtime referenced for the duration of the batch.
    phoenix_runtime = _ensure_phoenix_ready()
    project = phoenix_project or (
        f"AppWorld {expected_split} Half {half} seed {manifest['seed']}"
    )
    monitor_batch = f"{batch.name}-half-{half.lower()}"
    log = batch / f"half-{half.lower()}.log"
    task_logs = batch / "task-logs" / half
    task_logs.mkdir(parents=True, exist_ok=True)
    outcomes: list[dict[str, Any]] = []
    progress_path = batch / "progress.json"

    active: dict[int, dict[str, Any]] = {}
    next_task = 0
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

    def progress(status: str) -> dict[str, Any]:
        active_indices = sorted(active)
        return {
            "status": status,
            "half": half,
            "total": len(task_ids),
            **_counts(outcomes),
            "parallelism": parallelism,
            "pending": len(task_ids) - len(outcomes) - len(active),
            # Keep the old scalar for existing readers and add the real list.
            "active_task_index": active_indices[0] if active_indices else None,
            "active_task_indices": active_indices,
            "active_tasks": [
                {
                    "index": index,
                    "task_id": active[index]["task_id"],
                    "started_at_utc": active[index]["started_at_utc"],
                    "log": str(active[index]["log"]),
                }
                for index in active_indices
            ],
            "phoenix_project": project,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    def save_outcomes() -> None:
        outcomes.sort(key=lambda item: item["index"])
        _atomic_json(results_path, outcomes)

    def launch(index: int, task_id: str) -> None:
        trial = planned_trials[index - 1]
        task_log = task_logs / f"{index:03d}-{task_id}.log"
        command = [
            sys.executable,
            "-m",
            "evals.appworld.run_conversation",
            "--task",
            task_id,
            "--split",
            expected_split,
            "--image",
            manifest["image"],
            "--output",
            str(trial),
            "--phoenix-project",
            project,
            # The parent indexes completions serially.  Child-side monitoring
            # would make several processes rewrite the same aggregate files.
            "--no-monitor",
            "--max-calls",
            "60",
            "--max-interactions",
            "90",
            "--allow-paid",
        ]
        started = datetime.now(timezone.utc).isoformat()
        output_handle = task_log.open("x", encoding="utf-8")
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=output_handle,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
        except Exception as error:
            output_handle.close()
            outcomes.append(
                {
                    "index": index,
                    "task_id": task_id,
                    "outcome": "UNSCORED",
                    "exit_code": None,
                    "launcher_error": f"{type(error).__name__}: {error}",
                    "started_at_utc": started,
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "log": str(task_log),
                }
            )
            save_outcomes()
            _append_log(
                log,
                f"[{datetime.now(timezone.utc).isoformat()}] "
                f"LAUNCH_ERROR {index}/{len(task_ids)} {type(error).__name__}",
            )
            return
        active[index] = {
            "process": process,
            "output_handle": output_handle,
            "task_id": task_id,
            "trial": trial,
            "log": task_log,
            "started_at_utc": started,
        }
        _append_log(
            log,
            f"[{started}] START {index}/{len(task_ids)} pid={process.pid} "
            f"active={len(active)}/{parallelism} log={task_log}",
        )

    try:
        while next_task < len(task_ids) or active:
            while next_task < len(task_ids) and len(active) < parallelism:
                next_task += 1
                launch(next_task, task_ids[next_task - 1])
            _atomic_json(progress_path, progress("RUNNING"))
            if not active:
                continue

            finished = [
                index
                for index, item in active.items()
                if item["process"].poll() is not None
            ]
            if not finished:
                time.sleep(0.25)
                continue

            for index in sorted(finished):
                item = active.pop(index)
                process = item["process"]
                item["output_handle"].close()
                outcome = _result_outcome(item["trial"])
                row = {
                    "index": index,
                    "task_id": item["task_id"],
                    "outcome": outcome,
                    "exit_code": process.returncode,
                    "started_at_utc": item["started_at_utc"],
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "log": str(item["log"]),
                }
                try:
                    from evals.appworld.batch_monitor import record_trial

                    record_trial(item["trial"], monitor_batch)
                except Exception as error:
                    row["monitor_error"] = f"{type(error).__name__}: {error}"
                outcomes.append(row)
                save_outcomes()
                _append_log(
                    log,
                    f"[{row['finished_at_utc']}] END {index}/{len(task_ids)} "
                    f"exit={process.returncode} outcome={outcome} "
                    f"active={len(active)}/{parallelism}",
                )
    except BaseException:
        # Only stop child processes launched by this batch.  Their partial
        # trial paths remain as immutable evidence and prevent silent retries.
        for item in active.values():
            process = item["process"]
            if process.poll() is None:
                process.terminate()
        for item in active.values():
            process = item["process"]
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            item["output_handle"].close()
        _atomic_json(progress_path, progress("ABORTED"))
        raise

    # Retain the reference until all children and monitor writes have finished.
    del phoenix_runtime
    _atomic_json(progress_path, progress("FINISHED"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--half", choices=("A", "B"))
    parser.add_argument(
        "--parallelism",
        "--num-processes",
        dest="parallelism",
        type=int,
        default=DEFAULT_PARALLELISM,
        help=(
            f"Concurrent task processes (default: {DEFAULT_PARALLELISM}; "
            f"range: 1-{MAX_PARALLELISM})."
        ),
    )
    parser.add_argument(
        "--phoenix-project",
        help="Optional shared Phoenix project name for this half.",
    )
    parser.add_argument("--allow-paid", action="store_true")
    args = parser.parse_args()
    batch = ROOT / ".agent/appworld-test-normal" / args.batch_id
    if args.prepare:
        batch = prepare(args.batch_id, args.seed, args.image)
        print(batch)
    if args.half:
        try:
            run_half(
                batch,
                args.half,
                allow_paid=args.allow_paid,
                parallelism=args.parallelism,
                phoenix_project=args.phoenix_project,
            )
        except ValueError as error:
            parser.error(str(error))
        print(batch)
    if not args.prepare and not args.half:
        parser.error("Choose --prepare and/or --half")


if __name__ == "__main__":
    main()
