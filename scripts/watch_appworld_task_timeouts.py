"""Stop only verified AppWorld batch children that exceed a wall-time limit.

This external watchdog can protect a batch whose frozen runner is already in
memory. It never retries a task or changes its official result ledger. Timeout
interventions are private sidecar evidence for later scored-only reporting.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

import psutil


ROOT = Path(__file__).resolve().parents[1]
BATCH_ROOT = ROOT / ".agent"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
START_LINE = re.compile(r"\bSTART (?P<index>\d+)/\d+ pid=(?P<pid>\d+)\b")
SIDE_CAR = "timeout-interventions.private.jsonl"


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _arg_after(command: list[str], flag: str) -> str | None:
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def _pids_for_half(batch: Path, half: str) -> dict[int, int]:
    try:
        lines = (batch / f"half-{half.lower()}.log").read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    result: dict[int, int] = {}
    for line in lines:
        match = START_LINE.search(line)
        if match:
            result[int(match["index"])] = int(match["pid"])
    return result


def _verified_process(pid: int, batch: Path, half: str, index: int,
                      task_id: str, started_at: datetime, split: str = "test_normal") -> psutil.Process | None:
    try:
        process = psutil.Process(pid)
        command = process.cmdline()
        born = datetime.fromtimestamp(process.create_time(), timezone.utc)
        expected_output = batch / "trials" / half / f"{index:03d}-{task_id}"
        actual_output = _arg_after(command, "--output")
        if (
            not process.is_running()
            or "evals.appworld.run_conversation" not in command
            or _arg_after(command, "--task") != task_id
            or _arg_after(command, "--split") != split
            or actual_output is None
            or Path(actual_output).resolve() != expected_output.resolve()
            or abs((born - started_at).total_seconds()) > 120
        ):
            return None
        return process
    except (psutil.Error, OSError, ValueError):
        return None


def overdue_children(batch: Path, now: datetime, timeout_seconds: float, split: str = "test_normal") -> list[dict]:
    progress = _read_json(batch / "progress.json")
    half = progress.get("half")
    if progress.get("status") != "RUNNING" or half not in ("A", "B"):
        return []
    pids = _pids_for_half(batch, half)
    overdue = []
    for active in progress.get("active_tasks") or []:
        try:
            index = int(active["index"])
            task_id = str(active["task_id"])
            started_at = datetime.fromisoformat(active["started_at_utc"])
            if started_at.tzinfo is None:
                continue
            elapsed = (now - started_at).total_seconds()
            pid = pids.get(index)
            if pid is None or elapsed < timeout_seconds:
                continue
            process = _verified_process(pid, batch, half, index, task_id, started_at, split)
            if process is None:
                continue
            overdue.append({
                "half": half, "index": index, "task_id": task_id,
                "pid": pid, "elapsed_seconds": elapsed, "process": process,
            })
        except (KeyError, TypeError, ValueError):
            continue
    return overdue


def _append_event(batch: Path, event: dict) -> None:
    with (batch / SIDE_CAR).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def stop_overdue(batch: Path, now: datetime, timeout_seconds: float,
                 *, dry_run: bool = False, ignored: set[tuple[str, int, int]] | None = None,
                 split: str = "test_normal") -> list[dict]:
    stopped = []
    for item in overdue_children(batch, now, timeout_seconds, split):
        key = (item["half"], item["index"], item["pid"])
        if ignored and key in ignored:
            continue
        process = item.pop("process")
        trial = batch / "trials" / item["half"] / f"{item['index']:03d}-{item['task_id']}"
        result_exists = (trial / "result.private.json").is_file()
        event = {
            **item,
            "event": "wall_timeout_stop_requested",
            "recorded_at_utc": now.isoformat(),
            "timeout_seconds": timeout_seconds,
            "result_existed_before_stop": result_exists,
            "classification": "TIMEOUT_CENSORED" if not result_exists else "RESULT_EXISTS_CLEANUP_TIMEOUT",
        }
        if not dry_run:
            # Write the intervention before stopping the child. The official
            # runner then observes its exit and records the original outcome.
            _append_event(batch, event)
            try:
                process.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied) as error:
                _append_event(batch, {**event, "event": "wall_timeout_stop_error", "error": type(error).__name__})
                continue
            _append_event(batch, {**event, "event": "wall_timeout_terminate_sent"})
        stopped.append(event)
    return stopped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--split", choices=("test_normal", "test_challenge"), default="test_normal")
    parser.add_argument("--minutes", type=float, default=40)
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not SAFE_ID.fullmatch(args.batch_id):
        parser.error("Invalid batch ID")
    if args.minutes <= 0 or args.poll_seconds <= 0:
        parser.error("Timeout and poll interval must be positive")
    batch = BATCH_ROOT / ("appworld-test-normal" if args.split == "test_normal" else "appworld-test-challenge") / args.batch_id
    manifest = _read_json(batch / "manifest.private.json")
    if manifest.get("split") != args.split:
        parser.error(f"Frozen {args.split} batch not found")
    timeout_seconds = args.minutes * 60
    seen: set[tuple[str, int, int]] = set()
    while True:
        now = datetime.now(timezone.utc)
        for event in stop_overdue(batch, now, timeout_seconds, dry_run=args.dry_run, ignored=seen, split=args.split):
            seen.add((event["half"], event["index"], event["pid"]))
            print(json.dumps({key: value for key, value in event.items() if key != "process"}), flush=True)
        progress = _read_json(batch / "progress.json")
        if args.once or (progress.get("half") == "B" and progress.get("status") in {"FINISHED", "ABORTED"}):
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
