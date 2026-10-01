"""Retry only the two infrastructure-censored low-temperature Test-N starts.

This launch step leaves the original 168-task batch untouched. The retries run
concurrently in new private trial directories with the exact frozen agent
source and model configuration. A separate audited merge may replace the two
empty startup attempts after official results exist. Scored tasks are never retried.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.appworld.run_test_normal_batch import _atomic_json, _selection_digest, _source_digest
from run_appworld_temperature_variant import _build_variant_environment
from watch_appworld_task_timeouts import stop_overdue


ORIGINAL_ID = "test-normal-full-p4-lowtemp-20260916-demo"
RETRY_ID = "test-normal-lowtemp-censored2-20260917"
PROJECT = "AppWorld Test-N Low Temp Censored Starts 20260917"
CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_censored(original: Path, manifest: dict) -> list[dict]:
    events = [json.loads(line) for line in (original / "timeout-interventions.private.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(events) != 2 or len({event["task_id"] for event in events}) != 2:
        raise RuntimeError("Expected exactly two distinct censored starts")
    ledgers = {half: _read(original / f"half-{half.lower()}-results.private.json") for half in ("A", "B")}
    for event in events:
        half, index, task_id = event["half"], int(event["index"]), event["task_id"]
        if event.get("classification") != "TIMEOUT_CENSORED" or event.get("result_existed_before_stop") is not False:
            raise RuntimeError("Retry target is not an unscored infrastructure stop")
        if manifest["halves"][half]["task_ids"][index - 1] != task_id:
            raise RuntimeError("Frozen manifest and intervention disagree")
        matching = [row for row in ledgers[half] if row["index"] == index]
        if len(matching) != 1 or matching[0]["task_id"] != task_id or matching[0]["outcome"] != "UNSCORED":
            raise RuntimeError("Original official ledger is not unscored")
        trial = original / "trials" / half / f"{index:03d}-{task_id}"
        files = [path for path in trial.rglob("*") if path.is_file()]
        if len(files) != 1 or files[0].name != "runtime.log" or files[0].stat().st_size != 0:
            raise RuntimeError("Original trial has more than a zero-byte startup log")
        task_log = original / "task-logs" / half / f"{index:03d}-{task_id}.log"
        if not task_log.is_file() or task_log.stat().st_size != 0:
            raise RuntimeError("Original task launcher log is not empty")
    return sorted(events, key=lambda event: (event["half"], event["index"]))


def main() -> None:
    original = ROOT / ".agent" / "appworld-test-normal" / ORIGINAL_ID
    retry = ROOT / ".agent" / "appworld-test-normal" / RETRY_ID
    if retry.exists():
        raise RuntimeError("Retry directory already exists; refusing overwrite or automatic repeat")
    original_manifest = _read(original / "manifest.private.json")
    original_config = _read(original / "experiment-config.private.json")
    comparison_status = _read(ROOT / ".agent" / "appworld-comparisons" / "test-normal-lowtemp-20260916" / "completion-status.private.json")
    if original_manifest.get("task_count") != 168 or comparison_status.get("state") != "FINISHED" or not (original / "full-aggregate.private.json").is_file():
        raise RuntimeError("Original low-temperature batch is not frozen and complete")
    if original_manifest["source_digest"] != _source_digest():
        raise RuntimeError("Agent source changed since the low-temperature evaluation")
    for relative, digest in original_config["baseline_config_sha256"].items():
        if _sha256(ROOT / relative) != digest:
            raise RuntimeError(f"Model configuration changed: {relative}")
    events = _verify_censored(original, original_manifest)
    environment, effective = _build_variant_environment()
    if effective != original_config["temperature_overrides"]:
        raise RuntimeError("Temperature overrides differ from the original low-temperature batch")

    retry.mkdir(parents=True, exist_ok=False)
    task_ids = [event["task_id"] for event in events]
    manifest = {
        "schema": "appworld-test-normal-infrastructure-retry",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "test_normal",
        "seed": original_manifest["seed"],
        "image": original_manifest["image"],
        "task_count": 2,
        "source_digest": original_manifest["source_digest"],
        "original_batch_id": ORIGINAL_ID,
        "original_censored_starts": [{"half": e["half"], "index": e["index"], "task_id": e["task_id"]} for e in events],
        "temperature_overrides": effective,
        "halves": {
            "A": {"count": 2, "selection_sha256": _selection_digest(task_ids), "task_ids": task_ids},
            "B": {"count": 0, "selection_sha256": _selection_digest([]), "task_ids": []},
        },
    }
    _atomic_json(retry / "manifest.private.json", manifest)
    _atomic_json(retry / "progress.json", {"status": "PREPARED", "completed": 0, "passed": 0, "failed": 0, "unscored": 0})
    command = [
        sys.executable, "-X", "utf8", "-m", "evals.appworld.run_test_normal_batch",
        "--batch-id", RETRY_ID, "--half", "A", "--parallelism", "2",
        "--phoenix-project", PROJECT, "--allow-paid",
    ]
    print(f"Starting two concurrent infrastructure-only retries in {retry}", flush=True)
    with (retry / "launcher.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, creationflags=CREATE_NO_WINDOW)
        last_report = 0.0
        seen: set[tuple[str, int, int]] = set()
        while process.poll() is None:
            now = datetime.now(timezone.utc)
            for event in stop_overdue(retry, now, 40 * 60, ignored=seen):
                seen.add((event["half"], event["index"], event["pid"]))
                print(f"Stopped verified over-40-minute child at retry index {event['index']}", flush=True)
            if time.monotonic() - last_report >= 60:
                progress = _read(retry / "progress.json")
                print(f"Progress: completed={progress.get('completed')} active={len(progress.get('active_tasks') or [])}", flush=True)
                last_report = time.monotonic()
            time.sleep(5)
        if process.returncode != 0:
            raise RuntimeError(f"Two-task runner exited {process.returncode}; see private launcher.log")

    rows = _read(retry / "half-a-results.private.json")
    if len(rows) != 2 or {row["task_id"] for row in rows} != set(task_ids):
        raise RuntimeError("Retry official result ledger is incomplete")
    counts = Counter(row["outcome"] for row in rows)
    original_counts = Counter(_read(original / "half-a-results.private.json")[i]["outcome"] for i in range(84))
    original_counts.update(row["outcome"] for row in _read(original / "half-b-results.private.json"))
    original_counts["UNSCORED"] -= 2
    original_counts.update(counts)
    correction = {
        "original_batch_id": ORIGINAL_ID,
        "retry_batch_id": RETRY_ID,
        "original_official_counts": {"PASSED": 138, "FAILED": 27, "UNSCORED": 3},
        "retry_outcomes": dict(counts),
        "reconciled_official_counts": {name: original_counts[name] for name in ("PASSED", "FAILED", "UNSCORED")},
        "reconciled_passed_over_168": original_counts["PASSED"] / 168,
        "note": "Only the two pre-task infrastructure-censored starts are replaced. Original ledgers are not modified.",
    }
    _atomic_json(retry / "reconciled-outcomes.private.json", correction)
    print(json.dumps(correction, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
