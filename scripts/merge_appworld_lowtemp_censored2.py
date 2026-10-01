"""Replace two empty startup attempts in the private low-temperature batch.

This is an explicit, audited infrastructure correction, not a scored-task
retry. The prior empty attempts and prior aggregate remain in a private backup.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.appworld.run_test_normal_batch import _atomic_json, _result_outcome
from appworld_full_batch_stats import summarize
from complete_appworld_full_batch import _zip_traces
from run_appworld_temperature_variant import _paired_comparison


ORIGINAL_ID = "test-normal-full-p4-lowtemp-20260916-demo"
RETRY_ID = "test-normal-lowtemp-censored2-20260917"
BASELINE_ID = "test-normal-full-p4-20260916-demo"
COMPARISON_ID = "test-normal-lowtemp-20260916"
BACKUP_NAME = "pre-infrastructure-reconciliation-20260917"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _within(path: Path, parent: Path) -> Path:
    absolute = path.resolve()
    if not absolute.is_relative_to(parent.resolve()):
        raise RuntimeError(f"Path escaped the intended private directory: {path}")
    return absolute


def main() -> None:
    private_root = ROOT / ".agent" / "appworld-test-normal"
    original = _within(private_root / ORIGINAL_ID, private_root)
    retry = _within(private_root / RETRY_ID, private_root)
    comparison = _within(ROOT / ".agent" / "appworld-comparisons" / COMPARISON_ID, ROOT / ".agent" / "appworld-comparisons")
    baseline = _within(private_root / BASELINE_ID, private_root)
    backup = _within(original / BACKUP_NAME, original)
    if backup.exists() or (original / "reconciliation.private.json").exists():
        raise RuntimeError("This reconciliation has already started or completed; refusing a second overwrite")
    original_manifest = _read(original / "manifest.private.json")
    retry_manifest = _read(retry / "manifest.private.json")
    if retry_manifest.get("original_batch_id") != ORIGINAL_ID or retry_manifest.get("source_digest") != original_manifest.get("source_digest"):
        raise RuntimeError("Retry provenance differs from the frozen low-temperature batch")
    events = retry_manifest["original_censored_starts"]
    if len(events) != 2 or len({e["task_id"] for e in events}) != 2:
        raise RuntimeError("Expected exactly two distinct infrastructure-censored starts")
    ledger_path = original / "half-a-results.private.json"
    ledger = _read(ledger_path)
    retry_ledger = _read(retry / "half-a-results.private.json")
    retry_by_task = {row["task_id"]: row for row in retry_ledger}
    if len(retry_by_task) != 2 or any(row["outcome"] not in {"PASSED", "FAILED"} for row in retry_ledger):
        raise RuntimeError("Both retries must have official scored results before merging")
    original_monitor_path = ROOT / ".agent" / "appworld-monitor" / f"{ORIGINAL_ID}-half-a" / "index.json"
    retry_monitor_path = ROOT / ".agent" / "appworld-monitor" / f"{RETRY_ID}-half-a" / "index.json"
    original_monitor = _read(original_monitor_path)
    retry_monitor = _read(retry_monitor_path)
    monitor_by_task = {row["task_id"]: row for row in retry_monitor}
    if len(monitor_by_task) != 2:
        raise RuntimeError("Retry monitoring index is incomplete")

    replacements = []
    for event in events:
        half, index, task_id = event["half"], int(event["index"]), event["task_id"]
        if half != "A" or original_manifest["halves"][half]["task_ids"][index - 1] != task_id:
            raise RuntimeError("Original frozen selection disagrees with retry provenance")
        old_ledger = ledger[index - 1]
        old_monitor = original_monitor[index - 1]
        old_trial = _within(original / "trials" / half / f"{index:03d}-{task_id}", original)
        old_log = _within(original / "task-logs" / half / f"{index:03d}-{task_id}.log", original)
        new_row = retry_by_task[task_id]
        new_trial = _within(retry / "trials" / "A" / f"{int(new_row['index']):03d}-{task_id}", retry)
        new_log = _within(retry / "task-logs" / "A" / f"{int(new_row['index']):03d}-{task_id}.log", retry)
        if old_ledger["task_id"] != task_id or old_ledger["outcome"] != "UNSCORED" or old_monitor["outcome"] != "UNSCORED":
            raise RuntimeError("Target is not an unscored original attempt")
        old_files = [p for p in old_trial.rglob("*") if p.is_file()]
        if len(old_files) != 1 or old_files[0].name != "runtime.log" or old_files[0].stat().st_size != 0 or old_log.stat().st_size != 0:
            raise RuntimeError("Original attempt is not an empty startup stall")
        if _result_outcome(new_trial) != new_row["outcome"] or monitor_by_task[task_id]["outcome"] != new_row["outcome"]:
            raise RuntimeError("Retry official result, ledger and monitor disagree")
        for required in ("manifest.json", "task.private.json", "usage.private.json", "spans.private.json", "result.private.json", "execution.private.zip"):
            if not (new_trial / required).is_file():
                raise RuntimeError(f"Retry trial is missing {required}")
        replacements.append((event, old_trial, old_log, new_trial, new_log))

    backup.mkdir(parents=False, exist_ok=False)
    for path in (
        ledger_path, original_monitor_path, original / "full-aggregate.private.json",
        original / "demo-trace-index.private.json", original / "demo-traces.private.zip",
        comparison / "paired-comparison.private.json",
    ):
        if not path.is_file():
            raise RuntimeError(f"Required original evidence is missing: {path}")
        shutil.copy2(path, backup / ("monitor-index.json" if path == original_monitor_path else "paired-comparison.private.json" if path.parent == comparison else path.name))

    for event, old_trial, old_log, new_trial, new_log in replacements:
        index, task_id = int(event["index"]), event["task_id"]
        moved_trial = backup / old_trial.name
        moved_log = backup / old_log.name
        shutil.move(str(old_trial), str(moved_trial))
        shutil.move(str(old_log), str(moved_log))
        shutil.copytree(new_trial, old_trial)
        shutil.copy2(new_log, old_log)
        new_ledger_row = dict(retry_by_task[task_id])
        new_ledger_row.update(index=index, log=str(old_log), replaced_empty_startup_attempt=True)
        ledger[index - 1] = new_ledger_row
        new_monitor_row = dict(monitor_by_task[task_id])
        new_monitor_row.update(trial_id=old_trial.name, trial_directory=str(old_trial), replaced_empty_startup_attempt=True)
        original_monitor[index - 1] = new_monitor_row

    _atomic_json(ledger_path, ledger)
    _atomic_json(original_monitor_path, original_monitor)
    metrics, traces, _ = summarize(ORIGINAL_ID)
    expected = Counter(row["outcome"] for row in ledger + _read(original / "half-b-results.private.json"))
    if (metrics["official_outcomes"]["PASSED"], metrics["official_outcomes"]["FAILED"], metrics["official_outcomes"]["UNSCORED"]) != (
        expected["PASSED"], expected["FAILED"], expected["UNSCORED"]
    ):
        raise RuntimeError("Rebuilt aggregate does not match official ledgers")
    _atomic_json(original / "full-aggregate.private.json", metrics)
    _atomic_json(original / "demo-trace-index.private.json", traces)
    (original / "demo-traces.private.zip").unlink()
    files, archive_bytes = _zip_traces(original)
    _atomic_json(comparison / "paired-comparison.private.json", _paired_comparison(baseline, original))
    record = {
        "reconciled_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "Two pre-task startup stalls had no model call, task manifest, official result or trace; user explicitly approved two concurrent scored replacements in the original batch.",
        "retry_batch_id": RETRY_ID,
        "backup_directory": str(backup),
        "replaced_trial_indices": [event["index"] for event in events],
        "official_outcomes": {name: expected[name] for name in ("PASSED", "FAILED", "UNSCORED")},
        "trace_archive_evidence_files": files,
        "trace_archive_bytes": archive_bytes,
        "warning": "Original batch wall-clock span now includes later corrective attempts; task-level corrected latency uses the replacement runs.",
    }
    _atomic_json(original / "reconciliation.private.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
