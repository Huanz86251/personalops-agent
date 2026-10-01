"""Run one frozen four-way AppWorld Challenge pass with the Test-N low-temp setup.

No task is automatically retried. A 40-minute watchdog records and stops only
verified stuck child processes; their results remain unscored evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from complete_appworld_full_batch import _complete_half, _read, _status, _zip_traces
from run_appworld_temperature_variant import _build_variant_environment


ROOT = Path(__file__).resolve().parents[1]
SPLIT = "test_challenge"
IMAGE = "personalops-appworld-test-challenge:0.1.3.post1"
BATCH_ID = "test-challenge-full-p4-lowtemp-20260917-demo-r1"
PROJECT = "AppWorld Test-C Low Temp 20260917 Demo p4 r1"
BATCH = ROOT / ".agent" / "appworld-test-challenge" / BATCH_ID
COMPARISON = ROOT / ".agent" / "appworld-comparisons" / "test-normal-lowtemp-20260916"
CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def _run(command: list[str], log: Path, env: dict[str, str]) -> None:
    with log.open("x", encoding="utf-8") as output:
        result = subprocess.run(
            command, cwd=ROOT, env=env, stdout=output,
            stderr=subprocess.STDOUT, check=False, creationflags=CREATE_NO_WINDOW,
        )
    if result.returncode:
        raise RuntimeError(f"Command exited {result.returncode}; inspect {log}")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-paid", action="store_true")
    args = parser.parse_args()
    if not args.allow_paid:
        parser.error("Challenge paid execution requires --allow-paid")
    if BATCH.exists():
        raise RuntimeError("Challenge batch already exists; refusing automatic retry or overwrite")
    prior = _read(COMPARISON / "comparison-plan.private.json")
    if not isinstance(prior, dict):
        raise RuntimeError("Frozen low-temperature Normal configuration is unavailable")
    for relative, expected in prior["baseline_config_sha256"].items():
        if _digest(ROOT / relative) != expected:
            raise RuntimeError(f"Model configuration drifted since Normal: {relative}")
    environment, effective = _build_variant_environment()
    if effective != prior["temperature_overrides"]:
        raise RuntimeError("Challenge temperature overrides differ from the Normal run")

    seed = int(prior["seed"])
    command = [
        sys.executable, "-X", "utf8", "-m", "evals.appworld.run_test_normal_batch",
        "--batch-id", BATCH_ID, "--split", SPLIT, "--seed", str(seed), "--image", IMAGE,
    ]
    # Preparation is read-only against AppWorld: freeze all IDs before any model call.
    BATCH.parent.mkdir(parents=True, exist_ok=True)
    prepare_log = BATCH.parent / f"{BATCH_ID}-prepare.log"
    _run([*command, "--prepare"], prepare_log, environment)
    manifest = _read(BATCH / "manifest.private.json")
    if not isinstance(manifest, dict) or manifest.get("split") != SPLIT:
        raise RuntimeError("Challenge manifest was not frozen correctly")
    plan = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": SPLIT,
        "normal_reference_batch": prior["variant_batch_id"],
        "seed": seed,
        "image": IMAGE,
        "task_count": manifest["task_count"],
        "source_digest": manifest["source_digest"],
        "temperature_overrides": effective,
        "other_roles": prior["other_roles"],
        "model_config_sha256": prior["baseline_config_sha256"],
        "max_model_calls_per_task": 80,
        "max_appworld_interactions_per_task": 90,
        "parallelism": 4,
        "task_wall_timeout_minutes": 40,
        "no_automatic_task_retry": True,
        "phoenix_project": PROJECT,
        "pre_challenge_agent_source_digest": "3812acb0b5b3002eab8813d8a26ebc7776ccbe788b7916e0156e8197b27ea5ee",
        "source_change_scope": "Challenge-only evaluation plumbing; agent policy unchanged",
        "prior_startup_attempt": "test-challenge-full-p4-lowtemp-20260917-demo: 417 children exited before Agent imports, all missing langchain_core in the base interpreter; zero model calls and zero AppWorld executions",
    }
    (BATCH / "experiment-config.private.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    watch_log = (BATCH / "watchdog.log").open("x", encoding="utf-8")
    watcher = subprocess.Popen(
        [sys.executable, "-X", "utf8", "scripts/watch_appworld_task_timeouts.py",
         "--batch-id", BATCH_ID, "--split", SPLIT, "--minutes", "40"],
        cwd=ROOT, env=environment, stdout=watch_log, stderr=subprocess.STDOUT,
        creationflags=CREATE_NO_WINDOW,
    )
    try:
        for half in ("A", "B"):
            _status(BATCH, f"RUNNING_{half}", "Four-way paid Challenge run; max 80 model calls per task")
            _run(
                [*command, "--half", half, "--parallelism", "4",
                 "--phoenix-project", PROJECT, "--allow-paid"],
                BATCH / f"half-{half.lower()}-launcher.log", environment,
            )
            if not _complete_half(BATCH, half, int(manifest["halves"][half]["count"])):
                raise RuntimeError(f"Challenge half {half} ledger is incomplete")
        _status(BATCH, "AGGREGATING", "Computing official outcomes and private traces")
        _run(
            [sys.executable, "-X", "utf8", "scripts/appworld_full_batch_stats.py",
             "--batch-id", BATCH_ID, "--split", SPLIT],
            BATCH / "aggregate-launcher.log", environment,
        )
        files, size = _zip_traces(BATCH)
        _status(BATCH, "FINISHED", f"All tasks complete; {files} evidence files ({size} bytes)")
    except Exception as error:
        _status(BATCH, "ERROR", f"{type(error).__name__}: {error}")
        raise
    finally:
        if watcher.poll() is None:
            watcher.terminate()
            watcher.wait(timeout=10)
        watch_log.close()


if __name__ == "__main__":
    main()
