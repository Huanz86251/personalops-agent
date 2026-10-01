"""Privately stage the frozen Challenge run for AppWorld's official evaluator.

The 414 original environment exports are copied unchanged. Three unfinished
trials have no export; their recorded world calls are replayed without an
agent or model, and every returned value must exactly match the original log.
Original trial evidence and scores are never overwritten.
"""
from __future__ import annotations

import base64
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.appworld.protocol import DockerWorld
from evals.appworld.run_test_normal_batch import _atomic_json
from stage_appworld_lowtemp_official_output import _hash, _read, _safe_extract


BATCH_ID = "test-challenge-full-p4-lowtemp-20260917-demo-r1"
EXPERIMENT_NAME = "personalops_dualflash_test_challenge"
STAGE = ROOT / ".agent" / "appworld-leaderboard-challenge-staging"
EXPERIMENT = STAGE / "experiments" / "outputs" / EXPERIMENT_NAME
REQUIRED_PER_TASK = (
    "dbs/model_hashes.json",
    "logs/environment_io.md",
    "logs/api_calls.jsonl",
    "evaluation/report.md",
    "version/code.txt",
    "version/data.txt",
)


def _reconstruct_recorded_trial(entry: dict, image: str) -> dict:
    trial = Path(entry["trial_directory"])
    result = _read(trial / "result.private.json")
    calls = _read(trial / "world-calls.private.json")
    if result.get("status") != "finished" or result.get("official_evaluation") is not None:
        raise RuntimeError("Reconstruction target is not a genuinely unscored finished trial")
    if not isinstance(calls, list) or not calls:
        raise RuntimeError("Unscored trial lacks recorded environment calls")
    if any(not isinstance(call.get("code"), str) or not isinstance(call.get("output"), str) for call in calls):
        raise RuntimeError("A recorded call lacks exact code or output")
    if [call.get("index") for call in calls] != list(range(1, len(calls) + 1)):
        raise RuntimeError("Recorded environment calls are not consecutively ordered")
    with DockerWorld(image=image) as world:
        world.request(
            "initialize", split="test_challenge", task_id=entry["task_id"],
            trial_id="leaderboard-evidence-reconstruction", max_interactions=90,
        )
        for call in calls:
            output = world.execute(call["code"])
            if output != call["output"]:
                raise RuntimeError(
                    f"Recorded/replayed environment output differs at call {call['index']}; "
                    "aborting without an official-format export"
                )
        world.request("finish")
        evaluation = world.request("evaluate")
        if evaluation.get("success") is not False:
            raise RuntimeError("Reconstruction did not reproduce a failed official result")
        archive_bytes = base64.b64decode(world.request("export")["zip_base64"])

    evidence_dir = STAGE / "private-reconstruction"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    archive = evidence_dir / f"{entry['task_id']}.private.zip"
    archive.write_bytes(archive_bytes)
    files = _safe_extract(archive_bytes, EXPERIMENT, entry["task_id"])
    return {
        "task_id": entry["task_id"],
        "original_outcome": "UNSCORED",
        "recorded_world_calls": len(calls),
        "matched_replayed_world_calls": len(calls),
        "original_model_calls": entry["model_calls"],
        "replayed_model_calls": 0,
        "reconstructed_official_success": False,
        "archive_sha256": _hash(archive_bytes),
        "archive_file": str(archive),
        "extracted_files": files,
    }


def main() -> None:
    if STAGE.exists():
        raise RuntimeError("Private Challenge staging already exists; refusing overwrite")
    batch = ROOT / ".agent" / "appworld-test-challenge" / BATCH_ID
    manifest = _read(batch / "manifest.private.json")
    aggregate = _read(batch / "full-aggregate.private.json")
    index = _read(batch / "demo-trace-index.private.json")
    completion = _read(batch / "completion-status.private.json")
    entries = index["traces"]
    expected_ids = manifest["halves"]["A"]["task_ids"] + manifest["halves"]["B"]["task_ids"]
    if manifest.get("split") != "test_challenge" or completion.get("state") != "FINISHED":
        raise RuntimeError("Frozen Challenge batch is not finished")
    if manifest.get("task_count") != 417 or [entry["task_id"] for entry in entries] != expected_ids:
        raise RuntimeError("Challenge trace index disagrees with the 417-task manifest")
    if len(set(expected_ids)) != 417 or tuple(
        aggregate["official_outcomes"][name] for name in ("PASSED", "FAILED", "UNSCORED")
    ) != (315, 99, 3):
        raise RuntimeError("Challenge manifest or audited outcomes changed")
    missing = [entry for entry in entries if not entry.get("execution_archive")]
    if len(missing) != 3 or any(entry["outcome"] != "UNSCORED" for entry in missing):
        raise RuntimeError("The three missing exports are not exactly the original unscored cases")
    if Counter(entry["model_calls"] for entry in missing) != Counter({6: 2, 63: 1}):
        raise RuntimeError("Unscored model-call evidence changed")

    EXPERIMENT.mkdir(parents=True, exist_ok=False)
    original_archives = original_files = 0
    for entry in entries:
        archive = entry.get("execution_archive")
        if not archive:
            continue
        source = Path(archive)
        if not source.is_file() or not source.resolve().is_relative_to(batch.resolve()):
            raise RuntimeError("Original execution export is missing or outside the frozen batch")
        original_files += _safe_extract(source.read_bytes(), EXPERIMENT, entry["task_id"])
        original_archives += 1
    if original_archives != 414:
        raise RuntimeError("Did not stage all 414 original environment exports")

    reconstructed = []
    for entry in missing:
        reconstructed.append(_reconstruct_recorded_trial(entry, manifest["image"]))
    task_dirs = list((EXPERIMENT / "tasks").iterdir())
    if len(task_dirs) != 417 or {path.name for path in task_dirs} != set(expected_ids):
        raise RuntimeError("Staged Challenge does not have exactly 417 task directories")
    for task in task_dirs:
        if any(not (task / name).is_file() for name in REQUIRED_PER_TASK):
            raise RuntimeError("Staged Challenge task lacks a required official output file")
    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_batch_id": BATCH_ID,
        "experiment_name": EXPERIMENT_NAME,
        "source_digest": manifest["source_digest"],
        "original_official_counts": {
            name: aggregate["official_outcomes"][name] for name in ("PASSED", "FAILED", "UNSCORED")
        },
        "original_environment_archives": original_archives,
        "original_extracted_files": original_files,
        "model_free_evidence_reconstructions": reconstructed,
        "task_directories": len(task_dirs),
        "stage_directory": str(EXPERIMENT),
        "warning": (
            "Private staging only. The three original trials remain UNSCORED; "
            "their existing environment calls were replayed exactly, with no model calls. "
            "Do not publish unencrypted task output. Official whole-set re-evaluation is still required."
        ),
    }
    _atomic_json(STAGE / "conversion-manifest.private.json", record)
    print(json.dumps({key: value for key, value in record.items() if key != "model_free_evidence_reconstructions"}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
