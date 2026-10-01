"""Stage corrected Test-N AppWorld outputs privately for official format checks.

This does not run the agent, change its score, pack a public bundle, or upload
anything. One genuinely unscored task is reconstructed from its sole recorded
read-only API-docs call; the original model run and result remain untouched.
"""
from __future__ import annotations

import ast
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.appworld.protocol import DockerWorld
from evals.appworld.run_test_normal_batch import _atomic_json


BATCH_ID = "test-normal-full-p4-lowtemp-20260916-demo"
EXPERIMENT_NAME = "personalops_dualflash_test_normal"
STAGE = ROOT / ".agent" / "appworld-leaderboard-staging"
EXPERIMENT = STAGE / "experiments" / "outputs" / EXPERIMENT_NAME


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_extract(archive_bytes: bytes, experiment: Path, task_id: str) -> int:
    count = 0
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as source:
        for member in source.infolist():
            if member.is_dir():
                continue
            parts = Path(member.filename).parts
            if len(parts) < 4 or parts[:2] != ("tasks", task_id) or any(part in ("", ".", "..") for part in parts):
                raise RuntimeError("Execution archive contains an unexpected path")
            destination = (experiment / member.filename).resolve()
            if not destination.is_relative_to(experiment.resolve()):
                raise RuntimeError("Execution archive path escaped private staging")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as incoming, destination.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            count += 1
    return count


def _verify_read_only_documentation(code: str) -> None:
    tree = ast.parse(code)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        raise RuntimeError("Unscored task's recorded world call is not a single expression")
    names = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    if names != ["print", "apis.api_docs.show_api_doc"]:
        raise RuntimeError("Refusing to replay anything beyond one read-only API-docs call")


def _reconstruct_read_only_failure(entry: dict, experiment: Path, image: str) -> dict:
    trial = Path(entry["trial_directory"])
    result = _read(trial / "result.private.json")
    calls = _read(trial / "world-calls.private.json")
    if result.get("official_evaluation") is not None or result.get("status") != "finished":
        raise RuntimeError("Remaining task is not genuinely unscored")
    if not isinstance(calls, list) or len(calls) != 1 or calls[0].get("role") != "harness_documentation":
        raise RuntimeError("Remaining task's world calls are not exactly one documented read")
    call = calls[0]
    code = call["code"]
    _verify_read_only_documentation(code)
    if call["execution_state"]["recent_calls"][0]["apis"] != ["api_docs.show_api_doc"]:
        raise RuntimeError("Recorded call was not the expected read-only documentation API")
    with DockerWorld(image=image) as world:
        world.request("initialize", split="test_normal", task_id=entry["task_id"],
                      trial_id="leaderboard-readonly-reconstruction", max_interactions=90)
        replay_output = world.execute(code)
        if replay_output != call["output"]:
            raise RuntimeError("Deterministic documentation replay did not reproduce the original output")
        world.request("finish")
        evaluation = world.request("evaluate")
        if evaluation.get("success") is not False:
            raise RuntimeError("The reconstructed unchanged world did not receive an official failure")
        archive_bytes = base64.b64decode(world.request("export")["zip_base64"])
    evidence_dir = STAGE / "private-reconstruction"
    evidence_dir.mkdir(parents=True, exist_ok=False)
    archive = evidence_dir / "read-only-unscored-execution.private.zip"
    archive.write_bytes(archive_bytes)
    files = _safe_extract(archive_bytes, experiment, entry["task_id"])
    return {
        "task_id": entry["task_id"], "original_status": "UNSCORED",
        "recorded_world_calls": 1, "replayed_world_calls": 1,
        "original_output_sha256": _hash(call["output"].encode("utf-8")),
        "replayed_output_sha256": _hash(replay_output.encode("utf-8")),
        "reconstructed_official_success": False,
        "execution_archive": str(archive), "archive_sha256": _hash(archive_bytes),
        "extracted_files": files,
        "warning": "Model was not rerun. This is a model-free replay of its one recorded read-only environment call; original run remains UNSCORED in the local ledger.",
    }


def main() -> None:
    if STAGE.exists():
        raise RuntimeError("Private staging directory already exists; refusing overwrite")
    batch = ROOT / ".agent" / "appworld-test-normal" / BATCH_ID
    manifest = _read(batch / "manifest.private.json")
    trace_index = _read(batch / "demo-trace-index.private.json")
    aggregate = _read(batch / "full-aggregate.private.json")
    entries = trace_index["traces"]
    if manifest.get("task_count") != 168 or len(entries) != 168 or len({entry["task_id"] for entry in entries}) != 168:
        raise RuntimeError("Corrected Test-N task set is not exactly 168 unique tasks")
    if tuple(aggregate["official_outcomes"][name] for name in ("PASSED", "FAILED", "UNSCORED")) != (140, 27, 1):
        raise RuntimeError("Corrected official outcome ledger no longer matches the audited result")
    STAGE.mkdir(parents=True, exist_ok=False)
    EXPERIMENT.mkdir(parents=True, exist_ok=False)
    complete = 0
    extracted = 0
    missing = []
    for entry in entries:
        path = entry.get("execution_archive")
        if not path:
            missing.append(entry)
            continue
        source = Path(path)
        if not source.is_file() or not source.resolve().is_relative_to(batch.resolve()):
            raise RuntimeError("Execution archive is missing or escaped its private batch")
        extracted += _safe_extract(source.read_bytes(), EXPERIMENT, entry["task_id"])
        complete += 1
    if complete != 167 or len(missing) != 1 or missing[0]["outcome"] != "UNSCORED":
        raise RuntimeError("Expected 167 real exports and one model-executed unscored case")
    reconstruction = _reconstruct_read_only_failure(missing[0], EXPERIMENT, manifest["image"])
    task_dirs = list((EXPERIMENT / "tasks").iterdir())
    if len(task_dirs) != 168:
        raise RuntimeError("Official-format task directory coverage is not 168")
    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_batch_id": BATCH_ID,
        "experiment_name": EXPERIMENT_NAME,
        "source_digest": manifest["source_digest"],
        "original_official_counts": {name: aggregate["official_outcomes"][name] for name in ("PASSED", "FAILED", "UNSCORED")},
        "real_execution_archives": complete,
        "real_extracted_files": extracted,
        "model_free_read_only_reconstruction": reconstruction,
        "task_directories": len(task_dirs),
        "stage_directory": str(EXPERIMENT),
        "warning": "Private staging only. Do not publish unencrypted task data. Official whole-dataset evaluation and pack/unpack checks remain required.",
    }
    _atomic_json(STAGE / "conversion-manifest.private.json", record)
    print(json.dumps({key: value for key, value in record.items() if key != "model_free_read_only_reconstruction"}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
