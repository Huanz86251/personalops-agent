"""Create the private encrypted Challenge bundle with AppWorld's pack API.

Run inside the pinned AppWorld image. Mount the staged experiment outputs at
/world/experiments/outputs and the validated Normal experiment read-only at
/normal-reference. Nothing is uploaded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from appworld import update_root
from appworld.common.utils import pack_bundle
from appworld.leaderboard import PASSWORD, SALT, path_store


NAME = "personalops_dualflash_test_challenge"
update_root("/world")
outputs = Path(path_store.experiment_outputs)
experiment = outputs / NAME
reference = Path("/normal-reference")
bundle = experiment / "leaderboard.bundle"
if bundle.exists():
    raise RuntimeError("Challenge bundle already exists; refusing overwrite")

evaluation_path = experiment / "evaluations/test_challenge.json"
evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
individual = evaluation["individual"]
if len(individual) != 417 or sum(bool(row["success"]) for row in individual.values()) != 315:
    raise RuntimeError("Official Challenge evaluation does not match the frozen 315/417 result")
if evaluation["aggregate"] != {"task_goal_completion": 75.5, "scenario_goal_completion": 51.8}:
    raise RuntimeError("Official Challenge aggregate changed")

tasks = sorted((experiment / "tasks").iterdir())
if len(tasks) != 417 or any(not task.is_dir() for task in tasks):
    raise RuntimeError("Expected exactly 417 staged Challenge task directories")
required_per_task = (
    "dbs/model_hashes.json", "logs/environment_io.md", "logs/api_calls.jsonl",
    "evaluation/report.md", "version/code.txt", "version/data.txt",
)
for task in tasks:
    if any(not (task / name).is_file() for name in required_per_task):
        raise RuntimeError("Staged Challenge task is missing an official output file")

metadata = json.loads((reference / "metadata.json").read_text(encoding="utf-8"))
if metadata.get("dataset") != "test_normal":
    raise RuntimeError("Normal reference metadata has an unexpected dataset")
metadata["dataset"] = "test_challenge"
metadata["id"] = hashlib.sha256(NAME.encode("utf-8")).hexdigest()[:8]
metadata_path = experiment / "metadata.json"
if metadata_path.exists():
    if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
        raise RuntimeError("Challenge metadata already exists with different content")
else:
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")

license_path = experiment / "LICENSE"
if not license_path.exists():
    shutil.copyfile(reference / "LICENSE", license_path)
readme_path = experiment / "README_BEFORE_SHARING.md"
if not readme_path.exists():
    readme_path.write_text(
        "Private AppWorld Challenge experiment output. Do not publish this directory, "
        "raw task data, or traces. Only the encrypted leaderboard.bundle may be "
        "submitted under the official leaderboard procedure.\n",
        encoding="utf-8",
    )

pack_bundle(
    bundle_file_path=str(bundle),
    base_directory=str(outputs),
    include_directories=[NAME],
    include_file_path_substrings=[],
    exclude_extensions=[],
    password=PASSWORD,
    salt=SALT,
)
print(json.dumps({
    "experiment_name": NAME,
    "task_directories": len(tasks),
    "official_tgc": evaluation["aggregate"]["task_goal_completion"],
    "official_sgc": evaluation["aggregate"]["scenario_goal_completion"],
    "bundle_bytes": bundle.stat().st_size,
    "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
}, ensure_ascii=False), flush=True)
