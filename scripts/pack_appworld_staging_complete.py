"""Use AppWorld's encrypted bundle API without the 0.1.3.post1 CLI filter typo.

Run inside the pinned AppWorld container with APPWORLD_ROOT=/world and the
private experiment-output directory bind-mounted. This includes only the
official experiment output tree and never uploads it.
"""
from __future__ import annotations

from pathlib import Path

from appworld import update_root
from appworld.common.utils import pack_bundle
from appworld.leaderboard import PASSWORD, SALT, path_store


NAME = "personalops_dualflash_test_normal"
update_root("/world")
outputs = Path(path_store.experiment_outputs)
experiment = outputs / NAME
required_global = (
    "metadata.json", "evaluations/test_normal.json", "evaluations/test_normal.txt",
)
required_per_task = (
    "dbs/model_hashes.json", "logs/environment_io.md", "logs/api_calls.jsonl",
    "evaluation/report.md", "version/code.txt", "version/data.txt",
)
if any(not (experiment / name).is_file() for name in required_global):
    raise RuntimeError("Official experiment output is missing a global required file")
tasks = sorted((experiment / "tasks").iterdir())
if len(tasks) != 168 or any(not task.is_dir() for task in tasks):
    raise RuntimeError("Expected 168 official task output directories")
for task in tasks:
    if any(not (task / name).is_file() for name in required_per_task):
        raise RuntimeError("An official task output is missing a required file")

bundle = experiment / "leaderboard.bundle"
pack_bundle(
    bundle_file_path=str(bundle),
    base_directory=str(outputs),
    include_directories=[NAME],
    include_file_path_substrings=[],
    exclude_extensions=[],
    password=PASSWORD,
    salt=SALT,
)
print(f"complete_bundle_written task_directories={len(tasks)} bytes={bundle.stat().st_size}")
