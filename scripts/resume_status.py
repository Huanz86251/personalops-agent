"""Read-only session recovery check. Never loads keys, calls models, or starts services."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    handoff = root / ".agent/handoffs/2026-08-31"
    manifest_path = handoff / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("Read docs/SESSION_HANDOFF.md; local private evidence manifest is missing.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    changed, missing = [], []
    for relative, expected in manifest["working_source_sha256"].items():
        path = root / relative
        if not path.is_file():
            missing.append(relative)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            changed.append(relative)
    evidence_missing = [p for p in manifest["evidence_sha256"] if not (handoff / p).is_file()]
    rows = []
    for directory in sorted((root / ".agent/evaluations").glob("aw_*")):
        result = directory / "result.json"
        if not result.is_file():
            rows.append({"trial_id": directory.name, "terminal_result_found": False})
            continue
        value = json.loads(result.read_text(encoding="utf-8-sig"))
        rows.append({"trial_id": directory.name,
                     "status": value.get("status"),
                     "official_task_success": value.get("official_task_success"),
                     "model_calls": value.get("usage", {}).get("model_calls_started")})
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = None
    print(json.dumps({
        "read_first": str(root / "docs/SESSION_HANDOFF.md"),
        "detailed_log": str(root / "docs/worklogs/2026-08-31.md"),
        "checkpoint_git_sha": manifest["git_sha"], "current_git_sha": git_sha,
        "changed_checkpoint_source_files": changed,
        "missing_checkpoint_source_files": missing, "missing_private_evidence_files": evidence_missing,
        "trials": rows,
        "user_boundary": "Stop after today's integration verification. No automatic paid/batch/SkillOpt continuation.",
        "next_step": "Explain the saved calibration evidence to the user before planning more experiments.",
        "limitations": ["No live process/network/model/provider-billing check is performed.",
                       "New files not listed at checkpoint are not included in source comparison.",
                       "Evidence files are checked for existence here; manifest hashes support deeper verification."]
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
