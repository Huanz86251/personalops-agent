"""Start Phoenix for a demo or print a safe evaluation summary.

This helper never calls a model and never prints task text, tool code, tool
outputs, credentials, or private grader details.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EVALUATIONS = ROOT / ".agent" / "evaluations"
PHOENIX_URL = "http://127.0.0.1:6007"


def result_files():
    if not EVALUATIONS.is_dir():
        return []
    return sorted(
        EVALUATIONS.glob("*/result.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def select_result(trial_id=None):
    if trial_id:
        path = EVALUATIONS / trial_id / "result.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    files = result_files()
    if not files:
        raise FileNotFoundError("No evaluation result.json files were found")
    return files[0]


def _tool_counts(directory):
    path = directory / "trajectory.private.json"
    if not path.is_file():
        return None, None
    trajectory = json.loads(path.read_text(encoding="utf-8-sig"))
    failures = sum(
        str(item.get("output", "")).startswith("Execution failed.")
        for item in trajectory
    )
    return len(trajectory), failures


def safe_summary(path):
    result = json.loads(path.read_text(encoding="utf-8-sig"))
    metadata = result.get("metadata") or {}
    if metadata.get("trial_id"):
        tool_calls, tool_failures = _tool_counts(path.parent)
        usage = result.get("usage") or {}
        agent = result.get("agent") or {}
        summary = {
            "kind": "appworld_agent_evaluation",
            "trial_id": metadata["trial_id"],
            "split": metadata.get("split"),
            "purpose": metadata.get("purpose"),
            "status": result.get("status"),
            "official_task_success": result.get("official_task_success"),
            "self_reported_status": agent.get("self_reported_final_status"),
            "model_calls": usage.get("model_calls_started"),
            "usage_complete": usage.get("usage_complete"),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "tool_calls": tool_calls,
            "tool_failures": tool_failures,
            "elapsed_seconds": result.get("elapsed_seconds"),
            "phoenix": {
                "ui": PHOENIX_URL,
                "project": metadata.get("phoenix_project"),
                "trace_id": metadata.get("phoenix_trace_id"),
                "trace_profile": metadata.get("phoenix_trace_profile"),
                "root_persisted": (result.get("trace_delivery") or {}).get(
                    "root_persisted"
                ),
            },
            "result_path": str(path),
        }
        warnings = []
        if summary["official_task_success"] is False:
            warnings.append("official_task_failed")
        if summary["official_task_success"] != (
            summary["self_reported_status"] == "COMPLETED"
        ):
            warnings.append("official_grade_and_agent_self_report_disagree")
        if not summary["usage_complete"]:
            warnings.append("token_usage_is_a_known_lower_bound")
        if summary["phoenix"]["trace_id"] and not summary["phoenix"]["root_persisted"]:
            warnings.append("trace_root_was_not_confirmed_durable")
        summary["warnings"] = warnings
        return summary

    world = result.get("world") or {}
    return {
        "kind": result.get("kind", "evaluation_check"),
        "check_id": result.get("check_id"),
        "passed": result.get("passed"),
        "external_model_calls": result.get("external_model_calls"),
        "compatibility": {
            "docker_and_appworld_started": bool(world),
            "persistent_execution_verified": world.get(
                "persistent_execution_verified"
            ),
            "premature_grading_rejected": world.get(
                "premature_evaluation_rejected"
            ),
            "execution_after_finish_rejected": world.get(
                "post_finish_execution_rejected"
            ),
            "official_evaluator_returned": world.get(
                "official_evaluator_returned"
            ),
        },
        "official_task_success": world.get("official_task_success"),
        "official_task_success_explanation": (
            "Expected false: this check verifies infrastructure and deliberately "
            "does not solve the sampled task."
        ),
        "phoenix": {
            "ui": PHOENIX_URL,
            "project": "personalops-eval-infrastructure",
            "trace_id": result.get("trace_id"),
            "root_persisted_before_stop": (
                result.get("before_stop") or {}
            ).get("root_persisted"),
            "root_persisted_after_stop": (
                result.get("after_stop") or {}
            ).get("root_persisted"),
        },
        "result_path": str(path),
    }


def serve():
    os.environ.update({
        "PHOENIX_HOST": "127.0.0.1",
        "PHOENIX_PORT": "6007",
        "PHOENIX_TELEMETRY_ENABLED": "false",
        "PHOENIX_DISABLE_AGENT_ASSISTANT": "true",
        "PHOENIX_ALLOWED_SANDBOX_PROVIDERS": "NONE",
    })
    from phoenix_runtime import PhoenixServerRuntime

    runtime = PhoenixServerRuntime()
    runtime.start()
    print(json.dumps({
        "phoenix_ui": runtime.ui_url,
        "database": str(runtime.database_path),
        "model_calls": 0,
        "next": [
            "Open the Phoenix URL in a browser.",
            "Choose project personalops-eval-infrastructure for the zero-model check.",
            "Choose personalops-eval-private-learning for complete local Agent traces.",
            "Press Ctrl+C here when the demo is finished.",
        ],
    }, ensure_ascii=False, indent=2))
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        runtime.stop()


def list_results(limit):
    rows = []
    for path in result_files()[:limit]:
        result = json.loads(path.read_text(encoding="utf-8-sig"))
        rows.append({
            "id": (result.get("metadata") or {}).get("trial_id")
                  or result.get("check_id")
                  or path.parent.name,
            "kind": result.get("kind") or "appworld_agent_evaluation",
            "status": result.get("status")
                      or ("passed" if result.get("passed") else "failed"),
            "modified": datetime.fromtimestamp(
                path.stat().st_mtime
            ).isoformat(timespec="seconds"),
        })
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("serve", help="Start/reuse the local Phoenix UI; zero model calls")
    inspect_parser = subparsers.add_parser(
        "inspect", help="Print a safe summary for one or the latest evaluation"
    )
    inspect_parser.add_argument("--trial-id")
    list_parser = subparsers.add_parser("list", help="List recent evaluation IDs")
    list_parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    if args.command == "serve":
        serve()
    elif args.command == "inspect":
        print(json.dumps(
            safe_summary(select_result(args.trial_id)),
            ensure_ascii=False,
            indent=2,
        ))
    else:
        list_results(args.limit)


if __name__ == "__main__":
    main()
