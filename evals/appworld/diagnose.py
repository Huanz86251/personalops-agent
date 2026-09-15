"""Deterministic Train/Dev triage. Emits metadata/counts, never task or response text.

The official grade remains authoritative. Detector labels describe observations,
not proven root causes. Does not call models, rewrite a trial, or inspect test splits.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3


def tool_failure_label(output):
    if not output.startswith("Execution failed."):
        return None
    if "No API named '" in output:
        return "unknown_api_name"
    if "takes 0 positional arguments but" in output:
        return "positional_arguments"
    if "Response status code is 422:" in output:
        return "api_validation"
    if "Usage of the following module is not allowed:" in output:
        return "disallowed_module"
    return "other_execution_failure"


def summarize_trial(result, trajectory):
    metadata = result["metadata"]
    if metadata.get("split") not in ("train", "dev"):
        raise ValueError("Per-task diagnostic analysis is restricted to Train/Dev")
    tool_failures = []
    for index, step in enumerate(trajectory, 1):
        label = tool_failure_label(step.get("output", ""))
        if label:
            tool_failures.append({"tool_call_index": index, "label": label})
    records = result.get("usage", {}).get("records", [])
    truncations = [i for i, r in enumerate(records, 1)
                   if r.get("finish_reason") == "length" or r.get("error_type") == "LengthFinishReasonError"]
    report = result.get("agent") or {}
    alerts = []
    if result.get("status") == "infrastructure_error":
        alerts.append("infrastructure_failure_not_a_model_quality_score")
    if result.get("official_task_success") is False:
        alerts.append("official_task_failed")
    if report.get("self_reported_final_status") == "COMPLETED" and result.get("official_task_success") is False:
        alerts.append("self_report_disagrees_with_official_grade")
    if report.get("self_reported_final_status") in ("FAILED", "PARTIAL") and result.get("official_task_success") is True:
        alerts.append("official_success_but_self_report_incomplete")
    usage = result.get("usage", {})
    if not usage.get("usage_complete", False):
        alerts.append("incomplete_token_usage")
    if truncations:
        alerts.append("model_output_truncated")
    # Reaching a ceiling does not itself prove why the task failed.
    if usage.get("model_calls_started", 0) >= metadata.get("max_model_calls", float("inf")):
        alerts.append("model_call_ceiling_reached")
    return {
        "trial_id": metadata["trial_id"], "split": metadata["split"],
        "purpose": metadata.get("purpose", "legacy_unspecified"),
        "status": result.get("status"),
        "official_task_success": result.get("official_task_success"),
        "self_reported_final_status": report.get("self_reported_final_status"),
        "model_calls_started": usage.get("model_calls_started"),
        "model_calls_returned": usage.get("model_calls_returned"),
        "usage_complete": usage.get("usage_complete"),
        "known_input_tokens": sum(r.get("input_tokens") or 0 for r in records),
        "known_output_tokens": sum(r.get("output_tokens") or 0 for r in records),
        "complete_input_tokens": usage.get("input_tokens"),
        "complete_output_tokens": usage.get("output_tokens"),
        "truncated_model_call_indices": truncations,
        "tool_calls_executed": len(trajectory), "tool_failures": tool_failures,
        "tool_failure_counts": dict(Counter(x["label"] for x in tool_failures)),
        "first_observed_tool_failure": tool_failures[0] if tool_failures else None,
        "elapsed_seconds": result.get("elapsed_seconds"), "alerts": alerts,
        "limits": [
            "Tool detectors recognize AppWorld's execution-error envelope; a model can print similar text.",
            "Indices follow recorded callback completion/serialized tool order, not inferred causality.",
            "No alert overrides official grading; a successful execution need not fulfill the user request.",
            "Known token sums are lower bounds if usage is incomplete, not a provider invoice.",
        ],
    }


def trace_summary(database, trial_id):
    if not database.is_file():
        raise FileNotFoundError(database)
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only = ON")
        db.execute("BEGIN")
        roots = list(db.execute(
            "SELECT s.trace_rowid, s.status_code, t.trace_id, s.name FROM spans s "
            "JOIN traces t ON t.id=s.trace_rowid "
            "WHERE s.name IN ('appworld.trial', 'EVAL / AppWorld trial') "
            "AND json_extract(s.attributes, '$.session.id')=?", (trial_id,)))
        if len(roots) != 1:
            return {"matched_trial_roots": len(roots), "coverage_verified": False}
        root = roots[0]
        spans = list(db.execute(
            "SELECT name, span_kind, status_code FROM spans WHERE trace_rowid=?", (root["trace_rowid"],)))
        return {
            "matched_trial_roots": 1, "coverage_verified": True,
            "trace_id": root["trace_id"], "root_span_name": root["name"],
            "root_span_status": root["status_code"],
            "span_count": len(spans),
            "span_kind_counts": dict(Counter(s["span_kind"] for s in spans)),
            "error_span_count": sum(s["status_code"] == "ERROR" for s in spans),
            "limits": "A matching root proves trace association, not completeness of every model event.",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial_directory", type=Path)
    parser.add_argument("--phoenix-db", type=Path)
    args = parser.parse_args()
    result_path = args.trial_directory / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8-sig"))
    if result.get("metadata", {}).get("split") not in ("train", "dev"):
        parser.error("Do not run per-task triage on final Test-N/Test-C tasks")
    trajectory_path = args.trial_directory / "trajectory.private.json"
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8-sig"))
    report = summarize_trial(result, trajectory)
    report["input_hashes"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (result_path, trajectory_path)}
    if args.phoenix_db:
        report["phoenix"] = trace_summary(args.phoenix_db, result["metadata"]["trial_id"])
    output = args.trial_directory / "diagnostics.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
