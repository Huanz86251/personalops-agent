"""Read-only, local Phoenix SQLite audit. Never exports prompt/response text.

Schema adapter for this repository's Phoenix database, not a general Phoenix API.
Requires SQLite with JSONB support (SQLite >= 3.45 when attributes use JSONB).
No Phoenix import, server startup, model calls, or network access.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys


def object_value(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            result = json.loads(value)
            return result if isinstance(result, dict) else {}
        except (ValueError, TypeError):
            pass
    return {}


def response_id(span):
    """Correlate SDK and LangChain spans by provider ID, never token similarity."""
    output = object_value(span["attrs"].get("output", {}).get("value"))
    if span["name"] == "ChatCompletion":
        return output.get("id")
    if span["name"] == "ChatDeepSeek":
        generations = output.get("generations", [])
        if generations and generations[0]:
            message = generations[0][0].get("message", {})
            kwargs = message.get("kwargs", message)
            return kwargs.get("response_metadata", {}).get("id")
    return None


def summarize(spans):
    kinds = Counter(s["span_kind"] for s in spans)
    roots = [s for s in spans if s["parent_id"] is None]
    llms = [s for s in spans if s["span_kind"] == "LLM"]
    local = [s for s in llms if "local_model" in s["attrs"]]
    remote = [s for s in llms if "local_model" not in s["attrs"]]
    groups = defaultdict(list)
    unresolved = 0
    for span in remote:
        rid = response_id(span)
        if not rid:
            unresolved += 1
            continue
        llm = span["attrs"].get("llm", {})
        groups[(llm.get("provider"), llm.get("model_name"), rid)].append(span)

    prompt_total = completion_total = 0
    conflicts = missing_usage = 0
    for group in groups.values():
        # Provider identity can be reused by a replay; count once only within
        # a single trace. Across traces we refuse an authoritative total.
        counts = {(s["llm_token_count_prompt"], s["llm_token_count_completion"])
                  for s in group}
        if len(counts) != 1 or len({s["trace_rowid"] for s in group}) != 1:
            conflicts += 1
            continue
        prompt, completion = next(iter(counts))
        if prompt is None or completion is None:
            missing_usage += 1
            continue
        prompt_total += prompt
        completion_total += completion
    totals_valid = not (unresolved or conflicts or missing_usage)

    business_errors = []
    for span in spans:
        if span["span_kind"] != "TOOL":
            continue
        output = object_value(span["attrs"].get("output", {}).get("value"))
        # ToolMessage uses data.status, not a keyword search of retrieved text.
        data = object_value(output.get("data"))
        if data.get("status") == "error":
            business_errors.append(span)

    turns = []
    for root in roots:
        if root["name"] != "conversation_turn":
            continue
        members = [s for s in spans if s["trace_rowid"] == root["trace_rowid"]]
        output = object_value(root["attrs"].get("output", {}).get("value"))
        planning = object_value(output.get("planning"))
        turns.append({
            "trace_rowid": root["trace_rowid"],
            "span_status": root["status_code"],
            "duration_seconds": round((datetime.fromisoformat(root["end_time"])
                                      - datetime.fromisoformat(root["start_time"])).total_seconds(), 3),
            "span_count": len(members),
            "self_reported_final_status": planning.get("final_status"),
            "stop_reason": planning.get("overall_stop_reason"),
            "reported_model_rounds": planning.get("model_rounds_used"),
            "reported_tool_calls": planning.get("tool_calls_used"),
            "tool_errors": sum(s["trace_rowid"] == root["trace_rowid"]
                               for s in business_errors),
            "independent_task_success": None,
        })
    metadata = Counter(k for s in spans
                       for k in object_value(s["attrs"].get("metadata")))
    return {
        "span_count": len(spans),
        "trace_count": len({s["trace_rowid"] for s in spans}),
        "root_names": dict(Counter(s["name"] for s in roots)),
        "span_kinds": dict(kinds),
        "span_statuses": dict(Counter(s["status_code"] for s in spans)),
        "first_start_as_stored": min((s["start_time"] for s in spans), default=None),
        "last_end_as_stored": max((s["end_time"] for s in spans), default=None),
        "cloud_usage": {
            "raw_llm_spans": len(remote),
            "identified_provider_calls": len(groups),
            "extra_records_for_identified_calls": sum(len(g) - 1 for g in groups.values()),
            "unresolved_spans": unresolved,
            "conflicting_groups": conflicts,
            "groups_missing_usage": missing_usage,
            "total_is_valid_for_identified_cloud_calls": totals_valid,
            "prompt_tokens": prompt_total if totals_valid else None,
            "completion_tokens": completion_total if totals_valid else None,
            "raw_prompt_sum_do_not_bill": sum(s["llm_token_count_prompt"] or 0 for s in remote),
            "raw_completion_sum_do_not_bill": sum(s["llm_token_count_completion"] or 0 for s in remote),
        },
        "local_llm_usage": {
            "spans": len(local),
            # Database columns can default to zero even with absent attributes.
            "spans_with_explicit_usage": sum(
                "token_count" in s["attrs"].get("llm", {}) for s in local),
        },
        "tool_message_errors": len(business_errors),
        "tool_message_errors_with_ok_span": sum(s["status_code"] == "OK" for s in business_errors),
        "tool_message_errors_by_name": dict(Counter(s["name"] for s in business_errors)),
        "conversation_turns": sorted(turns, key=lambda t: t["trace_rowid"]),
        "metadata_keys": dict(sorted(metadata.items())),
        "limitations": [
            "Historical traffic, not an AppWorld benchmark or independent success rate.",
            "Token totals are provider-reported cloud usage, not a reconciled invoice.",
            "Missing local-model usage is unknown, not zero compute or zero tokens.",
            "Business-error detector covers the observed ToolMessage envelope only.",
            "No prompt, response text, session ID, or provider response ID is exported.",
        ],
    }


def audit(database):
    if not database.is_file():
        raise FileNotFoundError(f"Phoenix database not found: {database}")
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connection.execute("BEGIN")  # One consistent read snapshot, including WAL.
        spans = []
        for row in connection.execute(
                "SELECT span_id, parent_id, trace_rowid, name, span_kind, "
                "start_time, end_time, status_code, llm_token_count_prompt, "
                "llm_token_count_completion, json(attributes) AS attributes FROM spans"):
            span = dict(row)
            span["attrs"] = object_value(span.pop("attributes"))
            spans.append(span)
        result = summarize(spans)
        result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        result["annotation_and_experiment_counts"] = {
            table: connection.execute("SELECT count(*) FROM " + table).fetchone()[0]
            for table in ("span_annotations", "trace_annotations", "datasets",
                          "experiments", "experiment_runs")
        }
        cost = connection.execute(
            "SELECT count(*) rows, count(total_cost) priced_rows FROM span_costs").fetchone()
        result["cost_records"] = dict(cost)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path,
                        default=Path(__file__).resolve().parents[1] / ".agent/phoenix/phoenix.db")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(audit(args.database), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
