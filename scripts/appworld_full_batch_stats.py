"""Summarize a frozen, completed AppWorld Test-N batch without task contents.

Reads both official result ledgers and monitor indexes. The generated files are
private because their trace links and local paths point to benchmark evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import statistics
from typing import Any

from appworld_batch_stats import distribution, load_json, percentile, wilson


ROOT = Path(__file__).resolve().parents[1]


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _dist(values: list[float]) -> dict[str, float | int | None]:
    result = distribution(values)
    result["p90"] = percentile(values, 0.90)
    return result


def _sum_role_usage(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        for role, usage in (row.get("role_usage") or {}).items():
            for field, value in (usage or {}).items():
                if isinstance(value, (int, float)):
                    totals[str(role)][str(field)] += int(value)
    return {role: dict(values) for role, values in sorted(totals.items())}


def _peak_concurrency(ledgers: list[dict[str, Any]]) -> int:
    events: list[tuple[datetime, int]] = []
    for row in ledgers:
        if row.get("started_at_utc") and row.get("finished_at_utc"):
            events.append((_time(row["started_at_utc"]), 1))
            events.append((_time(row["finished_at_utc"]), -1))
    active = peak = 0
    for _, delta in sorted(events, key=lambda event: (event[0], event[1])):
        active += delta
        peak = max(peak, active)
    return peak


def _timeout_events(batch: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Read explicit interventions; never infer a timeout from UNSCORED alone."""
    path = batch / "timeout-interventions.private.jsonl"
    if not path.is_file():
        return {}
    events: dict[tuple[str, int], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") not in {"manual_startup_stall_stop", "wall_timeout_terminate_sent"}:
            continue
        if event.get("classification") != "TIMEOUT_CENSORED":
            continue
        key = (str(event["half"]), int(event["index"]))
        if key in events and events[key]["task_id"] != event["task_id"]:
            raise RuntimeError(f"Conflicting timeout evidence for {key}")
        events[key] = event
    return events


def summarize(batch_id: str, split: str = "test_normal") -> tuple[dict[str, Any], dict[str, Any], Path]:
    if split not in {"test_normal", "test_challenge"}:
        raise ValueError("Unsupported test split")
    batch = ROOT / ".agent" / ("appworld-test-normal" if split == "test_normal" else "appworld-test-challenge") / batch_id
    manifest = load_json(batch / "manifest.private.json")
    progress = load_json(batch / "progress.json", {})
    if not isinstance(manifest, dict) or manifest.get("split") != split:
        raise RuntimeError(f"Missing frozen {split} manifest")
    all_rows: list[dict[str, Any]] = []
    all_ledgers: list[dict[str, Any]] = []
    half_summary: dict[str, Any] = {}
    trace_index: list[dict[str, Any]] = []
    request_latencies: list[float] = []
    ttft_values: list[float] = []
    ttft_status: Counter[str] = Counter()
    timeout_events = _timeout_events(batch)
    matched_timeouts: set[tuple[str, int]] = set()

    for half in ("A", "B"):
        expected_ids = manifest["halves"][half]["task_ids"]
        ledger = load_json(batch / f"half-{half.lower()}-results.private.json")
        monitor_dir = ROOT / ".agent" / "appworld-monitor" / f"{batch_id}-half-{half.lower()}"
        rows = load_json(monitor_dir / "index.json")
        if not isinstance(ledger, list) or not isinstance(rows, list):
            raise RuntimeError(f"Half {half}: missing result ledger or monitor index")
        if len(ledger) != len(expected_ids) or len(rows) != len(expected_ids):
            raise RuntimeError(
                f"Half {half}: incomplete; expected {len(expected_ids)}, "
                f"ledger {len(ledger)}, indexed {len(rows)}"
            )
        ledger_by_index = {int(item["index"]): item for item in ledger}
        row_by_index = {int(str(item["trial_id"]).split("-", 1)[0]): item for item in rows}
        if len(ledger_by_index) != len(expected_ids) or len(row_by_index) != len(expected_ids):
            raise RuntimeError(f"Half {half}: duplicate or missing task indices")
        for index, task_id in enumerate(expected_ids, start=1):
            item = ledger_by_index.get(index)
            row = row_by_index.get(index)
            if item is None or row is None or item["task_id"] != task_id:
                raise RuntimeError(f"Half {half}, task {index}: frozen manifest and results disagree")
            # A process may stop before writing its own manifest. In that case
            # the parent ledger still identifies the attempted task, while the
            # monitor row correctly records missing trial evidence.
            if row.get("task_id") not in (None, task_id):
                raise RuntimeError(f"Half {half}, task {index}: monitor task ID disagrees")
            if item["outcome"] != row["outcome"]:
                raise RuntimeError(f"Half {half}, task {index}: official ledger and monitor disagree")
            timeout_event = timeout_events.get((half, index))
            if timeout_event:
                if timeout_event["task_id"] != task_id:
                    raise RuntimeError(f"Half {half}, task {index}: timeout task ID disagrees")
                matched_timeouts.add((half, index))

        counts = Counter(row["outcome"] for row in rows)
        started = min(_time(item["started_at_utc"]) for item in ledger)
        finished = max(_time(item["finished_at_utc"]) for item in ledger)
        half_summary[half] = {
            "tasks": len(rows),
            "outcomes": dict(counts),
            "started_at_utc": started.isoformat(),
            "finished_at_utc": finished.isoformat(),
            "wall_seconds": (finished - started).total_seconds(),
        }
        for index, task_id in enumerate(expected_ids, start=1):
            row = dict(row_by_index[index])
            row["task_id"] = task_id
            item = ledger_by_index[index]
            censored = (half, index) in timeout_events and row["outcome"] == "UNSCORED"
            row["timeout_censored"] = censored
            trial = Path(row["trial_directory"])
            if not trial.is_dir():
                raise RuntimeError(f"Half {half}, task {index}: trial directory absent")
            spans_file = trial / "spans.private.json"
            result_file = trial / "result.private.json"
            execution_file = trial / "execution.private.zip"
            usage_file = trial / "usage.private.json"
            trace_index.append({
                "half": half,
                "index": index,
                "task_id": task_id,
                "outcome": row["outcome"],
                "timeout_censored": censored,
                "trace_id": row.get("trace_id"),
                "phoenix_url": row.get("phoenix_url"),
                "trial_directory": str(trial),
                "spans_file": str(spans_file) if spans_file.is_file() else None,
                "execution_archive": str(execution_file) if execution_file.is_file() else None,
                "usage_file": str(usage_file) if usage_file.is_file() else None,
                "result_file": str(result_file) if result_file.is_file() else None,
                "model_calls": row.get("model_calls"),
                "elapsed_seconds": row.get("elapsed_seconds"),
                "known_cost_cny": row.get("known_cost_cny"),
            })
            usage = load_json(usage_file, [])
            usage_rows = list(usage.values()) if isinstance(usage, dict) else usage or []
            for request in usage_rows:
                if isinstance(request, dict) and isinstance(request.get("latency_seconds"), (int, float)):
                    request_latencies.append(float(request["latency_seconds"]))
            spans = load_json(spans_file, []) or []
            if not isinstance(spans, list):
                raise RuntimeError(f"Half {half}, task {index}: invalid span evidence")
            for span in spans:
                attributes = span.get("attributes") or {}
                status = attributes.get("timing.first_token_status")
                if status:
                    ttft_status[str(status)] += 1
                ttft = attributes.get("timing.first_token_ms")
                if status == "observed" and isinstance(ttft, (int, float)):
                    ttft_values.append(float(ttft))
            all_rows.append(row)
            all_ledgers.append(item)

    expected_total = int(manifest["task_count"])
    if len(all_rows) != expected_total or len({row["task_id"] for row in all_rows}) != expected_total:
        raise RuntimeError(f"Full {split} task coverage is not complete and unique")
    if matched_timeouts != set(timeout_events):
        raise RuntimeError("Timeout intervention does not match the frozen task manifest")
    counts = Counter(row["outcome"] for row in all_rows)
    passed = counts["PASSED"]
    officially_scored = passed + counts["FAILED"]
    censored = sum(bool(row["timeout_censored"]) for row in all_rows)
    other_unscored = counts["UNSCORED"] - censored
    score_eligible = expected_total - censored
    starts = [_time(row["started_at_utc"]) for row in all_ledgers]
    finishes = [_time(row["finished_at_utc"]) for row in all_ledgers]
    wall_seconds = (max(finishes) - min(starts)).total_seconds()
    input_tokens = sum(int(row.get("input_tokens") or 0) for row in all_rows)
    cache_tokens = sum(int(row.get("cache_read_tokens") or 0) for row in all_rows)
    known_cost = sum(float(row.get("known_cost_cny") or 0) for row in all_rows)
    missing_cost = sum(int(row.get("cost_missing_requests") or 0) for row in all_rows)
    missing_usage = sum(int(row.get("usage_missing_requests") or 0) for row in all_rows)
    outcomes = {name: counts[name] for name in ("PASSED", "FAILED", "UNSCORED")}
    failure_kinds = Counter(str(row.get("failure_kind") or "unknown") for row in all_rows if row["outcome"] != "PASSED" and not row["timeout_censored"])
    metric = {
        "batch": {
            "batch_id": batch_id,
            "split": manifest["split"],
            "seed": manifest["seed"],
            "image": manifest["image"],
            "source_digest": manifest["source_digest"],
            "complete": True,
            "task_count": expected_total,
            "parallelism_requested": progress.get("parallelism"),
            "peak_concurrency_observed": _peak_concurrency(all_ledgers),
            "started_at_utc": min(starts).isoformat(),
            "finished_at_utc": max(finishes).isoformat(),
            "wall_seconds": wall_seconds,
            "throughput_tasks_per_hour": expected_total * 3600 / wall_seconds,
            "halves": half_summary,
        },
        "official_outcomes": {
            **outcomes,
            "score_eligible_count": score_eligible,
            "officially_scored_count": officially_scored,
            "timeout_censored_count": censored,
            "other_unscored_count": other_unscored,
            "scoring_rule": "Timeout-censored trials are excluded. Other unscored trials remain nonpasses in score_eligible_success_rate; scored_success_rate is a separate conditional diagnostic.",
            "conservative_success_rate": passed / expected_total,
            "score_eligible_success_rate": passed / score_eligible if score_eligible else None,
            "scored_success_rate": passed / officially_scored if officially_scored else None,
            "conservative_wilson_95": wilson(passed, expected_total),
            "score_eligible_wilson_95": wilson(passed, score_eligible) if score_eligible else None,
            "failure_kinds": dict(failure_kinds),
        },
        "task_elapsed_seconds": _dist([
            float(row["elapsed_seconds"]) for row in all_rows
            if isinstance(row.get("elapsed_seconds"), (int, float))
        ]),
        "task_end_to_end_seconds": _dist([
            (_time(row["finished_at_utc"]) - _time(row["started_at_utc"])).total_seconds()
            for row in all_ledgers
        ]),
        "task_end_to_end_scored_seconds": _dist([
            (_time(row["finished_at_utc"]) - _time(row["started_at_utc"])).total_seconds()
            for row in all_ledgers if row["outcome"] in {"PASSED", "FAILED"}
        ]),
        "model_calls": {
            "total": sum(int(row.get("model_calls") or 0) for row in all_rows),
            "per_task": _dist([float(row.get("model_calls") or 0) for row in all_rows]),
        },
        "model_request_latency_seconds": _dist(request_latencies),
        "time_to_first_token_ms": {
            **_dist(ttft_values),
            "status_counts": dict(ttft_status),
            "available": bool(ttft_values),
        },
        "tokens": {
            "input_total": input_tokens,
            "output_total": sum(int(row.get("output_tokens") or 0) for row in all_rows),
            "reasoning_total": sum(int(row.get("reasoning_tokens") or 0) for row in all_rows),
            "cache_read_total": cache_tokens,
            "cache_hit_ratio_weighted": cache_tokens / input_tokens if input_tokens else None,
            "cache_hit_ratio_per_task": _dist([
                float(row["cache_hit_ratio"]) for row in all_rows
                if isinstance(row.get("cache_hit_ratio"), (int, float))
            ]),
            "input_per_task": _dist([float(row.get("input_tokens") or 0) for row in all_rows]),
            "output_per_task": _dist([float(row.get("output_tokens") or 0) for row in all_rows]),
        },
        "cost_cny": {
            "known_total": known_cost,
            "missing_request_count": missing_cost,
            "status": "lower_bound_public_price_estimate" if missing_cost else "public_price_estimate",
            "per_task_known_cost": _dist([float(row.get("known_cost_cny") or 0) for row in all_rows]),
            "known_cost_per_pass": known_cost / passed if passed else None,
        },
        "role_usage": _sum_role_usage(all_rows),
        "trace_quality": {
            "indexed_tasks": len(trace_index),
            "tasks_with_trace_id": sum(bool(row.get("trace_id")) for row in all_rows),
            "tasks_with_phoenix_url": sum(bool(row.get("phoenix_url")) for row in all_rows),
            "tasks_with_span_file": sum(bool(row.get("spans_file")) for row in trace_index),
            "tasks_with_usage_file": sum(bool(row.get("usage_file")) for row in trace_index),
            "tasks_with_result_file": sum(bool(row.get("result_file")) for row in trace_index),
            "tasks_with_execution_archive": sum(bool(row.get("execution_archive")) for row in trace_index),
            "tasks_with_error_spans": sum(int(row.get("error_span_count") or 0) > 0 for row in all_rows),
            "total_error_spans": sum(int(row.get("error_span_count") or 0) for row in all_rows),
            "missing_usage_requests": missing_usage,
        },
        "outcome_cohorts": {
            name: {
                "tasks": len(selected),
                "mean_elapsed_seconds": statistics.fmean(float(row.get("elapsed_seconds") or 0) for row in selected) if selected else None,
                "mean_model_calls": statistics.fmean(float(row.get("model_calls") or 0) for row in selected) if selected else None,
                "mean_known_cost_cny": statistics.fmean(float(row.get("known_cost_cny") or 0) for row in selected) if selected else None,
            }
            for name, selected in {
                "PASSED": [row for row in all_rows if row["outcome"] == "PASSED"],
                "FAILED": [row for row in all_rows if row["outcome"] == "FAILED"],
                "UNSCORED_OTHER": [row for row in all_rows if row["outcome"] == "UNSCORED" and not row["timeout_censored"]],
                "TIMEOUT_CENSORED": [row for row in all_rows if row["timeout_censored"]],
            }.items()
        },
    }
    index = {
        "batch_id": batch_id,
        "phoenix_project": progress.get("phoenix_project"),
        "warning": "Private benchmark evidence; do not publish raw traces or task data.",
        "traces": trace_index,
    }
    return metric, index, batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--split", choices=("test_normal", "test_challenge"), default="test_normal")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    metrics, traces, batch = summarize(args.batch_id, args.split)
    if not args.check_only:
        (batch / "full-aggregate.private.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (batch / "demo-trace-index.private.json").write_text(
            json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
