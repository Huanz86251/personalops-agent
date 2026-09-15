"""Aggregate one frozen AppWorld half without reading task or message contents."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    observed = successes / total
    denominator = 1 + z * z / total
    center = (observed + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(observed * (1 - observed) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def iso_seconds(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    try:
        return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
    except ValueError:
        return None


def aggregate(batch_id: str, half: str) -> tuple[dict[str, Any], Path]:
    half = half.upper()
    batch = ROOT / ".agent/appworld-test-normal" / batch_id
    manifest = load_json(batch / "manifest.private.json", {})
    progress = load_json(batch / "progress.json", {})
    launcher = load_json(batch / "launcher.json", {})
    monitor = ROOT / ".agent/appworld-monitor" / f"{batch_id}-half-{half.lower()}"
    rows = load_json(monitor / "index.json", [])
    if not isinstance(rows, list):
        raise RuntimeError("Monitor index is missing or invalid")
    expected = int(manifest.get("halves", {}).get(half, {}).get("count", 0))
    outcomes = {"PASSED": 0, "FAILED": 0, "UNSCORED": 0}
    for row in rows:
        outcome = str(row.get("outcome") or "UNSCORED")
        outcomes[outcome if outcome in outcomes else "UNSCORED"] += 1

    input_tokens = sum(int(row.get("input_tokens") or 0) for row in rows)
    output_tokens = sum(int(row.get("output_tokens") or 0) for row in rows)
    cache_tokens = sum(int(row.get("cache_read_tokens") or 0) for row in rows)
    reasoning_tokens = sum(int(row.get("reasoning_tokens") or 0) for row in rows)
    model_calls = sum(int(row.get("model_calls") or 0) for row in rows)
    known_cost = sum(float(row.get("known_cost_cny") or 0) for row in rows)
    missing_cost_requests = sum(int(row.get("cost_missing_requests") or 0) for row in rows)
    elapsed = [float(row["elapsed_seconds"]) for row in rows if isinstance(row.get("elapsed_seconds"), (int, float))]
    costs = [float(row["known_cost_cny"]) for row in rows if isinstance(row.get("known_cost_cny"), (int, float))]
    calls_per_task = [float(row.get("model_calls") or 0) for row in rows]
    per_task_cache = [float(row["cache_hit_ratio"]) for row in rows if isinstance(row.get("cache_hit_ratio"), (int, float))]
    task_input = [float(row.get("input_tokens") or 0) for row in rows]
    task_output = [float(row.get("output_tokens") or 0) for row in rows]

    request_latencies: list[float] = []
    ttft_ms: list[float] = []
    ttft_status: dict[str, int] = {}
    role_usage: dict[str, dict[str, int]] = {}
    for row in rows:
        for role, usage in (row.get("role_usage") or {}).items():
            target = role_usage.setdefault(
                str(role),
                {"model_calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0},
            )
            for field in target:
                target[field] += int((usage or {}).get(field) or 0)
        trial = Path(row["trial_directory"])
        usage = load_json(trial / "usage.private.json", [])
        usage_rows = list(usage.values()) if isinstance(usage, dict) else usage or []
        for record in usage_rows:
            value = record.get("latency_seconds") if isinstance(record, dict) else None
            if isinstance(value, (int, float)):
                request_latencies.append(float(value))
        spans = load_json(trial / "spans.private.json", []) or []
        for span in spans:
            attributes = span.get("attributes") or {}
            status = attributes.get("timing.first_token_status")
            if status:
                ttft_status[str(status)] = ttft_status.get(str(status), 0) + 1
            value = attributes.get("timing.first_token_ms")
            if status == "observed" and isinstance(value, (int, float)):
                ttft_ms.append(float(value))

    passed = outcomes["PASSED"]
    scored = passed + outcomes["FAILED"]
    task_count = len(rows)
    wall_seconds = iso_seconds(launcher.get("started_at"), progress.get("updated_at_utc"))
    cohorts: dict[str, dict[str, float | int | None]] = {}
    for name, selected in {
        "PASSED": [row for row in rows if row.get("outcome") == "PASSED"],
        "FAILED_OR_UNSCORED": [row for row in rows if row.get("outcome") != "PASSED"],
    }.items():
        count = len(selected)
        cohorts[name] = {
            "tasks": count,
            "mean_cost_cny": statistics.fmean(float(row.get("known_cost_cny") or 0) for row in selected) if count else None,
            "mean_elapsed_seconds": statistics.fmean(float(row.get("elapsed_seconds") or 0) for row in selected) if count else None,
            "mean_model_calls": statistics.fmean(float(row.get("model_calls") or 0) for row in selected) if count else None,
            "mean_total_tokens": statistics.fmean(float(row.get("input_tokens") or 0) + float(row.get("output_tokens") or 0) for row in selected) if count else None,
        }
    metrics = {
        "batch": {
            "batch_id": batch_id,
            "half": half,
            "split": manifest.get("split"),
            "seed": manifest.get("seed"),
            "status": progress.get("status"),
            "expected_tasks": expected,
            "indexed_tasks": task_count,
            "complete": bool(expected and expected == task_count and progress.get("status") == "FINISHED"),
            "wall_seconds": wall_seconds,
            "throughput_tasks_per_hour": task_count / (wall_seconds / 3600) if wall_seconds else None,
        },
        "official_outcomes": {
            **outcomes,
            "scored_tasks": scored,
            "conservative_success_rate": passed / expected if expected else None,
            "scored_success_rate": passed / scored if scored else None,
            "conservative_wilson_95": wilson(passed, expected),
        },
        "cost_cny": {
            "known_total": known_cost,
            "missing_request_count": missing_cost_requests,
            "status": "lower_bound_public_price_estimate" if missing_cost_requests else "public_price_estimate",
            "per_task": distribution(costs),
            "known_cost_per_pass": known_cost / passed if passed else None,
        },
        "tokens": {
            "input_total": input_tokens,
            "output_total": output_tokens,
            "reasoning_total": reasoning_tokens,
            "cache_read_total": cache_tokens,
            "cache_hit_ratio_weighted": cache_tokens / input_tokens if input_tokens else None,
            "cache_hit_ratio_per_task": distribution(per_task_cache),
            "input_per_task": distribution(task_input),
            "output_per_task": distribution(task_output),
        },
        "model_calls": {"total": model_calls, "per_task": distribution(calls_per_task)},
        "role_usage": role_usage,
        "outcome_cohorts": cohorts,
        "task_elapsed_seconds": distribution(elapsed),
        "model_request_latency_seconds": distribution(request_latencies),
        "time_to_first_token_ms": {
            **distribution(ttft_ms),
            "status_counts": ttft_status,
            "available": bool(ttft_ms),
            "definition": "provider streaming first token; unavailable must not be replaced by full-request latency",
        },
        "trace_quality": {
            "tasks_with_error_spans": sum(int(row.get("error_span_count") or 0) > 0 for row in rows),
            "total_error_spans": sum(int(row.get("error_span_count") or 0) for row in rows),
            "missing_usage_requests": sum(int(row.get("usage_missing_requests") or 0) for row in rows),
        },
    }
    return metrics, batch


def fmt(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def markdown(metrics: dict[str, Any]) -> str:
    batch = metrics["batch"]
    outcomes = metrics["official_outcomes"]
    costs = metrics["cost_cny"]
    tokens = metrics["tokens"]
    elapsed = metrics["task_elapsed_seconds"]
    calls = metrics["model_calls"]
    latency = metrics["model_request_latency_seconds"]
    ttft = metrics["time_to_first_token_ms"]
    cohorts = metrics["outcome_cohorts"]
    cache_task = tokens["cache_hit_ratio_per_task"]
    interval = outcomes["conservative_wilson_95"] or [None, None]
    lines = [
        f"# AppWorld {batch['split']} Half {batch['half']} 统计", "",
        f"- 完整性：{batch['indexed_tasks']}/{batch['expected_tasks']}，status={batch['status']}",
        f"- 官方结果：{outcomes['PASSED']} 成功 / {outcomes['FAILED']} 失败 / {outcomes['UNSCORED']} 未评分。",
        f"- 保守成功率：{outcomes['conservative_success_rate']:.2%}（分母含未评分）；Wilson 95% CI {interval[0]:.2%}–{interval[1]:.2%}。",
        f"- 已评分成功率：{outcomes['scored_success_rate']:.2%}（仅作诊断，不取代保守口径）。",
        f"- 已知成本：¥{costs['known_total']:.6f}；缺成本记录请求 {costs['missing_request_count']} 次；已知平均每题 ¥{costs['per_task']['mean']:.4f}。",
        f"- Token：input {tokens['input_total']:,}，output {tokens['output_total']:,}，reasoning {tokens['reasoning_total']:,}，cache-read {tokens['cache_read_total']:,}。",
        f"- 加权缓存命中率：{tokens['cache_hit_ratio_weighted']:.2%}；逐题算术平均 {cache_task['mean']:.2%}，P50 {cache_task['p50']:.2%}，P95 {cache_task['p95']:.2%}。",
        f"- 每题耗时：平均 {elapsed['mean']:.1f}s，P50 {elapsed['p50']:.1f}s，P95 {elapsed['p95']:.1f}s，最大 {elapsed['max']:.1f}s。",
        f"- 整批墙钟时间：{fmt(batch['wall_seconds'],1)}s，吞吐 {fmt(batch['throughput_tasks_per_hour'],2)} 题/小时。",
        f"- 模型调用：总计 {calls['total']}，平均每题 {calls['per_task']['mean']:.2f}，P95 {calls['per_task']['p95']:.2f}。",
        f"- 完整请求延迟：平均 {latency['mean']:.2f}s，P50 {latency['p50']:.2f}s，P95 {latency['p95']:.2f}s。",
    ]
    if ttft["available"]:
        lines.append(f"- TTFT：平均 {ttft['mean']:.1f}ms，P50 {ttft['p50']:.1f}ms，P95 {ttft['p95']:.1f}ms。")
    else:
        lines.append(f"- TTFT：不可得。现有 {sum(ttft['status_counts'].values())} 个调用 span 均标记 unavailable，不用完整请求延迟冒充首 Token 时间。")
    lines += [
        "", "## 成功与非成功成本", "",
        "| 口径 | 题数 | 平均费用 | 平均耗时 | 平均模型调用 | 平均总Token |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("PASSED", "FAILED_OR_UNSCORED"):
        row = cohorts[name]
        lines.append(
            f"| {name} | {row['tasks']} | ¥{row['mean_cost_cny']:.4f} | "
            f"{row['mean_elapsed_seconds']:.1f}s | {row['mean_model_calls']:.2f} | "
            f"{row['mean_total_tokens']:.0f} |"
        )
    lines += ["", "费用是现有公开价估算，不是供应商账单；未评分任务保留在总分母。", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--half", choices=("A", "B", "a", "b"), required=True)
    args = parser.parse_args()
    metrics, batch = aggregate(args.batch_id, args.half)
    stem = f"half-{args.half.lower()}-aggregate"
    json_path = batch / f"{stem}.json"
    md_path = batch / f"{stem}.md"
    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(markdown(metrics), encoding="utf-8")
    print(md_path)


if __name__ == "__main__":
    main()
