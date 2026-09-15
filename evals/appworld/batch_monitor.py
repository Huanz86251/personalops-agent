"""Deterministic private index for AppWorld runs and Phoenix traces.

The monitor never calls a model and never copies task instructions, tool outputs,
credentials, or model messages.  It can be called after each run or rebuilt from
existing private run directories.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from contextlib import closing
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = ROOT / ".agent" / "appworld-conversations"
DEFAULT_OUTPUT_ROOT = ROOT / ".agent" / "appworld-monitor"
DEFAULT_PHOENIX_DB = ROOT / ".agent" / "phoenix" / "phoenix.db"


def _load(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, (int, float)) else None


def _usage_rows(value: Any) -> list[dict[str, Any]]:
    rows = list(value.values()) if isinstance(value, dict) else value
    return [row for row in (rows or []) if isinstance(row, dict)]


def _span_seconds(span: dict[str, Any]) -> float | None:
    start, end = _number(span.get("start_time")), _number(span.get("end_time"))
    if start is None or end is None or end < start:
        return None
    # OpenTelemetry exports integer timestamps in nanoseconds. Float timestamps
    # are treated as seconds so synthetic/imported span formats remain usable.
    divisor = 1_000_000_000 if isinstance(start, int) and isinstance(end, int) else 1
    return (end - start) / divisor


def _official_success(result: dict[str, Any]) -> bool | None:
    value = result.get("official_evaluation")
    if isinstance(value, dict) and isinstance(value.get("success"), bool):
        return value["success"]
    value = result.get("official_task_success")
    return value if isinstance(value, bool) else None


def _phoenix_url(trace_id: str | None, database: Path) -> str | None:
    if not trace_id or not database.is_file():
        return None
    try:
        # sqlite.Connection context management does not close the handle.
        # closing() gives this short read a deterministic lifetime on Windows.
        with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            row = db.execute(
                "SELECT p.id FROM traces t JOIN projects p ON p.id=t.project_rowid "
                "WHERE t.trace_id=? ORDER BY t.id DESC LIMIT 1",
                (trace_id,),
            ).fetchone()
    except (sqlite3.Error, OSError):
        return None
    if row is None:
        return None
    project_ref = base64.b64encode(f"Project:{row[0]}".encode()).decode()
    return f"http://127.0.0.1:6007/projects/{project_ref}/spans/{trace_id}?timeRangeKey=7d"


def summarize_trial(trial_directory: Path, *, phoenix_db: Path = DEFAULT_PHOENIX_DB) -> dict[str, Any]:
    trial_directory = trial_directory.resolve()
    manifest = _load(trial_directory / "manifest.json", {})
    result = _load(trial_directory / "result.private.json", {})
    usage = _usage_rows(_load(trial_directory / "usage.private.json", []))
    costs = _load(trial_directory / "cost-summary.json", {})
    spans = _load(trial_directory / "spans.private.json", [])
    spans = [span for span in spans if isinstance(span, dict)]

    def total(field: str) -> tuple[int, int]:
        values = [_number(row.get(field)) for row in usage]
        return int(sum(value for value in values if value is not None)), sum(value is None for value in values)

    input_tokens, missing_input = total("input_tokens")
    output_tokens, missing_output = total("output_tokens")
    cache_tokens, missing_cache = total("cache_read_input_tokens")
    reasoning_tokens, missing_reasoning = total("reasoning_output_tokens")
    role_usage: dict[str, dict[str, int]] = defaultdict(lambda: {
        "model_calls": 0, "input_tokens": 0, "output_tokens": 0,
        "cache_read_tokens": 0, "reasoning_tokens": 0,
    })
    for row in usage:
        role = str(row.get("model_role") or "unknown")
        target = role_usage[role]
        target["model_calls"] += 1
        target["input_tokens"] += int(_number(row.get("input_tokens")) or 0)
        target["output_tokens"] += int(_number(row.get("output_tokens")) or 0)
        target["cache_read_tokens"] += int(_number(row.get("cache_read_input_tokens")) or 0)
        target["reasoning_tokens"] += int(_number(row.get("reasoning_output_tokens")) or 0)

    trace_counts = Counter(str(span.get("trace_id")) for span in spans if span.get("trace_id"))
    trace_id = trace_counts.most_common(1)[0][0] if trace_counts else None
    seconds = [_span_seconds(span) for span in spans]
    known_seconds = [value for value in seconds if value is not None]
    elapsed_seconds = max(known_seconds, default=None)

    component_candidates = []
    for span, duration in zip(spans, seconds):
        name = str(span.get("name") or "")
        lowered = name.lower()
        is_wrapper = (
            name.startswith("AppWorld / ")
            or name.startswith("Conversation")
            or name == "🎛️ Scheduler"
            or name.startswith("Scheduler / ")
        )
        if duration is None or is_wrapper or not any(word in lowered for word in (
            "step", "worker", "reviewer", "general", "code", "web", "llm"
        )):
            continue
        component_candidates.append((duration, name))
    longest = max(component_candidates, default=(None, None))
    error_names = [str(span.get("name") or "unknown") for span in spans if span.get("status") == "ERROR"]

    total_cost = costs.get("total", {}) if isinstance(costs, dict) else {}
    known_cost = _number(total_cost.get("known_total_cny")) if isinstance(total_cost, dict) else None
    missing_cost = int(_number(total_cost.get("missing_requests")) or 0) if isinstance(total_cost, dict) else len(usage)
    official = _official_success(result)
    if result.get("status") == "error":
        outcome = "ERROR"
    elif official is True:
        outcome = "PASSED"
    elif official is False:
        outcome = "FAILED"
    else:
        outcome = "UNSCORED"

    reasons = []
    if result.get("status") == "error":
        reasons.append("runtime_error")
    if official is False:
        reasons.append("official_evaluation_failed")
    if official is None:
        reasons.append(str(result.get("evaluation_skipped") or "official_evaluation_missing"))
    if error_names:
        reasons.append("error_spans=" + ", ".join(error_names[:3]))
    if result.get("stop_reason"):
        reasons.append("stop_reason=" + str(result["stop_reason"])[:240])

    failure_codes = []
    if result.get("status") == "error":
        failure_codes.append("runtime_error")
    if official is False:
        failure_codes.append("official_failed")
    if official is None:
        failure_codes.append("unscored")
    if error_names:
        failure_codes.append("error_span")

    token_by_role = sorted(
        role_usage.items(), key=lambda item: item[1]["input_tokens"] + item[1]["output_tokens"], reverse=True
    )
    return {
        "trial_id": trial_directory.name,
        "task_id": manifest.get("task_id") or result.get("task_id"),
        "split": manifest.get("split"),
        "status": result.get("status"),
        "outcome": outcome,
        "official_task_success": official,
        "failure_kind": ",".join(failure_codes) or "none",
        "failure_summary": "; ".join(reasons),
        "model_calls": len(usage),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cache_hit_ratio": (cache_tokens / input_tokens) if input_tokens and not missing_cache else None,
        "usage_missing_requests": max(missing_input, missing_output, missing_cache, missing_reasoning),
        "known_cost_cny": known_cost,
        "cost_missing_requests": missing_cost,
        "elapsed_seconds": round(elapsed_seconds, 3) if elapsed_seconds is not None else None,
        "span_count": len(spans),
        "error_span_count": len(error_names),
        "longest_component": longest[1],
        "longest_component_seconds": round(longest[0], 3) if longest[0] is not None else None,
        "top_token_role": token_by_role[0][0] if token_by_role else None,
        "top_token_role_tokens": (
            token_by_role[0][1]["input_tokens"] + token_by_role[0][1]["output_tokens"]
            if token_by_role else None
        ),
        "role_usage": dict(role_usage),
        "trace_id": trace_id,
        "phoenix_url": _phoenix_url(trace_id, phoenix_db),
        "trial_directory": str(trial_directory),
        "indexed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


CSV_FIELDS = [
    "trial_id", "task_id", "split", "outcome", "official_task_success", "status",
    "model_calls", "input_tokens", "output_tokens", "cache_read_tokens", "cache_hit_ratio",
    "reasoning_tokens", "known_cost_cny", "cost_missing_requests", "elapsed_seconds",
    "span_count", "error_span_count", "longest_component", "longest_component_seconds",
    "top_token_role", "top_token_role_tokens", "trace_id", "phoenix_url", "failure_kind", "failure_summary",
    "trial_directory",
]


def _markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| 任务 | 运行 | 结果 | 原因 | 模型调用 | 输入/输出Token | 缓存命中 | 费用(元) | 秒 | 最长组件 | Trace |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        ratio = row.get("cache_hit_ratio")
        ratio_text = "未知" if ratio is None else f"{ratio:.1%}"
        cost = row.get("known_cost_cny")
        cost_text = "未知" if cost is None else f"{cost:.6f}"
        trace = f"[打开]({row['phoenix_url']})" if row.get("phoenix_url") else str(row.get("trace_id") or "未知")
        component = str(row.get("longest_component") or "未知")
        if row.get("longest_component_seconds") is not None:
            component += f" ({row['longest_component_seconds']:.1f}s)"
        lines.append(
            f"| {row.get('task_id') or '未知'} | {row['trial_id']} | {row['outcome']} | "
            f"{row.get('failure_kind') or 'none'} | {row['model_calls']} | "
            f"{row['input_tokens']}/{row['output_tokens']} | {ratio_text} | {cost_text} | "
            f"{row.get('elapsed_seconds') if row.get('elapsed_seconds') is not None else '未知'} | {component} | {trace} |"
        )
    return lines


def write_batch(rows: Iterable[dict[str, Any]], batch_id: str, *, output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    batch = output_root / batch_id
    runs = batch / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    by_trial = {str(row["trial_id"]): row for row in rows}
    for row in by_trial.values():
        path = runs / f"{row['trial_id']}.json"
        path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    all_rows = [_load(path, {}) for path in sorted(runs.glob("*.json"))]
    all_rows = [row for row in all_rows if isinstance(row, dict) and row.get("trial_id")]

    (batch / "index.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (batch / "index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    failures = [row for row in all_rows if row["outcome"] != "PASSED"]
    by_cost = sorted(all_rows, key=lambda row: row.get("known_cost_cny") or -1, reverse=True)[:10]
    by_tokens = sorted(all_rows, key=lambda row: row.get("input_tokens", 0) + row.get("output_tokens", 0), reverse=True)[:10]
    by_time = sorted(all_rows, key=lambda row: row.get("elapsed_seconds") or -1, reverse=True)[:10]
    roles: dict[str, dict[str, int]] = defaultdict(lambda: {"model_calls": 0, "input_tokens": 0, "output_tokens": 0})
    for row in all_rows:
        for role, usage in row.get("role_usage", {}).items():
            roles[role]["model_calls"] += int(usage.get("model_calls", 0))
            roles[role]["input_tokens"] += int(usage.get("input_tokens", 0))
            roles[role]["output_tokens"] += int(usage.get("output_tokens", 0))

    report = [
        f"# AppWorld批次监控：{batch_id}", "",
        f"共{len(all_rows)}条；通过{len(all_rows)-len(failures)}条；失败或未评分{len(failures)}条。",
        "本报告只含索引和统计，不含任务原文、模型消息、工具返回或凭据。", "",
        "## 失败与未评分", "",
    ]
    report += _markdown_table(failures) if failures else ["无。"]
    for title, ranked in (("费用最高", by_cost), ("Token最高", by_tokens), ("耗时最长", by_time)):
        report += ["", f"## {title}", "", *_markdown_table(ranked)]
    report += ["", "## 按角色Token热点", "", "| 角色 | 模型调用 | 输入Token | 输出Token |", "| --- | ---: | ---: | ---: |"]
    for role, usage in sorted(roles.items(), key=lambda item: item[1]["input_tokens"] + item[1]["output_tokens"], reverse=True):
        report.append(f"| {role} | {usage['model_calls']} | {usage['input_tokens']} | {usage['output_tokens']} |")
    report += ["", "## 字段边界", "", "- PASSED/FAILED仅采用官方评测；UNSCORED不能算成功。", "- 费用是现有公开价估算，不是供应商账单；缺失调用保持未知。", "- 最长组件来自本地Span持续时间，父子范围可能重叠，不能相加。"]
    (batch / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return batch


def record_trial(trial_directory: Path, batch_id: str, *, output_root: Path = DEFAULT_OUTPUT_ROOT, phoenix_db: Path = DEFAULT_PHOENIX_DB) -> Path:
    row = summarize_trial(trial_directory, phoenix_db=phoenix_db)
    return write_batch([row], batch_id, output_root=output_root)


def scan_runs(runs_root: Path, *, latest: int | None = None) -> list[Path]:
    paths = [path for path in runs_root.iterdir() if path.is_dir() and (path / "manifest.json").is_file()]
    paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return paths[:latest] if latest else paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial_directories", type=Path, nargs="*")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--phoenix-db", type=Path, default=DEFAULT_PHOENIX_DB)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--latest", type=int)
    args = parser.parse_args()
    paths = args.trial_directories or scan_runs(args.runs_root, latest=args.latest)
    rows = [summarize_trial(path, phoenix_db=args.phoenix_db) for path in paths]
    output = write_batch(rows, args.batch_id, output_root=args.output_root)
    print(output)


if __name__ == "__main__":
    main()
