"""Read-only analysis of one private audit run and its persisted Phoenix spans."""
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import sqlite3
import sys

folder = Path(sys.argv[1]).resolve()
usage = json.loads((folder / "usage.json").read_text(encoding="utf-8"))
result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
events = [json.loads(line) for line in (folder / "events.private.jsonl").read_text(encoding="utf-8").splitlines()]
starts = {e["id"]: e for e in events if e["event"] == "model_start"}
by_index = {e["metadata"]["call_index"]: e for e in starts.values()}

def classify(event):
    messages = event.get("messages", [])
    system = "\n".join(str(m.get("content", "")) for m in messages if m.get("type", m.get("role")) in {"system"})
    for m in messages:
        try:
            value = json.loads(m.get("content", ""))
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict) and "available_skills" in value:
            return value.get("role", "unknown") + ".skills"
    for text, role in [("对照 code_task 独立检查", "code.review"), ("按当前 code_task 实现", "code.execute"), ("完成当前网页任务", "web.execute"), ("对照 task_contract 审核", "step.report")]:
        if text in system:
            return role
    metadata = event.get("metadata", {})
    return (metadata.get("runtime_metadata") or {}).get("langgraph_node") or metadata.get("model_role", "unknown")

stages = defaultdict(lambda: dict(calls=0, input_tokens=0, output_tokens=0, known_reasoning_tokens=0, missing_reasoning_usage=0, estimated_cny=0.0))
calls = []
ids = []
for record in usage["records"]:
    stage = classify(by_index.get(record["call_index"], {}))
    started = datetime.fromisoformat(record["started_at_utc"]).astimezone(timezone(timedelta(hours=8)))
    peak = started.weekday() < 5 and (9 <= started.hour < 12 or 14 <= started.hour < 18)
    pro = "pro" in record.get("model_name", "")
    input_rate, cache_rate, output_rate = (4.5, .15, 13.5) if pro else (1.5, .05, 4.5)
    multiplier = 2 if peak else 1
    inp, out = record.get("input_tokens"), record.get("output_tokens")
    cache = record.get("cache_read_input_tokens")
    cost = None if inp is None or out is None or cache is None else ((inp-cache)*input_rate + cache*cache_rate + out*output_rate)*multiplier/1e6
    item = {**record, "stage": stage, "estimated_cny": cost}
    calls.append(item)
    if record.get("provider_response_id"):
        ids.append(record["provider_response_id"])
    row = stages[stage]
    row["calls"] += 1
    row["input_tokens"] += inp or 0
    row["output_tokens"] += out or 0
    row["known_reasoning_tokens"] += record.get("reasoning_output_tokens") or 0
    row["missing_reasoning_usage"] += record.get("reasoning_output_tokens") is None
    row["estimated_cny"] += cost or 0

db_path = Path(__file__).resolve().parents[1] / ".agent/phoenix/phoenix.db"
with sqlite3.connect(db_path.resolve().as_uri()+"?mode=ro", uri=True) as db:
    db.row_factory = sqlite3.Row
    spans = [dict(row) for row in db.execute("SELECT s.* FROM spans s JOIN traces t ON t.id=s.trace_rowid WHERE t.trace_id=? ORDER BY s.start_time", (result["trace_id"],))]
for span in spans:
    if isinstance(span.get("attributes"), str):
        span["attributes"] = json.loads(span["attributes"])
(folder / "phoenix-spans.private.json").write_text(json.dumps(spans, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
names = Counter(s["name"] for s in spans)
summary = dict(trace_id=result["trace_id"], status=result.get("status"), usage_complete=usage["usage_complete"], calls_started=usage["model_calls_started"], calls_recorded=len(calls), known_input_tokens=usage["known_input_tokens"], known_output_tokens=usage["known_output_tokens"], estimated_known_cny=sum(c["estimated_cny"] or 0 for c in calls), duplicate_provider_response_ids=[k for k,v in Counter(ids).items() if v>1], span_count=len(spans), llm_span_count=sum(s["name"].startswith("LLM / ") for s in spans), tool_span_count=sum(s["name"].startswith("TOOL / ") for s in spans), stage_totals=dict(stages), span_names=dict(names), calls=calls, pricing_source="https://api-docs.deepseek.com/zh-cn/quick_start/pricing/", pricing_note="Public list-price estimate, China weekday peak schedule; not provider invoice. Unknown cache/usage remains unknown.")
(folder / "analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k:v for k,v in summary.items() if k not in {"calls", "span_names"}}, ensure_ascii=False, indent=2))
