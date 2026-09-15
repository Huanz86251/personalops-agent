"""Free synthetic callback rehearsal; exports inspectable usage JSON and HTML."""
import asyncio
import html
import json
import os
from pathlib import Path
import socket
import sys
from datetime import datetime, timezone
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trace_presentation import run_name, role_badge
os.environ.update(PHOENIX_TRACING_ENABLED="true", PHOENIX_PROJECT=run_name("Usage Check"),
                  PHOENIX_COLLECTOR_ENDPOINT="http://127.0.0.1:6007/v1/traces")
original_connect = socket.socket.connect
def local_only(sock, address):
    if isinstance(address, tuple) and address[0] not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError("Synthetic usage probe forbids external connections")
    return original_connect(sock, address)
socket.socket.connect = local_only
from observability import setup_observability, trace_span
from runtime_tracing import operation
from trace_callbacks import CALLBACK
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import LLMResult, ChatGeneration
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
provider = setup_observability()
assert provider is not None
exporter = InMemorySpanExporter()
provider.add_span_processor(SimpleSpanProcessor(exporter))

def request(role, incoming=None, outgoing=None, cached=None):
    key = uuid4()
    CALLBACK.on_chat_model_start({}, [[HumanMessage(content="Synthetic usage fixture; no provider request")]],
        run_id=key, metadata={"runtime.model_role": role})
    usage = None if incoming is None else {"input_tokens": incoming, "output_tokens": outgoing,
        "total_tokens": incoming + outgoing, "input_token_details": {"cache_read": cached}}
    response = LLMResult(generations=[[ChatGeneration(message=AIMessage(content="Synthetic response", usage_metadata=usage))]])
    CALLBACK.on_llm_end(response, run_id=key)

@operation(role_badge("Code Reviewer"), fields=("state",))
async def review(state):
    await asyncio.sleep(.001)
    request("code_reviewer", 100, 20, 60)

async def main():
    with trace_span("🎛️ Scheduler", attributes={"audit.synthetic": True, "audit.paid_calls": 0}):
        with trace_span("Scheduler / Plan"):
            request("scheduler", 10, 2, 0)
        with trace_span(role_badge("Code Agent / Step 1")):
            await asyncio.gather(*(review({"step_id": 1, "attempt_id": "fixture", "candidate_revision": n}) for n in (1, 2)))
        with trace_span("Scheduler / Final Review"):
            request("final_reviewer")

asyncio.run(main())
provider.force_flush(timeout_millis=15000)
root = next(s for s in exporter.get_finished_spans() if s.name == "🎛️ Scheduler")
report = json.loads(root.attributes["usage.summary_json"])
assert report["subtree"]["total_tokens"] == 252
assert report["subtree"]["total_tokens_missing_requests"] == 1
report.update(synthetic=True, paid_calls=0, trace_id=format(root.context.trace_id, "032x"),
    timing=[{"name":s.name, **{k:v for k,v in s.attributes.items() if k.startswith("timing.")}} for s in exporter.get_finished_spans()])
out = ROOT / ".agent/usage-accounting" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
out.mkdir(parents=True, exist_ok=False)
(out / "usage.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
def table(values):
    keys = ("requests", "input_tokens", "output_tokens", "total_tokens", "cache_read_tokens", "total_tokens_missing_requests")
    return '<table><tr><th>范围</th>' + ''.join('<th>'+k+'</th>' for k in keys) + '</tr>' + ''.join(
        '<tr><td>'+html.escape(label)+'</td>'+''.join('<td>'+('未知' if row[k] is None else str(row[k]))+'</td>' for k in keys)+'</tr>' for label,row in values.items())+'</table>'
page = '<!doctype html><meta charset="utf-8"><title>Token 用量验证</title><style>body{font:16px system-ui;margin:40px;color:#172033}table{border-collapse:collapse;margin:20px 0}td,th{padding:12px;border:1px solid #ccd4df}pre{white-space:pre-wrap}</style><h1>Token 用量汇总 · 免费合成演练</h1><p>这些数字来自测试夹具，不是真实模型账单。4 次模拟请求：已知 252 token，另 1 次未知。缓存已包含在输入中。</p>'
page += table({k:report[k] for k in ('subtree','scheduler_control','code_review')})
page += '<h2>Reviewer 轮次：步骤 / 尝试 / 修订 / 返修轮</h2>'+table(report['by']['code_review_round'])
page += '<details><summary>逐请求增量、累计和完整 JSON</summary><pre>'+html.escape(json.dumps(report,ensure_ascii=False,indent=2))+'</pre></details>'
(out / "usage.html").write_text(page, encoding="utf-8")
print(json.dumps({"evidence": str(out), "trace_id": report["trace_id"], "paid_calls": 0}))
