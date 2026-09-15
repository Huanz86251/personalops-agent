"""Free boundary probe: graph-inherited callbacks plus local audit budget."""
import asyncio
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PHOENIX_TRACING_ENABLED"] = "false"
import langchain_openai
import httpx
from tests.test_model_roles import defaults
from model_roles import load_role_models
from agent import build_role_model
from trace_callbacks import CALLBACK
from trace_presentation import attach_audit_callbacks
from evals.appworld.adapter import UsageMeter
from observability import trace_span
from openinference.instrumentation import OITracer, TraceConfig
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

OUT = ROOT / ".agent/inherited-callbacks" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
OUT.mkdir(parents=True, exist_ok=False)
provider = TracerProvider()
exporter = InMemorySpanExporter()
provider.add_span_processor(SimpleSpanProcessor(exporter))
tracer = OITracer(provider.get_tracer("offline"), config=TraceConfig())
settings = defaults()
with patch.dict(os.environ, {"DASHSCOPE_API_KEY":"offline", "OPENAI_API_KEY":"offline", "DEEPSEEK_API_KEY":"offline"}):
    settings.role_models = load_role_models(settings)

results = []
async def probe(mode):
    model = build_role_model(settings, "code")
    meter = UsageMeter(max_calls=0)
    attach_audit_callbacks(model, meter)
    requests = []
    def handler(request):
        requests.append(str(request.url))
        return httpx.Response(500)
    before = set(CALLBACK.spans)
    model.root_client._client.close()
    await model.root_async_client._client.aclose()
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as async_client:
            model.root_client._client = client
            model.root_async_client._client = async_client
            with trace_span("Graph callback boundary"):
                error = None
                try:
                    if mode == "async":
                        await model.ainvoke("synthetic", config={"callbacks":[CALLBACK]})
                    else:
                        model.invoke("synthetic", config={"callbacks":[CALLBACK]})
                except Exception as caught:
                    error = type(caught).__name__
    leaked = set(CALLBACK.spans) - before
    result = {"mode":mode, "provider_requests":len(requests), "budget_records":len(meter.started),
              "exception":error, "unfinished_trace_nodes":len(leaked)}
    results.append(result)
    # Close only this probe's outstanding records after measuring, before exit.
    for key in leaked:
        CALLBACK.on_llm_error(RuntimeError("Probe cleanup: budget rejected before model invocation"), run_id=key)

with patch("observability._TRACER", tracer):
    asyncio.run(probe("sync"))
    asyncio.run(probe("async"))
(OUT / "result.json").write_text(json.dumps({"paid_calls":0,"results":results},indent=2),encoding="utf-8")
provider.shutdown()
print(json.dumps({"evidence":str(OUT),"paid_calls":0,"results":results}))
