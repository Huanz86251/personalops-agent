"""Human-driven real AppWorld trial; responses supplied through private files, no provider."""
from __future__ import annotations
import asyncio
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import socket
import sys
import traceback
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True, help="Train task ID; responses are supplied through private files")
TASK_ID = parser.parse_args().task
OUT = ROOT / ".agent/appworld-human" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
OUT.mkdir(parents=True, exist_ok=False)
import path
path.AGENT_DATA_ROOT = OUT / "state"
path.WORKSPACE_ROOT = OUT / "workspace"
path.AGENT_DATA_ROOT.mkdir()
path.WORKSPACE_ROOT.mkdir()
os.environ.update(SKILL_ROUTING_MODE="off", EMAIL_MCP_ENABLED="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
from trace_presentation import run_name
MODE = "code"
assert MODE in {"code", "general"}
PROJECT = run_name("AppWorld Human")
os.environ.update(PHOENIX_TRACING_ENABLED="true", PHOENIX_PROJECT=PROJECT,
    PHOENIX_COLLECTOR_ENDPOINT="http://127.0.0.1:6007/v1/traces")
logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(OUT / "runtime.log", encoding="utf-8")])
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.embeddings import DeterministicFakeEmbedding
from trace_callbacks import CALLBACK
from config import load_settings
import conversation_runtime as cr
from evals.appworld.conversation import AppWorldConversation
from evals.appworld.protocol import DockerWorld
from observability import setup_observability, trace_span, set_span_output
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

class EmptyRetrieval:
    """No previous memory/history exists; any scoring request fails explicitly."""
    def __init__(self, **kwargs):
        self.langchain_embeddings = DeterministicFakeEmbedding(size=1024)
    async def aload(self): pass
    async def aclose(self): pass
    async def arerank(self, **kwargs):
        return []

CALLS = []
def block(messages, title):
    marker = "[" + title + "]\n"
    for message in reversed(messages):
        text = str(message.content)
        if marker in text:
            return json.JSONDecoder().raw_decode(text.rsplit(marker, 1)[1])[0]
    raise AssertionError("Missing runtime block: " + title)

def tool(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name":name,"args":args,"id":call_id,"type":"tool_call"}])

class ManualModel(BaseChatModel):
    role: str
    phase: int = 0
    schema_name: str = ""
    @property
    def _llm_type(self): return "offline-conversation-fixture"
    def bind_tools(self, tools, **kwargs):
        names = [getattr(t, "name", str(t)) for t in tools]
        if self.role in {"code", "general"}: assert {"appworld_discover", "appworld_execute"} <= set(names), names
        if self.role == "code_reviewer": assert {"appworld_discover", "appworld_verify"} <= set(names), names
        save("tools-" + self.role + ".json", names)
        return self
    def with_structured_output(self, schema, **kwargs):
        bound = self.model_copy(update={"schema_name":schema.__name__})
        async def respond(messages, config=None):
            raw = await bound.ainvoke(messages, config=config)
            return {"raw":raw,"parsed":schema.model_validate_json(raw.content),"parsing_error":None}
        return RunnableLambda(respond)
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        import time
        self.phase += 1
        index = len(CALLS)+1
        request = {"index":index,"role":self.role,"phase":self.phase,"schema":self.schema_name,
                   "messages":[m.model_dump(mode="json") for m in messages]}
        CALLS.append(request)
        save("requests.private.json",CALLS)
        save(f"request-{index:03}.private.json",request)
        print(json.dumps({"waiting":index,"role":self.role,"schema":self.schema_name,"out":str(OUT)}),flush=True)
        reply=OUT/f"response-{index:03}.private.json"
        deadline=time.monotonic()+1200
        while not reply.exists():
            if time.monotonic()>deadline:raise TimeoutError("No manual response; stop without provider")
            time.sleep(0.25)
        answer=AIMessage(**json.loads(reply.read_text(encoding="utf-8")))
        return ChatResult(generations=[ChatGeneration(message=answer)])

async def main():
    original_connect = socket.socket.connect
    def local_only(sock,address):
        if isinstance(address,tuple) and address[0] not in {"127.0.0.1","::1","localhost"}:
            raise RuntimeError("Free rehearsal forbids external sockets")
        return original_connect(sock,address)
    socket.socket.connect = local_only
    cr.build_role_model = lambda settings,role: ManualModel(role=role, callbacks=[CALLBACK], metadata={"runtime.model_role":role,"trace.owner":"personalops"})
    cr.RetrievalModelManager = EmptyRetrieval
    settings = load_settings()
    settings = replace(settings, memory_write_gate_enabled=False,memory_extraction_enabled=False,memory_router_enabled=False,
        email_mcp=replace(settings.email_mcp,enabled=False),
        prompt_injection_guard=replace(settings.prompt_injection_guard,enabled=False),
        code_sandbox=replace(settings.code_sandbox,auto_build=False))
    world = DockerWorld()
    ids = world.request("list_tasks", split="train")
    task = world.request("initialize", split="train", task_id=TASK_ID, trial_id=OUT.name)
    save("task.private.json", task)
    save("isolation.json", world.isolation_report())
    runtime = AppWorldConversation(settings, world, task)
    provider = setup_observability()
    assert provider is not None
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    result = {"project":PROJECT,"paid_calls":0,"scope":"Human-authored responses through normal runtime; real AppWorld and Docker; skills off, empty memory, no provider"}
    progress_events = []
    async def progress(event):
        progress_events.append(event)
        save("progress.private.json", progress_events)
    try:
        with trace_span(run_name("AppWorld " + MODE), attributes={"audit.synthetic":True}) as span:
            result["trace_id"] = format(span.get_span_context().trace_id,"032x")
            await runtime.start()
            outcome = await asyncio.wait_for(runtime.run_task(progress), timeout=7200)
            result.update(outcome)
            assert outcome["official_evaluation"] is not None, outcome
            import base64
            (OUT/"execution.private.zip").write_bytes(base64.b64decode(world.request("export")["zip_base64"]))
            save("world-calls.private.json", runtime.task_tools.calls)
            result["status"] = "returned"
            set_span_output(span,result)
    except BaseException as error:
        result.update(status="failed",error=repr(error))
        (OUT/"exception.txt").write_text(traceback.format_exc(),encoding="utf-8")
    finally:
        save("world-calls.private.json", runtime.task_tools.calls)
        world.close()
        await runtime.stop()
        provider.force_flush(timeout_millis=15000)
        spans=exporter.get_finished_spans()
        rows=[{"name":s.name,"span_id":format(s.context.span_id,"016x"),"parent_id":format(s.parent.span_id,"016x") if s.parent else None,
            "status":s.status.status_code.name,"status_description":s.status.description,
            "trace_id":format(s.context.trace_id,"032x"),
            "events":[{"name":e.name,"attributes":dict(e.attributes)} for e in s.events],
            "attributes":dict(s.attributes)} for s in spans]
        save("spans.private.json",rows)
        result.update(span_count=len(rows),model_calls=len(CALLS),unfinished_callbacks=len(CALLBACK.spans))
        save("result.json",result)
        print(json.dumps({"evidence":str(OUT),"status":result["status"],"model_calls":len(CALLS),
                         "spans":len(rows),"paid_calls":0,"human_trial":True},ensure_ascii=False),flush=True)
    if result["status"] != "returned": raise SystemExit(1)

if __name__ == "__main__": asyncio.run(main())
