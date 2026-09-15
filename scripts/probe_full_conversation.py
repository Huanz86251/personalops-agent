"""One isolated, provider-free ConversationRuntime -> real Code/Docker task."""
from __future__ import annotations
import asyncio
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
OUT = ROOT / ".agent/full-conversation" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
OUT.mkdir(parents=True, exist_ok=False)
import path
path.AGENT_DATA_ROOT = OUT / "state"
path.WORKSPACE_ROOT = OUT / "workspace"
path.AGENT_DATA_ROOT.mkdir()
path.WORKSPACE_ROOT.mkdir()
os.environ.update(SKILL_ROUTING_MODE="off", EMAIL_MCP_ENABLED="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
from trace_presentation import run_name
PROJECT = run_name("Full Conversation")
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

class ScriptModel(BaseChatModel):
    role: str
    phase: int = 0
    schema_name: str = ""
    @property
    def _llm_type(self): return "offline-conversation-fixture"
    def bind_tools(self, tools, **kwargs): return self
    def with_structured_output(self, schema, **kwargs):
        bound = self.model_copy(update={"schema_name":schema.__name__})
        async def respond(messages, config=None):
            raw = await bound.ainvoke(messages, config=config)
            return {"raw":raw,"parsed":schema.model_validate_json(raw.content),"parsing_error":None}
        return RunnableLambda(respond)
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.phase += 1
        CALLS.append({"role":self.role,"phase":self.phase,"schema":self.schema_name,
                      "messages":[{"role":m.type,"content":m.content} for m in messages]})
        save("requests.private.json", CALLS)
        if len(CALLS)>30: raise AssertionError("Script request ceiling")
        if self.schema_name == "SupervisorDecision":
            payload = {"action":"PLAN", "plan_objective":"Deliver a checked addition function.",
                "plan_success_criteria":["app.py contains add(a,b) and passes independent checks."],
                "steps":[{"step_id":1,"objective":"Create app.py implementing add(a,b).", "success_criteria":["Addition checks pass."],
                "worker_kind":"CODE","code_task":{"delivery_mode":"ARTIFACT","requirements":[{"requirement_id":"addition","statement":"app.py implements add(a,b) returning a+b.","priority":"MUST"}],
                "validation_expectations":["Check positive, negative and zero operands." ]}}]}
            answer = AIMessage(content=json.dumps(payload))
        elif self.schema_name == "FinalReviewDecision":
            text = "\n".join(str(m.content) for m in messages)
            if "PASSED" not in text and "APPLIED" not in text: raise AssertionError("Final review lacks passed Code evidence")
            answer = AIMessage(content=json.dumps({"action":"FINAL","status":"COMPLETED","final_answer":"app.py 已交付，独立加法检查通过。"}))
        elif self.role == "title": answer = AIMessage(content="加法交付验证")
        elif self.role == "code":
            candidate = block(messages,"当前候选")
            if self.phase == 1:
                answer = tool("execute", {"command":"printf 'def add(a, b):\\n    return a + b\\n' > /workspace/app.py\npython -c \"import sys;sys.path.insert(0,'/workspace');from app import add;assert add(2,3)==5;print('WORKER_CHECK_OK')\""}, "worker-check")
            else:
                if not any("WORKER_CHECK_OK" in str(m.content) for m in messages if m.type=="tool"): raise AssertionError("Worker check absent")
                answer = tool("submit_code_for_review", {"submission":{"candidate":candidate,"summary":"Implemented addition.","requirement_status":{"addition":"MET"},
                    "changed_files":[{"path":"app.py","change_summary":"Add function"}],"proposed_artifact_paths":["app.py"],
                    "self_checks":[{"check":"positive operands","outcome":"PASSED","summary":"2+3=5"}],"evidence_tool_call_ids":["worker-check"]}},"worker-submit")
        elif self.role == "code_reviewer":
            candidate = block(messages,"当前候选")
            if self.phase == 1:
                answer = tool("execute", {"command":"python -c \"import sys;sys.path.insert(0,'/workspace');from app import add;assert add(2,3)==5;assert add(-2,3)==1;assert add(0,0)==0;print('REVIEW_CHECK_OK:3')\""},"review-check")
            elif self.phase == 2:
                if not any("REVIEW_CHECK_OK:3" in str(m.content) for m in messages if m.type=="tool"): raise AssertionError("Independent checks absent")
                answer = tool("publish_reviewed_candidate",{"approved_artifact_paths":["app.py"]},"review-publish")
            else:
                receipt = block(messages,"发布回执")
                answer = tool("submit_code_review", {"report":{"candidate":candidate,"verdict":"PASSED","summary":"Addition independently checked.",
                    "verification_summary":"3 assertions passed in reviewer sandbox", "verified_requirement_ids":["addition"],
                    "check_results":[{"check_id":"addition","description":"positive/negative/zero","status":"PASSED","summary":"3 assertions passed"}],
                    "evidence_refs":["review-check"], "approved_artifact_paths":["app.py"],"published_artifact_paths":["app.py"],
                    "delivery_location":receipt["target_root"],"publication_id":receipt["publication_id"],"applied_revision":receipt["applied_revision"]}},"review-submit")
        else: raise AssertionError(f"Unscripted role/schema: {self.role}/{self.schema_name}")
        return ChatResult(generations=[ChatGeneration(message=answer)])

async def main():
    original_connect = socket.socket.connect
    def local_only(sock,address):
        if isinstance(address,tuple) and address[0] not in {"127.0.0.1","::1","localhost"}:
            raise RuntimeError("Free rehearsal forbids external sockets")
        return original_connect(sock,address)
    socket.socket.connect = local_only
    cr.build_role_model = lambda settings,role: ScriptModel(role=role, callbacks=[CALLBACK], metadata={"runtime.model_role":role,"trace.owner":"personalops"})
    cr.RetrievalModelManager = EmptyRetrieval
    settings = load_settings()
    settings = replace(settings, memory_write_gate_enabled=False,memory_extraction_enabled=False,memory_router_enabled=False,
        email_mcp=replace(settings.email_mcp,enabled=False),
        prompt_injection_guard=replace(settings.prompt_injection_guard,enabled=False),
        code_sandbox=replace(settings.code_sandbox,auto_build=False))
    runtime = cr.ConversationRuntime(settings, [])
    provider = setup_observability()
    assert provider is not None
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    result = {"project":PROJECT,"paid_calls":0,"scope":"Real ConversationRuntime/Code graph/Docker; scripted model, empty memory fixture, skills off"}
    progress_events = []
    async def progress(event):
        progress_events.append(event)
        save("progress.private.json", progress_events)
    try:
        with trace_span(run_name("Addition Task"), attributes={"audit.synthetic":True}) as span:
            result["trace_id"] = format(span.get_span_context().trace_id,"032x")
            await runtime.start()
            result["answer"] = await asyncio.wait_for(runtime.ask(user_text="生成 app.py，add(a,b) 返回两数之和；独立检查正数、负数、零，然后交付文件。",channel="offline-check",external_chat_id="isolated",event_id="offline-"+uuid4().hex,progress_callback=progress),timeout=240)
            if "独立加法检查通过" not in result["answer"]:
                raise AssertionError("Conversation returned without the expected checked delivery")
            result["status"] = "returned"
            set_span_output(span,result)
    except BaseException as error:
        result.update(status="failed",error=repr(error))
        (OUT/"exception.txt").write_text(traceback.format_exc(),encoding="utf-8")
    finally:
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
        print(json.dumps({"evidence":str(OUT),**result},ensure_ascii=False),flush=True)
    if result["status"] != "returned": raise SystemExit(1)

if __name__ == "__main__": asyncio.run(main())
