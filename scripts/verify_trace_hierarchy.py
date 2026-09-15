"""Provider-free semantic trace verification, with optional existing Docker probe."""
import asyncio
import contextvars
import importlib
import json
import os
from pathlib import Path
import socket
import sys
import unittest
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
names = ["test_planning_web_to_code_pipeline.WebToCodePipelineTests.test_parallel_web_report_feeds_reviewed_code_step",
         "test_planning_failure_stop.FailureStopTests.test_two_invalid_plans_stop_without_other_model_or_worker",
         "test_code_runtime.CodeRuntimeTests.test_continue_reuses_frozen_pair_and_applies_revision",
         "test_code_runtime.CodeRuntimeTests.test_restart_supersedes_old_attempt_and_starts_fresh_pair",
         "test_code_runtime.CodeRuntimeTests.test_stop_archives_and_releases_frozen_pair",
         "test_skill_preparation.RolePreparationTests.test_multiple_policy_bodies_reach_only_their_role_and_survive_resume"]
if "--docker-only" in sys.argv:
    names = []
if "--model-only" in sys.argv:
    # Initialize the installed SSL/provider runtime before fixture tests clear
    # environment variables to verify per-role configuration isolation.
    import langchain_openai
    names = ["test_model_roles.RoleConfigurationTests.test_qwen_role_defaults_and_actual_request_payloads",
             "test_model_roles.RoleConfigurationTests.test_actual_mock_http_requests_keep_keys_and_endpoints_separate"]
if "--agent-only" in sys.argv:
    names = ["test_code_agents.CodeAgentFactoryTests.test_worker_submits_a_real_structured_manifest",
             "test_code_agents.CodeAgentFactoryTests.test_reviewer_requests_bounded_repair_and_stops",
             "test_code_agents.CodeAgentFactoryTests.test_same_worker_consumes_scheduler_continue_directive"]
cases = [unittest.defaultTestLoader.loadTestsFromName(name) for name in names]
from observability import setup_observability, trace_span, set_span_output
from trace_presentation import run_name
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from trace_callbacks import CALLBACK
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import LLMResult, ChatGeneration
from uuid import uuid4

OUT = ROOT / ".agent/trace-hierarchy" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
OUT.mkdir(parents=True, exist_ok=False)
os.environ.update(PHOENIX_TRACING_ENABLED="true", PHOENIX_TRACE_PROFILE="curated",
                  PHOENIX_PROJECT=run_name("Route Check"),
                  PHOENIX_COLLECTOR_ENDPOINT="http://127.0.0.1:6007/v1/traces")
provider = setup_observability()
assert provider is not None
exporter = InMemorySpanExporter()
provider.add_span_processor(SimpleSpanProcessor(exporter))

original_connect = socket.socket.connect
def local_only(sock, address):
    if isinstance(address, tuple) and address[0] not in {"127.0.0.1", "::1", "localhost"}:
        raise RuntimeError("Offline trace rehearsal blocked external socket")
    return original_connect(sock, address)
socket.socket.connect = local_only

class Result(unittest.TextTestResult):
    def startTest(self, test):
        if hasattr(test, "_asyncioTestContext"):
            test._asyncioTestContext = contextvars.copy_context()
        print(test.id(), flush=True)
        super().startTest(test)

results = []
with (OUT / "tests.log").open("w", encoding="utf-8") as handle:
    for name, case in zip(names, cases):
        with trace_span("Rehearsal / " + name.split(".")[-1], attributes={"audit.synthetic": True}) as span:
            result = unittest.TextTestRunner(stream=handle, verbosity=2, resultclass=Result).run(case)
            item = {"case": name, "passed": result.wasSuccessful(), "trace_id": format(span.get_span_context().trace_id, "032x")}
            set_span_output(span, item)
            results.append(item)
            handle.flush()

async def memory_probe():
    from memory import MemoryService
    class Model:
        responses = iter([
            {"candidates": [{"candidate_id": "synthetic-user", "frames": [{"frame_id": "name", "frame_type": "profile"}]}]},
            {"records": [{"candidate_id": "synthetic-user", "frame_id": "name", "record_type": "profile", "summary": "用户自称小林",
                          "importance": "medium", "confidence": "high", "field": "name", "value": "小林"}]}])
        async def ainvoke(self, messages):
            key = uuid4()
            CALLBACK.on_chat_model_start({}, [[HumanMessage(content=str(messages))]], run_id=key, metadata={"runtime.model_role": "extraction"})
            response = AIMessage(content=json.dumps(next(self.responses), ensure_ascii=False))
            CALLBACK.on_llm_end(LLMResult(generations=[[ChatGeneration(message=response)]]), run_id=key)
            return response
    service = object.__new__(MemoryService)
    service.timezone_name = "Asia/Shanghai"
    service.model = Model()
    plan, records = await service.extract_progressive_batch([("synthetic-user", {"raw_user_text": "我叫小林"})])
    assert len(records) == 1

with trace_span("Rehearsal / Memory Extractor", attributes={"audit.synthetic": True}):
    asyncio.run(memory_probe())
if "--docker" in sys.argv:
    try:
        with trace_span("Rehearsal / Real Docker Repair", attributes={"audit.synthetic": True, "audit.real_docker": True}):
            from scripts.probe_code_skill_handoff import main
            main()
        results.append({"case": "real_docker", "passed": True})
    except Exception as error:
        results.append({"case": "real_docker", "passed": False, "error": str(error)})
provider.force_flush(timeout_millis=15000)
spans = exporter.get_finished_spans()
rows = [{"name": s.name, "trace_id": format(s.context.trace_id, "032x"), "span_id": format(s.context.span_id, "016x"),
         "parent_id": format(s.parent.span_id, "016x") if s.parent else None,
         "start": s.start_time, "end": s.end_time, "status": s.status.status_code.name,
         "attributes": dict(s.attributes)} for s in spans]
(OUT / "spans.private.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
ids = {s["span_id"] for s in rows}
assert not [s for s in rows if s["parent_id"] and s["parent_id"] not in ids]
summary = {"paid_calls": 0, "span_count": len(rows), "root_count": sum(s["parent_id"] is None for s in rows),
           "missing_parents": 0, "results": results, "evidence": str(OUT)}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False), flush=True)
assert all(r["passed"] for r in results), results
