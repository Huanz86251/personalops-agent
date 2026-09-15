"""Free routing rehearsal. Scripted decisions are not model-quality evidence."""
from __future__ import annotations

import asyncio
import contextvars
import importlib
import inspect
import json
import os
from pathlib import Path
import socket
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
OUT = ROOT / ".agent/chain-audit/20260906" / (sys.argv[1] if len(sys.argv) > 1 else "scripted-routes")
OUT.mkdir(parents=True, exist_ok=True)
if (OUT / "STARTED").exists():
    raise SystemExit("Evidence exists; choose a new output directory before rerunning")
(OUT / "STARTED").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))

MODULES = ["test_planning_web_to_code_pipeline", "test_planning_code_execution",
           "test_code_agents", "test_code_review_models", "test_code_runtime",
           "test_skill_preparation", "test_web_runtime"]
modules = [importlib.import_module(name) for name in MODULES]
from trace_presentation import run_name
project_name = run_name("Scripted Routes")
os.environ.update(PHOENIX_TRACING_ENABLED="true", PHOENIX_TRACE_PROFILE="curated",
                  PHOENIX_COLLECTOR_ENDPOINT="http://127.0.0.1:6007/v1/traces",
                  PHOENIX_PROJECT_NAME=project_name, PHOENIX_PROJECT=project_name)
from observability import setup_observability, trace_span, set_span_output
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.utils import count_tokens_approximately

provider = setup_observability()
records = []
current_case = ""
original_connect = socket.socket.connect
def local_connect(sock, address):
    if isinstance(address, tuple) and address[0] not in {"127.0.0.1", "::1", "localhost"}:
        raise RuntimeError(f"Audit blocked outbound connection: {address[0]}")
    return original_connect(sock, address)
socket.socket.connect = local_connect

def safe(value):
    if hasattr(value, "model_dump"):
        return safe(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

def observe(role, messages, result, schema=None):
    body = safe(messages)
    encoded = json.dumps(body, ensure_ascii=False)
    try:
        tokens = count_tokens_approximately(messages)
    except Exception:
        tokens = None
    entry = {"case": current_case, "role": role, "input_chars": len(encoded),
             "approx_message_tokens": tokens, "schema_chars": len(json.dumps(schema, ensure_ascii=False)) if schema else 0,
             "messages": body, "scripted_output": safe(result), "actual_usage": None}
    records.append(entry)
    with (OUT / "requests.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

def instrument_structured(cls):
    original = cls.with_structured_output
    def wrapped(self, schema, **kwargs):
        runnable = original(self, schema, **kwargs)
        async def run(messages, config=None):
            with trace_span("AUDIT scripted " + schema.__name__, input_value=messages,
                            attributes={"audit.synthetic": True, "audit.role": schema.__name__}) as span:
                try:
                    result = await runnable.ainvoke(messages, config=config)
                except Exception as error:
                    observe(schema.__name__, messages, {"audit_error": repr(error)}, schema.model_json_schema())
                    raise
                observe(schema.__name__, messages, result, schema.model_json_schema())
                set_span_output(span, result)
                return result
        def run_sync(messages, config=None):
            with trace_span("AUDIT scripted " + schema.__name__, input_value=messages,
                            attributes={"audit.synthetic": True, "audit.role": schema.__name__}) as span:
                try:
                    result = runnable.invoke(messages, config=config)
                except Exception as error:
                    observe(schema.__name__, messages, {"audit_error": repr(error)}, schema.model_json_schema())
                    raise
                observe(schema.__name__, messages, result, schema.model_json_schema())
                set_span_output(span, result)
                return result
        return RunnableLambda(run_sync, afunc=run)
    cls.with_structured_output = wrapped

for module in modules:
    for cls in vars(module).values():
        if not inspect.isclass(cls) or cls.__module__ != module.__name__:
            continue
        if "with_structured_output" in cls.__dict__:
            instrument_structured(cls)
        if issubclass(cls, BaseChatModel) and "_generate" in cls.__dict__:
            original = cls._generate
            def generate(self, messages, *args, _original=original, **kwargs):
                with trace_span("AUDIT scripted " + type(self).__name__, input_value=messages,
                                attributes={"audit.synthetic": True}) as span:
                    result = _original(self, messages, *args, **kwargs)
                    observe(type(self).__name__, messages, result)
                    set_span_output(span, result)
                    return result
            cls._generate = generate

class ReplanRoutes(unittest.IsolatedAsyncioTestCase):
    def test_repeated_test_dispute_exhausts_local_budget(self):
        from workers.code_review_models import create_code_review_loop, request_code_repair, receive_code_worker_response, CodeWorkerRepairResponse
        m = modules[3]
        loop = create_code_review_loop(candidate=m.candidate(), worker_checkpoint_id="worker", reviewer_checkpoint_id="reviewer", max_repair_rounds=2)
        states = []
        for n in (1, 2):
            loop = request_code_repair(loop, summary="Threshold check disputed", required_changes=("Meet inclusive threshold",), findings=(m.finding(),))
            loop = receive_code_worker_response(loop, CodeWorkerRepairResponse(round_no=n, action="TEST_DISPUTE", candidate=loop.candidate,
                summary="Worker disputes reviewer assertion", reasons=("Worker thinks threshold should be exclusive",)))
            states.append(safe(loop))
        with self.assertRaisesRegex(ValueError, "预算"):
            request_code_repair(loop, summary="Try again", required_changes=("Fix threshold",), findings=(m.finding(),))
        self.assertEqual(len(loop.exchanges), 2)
        self.assertEqual(loop.candidate.candidate_revision, 1)
        (OUT / "dispute-result.json").write_text(json.dumps(states, ensure_ascii=False, indent=2), encoding="utf-8")

    async def test_code_stop_then_replan(self):
        await self._replan(False)

    async def test_code_stop_then_replan_new_code_step(self):
        await self._replan(True)

    async def _replan(self, continue_plan):
        m = modules[1]
        from tempfile import TemporaryDirectory
        from planning_models import PlanningContextPack
        class Model(m.ScriptedCodePlanningModel):
            finals = 0
            replans = 0
            def with_structured_output(self, schema, **kwargs):
                if schema.__name__ not in {"FinalReviewDecision", "ReplanDecision"}:
                    return super().with_structured_output(schema, **kwargs)
                async def run(messages):
                    if schema.__name__ == "ReplanDecision":
                        self.replans += 1
                        payload = ({"action": "CONTINUE", "reason": "Try a narrower code contract.", "remaining_steps": [{
                            "step_id": 2, "objective": "Publish a narrowly scoped app.py", "success_criteria": ["The file is published."],
                            "worker_kind": "CODE", "execution_mode": "SINGLE", "code_task": {"requirements": [{"requirement_id": "publish_file", "statement": "Publish app.py."}], "validation_expectations": ["Focused check"]}}]}
                            if continue_plan else {"action": "FINISH", "reason": "No safe remaining implementation within scope."})
                    else:
                        self.finals += 1
                        payload = ({"action": "REPLAN", "replan_reason": "Code was stopped; assess a safe alternative."}
                                   if self.finals == 1 else {"action": "FINAL", "status": "COMPLETED" if continue_plan else "FAILED",
                                    "final_answer": "Narrower plan completed." if continue_plan else "Code stopped; no safe alternative.", "unmet_success_criteria": [] if continue_plan else ["File publication"]})
                    result = {"parsed": schema.model_validate(payload), "raw": AIMessage(content=json.dumps(payload)), "parsing_error": None}
                    observe(schema.__name__, messages, result, schema.model_json_schema())
                    return result
                return RunnableLambda(run)
        with TemporaryDirectory() as temporary:
            store = m.AsyncEventStore(Path(temporary) / "events.sqlite3")
            await store.start()
            try:
                model = Model(code_action="STOP")
                class Runtime(m.EscalateThenApplyCodeRuntime):
                    async def ainvoke(self, input_state, config=None):
                        if len(self.calls) >= 2 and "code_scheduler_decision" not in input_state:
                            return await m.AppliedCodeRuntime().ainvoke(input_state, config=config)
                        return await super().ainvoke(input_state, config=config)
                runtime = Runtime(expected_action="STOP")
                graph = m.build_planning_graph(simple_model=model, hard_model=model,
                    worker_registry=m.WorkerAgentRegistry.with_code_worker(m.FailingGeneralRuntime(), runtime),
                    worker_group_coordinator=m.WorkerGroupCoordinator(store), planning=m.planning_settings(), model_output_max_tokens=2048)
                result = await graph.ainvoke({"context": PlanningContextPack(current_time="now", user_request="Implement app.py; if code stops, assess a safe alternative."),
                    "event_id": "audit-replan", "conversation_thread_id": "audit-replan", "planning_run_id": "audit-replan"},
                    config={"configurable": {"progress_callback": lambda event: m._ignore_progress(event)}})
                (OUT / ("replan-continue-result.json" if continue_plan else "replan-result.json")).write_text(json.dumps(safe(result), ensure_ascii=False, indent=2), encoding="utf-8")
                self.assertEqual(model.replans, 1)
                self.assertEqual(model.finals, 2)
                self.assertEqual(result["final_status"], "COMPLETED" if continue_plan else "FAILED")
            finally:
                await store.close()

families = {
    "01_web_code": ["test_planning_web_to_code_pipeline", "test_planning_code_execution.PlanningCodeExecutionTests.test_applied_report_returns_to_scheduler_without_second_reporter"],
    "02_repair_disagreement": ["test_code_agents", "test_code_review_models"],
    "03_continue": ["test_planning_code_execution.PlanningCodeExecutionTests.test_reviewer_escalation_runs_controller_then_continues_same_step", "test_code_runtime.CodeRuntimeTests.test_continue_reuses_frozen_pair_and_applies_revision"],
    "04_restart": ["test_planning_code_execution.PlanningCodeExecutionTests.test_controller_restart_creates_a_new_planning_attempt", "test_code_runtime.CodeRuntimeTests.test_restart_supersedes_old_attempt_and_starts_fresh_pair"],
    "05_stop": ["test_planning_code_execution.PlanningCodeExecutionTests.test_controller_stop_finalizes_without_generic_retry", "test_code_runtime.CodeRuntimeTests.test_stop_archives_and_releases_frozen_pair"],
    "06_replan": [],
    "07_skills_web_boundaries": ["test_skill_preparation", "test_web_runtime"],
}
summary = {"paid_requests": 0, "usage": "unknown; synthetic responses do not report billable tokens", "families": []}
class Result(unittest.TextTestResult):
    def startTest(self, test):
        global current_case
        current_case = test.id()
        if hasattr(test, "_asyncioTestContext"):
            test._asyncioTestContext = contextvars.copy_context()
        print("CASE " + current_case, flush=True)
        super().startTest(test)

with (OUT / "test-output.txt").open("w", encoding="utf-8") as log:
    for family, names in families.items():
        if len(sys.argv) > 2 and family not in sys.argv[2:]:
            continue
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in names
                                   if not (os.getenv("AUDIT_SKIP_RUNTIME") == "1" and name.startswith("test_code_runtime")))
        if family == "06_replan":
            suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(ReplanRoutes))
        start = time.monotonic()
        with trace_span("AUDIT " + family, attributes={"audit.synthetic": True}) as span:
            result = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=Result).run(suite)
            row = {"family": family, "tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
                   "seconds": round(time.monotonic() - start, 2), "trace_id": format(span.get_span_context().trace_id, "032x") if span else None}
            set_span_output(span, row)
            summary["families"].append(row)
        log.flush()
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
if provider:
    provider.force_flush(timeout_millis=15000)
summary["scripted_requests"] = len(records)
summary["approx_message_tokens"] = sum(r["approx_message_tokens"] or 0 for r in records)
summary["schema_chars"] = sum(r["schema_chars"] for r in records)
summary["case_requests"] = {case: {"requests": len(rs := [r for r in records if r["case"] == case]),
                                      "approx_message_tokens": sum(r["approx_message_tokens"] or 0 for r in rs)}
                            for case in sorted({r["case"] for r in records})}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False), flush=True)
