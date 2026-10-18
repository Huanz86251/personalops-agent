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
           "test_skill_preparation", "test_web_runtime", "test_code_global_replan"]
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
        m = modules[-1]
        model = m.ReplanModel(finish=not continue_plan)
        runtime = m.FailedThenAppliedRuntime()
        result = await m.CodeGlobalReplanTests().run_graph(model, runtime)
        (OUT / ("replan-continue-result.json" if continue_plan else "replan-result.json")).write_text(
            json.dumps(safe(result), ensure_ascii=False, indent=2), encoding="utf-8")
        self.assertEqual(result["replans_used"], 1)
        self.assertEqual(len(runtime.inputs), 2 if continue_plan else 1)

families = {
    "01_web_code": ["test_planning_web_to_code_pipeline", "test_planning_code_execution.PlanningCodeExecutionTests.test_applied_report_returns_to_scheduler_without_second_reporter"],
    "02_repair_disagreement": ["test_code_agents", "test_code_review_models"],
    "03_replan_new_step": ["test_code_global_replan.CodeGlobalReplanTests.test_failed_report_replans_to_fresh_step_without_controller"],
    "04_local_repairs": ["test_code_global_replan.CodeGlobalReplanTests.test_three_failed_repairs_archive_before_replan", "test_code_global_replan.CodeGlobalReplanTests.test_last_repaired_candidate_can_still_pass"],
    "05_finish": ["test_code_global_replan.CodeGlobalReplanTests.test_replanner_can_finish_without_new_worker", "test_code_global_replan.CodeGlobalReplanTests.test_no_replan_quota_goes_to_final_review"],
    "06_replan": [],
    "07_skills_web_boundaries": ["test_skill_preparation", "test_web_runtime", "test_code_global_replan"],
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
