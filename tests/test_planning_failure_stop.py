"""Exercise the real planning graph with provider-free failure responses."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
import test_planning_code_execution as fixture


class WorkerSpy:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("No Worker should start")


class Model(fixture.ScriptedCodePlanningModel):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.calls = []
        self.supervisor_calls = 0

    def with_structured_output(self, schema, **kwargs):
        original = super().with_structured_output(schema, **kwargs)
        async def respond(messages):
            self.calls.append(schema.__name__)
            if schema.__name__ == "SkillChoice":
                return {"parsed": {"skill_ids": [], "reason": "Offline baseline"}}
            if schema.__name__ == "SupervisorDecision":
                self.supervisor_calls += 1
                if self.mode == "timeout":
                    raise TimeoutError("offline timeout")
                if self.mode == "fail" or (self.mode == "repair" and self.supervisor_calls == 1):
                    return {"parsed": {"action": "INVALID"}, "raw": AIMessage(content='{"action":"INVALID"}')}
                if self.mode == "final":
                    return {"parsed": {"action": "FINAL", "final_answer": "Direct answer"}}
            return await original.ainvoke(messages)
        return RunnableLambda(respond)


class FailureStopTests(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, mode):
        with TemporaryDirectory() as temporary:
            store = fixture.AsyncEventStore(Path(temporary) / "events.sqlite3")
            await store.start()
            try:
                model, general = Model(mode), WorkerSpy()
                code = fixture.AppliedCodeRuntime() if mode == "repair" else WorkerSpy()
                graph = fixture.build_planning_graph(simple_model=model, hard_model=model,
                    worker_registry=fixture.WorkerAgentRegistry.with_code_worker(general, code),
                    worker_group_coordinator=fixture.WorkerGroupCoordinator(store),
                    planning=fixture.planning_settings(), model_output_max_tokens=2048)
                result = await graph.ainvoke({"context": fixture.PlanningContextPack(current_time="now", user_request="Implement app.py"),
                    "event_id": "failure-audit", "conversation_thread_id": "failure-audit", "planning_run_id": "failure-audit"},
                    config={"configurable": {"progress_callback": fixture._ignore_progress}})
                self.assertEqual(general.calls, 0)
                return result, model, code
            finally:
                await store.close()

    async def test_two_invalid_plans_stop_without_other_model_or_worker(self):
        result, model, code = await self.run_case("fail")
        self.assertEqual(result["final_status"], "FAILED")
        self.assertEqual(code.calls, 0)
        self.assertEqual(model.calls, ["ScopeContract"] * 3 + ["SkillChoice"] + ["SupervisorDecision"] * 4)
        self.assertEqual(result["planning_failure"]["attempts"], 4)
        self.assertEqual(len(result["planning_failure"]["errors"]), 4)
        self.assertEqual(result["overall_stop_reason"], "supervisor_generation_failed")
        self.assertIn("原因", result["final_answer"])
        self.assertFalse(result.get("remaining_steps"))

    async def test_two_timeouts_stop_without_summary_model(self):
        result, model, code = await self.run_case("timeout")
        self.assertEqual(result["final_status"], "FAILED")
        self.assertEqual(code.calls, 0)
        self.assertEqual(model.calls, ["ScopeContract"] * 3 + ["SkillChoice", "SupervisorDecision"])
        self.assertIn("offline timeout", result["final_answer"])

    async def test_valid_repair_still_executes_code(self):
        result, model, code = await self.run_case("repair")
        self.assertEqual(result["final_status"], "COMPLETED")
        self.assertEqual(model.supervisor_calls, 2)
        self.assertIsNotNone(code.received_contract)

    async def test_valid_direct_answer_is_not_marked_failed(self):
        result, model, code = await self.run_case("final")
        self.assertEqual(result["final_status"], "COMPLETED")
        self.assertEqual(code.calls, 0)
        self.assertEqual(result["final_answer"], "Direct answer")
