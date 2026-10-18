"""Offline coverage of the single CODE failure -> global Replanner boundary."""
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from eventing import AsyncEventStore
from planning_graph import build_planning_graph
from planning_models import PlanningContextPack
from tests.test_planning_code_execution import (
    AppliedCodeRuntime, FailingGeneralRuntime, ScriptedCodePlanningModel,
    planning_settings, _ignore_progress,
)
from tests.test_code_runtime import (
    FakeSandboxManager, FakeReviewerGraph, RepairAwareWorkerGraph,
    contract, failed_review,
)
from workers import (
    CodeCandidateRef, CodeReviewFinding, CodeReviewLoopState, WorkerAgentRegistry,
    WorkerGroupCoordinator, create_code_review_loop, request_code_repair,
)
from workers.code_finalization import CodeReviewerProgressMiddleware
from workers.code_runtime import CodeStepRuntime
from workers.code_runtime_checkpoint import CodeRuntimeCheckpointStore


class ReplanModel(ScriptedCodePlanningModel):
    def __init__(self, finish=False):
        super().__init__()
        self.finish = finish
        self.replan_inputs = []

    def with_structured_output(self, schema, **kwargs):
        if schema.__name__ == "FinalReviewDecision":
            def respond(messages):
                packet = json.loads(messages[1]["content"])
                reports = packet["独立审核材料"]["step_reports"]
                completed = bool(reports) and reports[-1].get("status") == "COMPLETED"
                payload = {"action": "FINAL", "status": "COMPLETED" if completed else "FAILED",
                    "final_answer": "Reviewed file published." if completed else "Focused behavior remains unverified.",
                    "unmet_success_criteria": [] if completed else ["Publish app.py."]}
                return {"parsed": schema.model_validate(payload), "raw": AIMessage(content=json.dumps(payload)), "parsing_error": None}
            return RunnableLambda(respond)
        if schema.__name__ == "ScopeContract":
            payload = dict(target_entity="file", effect_mode="MUTATION", sets=[
                dict(set_id="A", definition="Requested app.py", result_entity="file")
            ], operation="DIRECT", operands=["A"])
            return RunnableLambda(lambda _: {"parsed": schema.model_validate(payload), "raw": None, "parsing_error": None})
        if schema.__name__ == "ReplanDecision":
            async def respond(messages):
                self.replan_inputs.append(messages)
                payload = {"action": "FINISH", "reason": "No credible path remains."} if self.finish else {
                    "action": "CONTINUE", "reason": "Use a fresh candidate and recheck the failed behavior.",
                    "remaining_steps": [{"step_id": 2, "objective": "Repair and publish app.py using a fresh attempt.",
                        "worker_kind": "CODE", "success_criteria": [], "code_task": {
                            "requirements": [{"requirement_id": "publish_file", "statement": "Publish app.py.", "priority": "MUST"}],
                            "validation_expectations": ["Recheck focused behavior."],
                        }}],
                }
                return {"parsed": schema.model_validate(payload), "raw": AIMessage(content=json.dumps(payload)), "parsing_error": None}
            return RunnableLambda(respond)
        return super().with_structured_output(schema, **kwargs)


class FailedThenAppliedRuntime(AppliedCodeRuntime):
    def __init__(self, always_fail=False):
        super().__init__()
        self.inputs = []
        self.always_fail = always_fail

    async def ainvoke(self, state, config=None):
        self.inputs.append(state)
        if len(self.inputs) > 1 and not self.always_fail:
            return await super().ainvoke(state, config=config)
        candidate = CodeCandidateRef(event_id=state["event_id"], step_id=int(state["step_id"]),
            attempt_id=config["configurable"]["thread_id"], workspace_id=state["worker_id"], candidate_revision=4)
        loop = create_code_review_loop(candidate=candidate, worker_checkpoint_id="w", reviewer_checkpoint_id="r")
        loop, report = failed_review(loop)
        return {"messages": [AIMessage(content=report.summary)], "executor_model_calls_used": 2,
            "executor_tool_calls_used": 2, "code_review_loop": loop.model_dump(mode="json"),
            "code_review_report": report.model_dump(mode="json"), "code_runtime_session_id": None}


class ThreeRepairsReviewer(FakeReviewerGraph):
    def __init__(self, pass_last=False):
        self.calls = 0
        self.pass_last = pass_last

    async def ainvoke(self, state, config=None):
        self.calls += 1
        loop = CodeReviewLoopState.model_validate(state["code_review_loop"])
        if loop.repair_round < loop.max_repair_rounds:
            loop = request_code_repair(loop, summary="Focused behavior failed.", required_changes=("Fix the behavior.",), findings=(CodeReviewFinding(finding_id="behavior", summary="Focused behavior failed.", affected_requirement_ids=("feature",)),))
            return {"code_review_loop": loop.model_dump(mode="json"), "executor_model_calls_used": 1, "executor_tool_calls_used": 1}
        if self.pass_last:
            return await super().ainvoke(state, config=config)
        loop, report = failed_review(loop)
        return {"code_review_loop": loop.model_dump(mode="json"), "code_review_report": report.model_dump(mode="json"),
            "executor_model_calls_used": 1, "executor_tool_calls_used": 1}


class CodeGlobalReplanTests(unittest.IsolatedAsyncioTestCase):
    async def run_graph(self, model, runtime, max_replans=1):
        with TemporaryDirectory() as tmp:
            store = AsyncEventStore(Path(tmp) / "events.sqlite3")
            await store.start()
            try:
                graph = build_planning_graph(simple_model=model, hard_model=model,
                    worker_registry=WorkerAgentRegistry.with_code_worker(FailingGeneralRuntime(), runtime),
                    worker_group_coordinator=WorkerGroupCoordinator(store),
                    planning=replace(planning_settings(), max_replans=max_replans, max_plan_model_rounds=40), model_output_max_tokens=2048)
                self.assertNotIn("code_controller", graph.get_graph().nodes)
                return await graph.ainvoke({"context": PlanningContextPack(current_time="now", user_request="Implement app.py.", skill_mode="off"),
                    "event_id": "replan-test", "conversation_thread_id": "c", "planning_run_id": "p"},
                    config={"configurable": {"progress_callback": _ignore_progress}})
            finally:
                await store.close()

    async def test_failed_report_replans_to_fresh_step_without_controller(self):
        model, runtime = ReplanModel(), FailedThenAppliedRuntime()
        result = await self.run_graph(model, runtime)
        self.assertEqual(result["replans_used"], 1)
        self.assertEqual(len(runtime.inputs), 2)
        self.assertNotEqual(runtime.inputs[0]["step_id"], runtime.inputs[1]["step_id"])
        self.assertTrue(result["completed_step_reports"][0].request_replan)
        self.assertEqual(result["completed_step_reports"][-1].status, "COMPLETED")
        self.assertIn("focused-check: old behavior", str(model.replan_inputs))
        self.assertEqual(model.code_control_calls, 0)

    async def test_replanner_can_finish_without_new_worker(self):
        model, runtime = ReplanModel(finish=True), FailedThenAppliedRuntime()
        result = await self.run_graph(model, runtime)
        self.assertEqual(result["replans_used"], 1)
        self.assertEqual(len(runtime.inputs), 1)
        self.assertIn("final_answer", result)

    async def test_no_replan_quota_goes_to_final_review(self):
        model, runtime = ReplanModel(), FailedThenAppliedRuntime()
        result = await self.run_graph(model, runtime, max_replans=0)
        self.assertEqual(len(model.replan_inputs), 0)
        self.assertEqual(len(runtime.inputs), 1)
        self.assertIn("final_answer", result)

    async def test_last_replan_runs_new_step_then_stops_on_another_failure(self):
        model, runtime = ReplanModel(), FailedThenAppliedRuntime(always_fail=True)
        result = await self.run_graph(model, runtime, max_replans=1)
        self.assertEqual(result["replans_used"], 1)
        self.assertEqual(len(model.replan_inputs), 1)
        self.assertEqual(len(runtime.inputs), 2)
        self.assertEqual(result["final_status"], "FAILED")

    async def run_local_repairs(self, pass_last):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            manager, worker, reviewer = FakeSandboxManager(), RepairAwareWorkerGraph(), ThreeRepairsReviewer(pass_last)
            runtime = CodeStepRuntime(object(), sandbox_manager=manager, source_root=workspace, target_root=workspace,
                archive_root=root / "archives", run_storage_root=root / "runs",
                worker_factory=lambda *a, **k: worker, reviewer_factory=lambda *a, **k: reviewer)
            result = await runtime.ainvoke({"messages": [{"role": "user", "content": "Implement app.py"}],
                "event_id": "three-repairs", "step_id": "1", "worker_id": "w", "code_task": contract().model_dump(mode="json"),
                "executor_model_run_limit": 24, "executor_tool_run_limit": 24}, config={"configurable": {"thread_id": "local"}})
            self.assertEqual(worker.calls, 4)
            self.assertEqual(reviewer.calls, 4)
            self.assertEqual(result["code_review_loop"]["repair_round"], 3)
            self.assertTrue(manager.cleaned)
            self.assertFalse(runtime.recovered_session_ids)
            checkpoint = CodeRuntimeCheckpointStore().load(Path(result["code_attempt_archive"]["root_path"]))
            self.assertEqual(checkpoint.phase, "TERMINAL")
            self.assertTrue(checkpoint.cleanup_authorized)
            return result

    async def test_three_failed_repairs_archive_before_replan(self):
        result = await self.run_local_repairs(False)
        self.assertEqual(result["code_review_loop"]["status"], "REPLAN_REQUIRED")
        self.assertIsNone(result["code_runtime_session_id"])
        self.assertEqual(result["code_attempt_final_record"]["outcome"], "FAILED")

    async def test_last_repaired_candidate_can_still_pass(self):
        result = await self.run_local_repairs(True)
        self.assertEqual(result["code_review_loop"]["status"], "APPLIED")

    def test_last_review_hides_repair_tool_but_keeps_verification_and_submission(self):
        class Request(SimpleNamespace):
            def override(self, **kwargs):
                return Request(**{**vars(self), **kwargs})
        request = Request(state={"code_review_loop": {"repair_round": 3, "max_repair_rounds": 3}},
            tools=[{"name": name} for name in ("request_code_worker_repair", "execute", "submit_code_review", "publish_reviewed_candidate")])
        changed = CodeReviewerProgressMiddleware()._require_control_request(request)
        self.assertEqual([tool["name"] for tool in changed.tools], ["execute", "submit_code_review", "publish_reviewed_candidate"])
