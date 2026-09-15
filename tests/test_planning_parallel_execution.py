"""No-provider integration test for PlanningGraph parallel Worker fan-out."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ["PHOENIX_TRACING_ENABLED"] = "false"

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from config import PlanningSettings
from eventing import AsyncEventStore
from planning_graph import build_planning_graph
from planning_models import PlanningContextPack
from planning_models import PlanStep
from workers import (
    StepWorkerGroup,
    WorkerAgentRegistry,
    WorkerAttemptStatus,
    WorkerGroupCoordinator,
    WorkerGroupReviewStatus,
)


class ScriptedPlanningModel:
    def __init__(self) -> None:
        self.reporter_messages = []
        self.scheduler_messages = []

    def with_structured_output(self, schema, **kwargs):
        payloads = {
            "SupervisorDecision": {
                "action": "PLAN",
                "plan_objective": "Research three independent sources.",
                "plan_success_criteria": ["Return all three findings."],
                "steps": [
                    {
                        "step_id": 1,
                        "objective": "Research three sources in parallel.",
                        "success_criteria": ["Return all three findings."],
                        "worker_kind": "WEB",
                        "execution_mode": "PARALLEL",
                        "worker_assignments": [
                            {"assignment_key": "first", "objective": "Source A."},
                            {"assignment_key": "second", "objective": "Source B."},
                            {"assignment_key": "third", "objective": "Source C."},
                        ],
                    }
                ],
            },
            "StepReport": {
                "step_id": 1,
                "status": "COMPLETED",
                "summary": "All three Worker contributions were reviewed.",
                "stop_reason": "ALL_TERMINAL",
                "criterion_results": [
                    {
                        "criterion": "Return all three findings.",
                        "criterion_id": "C1",
                        "status": "MET",
                        "evidence": [],
                    }
                ],
                "confirmed_results": ["A", "B", "C"],
                "evidence": [],
            },
            "FinalReviewDecision": {
                "action": "FINAL",
                "status": "COMPLETED",
                "final_answer": "A, B, C",
            },
        }

        async def respond(messages):
            if schema.__name__ == "StepReport":
                self.reporter_messages.append(messages)
            else:
                self.scheduler_messages.append(copy.deepcopy(messages))
            payload = payloads[schema.__name__]
            return {
                "parsed": schema.model_validate(payload),
                "raw": AIMessage(content=json.dumps(payload)),
                "parsing_error": None,
            }

        return RunnableLambda(respond)


class BarrierWorker:
    """Fake Worker proving all three invocations overlap in wall-clock time."""

    def __init__(self, expected: int) -> None:
        self.expected = expected
        self.arrived = 0
        self.active = 0
        self.max_active = 0
        self.release = asyncio.Event()
        self.thread_ids: list[str] = []
        self.model_limits: list[int] = []
        self.tool_limits: list[int] = []

    async def ainvoke(self, input_state, config=None):
        thread_id = config["configurable"]["thread_id"]
        self.thread_ids.append(thread_id)
        self.model_limits.append(input_state["executor_model_run_limit"])
        self.tool_limits.append(input_state["executor_tool_run_limit"])
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.arrived += 1
        if self.arrived == self.expected:
            self.release.set()
        try:
            await asyncio.wait_for(self.release.wait(), timeout=1)
            worker_id = input_state["worker_id"]
            return {
                "messages": [AIMessage(content=f"Completed {worker_id}")],
                "executor_model_calls_used": 1,
                "executor_tool_calls_used": 1,
                "show_all_toolsets_calls_used": 0,
                "worker_submission": {
                    "submitted_at": "2026-09-03T00:00:00+00:00",
                    "worker_id": worker_id,
                    "event_id": input_state["event_id"],
                    "step_id": input_state["step_id"],
                    "total_tool_calls": 1,
                    "submission": {
                        "summary": f"Completed {worker_id}",
                        "final_conclusion": f"Finding from {worker_id}",
                        "criterion_claims": [],
                        "artifact_candidates": [],
                        "unresolved_items": [],
                    },
                    "resolved_evidence": [],
                },
            }
        finally:
            self.active -= 1


def planning_settings() -> PlanningSettings:
    return PlanningSettings(
        max_steps_per_plan=3,
        max_total_steps=6,
        max_replans=1,
        max_step_model_rounds=12,
        max_step_executor_rounds=9,
        max_step_report_rounds=2,
        max_step_tool_calls=9,
        max_step_attempts=2,
        max_plan_model_rounds=20,
        max_plan_tool_calls=12,
        hard_recent_dialogue_turns=4,
        hard_recent_dialogue_max_chars=8000,
        conversation_summary_trigger_turns=8,
        conversation_summary_max_chars=4000,
        executor_summary_trigger_tokens=5000,
        executor_summary_trigger_messages=18,
        executor_summary_keep_messages=10,
    )


class PlanningParallelExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_scheduler_history_survives_graph_checkpoint_resume(self):
        from langgraph.checkpoint.memory import InMemorySaver
        with TemporaryDirectory() as temporary:
            store = AsyncEventStore(Path(temporary) / "events.sqlite3")
            await store.start()
            try:
                model = ScriptedPlanningModel()
                worker = BarrierWorker(3)
                graph = build_planning_graph(
                    simple_model=model, hard_model=model,
                    worker_registry=WorkerAgentRegistry.general_worker_first(worker),
                    worker_group_coordinator=WorkerGroupCoordinator(store),
                    planning=planning_settings(), model_output_max_tokens=2048,
                    checkpointer=InMemorySaver(),
                )
                async def ignore_progress(*args, **kwargs):
                    pass
                config = {"configurable": {"thread_id": "progressive-checkpoint", "progress_callback": ignore_progress}}
                paused = await graph.ainvoke({"context": PlanningContextPack(
                    current_time="now", user_request="Research three sources", skill_mode="off"),
                    "event_id": "event-checkpoint", "planning_run_id": "run-checkpoint",
                    "conversation_thread_id": "conversation-checkpoint"},
                    config=config, interrupt_after=["supervisor"])
                context = PlanningContextPack.model_validate(paused["context"])
                self.assertTrue(context.scheduler_session["records"])
                self.assertEqual(worker.arrived, 0)
                result = await graph.ainvoke(None, config=config)
                self.assertEqual(result["final_answer"], "A, B, C")
                self.assertEqual(len(model.scheduler_messages), 2)
                first, final = model.scheduler_messages
                self.assertEqual(first[0]["role"], final[0]["role"])
                self.assertTrue(first[0]["content"])
                self.assertTrue(final[0]["content"])
                self.assertNotIn("task_contract", first[0]["content"])
                final_context = PlanningContextPack.model_validate(result["context"])
                self.assertIn("活动计划", final_context.scheduler_session["active_records"])
                self.assertTrue(any(
                    record["kind"] == "StepReport"
                    for record in final_context.scheduler_session["records"]
                ))
                self.assertFalse(any(
                    str(record.get("key", "")).startswith("transient:")
                    for record in final_context.scheduler_session["records"]
                ))
                self.assertTrue(model.reporter_messages)
            finally:
                await store.close()

    async def test_join_ready_group_rebuilds_report_without_rerunning_workers(self):
        with TemporaryDirectory() as temporary:
            store = AsyncEventStore(Path(temporary) / "events.sqlite3")
            await store.start()
            try:
                coordinator = WorkerGroupCoordinator(store)
                step = PlanStep(
                    step_id=1,
                    objective="Research three sources in parallel.",
                    success_criteria=["Return all three findings."],
                    worker_kind="WEB",
                    execution_mode="PARALLEL",
                    worker_assignments=[
                        {"assignment_key": "first", "objective": "Source A."},
                        {"assignment_key": "second", "objective": "Source B."},
                        {"assignment_key": "third", "objective": "Source C."},
                    ],
                )
                stored = await coordinator.create_group(
                    step,
                    event_id="event-recovery",
                    group_id="group:event-recovery:step:1:execution:1",
                )
                group = StepWorkerGroup.model_validate(stored.snapshot)
                for slot in group.slots:
                    attempt = slot.current_attempt
                    await coordinator.start_attempt(
                        group_id=group.group_id,
                        assignment_key=slot.assignment_key,
                        attempt_no=attempt.attempt_no,
                        worker_id=attempt.worker_id,
                        occurred_at=datetime.now(timezone.utc),
                    )
                    await coordinator.finish_attempt(
                        group_id=group.group_id,
                        assignment_key=slot.assignment_key,
                        attempt_no=attempt.attempt_no,
                        worker_id=attempt.worker_id,
                        status=WorkerAttemptStatus.SUBMITTED,
                        terminal_reason="Worker submitted before outer graph crash.",
                        review_payload={
                            "assignment_key": slot.assignment_key,
                            "assignment_objective": attempt.objective,
                            "attempt": attempt.attempt_no,
                            "worker_id": attempt.worker_id,
                            "workspace_id": attempt.workspace.workspace_id,
                            "checkpoint_thread_id": (
                                attempt.workspace.checkpoint_thread_id
                            ),
                            "finish_reason": "READY_FOR_REVIEW",
                            "stop_reason": "Submitted before crash.",
                            "final_answer": f"Recovered {slot.assignment_key}",
                            "execution_summary": {
                                "model_call_count": 1,
                                "tool_call_count": 1,
                            },
                        },
                        occurred_at=datetime.now(timezone.utc),
                    )

                model = ScriptedPlanningModel()
                worker = BarrierWorker(expected=1)
                graph = build_planning_graph(
                    simple_model=model,
                    hard_model=model,
                    worker_registry=WorkerAgentRegistry.general_worker_first(worker),
                    worker_group_coordinator=coordinator,
                    planning=planning_settings(),
                    model_output_max_tokens=2048,
                    web_max_parallelism=3,
                )

                async def ignore_progress(event):
                    return None

                result = await graph.ainvoke(
                    {
                        "context": PlanningContextPack(
                            current_time="now",
                            user_request="Research A, B, and C.",
                            skill_mode="off",
                        ),
                        "event_id": "event-recovery",
                        "conversation_thread_id": "conversation-recovery",
                        "planning_run_id": "planning-recovery",
                    },
                    config={
                        "configurable": {
                            "progress_callback": ignore_progress,
                        }
                    },
                )

                self.assertEqual(result["final_status"], "COMPLETED")
                self.assertEqual(worker.thread_ids, [])
                self.assertEqual(len(model.reporter_messages), 1)
                reporter_prompt = str(model.reporter_messages[0])
                for assignment_key in ("first", "second", "third"):
                    self.assertIn(assignment_key, reporter_prompt)
                    self.assertIn(f"Recovered {assignment_key}", reporter_prompt)
                recovered = await store.require_worker_group(group.group_id)
                self.assertEqual(
                    recovered.snapshot["review_status"],
                    WorkerGroupReviewStatus.REPORTED.value,
                )
            finally:
                await store.close()

    async def test_three_workers_fan_out_join_and_run_one_reporter(self):
        with TemporaryDirectory() as temporary:
            store = AsyncEventStore(Path(temporary) / "events.sqlite3")
            await store.start()
            try:
                model = ScriptedPlanningModel()
                worker = BarrierWorker(expected=3)
                graph = build_planning_graph(
                    simple_model=model,
                    hard_model=model,
                    worker_registry=WorkerAgentRegistry.general_worker_first(worker),
                    worker_group_coordinator=WorkerGroupCoordinator(store),
                    planning=planning_settings(),
                    model_output_max_tokens=2048,
                    web_max_parallelism=3,
                )

                async def ignore_progress(event):
                    return None

                result = await graph.ainvoke(
                    {
                        "context": PlanningContextPack(
                            current_time="now",
                            user_request="Research A, B, and C.",
                            skill_mode="off",
                        ),
                        "event_id": "event-parallel",
                        "conversation_thread_id": "conversation-parallel",
                        "planning_run_id": "planning-parallel",
                    },
                    config={
                        "configurable": {
                            "progress_callback": ignore_progress,
                        }
                    },
                )

                self.assertEqual(result["final_status"], "COMPLETED")
                self.assertEqual(worker.max_active, 3)
                self.assertEqual(len(set(worker.thread_ids)), 3)
                self.assertEqual(sum(worker.model_limits), 9)
                self.assertEqual(sum(worker.tool_limits), 9)
                self.assertEqual(len(model.reporter_messages), 1)
                reporter_prompt = str(model.reporter_messages[0])
                for assignment_key in ("first", "second", "third"):
                    self.assertIn(assignment_key, reporter_prompt)

                stored = await store.require_worker_group(
                    "group:event-parallel:step:1:execution:1"
                )
                group = StepWorkerGroup.model_validate(stored.snapshot)
                self.assertEqual(
                    group.review_status,
                    WorkerGroupReviewStatus.REPORTED,
                )
                self.assertTrue(
                    all(
                        attempt.status is WorkerAttemptStatus.SUBMITTED
                        for attempt in group.current_attempts
                    )
                )
                audits = await store.list_worker_attempt_audit(
                    group_id=group.group_id
                )
                self.assertEqual(len(audits), 6)
            finally:
                await store.close()


if __name__ == "__main__":
    unittest.main()
