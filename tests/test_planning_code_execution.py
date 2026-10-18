"""Planning Graph integration tests for the reviewed CODE path."""

from __future__ import annotations

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
from workers import (
    CodeCandidateRef,
    CodeCheckResult,
    CodeChangedFile,
    CodeReviewReport,

    WorkerAgentRegistry,
    WorkerGroupCoordinator,

    create_code_review_loop,
    finish_code_review,
    record_code_publication,
)


def planning_settings() -> PlanningSettings:
    return PlanningSettings(
        max_steps_per_plan=3,
        max_total_steps=6,
        max_replans=1,
        max_step_model_rounds=12,
        max_step_executor_rounds=9,
        max_step_report_rounds=2,
        max_step_tool_calls=12,
        max_step_attempts=2,
        max_plan_model_rounds=20,
        max_plan_tool_calls=20,
        hard_recent_dialogue_turns=4,
        hard_recent_dialogue_max_chars=8000,
        conversation_summary_trigger_turns=8,
        conversation_summary_max_chars=4000,
        executor_summary_trigger_tokens=5000,
        executor_summary_trigger_messages=18,
        executor_summary_keep_messages=10,
    )


async def _ignore_progress(event):
    return None


class ScriptedCodePlanningModel:
    def __init__(self, code_action: str = "CONTINUE") -> None:
        self.step_report_calls = 0
        self.code_control_calls = 0
        self.code_action = code_action

    def with_structured_output(self, schema, **kwargs):
        async def respond(messages):
            if schema.__name__ == "SkillChoice":
                request = json.loads(messages[0]["content"])
                selected = ["plan-task-dependencies"] if request.get("role") == "scheduler" else []
                return {"parsed": schema.model_validate({"skill_ids": selected, "reason": "Offline fixture"}), "raw": None, "parsing_error": None}
            if schema.__name__ == "SupervisorDecision":
                payload = {
                    "action": "PLAN",
                    "plan_objective": "Implement and publish one file.",
                    "plan_success_criteria": ["The file is published."],
                    "steps": [
                        {
                            "step_id": 1,
                            "objective": "Implement the requested file.",
                            "success_criteria": ["The file is published."],
                            "worker_kind": "CODE",
                            "target_selection": {"target_entity": "file", "sets": [{"set_id": "A", "definition": "Requested app.py", "condition_owner": "file", "result_entity": "file"}], "operation": "DIRECT", "operands": ["A"]},
                            "execution_mode": "SINGLE",
                            "code_task": {
                                "delivery_mode": "PATCH",
                                "requirements": [
                                    {
                                        "requirement_id": "publish_file",
                                        "statement": "Publish app.py.",
                                        "priority": "MUST",
                                    }
                                ],
                                "validation_expectations": [
                                    "Run a focused behavior check."
                                ],
                            },
                        }
                    ],
                }
            elif schema.__name__ == "StepReport":
                self.step_report_calls += 1
                raise AssertionError(
                    "APPLIED CODE results must bypass generic Step Reporter"
                )
            else:
                payload = {
                    "action": "FINAL",
                    "status": (
                        "FAILED" if self.code_action == "STOP" else "COMPLETED"
                    ),
                    "final_answer": (
                        "The code attempt was stopped."
                        if self.code_action == "STOP"
                        else "The reviewed file was published."
                    ),
                    "unmet_success_criteria": (
                        ["The file is published."]
                        if self.code_action == "STOP"
                        else []
                    ),
                }
            return {
                "parsed": schema.model_validate(payload),
                "raw": AIMessage(content=json.dumps(payload)),
                "parsing_error": None,
            }

        return RunnableLambda(respond)


class FailingGeneralRuntime:
    async def ainvoke(self, input_state, config=None):
        raise AssertionError("CODE Step was incorrectly routed to general_worker")


class AppliedCodeRuntime:
    def __init__(self) -> None:
        self.received_contract = None

    async def ainvoke(self, input_state, config=None):
        self.received_contract = input_state.get("code_task")
        thread_id = config["configurable"]["thread_id"]
        candidate = CodeCandidateRef(
            event_id=input_state["event_id"],
            step_id=int(input_state["step_id"]),
            attempt_id=thread_id,
            workspace_id=input_state["worker_id"],
            candidate_revision=1,
        )
        loop = create_code_review_loop(
            candidate=candidate,
            worker_checkpoint_id=f"{thread_id}:worker",
            reviewer_checkpoint_id=f"{thread_id}:reviewer",
        )
        loop = record_code_publication(loop, candidate=candidate)
        report = CodeReviewReport(
            candidate=candidate,
            verdict="PASSED",
            summary="Reviewer approved and published app.py.",
            verification_summary="The focused behavior check passed.",
            verified_requirement_ids=("publish_file",),
            check_results=(
                CodeCheckResult(
                    check_id="behavior",
                    description="Run the focused behavior check.",
                    status="PASSED",
                    summary="Observed the expected result.",
                ),
            ),
            changed_files=(
                CodeChangedFile(
                    path="app.py",
                    change_summary="Implemented the requested behavior.",
                ),
            ),
            approved_artifact_paths=("app.py",),
            published_artifact_paths=("app.py",),
            delivery_location="D:/workspace",
            publication_id="publication-1",
            applied_revision="sha256:applied",
        )
        loop = finish_code_review(loop, report)
        return {
            "messages": [AIMessage(content=report.summary)],
            "executor_model_calls_used": 3,
            "executor_tool_calls_used": 5,
            "code_review_loop": loop.model_dump(mode="json"),
            "code_review_report": report.model_dump(mode="json"),
            "code_artifact_manifest": {
                "manifest_id": "manifest-1",
                "candidate": candidate.model_dump(mode="json"),
                "files": [],
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            "code_publication_receipt": {
                "publication_id": "publication-1"
            },
        }


class PlanningCodeExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_unhandled_code_failure_stops_without_generic_reporter(self):
        class BrokenCode:
            async def ainvoke(self, *args, **kwargs):
                raise RuntimeError("source import denied")
        class StrictModel(ScriptedCodePlanningModel):
            schemas = []
            def with_structured_output(self, schema, **kwargs):
                self.schemas.append(schema.__name__)
                if schema.__name__ not in {"ScopeContract", "SupervisorDecision", "SkillChoice"}:
                    raise AssertionError("Unexpected post-failure model: " + schema.__name__)
                return super().with_structured_output(schema, **kwargs)
        with TemporaryDirectory() as temporary:
            store = AsyncEventStore(Path(temporary) / "events.sqlite3")
            await store.start()
            try:
                model = StrictModel()
                graph = build_planning_graph(simple_model=model, hard_model=model,
                    worker_registry=WorkerAgentRegistry.with_code_worker(FailingGeneralRuntime(), BrokenCode()),
                    worker_group_coordinator=WorkerGroupCoordinator(store), planning=planning_settings(), model_output_max_tokens=2048)
                result = await graph.ainvoke({"context":PlanningContextPack(current_time="now",user_request="Implement app.py",skill_mode="off"),
                    "event_id":"broken-code","conversation_thread_id":"broken","planning_run_id":"broken"},
                    config={"configurable":{"progress_callback":_ignore_progress}})
                self.assertEqual(result["final_status"], "FAILED")
                self.assertEqual(result["overall_stop_reason"], "code_runtime_error")
                self.assertIn("source import denied", result["final_answer"])
                self.assertEqual(
                    model.schemas,
                    ["ScopeContract", "SupervisorDecision"],
                )
            finally:
                await store.close()
    async def test_applied_report_returns_to_scheduler_without_second_reporter(self):
        with TemporaryDirectory() as temporary:
            store = AsyncEventStore(Path(temporary) / "events.sqlite3")
            await store.start()
            try:
                model = ScriptedCodePlanningModel()
                code_runtime = AppliedCodeRuntime()
                graph = build_planning_graph(
                    simple_model=model,
                    hard_model=model,
                    worker_registry=WorkerAgentRegistry.with_code_worker(
                        FailingGeneralRuntime(),
                        code_runtime,
                    ),
                    worker_group_coordinator=WorkerGroupCoordinator(store),
                    planning=planning_settings(),
                    model_output_max_tokens=2048,
                )

                async def ignore_progress(event):
                    return None

                result = await graph.ainvoke(
                    {
                        "context": PlanningContextPack(
                            current_time="now",
                            user_request="Implement app.py.",
                        ),
                        "event_id": "event-code",
                        "conversation_thread_id": "conversation-code",
                        "planning_run_id": "planning-code",
                    },
                    config={
                        "configurable": {
                            "progress_callback": ignore_progress,
                        }
                    },
                )

                self.assertEqual(result["final_status"], "COMPLETED")
                self.assertEqual(model.step_report_calls, 0)
                self.assertIsNotNone(code_runtime.received_contract)
                self.assertEqual(
                    code_runtime.received_contract["requirements"][0]
                    ["requirement_id"],
                    "publish_file",
                )
                report = result["completed_step_reports"][0]
                self.assertEqual(report.status, "COMPLETED")
                self.assertEqual(report.artifacts[0].path, "app.py")
                self.assertIn("publication_id=publication-1", report.evidence)
                self.assertEqual(
                    result["current_step_trace"]["finish_reason"],
                    "CODE_APPLIED",
                )
            finally:
                await store.close()


if __name__ == "__main__":
    unittest.main()
