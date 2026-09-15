"""Offline end-to-end test for Reporter-approved shared publication."""

from __future__ import annotations

import hashlib
import json
import os
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ["PHOENIX_TRACING_ENABLED"] = "false"

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from config import PlanningSettings
from eventing import AsyncEventStore
from planning_graph import build_planning_graph
from planning_graph import _publish_step_handoff
from planning_models import PlanningContextPack, StepArtifactOutput, StepReport
from reporting.models import ReviewAttempt, ReviewTaskContract, StepReviewPacket
from workers import WorkerAgentRegistry, WorkerGroupCoordinator


async def ignore_progress(event):
    return None


def planning_settings(*, max_step_attempts: int = 1) -> PlanningSettings:
    return PlanningSettings(
        max_steps_per_plan=2,
        max_total_steps=3,
        max_replans=1,
        max_step_model_rounds=8,
        max_step_executor_rounds=5,
        max_step_report_rounds=2,
        max_step_tool_calls=5,
        max_step_attempts=max_step_attempts,
        max_plan_model_rounds=12,
        max_plan_tool_calls=8,
        hard_recent_dialogue_turns=4,
        hard_recent_dialogue_max_chars=8000,
        conversation_summary_trigger_turns=8,
        conversation_summary_max_chars=4000,
        executor_summary_trigger_tokens=5000,
        executor_summary_trigger_messages=18,
        executor_summary_keep_messages=10,
    )


class ArtifactWorker:
    def __init__(self, source: Path) -> None:
        self.source = source
        self.calls = 0

    async def ainvoke(self, input_state, config=None):
        self.calls += 1
        payload = self.source.read_bytes()
        worker_id = input_state["worker_id"]
        return {
            "messages": [AIMessage(content="Prepared a reusable research note.")],
            "executor_model_calls_used": 1,
            "executor_tool_calls_used": 1,
            "show_all_toolsets_calls_used": 0,
            "worker_submission": {
                "submitted_at": "2026-09-04T00:00:00+00:00",
                "worker_id": worker_id,
                "event_id": input_state["event_id"],
                "step_id": input_state["step_id"],
                "total_tool_calls": 1,
                "submission": {
                    "summary": "Prepared a reusable research note.",
                    "final_conclusion": "The note is ready for review.",
                    "criterion_claims": [],
                    "artifact_candidates": [],
                    "unresolved_items": [],
                },
                "resolved_evidence": [],
                "resolved_artifacts": [
                    {
                        "candidate_id": "research-note",
                        "output_id": "requested_note",
                        "kind": "DOWNLOADED_FILE",
                        "description": "Reviewer-approved research handoff.",
                        "verified": True,
                        "location": str(self.source),
                        "storage_path": str(self.source),
                        "size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            },
        }


class PublicationModel:
    def __init__(self, *, worker_kind="WEB", step_status="COMPLETED"):
        self.worker_kind = worker_kind
        self.step_status = step_status
        self.schemas = []

    def with_structured_output(self, schema, **kwargs):
        async def respond(messages):
            if schema.__name__ == "SkillChoice":
                return {"parsed": {"skill_ids": [], "reason": "Offline publication fixture has no matching skill"}}
            self.schemas.append(schema.__name__)
            prompt = str(messages)
            if schema.__name__ == "SupervisorDecision":
                payload = {
                    "action": "PLAN",
                    "plan_objective": "Create a reusable handoff note.",
                    "plan_success_criteria": ["Publish one reviewed note."],
                    "steps": [
                        {
                            "step_id": 1,
                            "objective": "Prepare one research note.",
                            "success_criteria": ["Publish one reviewed note."],
                            "worker_kind": self.worker_kind,
                            "artifact_outputs": [
                                {
                                    "output_id": "requested_note",
                                    "description": "The note requested by the user.",
                                    "disposition": "USER_DELIVERABLE",
                                    "target_path": "research/research.md",
                                    "required": True,
                                }
                            ],
                        }
                    ],
                }
            elif schema.__name__ == "StepReport":
                match = re.search("\\\"review_ref\\\":\\s*\\\"([^\\\"]+)\\\"", prompt)
                if match is None:
                    raise AssertionError("Reporter did not receive review_ref")
                if self.step_status == "COMPLETED":
                    payload = {
                        "step_id": 1,
                        "status": "COMPLETED",
                        "summary": "The note was reviewed and selected for handoff.",
                        "stop_reason": "Independent review completed.",
                        "criterion_results": [
                            {
                                "criterion_id": "C1",
                                "status": "MET",
                                "evidence": [match.group(1)],
                            }
                        ],
                        "confirmed_results": ["One note was approved."],
                        "approved_artifact_refs": [match.group(1)],
                        # A model-authored path must be discarded by the Harness.
                        "artifacts": [
                            {"path": "C:/invented.txt", "description": "untrusted"}
                        ],
                    }
                else:
                    payload = {
                        "step_id": 1,
                        "status": "BLOCKED",
                        "summary": "The evidence was insufficient.",
                        "stop_reason": "Independent review completed without approval.",
                        "criterion_results": [{
                            "criterion_id": "C1",
                            "status": "UNKNOWN",
                            "evidence": [],
                        }],
                        "unresolved_items": ["A reliable source is still missing."],
                        "next_action": "Scheduler may choose a different route.",
                        "request_replan": False,
                    }
            else:
                payload = {
                    "action": "FINAL",
                    "status": "COMPLETED",
                    "final_answer": "The reviewed note is available to later Steps.",
                    "unmet_success_criteria": [],
                }
            return {
                "parsed": schema.model_validate(payload),
                "raw": AIMessage(content=json.dumps(payload)),
                "parsing_error": None,
            }

        return RunnableLambda(respond)


class PlanningHandoffPublicationTests(unittest.IsolatedAsyncioTestCase):
    async def test_completed_report_cannot_omit_required_artifact(self):
        packet = StepReviewPacket(
            created_at="2026-09-04T00:00:00+00:00",
            task_contract=ReviewTaskContract(
                user_request="Download the HTML.",
                plan_objective="Deliver an HTML file.",
                step_id=1,
                step_assignment="Download one HTML file.",
                success_criteria=["The HTML is downloaded."],
                artifact_outputs=[
                    StepArtifactOutput(
                        output_id="requested_html",
                        description="The requested HTML file.",
                        disposition="USER_DELIVERABLE",
                        target_path="downloads/template.html",
                    )
                ],
            ),
            attempts=[
                ReviewAttempt(
                    attempt=1,
                    submission_source="WORKER",
                    finish_reason="READY_FOR_REVIEW",
                    stop_reason="Worker finished.",
                    summary="No artifact was submitted.",
                    final_conclusion="Claimed completion without a file.",
                )
            ],
        )
        report = StepReport(
            step_id=1,
            status="COMPLETED",
            summary="Claimed complete.",
            stop_reason="Reporter finished.",
        )
        with TemporaryDirectory() as temporary:
            checked, receipts = _publish_step_handoff(
                report=report,
                packet=packet,
                run_id="event-required-artifact",
                run_storage_root=Path(temporary) / "runs",
            )

        self.assertEqual(checked.status, "FAILED")
        self.assertEqual(receipts, [])
        self.assertIn("required artifacts", checked.errors[0])

    async def test_reporter_selection_is_published_by_harness_once(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "private-source" / "research.md"
            source.parent.mkdir(parents=True)
            source.write_text("reviewed research", encoding="utf-8")
            store = AsyncEventStore(root / "events.sqlite3")
            await store.start()
            try:
                model = PublicationModel()
                independent_reporter = PublicationModel()
                graph = build_planning_graph(
                    role_models={"web_reporter": independent_reporter},
                    simple_model=model,
                    hard_model=model,
                    worker_registry=WorkerAgentRegistry.general_worker_first(
                        ArtifactWorker(source)
                    ),
                    worker_group_coordinator=WorkerGroupCoordinator(store),
                    planning=planning_settings(),
                    model_output_max_tokens=2048,
                    run_storage_root=root / "runs",
                )

                result = await graph.ainvoke(
                    {
                        "context": PlanningContextPack(
                            current_time="now",
                            user_request="Prepare a reusable note.",
                        ),
                        "event_id": "event-publication",
                        "conversation_thread_id": "conversation-publication",
                        "planning_run_id": "planning-publication",
                    },
                    config={"configurable": {"progress_callback": ignore_progress}},
                )

                self.assertIn("StepReport", independent_reporter.schemas)
                self.assertNotIn("StepReport", model.schemas)
                report = result["completed_step_reports"][0]
                self.assertEqual(report.status, "COMPLETED")
                self.assertEqual(len(report.artifacts), 1)
                self.assertTrue(report.artifacts[0].path.startswith("/handoff/"))
                self.assertNotIn("invented", report.artifacts[0].path)
                receipts = result["handoff_publication_receipts"]
                self.assertEqual(len(receipts), 1)
                self.assertEqual(receipts[0]["output_id"], "requested_note")
                self.assertEqual(
                    receipts[0]["disposition"],
                    "USER_DELIVERABLE",
                )
                self.assertEqual(
                    receipts[0]["target_path"],
                    "research/research.md",
                )
                published = Path(receipts[0]["storage_path"])
                self.assertEqual(published.read_text(encoding="utf-8"), "reviewed research")
            finally:
                await store.close()

    async def test_general_file_delivery_also_passes_through_step_reporter(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "private-source" / "general.md"
            source.parent.mkdir(parents=True)
            source.write_text("general artifact", encoding="utf-8")
            store = AsyncEventStore(root / "events.sqlite3")
            await store.start()
            try:
                model = PublicationModel(worker_kind="GENERAL")
                worker = ArtifactWorker(source)
                independent_reporter = PublicationModel()
                graph = build_planning_graph(
                    role_models={"reporter": independent_reporter},
                    simple_model=model,
                    hard_model=model,
                    worker_registry=WorkerAgentRegistry.general_worker_first(worker),
                    worker_group_coordinator=WorkerGroupCoordinator(store),
                    planning=planning_settings(),
                    model_output_max_tokens=2048,
                    run_storage_root=root / "runs",
                )

                result = await graph.ainvoke(
                    {
                        "context": PlanningContextPack(
                            current_time="now",
                            user_request="Prepare a reusable note.",
                        ),
                        "event_id": "event-general-publication",
                        "conversation_thread_id": "conversation-general-publication",
                        "planning_run_id": "planning-general-publication",
                    },
                    config={"configurable": {"progress_callback": ignore_progress}},
                )

                self.assertIn("StepReport", independent_reporter.schemas)
                self.assertNotIn("StepReport", model.schemas)
                self.assertEqual(result["completed_step_reports"][0].status, "COMPLETED")
                self.assertEqual(len(result["handoff_publication_receipts"]), 1)
                self.assertEqual(worker.calls, 1)
            finally:
                await store.close()

    async def test_web_blocked_report_does_not_create_automatic_worker_retry(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "private-source" / "blocked.md"
            source.parent.mkdir(parents=True)
            source.write_text("candidate", encoding="utf-8")
            store = AsyncEventStore(root / "events.sqlite3")
            await store.start()
            try:
                model = PublicationModel(step_status="BLOCKED")
                worker = ArtifactWorker(source)
                graph = build_planning_graph(
                    simple_model=model,
                    hard_model=model,
                    worker_registry=WorkerAgentRegistry.general_worker_first(worker),
                    worker_group_coordinator=WorkerGroupCoordinator(store),
                    planning=planning_settings(max_step_attempts=2),
                    model_output_max_tokens=2048,
                    run_storage_root=root / "runs",
                )

                result = await graph.ainvoke(
                    {
                        "context": PlanningContextPack(
                            current_time="now",
                            user_request="Prepare a reusable note.",
                        ),
                        "event_id": "event-web-blocked",
                        "conversation_thread_id": "conversation-web-blocked",
                        "planning_run_id": "planning-web-blocked",
                    },
                    config={"configurable": {"progress_callback": ignore_progress}},
                )

                self.assertEqual(worker.calls, 1)
                self.assertEqual(result["completed_step_reports"][0].status, "BLOCKED")
                self.assertEqual(result["current_step_attempt"], 1)
            finally:
                await store.close()


if __name__ == "__main__":
    unittest.main()
