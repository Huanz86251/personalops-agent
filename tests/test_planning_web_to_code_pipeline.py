"""Offline contract test for WEB fan-out -> report -> CODE APPLIED."""

from __future__ import annotations

import asyncio
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


WEB_CRITERION = "Return three complementary, evidence-backed design directions."


def planning_settings() -> PlanningSettings:
    return PlanningSettings(
        max_steps_per_plan=3,
        max_total_steps=5,
        max_replans=1,
        max_step_model_rounds=15,
        max_step_executor_rounds=12,
        max_step_report_rounds=2,
        max_step_tool_calls=15,
        max_step_attempts=2,
        max_plan_model_rounds=30,
        max_plan_tool_calls=30,
        hard_recent_dialogue_turns=4,
        hard_recent_dialogue_max_chars=8000,
        conversation_summary_trigger_turns=8,
        conversation_summary_max_chars=4000,
        executor_summary_trigger_tokens=5000,
        executor_summary_trigger_messages=18,
        executor_summary_keep_messages=10,
    )


class MusicPipelineModel:
    def __init__(self) -> None:
        self.reporter_calls = 0
        self.reporter_prompt = ""
        self.final_prompt = ""

    def with_structured_output(self, schema, **kwargs):
        async def respond(messages):
            prompt = str(messages)
            if schema.__name__ == "SkillChoice":
                return {"parsed": schema.model_validate({"skill_ids": [], "reason": "Offline fixture"}), "raw": None, "parsing_error": None}
            if schema.__name__ == "SupervisorDecision":
                payload = {
                    "action": "PLAN",
                    "plan_objective": (
                        "Research music-course design directions and publish "
                        "a reviewed HTML prototype."
                    ),
                    "plan_success_criteria": [
                        "The research is evidence-backed.",
                        "The reviewed HTML is published.",
                    ],
                    "steps": [
                        {
                            "step_id": 1,
                            "objective": "Research three music-course directions.",
                            "success_criteria": [WEB_CRITERION],
                            "worker_kind": "WEB",
                            "execution_mode": "PARALLEL",
                            "worker_assignments": [
                                {
                                    "assignment_key": "modern",
                                    "objective": "Research modern music courses.",
                                },
                                {
                                    "assignment_key": "classical",
                                    "objective": "Research classical music courses.",
                                },
                                {
                                    "assignment_key": "accessible",
                                    "objective": (
                                        "Research accessible beginner courses."
                                    ),
                                },
                            ],
                            "artifact_outputs": [
                                {
                                    "output_id": "design_reference",
                                    "description": (
                                        "A reviewed cinema-design reference file "
                                        "for the later Code Step."
                                    ),
                                    "disposition": "INTERNAL_HANDOFF",
                                    "required": True,
                                }
                            ],
                        },
                        {
                            "step_id": 2,
                            "objective": (
                                "Use the reviewed research to build index.html."
                            ),
                            "success_criteria": [
                                "A reviewed music-course HTML page is published."
                            ],
                            "worker_kind": "CODE",
                            "execution_mode": "SINGLE",
                            "code_task": {
                                "delivery_mode": "ARTIFACT",
                                "requirements": [
                                    {
                                        "requirement_id": "music_page",
                                        "priority": "MUST",
                                        "statement": (
                                            "Create index.html using the reviewed "
                                            "research directions."
                                        ),
                                    }
                                ],
                                "interfaces": [
                                    {
                                        "interface_id": "html_artifact",
                                        "kind": "FILE_ARTIFACT",
                                        "description": (
                                            "The published standalone HTML page."
                                        ),
                                        "details": {"path": "index.html"},
                                    }
                                ],
                                "validation_expectations": [
                                    "Confirm all three directions are represented."
                                ],
                                "non_goals": ["Do not build a backend."],
                            },
                        },
                    ],
                }
            elif schema.__name__ == "StepReport":
                self.reporter_calls += 1
                self.reporter_prompt = prompt
                for marker in (
                    "modern.example",
                    "classical.example",
                    "accessible.example",
                ):
                    if marker not in prompt:
                        raise AssertionError(
                            f"Reporter did not receive Web evidence: {marker}"
                        )
                artifact_match = re.search(
                    r'\"review_ref\":\s*\"([^\"]+)\"',
                    prompt,
                )
                if artifact_match is None:
                    raise AssertionError(
                        "Reporter did not receive the downloaded reference."
                    )
                payload = {
                    "step_id": 1,
                    "status": "COMPLETED",
                    "summary": (
                        "Three complementary music-course directions were "
                        "reviewed for the implementation step."
                    ),
                    "stop_reason": "ALL_TERMINAL",
                    "criterion_results": [
                        {
                            "criterion_id": "C1",
                            "status": "MET",
                            "evidence": list(dict.fromkeys(re.findall(r'"tool_call_id":\s*"(E\d+)"', prompt))),
                        }
                    ],
                    "confirmed_results": [
                        "Modern direction: modular electronic-music cards.",
                        "Classical direction: repertoire and teacher profiles.",
                        "Accessible direction: clear beginner pathways.",
                    ],
                    "worker_contributions": [
                        {
                            "worker_id": "modern-worker",
                            "contribution": "Modern direction.",
                        },
                        {
                            "worker_id": "classical-worker",
                            "contribution": "Classical direction.",
                        },
                        {
                            "worker_id": "accessible-worker",
                            "contribution": "Accessible direction.",
                        },
                    ],
                    "evidence": list(dict.fromkeys(re.findall(r'"tool_call_id":\s*"(E\d+)"', prompt))),
                    "approved_artifact_refs": [artifact_match.group(1)],
                }
            else:
                self.final_prompt = prompt
                payload = {
                    "action": "FINAL",
                    "status": "COMPLETED",
                    "final_answer": (
                        "The three research directions were reviewed and the "
                        "HTML prototype was published."
                    ),
                    "unmet_success_criteria": [],
                }
            return {
                "parsed": schema.model_validate(payload),
                "raw": AIMessage(content=json.dumps(payload)),
                "parsing_error": None,
            }

        return RunnableLambda(respond)


class ParallelWebRuntime:
    def __init__(self, reference_file: Path) -> None:
        self.reference_file = reference_file
        self.active = 0
        self.max_active = 0
        self.arrived = 0
        self.release = asyncio.Event()

    async def ainvoke(self, input_state, config=None):
        instruction = str(input_state["messages"][-1]["content"])
        assignment = re.search(
            r"assignment_key:\s*(modern|classical|accessible)",
            instruction,
        )
        if assignment is None:
            raise AssertionError("Parallel Worker is missing assignment_key")
        direction = assignment.group(1)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.arrived += 1
        if self.arrived == 3:
            self.release.set()
        try:
            await asyncio.wait_for(self.release.wait(), timeout=1)
            worker_id = input_state["worker_id"]
            call_id = f"search-{direction}"
            url = f"https://{direction}.example/course"
            resolved_artifacts = []
            if direction == "modern":
                payload = self.reference_file.read_bytes()
                resolved_artifacts.append(
                    {
                        "candidate_id": "laixi-design-reference",
                        "output_id": "design_reference",
                        "kind": "DOWNLOADED_FILE",
                        "description": (
                            "Downloaded reference for the cinema poster."
                        ),
                        "verified": True,
                        "location": str(self.reference_file),
                        "storage_path": str(self.reference_file),
                        "size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
            return {
                "messages": [AIMessage(content=f"Found {url}")],
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
                        "summary": f"Researched the {direction} direction.",
                        "final_conclusion": f"Use the {direction} direction.",
                        "criterion_claims": [
                            {
                                "criterion": WEB_CRITERION,
                                "conclusion": f"Found {direction} evidence.",
                                "evidence_tool_call_ids": [call_id],
                            }
                        ],
                        "artifact_candidates": [],
                        "unresolved_items": [],
                    },
                    "resolved_evidence": [
                        {
                            "tool_call_id": call_id,
                            "tool_name": "web_search",
                            "arguments": {"query": f"{direction} music course"},
                            "result": json.dumps(
                                {"results": [{"title": direction, "href": url}]}
                            ),
                            "result_chars": 120,
                        }
                    ],
                    "resolved_artifacts": resolved_artifacts,
                },
            }
        finally:
            self.active -= 1


class ResearchAwareCodeRuntime:
    def __init__(self) -> None:
        self.instruction = ""

    async def ainvoke(self, input_state, config=None):
        self.instruction = str(input_state["messages"][-1]["content"])
        for marker in (
            "Modern direction: modular electronic-music cards.",
            "Classical direction: repertoire and teacher profiles.",
            "Accessible direction: clear beginner pathways.",
        ):
            if marker not in self.instruction:
                raise AssertionError(
                    f"Code Worker did not receive prior StepReport: {marker}"
                )
        if "/handoff/" not in self.instruction:
            raise AssertionError(
                "Code Worker did not receive the reviewed handoff path."
            )
        if "laixi-cinema-reference.md" not in self.instruction:
            raise AssertionError(
                "Code Worker did not receive the expected reference filename."
            )
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
            summary="Reviewer published the researched music-course page.",
            verification_summary=(
                "The page contains modern, classical, and beginner sections."
            ),
            verified_requirement_ids=("music_page",),
            check_results=(
                CodeCheckResult(
                    check_id="three-directions",
                    description="Check all researched directions are present.",
                    status="PASSED",
                    summary="All three directions are present in index.html.",
                ),
            ),
            changed_files=(
                CodeChangedFile(
                    path="index.html",
                    change_summary="Created the music-course prototype.",
                ),
            ),
            approved_artifact_paths=("index.html",),
            published_artifact_paths=("index.html",),
            delivery_location="D:/workspace",
            publication_id="publication-music-page",
            applied_revision="sha256:music-page",
        )
        loop = finish_code_review(loop, report)
        return {
            "messages": [AIMessage(content=report.summary)],
            "executor_model_calls_used": 3,
            "executor_tool_calls_used": 4,
            "show_all_toolsets_calls_used": 0,
            "code_review_loop": loop.model_dump(mode="json"),
            "code_review_report": report.model_dump(mode="json"),
        }


class WebToCodePipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_web_report_feeds_reviewed_code_step(self):
        with TemporaryDirectory() as temporary:
            store = AsyncEventStore(Path(temporary) / "events.sqlite3")
            await store.start()
            try:
                reference_file = Path(temporary) / "laixi-cinema-reference.md"
                reference_file.write_text(
                    "Reviewed visual references for a Laixi cinema poster.",
                    encoding="utf-8",
                )
                model = MusicPipelineModel()
                web_runtime = ParallelWebRuntime(reference_file)
                code_runtime = ResearchAwareCodeRuntime()
                graph = build_planning_graph(
                    simple_model=model,
                    hard_model=model,
                    worker_registry=WorkerAgentRegistry(
                        {
                            "GENERAL": web_runtime,
                            "WEB": web_runtime,
                            "CODE": code_runtime,
                        }
                    ),
                    worker_group_coordinator=WorkerGroupCoordinator(store),
                    planning=planning_settings(),
                    model_output_max_tokens=4096,
                    web_max_parallelism=3,
                )

                async def ignore_progress(event):
                    return None

                result = await graph.ainvoke(
                    {
                        "context": PlanningContextPack(
                            current_time="now",
                            user_request=(
                                "Research music-course sites and build an HTML page."
                            ),
                        ),
                        "event_id": "event-music-page",
                        "conversation_thread_id": "conversation-music-page",
                        "planning_run_id": "planning-music-page",
                    },
                    config={
                        "configurable": {
                            "progress_callback": ignore_progress,
                        }
                    },
                )

                self.assertEqual(result["final_status"], "COMPLETED")
                self.assertEqual(web_runtime.max_active, 3)
                self.assertEqual(model.reporter_calls, 1)
                self.assertEqual(len(result["handoff_publication_receipts"]), 1)
                self.assertEqual(
                    result["handoff_publication_receipts"][0]["output_id"],
                    "design_reference",
                )
                self.assertEqual(
                    result["handoff_publication_receipts"][0]["disposition"],
                    "INTERNAL_HANDOFF",
                )
                self.assertEqual(len(result["completed_step_reports"]), 2)
                self.assertEqual(
                    [item.status for item in result["completed_step_reports"]],
                    ["COMPLETED", "COMPLETED"],
                )
                self.assertEqual(
                    result["completed_step_reports"][1].artifacts[0].path,
                    "index.html",
                )
                self.assertIn(
                    "publication-music-page",
                    model.final_prompt,
                )
            finally:
                await store.close()


if __name__ == "__main__":
    unittest.main()
