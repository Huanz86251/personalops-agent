"""Tests for the bounded, independent Step Reporter context."""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from planning_models import PlanStep
from reporting import (
    build_step_review_packet,
    materialize_worker_review_trace,
)
from step_execution import _reporter_evidence_manifest, run_step_reporter
from reporting.criteria import normalize_report
from reporting.materials import build_material_reader
from workers.submission import (
    ResolvedToolEvidence,
    WorkerCriterionClaim,
    WorkerSubmission,
    WorkerSubmissionRecord,
)


class FakeStructuredReporter:
    def __init__(self, parsed):
        self.parsed = parsed
        self.messages = []

    async def ainvoke(self, messages):
        self.messages.append(messages)
        return {
            "parsed": self.parsed,
            "parsing_error": None,
        }


class FakeReporterModel:
    def __init__(self, parsed):
        self.structured = FakeStructuredReporter(parsed)

    def with_structured_output(self, schema, **kwargs):
        return self.structured


def sample_step() -> PlanStep:
    return PlanStep(
        step_id=1,
        objective="Find one useful cinema source.",
        success_criteria=[
            "Find one cinema source with tool evidence."
        ],
    )


def sample_submission_record() -> WorkerSubmissionRecord:
    return WorkerSubmissionRecord(
        submitted_at=datetime.now(timezone.utc),
        worker_id="worker-a",
        event_id="event-a",
        step_id="1",
        total_tool_calls=2,
        submission=WorkerSubmission(
            summary="Found one cinema source.",
            final_conclusion="The source satisfies the Step.",
            criterion_claims=[
                WorkerCriterionClaim(
                    criterion=(
                        "Find one cinema source with tool evidence."
                    ),
                    conclusion="One source was found.",
                    evidence_tool_call_ids=["call-1"],
                )
            ],
        ),
        resolved_evidence=[
            ResolvedToolEvidence(
                tool_call_id="call-1",
                tool_name="web_search",
                arguments={"query": "cinema"},
                result='{"title":"Cinema","url":"https://example.com"}',
                result_chars=52,
            )
        ],
    )


class StepReviewPacketTests(unittest.TestCase):
    def test_reporter_manifest_keeps_every_registered_evidence_row(self) -> None:
        record = sample_submission_record().model_copy(deep=True)
        record.resolved_evidence = [
            ResolvedToolEvidence(
                tool_call_id=f"call-{index}",
                tool_name="cinema_showtimes",
                arguments={"cinema": index},
                result=(f"影院{index}: " + "x" * 1400),
                result_chars=1405,
            )
            for index in range(1, 7)
        ]
        packet = build_step_review_packet(
            user_request="查六家影院",
            plan_objective="核对六家影院的排片",
            current_step=sample_step(),
            current_attempt={
                "attempt": 1,
                "worker_submission": record.model_dump(mode="json"),
                "finish_reason": "READY_FOR_REVIEW",
            },
            stop_reason="Worker submitted.",
        )
        refs = {f"call-{index}": f"E{index}" for index in range(1, 7)}
        manifest = _reporter_evidence_manifest(packet, refs)

        self.assertEqual([row["reference"] for row in manifest], [f"E{i}" for i in range(1, 7)])
        self.assertTrue(all(len(row["preview"]) <= 700 for row in manifest))
        self.assertTrue(all(row["preview_source"] == "HARNESS_DIRECT_TOOL_RESULT" for row in manifest))
        self.assertTrue(all(row["preview_truncated"] for row in manifest))
        self.assertTrue(all(row["full_text_available_via_read_review_materials"] for row in manifest))

    def test_durable_review_trace_excludes_messages_and_bounds_evidence(self) -> None:
        record = sample_submission_record().model_dump(mode="json")
        record["resolved_evidence"][0]["result"] = "x" * 7000
        record["resolved_evidence"][0]["result_chars"] = 7000
        durable = materialize_worker_review_trace(
            {
                "assignment_key": "primary",
                "attempt": 1,
                "messages": ["private raw trajectory"],
                "worker_leadership_decisions": ["not needed by Reporter"],
                "worker_submission": record,
            }
        )

        self.assertNotIn("messages", durable)
        self.assertNotIn("worker_leadership_decisions", durable)
        result = durable["worker_submission"]["resolved_evidence"][0]["result"]
        self.assertLessEqual(len(result), 6000)
        self.assertIn("middle omitted", result)

    def test_parallel_packet_keeps_assignment_identity(self) -> None:
        step = PlanStep(
            step_id=1,
            objective="Compare two sources.",
            success_criteria=["Return both findings."],
            worker_kind="WEB",
            execution_mode="PARALLEL",
            worker_assignments=[
                {"assignment_key": "first", "objective": "Find source A."},
                {"assignment_key": "second", "objective": "Find source B."},
            ],
        )
        packet = build_step_review_packet(
            user_request="Compare sources.",
            plan_objective="Return a comparison.",
            current_step=step,
            current_attempt={},
            worker_attempts=[
                {
                    "assignment_key": "first",
                    "assignment_objective": "Find source A.",
                    "attempt": 1,
                    "final_answer": "A",
                },
                {
                    "assignment_key": "second",
                    "assignment_objective": "Find source B.",
                    "attempt": 1,
                    "final_answer": "B",
                },
            ],
            stop_reason="Both Workers ended.",
        )

        self.assertEqual(
            [attempt.assignment_key for attempt in packet.attempts],
            ["first", "second"],
        )
        self.assertEqual(
            [attempt.assignment_objective for attempt in packet.attempts],
            ["Find source A.", "Find source B."],
        )

    def test_packet_excludes_raw_and_unreferenced_messages(self) -> None:
        packet = build_step_review_packet(
            user_request="Find a cinema.",
            plan_objective="Return verified cinema information.",
            current_step=sample_step(),
            current_attempt={
                "attempt": 1,
                "worker_submission": (
                    sample_submission_record().model_dump(mode="json")
                ),
                "messages": [
                    {
                        "role": "tool",
                        "content": "UNREFERENCED_PRIVATE_TRACE",
                    }
                ],
                "execution_summary": {
                    "model_call_count": 2,
                    "tool_call_count": 2,
                    "timeline": ["UNREFERENCED_PRIVATE_TIMELINE"],
                },
                "applied_limits": {
                    "model_rounds": 8,
                    "tool_calls": 15,
                },
                "finish_reason": "READY_FOR_REVIEW",
                "stop_reason": "Worker submitted for review.",
            },
            stop_reason="Worker submitted for review.",
        )

        packet_text = json.dumps(
            packet.model_dump(mode="json"),
            ensure_ascii=False,
        )
        self.assertNotIn("UNREFERENCED_PRIVATE_TRACE", packet_text)
        self.assertNotIn("UNREFERENCED_PRIVATE_TIMELINE", packet_text)
        self.assertIn("https://example.com", packet_text)
        self.assertEqual(
            packet.attempts[0].submission_source,
            "WORKER",
        )
        self.assertEqual(
            packet.attempts[0].resolved_evidence[0].tool_call_id,
            "call-1",
        )

    def test_natural_exit_becomes_explicit_harness_fallback(self) -> None:
        packet = build_step_review_packet(
            user_request="Find a cinema.",
            plan_objective="Return verified cinema information.",
            current_step=sample_step(),
            current_attempt={
                "attempt": 1,
                "final_answer": "I stopped without using the submission tool.",
                "finish_reason": "NATURAL_EXIT",
            },
            stop_reason="General Worker ended.",
        )

        attempt = packet.attempts[0]
        self.assertEqual(
            attempt.submission_source,
            "HARNESS_FALLBACK",
        )
        self.assertEqual(attempt.criterion_claims, [])
        self.assertEqual(
            attempt.unresolved_items,
            sample_step().success_criteria,
        )


class StepReporterAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_replan_report_gets_role_specific_recovery_proposal(self) -> None:
        step = sample_step().model_copy(update={"worker_kind": "GENERAL"})
        packet = build_step_review_packet(
            user_request="Find a cinema, or explain the verified limitation.",
            plan_objective="Return verified cinema information.",
            current_step=step,
            current_attempt={
                "attempt": 1,
                "worker_submission": sample_submission_record().model_dump(mode="json"),
                "finish_reason": "READY_FOR_REVIEW",
                "stop_reason": "Worker submitted for review.",
            },
            stop_reason="Worker submitted for review.",
        )
        model = FakeReporterModel(
            {
                "step_id": 1,
                "status": "PARTIAL",
                "summary": "The available result does not satisfy the frozen criterion.",
                "stop_reason": "Independent review found a material gap.",
                "criterion_results": [
                    {
                        "criterion": "Find one cinema source with tool evidence.",
                        "status": "NOT_MET",
                        "evidence": [],
                    }
                ],
                "confirmed_results": [],
                "completed_work": ["Reviewed the submitted result."],
                "artifacts": [],
                "approved_artifact_refs": [],
                "worker_contributions": [],
                "evidence": [],
                "errors": [],
                "unresolved_items": ["No accepted source evidence."],
                "next_action": "Ask Scheduler to choose a safe fallback.",
                "request_replan": True,
                "replan_reason": "The original path did not produce accepted evidence.",
            }
        )
        proposal = {
            "proposal_kind": "GENERAL",
            "reason": "No accepted source evidence is available.",
            "preserved_results": [],
            "downgrade_recommended": False,
            "downgrade_reason": None,
            "proposed_degraded_outcome": None,
            "user_visible_disclosure": None,
            "user_confirmation_request": None,
            "affected_criterion_ids": ["C1"],
            "blocked_requirements": ["Find one verified source."],
            "external_state_impact": "No external state change.",
            "preserved_safety_boundaries": [],
        }
        recovery_call = AsyncMock(return_value=(proposal, 1, None))

        with patch("review_recovery.build_recovery_proposal", recovery_call):
            result = await run_step_reporter(
                model,
                current_step=step,
                review_packet=packet,
                max_model_rounds=2,
                model_output_max_tokens=1024,
            )

        self.assertEqual(result.report.recovery_proposal, proposal)
        self.assertEqual(result.model_rounds_used, 2)
        recovery_call.assert_awaited_once()
        self.assertEqual(recovery_call.await_args.kwargs["worker_kind"], "GENERAL")

    async def test_material_reader_accepts_public_evidence_alias(self) -> None:
        packet = build_step_review_packet(
            user_request="Find a cinema.",
            plan_objective="Return verified cinema information.",
            current_step=sample_step(),
            current_attempt={
                "attempt": 1,
                "worker_submission": sample_submission_record().model_dump(mode="json"),
                "finish_reason": "READY_FOR_REVIEW",
                "stop_reason": "Worker submitted for review.",
            },
            stop_reason="Worker submitted for review.",
        )
        reader, _ = build_material_reader(
            packet,
            {"E1": "evidence:call-1"},
        )

        result = await reader.ainvoke({
            "requests": [{"reference": "E1", "offset": 0}],
        })

        self.assertIn("Cinema", result)

    def test_material_alias_is_normalized_without_business_retry(self) -> None:
        report = normalize_report(
            {
                "step_id": 1,
                "status": "COMPLETED",
                "summary": "Verified.",
                "stop_reason": "Review complete.",
                "criterion_results": [
                    {"criterion_id": "C1", "evidence": ["M1"], "status": "MET"}
                ],
                "confirmed_results": ["Verified one source."],
                "completed_work": ["Reviewed evidence."],
                "artifacts": [],
                "approved_artifact_refs": [],
                "worker_contributions": [],
                "evidence": ["M1"],
                "errors": [],
                "unresolved_items": [],
                "request_replan": False,
                "replan_reason": None,
            },
            sample_step().success_criteria,
            {"call-1": "E1"},
            aliases={"M1": "E1"},
        )

        self.assertEqual(report.status, "COMPLETED")
        self.assertEqual(report.evidence, ["call-1"])
        self.assertEqual(report.criterion_results[0].evidence, ["call-1"])

    async def test_reporter_reads_packet_and_returns_planner_handoff(self) -> None:
        packet = build_step_review_packet(
            user_request="Find a cinema.",
            completion_api_contract="PUBLIC-COMPLETE-TASK-DOC",
            plan_objective="Return verified cinema information.",
            current_step=sample_step(),
            current_attempt={
                "attempt": 1,
                "worker_submission": (
                    sample_submission_record().model_dump(mode="json")
                ),
                "finish_reason": "READY_FOR_REVIEW",
                "stop_reason": "Worker submitted for review.",
            },
            stop_reason="Worker submitted for review.",
        )
        model = FakeReporterModel(
            {
                "step_id": 1,
                "status": "COMPLETED",
                "summary": (
                    "The Worker found one cinema source and cited the "
                    "corresponding Web Search result."
                ),
                "stop_reason": "Independent review completed.",
                "criterion_results": [
                    {
                        "criterion": (
                            "Find one cinema source with tool evidence."
                        ),
                        "status": "MET",
                        "evidence": ["E1"],
                    }
                ],
                "confirmed_results": ["One cinema source was confirmed."],
                "completed_work": ["Searched for one cinema source."],
                "artifacts": [],
                "approved_artifact_refs": [],
                "worker_contributions": [
                    {
                        "worker_id": "worker-a",
                        "contribution": "Found and cited the source.",
                    }
                ],
                "evidence": ["call-1"],
                "errors": [],
                "unresolved_items": [],
                "next_action": "Continue to the next planned Step.",
                "request_replan": False,
                "replan_reason": None,
            }
        )

        result = await run_step_reporter(
            model,
            current_step=sample_step(),
            review_packet=packet,
            max_model_rounds=2,
            model_output_max_tokens=1024,
        )

        self.assertEqual(result.report.status, "COMPLETED")
        self.assertEqual(
            result.report.worker_contributions[0].worker_id,
            "worker-a",
        )
        prompt = "\n".join(m["content"] for m in model.structured.messages[0])
        self.assertEqual(sum(m["content"] == "PUBLIC-COMPLETE-TASK-DOC"
                             for m in model.structured.messages[0]), 1)
        self.assertNotIn("call-1", model.structured.messages[0][0]["content"])
        self.assertIn("StepReviewPacket", prompt)
        self.assertNotIn("call-1", prompt)
        self.assertIn('"tool_call_id": "E1"', prompt)
        self.assertNotIn("current_step_trace", prompt)
        contracts = [m["content"] for m in model.structured.messages[0]
                     if m["content"].startswith("原始验收契约")]
        self.assertEqual(len(contracts), 1)
        for criterion in sample_step().success_criteria:
            self.assertIn(criterion, contracts[0])

    async def test_web_result_reference_reaches_reporter_without_becoming_e_number(self) -> None:
        record = sample_submission_record().model_copy(deep=True)
        record.submission.criterion_claims[0].evidence_tool_call_ids = ["WA2"]
        result_text = json.dumps({
            "source_ref": "WA2",
            "title": "Cinema result two",
            "href": "https://example.com/two",
            "body": "Observed listing",
        })
        record.resolved_evidence = [ResolvedToolEvidence(
            tool_call_id="WA2",
            tool_name="web_search",
            arguments={
                "source_ref": "WA2",
                "search_arguments": {"query": "cinema"},
            },
            result=result_text,
            result_chars=len(result_text),
        )]
        packet = build_step_review_packet(
            user_request="Find a cinema.",
            plan_objective="Return verified cinema information.",
            current_step=sample_step(),
            current_attempt={
                "attempt": 1,
                "worker_submission": record.model_dump(mode="json"),
                "finish_reason": "READY_FOR_REVIEW",
                "stop_reason": "Worker submitted for review.",
            },
            stop_reason="Worker submitted for review.",
        )
        model = FakeReporterModel({
            "step_id": 1,
            "status": "COMPLETED",
            "summary": "The exact cited Web Search row supports the criterion.",
            "stop_reason": "Independent review completed.",
            "criterion_results": [{
                "criterion": "Find one cinema source with tool evidence.",
                "status": "MET",
                "evidence": ["WA2"],
            }],
            "confirmed_results": ["One source was confirmed."],
            "completed_work": ["Reviewed the cited Web Search row."],
            "artifacts": [],
            "approved_artifact_refs": [],
            "worker_contributions": [{
                "worker_id": "worker-a",
                "contribution": "Found and cited the source.",
            }],
            "evidence": ["WA2"],
            "errors": [],
            "unresolved_items": [],
            "next_action": "Continue to the next planned Step.",
            "request_replan": False,
            "replan_reason": None,
        })

        reviewed = await run_step_reporter(
            model,
            current_step=sample_step(),
            review_packet=packet,
            max_model_rounds=2,
            model_output_max_tokens=1024,
        )

        self.assertEqual(reviewed.report.evidence, ["WA2"])
        self.assertEqual(reviewed.report.criterion_results[0].evidence, ["WA2"])
        prompt = "\n".join(m["content"] for m in model.structured.messages[0])
        self.assertIn('"tool_call_id": "WA2"', prompt)
        self.assertNotIn('"tool_call_id": "E1"', prompt)


if __name__ == "__main__":
    unittest.main()
