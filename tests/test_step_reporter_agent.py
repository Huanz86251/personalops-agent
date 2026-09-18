"""Tests for the bounded, independent Step Reporter context."""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone

from planning_models import PlanStep
from reporting import (
    build_step_review_packet,
    materialize_worker_review_trace,
)
from step_execution import run_step_reporter
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


if __name__ == "__main__":
    unittest.main()
