import unittest
from datetime import datetime, timezone
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from reporting.context import _tool_audit, materialize_worker_review_trace, build_step_review_packet
from planning_models import PlanStep
from workers.submission import (
    ResolvedToolEvidence,
    WorkerCriterionClaim,
    WorkerSubmission,
    WorkerSubmissionRecord,
    resolve_all_tool_evidence,
)

class AuditTests(unittest.TestCase):
    def test_records_are_not_selected_by_worker(self):
        trace = {"messages": [HumanMessage(content="private conversation must not pass"),
            AIMessage(content="untrusted explanation", tool_calls=[{"id":"x", "name":"appworld_execute", "args":{"code":"check()"}}]),
            ToolMessage(content="Execution failed: missing", tool_call_id="x", status="error")]}
        rows, available = _tool_audit(trace)
        self.assertTrue(available)
        self.assertEqual(rows[0]["status"], "ERROR")
        durable = materialize_worker_review_trace(trace)
        self.assertNotIn("messages", durable)
        self.assertNotIn("private conversation", str(durable))
        self.assertEqual(_tool_audit(durable), (rows, True))
    def test_final_reviewer_request_survives_into_role_reviewer_packet(self):
        request = {
            "step_id": 1,
            "worker_kind": "GENERAL",
            "failed_criterion_ids": ["C2"],
            "evidence_refs": ["E1"],
            "observed_problem": "未观察到完成提交。",
            "missing_requirement": "缺少完成提交回执。",
        }
        trace = {
            "attempt": 1,
            "worker_id": "worker-1",
            "finish_reason": "GENERAL_SELF_REPORT",
            "stop_reason": "done",
            "final_reviewer_request": request,
            "final_answer": "repaired",
        }
        durable = materialize_worker_review_trace(trace)
        self.assertEqual(durable["final_reviewer_request"], request)
        packet = build_step_review_packet(
            user_request="完成任务",
            plan_objective="完成并核验",
            current_step=PlanStep(
                step_id=1,
                worker_kind="GENERAL",
                objective="执行任务",
                success_criteria=["完成任务"],
            ),
            current_attempt=trace,
            stop_reason="done",
        )
        self.assertEqual(packet.attempts[0].final_reviewer_request, request)

    def test_missing_results_and_unavailable_archive(self):
        self.assertEqual(_tool_audit({}), ([], False))
        rows, available = _tool_audit({"messages":[AIMessage(content="", tool_calls=[{"id":"a","name":"probe","args":{}}])]})
        self.assertTrue(available)
        self.assertEqual(rows[0]["status"], "NO_RESULT")

    def test_web_status_is_explicit_not_inferred_from_prose(self):
        trace={"messages":[AIMessage(content="",tool_calls=[{"id":"w","name":"fetch","args":{}}]),
                           ToolMessage(content='{"fetch_status":"ACCESS_DENIED","content":""}',tool_call_id="w")]}
        rows,_=_tool_audit(trace)
        self.assertEqual(rows[0]["fetch_status"], "ACCESS_DENIED")
        self.assertEqual(rows[0]["status"], "RETURNED")
        trace["messages"][-1]=ToolMessage(content="Documentation mentions ValueError and SUCCESS",tool_call_id="w")
        rows,_=_tool_audit(trace)
        self.assertIsNone(rows[0]["fetch_status"])
        self.assertIsNone(rows[0]["error_type"])

    def test_successful_retry_is_visible_even_when_worker_cites_failed_call(self):
        messages = [
            AIMessage(content="", tool_calls=[
                {"id": "failed-map", "name": "maps_text_search", "args": {"keywords": "景点"}},
            ]),
            ToolMessage(
                content="Execution failed: Unknown source_refs: S1",
                tool_call_id="failed-map",
                status="error",
            ),
            AIMessage(content="", tool_calls=[
                {"id": "successful-map", "name": "maps_text_search", "args": {"keywords": "景点"}},
            ]),
            ToolMessage(
                content='{"pois":[{"name":"故宫博物院"}]}',
                tool_call_id="successful-map",
            ),
        ]
        all_evidence = resolve_all_tool_evidence(messages)
        self.assertEqual(
            [item.tool_call_id for item in all_evidence],
            ["failed-map", "successful-map"],
        )

        cited_failure = ResolvedToolEvidence(
            tool_call_id="failed-map",
            tool_name="maps_text_search",
            arguments={"keywords": "景点"},
            result="Execution failed: Unknown source_refs: S1",
            result_chars=41,
        )
        record = WorkerSubmissionRecord(
            submitted_at=datetime.now(timezone.utc),
            worker_id="general-1",
            event_id="event-1",
            step_id="1",
            total_tool_calls=2,
            submission=WorkerSubmission(
                summary="地图查询和文件交付已完成。",
                final_conclusion="完成。",
                criterion_claims=[WorkerCriterionClaim(
                    criterion="查询景点",
                    evidence_tool_call_ids=["failed-map"],
                    conclusion="已查询。",
                )],
            ),
            resolved_evidence=[cited_failure],
        )
        trace = materialize_worker_review_trace({
            "attempt": 1,
            "worker_submission": record.model_dump(mode="json"),
            "review_evidence": [
                item.model_dump(mode="json") for item in all_evidence
            ],
            "messages": messages,
        })
        packet = build_step_review_packet(
            user_request="查询景点",
            plan_objective="查询景点并交付",
            current_step=PlanStep(
                step_id=1,
                worker_kind="GENERAL",
                objective="查询景点",
                success_criteria=["查询景点"],
            ),
            current_attempt=trace,
            stop_reason="done",
        )
        self.assertEqual(
            [item.tool_call_id for item in packet.attempts[0].resolved_evidence],
            ["failed-map", "successful-map"],
        )
        self.assertIn(
            "故宫博物院",
            packet.attempts[0].resolved_evidence[1].result,
        )
