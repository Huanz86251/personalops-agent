import unittest
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from reporting.context import _tool_audit, materialize_worker_review_trace, build_step_review_packet
from planning_models import PlanStep

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
