"""Observable report/consumer contracts, not a model-quality evaluation."""
import copy
import json
import unittest

from planning_models import StepReport
from reporting import build_step_review_packet
from scheduler_runtime import SchedulerConversation, record_graph_facts
from step_execution import _report_matches_step, run_step_reporter
from tests.test_step_reporter_agent import FakeReporterModel, sample_step, sample_submission_record


def report_payload(criterion_status="MET", status="COMPLETED"):
    return dict(step_id=1, status=status, summary="Public source reviewed.",
                stop_reason="Submitted evidence reviewed.", criterion_results=[dict(
                    criterion=sample_step().success_criteria[0], status=criterion_status,
                    evidence=["call-1"])], confirmed_results=["Public source exists."],
                evidence=["call-1"])


class ReportDecisionTests(unittest.TestCase):
    def test_completed_rejects_each_unfinished_criterion(self):
        for status in ("UNKNOWN", "PARTIAL", "NOT_MET"):
            with self.subTest(status=status):
                self.assertFalse(_report_matches_step(StepReport(**report_payload(status)), sample_step()))

    def test_completed_keeps_recovered_tool_errors(self):
        payload = report_payload()
        payload["errors"] = ["Earlier fetch returned an empty shell; alternate source supplied evidence."]
        self.assertTrue(_report_matches_step(StepReport(**payload), sample_step()))

    def test_scheduler_receives_each_report_field_without_worker_history(self):
        payload = report_payload("UNKNOWN", "PARTIAL")
        payload.update(completed_work=["Read an alternate reference quote."],
                       errors=["Target page was a shell."], unresolved_items=["Target seller live price."],
                       next_action="Retain reference quote; verify exact seller and SKU through the observed public item link.",
                       request_replan=True, replan_reason="Original path cannot establish current seller price.",
                       worker_contributions=[dict(worker_id="worker-a", contribution="Located alternate quote.")])
        report = StepReport(**payload)
        session = SchedulerConversation({}, None, threshold=1)
        record_graph_facts(session, {"completed_step_reports": [report.model_dump(mode="json")]})
        session.compact()
        delivered = json.loads(session.wire()[0]["content"])["StepReport"]
        self.assertEqual(delivered, report.model_dump(mode="json"))
        self.assertEqual(set(delivered), set(StepReport.model_fields))
        self.assertNotIn("messages", delivered)

    def test_targeted_tool_evidence_survives_bounded_packet(self):
        record = sample_submission_record().model_dump(mode="json")
        marker = "TARGET_MODEL reference_price=269 CNY; not a live seller quote"
        def packet_for(body):
            current = copy.deepcopy(record)
            current["resolved_evidence"][0].update(result=body, result_chars=len(body))
            return build_step_review_packet(user_request="Read a reference quote", plan_objective="Reference only",
                current_step=sample_step(), current_attempt={"attempt": 1, "worker_submission": current,
                "messages": ["PRIVATE_WORKER_HISTORY"]}, stop_reason="Submitted")
        whole = packet_for("navigation " * 400 + marker + " footer" * 700)
        self.assertNotIn(marker, whole.attempts[0].resolved_evidence[0].result)
        targeted = packet_for(marker)
        self.assertIn(marker, targeted.attempts[0].resolved_evidence[0].result)
        self.assertNotIn("PRIVATE_WORKER_HISTORY", targeted.model_dump_json())


class ReportRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_inconsistent_completion_cannot_reach_scheduler_as_completed(self):
        packet = build_step_review_packet(user_request="Find a source", plan_objective="Find source",
            current_step=sample_step(), current_attempt={"attempt": 1,
            "worker_submission": sample_submission_record().model_dump(mode="json")}, stop_reason="Submitted")
        result = await run_step_reporter(FakeReporterModel(report_payload("UNKNOWN")),
            current_step=sample_step(), review_packet=packet, max_model_rounds=2,
            model_output_max_tokens=2048, skill_mode="off")
        self.assertNotEqual(result.report.status, "COMPLETED")


if __name__ == "__main__":
    unittest.main()
