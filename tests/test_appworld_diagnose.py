import json
import unittest
from evals.appworld.diagnose import summarize_trial


class TriageTests(unittest.TestCase):
    def fixture(self):
        return {"metadata": {"trial_id": "fixture", "split": "train", "max_model_calls": 2},
                "status": "agent_returned", "official_task_success": False,
                "agent": {"self_reported_final_status": "COMPLETED", "final_answer": "PRIVATE_ANSWER"},
                "usage": {"model_calls_started": 1, "model_calls_returned": 1,
                          "usage_complete": True, "input_tokens": 10, "output_tokens": 5,
                          "records": [{"input_tokens": 10, "output_tokens": 5}]}}

    def test_self_report_cannot_override_grade_and_content_is_not_exported(self):
        result = summarize_trial(self.fixture(), [{"code": "print('PRIVATE_CODE')",
                    "output": "Documentation says: Execution failed. PRIVATE_DATA"}])
        self.assertFalse(result["official_task_success"])
        self.assertIn("self_report_disagrees_with_official_grade", result["alerts"])
        self.assertEqual(result["tool_failures"], [])
        self.assertNotIn("PRIVATE_", json.dumps(result))

    def test_official_success_is_kept_when_final_reporting_fails(self):
        result = self.fixture()
        result["official_task_success"] = True
        result["agent"]["self_reported_final_status"] = "FAILED"
        report = summarize_trial(result, [])
        self.assertTrue(report["official_task_success"])
        self.assertIn("official_success_but_self_report_incomplete", report["alerts"])

    def test_final_test_tasks_are_refused(self):
        result = self.fixture()
        result["metadata"]["split"] = "test_normal"
        with self.assertRaises(ValueError):
            summarize_trial(result, [])

    def test_infrastructure_failure_remains_unscored_and_usage_unknown(self):
        result = self.fixture()
        result.update(status="infrastructure_error", official_task_success=None, agent={})
        result["usage"] = {"model_calls_started": 1, "usage_complete": False,
                           "input_tokens": None, "output_tokens": None,
                           "records": [{"status": "error", "input_tokens": None, "output_tokens": None}]}
        report = summarize_trial(result, [])
        self.assertIsNone(report["official_task_success"])
        self.assertIsNone(report["complete_input_tokens"])
        self.assertNotIn("official_task_failed", report["alerts"])
        self.assertIn("incomplete_token_usage", report["alerts"])


if __name__ == "__main__":
    unittest.main()
