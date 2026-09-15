"""No-network checks for the curated Phoenix evaluation profile."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from observability import get_phoenix_trace_profile
from scripts.eval_demo import safe_summary


class ObservabilityProfileTests(unittest.TestCase):
    def test_curated_profile_is_selected_explicitly(self):
        with patch.dict(os.environ, {"PHOENIX_TRACE_PROFILE": "curated"}):
            self.assertEqual(get_phoenix_trace_profile(), "curated")

    def test_curated_profile_is_the_application_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PHOENIX_TRACE_PROFILE", None)
            self.assertEqual(get_phoenix_trace_profile(), "curated")

    def test_safe_summary_exposes_metrics_without_private_content(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            result = {
                "metadata": {
                    "trial_id": "aw_fixture",
                    "split": "train",
                    "purpose": "calibration",
                    "phoenix_project": "personalops-eval-train",
                    "phoenix_trace_profile": "curated",
                    "phoenix_trace_id": "abc",
                },
                "status": "agent_returned",
                "official_task_success": True,
                "agent": {"self_reported_final_status": "FAILED"},
                "usage": {
                    "model_calls_started": 2,
                    "usage_complete": True,
                    "input_tokens": 10,
                    "output_tokens": 3,
                },
                "trace_delivery": {"root_persisted": True},
                "elapsed_seconds": 1.5,
            }
            trajectory = [
                {"code": "SECRET_CODE", "output": "SECRET_OUTPUT"},
                {"code": "MORE_SECRET", "output": "Execution failed. SECRET"},
            ]
            path = directory / "result.json"
            path.write_text(json.dumps(result), encoding="utf-8")
            (directory / "trajectory.private.json").write_text(
                json.dumps(trajectory), encoding="utf-8"
            )
            rendered = json.dumps(safe_summary(path))
            self.assertNotIn("SECRET", rendered)
            self.assertIn("official_grade_and_agent_self_report_disagree", rendered)
            self.assertIn('"tool_calls": 2', rendered)
            self.assertIn('"tool_failures": 1', rendered)


if __name__ == "__main__":
    unittest.main()
