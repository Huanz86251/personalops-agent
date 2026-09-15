"""Offline tests for Harness-authored Worker cancellation records."""

import unittest

from agent import ask_worker
from worker_termination import build_worker_cancellation_record


class _CancelledAgent:
    async def ainvoke(self, input_state, config=None):
        return {
            **input_state,
            "worker_terminal_action": "CANCEL",
            "worker_total_tool_calls": 4,
            "worker_progress_reports": [
                {
                    "sequence": 1,
                    "summary": "Collected partial evidence.",
                }
            ],
            "worker_leadership_decisions": [
                {
                    "decision": {
                        "action": "CANCEL",
                        "reason": "The direction is no longer useful.",
                    }
                }
            ],
        }


class WorkerCancellationRecordTests(unittest.IsolatedAsyncioTestCase):
    def test_record_uses_only_existing_state(self):
        record = build_worker_cancellation_record(
            {
                "worker_id": "worker-1",
                "event_id": "event-1",
                "step_id": "1",
                "worker_total_tool_calls": 3,
                "worker_progress_reports": [{"sequence": 1}],
            },
            reason="User cancelled the branch.",
        )
        self.assertEqual(record.total_tool_calls, 3)
        self.assertEqual(record.latest_progress, {"sequence": 1})

    async def test_ask_worker_materializes_cancel_without_second_model(self):
        details = await ask_worker(
            _CancelledAgent(),
            "Run a cancellable branch.",
            thread_id="cancel-thread",
            state_update={
                "worker_id": "worker-cancel",
                "event_id": "event-cancel",
                "step_id": "2",
            },
            return_details=True,
        )
        record = details["worker_cancellation_record"]
        self.assertEqual(record["worker_id"], "worker-cancel")
        self.assertEqual(record["total_tool_calls"], 4)
        self.assertEqual(
            record["reason"],
            "The direction is no longer useful.",
        )


if __name__ == "__main__":
    unittest.main()
