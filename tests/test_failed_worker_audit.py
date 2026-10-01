"""A crashed Worker must not erase tool calls already in its checkpoint."""

import unittest
from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage

from planning_graph import _audit_from_failed_worker_checkpoint


class CheckpointWorker:
    def __init__(self, messages):
        self.messages = messages
        self.requested_config = None

    async def aget_state(self, config):
        self.requested_config = config
        return SimpleNamespace(values={"messages": self.messages})


class FailedWorkerAuditTests(unittest.IsolatedAsyncioTestCase):
    async def test_recovers_completed_calls_for_reviewer(self):
        worker = CheckpointWorker([
            AIMessage(content="", tool_calls=[{
                "name": "web_search",
                "args": {"query": "Laixi cinema"},
                "id": "search-1",
                "type": "tool_call",
            }]),
            ToolMessage(content="Search returned no results", tool_call_id="search-1"),
        ])

        recovered = await _audit_from_failed_worker_checkpoint(worker, "thread-1")

        self.assertEqual(
            worker.requested_config,
            {"configurable": {"thread_id": "thread-1"}},
        )
        self.assertTrue(recovered["tool_audit_available"])
        self.assertEqual(len(recovered["tool_audit"]), 1)
        self.assertEqual(recovered["tool_audit"][0]["tool_name"], "web_search")
        self.assertEqual(recovered["tool_audit"][0]["status"], "RETURNED")

    async def test_empty_checkpoint_does_not_claim_no_tools_were_used(self):
        recovered = await _audit_from_failed_worker_checkpoint(
            CheckpointWorker([]), "thread-2"
        )
        self.assertEqual(recovered, {})
