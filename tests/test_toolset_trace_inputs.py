import json
import unittest
from toolset_router import ToolsetRouter
from test_toolset_router_cross_encoder import FakeReranker, _spec, _tool
from toolsets import ToolsetRegistry
from test_runtime_tracing import TraceTests

class ToolsetTraceInputs(unittest.IsolatedAsyncioTestCase):
    setUp = TraceTests.setUp
    tearDown = TraceTests.tearDown

    async def test_trace_records_exact_scoring_documents_and_separate_full_profile(self):
        scorer = FakeReranker([0.1, 0.9])
        router = ToolsetRouter(scorer, registry=ToolsetRegistry([_spec("APPWORLD", "appworld_execute")]))
        await router.route("fixture query", [_tool("appworld_execute")])
        span = self.exporter.get_finished_spans()[-1]
        data = json.loads(span.attributes["input.value"])
        self.assertEqual([x["scoring_text"] for x in data["candidate_documents"]], scorer.calls[0][1])
        self.assertEqual(data["routing_task"], scorer.calls[0][0])
        candidate = data["candidate_documents"][1]
        self.assertIn("Misleading negative keyword", candidate["full_profile_for_audit_only"])
        self.assertNotIn("Misleading negative keyword", candidate["scoring_text"])
        self.assertEqual(span.attributes["toolset.scorer_returned_count"], 2)
