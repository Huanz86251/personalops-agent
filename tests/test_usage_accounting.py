import asyncio
import json
from uuid import uuid4
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import LLMResult, ChatGeneration
from tests.test_runtime_tracing import TraceTests
from observability import trace_span
from trace_callbacks import RuntimeTraceCallback
from runtime_tracing import operation
from usage_accounting import normalize_usage


class UsageTests(TraceTests):
    def emit(self, role, usage=None, error=False):
        callback, key = RuntimeTraceCallback(), uuid4()
        callback.on_chat_model_start({}, [[HumanMessage(content="synthetic")]], run_id=key,
                                     metadata={"runtime.model_role": role})
        if error:
            callback.on_llm_error(RuntimeError("synthetic timeout"), run_id=key)
        else:
            response = LLMResult(generations=[[ChatGeneration(message=AIMessage(content="synthetic", usage_metadata=usage))]])
            callback.on_llm_end(response, run_id=key)
            callback.on_llm_end(response, run_id=key)

    async def test_subtree_control_parallel_rounds_and_missing(self):
        @operation("Code Reviewer", fields=("state",))
        async def review(state):
            await asyncio.sleep(.001)
            self.emit("code_reviewer", {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
                "input_token_details": {"cache_read": 60}, "output_token_details": {"reasoning": 10}})
        with trace_span("Scheduler"):
            self.emit("scheduler", {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12})
            await asyncio.gather(*(review({"step_id": 1, "attempt_id": "a", "candidate_revision": n}) for n in (1, 2)))
            self.emit("final_reviewer", error=True)
        spans = self.exporter.get_finished_spans()
        root = next(s for s in spans if s.name == "Scheduler")
        report = json.loads(root.attributes["usage.summary_json"])
        self.assertEqual(report["subtree"]["total_tokens"], 252)
        self.assertEqual(report["subtree"]["requests"], 4)
        self.assertEqual(report["subtree"]["total_tokens_missing_requests"], 1)
        self.assertEqual(report["subtree"]["status"], "partial")
        self.assertIsNone(report["subtree"]["cache_hit_ratio"])
        self.assertEqual(report["scheduler_control"]["total_tokens"], 12)
        self.assertEqual(report["code_review"]["total_tokens"], 240)
        self.assertEqual(len(report["by"]["code_review_round"]), 2)
        self.assertEqual(report["requests_in_completion_order"][-1]["cumulative_known_total_tokens"], 252)
        self.assertNotIn("llm.token_count.total", root.attributes)
        for span in [s for s in spans if s.name == "Code Reviewer"]:
            detail = json.loads(span.attributes["usage.summary_json"])
            self.assertEqual(detail["subtree"]["requests"], 1)
            self.assertEqual(detail["subtree"]["cache_hit_ratio"], .6)
        self.assertEqual(sum(s.attributes.get("llm.token_count.total", 0) for s in spans), 252)

    def test_raw_metadata_usage_and_explicit_zero(self):
        message = AIMessage(content="x", response_metadata={"token_usage": {
            "prompt_tokens": 100, "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 3}}})
        usage = normalize_usage(None, [message])
        self.assertEqual(usage["total_tokens"], 110)
        self.assertEqual(usage["cache_read_tokens"], 0)
        self.assertEqual(usage["reasoning_tokens"], 3)
        self.assertIsNone(usage["cache_creation_tokens"])

    def test_independent_tasks_do_not_accumulate(self):
        for _ in range(2):
            with trace_span("Task"):
                self.emit("code", {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3})
        for span in [s for s in self.exporter.get_finished_spans() if s.name == "Task"]:
            self.assertEqual(json.loads(span.attributes["usage.summary_json"])["subtree"]["total_tokens"], 3)
