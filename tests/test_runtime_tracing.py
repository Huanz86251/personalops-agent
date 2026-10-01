"""Offline trace hierarchy, identities, deduplication and missing-usage checks."""
import asyncio
import json
import unittest
from unittest.mock import patch
from uuid import uuid4
from openinference.instrumentation import OITracer, TraceConfig
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import LLMResult, ChatGeneration
from observability import _curated_span_name, trace_span
from runtime_tracing import operation, ROLE_NAMES
from trace_callbacks import RuntimeTraceCallback
from skill_runtime.preparation import prepare_skills, load_catalog
from model_roles import ROLE_DEFAULTS


class TraceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider()
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.tracer = OITracer(self.provider.get_tracer("offline"), config=TraceConfig())
        self.patch = patch("observability._TRACER", self.tracer)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.provider.shutdown()

    def test_step_reporter_name_exposes_attempt_and_repair_round(self):
        self.assertEqual(
            _curated_span_name("step_report.step_2.attempt_1.repair_3"),
            "Step Reporter / Step 2 / Attempt 1 / Repair 3",
        )

    async def test_parallel_children_keep_their_own_identity(self):
        @operation("Web Agent", fields=("state",), kind="agent")
        async def worker(state):
            await asyncio.sleep(.01)
            callback = RuntimeTraceCallback()
            call = uuid4()
            callback.on_chat_model_start({}, [[HumanMessage(content="fixture")]], run_id=call)
            callback.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="done"))]]), run_id=call)
            return {"status": "COMPLETED"}
        with trace_span("Scheduler"):
            await asyncio.gather(*(worker({"worker_id": str(n), "step_id": 1}) for n in range(3)))
        spans = self.exporter.get_finished_spans()
        root = next(s for s in spans if s.name == "Scheduler")
        workers = [s for s in spans if s.name == "Web Agent"]
        self.assertEqual(len(workers), 3)
        self.assertTrue(all(s.parent.span_id == root.context.span_id for s in workers))
        self.assertLess(max(s.start_time for s in workers), min(s.end_time for s in workers))
        for child in [s for s in spans if s.name.startswith("LLM /")]:
            parent = next(s for s in workers if s.context.span_id == child.parent.span_id)
            self.assertEqual(child.attributes["runtime.worker_id"], parent.attributes["runtime.worker_id"])

    def test_duplicate_callback_is_one_leaf_and_usage_is_not_double_counted(self):
        cb, key = RuntimeTraceCallback(), uuid4()
        with trace_span("Code Worker"):
            for _ in range(2):
                cb.on_chat_model_start({}, [[HumanMessage(content="x")]], run_id=key, metadata={"runtime.model_role": "code"})
            response = LLMResult(generations=[[ChatGeneration(message=AIMessage(content="ok", usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}))]])
            cb.on_llm_end(response, run_id=key)
            cb.on_llm_end(response, run_id=key)
        leaves = [s for s in self.exporter.get_finished_spans() if s.name.startswith("LLM /")]
        self.assertEqual(len(leaves), 1)
        self.assertEqual(
            leaves[0].name,
            "LLM / Code Worker · 01 · I/O 5→3 · Cache n/a",
        )
        self.assertEqual(leaves[0].attributes["llm.token_count.total"], 8)

    def test_llm_leaf_exposes_current_request_cache_metrics(self):
        cb, key = RuntimeTraceCallback(), uuid4()
        cb.on_chat_model_start(
            {},
            [[HumanMessage(content="fixture")]],
            run_id=key,
            metadata={"runtime.model_role": "web"},
        )
        response = LLMResult(
            generations=[[
                ChatGeneration(
                    message=AIMessage(
                        content="done",
                        usage_metadata={
                            "input_tokens": 100,
                            "output_tokens": 20,
                            "total_tokens": 120,
                            "input_token_details": {"cache_read": 75},
                        },
                    )
                )
            ]]
        )
        cb.on_llm_end(response, run_id=key)

        leaf = next(
            span
            for span in self.exporter.get_finished_spans()
            if span.name.startswith("LLM / Web Agent")
        )
        self.assertEqual(leaf.attributes["usage.current.scope"], "single_provider_request")
        self.assertEqual(leaf.attributes["usage.current.input_tokens"], 100)
        self.assertEqual(leaf.attributes["usage.current.cache_read_tokens"], 75)
        self.assertEqual(leaf.attributes["usage.current.uncached_input_tokens"], 25)
        self.assertEqual(leaf.attributes["usage.current.output_tokens"], 20)
        self.assertEqual(leaf.attributes["usage.current.cache_hit_ratio"], .75)
        self.assertEqual(leaf.attributes["usage.current.cache_hit_percent"], 75.0)
        self.assertIn("cache_read=75 (75.0%)", leaf.attributes["usage.current.summary"])
        self.assertEqual(
            leaf.name,
            "LLM / Web Agent · 01 · I/O 100→20 · Cache 75.0%",
        )
        event = next(event for event in leaf.events if event.name == "Current LLM request usage")
        self.assertEqual(event.attributes["usage.current.cache_hit_percent"], 75.0)

    def test_unknown_usage_and_tool_failure_are_explicit(self):
        cb, key = RuntimeTraceCallback(), uuid4()
        cb.on_chat_model_start({}, [[HumanMessage(content="x")]], run_id=key)
        cb.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="ok"))]]), run_id=key)
        leaf = next(s for s in self.exporter.get_finished_spans() if s.attributes.get("openinference.span.kind") == "LLM")
        self.assertEqual(leaf.attributes["usage.status"], "missing")
        self.assertNotIn("llm.token_count.total", leaf.attributes)
        key = uuid4()
        cb.on_tool_start({"name": "fetch_webpage"}, "fixture", run_id=key)
        cb.on_tool_error(RuntimeError("HTTP 403"), run_id=key)
        self.assertEqual(self.exporter.get_finished_spans()[-1].status.status_code.name, "ERROR")

    def test_web_outcome_is_recorded_without_claiming_evidence(self):
        from langchain_core.messages import ToolMessage
        cb, key = RuntimeTraceCallback(), uuid4()
        cb.on_tool_start({"name": "fetch_webpage"}, "fixture", run_id=key)
        cb.on_tool_end(ToolMessage(content=json.dumps({"fetch_status":"EMPTY_CONTENT", "http_status":200,
            "evidence_available":False, "content":""}), tool_call_id="fixture"), run_id=key)
        leaf = self.exporter.get_finished_spans()[-1]
        self.assertEqual(leaf.attributes["business.status"], "EMPTY_CONTENT")
        self.assertFalse(leaf.attributes["web.evidence_available"])
        self.assertEqual(leaf.attributes["http.status_code"], 200)

    def test_progress_interrupt_and_resume_are_one_non_error_span(self):
        from langgraph.errors import GraphInterrupt

        cb = RuntimeTraceCallback()
        parent = uuid4()
        first_run = uuid4()
        resumed_run = uuid4()
        payload = {"progress": {"phase": "researching"}}

        cb.on_tool_start(
            {"name": "publish_worker_progress"},
            payload,
            run_id=first_run,
            parent_run_id=parent,
            inputs=payload,
        )
        cb.on_tool_error(GraphInterrupt(()), run_id=first_run)
        self.assertEqual(self.exporter.get_finished_spans(), ())

        cb.on_tool_start(
            {"name": "publish_worker_progress"},
            payload,
            run_id=resumed_run,
            parent_run_id=parent,
            inputs=payload,
        )
        cb.on_tool_end("continued", run_id=resumed_run)

        leaves = self.exporter.get_finished_spans()
        self.assertEqual(len(leaves), 1)
        self.assertEqual(leaves[0].name, "Worker Progress Gate")
        self.assertEqual(leaves[0].status.status_code.name, "OK")
        self.assertTrue(leaves[0].attributes["control_flow.expected_interrupt"])
        self.assertEqual(
            leaves[0].attributes["worker.progress_gate.transition"],
            "PAUSED_TO_CONTINUED",
        )

    def test_graph_interrupt_is_expected_pause_not_error(self):
        from langgraph.errors import GraphInterrupt

        with self.assertRaises(GraphInterrupt):
            with trace_span("Scheduler / Dispatch Step"):
                raise GraphInterrupt(())

        leaf = self.exporter.get_finished_spans()[-1]
        self.assertEqual(leaf.status.status_code.name, "OK")
        self.assertEqual(leaf.attributes["business.status"], "PAUSED")
        self.assertTrue(leaf.attributes["control_flow.expected_interrupt"])

    async def test_skills_off_fixed_and_reuse_all_have_spans(self):
        catalog = load_catalog()
        off = await prepare_skills(None, role="code", task="fixture", catalog=catalog, mode="off")
        fixed = await prepare_skills(None, role="scheduler", task="fixture", catalog=catalog, mode="fixed", fixed_ids=["plan-task-dependencies"])
        restored = await prepare_skills(None, role="scheduler", task="fixture", saved=fixed.model_dump(mode="json"))
        spans = self.exporter.get_finished_spans()
        self.assertEqual(len(spans), 3)
        self.assertTrue(spans[-1].attributes["skills.reused"])
        data = json.loads(spans[-1].attributes["output.value"])
        self.assertNotIn("content", data["selected"][0])
        self.assertEqual(restored, fixed)
        self.assertEqual(off.model_calls, 0)

    def test_all_configured_model_roles_have_display_names(self):
        self.assertFalse(set(ROLE_DEFAULTS) - set(ROLE_NAMES))

    def test_evaluation_meter_does_not_duplicate_runtime_leaf(self):
        from evals.appworld.adapter import UsageMeter
        meter = UsageMeter(max_calls=2)
        with patch.object(meter, "_start_trace_span") as create:
            meter.on_chat_model_start({}, [[HumanMessage(content="fixture")]], run_id=uuid4(),
                                      metadata={"trace.owner": "personalops"})
            create.assert_not_called()
        self.assertEqual(len(meter.started), 1)

    async def test_cancelled_operation_closes_error_span(self):
        @operation("Web Runtime / Browser Session")
        async def cancelled():
            raise asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await cancelled()
        self.assertEqual(self.exporter.get_finished_spans()[-1].status.status_code.name, "ERROR")
