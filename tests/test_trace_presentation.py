import json
import os
import asyncio
from unittest.mock import patch
from uuid import uuid4
import httpx
from langchain_core.messages import HumanMessage
from tests import test_runtime_tracing as base
from tests.test_model_roles import defaults
from agent import build_role_model
from model_roles import load_role_models
from trace_callbacks import RuntimeTraceCallback, CALLBACK
from trace_presentation import attach_audit_callbacks, run_name
from observability import trace_span
from evals.appworld.adapter import UsageMeter


class PresentationTests(base.TraceTests):
    async def test_inherited_callback_budget_rejection_closes_without_usage(self):
        settings = defaults()
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY":"offline", "OPENAI_API_KEY":"offline", "DEEPSEEK_API_KEY":"offline"}):
            settings.role_models = load_role_models(settings)
        for asynchronous in (False, True):
            model = build_role_model(settings, "code")
            meter = UsageMeter(max_calls=0)
            attach_audit_callbacks(model, meter)
            active = set(CALLBACK.spans)
            with trace_span("Rejected Task"):
                with patch.object(type(model), "_generate", side_effect=AssertionError("Provider must not run")), patch.object(type(model), "_agenerate", side_effect=AssertionError("Provider must not run")):
                    from evals.appworld.adapter import EvaluationBudgetExceeded
                    with self.assertRaises(EvaluationBudgetExceeded):
                        if asynchronous:
                            await model.ainvoke("fixture", config={"callbacks":[CALLBACK]})
                        else:
                            model.invoke("fixture", config={"callbacks":[CALLBACK]})
            self.assertEqual(set(CALLBACK.spans), active)
            self.assertFalse(meter.started)
            model.root_client._client.close()
            await model.root_async_client._client.aclose()
        spans = self.exporter.get_finished_spans()
        rejected = [s for s in spans if s.name == "Request / Rejected"]
        self.assertEqual(len(rejected), 2)
        self.assertTrue(all(s.attributes["usage.status"] == "not_sent" for s in rejected))
        self.assertTrue(all(not s.attributes["request.sent"] for s in rejected))
        self.assertTrue(all("usage.summary_json" not in s.attributes for s in spans if s.name == "Rejected Task"))

    def test_all_roles_retain_meter_and_canonical_and_budget(self):
        settings = defaults()
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "offline", "OPENAI_API_KEY": "offline", "DEEPSEEK_API_KEY": "offline"}):
            settings.role_models = load_role_models(settings)
        requests = []
        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"id":"fixture", "object":"chat.completion", "created":1,"model":"fixture",
                "choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"OK"}}],
                "usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}})
        role_count = len(settings.role_models)
        meter = UsageMeter(max_calls=role_count)
        with trace_span("Audit"):
            for role in settings.role_models:
                model = build_role_model(settings, role)
                attach_audit_callbacks(model, meter)
                attach_audit_callbacks(model, meter)
                self.assertEqual(model.callbacks.count(CALLBACK), 1)
                self.assertEqual(model.callbacks.count(meter), 1)
                self.assertIs(model.callbacks[0], meter)
                model.tags = ["appworld:"+role]
                model.root_client._client.close()
                with httpx.Client(transport=httpx.MockTransport(handler)) as client:
                    model.root_client._client = client
                    model.invoke("offline")
                    if len(requests) == role_count:
                        with self.assertRaises(Exception):
                            model.invoke("budget rejection")
        self.assertEqual(len(requests), role_count)
        self.assertEqual(len(meter.records), role_count)
        spans = self.exporter.get_finished_spans()
        leaves = [s for s in spans if s.name.startswith("LLM /")]
        self.assertEqual(len(leaves), role_count)
        root = next(s for s in spans if s.name == "Audit")
        self.assertEqual(json.loads(root.attributes["usage.summary_json"])["subtree"]["total_tokens"], role_count * 8)

    async def test_timing_and_first_content(self):
        callback, key = RuntimeTraceCallback(), uuid4()
        with trace_span("Pipeline"):
            with trace_span("First"):
                await asyncio.sleep(.005)
            with trace_span("Second"):
                callback.on_chat_model_start({}, [[HumanMessage(content="x")]], run_id=key)
                callback.on_llm_new_token("", run_id=key)
                callback.on_llm_new_token("hello", run_id=key)
                callback.on_llm_error(RuntimeError("fixture"), run_id=key)
        spans = self.exporter.get_finished_spans()
        second = next(s for s in spans if s.name == "Second")
        self.assertGreaterEqual(second.attributes["timing.previous_sibling_gap_ms"], 0)
        leaf = next(s for s in spans if s.name.startswith("LLM /"))
        self.assertEqual(leaf.attributes["timing.first_token_status"], "observed")
        self.assertLessEqual(leaf.attributes["timing.first_token_ms"], leaf.attributes["timing.duration_ms"])
        self.assertNotEqual(run_name("Chain Audit"), run_name("Chain Audit"))
