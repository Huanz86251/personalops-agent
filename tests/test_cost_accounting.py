import unittest
from cost_accounting import estimate_cost, cost_attributes, summarize_cost
from test_runtime_tracing import TraceTests
from trace_callbacks import RuntimeTraceCallback
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.outputs import LLMResult, ChatGeneration
from uuid import uuid4

class PriceTests(unittest.TestCase):
    def test_cache_reasoning_and_fx(self):
        c=estimate_cost("qwen3.8-flash",dict(input_tokens=10000,output_tokens=1000,cache_read_tokens=8000,reasoning_tokens=500))
        self.assertAlmostEqual(c["total_cny"],.0051)
        self.assertAlmostEqual(c["total_usd"]*c["cny_per_usd"],.0051)
    def test_glm_bailian_cache_and_reasoning(self):
        c = estimate_cost("ZHIPU/GLM-5.3-Flash", dict(input_tokens=10000,
            output_tokens=1000, cache_read_tokens=8000, reasoning_tokens=500))
        self.assertEqual(c["status"], "estimated")
        self.assertAlmostEqual(c["total_cny"], .00624)
        self.assertEqual(c["verified_on"], "2026-09-11")
        self.assertAlmostEqual(cost_attributes(c)["cost.total_cny"], .00624)
        self.assertEqual(estimate_cost("ZHIPU/GLM-5.3-Flash", {})["status"], "unknown")

    def test_tier_and_missing(self):
        c=estimate_cost("qwen3.7-flash",dict(input_tokens=32001,output_tokens=1000,cache_read_tokens=0))
        self.assertEqual(c["rates_cny_per_million"]["input"],.6)
        self.assertEqual(estimate_cost("unknown",{})["status"],"unknown")
        self.assertEqual(estimate_cost("qwen3.8-flash",dict(input_tokens=10,output_tokens=2))["status"],"unknown")
        self.assertEqual(summarize_cost([c,{}])["missing_requests"],1)

class CallbackPriceTests(unittest.TestCase):
    setUp=TraceTests.setUp
    tearDown=TraceTests.tearDown
    def test_only_llm_leaf_is_priced(self):
        cb=RuntimeTraceCallback();key=uuid4()
        cb.on_chat_model_start({},[[HumanMessage(content="synthetic")]],run_id=key,metadata={"ls_model_name":"qwen3.8-flash"})
        response=LLMResult(generations=[[ChatGeneration(message=AIMessage(content="synthetic",usage_metadata={"input_tokens":10000,"output_tokens":1000,"total_tokens":11000,"input_token_details":{"cache_read":8000}}))]])
        cb.on_llm_end(response,run_id=key)
        spans=self.exporter.get_finished_spans();priced=[s for s in spans if "llm.cost.total" in s.attributes]
        self.assertEqual(len(priced),1)
        self.assertAlmostEqual(priced[0].attributes["cost.total_cny"],.0051)
