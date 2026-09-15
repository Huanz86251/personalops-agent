"""Offline checks for role isolation and the actual OpenAI request payload."""
import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent import build_summary_model, build_memory_model
from tests.test_general_worker import StaticWorkerModel
from workers.compaction import WorkerCompactionMiddleware


class SummaryModelRoutingTests(unittest.TestCase):
    def test_nano_payload_has_minimal_reasoning_and_completion_limit(self):
        settings = SimpleNamespace(
            summary_llm_provider="openai", summary_llm_model="gpt-5-nano",
            extraction_llm_provider="openai", extraction_llm_model="gpt-5-nano",
            cloud_llm_max_tokens=5000, memory_extraction_max_tokens=16000,
        )
        with patch.dict("os.environ", {"OPENAI_API_KEY": "offline-placeholder"}):
            for factory, limit in [(build_summary_model, 5000), (build_memory_model, 16000)]:
                model = factory(settings)
                payload = model._get_request_payload([("human", "offline")])
                self.assertEqual(payload["model"], "gpt-5-nano")
                self.assertEqual(payload["reasoning_effort"], "minimal")
                self.assertEqual(payload["max_completion_tokens"], limit)
                self.assertNotIn("max_tokens", payload)
                self.assertNotIn("temperature", payload)

    def test_all_workers_keep_inactive_compactor_for_explicit_opt_in(self):
        for role in ["general_worker", "web_worker", "code_worker", "code_reviewer"]:
            with self.subTest(role=role):
                module = importlib.import_module("workers." + role)
                main, summary = StaticWorkerModel(), StaticWorkerModel()
                from tools import ALL_TOOLS
                factory_name = "create_agent" if role == "general_worker" else "create_deep_agent"
                with patch.object(module, factory_name) as create:
                    getattr(module, "create_" + role)(
                        main, summary_model=summary,
                        tools=ALL_TOOLS if role == "web_worker" else [],
                    )
                compactors = [m for m in create.call_args.kwargs["middleware"]
                              if isinstance(m, WorkerCompactionMiddleware)]
                self.assertEqual(len(compactors), 1)
                self.assertIs(compactors[0].model, summary)
                self.assertFalse(compactors[0].enabled)


if __name__ == "__main__":
    unittest.main()
