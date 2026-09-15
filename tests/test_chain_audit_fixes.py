import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import torch

from agent import build_model, build_hard_model
from memory import MemoryService
from retrieval_models import _restore_gte_position_ids


class ThinkingConfigTests(unittest.TestCase):
    def test_role_switches_are_independent(self):
        for regular, scheduler in [(False, False), (True, False), (False, True), (True, True)]:
            settings = SimpleNamespace(llm_provider="deepseek", llm_model="flash", hard_llm_provider="deepseek", hard_llm_model="pro", cloud_llm_max_tokens=5000, llm_thinking_enabled=regular, scheduler_thinking_enabled=scheduler)
            with patch("agent.init_chat_model", return_value=SimpleNamespace(profile={})) as factory:
                build_model(settings)
                self.assertEqual(factory.call_args.kwargs["extra_body"]["thinking"]["type"], "enabled" if regular else "disabled")
                build_hard_model(settings)
                self.assertEqual(factory.call_args.kwargs["extra_body"]["thinking"]["type"], "enabled" if scheduler else "disabled")


class EmptyMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_store_does_not_embed_query(self):
        store = SimpleNamespace(asearch=AsyncMock(return_value=[]))
        service = MemoryService(store=store, retrieval_models=SimpleNamespace(), model=None, timezone_name="UTC")
        self.assertEqual(await service._dense_retrieve("非空用户请求"), [])
        store.asearch.assert_awaited_once()
        self.assertNotIn("query", store.asearch.call_args.kwargs)

    async def test_nonempty_store_still_runs_semantic_search(self):
        store = SimpleNamespace(asearch=AsyncMock(side_effect=[[object()], []]))
        service = MemoryService(store=store, retrieval_models=SimpleNamespace(), model=None, timezone_name="UTC")
        await service._dense_retrieve("查项目")
        self.assertEqual(store.asearch.await_count, 2)
        self.assertEqual(store.asearch.call_args.kwargs["query"], "查项目")


class PositionBufferTests(unittest.TestCase):
    def test_repairs_only_deterministic_buffer_and_is_idempotent(self):
        class NewEmbeddings(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("position_ids", torch.full((8,), 9999999, dtype=torch.long), persistent=False)
                self.weight = torch.nn.Parameter(torch.ones(2))
        model = torch.nn.Module()
        model.embeddings = NewEmbeddings()
        weights = model.embeddings.weight.detach().clone()
        self.assertEqual(_restore_gte_position_ids(model, "unrelated"), 0)
        self.assertEqual(_restore_gte_position_ids(model, "Alibaba-NLP/gte-multilingual-base"), 1)
        self.assertTrue(torch.equal(model.embeddings.position_ids, torch.arange(8)))
        self.assertTrue(torch.equal(weights, model.embeddings.weight))
        self.assertNotIn("embeddings.position_ids", model.state_dict())
        self.assertEqual(_restore_gte_position_ids(model, "Alibaba-NLP/gte-multilingual-base"), 0)
