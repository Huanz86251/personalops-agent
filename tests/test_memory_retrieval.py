import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from memory import MEMORY_NAMESPACE, MemoryService
from memory_graph import MemoryGraphIndex
from memory_lexical import MemoryBM25Index, tokenize_memory_text
from retrieval_models import RerankResult


def _memory_value(
    content: str,
    *,
    structured_data: dict | None = None,
    triples: list[dict] | None = None,
    confidence: int = 2,
    importance: int = 2,
) -> dict:
    return {
        "status": "active",
        "content": content,
        "memory_type": "semantic",
        "structured_data": structured_data or {},
        "triples": triples or [],
        "confidence": confidence,
        "importance": importance,
        "valid_from": None,
        "expires_at": None,
    }


class _Store:
    def __init__(self, rows: dict[str, dict]):
        self.rows = rows

    async def aget(self, namespace, key):
        if namespace != MEMORY_NAMESPACE or key not in self.rows:
            return None
        return SimpleNamespace(key=key, value=dict(self.rows[key]))

    async def asearch(self, namespace, **kwargs):
        if namespace != MEMORY_NAMESPACE:
            return []
        offset = kwargs.get("offset", 0)
        limit = kwargs.get("limit", len(self.rows))
        rows = [
            SimpleNamespace(key=key, value=dict(value))
            for key, value in sorted(self.rows.items())
            if value.get("status") == "active"
        ]
        return rows[offset:offset + limit]


class MemoryLexicalIndexTests(unittest.TestCase):
    def test_mixed_chinese_and_latin_tokens_are_normalized(self):
        self.assertEqual(
            tokenize_memory_text("Cafe\u0301 CAFÉ"),
            ["café", "café"],
        )
        self.assertIn("哈莫", tokenize_memory_text("阿哈莫德"))

    def test_bm25_searches_summary_structured_fields_and_triples(self):
        index = MemoryBM25Index()
        index.add_memory(
            "mentor",
            _memory_value(
                "用户有一位长期导师",
                structured_data={
                    "record_type": "person_relation",
                    "person_name": "阿哈莫德",
                    "relation": "mentor",
                },
                triples=[{
                    "subject": "阿哈莫德",
                    "relation": "mentor_of",
                    "object": "用户",
                }],
            ),
        )
        index.add_memory(
            "project",
            _memory_value("PersonalOps 使用 SQLite 保存记忆"),
        )

        self.assertEqual(index.search("阿哈莫德是谁")[0].memory_id, "mentor")
        self.assertEqual(index.search("SQLite")[0].memory_id, "project")

        self.assertTrue(index.remove_memory("mentor"))
        self.assertEqual(index.search("阿哈莫德"), [])


class HybridMemoryRetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_scan_rebuilds_bm25_and_graph_together(self):
        value = _memory_value(
            "阿哈莫德是用户的导师",
            structured_data={"person_name": "阿哈莫德", "relation": "mentor"},
            triples=[{
                "subject": "阿哈莫德",
                "relation": "mentor_of",
                "object": "用户",
            }],
        )
        service = MemoryService(
            store=_Store({"mentor": value}),
            retrieval_models=SimpleNamespace(),
            model=None,
            timezone_name="UTC",
        )

        await service.rebuild_graph_index(page_size=1)

        self.assertEqual(service.lexical_index.stats()["documents"], 1)
        self.assertEqual(service.graph_index.stats()["memories"], 1)

    async def test_bm25_only_hit_enters_cross_encoder_and_graph_seed_stage(self):
        value = _memory_value(
            "阿哈莫德是用户的导师",
            structured_data={
                "record_type": "person_relation",
                "person_name": "阿哈莫德",
                "relation": "mentor",
            },
            triples=[{
                "subject": "阿哈莫德",
                "relation": "mentor_of",
                "object": "用户",
            }],
            confidence=3,
            importance=3,
        )
        retrieval_models = SimpleNamespace(
            arerank=AsyncMock(return_value=[
                RerankResult(index=0, text="阿哈莫德是用户的导师", score=0.91),
            ]),
        )
        service = MemoryService(
            store=_Store({"mentor": value}),
            retrieval_models=retrieval_models,
            model=None,
            timezone_name="UTC",
        )
        service.lexical_index.add_memory("mentor", value)
        service._dense_retrieve = AsyncMock(return_value=[])
        service._retrieve_graph_memories = AsyncMock(return_value=[])

        selected = await service.retrieve_for_turn("阿哈莫德是谁")

        self.assertEqual([item.memory_id for item in selected], ["mentor"])
        self.assertIsNotNone(selected[0].lexical_score)
        service._retrieve_graph_memories.assert_awaited_once()
        seeds = service._retrieve_graph_memories.await_args.args[0]
        self.assertEqual([item.memory_id for item in seeds], ["mentor"])
        retrieval_models.arerank.assert_awaited_once()


class MemoryGraphNormalizationTests(unittest.TestCase):
    def test_nfc_equivalent_entities_share_one_graph_node(self):
        graph = MemoryGraphIndex()
        graph.add_memory(
            "decomposed",
            _memory_value(
                "用户使用 Cafe\u0301",
                triples=[{
                    "subject": "Cafe\u0301",
                    "relation": "used_by",
                    "object": "用户",
                }],
            ),
        )
        graph.add_memory(
            "composed",
            _memory_value(
                "用户使用 Café",
                triples=[{
                    "subject": "Café",
                    "relation": "preferred_by",
                    "object": "用户",
                }],
            ),
        )

        self.assertEqual(graph.stats()["nodes"], 2)


if __name__ == "__main__":
    unittest.main()
