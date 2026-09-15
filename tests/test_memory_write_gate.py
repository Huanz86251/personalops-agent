import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from memory import (
    MEMORY_CONFLICT_NAMESPACE,
    MEMORY_NAMESPACE,
    MEMORY_WRITE_CANDIDATE_NAMESPACE,
    MEMORY_WRITE_INBOX_NAMESPACE,
    MemoryService,
)
from memory_extraction_models import (
    CandidateFramePlan,
    MemoryFrame,
    MemoryFramePlan,
    PersonRelationRecord,
    ProfileRecord,
)
from memory_write_gate import MemOperatorWriteGate, MemoryWriteDecision


class FakeStore:
    def __init__(self):
        self.rows = {}

    async def aput(self, namespace, key, value, index=None):
        self.rows[(namespace, key)] = dict(value)

    async def adelete(self, namespace, key):
        self.rows.pop((namespace, key), None)

    async def aget(self, namespace, key):
        value = self.rows.get((namespace, key))
        return None if value is None else SimpleNamespace(key=key, value=dict(value))

    async def asearch(self, namespace, **kwargs):
        return [
            SimpleNamespace(key=key, value=dict(value))
            for (stored_namespace, key), value in self.rows.items()
            if stored_namespace == namespace
        ]


class MemoryWriteGateTests(unittest.IsolatedAsyncioTestCase):
    def test_preferred_name_is_explicit_and_key_parts_use_nfc(self):
        record = ProfileRecord(
            candidate_id="candidate-001",
            frame_id="f1",
            record_type="profile",
            summary="用户希望被称为Café",
            importance="high",
            confidence="high",
            field="preferred_name",
            value="Cafe\u0301",
            valid_from=None,
            valid_to=None,
        )

        candidate = MemoryService._typed_record_to_candidate(record, {})

        self.assertEqual(candidate.structured_data["field"], "preferred_name")
        self.assertEqual(
            candidate.structured_data["dedupe_key"],
            "profile|preferred_name|café",
        )
        self.assertEqual(
            candidate.structured_data["conflict_key"],
            "profile|preferred_name",
        )

    def test_memoperator_parser_requires_one_decision_per_message(self):
        decisions = MemOperatorWriteGate._parse_decisions(
            '说明\n[{"index": 1, "decision": "SAVE"},'
            '{"index": 2, "decision": "skip"}]',
            2,
        )
        self.assertEqual(
            [(item.index, item.label) for item in decisions],
            [(1, "SAVE"), (2, "SKIP")],
        )

        with self.assertRaises(RuntimeError):
            MemOperatorWriteGate._parse_decisions(
                '[{"index": 1, "decision": "SAVE"}]',
                2,
            )

    async def test_three_user_messages_trigger_one_async_batch(self):
        store = FakeStore()
        gate = SimpleNamespace(
            model_name="MemTensor/MemOperator-0.6B",
            aclassify=AsyncMock(
                return_value=[
                    MemoryWriteDecision(1, "SAVE"),
                    MemoryWriteDecision(2, "SKIP"),
                    MemoryWriteDecision(3, "SAVE"),
                ]
            ),
        )
        service = MemoryService(
            store=store,
            retrieval_models=SimpleNamespace(),
            model=None,
            write_gate=gate,
            write_gate_batch_size=3,
        )

        common = {
            "source_platform": "feishu",
            "source_conversation_id": "conversation-1",
            "source_thread_id": "thread-1",
        }
        self.assertEqual(
            await service.consolidate_turn(user_text="今天在研究缓存策略", **common),
            [],
        )
        self.assertEqual(
            await service.consolidate_turn(user_text="这个方案可能还要调整", **common),
            [],
        )
        admitted = await service.consolidate_turn(
            user_text="项目下周进入测试阶段",
            **common,
        )

        gate.aclassify.assert_awaited_once_with(
            [
                "今天在研究缓存策略",
                "这个方案可能还要调整",
                "项目下周进入测试阶段",
            ]
        )
        self.assertEqual(len(admitted), 2)
        inbox = await store.asearch(MEMORY_WRITE_INBOX_NAMESPACE)
        candidates = await store.asearch(MEMORY_WRITE_CANDIDATE_NAMESPACE)
        self.assertEqual(inbox, [])
        self.assertEqual(len(candidates), 2)
        self.assertTrue(
            all(item.value["status"] == "pending_extraction" for item in candidates)
        )
        self.assertNotIn("assistant_reply", candidates[0].value)

    async def test_explicit_memory_rule_bypasses_model_into_candidate_buffer(self):
        store = FakeStore()
        gate = SimpleNamespace(
            model_name="MemTensor/MemOperator-0.6B",
            aclassify=AsyncMock(),
        )
        service = MemoryService(
            store=store,
            retrieval_models=SimpleNamespace(),
            model=None,
            write_gate=gate,
        )

        admitted = await service.consolidate_turn(
            user_text="请记住我以后偏好简短回答",
            source_platform="feishu",
            source_conversation_id="conversation-1",
            source_thread_id="thread-1",
        )

        self.assertEqual(len(admitted), 1)
        gate.aclassify.assert_not_awaited()
        candidates = await store.asearch(MEMORY_WRITE_CANDIDATE_NAMESPACE)
        self.assertEqual(
            candidates[0].value["decision_source"],
            "deterministic_explicit_rule",
        )

    async def test_gate_failure_keeps_the_durable_inbox(self):
        store = FakeStore()
        gate = SimpleNamespace(
            model_name="MemTensor/MemOperator-0.6B",
            aclassify=AsyncMock(side_effect=RuntimeError("offline")),
        )
        service = MemoryService(
            store=store,
            retrieval_models=SimpleNamespace(),
            model=None,
            write_gate=gate,
            write_gate_batch_size=3,
        )
        common = {
            "source_platform": "feishu",
            "source_conversation_id": "conversation-1",
            "source_thread_id": "thread-1",
        }
        for text in ("第一条普通消息", "第二条普通消息", "第三条普通消息"):
            await service.consolidate_turn(user_text=text, **common)

        inbox = await store.asearch(MEMORY_WRITE_INBOX_NAMESPACE)
        self.assertEqual(len(inbox), 3)
        self.assertTrue(all(item.value["attempt_count"] == 1 for item in inbox))
        self.assertEqual(
            await store.asearch(MEMORY_WRITE_CANDIDATE_NAMESPACE),
            [],
        )

    async def test_progressive_processor_backfills_source_and_keeps_level_labels(self):
        store = FakeStore()
        await store.aput(MEMORY_WRITE_CANDIDATE_NAMESPACE, "candidate-001", {
            "status": "pending_extraction",
            "raw_user_text": "阿哈默德是我的导师",
            "source_platform": "feishu",
            "source_conversation_id": "conversation-1",
            "source_thread_id": "thread-1",
            "queued_at": "2026-08-10T10:00:00+08:00",
        })
        plan = MemoryFramePlan(candidates=[CandidateFramePlan(
            candidate_id="candidate-001",
            frames=[MemoryFrame(frame_id="f1", frame_type="person_relation")],
        )])
        record = PersonRelationRecord(
            candidate_id="candidate-001", frame_id="f1",
            record_type="person_relation", summary="阿哈默德是用户的导师",
            importance="high", confidence="low", person_name="阿哈默德",
            relation="mentor", other_relation=None, state="current",
            valid_from=None, valid_to=None,
        )
        service = MemoryService(
            store=store, retrieval_models=SimpleNamespace(), model=SimpleNamespace(),
            extraction_enabled=True, extraction_batch_size=1,
        )
        service.extract_progressive_batch = AsyncMock(return_value=(plan, [record]))
        service._find_typed_duplicate_ids = AsyncMock(return_value=[])
        service._store_typed_candidate = AsyncMock(return_value=("mem-001", None))

        self.assertEqual(await service._process_extraction_batch(), ["mem-001"])
        candidate = service._store_typed_candidate.await_args.kwargs["candidate"]
        self.assertEqual(candidate.content, "阿哈默德是用户的导师")
        self.assertEqual(candidate.evidence[0]["source_text"], "阿哈默德是我的导师")
        self.assertEqual(candidate.structured_data["importance"], "high")
        self.assertEqual(candidate.structured_data["confidence"], "low")
        self.assertEqual(candidate.confidence, 1)
        self.assertEqual(candidate.triples[0].relation, "mentor_of")
        saved = await store.aget(MEMORY_WRITE_CANDIDATE_NAMESPACE, "candidate-001")
        self.assertEqual(saved.value["status"], "extracted")

    async def test_same_slot_same_confidence_creates_user_resolvable_group(self):
        store = FakeStore()
        service = MemoryService(
            store=store,
            retrieval_models=SimpleNamespace(),
            model=None,
            timezone_name="UTC",
        )
        records = [
            ProfileRecord(
                candidate_id=f"candidate-{index}", frame_id="f1",
                record_type="profile", summary=f"用户希望被称为{name}",
                importance="high", confidence="high",
                field="preferred_name", value=name,
                valid_from=None, valid_to=None,
            )
            for index, name in enumerate(("小黄", "小王"), start=1)
        ]
        stored = []
        for record in records:
            candidate = service._typed_record_to_candidate(record, {})
            stored.append(await service._store_typed_candidate(
                candidate,
                source_platform="feishu",
                source_conversation_id="conversation-1",
                source_thread_id="thread-1",
            ))

        self.assertIsNone(stored[0][1])
        self.assertIsNotNone(stored[1][1])
        pair = await service.next_conflict_pair()
        self.assertEqual(
            {pair.left["content"], pair.right["content"]},
            {"用户希望被称为小黄", "用户希望被称为小王"},
        )
        result = await service.resolve_conflict_pair(
            group_id=pair.group_id,
            keep_memory_id=pair.left["memory_id"],
            retire_memory_id=pair.right["memory_id"],
        )
        self.assertEqual(result["group_status"], "resolved")
        retired = await store.aget(MEMORY_NAMESPACE, pair.right["memory_id"])
        self.assertEqual(retired.value["status"], "retired")
        group = await store.aget(MEMORY_CONFLICT_NAMESPACE, pair.group_id)
        self.assertEqual(group.value["status"], "resolved")

    async def test_different_confidence_or_nonoverlap_does_not_conflict(self):
        store = FakeStore()
        service = MemoryService(
            store=store,
            retrieval_models=SimpleNamespace(),
            model=None,
            timezone_name="UTC",
        )
        first = ProfileRecord(
            candidate_id="c1", frame_id="f1", record_type="profile",
            summary="用户在上海", importance="high", confidence="high",
            field="location", value="上海",
            valid_from="2026-01-01T00:00:00+08:00",
            valid_to="2026-02-01T00:00:00+08:00",
        )
        second = ProfileRecord(
            candidate_id="c2", frame_id="f1", record_type="profile",
            summary="用户在北京", importance="high", confidence="high",
            field="location", value="北京",
            valid_from="2026-03-01T00:00:00+08:00",
            valid_to=None,
        )
        third = ProfileRecord(
            candidate_id="c3", frame_id="f1", record_type="profile",
            summary="用户在深圳", importance="high", confidence="low",
            field="location", value="深圳",
            valid_from="2026-03-01T00:00:00+08:00",
            valid_to=None,
        )
        for record in (first, second, third):
            await service._store_typed_candidate(
                service._typed_record_to_candidate(record, {}),
                source_platform="feishu",
                source_conversation_id="conversation-1",
                source_thread_id="thread-1",
            )
        self.assertIsNone(await service.next_conflict_pair())


if __name__ == "__main__":
    unittest.main()
