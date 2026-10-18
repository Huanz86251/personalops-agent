"""离线验证：历史原文与摘要隔离，旧 checkpoint 保持兼容。"""
import unittest
import json
from datetime import timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

from langchain.messages import AIMessage, HumanMessage

from context_middlewares import summarize_conversation_history
from conversation_runtime import ConversationRuntime, _planning_run_handoff_material
from hard_planning import _format_hard_context
from planning_models import PlanningContextPack
from memory import (
    MemoryService,
    RetrievedMemory,
)
from memory_extraction_models import PersonRelationRecord, TaskRecord
from prompt_loader import load_prompt, split_prompt
from retrieval_models import RerankResult


class PromptContextContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_pending_and_latest_pairs_are_exact_without_history_reranking(self):
        from tests.test_conversation_compaction import ConversationCompactionTests, messages
        harness = ConversationCompactionTests()
        runtime = harness.runtime(keep=3)
        original = messages(6)
        context, agent = await harness.prepare(runtime, {"messages": original})
        self.assertEqual([m.content for m in context.recent_dialogue], [m.content for m in original])
        self.assertEqual(context.user_instruction_history, [])
        self.assertEqual(context.previous_run_summary, "")
        restored = PlanningContextPack.model_validate_json(context.model_dump_json())
        rendered = _format_hard_context(restored)
        for message in original:
            self.assertIn(message.content, rendered)
        runtime.retrieval_models.arerank.assert_not_awaited()
        agent.aupdate_state.assert_not_awaited()

    async def test_feishu_json_summary_is_reused_and_internal_summary_is_ignored(self):
        from unittest.mock import patch
        from tests.test_conversation_compaction import ConversationCompactionTests, messages
        harness = ConversationCompactionTests()
        runtime = harness.runtime(keep=3)
        state = {"messages": messages(6), "previous_run_summary": "内部审核记录"}
        with patch("conversation_runtime.dialogue_token_estimate", return_value=10000):
            context, agent = await harness.prepare(runtime, state)
        saved = agent.aupdate_state.await_args.args[1]
        self.assertEqual(saved["conversation_summary_turn_count"], 3)
        self.assertEqual(context.previous_run_summary, "")
        self.assertNotIn("内部审核记录", _format_hard_context(context))
        resumed, _ = await harness.prepare(runtime, {**state, **saved})
        self.assertEqual(resumed.conversation_summary, context.conversation_summary)
        self.assertEqual(runtime.summary_model.ainvoke.await_count, 1)

    def test_previous_run_summary_is_separate_and_raw_trace_is_not_replayed(self):
        material = _planning_run_handoff_material({
            "plan_objective": "更新报表",
            "final_status": "FAILED",
            "overall_stop_reason": "验收未通过",
            "final_answer": "给用户的原文，不放入内部摘要",
            "completed_step_reports": [{
                "step_id": 1, "status": "FAILED", "summary": "报表已写入但未验收",
                "stop_reason": "缺少回读", "secret_token": "never-copy-this",
            }],
            "final_review_decision": {
                "action": "FINAL",
                "status": "COMPLETED",
                "review_reason": "最终审核确认回读已满足。",
                "criterion_reviews": [{
                    "criterion_id": "C1",
                    "status": "MET",
                    "observed_result": "最终材料包含回读结果。",
                    "missing_requirement": None,
                    "evidence_refs": ["E1"],
                }],
                "final_answer": "不要复制到内部摘要的用户正文",
                "unmet_success_criteria": [],
            },
        })
        self.assertIn("报表已写入但未验收", material)
        self.assertIn("最终审核确认回读已满足", material)
        self.assertIn('"status": "MET"', material)
        self.assertNotIn("给用户的原文", material)
        self.assertNotIn("不要复制到内部摘要的用户正文", material)
        self.assertNotIn("never-copy-this", material)
        context = PlanningContextPack(
            current_time="2026-09-18", user_request="请修复上一轮报表",
            previous_run_summary="上一轮写入了报表，但回读验收未完成",
        )
        rendered = _format_hard_context(context)
        self.assertIn("上一轮写入了报表", rendered)
        self.assertIn("请修复上一轮报表", rendered)

    async def test_internal_run_summary_uses_its_own_prompt(self):
        model = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content="上轮未完成回读")))
        result = await summarize_conversation_history(
            model, previous_summary="", messages=[{"role": "user", "content": "FAILED: 缺少回读"}],
            max_chars=200, prompt_name="conversation/run_summary",
            material_label="上一轮内部规划、执行与审核记录",
        )
        self.assertEqual(result, "上轮未完成回读")
        sent = model.ainvoke.call_args.args[0]
        self.assertIn("内部规划", sent[0]["content"])
        self.assertIn("FAILED: 缺少回读", sent[1]["content"])

    async def test_summary_material_is_separate_from_system_rules(self):
        material = "历史原文包含 {messages} 和 {{user_message}}，不要插值"
        model = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content="历史摘要")))
        result = await summarize_conversation_history(
            model, previous_summary="旧摘要", messages=[HumanMessage(content=material)], max_chars=100,
        )
        self.assertEqual(result, "历史摘要")
        messages = model.ainvoke.call_args.args[0]
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertNotIn(material, messages[0]["content"])
        self.assertIn(material, messages[1]["content"])

    async def test_summary_failure_stays_bounded_without_mutating_messages(self):
        message = HumanMessage(content="原始要求" * 100)
        model = SimpleNamespace(ainvoke=AsyncMock(side_effect=RuntimeError("offline failure")))
        result = await summarize_conversation_history(
            model, previous_summary="旧进展", messages=[message], max_chars=80,
        )
        self.assertLessEqual(len(result), 80)
        self.assertEqual(message.content, "原始要求" * 100)

    def test_old_context_loads_without_new_field(self):
        context = PlanningContextPack(current_time="2026-09-05", user_request="原始任务")
        self.assertEqual(context.user_instruction_history, [])
        self.assertIn("原始任务", _format_hard_context(context))

    def test_delete_request_is_not_forced_into_memory_write(self):
        self.assertIsNone(MemoryService._route_memory_write_by_rule("删除这条记忆"))
        self.assertEqual(MemoryService._route_memory_write_by_rule("记住我以后使用中文"), "RECORD")

    def test_secret_values_are_never_forced_into_memory_candidates(self):
        self.assertEqual(
            MemoryService._route_memory_write_by_rule("记住，我的密码是 Hunter2!"),
            "NOT_RECORD",
        )
        self.assertIsNone(
            MemoryService._route_memory_write_by_rule("我习惯使用密码管理器")
        )

    async def test_cross_encoder_threshold_is_the_read_gate(self):
        retrieval_models = SimpleNamespace(
            arerank=AsyncMock(return_value=[
                RerankResult(index=0, text="相关", score=0.82),
                RerankResult(index=1, text="不相关", score=0.39),
            ])
        )
        service = SimpleNamespace(
            final_limit=2,
            reranker_threshold=0.4,
            retrieval_models=retrieval_models,
            _memory_to_model_text=lambda memory: memory.content,
        )
        memories = [
            RetrievedMemory("m1", "相关", "preference", 3),
            RetrievedMemory("m2", "不相关", "episode", 2),
        ]

        selected = await MemoryService._rerank(
            service,
            query="当前问题",
            memories=memories,
        )

        self.assertEqual([memory.memory_id for memory in selected], ["m1"])
        self.assertEqual(selected[0].rerank_score, 0.82)

    async def test_confidence_reorders_but_does_not_drop_relevant_memories(self):
        retrieval_models = SimpleNamespace(
            arerank=AsyncMock(return_value=[
                RerankResult(index=0, text="低置信", score=0.80),
                RerankResult(index=1, text="高置信", score=0.75),
            ])
        )
        service = SimpleNamespace(
            final_limit=2,
            reranker_threshold=0.4,
            retrieval_models=retrieval_models,
            _memory_to_model_text=lambda memory: memory.content,
        )
        memories = [
            RetrievedMemory("low", "低置信", "preference", 3, confidence=1),
            RetrievedMemory("high", "高置信", "preference", 3, confidence=4),
        ]

        selected = await MemoryService._rerank(
            service,
            query="当前问题",
            memories=memories,
        )

        self.assertEqual([memory.memory_id for memory in selected], ["high", "low"])
        self.assertEqual(selected[0].rerank_score, 0.75)
        self.assertAlmostEqual(selected[1].retrieval_score, 0.64)
        retrieval_models.arerank.assert_awaited_once_with(
            query="当前问题",
            documents=["低置信", "高置信"],
            top_k=2,
        )

    async def test_progressive_memory_extraction_reuses_prefix_and_omits_quotes(self):
        first = {
            "candidates": [{
                "candidate_id": "candidate-001",
                "frames": [
                    {"frame_id": "f1", "frame_type": "person_relation"},
                    {"frame_id": "f2", "frame_type": "task"},
                ],
            }]
        }
        second = {"records": [
            {
                "candidate_id": "candidate-001", "frame_id": "f1",
                "record_type": "person_relation",
                "summary": "阿哈默德是用户的导师",
                "importance": "high", "confidence": "high",
                "person_name": "阿哈默德", "relation": "mentor",
                "other_relation": None, "state": "current",
                "valid_from": None, "valid_to": None,
            },
            {
                "candidate_id": "candidate-001", "frame_id": "f2",
                "record_type": "task", "summary": "阿哈默德要求用户三周内完成文档",
                "importance": "high", "confidence": "high",
                "event": "request", "from_person": "阿哈默德",
                "to_person": "用户", "action": "完成", "object": "记忆设计文档",
                "project_name": "PersonalOps", "status": "pending",
                "due_at": "2026-08-31T10:00:00+08:00",
            },
        ]}
        model = SimpleNamespace(ainvoke=AsyncMock(side_effect=[
            AIMessage(content=json.dumps(first, ensure_ascii=False)),
            AIMessage(content=json.dumps(second, ensure_ascii=False)),
        ]))
        service = MemoryService(
            store=SimpleNamespace(), retrieval_models=SimpleNamespace(), model=model,
            timezone_name="UTC",
        )
        plan, records = await service.extract_progressive_batch([(
            "candidate-001",
            {"queued_at": "2026-08-10T10:00:00+08:00", "raw_user_text":
             "阿哈默德是我的导师。他让我在未来三周内写完记忆设计文档。"},
        )])
        self.assertEqual(len(plan.candidates[0].frames), 2)
        self.assertIsInstance(records[0], PersonRelationRecord)
        self.assertIsInstance(records[1], TaskRecord)
        first_messages = model.ainvoke.await_args_list[0].args[0]
        second_messages = model.ainvoke.await_args_list[1].args[0]
        self.assertEqual(second_messages[:2], first_messages)
        self.assertEqual(second_messages[2]["content"], json.dumps(first, ensure_ascii=False))
        self.assertNotIn("阿哈默德是我的导师", second_messages[3]["content"])
        self.assertNotIn("routine", second_messages[3]["content"])
        self.assertNotIn("episode", second_messages[3]["content"])
        self.assertIn("preferred_name", second_messages[3]["content"])

    async def test_verified_memory_model_uses_provider_schema_without_prompt_copy(self):
        first = {
            "candidates": [{
                "candidate_id": "candidate-001",
                "frames": [{"frame_id": "f1", "frame_type": "profile"}],
            }],
        }
        second = {"records": [{
            "candidate_id": "candidate-001",
            "frame_id": "f1",
            "record_type": "profile",
            "summary": "用户希望被称为小黄",
            "importance": "medium",
            "confidence": "high",
            "field": "preferred_name",
            "value": "小黄",
            "valid_from": None,
            "valid_to": None,
        }]}

        class StrictMemoryModel:
            model_name = "qwen3.8-flash"

            def __init__(self):
                self.payloads = [first, second]
                self.requests = []
                self.options = []

            def with_structured_output(self, schema, **kwargs):
                self.options.append(kwargs)

                async def invoke(messages):
                    payload = self.payloads.pop(0)
                    parsed = schema.model_validate(payload) if hasattr(schema, "model_validate") else payload
                    self.requests.append(messages)
                    return {
                        "parsed": parsed,
                        "parsing_error": None,
                        "raw": AIMessage(content=json.dumps(payload, ensure_ascii=False)),
                    }

                return SimpleNamespace(ainvoke=invoke)

        model = StrictMemoryModel()
        service = MemoryService(
            store=SimpleNamespace(),
            retrieval_models=SimpleNamespace(),
            model=model,
            timezone_name="UTC",
        )
        plan, records = await service.extract_progressive_batch([(
            "candidate-001",
            {
                "queued_at": "2026-09-23T10:00:00+08:00",
                "raw_user_text": "以后叫我小黄",
            },
        )])

        self.assertEqual(plan.candidates[0].frames[0].frame_type, "profile")
        self.assertEqual(records[0].summary, "用户希望被称为小黄")
        self.assertNotIn('"properties"', model.requests[0][0]["content"])
        self.assertNotIn('"$defs"', model.requests[1][-1]["content"])
        self.assertTrue(all(item["method"] == "json_schema" for item in model.options))

    async def test_typed_extraction_does_not_call_cloud_conflict_resolver(self):
        service = SimpleNamespace(
            extraction_enabled=True,
            model=SimpleNamespace(),
            extraction_batch_size=1,
            _extraction_lock=__import__("asyncio").Lock(),
            store=SimpleNamespace(asearch=AsyncMock(return_value=[])),
        )
        self.assertEqual(await MemoryService._process_extraction_batch(service), [])

    def test_shared_rules_are_in_actual_fixed_role_prompts(self):
        self.assertIn("默认用中文", load_prompt("planning/scheduler"))
        for role in ("supervisor", "replanner", "final_reviewer", "worker_objection"):
            fixed, _ = split_prompt(load_prompt("planning/" + role))
            self.assertNotIn("你是", fixed)
            self.assertNotIn("历史用户原文", fixed)


if __name__ == "__main__":
    unittest.main()
