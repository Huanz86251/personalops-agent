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
    async def test_latest_pair_is_exact_and_only_high_relevance_older_pairs_return(self):
        unrelated = "上周讨论晚餐菜单。"
        related = "海报必须保留影院地址，并导出 PNG。"
        low_score = "顺便提过一次天气。"
        latest_user = "修改上一条：取消导出 PDF，保留其他要求。" + "精确原文" * 20
        latest_assistant = "收到修改，下一步会保留 PNG。"
        retrieval_models = SimpleNamespace(
            arerank=AsyncMock(return_value=[
                RerankResult(index=1, text=related + "\n相关回复", score=0.91),
                RerankResult(index=0, text=unrelated + "\n无关回复", score=0.12),
                RerankResult(index=2, text=low_score + "\n普通回复", score=0.64),
            ])
        )
        runtime = SimpleNamespace(
            settings=SimpleNamespace(planning=SimpleNamespace(
                hard_recent_dialogue_turns=1,
                hard_recent_dialogue_max_chars=220,
                conversation_summary_trigger_turns=1,
                conversation_summary_max_chars=40,
            )),
            toolset_catalog=[],
            toolset_router=None,
            retrieval_models=retrieval_models,
            _build_config=lambda thread_id: {"configurable": {"thread_id": thread_id}},
        )
        agent = SimpleNamespace(aupdate_state=AsyncMock())
        model = SimpleNamespace(ainvoke=AsyncMock(return_value=AIMessage(content="历史进展")))
        state = {"messages": [
            HumanMessage(content=unrelated), AIMessage(content="无关回复"),
            HumanMessage(content=related), AIMessage(content="相关回复"),
            HumanMessage(content=low_score), AIMessage(content="普通回复"),
            HumanMessage(content=latest_user), AIMessage(content=latest_assistant),
        ]}
        context = await ConversationRuntime._prepare_planning_context(
            runtime, agent=agent, hard_model=model,
            conversation=SimpleNamespace(thread_id="offline-test"),
            state_values=state, user_request="继续执行", memory_context="旧偏好",
        )
        self.assertEqual(context.user_request, "继续执行")
        self.assertEqual(context.user_instruction_history, [unrelated, related, low_score, latest_user])
        self.assertEqual(context.conversation_summary, "")
        restored = PlanningContextPack.model_validate_json(context.model_dump_json())
        rendered = _format_hard_context(restored)
        self.assertIn(related, rendered)
        # Earlier user instructions stay exact; irrelevant assistant replies do not.
        self.assertIn(unrelated, rendered)
        self.assertNotIn("无关回复", rendered)
        self.assertNotIn("普通回复", rendered)
        self.assertIn(latest_user, rendered)
        self.assertIn(latest_assistant, rendered)
        self.assertLess(rendered.index(related), rendered.index(latest_user))
        retrieval_models.arerank.assert_awaited_once()
        rerank_call = retrieval_models.arerank.await_args.kwargs
        self.assertEqual(rerank_call["query"], "继续执行")
        self.assertEqual(rerank_call["top_k"], 3)
        self.assertEqual(len(rerank_call["documents"]), 3)
        self.assertIn(unrelated, rerank_call["documents"][0])
        self.assertIn(related, rerank_call["documents"][1])
        self.assertIn(low_score, rerank_call["documents"][2])
        agent.aupdate_state.assert_not_awaited()
        model.ainvoke.assert_not_awaited()

    async def test_feishu_older_turns_are_summarized_once_without_rewriting_current_request(self):
        messages = []
        for index in range(6):
            messages.extend([
                HumanMessage(content=f"历史用户原话 {index}"),
                AIMessage(content=f"历史任务结果 {index}"),
            ])
        summary_model = SimpleNamespace(
            ainvoke=AsyncMock(return_value=AIMessage(content="较早两轮的已确认进展"))
        )
        runtime = SimpleNamespace(
            settings=SimpleNamespace(planning=SimpleNamespace(
                hard_recent_dialogue_max_chars=4000,
                conversation_summary_trigger_turns=2,
                conversation_summary_max_chars=1000,
            )),
            summary_model=summary_model,
            toolset_catalog=[],
            toolset_router=None,
            retrieval_models=None,
            _build_config=lambda thread_id: {"configurable": {"thread_id": thread_id}},
        )
        agent = SimpleNamespace(aupdate_state=AsyncMock())
        original_request = "请继续处理本轮文件，保留这句原始命令。"
        context = await ConversationRuntime._prepare_planning_context(
            runtime, agent=agent, hard_model=SimpleNamespace(),
            conversation=SimpleNamespace(thread_id="feishu-conversation"),
            state_values={"messages": messages, "previous_run_summary": "上一轮内部审核未通过"}, user_request=original_request,
            memory_context="",
        )
        self.assertEqual(context.user_request, original_request)
        self.assertEqual(context.conversation_summary, "较早两轮的已确认进展")
        self.assertEqual(context.previous_run_summary, "上一轮内部审核未通过")
        self.assertEqual(context.recent_dialogue[-2].content, "历史用户原话 5")
        self.assertEqual(context.recent_dialogue[-1].content, "历史任务结果 5")
        self.assertIn("历史用户原话 0", _format_hard_context(context))
        self.assertNotIn("历史任务结果 0", _format_hard_context(context))
        self.assertIn(original_request, _format_hard_context(context))
        self.assertEqual(summary_model.ainvoke.await_count, 1)
        self.assertNotIn("历史用户原话 0", summary_model.ainvoke.call_args.args[0][1]["content"])
        saved = agent.aupdate_state.await_args.args[1]
        self.assertEqual(saved["conversation_summary_message_count"], 2)

        resumed = await ConversationRuntime._prepare_planning_context(
            runtime, agent=agent, hard_model=SimpleNamespace(),
            conversation=SimpleNamespace(thread_id="feishu-conversation"),
            state_values={"messages": messages, **saved},
            user_request=original_request, memory_context="",
        )
        self.assertEqual(resumed.conversation_summary, context.conversation_summary)
        self.assertEqual(summary_model.ainvoke.await_count, 1)

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
        })
        self.assertIn("报表已写入但未验收", material)
        self.assertNotIn("给用户的原文", material)
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
        for role in ("supervisor", "replanner", "final_reviewer", "code_controller"):
            fixed, _ = split_prompt(load_prompt("planning/" + role))
            self.assertNotIn("你是", fixed)
            self.assertNotIn("历史用户原文", fixed)


if __name__ == "__main__":
    unittest.main()
