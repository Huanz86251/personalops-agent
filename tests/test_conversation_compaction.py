"""Offline conversation policy, JSON transport, persistence and failure checks."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain.messages import AIMessage, HumanMessage

from conversation_compaction import (
    ConversationSummary, SummaryCheckpoint, compress_dialogue, dialogue_token_estimate,
)
from conversation_runtime import ConversationRuntime
from config import _load_planning_settings
from hard_planning import _format_hard_context


def summary_text(label="当前只导出CSV，已取消PDF"):
    return ConversationSummary(
        user_requirements=[label], decisions=[], assistant_results=["助手报告尚未测试"],
        pending_items=["验证导出结果"], important_references=["export.py"],
    ).model_dump_json()


def messages(count):
    return [item for n in range(1, count + 1) for item in (
        HumanMessage(content=f"用户第{n}轮：保留原文，不删中间。"),
        AIMessage(content=f"最终回复第{n}轮：尚未验证。"),
    )]


class RecordingSummaryModel:
    """Provider double recording the schema binding separately from raw output."""
    def __init__(self, response=None):
        self.model_name = "qwen3.8-flash"
        self.ainvoke = AsyncMock(return_value=response or AIMessage(content=summary_text()))
        self.structured_calls = []

    def with_structured_output(self, schema, **options):
        self.structured_calls.append((schema, options))

        async def invoke(messages):
            raw = await self.ainvoke(messages)
            try:
                parsed = schema.model_validate_json(raw.content) if hasattr(schema, "model_validate_json") else json.loads(raw.content)
                error = None
            except ValueError as exception:
                parsed, error = None, exception
            return {"raw": raw, "parsed": parsed, "parsing_error": error}

        return SimpleNamespace(ainvoke=invoke)


class ConversationCompactionTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_restores_all_originals_without_summary_call(self):
        runtime = self.runtime(threshold=1)
        runtime.settings.planning.conversation_compaction_enabled = False
        from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content="第一个原始要求"), AIMessage(content="第一轮实际回复"),
                    HumanMessage(content="第二个原始要求"), AIMessage(content="第二轮实际回复")]
        agent = SimpleNamespace(aupdate_state=AsyncMock())
        result, agent = await self.prepare(runtime, {"messages": messages, "conversation_summary": "旧摘要"}, agent)
        self.assertEqual(len(result.recent_dialogue), 4)
        self.assertEqual(result.conversation_summary, "")
        agent.aupdate_state.assert_not_awaited()
        runtime.summary_model.ainvoke.assert_not_awaited()

    def runtime(self, keep=3, threshold=10000, response=None):
        return SimpleNamespace(
            settings=SimpleNamespace(planning=SimpleNamespace(
                conversation_compaction_enabled=True,
                hard_recent_dialogue_turns=keep,
                conversation_summary_trigger_tokens=threshold,
                conversation_summary_max_chars=6000,
            )),
            summary_model=RecordingSummaryModel(AIMessage(content=response or summary_text())),
            toolset_catalog=[], toolset_router=None,
            retrieval_models=SimpleNamespace(arerank=AsyncMock()),
            _build_config=lambda thread: {"configurable": {"thread_id": thread}},
        )

    async def prepare(self, runtime, state, agent=None):
        agent = agent or SimpleNamespace(aupdate_state=AsyncMock())
        result = await ConversationRuntime._prepare_planning_context(
            runtime, agent=agent, hard_model=None,
            conversation=SimpleNamespace(thread_id="offline-C"), state_values=state,
            user_request="现在执行，当前原话不能改。", memory_context="",
        )
        return result, agent

    async def test_below_threshold_keeps_all_pending_and_recent_exactly(self):
        runtime = self.runtime()
        original = messages(10)
        context, agent = await self.prepare(runtime, {
            "messages": original, "previous_run_summary": "内部失败记录不要注入",
        })
        self.assertEqual([m.content for m in context.recent_dialogue], [m.content for m in original])
        self.assertEqual(context.conversation_summary, "")
        self.assertEqual(context.previous_run_summary, "")
        self.assertEqual(context.user_instruction_history, [])
        self.assertNotIn("内部失败记录", _format_hard_context(context))
        runtime.retrieval_models.arerank.assert_not_awaited()
        runtime.summary_model.ainvoke.assert_not_awaited()
        agent.aupdate_state.assert_not_awaited()

    async def test_at_threshold_compresses_pairs_and_restores_without_resummarizing(self):
        runtime = self.runtime()
        state = {"messages": messages(8)}
        with patch("conversation_runtime.dialogue_token_estimate", return_value=10000):
            context, agent = await self.prepare(runtime, state)
        saved = agent.aupdate_state.await_args.args[1]
        self.assertEqual(saved["conversation_summary_turn_count"], 5)
        checkpoint = SummaryCheckpoint.model_validate_json(context.conversation_summary)
        self.assertEqual(checkpoint.covered_turns, 5)
        self.assertEqual([m.content for m in context.recent_dialogue], [m.content for m in state["messages"][10:]])
        call = runtime.summary_model.ainvoke.await_args
        self.assertEqual(runtime.summary_model.structured_calls, [(ConversationSummary.model_json_schema(), {
            "method": "json_schema", "strict": True, "include_raw": True,
        })])
        self.assertNotIn("JSON Schema:", call.args[0][0]["content"])
        self.assertNotIn('"properties"', call.args[0][0]["content"])
        source = json.loads(call.args[0][1]["content"])
        self.assertEqual([t["turn"] for t in source["new_dialogue"]], [1, 2, 3, 4, 5])
        self.assertEqual(source["new_dialogue"][0]["messages"][0]["content"], state["messages"][0].content)
        self.assertEqual(source["new_dialogue"][0]["messages"][1]["content"], state["messages"][1].content)
        restored, _ = await self.prepare(runtime, {**state, **saved})
        self.assertEqual(restored.conversation_summary, context.conversation_summary)
        self.assertEqual(runtime.summary_model.ainvoke.await_count, 1)

    async def test_next_batch_combines_old_summary_and_only_new_turns(self):
        runtime = self.runtime()
        previous = SummaryCheckpoint(covered_turns=5, summary=ConversationSummary.model_validate_json(summary_text()))
        with patch("conversation_runtime.dialogue_token_estimate", return_value=10000):
            context, _ = await self.prepare(runtime, {
                "messages": messages(13), "conversation_summary": previous.model_dump_json(),
            })
        source = json.loads(runtime.summary_model.ainvoke.await_args.args[0][1]["content"])
        self.assertEqual(source["previous_summary"], previous.summary.model_dump())
        self.assertEqual([t["turn"] for t in source["new_dialogue"]], list(range(6, 11)))
        self.assertEqual(SummaryCheckpoint.model_validate_json(context.conversation_summary).covered_turns, 10)

    async def test_invalid_json_failure_or_truncation_preserves_all_pending(self):
        for response in ["not JSON", "{}", summary_text()[:-1],
                         ConversationSummary(user_requirements=[], decisions=[], assistant_results=[],
                             pending_items=[], important_references=[]).model_dump_json(),
                         summary_text("x" * 7000)]:
            with self.subTest(response=response[:30]):
                runtime = self.runtime(response=response)
                old = SummaryCheckpoint(covered_turns=1, summary=ConversationSummary.model_validate_json(summary_text()))
                state = {"messages": messages(8), "conversation_summary": old.model_dump_json()}
                with patch("conversation_runtime.dialogue_token_estimate", return_value=10000):
                    context, agent = await self.prepare(runtime, state)
                self.assertEqual(context.conversation_summary, old.model_dump_json())
                self.assertEqual([m.content for m in context.recent_dialogue], [m.content for m in state["messages"][2:]])
                agent.aupdate_state.assert_not_awaited()
        runtime = self.runtime()
        runtime.summary_model.ainvoke.side_effect = RuntimeError("offline failure")
        with patch("conversation_runtime.dialogue_token_estimate", return_value=10000):
            context, agent = await self.prepare(runtime, {"messages": messages(8)})
        self.assertEqual(len(context.recent_dialogue), 16)
        agent.aupdate_state.assert_not_awaited()

    async def test_changing_recent_count_and_legacy_checkpoint(self):
        runtime = self.runtime(keep=5)
        old = SummaryCheckpoint(covered_turns=5, summary=ConversationSummary.model_validate_json(summary_text()))
        context, _ = await self.prepare(runtime, {"messages": messages(8), "conversation_summary": old.model_dump_json()})
        self.assertEqual(len(context.recent_dialogue), 10)
        runtime.summary_model.ainvoke.assert_not_awaited()
        context, _ = await self.prepare(runtime, {"messages": messages(8),
            "conversation_summary": "旧助手摘要", "conversation_summary_message_count": 999})
        self.assertEqual(len(context.recent_dialogue), 16)
        self.assertEqual(context.conversation_summary, "")

    async def test_finish_reason_length_does_not_accept_even_valid_json(self):
        model = RecordingSummaryModel(AIMessage(
            content=summary_text(), response_metadata={"finish_reason": "length"}))
        with self.assertRaisesRegex(ValueError, "truncated"):
            await compress_dialogue(model, previous=None, turns=[], first_turn=1, max_chars=6000)

    def test_token_estimate_counts_cjk_and_special_literal_without_error(self):
        from planning_models import DialogueMessage
        chinese = [[DialogueMessage(role="user", content="中文内容" * 100)]]
        latin = [[DialogueMessage(role="user", content="a" * 400)]]
        self.assertGreater(dialogue_token_estimate(chinese), dialogue_token_estimate(latin))
        self.assertGreater(dialogue_token_estimate([[DialogueMessage(role="user", content="<|endoftext|>")]]), 0)

    def test_configurable_defaults_and_bounds(self):
        with patch.dict("os.environ", {"PLANNING_HARD_RECENT_DIALOGUE_TURNS": "3",
             "PLANNING_CONVERSATION_SUMMARY_TRIGGER_TOKENS": "10000"}):
            config = _load_planning_settings()
            self.assertEqual(config.hard_recent_dialogue_turns, 3)
            self.assertEqual(config.conversation_summary_trigger_tokens, 10000)
        with patch.dict("os.environ", {"PLANNING_CONVERSATION_SUMMARY_TRIGGER_TOKENS": "0"}):
            with self.assertRaises(RuntimeError):
                _load_planning_settings()

    async def test_real_token_policy_simulation_over_sixteen_turns(self):
        runtime = self.runtime(keep=3, threshold=10000)
        state = {"messages": []}
        boundaries = []
        for n in range(1, 17):
            state["messages"].extend([
                HumanMessage(content=f"用户{n}要求CSV，取消PDF。" + "保留用户原话。" * 200),
                AIMessage(content=f"回复{n}：助手报告导出完成，测试未执行。" + "具体结果尚待验证。" * 200),
            ])
            context, agent = await self.prepare(runtime, state)
            if agent.aupdate_state.await_count:
                saved = agent.aupdate_state.await_args.args[1]
                state.update(saved)
                boundaries.append(saved["conversation_summary_turn_count"])
            # Every recent pair survives unchanged, even if individually large.
            self.assertEqual([m.content for m in context.recent_dialogue[-6:]],
                             [m.content for m in state["messages"][-6:]])
        self.assertGreaterEqual(len(boundaries), 2)
        self.assertEqual(boundaries, sorted(set(boundaries)))
        calls = runtime.summary_model.ainvoke.await_args_list
        source_ranges = [json.loads(call.args[0][1]["content"])["new_dialogue"] for call in calls]
        covered = [turn["turn"] for batch in source_ranges for turn in batch]
        self.assertEqual(covered, list(range(1, boundaries[-1] + 1)))
        self.assertIsNotNone(json.loads(calls[1].args[0][1]["content"])["previous_summary"])
        self.assertEqual(len(state["messages"]), 32)

    async def test_json_summary_survives_actual_sqlite_checkpoint_reopen(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        from langgraph.graph import StateGraph, START, END
        from workers.progress import WorkerProgressState
        config = {"configurable": {"thread_id": "offline-C"}}

        def build(saver):
            graph = StateGraph(WorkerProgressState)
            graph.add_node("model", lambda state: {})
            graph.add_edge(START, "model")
            graph.add_edge("model", END)
            return graph.compile(checkpointer=saver)

        with TemporaryDirectory() as directory:
            database = str(Path(directory) / "checkpoints.sqlite")
            async with AsyncSqliteSaver.from_conn_string(database) as saver:
                graph = build(saver)
                await graph.aupdate_state(config, {"messages": messages(8)}, as_node="model")
                state = (await graph.aget_state(config)).values
                runtime = self.runtime()
                with patch("conversation_runtime.dialogue_token_estimate", return_value=10000):
                    context, _ = await self.prepare(runtime, state, graph)
            async with AsyncSqliteSaver.from_conn_string(database) as saver:
                graph = build(saver)
                restored = (await graph.aget_state(config)).values
                self.assertEqual(restored["conversation_summary_turn_count"], 5)
                self.assertEqual(restored["conversation_summary"], context.conversation_summary)
                self.assertEqual(len(restored["messages"]), 16)
                runtime = self.runtime()
                await self.prepare(runtime, restored, graph)
                runtime.summary_model.ainvoke.assert_not_awaited()

    async def test_qwen_strict_schema_actual_sdk_http_payload_offline(self):
        import httpx
        from model_clients import CompatibleChatModel
        from planning_models import DialogueMessage
        payloads = []

        def serve(request):
            payloads.append(json.loads(request.content))
            return httpx.Response(200, json={
                "id": "offline-summary", "object": "chat.completion", "created": 0,
                "model": "qwen3.8-flash", "choices": [{"index": 0,
                    "finish_reason": "stop", "message": {"role": "assistant", "content": summary_text()}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            })

        async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
            model = CompatibleChatModel(model="qwen3.8-flash", api_key="offline-placeholder",
                base_url="https://example.invalid/v1", http_async_client=client,
                extra_body={"enable_thinking": False})
            result = await compress_dialogue(model, previous=None,
                turns=[[DialogueMessage(role="user", content="只导出CSV"),
                        DialogueMessage(role="assistant", content="尚未测试")]],
                first_turn=1, max_chars=6000)
        self.assertEqual(result.user_requirements, ["当前只导出CSV，已取消PDF"])
        self.assertEqual(len(payloads), 1)
        payload = payloads[0]
        self.assertEqual(payload["model"], "qwen3.8-flash")
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        contract = payload["response_format"]["json_schema"]
        self.assertTrue(contract["strict"])
        self.assertEqual(set(contract["schema"]["required"]), set(ConversationSummary.model_fields))
        self.assertFalse(contract["schema"]["additionalProperties"])
        self.assertNotIn("JSON Schema:", payload["messages"][0]["content"])
        self.assertNotIn("tools", payload)

    async def test_unverified_provider_keeps_shared_fallback(self):
        model = RecordingSummaryModel()
        model.model_name = "other-model"
        await compress_dialogue(model, previous=None, turns=[], first_turn=1, max_chars=6000)
        self.assertEqual(model.structured_calls, [(ConversationSummary.model_json_schema(), {
            "method": "json_mode", "include_raw": True,
        })])
        self.assertIn("JSON Schema:", model.ainvoke.await_args.args[0][0]["content"])


if __name__ == "__main__":
    unittest.main()
