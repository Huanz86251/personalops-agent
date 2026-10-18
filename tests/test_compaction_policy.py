"""Strict compression, one retry, harmless repair and protected history."""
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain.messages import AIMessage, HumanMessage, ToolMessage

from compaction_runtime import arun_compaction, CompactionFailure
from conversation_compaction import compress_dialogue
from planning_models import DialogueMessage, PlanningContextPack
from schemas.compaction import ConversationSummary, WorkerExecutionSummary
from scheduler_runtime import SchedulerConversation
from workers.compaction import WorkerCompactionMiddleware


class SequenceModel:
    model_name = "qwen3.8-flash"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.requests, self.bindings = [], []

    def with_structured_output(self, schema, **options):
        self.bindings.append((schema, options))

        def invoke(messages):
            self.requests.append(copy.deepcopy(messages))
            value = self.outputs.pop(0)
            if isinstance(value, Exception):
                raise value
            return {"parsed": value, "raw": AIMessage(content=json.dumps(value)), "parsing_error": None}

        async def ainvoke(messages):
            return invoke(messages)
        return SimpleNamespace(invoke=invoke, ainvoke=ainvoke)


def worker_payload():
    return {"observed_results": ["语法检查通过；行为测试尚未完成"],
        "errors_and_failures": ["数据库连接失败"], "pending_work": ["恢复连接并重试测试"],
        "files_and_artifacts": ["app.py"], "evidence_refs": ["E1", "invented"], "next_steps": []}


def worker_state():
    return {"messages": [HumanMessage(content="原始要求：不要修改数据库结构"),
        AIMessage(content="", tool_calls=[{"id": "check1", "name": "run_test", "args": {}}]),
        ToolMessage(content="数据库连接失败\n" * 1000, tool_call_id="check1", name="run_test"),
        HumanMessage(content="任务补充，保留原文"), HumanMessage(content="最新消息原文")],
        "executor_model_run_limit": 10}


class CompactionPolicyTests(unittest.IsolatedAsyncioTestCase):
    def test_scheduler_default_disabled_does_not_evict_under_pressure(self):
        context = PlanningContextPack(current_time="now", user_request="继续")
        with patch.dict("os.environ", {"SCHEDULER_COMPACTION_ENABLED": "false"}):
            session = SchedulerConversation(context.scheduler_session, context, threshold=1)
        session.initialize()
        session.add("progress", "完整历史" * 100, protected=False)
        original = copy.deepcopy(session.records)
        session.compact()
        self.assertEqual(session.records, original)
        self.assertNotIn("compactions", session.data)

    def test_scheduler_opt_in_preserves_all_raw_pairs_and_summary(self):
        turns = [DialogueMessage(role=role, content=f"{role}{n}")
                 for n in range(4) for role in ("user", "assistant")]
        context = PlanningContextPack(current_time="now", user_request="继续",
            recent_dialogue=turns, conversation_summary='{"summary":"已有摘要"}')
        session = SchedulerConversation(context.scheduler_session, context, threshold=1, enabled=True)
        session.initialize()
        session.add("progress", "可删过程", protected=False)
        session.compact()
        text = json.dumps(session.wire(), ensure_ascii=False)
        for item in turns:
            self.assertIn(item.content, text)
        self.assertIn("已有摘要", text)
        self.assertNotIn("可删过程", text)

    def test_scheduler_upgrades_legacy_retention_without_losing_content(self):
        context = PlanningContextPack(current_time="now", user_request="继续")
        session = SchedulerConversation(context.scheduler_session, context, threshold=1, enabled=True)
        session.add("相关较早对话", "旧的第二第三轮", protected=False)
        session.add("辅助背景", "已有摘要", protected=False)
        session.initialize()
        session.compact()
        self.assertEqual(len(session.records), 2)
        self.assertTrue(all(item["protected"] for item in session.records))

    async def test_optional_container_repair_does_not_retry_or_invent_facts(self):
        model = SequenceModel([{"assistant_results": "助手报告尚未测试", "important_references": {"bad": 1}}])
        summary = await compress_dialogue(model, previous=None, turns=[], first_turn=1, max_chars=6000)
        self.assertEqual(summary.assistant_results, ["助手报告尚未测试"])
        self.assertEqual(summary.user_requirements, [])
        self.assertEqual(summary.important_references, [])
        self.assertEqual(len(model.requests), 1)

    async def test_empty_summary_retries_exactly_once_then_falls_back(self):
        model = SequenceModel([{}, {}, {"assistant_results": ["第三次不得调用"]}])
        with self.assertRaises(CompactionFailure) as caught:
            await compress_dialogue(model, previous=None, turns=[], first_turn=1, max_chars=6000)
        self.assertEqual(caught.exception.attempts, 2)
        self.assertEqual(len(model.requests), 2)
        self.assertIn("empty_substantive_summary", model.requests[1][-1]["content"])
        self.assertEqual(len(model.outputs), 1)

    async def test_retry_can_recover_without_rebuilding_source(self):
        model = SequenceModel([{}, {"assistant_results": ["尚未测试"]}])
        result = await compress_dialogue(model, previous=None, turns=[], first_turn=1, max_chars=6000)
        self.assertEqual(result.assistant_results, ["尚未测试"])
        self.assertEqual(model.requests[1][:-1], model.requests[0])

    async def test_invalid_nonempty_core_is_not_silently_dropped(self):
        model = SequenceModel([{"observed_results": {"bad": "do not drop"}}, {}])
        with self.assertRaises(CompactionFailure):
            await arun_compaction(model, WorkerExecutionSummary, [],
                core_fields=("observed_results", "errors_and_failures", "pending_work"), max_chars=6000)
        self.assertIn("invalid_core_field:observed_results", model.requests[1][-1]["content"])

    async def test_worker_strict_json_keeps_task_control_and_real_evidence(self):
        state = worker_state()
        control = AIMessage(content="", tool_calls=[{"id": "review1", "name": "submit_code_for_review", "args": {}}])
        control_result = ToolMessage(content="审核中", tool_call_id="review1", name="submit_code_for_review")
        state["messages"][3:3] = [control, control_result]
        model = SequenceModel([worker_payload()])
        middleware = WorkerCompactionMiddleware(model, threshold=1, keep_messages=2, keep_tokens=1, enabled=True)
        update = await middleware.abefore_model(state, None)
        kept = update["messages"][1:]
        self.assertIn(state["messages"][0], kept)
        self.assertIn(control, kept)
        self.assertIn(control_result, kept)
        self.assertEqual(kept[-2:], state["messages"][-2:])
        generated = next(m for m in kept if m.additional_kwargs.get("worker_compaction_summary"))
        data = json.loads(generated.content.split("\n", 1)[1])
        self.assertEqual(data["evidence_refs"], ["E1"])
        self.assertNotIn("invented", generated.content)
        self.assertEqual(update["worker_compaction_calls_used"], 1)
        self.assertEqual(model.bindings[0][1], {"method": "json_schema", "strict": True, "include_raw": True})
        self.assertNotIn('"properties"', model.requests[0][0]["content"])

    async def test_worker_two_failures_do_not_replace_and_defer_same_source(self):
        state = worker_state()
        model = SequenceModel([{}, {}])
        middleware = WorkerCompactionMiddleware(model, threshold=1, keep_messages=2, keep_tokens=1, enabled=True)
        update = await middleware.abefore_model(state, None)
        self.assertNotIn("messages", update)
        self.assertEqual(update["worker_archived_messages"], [])
        self.assertEqual(update["worker_compaction_calls_used"], 2)
        next_state = {**state, **update}
        self.assertIsNone(await middleware.abefore_model(next_state, None))
        self.assertEqual(len(model.requests), 2)
        next_state["messages"] = [*state["messages"], HumanMessage(content="新的执行输入")]
        model.outputs.append(worker_payload())
        self.assertIsNotNone(await middleware.abefore_model(next_state, None))
        self.assertEqual(len(model.requests), 3)

    def test_worker_sync_retry_counts_budget_and_retains_originals(self):
        state = worker_state()
        model = SequenceModel([{}, worker_payload()])
        middleware = WorkerCompactionMiddleware(model, threshold=1, keep_messages=2, keep_tokens=1, enabled=True)
        update = middleware.before_model(state, None)
        self.assertEqual(update["worker_compaction_calls_used"], 2)
        self.assertIn("messages", update)
        state["executor_model_run_limit"] = 3
        self.assertIsNone(middleware.before_model(state, None))

    async def test_previous_worker_json_is_recompressed_not_accumulated(self):
        state = worker_state()
        model = SequenceModel([worker_payload(), worker_payload()])
        middleware = WorkerCompactionMiddleware(model, threshold=1, keep_messages=2, keep_tokens=1, enabled=True)
        update = await middleware.abefore_model(state, None)
        state = {**state, **update, "messages": [*update["messages"][1:],
            AIMessage(content="", tool_calls=[{"id": "check2", "name": "run_test", "args": {}}]),
            ToolMessage(content="第二次连接也失败\n" * 1000, tool_call_id="check2", name="run_test"),
            HumanMessage(content="近期1"), HumanMessage(content="近期2")]}
        update = await middleware.abefore_model(state, None)
        self.assertEqual(sum(bool(m.additional_kwargs.get("worker_compaction_summary"))
                             for m in update["messages"][1:]), 1)
        source = json.loads(model.requests[1][1]["content"])
        self.assertEqual(len(source["previous_summaries"]), 1)
        self.assertNotEqual(source["execution_messages"], [])

    async def test_worker_sdk_sends_strict_schema_and_repairs_optional_slots_once(self):
        import httpx
        from model_clients import CompatibleChatModel
        payloads = []

        def serve(request):
            payloads.append(json.loads(request.content))
            # Deliberately malformed optional slots test local Harness repair.
            output = {"observed_results": ["语法通过"], "files_and_artifacts": None,
                      "evidence_refs": {"invalid": 1}}
            return httpx.Response(200, json={"id": "offline-worker", "object": "chat.completion",
                "created": 0, "model": "qwen3.8-flash", "choices": [{"index": 0,
                    "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(output)}}]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
            model = CompatibleChatModel(model="qwen3.8-flash", api_key="offline-placeholder",
                base_url="https://example.invalid/v1", http_async_client=client)
            result, attempts = await arun_compaction(model, WorkerExecutionSummary,
                [{"role": "user", "content": "压缩执行材料，返回JSON"}],
                core_fields=("observed_results", "errors_and_failures", "pending_work"), max_chars=6000)
        self.assertEqual(attempts, 1)
        self.assertEqual(len(payloads), 1)
        self.assertEqual(result.files_and_artifacts, [])
        contract = payloads[0]["response_format"]
        self.assertEqual(contract["type"], "json_schema")
        self.assertTrue(contract["json_schema"]["strict"])
        self.assertFalse(contract["json_schema"]["schema"]["additionalProperties"])
        self.assertEqual(set(contract["json_schema"]["schema"]["required"]), set(WorkerExecutionSummary.model_fields))

    def test_scheduler_enable_setting_is_loaded_and_validated(self):
        from config import _load_planning_settings
        with patch.dict("os.environ", {"SCHEDULER_COMPACTION_ENABLED": "false"}):
            self.assertFalse(_load_planning_settings().scheduler_compaction_enabled)
        with patch.dict("os.environ", {"SCHEDULER_COMPACTION_ENABLED": "true"}):
            self.assertTrue(_load_planning_settings().scheduler_compaction_enabled)
        with patch.dict("os.environ", {"SCHEDULER_COMPACTION_ENABLED": "invalid"}):
            with self.assertRaises(RuntimeError):
                _load_planning_settings()

    def test_separate_compaction_settings_default_off_and_validate(self):
        from config import _load_planning_settings, load_worker_compaction_settings
        with patch.dict("os.environ", {"CONVERSATION_COMPACTION_ENABLED": "false", "WORKER_COMPACTION_ENABLED": "false"}):
            self.assertFalse(_load_planning_settings().conversation_compaction_enabled)
            self.assertFalse(load_worker_compaction_settings().enabled)
        with patch.dict("os.environ", {"WORKER_COMPACTION_ENABLED": "true", "WORKER_SUMMARY_KEEP_MESSAGES": "5"}):
            settings = load_worker_compaction_settings()
            self.assertTrue(settings.enabled)
            self.assertEqual(settings.keep_messages, 5)
        with patch.dict("os.environ", {"WORKER_COMPACTION_ENABLED": "invalid"}):
            with self.assertRaises(RuntimeError):
                load_worker_compaction_settings()


if __name__ == "__main__":
    unittest.main()
