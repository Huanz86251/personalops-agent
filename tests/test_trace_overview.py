import asyncio
import json
import unittest
from unittest.mock import patch
from trace_overview import (
    build_conversation_overview, build_overview, conversation_overview,
    capture_task, task_overview, tool_purpose,
)
import test_runtime_tracing as fixture


class OverviewTests(unittest.TestCase):
    def test_conversation_overview_groups_parallel_workers_under_real_parent(self):
        rows = [
            {"span_id": "scheduler", "parent_id": "turn", "name": "🎛️ Scheduler",
             "start_time": 1, "attributes": {}},
            {"span_id": "dispatch", "parent_id": "scheduler",
             "name": "Scheduler / Dispatch Step", "start_time": 2,
             "attributes": {"output.value": json.dumps({"current_step_attempt": 1,
                 "current_step": {"step_id": 1, "worker_kind": "WEB"}})}},
            {"span_id": "web-a", "parent_id": "dispatch",
             "name": "🌐 Web Agent / Step 1", "start_time": 3, "attributes": {}},
            {"span_id": "web-b", "parent_id": "dispatch",
             "name": "🌐 Web Agent / Step 1", "start_time": 4, "attributes": {}},
            {"span_id": "llm-a", "parent_id": "web-a", "name": "LLM / Web Agent",
             "start_time": 5, "attributes": {"openinference.span.kind": "LLM"}},
        ]
        md, root = build_conversation_overview(
            {"instruction": "Find a movie"},
            {"answer": "No confirmed screening", "planning": {
                "final_status": "FAILED", "planned_steps": [{"step_id": 1,
                    "worker_kind": "WEB", "objective": "Find screenings"}],
                "completed_step_reports": [{"step_id": 1, "status": "BLOCKED",
                    "summary": "No showtime evidence"}]}}, rows)
        self.assertIn("Step 1 · WEB · BLOCKED", md)
        self.assertIn("并行 Worker 按真实父子 Span 归组", md)
        self.assertIn("├─ 🌐 Web Agent / Step 1", md)
        self.assertIn("└─ 🌐 Web Agent / Step 1", md)
        self.assertEqual(len(root["04 模型调用索引"]), 1)
        self.assertNotIn("full worker input", str(root))

    def test_conversation_usage_keeps_cache_separate_from_input(self):
        rows = [{"span_id": "llm", "parent_id": "turn", "name": "LLM / Web Agent",
                 "attributes": {"openinference.span.kind": "LLM", "runtime.model_role": "web",
                                "trace_overview.usage": {"input_tokens": 100,
                                    "output_tokens": 20, "cache_read_tokens": 80,
                                    "cost_cny": 0.01}}}]
        md, _ = build_conversation_overview({"instruction": "Find a movie"},
                                             {"answer": "Done"}, rows)
        self.assertIn("| Web Agent | 1 | 100 | 20 | 80 | ¥0.0100 |", md)
        self.assertIn("缓存读取属于输入，不重复相加", md)

    def test_complete_context_and_unknown_usage(self):
        content = "original context " * 1000
        row = {"name":"LLM / Code Worker", "attributes":{
            "openinference.span.kind":"LLM", "runtime.model_role":"code",
            "input.value":json.dumps({"messages":[[{"content":content}]]}),
            "output.value":json.dumps({"responses":[],"usage":{"input_tokens":None}})}}
        md, root = build_overview({"instruction":"user task"}, {"scheduler_status":"FAILED"}, [row], preview=True)
        self.assertIn("不计入准确率", md)
        self.assertIn("未知", md)
        self.assertEqual(root["03 按角色查看每次完整上下文与输出"]["code"][0]["输入原文"]["messages"][0][0]["content"],content)

    def test_tool_purpose_and_empty_trace(self):
        self.assertEqual(tool_purpose("appworld_verify",{"code":"print(apis.api_docs.show_api_doc())"}),"AppWorld / API 文档")
        md, _ = build_overview({}, {}, [])
        self.assertIn("未评分",md)

    def test_overview_failure_does_not_mask_task_error(self):
        with patch("trace_overview.build_overview", side_effect=ValueError("presentation error")):
            with self.assertRaisesRegex(RuntimeError,"task error"):
                with task_overview(None, {}, {}):
                    raise RuntimeError("task error")


class CaptureTests(unittest.TestCase):
    setUp = fixture.TraceTests.setUp
    tearDown = fixture.TraceTests.tearDown
    def test_live_capture_has_child_context_at_root(self):
        from observability import trace_span
        from trace_callbacks import RuntimeTraceCallback
        from langchain_core.messages import HumanMessage, AIMessage
        from langchain_core.outputs import LLMResult, ChatGeneration
        from uuid import uuid4
        with patch("opentelemetry.trace.get_tracer_provider", return_value=self.provider):
            with trace_span("task") as root, task_overview(root,{"instruction":"original user task"},{"scheduler_status":"COMPLETED"}):
                cb=RuntimeTraceCallback();key=uuid4()
                cb.on_chat_model_start({},[[HumanMessage(content="full worker input")]],run_id=key)
                cb.on_llm_end(LLMResult(generations=[[ChatGeneration(message=AIMessage(content="done"))]]),run_id=key)
        root = self.exporter.get_finished_spans()[-1]
        self.assertIn("full worker input",root.attributes['input.value'])
        self.assertIn("任务总览",root.attributes['output.value'])

    def test_conversation_root_has_readable_overview_without_replacing_appworld(self):
        from observability import trace_span, set_span_output
        with patch("opentelemetry.trace.get_tracer_provider", return_value=self.provider):
            with trace_span("Conversation") as root, conversation_overview(
                root, {"instruction": "你好"}, {"answer": "你好", "status": "COMPLETED"}
            ):
                set_span_output(root, {"assistant_reply": "你好", "planning": {"final_status": "COMPLETED"}})
                with trace_span("🎛️ Scheduler"):
                    with trace_span("Scheduler / Plan"):
                        pass
        spans = self.exporter.get_finished_spans()
        root = spans[-1]
        self.assertIn("对话任务总览", root.attributes["output.value"])
        self.assertIn("Scheduler / Plan", root.attributes["output.value"])
        self.assertIn("你好", root.attributes["input.value"])
        details = next(span for span in spans if span.name == "Conversation / Result Details")
        self.assertEqual(json.loads(details.attributes["output.value"])["assistant_reply"], "你好")

        with trace_span("AppWorld Conversation") as unchanged, conversation_overview(
            unchanged, {}, {}, enabled=False
        ):
            set_span_output(unchanged, {"original": True})
        self.assertEqual(json.loads(self.exporter.get_finished_spans()[-1].attributes[
            "output.value"]), {"original": True})

    def test_parallel_conversations_do_not_mix_captured_spans(self):
        from observability import trace_span

        async def turn(label):
            with trace_span(label) as root, capture_task(root) as captured:
                await asyncio.sleep(0)
                with trace_span(label + " / child"):
                    await asyncio.sleep(0)
            return [row["name"] for row in captured["rows"]]

        async def both():
            return await asyncio.gather(turn("first"), turn("second"))

        with patch("opentelemetry.trace.get_tracer_provider", return_value=self.provider):
            first, second = asyncio.run(both())
        self.assertEqual(first, ["first / child"])
        self.assertEqual(second, ["second / child"])


if __name__=='__main__':unittest.main()
