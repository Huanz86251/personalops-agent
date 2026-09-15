import json
import unittest
from unittest.mock import patch
from trace_overview import build_overview, task_overview, tool_purpose
import test_runtime_tracing as fixture


class OverviewTests(unittest.TestCase):
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


if __name__=='__main__':unittest.main()
