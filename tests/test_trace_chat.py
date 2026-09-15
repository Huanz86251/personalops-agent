import unittest
from uuid import uuid4
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.outputs import LLMResult, ChatGeneration
from test_runtime_tracing import TraceTests
from trace_callbacks import RuntimeTraceCallback

class ChatTests(unittest.TestCase):
    setUp = TraceTests.setUp
    tearDown = TraceTests.tearDown

    def test_code_display_preserves_python_and_raw_input(self):
        import json
        cb = RuntimeTraceCallback(); key = uuid4()
        code = 'print("hello")\nprint("literal \\n")\n'
        raw = json.dumps({"code": code})
        cb.on_tool_start({"name": "appworld_execute"}, raw, run_id=key)
        cb.on_tool_start({"name": "appworld_execute"}, raw, run_id=key)
        cb.on_tool_end("done", run_id=key)
        spans = self.exporter.get_finished_spans()
        view = next(s for s in spans if s.attributes.get("audit.display_only"))
        self.assertEqual(view.attributes["input.value"], code)
        self.assertEqual(view.attributes["input.mime_type"], "text/plain")
        self.assertEqual(len(spans), 2)
        parent = next(s for s in spans if s.attributes.get("openinference.span.kind") == "TOOL")
        self.assertEqual(view.parent.span_id, parent.context.span_id)
        self.assertIn("code", parent.attributes["input.value"])
        self.assertNotIn("llm.token_count.total", view.attributes)

    def test_child_messages_preserve_tool_ids_and_do_not_duplicate_usage(self):
        cb = RuntimeTraceCallback(); key = uuid4()
        cb.on_chat_model_start({}, [[HumanMessage(content="context"), ToolMessage(content="result",tool_call_id="call-1")]],run_id=key)
        answer=AIMessage(content="answer", additional_kwargs={"reasoning_content":"returned explanation"}, usage_metadata={"input_tokens":2,"output_tokens":3,"total_tokens":5})
        result=LLMResult(generations=[[ChatGeneration(message=answer)]])
        cb.on_llm_end(result,run_id=key); cb.on_llm_end(result,run_id=key)
        spans=self.exporter.get_finished_spans()
        self.assertEqual(len(spans),2)
        view=next(s for s in spans if s.name=="Messages / 完整对话")
        parent=next(s for s in spans if s.attributes.get("openinference.span.kind")=="LLM")
        self.assertEqual(view.parent.span_id,parent.context.span_id)
        self.assertEqual(view.attributes["llm.input_messages.1.message.tool_call_id"],"call-1")
        self.assertIn("returned explanation", view.attributes["llm.output_messages.0.message.content"])
        self.assertTrue(view.attributes["llm.output_messages.0.message.content"].endswith("answer"))
        self.assertIn("returned explanation",view.attributes["output.value"])
        self.assertNotIn("llm.token_count.total",view.attributes)
        self.assertEqual(parent.attributes["llm.token_count.total"],5)
        self.assertEqual(cb.chat_inputs,{})
