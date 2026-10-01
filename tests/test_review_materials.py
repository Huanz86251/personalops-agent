import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from reporting.materials import build_material_reader, ReviewWithReads
from langchain_core.messages import AIMessage
from step_user_question import STEP_USER_QUESTIONS, ask_user_runtime_scope

class MaterialTests(unittest.TestCase):
    def test_registered_file_only_hash_and_pagination(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)/"note.txt"; data = b"x"*5000; p.write_bytes(data)
            item=NS(review_ref="candidate1",storage_path=str(p),sha256=hashlib.sha256(data).hexdigest())
            reader, refs=build_material_reader(NS(attempts=[NS(resolved_artifacts=[item],resolved_evidence=[])]))
            first=json.loads(reader.invoke({"requests":[{"reference":"M1"}]}))["materials"][0]
            self.assertEqual(first["next_offset"],4000)
            second=json.loads(reader.invoke({"requests":[{"reference":"M1","offset":4000}]}))["materials"][0]
            self.assertEqual(len(second["content"]),1000)
            with self.assertRaises(ValueError):reader.invoke({"requests":[{"reference":str(p)}]})
            p.write_text("changed")
            with self.assertRaises(ValueError):reader.invoke({"requests":[{"reference":"M1"}]})
    def test_read_then_reserved_final_round(self):
        packet=NS(attempts=[NS(resolved_artifacts=[],resolved_evidence=[NS(tool_call_id="e1",result="fact")])])
        class Model:
            def bind_tools(self, tools):return self
            async def ainvoke(self,messages):return AIMessage(content="",tool_calls=[{"id":"r1","name":"read_review_materials","args":{"requests":[{"reference":"M1"}]}}])
        class Final:
            async def ainvoke(self,messages):return {"parsed":{"done":True}}
        wrapper=ReviewWithReads(Model(),Final(),packet,3);messages=[]
        self.assertTrue(asyncio.run(wrapper.ainvoke(messages))["material_read"])
        self.assertEqual(messages[-1].tool_call_id,"r1")
        self.assertEqual(messages[-1].status, "success")
        self.assertIn('fact', messages[-1].content)
        self.assertTrue(asyncio.run(wrapper.ainvoke(messages))["parsed"]["done"])

    def test_batch_reader_returns_multiple_registered_pages_once(self):
        packet = NS(attempts=[NS(
            resolved_artifacts=[],
            resolved_evidence=[
                NS(tool_call_id="e1", result="first fact"),
                NS(tool_call_id="e2", result="second fact"),
                NS(tool_call_id="e3", result="third fact"),
            ],
        )])
        reader, refs = build_material_reader(packet)
        result = json.loads(reader.invoke({"requests": [
            {"reference": "M1"},
            {"reference": "M2"},
            {"reference": "M3"},
        ]}))
        self.assertEqual(len(refs), 3)
        self.assertEqual(
            [item["content"] for item in result["materials"]],
            ["first fact", "second fact", "third fact"],
        )

    def test_review_wrapper_exposes_only_batch_reader(self):
        packet = NS(attempts=[NS(
            resolved_artifacts=[],
            resolved_evidence=[NS(tool_call_id="e1", result="fact")],
        )])
        class Model:
            tool_names = []
            def bind_tools(self, tools):
                self.tool_names = [tool.name if hasattr(tool, "name") else tool.__name__ for tool in tools]
                return self
            async def ainvoke(self, messages):
                return AIMessage(content="", tool_calls=[{
                    "id": "batch-1",
                    "name": "read_review_materials",
                    "args": {"requests": [{"reference": "M1"}]},
                }])
        class Final:
            async def ainvoke(self, messages): return {"parsed": {"done": True}}
        model = Model()
        wrapper = ReviewWithReads(model, Final(), packet, 3)
        messages = []
        self.assertTrue(asyncio.run(wrapper.ainvoke(messages))["material_read"])
        self.assertEqual(
            set(model.tool_names),
            {"read_review_materials", "StepReport"},
        )
        self.assertIn("fact", messages[-1].content)

    def test_reporter_can_pause_for_user_and_continue_same_messages(self):
        packet = NS(attempts=[NS(resolved_artifacts=[], resolved_evidence=[])])

        class Model:
            def __init__(self):
                self.calls = 0
                self.saw_answer = False

            def bind_tools(self, tools):
                self.tools = tools
                return self

            async def ainvoke(self, messages):
                self.calls += 1
                if self.calls == 1:
                    return AIMessage(content="", tool_calls=[{
                        "id": "ask-review-1",
                        "name": "ask_user",
                        "args": {"question": "是否接受这个范围？"},
                    }])
                self.saw_answer = any(
                    getattr(item, "type", "") == "tool" and "接受" in item.content
                    for item in messages
                )
                return AIMessage(content="", tool_calls=[{
                    "id": "report-1",
                    "name": "StepReport",
                    "args": {"done": True},
                }])

        class Final:
            async def ainvoke(self, messages):
                return {"parsed": {"done": False}}

        async def scenario():
            async def notify(chat_id, request):
                asyncio.get_running_loop().call_soon(
                    STEP_USER_QUESTIONS.consume_user_reply,
                    chat_id,
                    "接受这个范围",
                )

            STEP_USER_QUESTIONS.configure_notifier(notify)
            try:
                model = Model()
                with ask_user_runtime_scope(event_id="evt-review", chat_id="chat-review"):
                    wrapper = ReviewWithReads(model, Final(), packet, 4)
                    messages = []
                    first = await wrapper.ainvoke(messages)
                    second = await wrapper.ainvoke(messages)
                return model, first, second
            finally:
                STEP_USER_QUESTIONS.configure_notifier(None)

        model, first, second = asyncio.run(scenario())
        self.assertTrue(first["human_answer"])
        self.assertEqual(second["parsed"], {"done": True})
        self.assertTrue(model.saw_answer)
