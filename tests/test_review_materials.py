import asyncio
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from reporting.materials import build_reader, ReviewWithReads
from langchain_core.messages import AIMessage

class MaterialTests(unittest.TestCase):
    def test_registered_file_only_hash_and_pagination(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)/"note.txt"; data = b"x"*5000; p.write_bytes(data)
            item=NS(review_ref="candidate1",storage_path=str(p),sha256=hashlib.sha256(data).hexdigest())
            reader, refs=build_reader(NS(attempts=[NS(resolved_artifacts=[item],resolved_evidence=[])]))
            import json
            first=json.loads(reader.invoke({"reference":"M1"}))
            self.assertEqual(first["next_offset"],4000)
            self.assertEqual(len(json.loads(reader.invoke({"reference":"M1","offset":4000}))["content"]),1000)
            with self.assertRaises(ValueError):reader.invoke({"reference":str(p)})
            p.write_text("changed")
            with self.assertRaises(ValueError):reader.invoke({"reference":"M1"})
    def test_read_then_reserved_final_round(self):
        packet=NS(attempts=[NS(resolved_artifacts=[],resolved_evidence=[NS(tool_call_id="e1",result="fact")])])
        class Model:
            def bind_tools(self, tools):return self
            async def ainvoke(self,messages):return AIMessage(content="",tool_calls=[{"id":"r1","name":"read_review_material","args":{"reference":"M1"}}])
        class Final:
            async def ainvoke(self,messages):return {"parsed":{"done":True}}
        wrapper=ReviewWithReads(Model(),Final(),packet,3);messages=[]
        self.assertTrue(asyncio.run(wrapper.ainvoke(messages))["material_read"])
        self.assertEqual(messages[-1].tool_call_id,"r1")
        self.assertEqual(messages[-1].status, "success")
        self.assertIn('fact', messages[-1].content)
        self.assertTrue(asyncio.run(wrapper.ainvoke(messages))["parsed"]["done"])
