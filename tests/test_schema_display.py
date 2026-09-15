import json
import unittest
from copy import deepcopy
from trace_chat import chat_attributes, readable_content

class SchemaDisplayTests(unittest.TestCase):
    def test_nested_schema_is_pretty_and_original_is_unchanged(self):
        schema={"properties":{"action":{"enum":["FINAL","PLAN"]}},"description":"line 1\nline 2"}
        original=json.dumps({"协议":"supervisor@abc","要求":"按规则填写。\nSchema:"+json.dumps(schema)},ensure_ascii=False)
        inputs={"messages":[[{"role":"human","content":original}]]};before=deepcopy(inputs)
        text=chat_attributes(inputs,{})["llm.input_messages.0.message.content"]
        self.assertIn('### Schema',text)
        self.assertIn(json.dumps(schema,ensure_ascii=False,indent=2),text)
        self.assertEqual(inputs,before)
    def test_plain_and_malformed_are_preserved(self):
        for text in ('plain text','{broken', '123'):
            self.assertEqual(readable_content(text),text)
        self.assertIn('无法解析',readable_content(json.dumps({"协议":"x","要求":"Schema:{bad"})))
    def test_data_fence_cannot_be_closed(self):
        text=readable_content({"example":"```"})
        self.assertIn('````json',text)
