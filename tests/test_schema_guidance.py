"""Verify guidance in actual prompt assembly, without provider requests."""
import copy
import json
import os
import unittest

os.environ["PHOENIX_TRACING_ENABLED"] = "false"
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field
from hard_planning import _invoke_structured, _invoke_scheduler_stage
from planning_models import PlanningContextPack, SupervisorDecision
from schema_utils import compact_schema
from memory_extraction_models import compact_schema as memory_compact


class ExampleAnswer(BaseModel):
    title: str = Field(description="Keep this business title.", examples=["Report"])
    payload: dict = Field(description="Preserve the example verbatim.", examples=[{"title": "Example title"}])


class CaptureModel:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    def with_structured_output(self, schema, **kwargs):
        async def respond(messages):
            self.requests.append(copy.deepcopy(messages))
            return {"parsed": schema.model_validate(self.payload)}
        return RunnableLambda(respond)


class SchemaGuidanceTests(unittest.IsolatedAsyncioTestCase):
    def test_literal_titles_and_schema_property_names_survive_both_paths(self):
        schema = ExampleAnswer.model_json_schema()
        before = copy.deepcopy(schema)
        for compact in (compact_schema, memory_compact):
            result = compact(schema)
            self.assertNotIn("title", result)
            self.assertNotIn("title", result["properties"]["title"])
            self.assertEqual(result["required"], schema["required"])
            for name in ("title", "payload"):
                for key in ("description", "examples"):
                    self.assertEqual(result["properties"][name][key], schema["properties"][name][key])
        self.assertEqual(schema, before)

    async def test_standalone_final_request_retains_examples_and_guidance(self):
        model = CaptureModel({"title": "Report", "payload": {}})
        await _invoke_structured(model, prompt="Answer briefly", output_schema=ExampleAnswer,
                                 trace_name="schema-test", fallback_factory=lambda error: None)
        sent = json.loads(model.requests[0][0]["content"].split("\nSchema:", 1)[1])
        self.assertEqual(sent["properties"]["payload"]["examples"], [{"title": "Example title"}])
        self.assertEqual(sent["properties"]["title"]["description"], "Keep this business title.")

    async def test_real_scheduler_request_retains_nested_contract_rules_and_reuses_protocol(self):
        model = CaptureModel({"action": "FINAL", "final_answer": "Done"})
        context = PlanningContextPack(current_time="now", user_request="Check schema")
        for _ in range(2):
            await _invoke_scheduler_stage(model, context=context, stage="supervisor", event={},
                output_schema=SupervisorDecision, trace_name="schema-test", fallback_factory=lambda error: None)
        for request in model.requests:
            protocols = [json.loads(m["content"]) for m in request if m["content"].startswith('{"协议":')]
            self.assertEqual(len(protocols), 1)
            sent = json.loads(protocols[0]["要求"].split("\nSchema:", 1)[1])
            original = SupervisorDecision.model_json_schema()
            for definition, field in (("PlanStep", "code_task"), ("StepArtifactOutput", "target_path")):
                self.assertEqual(sent["$defs"][definition]["properties"][field]["description"],
                                 original["$defs"][definition]["properties"][field]["description"])
        self.assertEqual(model.requests[0], model.requests[1])


if __name__ == "__main__":
    unittest.main()
