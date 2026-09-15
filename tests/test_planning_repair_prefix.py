"""Offline validation recovery and append-only message assembly checks."""
import copy
import json
import os
import unittest

os.environ["PHOENIX_TRACING_ENABLED"] = "false"
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, ValidationError
from hard_planning import _invoke_structured, _build_validation_repair_message
from planning_models import SupervisorDecision


class Answer(BaseModel):
    count: int


class Scripted:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def with_structured_output(self, schema, **kwargs):
        async def respond(messages):
            self.requests.append(copy.deepcopy(messages))
            value = next(self.responses)
            if isinstance(value, Exception):
                raise value
            return value
        return RunnableLambda(respond)


class RepairTests(unittest.IsolatedAsyncioTestCase):
    async def test_original_prefix_and_caller_history_unchanged(self):
        raw = '{"count":"wrong"}'
        model = Scripted([{"raw": AIMessage(content=raw), "parsing_error": ValueError("plan" * 3000 + "bad count")},
                          {"parsed": {"count": 2}}])
        prefix = [{"role": "system", "content": "fixed skills and schema"},
                  {"role": "user", "content": "original task"}]
        before = copy.deepcopy(prefix)
        result = await _invoke_structured(model, prompt="fixed", output_schema=Answer, trace_name="test",
            fallback_factory=lambda error: Answer(count=-1), scheduler_messages=prefix)
        self.assertEqual(prefix, before)
        self.assertEqual(model.requests[1][:-2], model.requests[0])
        self.assertEqual(model.requests[1][-2], {"role": "assistant", "content": raw})
        self.assertIn("count", model.requests[1][-1]["content"])
        self.assertNotIn("planplan", model.requests[1][-1]["content"])
        self.assertFalse(result.used_fallback)
        self.assertEqual(result.model_rounds_used, 2)

    async def test_timeout_does_not_consume_schema_repair_rounds(self):
        model = Scripted([TimeoutError("timed out"), {"parsed": {"count": 1}}])
        result = await _invoke_structured(
            model, prompt="fixed prompt", output_schema=Answer, trace_name="test",
            fallback_factory=lambda error: Answer(count=-1),
        )
        self.assertEqual(len(model.requests), 1)
        self.assertTrue(result.used_fallback)
        self.assertEqual(result.validation_retry_count, 0)

    async def test_failed_retry_remains_bounded(self):
        model = Scripted([{"parsed": {"count": "bad"}} for _ in range(4)])
        result = await _invoke_structured(model, prompt="fixed", output_schema=Answer, trace_name="test",
                                         fallback_factory=lambda error: Answer(count=-1))
        self.assertEqual(len(model.requests), 4)
        self.assertEqual(result.validation_retry_count, 3)
        self.assertTrue(result.used_fallback)

    def test_two_cross_field_errors_survive_long_sdk_wrapper(self):
        plan = {"action": "PLAN", "plan_objective": "Review and implement", "plan_success_criteria": ["Delivered"],
                "steps": [
                    {"step_id": 1, "objective": "Research", "success_criteria": ["Evidence"], "worker_kind": "WEB", "execution_mode": "SINGLE",
                     "artifact_outputs": [{"output_id": "reference", "description": "Notes", "disposition": "INTERNAL_HANDOFF", "target_path": "notes.md"}]},
                    {"step_id": 2, "objective": "Implement", "success_criteria": ["Works"], "worker_kind": "CODE", "execution_mode": "SINGLE",
                     "code_task": {"requirements": [{"requirement_id": "works", "statement": "Works"}], "validation_expectations": ["test"]},
                     "artifact_outputs": [{"output_id": "code", "description": "Code", "disposition": "USER_DELIVERABLE", "target_path": "app.py"}]}]}
        with self.assertRaises(ValidationError) as raised:
            SupervisorDecision.model_validate(plan)
        wrapper = ValueError("full plan " * 3000)
        wrapper.__cause__ = raised.exception
        message = _build_validation_repair_message(
            schema_name="SupervisorDecision", output_schema=SupervisorDecision, error=wrapper,
        )
        self.assertIn("steps.0", message)
        self.assertIn("steps.1", message)
        self.assertIn("INTERNAL_HANDOFF artifact不得提供target_path", message)
        self.assertIn("CODE Step使用code_task", message)
        self.assertNotIn("full plan", message)
        self.assertLessEqual(len(message), 2600)


if __name__ == "__main__":
    unittest.main()
