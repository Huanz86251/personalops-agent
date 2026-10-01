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
    @staticmethod
    def redundant_single_plan(*, assignments=1, legacy_entity="file"):
        worker_assignments = [
            {
                "assignment_key": f"delivery_{index}",
                "objective": f"Deliver the selected file, branch {index}",
            }
            for index in range(1, assignments + 1)
        ]
        return {
            "action": "PLAN",
            "plan_objective": "Locate and deliver the requested resume",
            "plan_success_criteria": ["The selected file is delivered"],
            "steps": [
                {
                    "step_id": 1,
                    "objective": "Locate and deliver the requested resume",
                    "success_criteria": ["The selected file is delivered"],
                    "worker_kind": "GENERAL",
                    "execution_mode": "SINGLE",
                    "worker_assignments": worker_assignments,
                    "target_selection": {
                        "target_entity": "file",
                        "result_entity": legacy_entity,
                        "effect_mode": "READ_ONLY",
                        "sets": [
                            {
                                "set_id": "A",
                                "definition": "Resume files in Downloads",
                                "condition_owner": "file",
                            }
                        ],
                        "operation": "DIRECT",
                        "operands": ["A"],
                    },
                }
            ],
        }

    async def test_supervisor_deterministic_shape_repair_avoids_model_retry(self):
        model = Scripted([{"parsed": self.redundant_single_plan()}])
        fallback = SupervisorDecision.model_validate(
            {
                "plan_objective": "Fallback",
                "plan_success_criteria": ["Stop"],
                "steps": [
                    {
                        "step_id": 1,
                        "objective": "Stop",
                        "success_criteria": ["Stop"],
                    }
                ],
            }
        )
        result = await _invoke_structured(
            model,
            prompt="fixed",
            output_schema=SupervisorDecision,
            trace_name="test",
            fallback_factory=lambda _error: fallback,
        )
        self.assertEqual(len(model.requests), 1)
        self.assertEqual(result.validation_retry_count, 0)
        self.assertFalse(result.used_fallback)
        step = result.output.steps[0]
        self.assertEqual(step.worker_assignments, [])
        self.assertEqual(step.target_selection.sets[0].result_entity, "file")

    def test_supervisor_shape_repair_does_not_guess_ambiguous_semantics(self):
        with self.assertRaises(ValidationError):
            SupervisorDecision.model_validate(
                self.redundant_single_plan(assignments=2)
            )
        with self.assertRaises(ValidationError):
            SupervisorDecision.model_validate(
                self.redundant_single_plan(legacy_entity="document")
            )

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
