"""No-provider regression tests for the real PersonalOps planning graph adapter."""
import os
os.environ["PHOENIX_TRACING_ENABLED"] = "false"

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch
from uuid import uuid4
from pydantic import Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from evals.appworld.adapter import (
    EvaluationBudgetExceeded, UsageMeter, run_personalops, make_discover_tool, make_execute_tool,
)
from hard_planning import _format_hard_context
from planning_models import PlanningContextPack, PlanningReplacementContext


class ScriptedModel(BaseChatModel):
    seen_systems: list[str] = Field(default_factory=list)
    seen_tools: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "scripted-test-only"

    def bind_tools(self, tools, **kwargs):
        self.seen_tools.extend(t.name for t in tools)
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen_systems.append("\n".join(str(m.content) for m in messages))
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        if any(getattr(m, "name", "") == "appworld_execute" for m in tool_messages):
            msg = AIMessage(content="", tool_calls=[{
                "name": "report_general_result", "args": {"result": {
                    "status": "COMPLETED", "summary": "Observed 42",
                    "evidence_tool_call_ids": ["E2"],
                }}, "id": "test-report", "type": "tool_call",
            }])
        elif tool_messages:
            msg = AIMessage(content="", tool_calls=[{
                "name": "appworld_execute", "args": {"source_refs": ["E1"],
                    "reason": "The discovered fixture API is available.",
                    "action_phase": "OTHER", "set_bindings": [],
                    "code": "print(6 * 7)"},
                "id": "test-tool", "type": "tool_call",
            }])
        else:
            msg = AIMessage(content="", tool_calls=[{
                "name": "appworld_discover", "args": {"source_refs": ["MODEL"],
                    "reason": "Discover fixture API.",
                    "code": "print(apis.api_docs.show_app_descriptions())"},
                "id": "discover-tool", "type": "tool_call",
            }])
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def with_structured_output(self, schema, **kwargs):
        payloads = {
            "SupervisorDecision": {
                "action": "PLAN", "plan_objective": "Observe a number",
                "plan_success_criteria": ["Observe 42"],
                "steps": [{"step_id": 1, "objective": "Observe 42", "success_criteria": ["Observe 42"]}],
            },
            "StepReport": {
                "step_id": 1, "status": "COMPLETED", "summary": "Observed 42",
                "stop_reason": "Observation received",
                "criterion_results": [{"criterion": "Observe 42", "status": "MET", "evidence": ["42"]}],
                "confirmed_results": ["42"], "evidence": ["42"],
            },
            "FinalReviewDecision": {
                "action": "FINAL", "status": "COMPLETED", "final_answer": "Observed 42",
            },
        }
        def respond(messages):
            self.seen_systems.append("\n".join(str(m["content"]) for m in messages))
            payload = payloads[schema.__name__]
            return {"parsed": schema.model_validate(payload),
                    "raw": AIMessage(content=json.dumps(payload)), "parsing_error": None}
        return RunnableLambda(respond)


class ExecuteOnlyWorld:
    def __init__(self):
        self.codes = []

    def execute(self, code):
        self.codes.append(code)
        return "42"


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_fixed_role_skills_reach_real_adapter_and_are_saved(self):
        world = ExecuteOnlyWorld()
        simple, hard = ScriptedModel(), ScriptedModel()
        result = await run_personalops(
            world, {"datetime":"2024-01-01", "instruction":"Observe 42", "supervisor":{}},
            simple_model=simple, hard_model=hard, trial_id="fixed-skill-unit",
            skill_mode="fixed", skill_fixed_ids={"general": ["appworld-execute-api"]},
        )
        self.assertEqual(world.codes, ["print(apis.api_docs.show_app_descriptions())", "print(6 * 7)"])
        self.assertTrue(any("[appworld-execute-api]" in s for s in simple.seen_systems))
        self.assertEqual(result["skill_routing_mode"], "fixed")
        snapshots = result["skill_routing_snapshots"].values()
        self.assertTrue(any(s["role"] == "general" and s["selected"] for s in snapshots))

    async def test_real_planning_graph_uses_only_world_tool_and_receives_skill(self):
        world = ExecuteOnlyWorld()
        simple, hard = ScriptedModel(), ScriptedModel()
        result = await run_personalops(
            world, {"datetime": "2024-01-01T00:00:00", "instruction": "Observe 42",
                    "supervisor": {"first_name": "Test"}},
            simple_model=simple, hard_model=hard, trial_id="unit-trial",
            skill="SKILL_MARKER_DO_NOT_PUBLISH_AS_RESULT",
        )
        self.assertEqual(world.codes, ["print(apis.api_docs.show_app_descriptions())", "print(6 * 7)"])
        self.assertEqual(
            set(simple.seen_tools),
            {"appworld_discover", "appworld_execute", "report_general_result"},
        )
        self.assertTrue(any("SKILL_MARKER" in s for s in simple.seen_systems))
        self.assertTrue(any("SKILL_MARKER" in s for s in hard.seen_systems))
        self.assertEqual(result["self_reported_final_status"], "COMPLETED")
        self.assertNotIn("official_task_success", result)

    async def test_outer_planning_graph_persists_final_snapshot(self):
        world = ExecuteOnlyWorld()
        simple, hard = ScriptedModel(), ScriptedModel()
        checkpoint_thread_id = "planning:unit-conversation:unit-checkpoint"

        with TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoints.sqlite3"
            async with AsyncSqliteSaver.from_conn_string(
                str(checkpoint_path)
            ) as checkpointer:
                result = await run_personalops(
                    world,
                    {
                        "datetime": "2024-01-01T00:00:00",
                        "instruction": "Observe 42",
                        "supervisor": {"first_name": "Test"},
                    },
                    simple_model=simple,
                    hard_model=hard,
                    trial_id="unit-checkpoint",
                    checkpointer=checkpointer,
                    checkpoint_thread_id=checkpoint_thread_id,
                )
                checkpoint_config = {
                    "configurable": {
                        "thread_id": checkpoint_thread_id,
                    }
                }
                saved = await checkpointer.aget_tuple(checkpoint_config)
                history = [
                    item
                    async for item in checkpointer.alist(checkpoint_config)
                ]

        self.assertEqual(result["self_reported_final_status"], "COMPLETED")
        self.assertIsNotNone(saved)
        saved_values = saved.checkpoint["channel_values"]
        self.assertEqual(saved_values["event_id"], "unit-checkpoint")
        self.assertEqual(saved_values["final_status"], "COMPLETED")
        self.assertEqual(saved_values["final_answer"], "Observed 42")
        self.assertGreater(len(history), 1)

    def test_normal_context_does_not_gain_benchmark_instructions(self):
        context = PlanningContextPack(current_time="now", user_request="hello")
        self.assertNotIn("[本轮执行环境与技能]", _format_hard_context(context))

    def test_replacement_context_is_explicit_in_hard_planning_input(self):
        context = PlanningContextPack(
            current_time="now",
            user_request="改成上海",
            replacement_context=PlanningReplacementContext(
                target_event_id="event-old",
                replacement_event_id="event-new",
                original_user_request="搜索北京音乐班并制作网页",
                replacement_instruction="不要北京，改成上海",
                previous_plan_objective="完成北京音乐班网页",
            ),
        )

        rendered = _format_hard_context(context)
        self.assertIn("[用户替换任务上下文]", rendered)
        self.assertIn("搜索北京音乐班并制作网页", rendered)
        self.assertIn("不要北京，改成上海", rendered)

    def test_execute_failure_is_a_tool_message_error(self):
        world = ExecuteOnlyWorld()
        world.execute = lambda code: "Execution failed. Traceback: invalid parameter"
        output = make_execute_tool(world).invoke({
            "name": "appworld_execute", "args": {"source_refs": ["U1"], "reason": "Fixture: documented API and fixture input.", "action_phase": "OTHER", "set_bindings": [], "code": "bad()"},
            "id": "failed-tool", "type": "tool_call",
        })
        self.assertEqual(output.status, "error")

    def test_budget_is_checked_before_next_model_call(self):
        meter = UsageMeter(max_calls=1)
        meter.on_chat_model_start({}, [], run_id=uuid4())
        with self.assertRaises(EvaluationBudgetExceeded):
            meter.on_chat_model_start({}, [], run_id=uuid4())

    def test_usage_meter_reads_runtime_model_role_metadata(self):
        meter, call = UsageMeter(), uuid4()
        meter.on_chat_model_start(
            {},
            [],
            run_id=call,
            tags=["map:key:raw"],
            metadata={"runtime.model_role": "scheduler", "trace.owner": "personalops"},
        )
        msg = AIMessage(content="ok", usage_metadata={
            "input_tokens": 2,
            "output_tokens": 1,
            "total_tokens": 3,
        })
        meter.on_llm_end(
            LLMResult(generations=[[ChatGeneration(message=msg)]]),
            run_id=call,
        )
        self.assertEqual(meter.report()["records"][0]["model_role"], "scheduler")

    def test_truncation_error_retains_billable_usage_without_success(self):
        from openai import LengthFinishReasonError
        from openai.types.chat import ChatCompletion
        meter, call = UsageMeter(), uuid4()
        meter.on_chat_model_start({}, [], run_id=call, tags=["appworld:hard"])
        completion = ChatCompletion.model_validate({
            "id": "fixture-response", "created": 0, "object": "chat.completion",
            "model": "fixture-model", "choices": [{"index": 0, "finish_reason": "length",
            "message": {"role": "assistant", "content": ""}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 2048, "total_tokens": 2148,
                      "prompt_tokens_details": {"cached_tokens": 64},
                      "completion_tokens_details": {"reasoning_tokens": 2048}},
        })
        meter.on_llm_error(LengthFinishReasonError(completion=completion), run_id=call)
        report = meter.report()
        self.assertTrue(report["usage_complete"])
        self.assertEqual(report["model_calls_returned"], 0)
        self.assertEqual(report["output_tokens"], 2048)
        self.assertEqual(report["records"][0]["finish_reason"], "length")
        self.assertEqual(report["records"][0]["reasoning_output_tokens"], 2048)
        self.assertEqual(report["records"][0]["cache_read_input_tokens"], 64)
        self.assertEqual(report["records"][0]["tags"], ["appworld:hard"])

    def test_transport_failure_is_unknown_usage_and_unstarted_rejection_is_not_a_call(self):
        meter, call = UsageMeter(max_calls=1), uuid4()
        meter.on_chat_model_start({}, [], run_id=call)
        meter.on_llm_error(TimeoutError("fixture timeout"), run_id=call)
        meter.on_llm_error(EvaluationBudgetExceeded("not started"), run_id=uuid4())
        report = meter.report()
        self.assertFalse(report["usage_complete"])
        self.assertIsNone(report["input_tokens"])
        self.assertEqual(report["model_calls_started"], 1)
        self.assertEqual(len(report["records"]), 1)

    def test_curated_model_callback_creates_one_manual_llm_span(self):
        meter = UsageMeter()
        call = uuid4()
        tracer = MagicMock()
        span = MagicMock()
        tracer.start_span.return_value = span
        with patch("observability.get_tracer", return_value=tracer):
            meter.on_chat_model_start(
                {},
                [[]],
                run_id=call,
                tags=["appworld:hard"],
                invocation_params={"model_name": "fixture-model"},
            )
            msg = AIMessage(
                content="private fixture output",
                usage_metadata={
                    "input_tokens": 12,
                    "output_tokens": 3,
                    "total_tokens": 15,
                },
            )
            result = LLMResult(generations=[[ChatGeneration(message=msg)]])
            meter.on_llm_end(result, run_id=call)
        tracer.start_span.assert_called_once_with(
            "LLM / hard / call 01",
            openinference_span_kind="llm",
        )
        span.end.assert_called_once()

    def test_learning_llm_span_keeps_messages_response_and_tool_decision(self):
        meter = UsageMeter()
        call = uuid4()
        tracer = MagicMock()
        span = MagicMock()
        tracer.start_span.return_value = span
        with patch("observability.get_tracer", return_value=tracer):
            meter.on_chat_model_start(
                {},
                [[HumanMessage(content="inspect the simulated app")]],
                run_id=call,
                tags=["appworld:simple"],
                invocation_params={"model_name": "fixture-model"},
            )
            msg = AIMessage(
                content="",
                tool_calls=[{
                    "name": "appworld_execute",
                    "args": {"source_refs": ["MODEL"], "reason": "Fixture: documented API and fixture input.", "code": "print(apis.api_docs.show_app_descriptions())"},
                    "id": "tool-1",
                    "type": "tool_call",
                }],
                usage_metadata={
                    "input_tokens": 12,
                    "output_tokens": 3,
                    "total_tokens": 15,
                },
            )
            meter.on_llm_end(
                LLMResult(generations=[[ChatGeneration(message=msg)]]),
                run_id=call,
            )

        trace_input = span.set_input.call_args.args[0]
        trace_output = span.set_output.call_args.args[0]
        self.assertEqual(
            trace_input["messages"][0]["content"],
            "inspect the simulated app",
        )
        self.assertEqual(
            trace_output["message"]["tool_calls"][0]["name"],
            "appworld_execute",
        )
        self.assertNotIn("content_redacted", trace_input)
        self.assertNotIn("content_redacted", trace_output)

    def test_learning_tool_span_keeps_exact_code_and_observation(self):
        world = ExecuteOnlyWorld()
        tracer = MagicMock()
        span = MagicMock()
        context = MagicMock()
        context.__enter__.return_value = span
        tracer.start_as_current_span.return_value = context

        with patch("observability.get_tracer", return_value=tracer):
            result = make_execute_tool(world).invoke({
                "name": "appworld_execute",
                "args": {"source_refs": ["U1"], "reason": "Fixture: documented API and fixture input.", "action_phase": "OTHER", "set_bindings": [], "code": "print(6 * 7)"},
                "id": "tool-1",
                "type": "tool_call",
            })

        self.assertEqual(result.content, "42")
        trace_input = span.set_input.call_args.args[0]
        trace_output = span.set_output.call_args.args[0]
        self.assertEqual(trace_input["code"], "print(6 * 7)")
        self.assertEqual(trace_output["observation"], "42")
        self.assertNotIn("raw_code_redacted", trace_input)
        self.assertNotIn("raw_output_redacted", trace_output)

    def test_repeated_callback_does_not_double_count_usage(self):
        meter = UsageMeter()
        call = uuid4()
        meter.on_chat_model_start({}, [], run_id=call)
        msg = AIMessage(content="ok", usage_metadata={"input_tokens": 12, "output_tokens": 3, "total_tokens": 15})
        result = LLMResult(generations=[[ChatGeneration(message=msg)]])
        meter.on_llm_end(result, run_id=call)
        meter.on_llm_end(result, run_id=call)
        self.assertEqual(meter.report()["input_tokens"], 12)
        self.assertEqual(meter.report()["model_calls_started"], 1)


if __name__ == "__main__":
    unittest.main()
