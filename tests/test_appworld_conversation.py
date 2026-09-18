"""Provider/Docker-free contract checks for the optional AppWorld boundary."""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_core.tools import ToolException
from evals.appworld.conversation import AppWorldConversation, TaskTools, can_judge
from planning_models import PlanningContextPack, PlanStep


class World:
    def __init__(self):
        self.operations = []
    def execute(self, code):
        self.operations.append(("execute", code))
        if 'api_name="complete_task"' in code:
            return '{"app_name":"supervisor","api_name":"complete_task","parameters":[{"name":"answer"},{"name":"status"}]}'
        return "shared result"
    def request(self, op):
        self.operations.append((op, None))
        return {"success": False}


class AppWorldTests(unittest.TestCase):
    def test_harness_time_includes_weekday_without_changing_the_source_timestamp(self):
        for timestamp, day in (
            ("2023-05-23 09:24:06", "星期二 / Tuesday"),
            ("2023-05-24 10:52:00", "星期三 / Wednesday"),
        ):
            context = PlanningContextPack(
                current_time=timestamp, user_request="fixture",
                execution_environment="appworld",
            )
            self.assertEqual(
                context.current_time_context(),
                f"AppWorld 当前时间：{timestamp}（{day}）",
            )
        real = PlanningContextPack(
            current_time="2026-09-16T23:30:00+08:00", user_request="fixture",
        )
        self.assertEqual(
            real.current_time_context(),
            "真实世界当前时间：2026-09-16T23:30:00+08:00（星期三 / Wednesday）",
        )
        actual_tool_shape = real.model_copy(update={
            "current_time": (
                "时区：Asia/Shanghai\n"
                "当前时间：2026-09-16T23:30:00+08:00\n"
                "星期：星期三"
            ),
        })
        self.assertEqual(
            actual_tool_shape.current_time_context(),
            "真实世界当前时间：2026-09-16T23:30:00+08:00"
            "（时区：Asia/Shanghai；星期三 / Wednesday）",
        )
        opaque = real.model_copy(update={"current_time": "now"})
        self.assertEqual(opaque.current_time_context(), "真实世界当前时间：now")

    def test_shared_tools_and_closed_gate(self):
        world = World()
        tools = TaskTools(world)
        discover, execute, verify = tools.build()
        self.assertEqual(discover.invoke({"source_refs": ["MODEL"], "reason": "Need API directory.",
            "code": "print(apis.api_docs.show_app_descriptions())"}), "shared result")
        self.assertEqual(execute.invoke({"source_refs": ["E1"], "reason": "Fixture: documented API and fixture input.", "action_phase": "OTHER", "set_bindings": [], "code": "set"}), "shared result")
        self.assertEqual(verify.invoke({"source_refs": ["E2"], "reason": "Fixture: documented API and fixture input.", "action_phase": "OTHER", "set_bindings": [], "code": "query"}), "shared result")
        self.assertEqual([r["role"] for r in tools.calls], ["discover", "executor", "code_reviewer"])
        tools.accepting = False
        with self.assertRaises(ToolException):
            verify.invoke({"source_refs": ["E2"], "reason": "Fixture: documented API and fixture input.", "action_phase": "OTHER", "set_bindings": [], "code": "late"})
        self.assertEqual(len(world.operations), 3)

    def test_world_error_is_not_success(self):
        world = World()
        world.execute = lambda code: "Execution failed. ValueError: bad input"
        tools = TaskTools(world)
        with self.assertRaises(ToolException):
            tools.execute("bad", "executor")
        self.assertIn("error", tools.calls[0])

    def test_final_review_gate(self):
        for status in ("COMPLETED", "FAILED", "PARTIAL"):
            for reviewed in (True, False):
                for pending in ((), ("replanner",)):
                    snapshot = SimpleNamespace(next=pending, values={
                        "final_status": status, "scheduler_final_decision": reviewed})
                    self.assertEqual(can_judge(snapshot), reviewed and not pending)
        self.assertFalse(can_judge(SimpleNamespace(next=(), values={"final_status": "COMPLETED"})))
        self.assertTrue(can_judge(SimpleNamespace(next=(), values={
            "final_status": "PARTIAL",
            "scheduler_final_decision": False,
            "harness_terminal_decision": True,
        })))

    def make_runtime(self, reviewed=True):
        runtime = object.__new__(AppWorldConversation)
        runtime.task = {"task_id": "fixture", "instruction": "original task", "datetime": "2023-01-01"}
        runtime.task_tools = TaskTools(World())
        runtime._task_started = False
        runtime._completion_api_contract = ""
        runtime.new_conversation = AsyncMock(return_value=SimpleNamespace(
            conversation_id="new-conversation", thread_id="new-thread"))
        runtime.ask = AsyncMock(return_value="answer")
        runtime.planning_graph = SimpleNamespace(aget_state=AsyncMock(return_value=SimpleNamespace(
            next=(), values={"final_status": "COMPLETED", "scheduler_final_decision": reviewed})))
        return runtime

    def test_task_input_judge_order_and_no_reuse(self):
        runtime = self.make_runtime()
        result = asyncio.run(runtime.run_task())
        self.assertEqual(runtime.ask.call_args.kwargs["user_text"], "当前我们正在 AppWorld 里面进行测试。\n\noriginal task")
        self.assertEqual([op for op, _ in runtime.task_tools.world.operations], ["execute", "finish", "evaluate"])
        self.assertEqual([call["role"] for call in runtime.task_tools.calls], ["harness_documentation"])
        self.assertNotIn("apis.supervisor.complete_task(", runtime.task_tools.calls[0]["code"])
        self.assertFalse(result["official_evaluation"]["success"])
        self.assertEqual(result["terminal_decision_source"], "scheduler_model")
        self.assertEqual(result["conversation_id"], "new-conversation")
        with self.assertRaises(RuntimeError):
            asyncio.run(runtime.run_task())

    def test_no_judging_early_final_or_failure(self):
        runtime = self.make_runtime(reviewed=False)
        result = asyncio.run(runtime.run_task())
        self.assertIsNone(result["official_evaluation"])
        self.assertEqual([op for op, _ in runtime.task_tools.world.operations], ["execute"])
        runtime = self.make_runtime()
        runtime.ask.side_effect = RuntimeError("cancelled")
        with self.assertRaises(RuntimeError):
            asyncio.run(runtime.run_task())
        self.assertFalse(runtime.task_tools.accepting)
        self.assertEqual([op for op, _ in runtime.task_tools.world.operations], ["execute"])

    def test_benchmark_does_not_buffer_personal_memory(self):
        runtime = self.make_runtime()
        service = SimpleNamespace(consolidate_turn=AsyncMock())
        asyncio.run(runtime._consolidate_memory_background(memory_service=service))
        service.consolidate_turn.assert_not_awaited()

    def test_environment_is_optional_and_schema_unchanged(self):
        schema = PlanStep.model_json_schema()
        runtime = self.make_runtime()
        runtime._completion_api_contract = runtime._read_completion_api_contract()
        context = PlanningContextPack(current_time="original", user_request="original task")
        with patch("conversation_runtime.ConversationRuntime._prepare_planning_context", new=AsyncMock(return_value=context)):
            adapted = asyncio.run(runtime._prepare_planning_context())
        self.assertEqual(context.execution_instructions, "")
        self.assertEqual(adapted.current_time, "2023-01-01")
        self.assertEqual(adapted.execution_environment, "appworld")
        self.assertEqual(adapted.current_time_context(), "AppWorld 当前时间：2023-01-01（星期日 / Sunday）")
        self.assertEqual(adapted.execution_instructions, context.execution_instructions)
        self.assertIn("complete_task", adapted.completion_api_contract)
        self.assertEqual(context.completion_api_contract, "")
        self.assertEqual(schema, PlanStep.model_json_schema())
        self.assertIn("WEB", str(schema))
        names = {a["name"] for a in adapted.skill_catalog}
        self.assertIn("plan-appworld", names)
        self.assertNotIn("plan-web-research", names)
        self.assertNotIn("plan-task-dependencies", names)
        self.assertFalse(any(a["source"] == "web" for a in adapted.skill_catalog))
        from scheduler_runtime import SchedulerConversation
        from planning_models import SupervisorDecision
        adapted_session = SchedulerConversation({}, adapted)
        adapted_session.initialize()
        self.assertTrue(any(r.get("kind") == "AppWorld公开完成接口合同" and r.get("protected")
                            for r in adapted_session.records))
        adapted_session.disclose("supervisor", SupervisorDecision.model_json_schema())
        contract = next(r["content"] for r in adapted_session.records if r.get("key") == "step_contract")
        self.assertNotIn("WEB", contract)
        self.assertIn("CODE", contract)

    def test_public_contract_rejects_missing_or_oversized_document(self):
        runtime = self.make_runtime()
        runtime.task_tools.world.execute = lambda code: "no API documentation"
        with self.assertRaisesRegex(RuntimeError, "bounded complete_task"):
            runtime._read_completion_api_contract()
        runtime.task_tools.world.execute = lambda code: "complete_task answer status " + "x" * 20000
        with self.assertRaisesRegex(RuntimeError, "bounded complete_task"):
            runtime._read_completion_api_contract()

    def test_only_terminal_appworld_step_receives_contract(self):
        from planning_graph import _terminal_completion_contract
        context = PlanningContextPack(current_time="now", user_request="task",
                                      execution_environment="appworld", completion_api_contract="DOC")
        self.assertEqual(_terminal_completion_contract(context, []), "DOC")
        self.assertEqual(_terminal_completion_contract(context, [object()]), "")
        self.assertEqual(_terminal_completion_contract(context.model_copy(update={
            "execution_environment": "real_world"}), []), "")



class RealGraphGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_final_reviewer_sets_provenance(self):
        from test_planning_failure_stop import FailureStopTests
        fixture = FailureStopTests()
        for mode, expected in (("repair", True), ("final", False), ("fail", False)):
            result, _, _ = await fixture.run_case(mode)
            self.assertEqual(result.get("scheduler_final_decision", False), expected)


if __name__ == "__main__":
    unittest.main()
