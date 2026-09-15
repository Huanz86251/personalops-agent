import unittest
from unittest.mock import AsyncMock, patch
from workers.code_runtime import CodeStepRuntime, _InvocationUsage


class RoleBudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_reserved_budget_appends_without_changing_prefix(self):
        runtime = object.__new__(CodeStepRuntime)
        usage = _InvocationUsage(model_limit=8, tool_limit=12)
        for role, state in (
            ("worker", usage.state({}, reserve_model=3, reserve_tools=3)),
            ("reviewer", _InvocationUsage(8, 12, model_calls=3, tool_calls=4).state({})),
            ("worker", _InvocationUsage(8, 12, model_calls=3, tool_calls=4).state({}, reserve_model=2, reserve_tools=2)),
        ):
            prefix = "Frozen instructions and previous evidence."
            with patch("workers.code_runtime.ask_worker", new=AsyncMock(return_value={})) as invoke:
                await runtime._invoke(object(), prefix, thread_id="trial:" + role, state=state)
            sent = invoke.call_args.args[1]
            self.assertTrue(sent.startswith(prefix))
            if role == "reviewer":
                self.assertIn(f"最多模型轮次：{state['executor_model_run_limit']}", sent)
                self.assertIn("提交报告", sent)
            else:
                self.assertEqual(sent, prefix)
            actual = invoke.call_args.kwargs['state_update']
            self.assertTrue(all(actual[k] == v for k, v in state.items()))
            self.assertFalse(actual['worker_finalize_requested'])
            self.assertEqual(actual['worker_finalization_model_calls_used'], 0)


if __name__ == '__main__':
    unittest.main()
