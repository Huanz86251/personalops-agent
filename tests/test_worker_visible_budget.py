import unittest
from workers.context import WorkerRuntimeContextMiddleware

class VisibleBudgetTests(unittest.TestCase):
    def test_only_actual_execution_capacity_after_preparation(self):
        state = {"executor_model_run_limit": 8, "executor_model_calls_used": 2,
                 "skill_preparation_calls_used": 1, "worker_compaction_calls_used": 1,
                 "executor_tool_run_limit": 9, "executor_tool_calls_used": 3}
        middleware = WorkerRuntimeContextMiddleware(execution_reserve=1)
        update = middleware.before_model(state, None)
        text = update['messages'][0].content
        self.assertIn('模型执行轮次：3；业务工具调用：6', text)
        self.assertNotIn('预留', text)
        self.assertIsNone(middleware.before_model(state | update, None))
        final = middleware.before_model(state | update | {'worker_finalize_requested':True}, None)
        self.assertIn('模型执行轮次：0；业务工具调用：0', final['messages'][0].content)
