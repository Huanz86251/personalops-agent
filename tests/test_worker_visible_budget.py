import unittest
from workers.context import WorkerRuntimeContextMiddleware

class VisibleBudgetTests(unittest.TestCase):
    def test_runtime_clock_is_visible_to_worker_and_role_reviewers(self):
        middleware = WorkerRuntimeContextMiddleware()
        update = middleware.before_model(
            {"current_time_context": "AppWorld 当前时间：2023-04-05 10:30:00"},
            None,
        )
        self.assertIn(
            "AppWorld 当前时间：2023-04-05 10:30:00",
            update["messages"][0].content,
        )

    def test_terminal_completion_contract_survives_runtime_context_refresh(self):
        middleware = WorkerRuntimeContextMiddleware()
        update = middleware.before_model({"completion_api_contract": "PUBLIC-COMPLETE-TASK-DOC"}, None)
        self.assertIn("PUBLIC-COMPLETE-TASK-DOC", update["messages"][0].content)
        self.assertTrue(update["messages"][0].additional_kwargs["personalops_runtime_event"])
        self.assertIsNone(middleware.before_model({"completion_api_contract": "PUBLIC-COMPLETE-TASK-DOC",
                                                    **update}, None))

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
