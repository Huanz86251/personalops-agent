"""Schema repair gets a separate, bounded original-role budget."""

from langchain_core.messages import AIMessage

from middlewares import DynamicExecutionBudgetMiddleware
from workers.code_finalization import CodeWorkerBudgetMiddleware
from workers.general_completion import GeneralBudgetMiddleware


def repair_state(*, used=0, tool_name="report_general_result"):
    return {
        "worker_finalize_requested": True,
        "worker_finalize_reason": "SCHEMA_REPAIR",
        "worker_schema_repair_model_calls_used": used,
        "executor_model_run_limit": 1,
        "executor_model_calls_used": 1,
        "executor_tool_run_limit": 1,
        "executor_tool_calls_used": 1,
        "show_all_toolsets_run_limit": 0,
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": tool_name,
                        "id": "repair-call",
                        "args": {},
                        "type": "tool_call",
                    }
                ],
            )
        ],
    }


def test_shared_budget_allows_three_repairs_after_business_budget_is_full():
    gate = DynamicExecutionBudgetMiddleware(
        enable_worker_finalization=True,
        schema_repair_max_rounds=3,
    )
    assert gate.before_model(repair_state(used=0), None) is None
    assert gate.before_model(repair_state(used=2), None) is None
    assert gate.before_model(repair_state(used=3), None) == {"jump_to": "end"}


def test_general_repairs_do_not_consume_business_model_budget():
    gate = GeneralBudgetMiddleware(schema_repair_max_rounds=3)
    state = repair_state(used=0)
    assert gate.before_model(state, None) is None
    update = gate.after_model(state, None)
    assert update["worker_schema_repair_model_calls_used"] == 1
    assert "executor_model_calls_used" not in update


def test_code_and_reviewer_gate_share_the_same_repair_reserve():
    gate = CodeWorkerBudgetMiddleware(schema_repair_max_rounds=3)
    state = repair_state(used=0, tool_name="submit_code_for_review")
    assert gate.before_model(state, None) is None
    update = gate.after_model(state, None)
    assert update["worker_schema_repair_model_calls_used"] == 1
    assert "executor_model_calls_used" not in update
    assert gate.before_model(repair_state(used=3), None) == {"jump_to": "end"}
