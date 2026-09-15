from types import SimpleNamespace

from evals.appworld.conversation import TaskTools
from workers.execution_state import (
    ExecutionStateLedger,
    ExecutionStateMiddleware,
    execution_state_scope,
)


def test_success_records_api_derived_variables_without_values():
    ledger = ExecutionStateLedger()
    ledger.record(
        index=1,
        role="executor",
        code="login = apis.music.login(username=user, password=password)\ntoken = login['access_token']",
        succeeded=True,
    )
    state = ledger.snapshot()
    assert [item["variable"] for item in state["available_variables"]] == ["login", "token"]
    assert state["available_variables"][0]["source_apis"] == ["music.login"]
    assert state["available_variables"][1]["source_apis"] == ["music.login"]
    assert "password" not in str(state)


def test_failed_assignments_remain_unverified_until_successful_use():
    ledger = ExecutionStateLedger()
    ledger.record(
        index=1,
        role="executor",
        code="login = apis.music.login(username=user, password=password)\ntoken = login['access_token']",
        succeeded=False,
        error_type="ToolException",
    )
    assert not ledger.snapshot()["available_variables"]
    ledger.record(
        index=2,
        role="executor",
        code="items = apis.music.show_library(access_token=token)",
        succeeded=True,
    )
    state = ledger.snapshot()
    assert {item["variable"] for item in state["available_variables"]} == {"items", "token"}
    assert next(item for item in state["available_variables"] if item["variable"] == "token")["status"] == "available"
    assert "login" in {item["variable"] for item in state["unverified_variables"]}
    assert "token" not in {item["variable"] for item in state["unverified_variables"]}


def test_middleware_injects_only_changed_snapshots():
    ledger = ExecutionStateLedger()
    middleware = ExecutionStateMiddleware()
    with execution_state_scope(ledger):
        assert middleware.before_model({}, SimpleNamespace()) is None
        ledger.record(index=1, role="executor", code="x = apis.demo.read()", succeeded=True)
        first = middleware.before_model({}, SimpleNamespace())
        assert "demo.read" in first["messages"][0].content
        assert middleware.before_model({}, SimpleNamespace()) is None
        ledger.record(index=2, role="executor", code="y = apis.other.write()", succeeded=True)
        second = middleware.before_model({}, SimpleNamespace())
        assert "other.write" in second["messages"][0].content
        assert "demo.read" not in second["messages"][0].content


class _World:
    def __init__(self, output):
        self.output = output

    def execute(self, code):
        return self.output


def test_task_tools_update_ledger_from_real_outcome():
    tools = TaskTools(_World("{'access_token': 'secret-value'}"))
    tools.execute("session = apis.demo.login(username=name, password=pw)", "executor", "documented")
    state = tools.execution_state.snapshot()
    assert state["recent_calls"][0]["status"] == "SUCCESS"
    assert state["available_variables"][0]["variable"] == "session"
    assert "secret-value" not in str(state)
    assert tools.calls[0]["execution_state"]["revision"] == 1
