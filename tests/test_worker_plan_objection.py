"""Provider-free execution tests for the explicit mid-run control channel."""
import json
import asyncio
from functools import wraps
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from tests.test_code_agents import ToolRecordingModel
from workers.plan_objection import (
    WorkerObjectionMiddleware, objection_scope, objection_tools,
    raise_plan_objection,
)
from workers.progress import WorkerProgressState


def offline_async(fn):
    @wraps(fn)
    def run(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))
    return run


def objection_call(call_id="objection-1"):
    return {"name": "raise_plan_objection", "id": call_id, "args": {
        "problem": "用户要求读取CSV，但当前步骤要求改数据库。",
        "reason": "任务契约和用户原始要求不一致。",
        "suggested_direction": "改为读取CSV并生成本地统计结果。",
    }}


@pytest.mark.parametrize("action", ["CONTINUE", "GUIDE", "REPLAN", "FINISH"])
@offline_async
async def test_real_tool_node_injection_and_control_routing(action):
    calls = []

    async def decide(objection, state):
        calls.append((objection, state["worker_id"]))
        return {"decision": {"action": action, "reason": "Checked current task.",
                "guidance": "只读取CSV。" if action == "GUIDE" else None},
                "model_rounds_used": 1}

    model = ToolRecordingModel(responses=[
        AIMessage(content="", tool_calls=[objection_call()]),
        AIMessage(content="Continuing with the current task."),
    ])
    graph = create_agent(model, tools=[raise_plan_objection],
                         middleware=[WorkerObjectionMiddleware()],
                         state_schema=WorkerProgressState)
    with patch.dict("os.environ", {"RUNTIME_WORKER_PLAN_OBJECTION_ENABLED": "true"}), objection_scope(decide):
        result = await graph.ainvoke({"messages": [{"role": "user", "content": "Read CSV."}],
                                     "worker_id": "w1"})
    assert len(calls) == 1, [str(m.content) for m in result["messages"] if getattr(m, "type", None) == "tool"]
    assert calls[0][1] == "w1"
    assert result["worker_leadership_model_rounds_used"] == 1
    assert len(result["worker_objection_records"]) == 1
    assert model.invocation_count == (1 if action in {"REPLAN", "FINISH"} else 2)
    assert bool(result.get("worker_objection_outcome")) == (action in {"REPLAN", "FINISH"})
    reply = next(m for m in result["messages"] if getattr(m, "type", None) == "tool")
    assert json.loads(reply.content)["decision"]["action"] == action


@offline_async
async def test_objection_blocks_business_calls_in_same_batch():
    executed = []

    @tool
    def change_database():
        """Offline stand-in for a side-effecting business tool."""
        executed.append(True)
        return "changed"

    async def decide(*args):
        return {"decision": {"action": "REPLAN", "reason": "Wrong task."}, "model_rounds_used": 1}

    model = ToolRecordingModel(responses=[AIMessage(content="", tool_calls=[
        {"name": "change_database", "id": "business", "args": {}}, objection_call(),
    ])])
    graph = create_agent(model, tools=[change_database, raise_plan_objection],
                         middleware=[WorkerObjectionMiddleware()], state_schema=WorkerProgressState)
    with patch.dict("os.environ", {"RUNTIME_WORKER_PLAN_OBJECTION_ENABLED": "true"}), objection_scope(decide):
        result = await graph.ainvoke({"messages": [{"role": "user", "content": "Read CSV."}]})
    assert executed == []
    assert result["worker_objection_outcome"]["decision"]["action"] == "REPLAN"


def test_disabled_registration_and_short_public_schema():
    with patch.dict("os.environ", {"RUNTIME_WORKER_PLAN_OBJECTION_ENABLED": "false"}):
        assert objection_tools() == []
    assert set(raise_plan_objection.args) == {"problem", "reason", "suggested_direction"}


def test_control_tool_remains_visible_during_progress_and_finalization():
    from workers.progress import WorkerProgressMiddleware
    from workers.general_completion import GeneralCompletionMiddleware
    from workers.code_finalization import CodeWorkerProgressMiddleware

    class Request(SimpleNamespace):
        def override(self, **changes):
            return Request(**{**vars(self), **changes})

    tools = [SimpleNamespace(name=n) for n in (
        "raise_plan_objection", "publish_worker_progress", "complete_web_step",
        "report_general_result", "submit_code_for_review", "execute")]
    progress = WorkerProgressMiddleware(every_tool_calls=2)
    due = Request(state={"worker_total_tool_calls": 10, "worker_last_progress_tool_call": 0}, tools=tools)
    assert "raise_plan_objection" in {t.name for t in progress._require_control_request(due).tools}
    final = Request(state={"worker_finalize_requested": True}, tools=tools)
    for control in (progress._require_control_request, GeneralCompletionMiddleware()._final_request,
                    CodeWorkerProgressMiddleware()._require_control_request):
        changed = control(final)
        assert "raise_plan_objection" in {t.name for t in changed.tools}
        assert "execute" not in {t.name for t in changed.tools}


@pytest.mark.parametrize("baseline_name", ["GENERAL_TOOLSET_BASELINE_TOOL_NAMES", "WEB_TOOLSET_BASELINE_TOOL_NAMES", "CODE_TOOLSET_BASELINE_TOOL_NAMES"])
@offline_async
async def test_actual_role_router_keeps_objection_when_business_tools_are_disabled(baseline_name):
    import conversation_runtime
    from middlewares import ToolsetRouterMiddleware
    from tests.test_worker_toolset_gating import FakeRequest, UncertainReranker

    middleware = ToolsetRouterMiddleware(UncertainReranker(),
        baseline_tool_names=getattr(conversation_runtime, baseline_name))
    captured = []

    async def handler(request):
        captured.extend(t.name for t in request.tools)
        return "ok"

    await middleware.awrap_model_call(FakeRequest(
        tools=[raise_plan_objection, SimpleNamespace(name="appworld_execute")],
        state={"step_tool_access": "DISABLED", "executor_tool_run_limit": 10,
               "conversation_id": "disabled-" + baseline_name}), handler)
    assert captured == ["raise_plan_objection"]


def test_old_objection_fields_removed_from_terminal_contracts():
    from workers.general_completion import GeneralResult
    from workers.submission import WorkerSubmission
    from workers.code_review_models import CodeWorkerSubmission, CodeReviewReport
    for schema in (GeneralResult, WorkerSubmission, CodeWorkerSubmission, CodeReviewReport):
        assert "plan_challenge" not in schema.model_fields
        assert "confirmed_plan_challenge" not in schema.model_fields


@pytest.mark.parametrize("role", ["GENERAL", "WEB", "CODE"])
@offline_async
async def test_all_worker_factories_offer_tool_and_stop_before_terminal_submission(role):
    from workers.general_worker import create_general_worker
    from workers.web_worker import create_web_worker
    from workers.code_worker import create_code_worker
    from tests.test_code_agents import candidate, contract
    from workers.code_review_models import create_code_review_loop

    async def decide(*args):
        return {"decision": {"action": "FINISH", "reason": "Current plan violates the request."},
                "model_rounds_used": 1}

    model = ToolRecordingModel(responses=[AIMessage(content="", tool_calls=[objection_call()])])
    @tool
    def web_search(query: str):
        """Offline search stand-in; objection test must never call this."""
        raise AssertionError("Search must not execute")
    factories = {"GENERAL": create_general_worker, "WEB": create_web_worker, "CODE": create_code_worker}
    c = candidate()
    state = {"messages": [{"role": "user", "content": "Read CSV."}], "skill_mode": "off",
             "worker_id": "w1", "event_id": "event1", "step_id": "1",
             "code_task": contract().model_dump(mode="json"), "code_candidate": c.model_dump(mode="json"),
             "code_review_loop": create_code_review_loop(candidate=c, worker_checkpoint_id="w", reviewer_checkpoint_id="r").model_dump(mode="json")}
    with patch.dict("os.environ", {"RUNTIME_WORKER_PLAN_OBJECTION_ENABLED": "true"}), objection_scope(decide):
        graph = factories[role](model, tools=[web_search] if role == "WEB" else [])
        result = await graph.ainvoke(state)
    assert "raise_plan_objection" in model.bound_tool_names
    assert model.invocation_count == 1
    assert result["worker_objection_outcome"]["decision"]["action"] == "FINISH"
    assert not result.get("worker_submission")


@pytest.mark.parametrize("action", ["REPLAN", "FINISH"])
@offline_async
async def test_code_runtime_archives_objection_without_review_or_publication(tmp_path, action):
    from tests.test_code_runtime import FakeSandboxManager, CodeRuntimeTests
    from workers.code_runtime import CodeStepRuntime

    class ObjectionWorker:
        async def ainvoke(self, state, config=None):
            return {"messages": [AIMessage(content="Stopped for Scheduler.")],
                    "executor_model_calls_used": 1, "executor_tool_calls_used": 1,
                    "worker_objection_outcome": {"decision": {"action": action, "reason": "Wrong input source."}}}

    class NeverReviewer:
        async def ainvoke(self, *args, **kwargs):
            raise AssertionError("Objection must not invoke Code Reviewer")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "app.py").write_text("print('original')\n", encoding="utf-8")
    manager = FakeSandboxManager()
    runtime = CodeStepRuntime(object(), sandbox_manager=manager, source_root=workspace,
        target_root=workspace, archive_root=tmp_path / "archives", run_storage_root=tmp_path / "runs",
        worker_factory=lambda *a, **k: ObjectionWorker(), reviewer_factory=lambda *a, **k: NeverReviewer())
    result = await runtime.ainvoke(CodeRuntimeTests.runtime_input(), config={"configurable": {"thread_id": "objection-code"}})
    assert manager.cleaned
    assert result["code_review_report"] is None
    assert result["code_publication_receipt"] is None
    assert result["code_attempt_final_record"]["outcome"] == ("SUPERSEDED" if action == "REPLAN" else "CANCELLED")
    assert (workspace / "app.py").read_text("utf-8") == "print('original')\n"
