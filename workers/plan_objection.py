"""Explicit control tool; Scheduler adjudicates before any global replan."""
from contextlib import contextmanager
from contextvars import ContextVar
import json
from typing import Annotated
from pydantic import Field

from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from config import _load_worker_runtime_settings
from prompt_loader import load_prompt
from schemas.worker_control import WorkerPlanObjection, WorkerObjectionDecision

RAISE_PLAN_OBJECTION_NAME = "raise_plan_objection"
_HANDLER = ContextVar("worker_plan_objection_handler", default=None)


@contextmanager
def objection_scope(handler=None):
    # Code Runtime's nested ask_worker calls inherit their Planning invocation.
    token = _HANDLER.set(handler) if handler is not None else None
    try:
        yield
    finally:
        if token is not None:
            _HANDLER.reset(token)


def objection_tools():
    return [raise_plan_objection] if _load_worker_runtime_settings().plan_objection_enabled else []


@tool(description=load_prompt("workers/raise_plan_objection_tool"))
async def raise_plan_objection(
    problem: Annotated[str, Field(min_length=1, max_length=1200, description="当前任务契约与用户目标的具体冲突。")],
    reason: Annotated[str, Field(min_length=1, max_length=1200, description="为什么继续执行会偏离用户要求。")],
    suggested_direction: Annotated[str, Field(min_length=1, max_length=1200, description="建议调整方向；不代写完整计划、不编造API。")],
    runtime: ToolRuntime,
) -> Command:
    state = runtime.state
    handler = _HANDLER.get()
    if not _load_worker_runtime_settings().plan_objection_enabled or handler is None:
        return Command(update={"messages": [ToolMessage(
            content="Plan objection unavailable: this execution has no enabled Scheduler adjudication channel. Continue the assigned task or report the actual blocker.",
            tool_call_id=runtime.tool_call_id, status="error",
        )]})
    objection = WorkerPlanObjection(problem=problem, reason=reason, suggested_direction=suggested_direction)
    result = await handler(objection, state)
    decision = WorkerObjectionDecision.model_validate(result["decision"])
    record = {"objection": objection.model_dump(mode="json"), **result}
    update = {
        "worker_objection_records": [*state.get("worker_objection_records", []), record],
        "worker_leadership_model_rounds_used": int(state.get("worker_leadership_model_rounds_used", 0) or 0) + int(result.get("model_rounds_used", 0)),
        "messages": [ToolMessage(content=json.dumps(record, ensure_ascii=False), tool_call_id=runtime.tool_call_id)],
    }
    if decision.action in {"REPLAN", "FINISH"}:
        update["worker_objection_outcome"] = record
    return Command(update=update)


class WorkerObjectionMiddleware(AgentMiddleware):
    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        if (state.get("worker_objection_outcome") or {}).get("decision", {}).get("action") in {"REPLAN", "FINISH"}:
            return {"jump_to": "end"}
        return None

    def _block(self, request):
        messages = request.state.get("messages") or []
        latest = next((m for m in reversed(messages) if getattr(m, "type", None) == "ai" or (isinstance(m, dict) and m.get("role") == "assistant")), None)
        calls = latest.get("tool_calls", []) if isinstance(latest, dict) else getattr(latest, "tool_calls", [])
        if request.tool_call["name"] == RAISE_PLAN_OBJECTION_NAME:
            objections = [c for c in calls if c.get("name") == RAISE_PLAN_OBJECTION_NAME]
            if objections and request.tool_call["id"] != objections[0].get("id"):
                return ToolMessage(content="Deferred: only one planning objection per model turn is allowed.", tool_call_id=request.tool_call["id"], status="error")
            return None
        if any(c.get("name") == RAISE_PLAN_OBJECTION_NAME for c in calls):
            return ToolMessage(content="Deferred: submit the planning objection alone, await Scheduler guidance, then choose the next tool.", tool_call_id=request.tool_call["id"], status="error")
        if request.state.get("worker_objection_outcome"):
            return ToolMessage(content="Execution stopped by Scheduler planning decision.", tool_call_id=request.tool_call["id"], status="error")
        return None

    async def awrap_tool_call(self, request, handler):
        blocked = self._block(request)
        return blocked if blocked is not None else await handler(request)

    def wrap_tool_call(self, request, handler):
        blocked = self._block(request)
        return blocked if blocked is not None else handler(request)
