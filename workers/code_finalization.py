"""Code Worker reporting reserve, inside its assigned role budget."""

from langchain.agents.middleware import hook_config
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from middlewares import DynamicExecutionBudgetMiddleware
from workers.progress import WorkerProgressMiddleware


def terminal_tool(state):
    loop = state.get("code_review_loop") or {}
    if loop.get("pending_scheduler_directive"):
        return "submit_continued_code_for_review"
    if loop.get("pending_instruction"):
        return "respond_to_code_review"
    return "submit_code_for_review"


class CodeWorkerBudgetMiddleware(DynamicExecutionBudgetMiddleware):
    terminal_name = staticmethod(terminal_tool)

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        if state.get("code_agent_finished"):
            return {"jump_to": "end"}
        if (
            state.get("worker_finalize_requested")
            and state.get("worker_finalize_reason") == "SCHEMA_REPAIR"
        ):
            repair_used = int(state.get("worker_schema_repair_model_calls_used", 0) or 0)
            repair_limit = max(
                int(
                    state.get(
                        "worker_schema_repair_model_run_limit",
                        self.schema_repair_max_rounds,
                    )
                    or self.schema_repair_max_rounds
                ),
                1,
            )
            return None if repair_used < repair_limit else {"jump_to": "end"}
        finalization_limit = max(
            int(state.get("worker_finalization_model_run_limit", self.finalization_model_rounds) or self.finalization_model_rounds),
            1,
        )
        if state.get("worker_finalize_requested") and state.get("worker_finalization_model_calls_used", 0) >= finalization_limit:
            return {"jump_to": "end"}
        limit = state.get("executor_model_run_limit")
        if limit is None:
            return None
        used = sum(int(state.get(k, 0) or 0) for k in (
            "executor_model_calls_used", "skill_preparation_calls_used", "worker_compaction_calls_used"))
        if used >= limit:
            return {"jump_to": "end"}
        tool_limit = state.get("executor_tool_run_limit")
        if used >= max(0, limit - 1) or (tool_limit is not None and state.get("executor_tool_calls_used", 0) >= tool_limit):
            return {"worker_finalize_requested": True,
                    "worker_finalize_reason": "角色执行额度到收尾边界；仅总结真实修改、测试结果、错误及阻碍，不再执行或修复。未完成要求标为NOT_MET/PARTIAL，不能虚构测试通过。"}
        return None

    @hook_config(can_jump_to=["model"])
    def after_model(self, state, runtime):
        if not state.get("worker_finalize_requested"):
            update = super().after_model(state, runtime)
            if update and update.get("messages") and not getattr(update["messages"][-1], "tool_calls", []):
                update.update(worker_finalize_requested=True, worker_finalize_reason="工具预算已耗尽，提交真实阻碍与未完成项", jump_to="model")
            return update
        messages = list(state.get("messages", []))
        last = messages[-1]
        calls = getattr(last, "tool_calls", [])
        allowed = [c for c in calls if c.get("name") == self.terminal_name(state)][:1]
        # WorkerProgressMiddleware counts all finalization calls. Schema
        # repairs have their own reserve and must not consume business rounds.
        if state.get("worker_finalize_reason") == "SCHEMA_REPAIR":
            update = {
                "worker_schema_repair_model_calls_used": (
                    int(state.get("worker_schema_repair_model_calls_used", 0) or 0) + 1
                )
            }
        else:
            update = {
                "executor_model_calls_used": int(state.get("executor_model_calls_used", 0)) + 1
            }
        if calls != allowed:
            update["messages"] = [RemoveMessage(id=REMOVE_ALL_MESSAGES), *messages[:-1],
                                  last.model_copy(update={"tool_calls": allowed})]
        return update


class CodeWorkerProgressMiddleware(WorkerProgressMiddleware):
    terminal_name = staticmethod(terminal_tool)

    @hook_config(can_jump_to=["model"])
    def after_model(self, state, runtime):
        update = super().after_model(state, runtime)
        if update and update.get("jump_to") == "model":
            # The jump skips the outer budget after_model hook.
            update["executor_model_calls_used"] = int(state.get("executor_model_calls_used", 0)) + 1
        return update

    def _require_control_request(self, request):
        if request.state.get("worker_finalize_requested") and not request.state.get("worker_review_requested"):
            name = self.terminal_name(request.state)
            tools = [t for t in request.tools if self._tool_name(t) == name]
            if len(tools) != 1:
                raise RuntimeError(f"Code finalization requires {name}")
            return request.override(tools=tools, tool_choice=name)
        return super()._require_control_request(request)


def missing_handoff_message(details, stage):
    """Bounded factual diagnostics; not a fabricated Reviewer verdict."""
    import json
    errors = []
    for message in details.get("current_turn_messages", []):
        kind = message.get("type", message.get("role")) if isinstance(message, dict) else getattr(message, "type", None)
        content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
        status = message.get("status") if isinstance(message, dict) else getattr(message, "status", None)
        if kind == "tool" and (status == "error" or any(m in str(content) for m in ("Traceback", "Error:", "rejected:", '"status": "error"'))):
            errors.append(str(content)[:700])
    return json.dumps({"status": "FAILED", "stage": stage,
        "reason": "Code Worker did not submit a valid handoff; independent review not completed for this handoff.",
        "usage": details.get("execution_summary", {}), "tool_errors": errors[-3:],
        "runtime_error": str(details.get("error", ""))[:700],
        "required_help": "Inspect execution errors and remaining requirements before deciding whether to replan; do not assume changes were verified or published."}, ensure_ascii=False, default=str)


class CodeReviewerBudgetMiddleware(CodeWorkerBudgetMiddleware):
    terminal_name = staticmethod(lambda state: "submit_code_review")

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        update = super().before_model(state, runtime)
        if update and update.get("worker_finalize_requested"):
            update["worker_finalize_reason"] = "审核收尾：只提交submit_code_review，列出已验证事实、失败和未验证项；没有验证或发布依据不能判PASSED，依照审核schema选择FAILED或ESCALATED。不再执行、发布或修复。"
        return update


class CodeReviewerProgressMiddleware(CodeWorkerProgressMiddleware):
    def _progress_is_due(self, state):
        return False

    terminal_name = staticmethod(lambda state: "submit_code_review")
