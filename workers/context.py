"""Checkpointed runtime-context events for the unified Deep Agent Worker."""

from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware

from workers.progress import WorkerProgressState
from workers.runtime_events import context_event


class WorkerRuntimeContextMiddleware(AgentMiddleware[WorkerProgressState]):
    """Expose assigned memory and environment instructions as append-only runtime events."""

    state_schema = WorkerProgressState

    def __init__(self, execution_reserve=0):
        self.execution_reserve = execution_reserve

    def before_model(self, state: WorkerProgressState, runtime):
        if state.get("code_agent_finished") or state.get("worker_review_requested"):
            return None
        blocks: list[str] = []
        if self.execution_reserve and state.get("executor_model_run_limit") is not None:
            used = sum(int(state.get(k, 0) or 0) for k in (
                "executor_model_calls_used", "skill_preparation_calls_used", "worker_compaction_calls_used"))
            remaining = max(0, int(state["executor_model_run_limit"]) - used - self.execution_reserve)
            tools = max(0, int(state.get("executor_tool_run_limit", 0)) - int(state.get("executor_tool_calls_used", 0)))
            if state.get("worker_finalize_requested"):
                remaining, tools = 0, 0
            blocks.append(f"[当前可用执行额度]\n模型执行轮次：{remaining}；业务工具调用：{tools}。以本次状态更新为准。")
        memory_context = state.get("memory_context", "")
        execution_instructions = state.get("execution_instructions", "")
        completion_api_contract = state.get("completion_api_contract", "")
        current_time_context = state.get("current_time_context", "")

        if isinstance(current_time_context, str) and current_time_context.strip():
            blocks.append(current_time_context.strip())

        if isinstance(memory_context, str) and memory_context.strip():
            blocks.append(f"[相关记忆]\n{memory_context.strip()}")
        if (
            isinstance(execution_instructions, str)
            and execution_instructions.strip()
        ):
            blocks.append(
                "[执行环境]\n"
                f"{execution_instructions.strip()}"
            )
        if isinstance(completion_api_contract, str) and completion_api_contract.strip():
            blocks.append(completion_api_contract.strip())

        return context_event(state, "\n\n".join(blocks), "runtime_context_fingerprint")


__all__ = ["WorkerRuntimeContextMiddleware"]
