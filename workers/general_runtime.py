"""Run-scoped General Worker runtime with read-only shared handoff access."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from eventing.store import AsyncEventStore
from feishu_exports import export_general_scope
from run_workspace import (
    RUN_WORKSPACE_ROOT,
    create_run_worker_backend,
    handoff_read_only_permissions,
    initialize_run_workspace,
)
from workers.general_worker import create_general_worker
from config import WORKER_FINALIZATION_DEFAULT_MODEL_ROUNDS
from workers.leadership import (
    LeadershipCoordinationState,
    LeadershipDecisionHandler,
    WorkerLeadershipBridge,
)
from workers.wake_policy import LeadershipWakePolicy


class GeneralStepRuntime:
    """Build one General Worker graph for the current Planning run."""

    def __init__(
        self,
        model: str | BaseChatModel,
        *,
        summary_model=None,
        event_store: AsyncEventStore,
        tools: Sequence[BaseTool],
        wake_policy: LeadershipWakePolicy | None = None,
        decision_handler: LeadershipDecisionHandler | None = None,
        progress_every_tool_calls: int,
        finalization_model_rounds: int = WORKER_FINALIZATION_DEFAULT_MODEL_ROUNDS,
        schema_repair_max_rounds: int = 3,
        middleware_factory: Callable[[], Sequence[AgentMiddleware]] | None = None,
        checkpointer=None,
        store=None,
        run_storage_root: Path = RUN_WORKSPACE_ROOT,
        worker_factory: Callable[..., Any] = create_general_worker,
    ) -> None:
        self.model = model
        self.summary_model = summary_model
        self.event_store = event_store
        self.tools = tuple(tools)
        self.wake_policy = wake_policy or LeadershipWakePolicy()
        self.decision_handler = decision_handler
        self.progress_every_tool_calls = progress_every_tool_calls
        self.finalization_model_rounds = finalization_model_rounds
        self.schema_repair_max_rounds = schema_repair_max_rounds
        self.middleware_factory = middleware_factory or (lambda: ())
        self.checkpointer = checkpointer
        self.store = store
        self.run_storage_root = Path(run_storage_root)
        self.worker_factory = worker_factory
        self.coordination_state = LeadershipCoordinationState()

    async def ainvoke(self, input_state, config=None, **kwargs):
        if not isinstance(input_state, dict):
            raise ValueError("General Worker invocation requires state mapping.")
        worker_id = str(input_state.get("worker_id") or "").strip()
        run_id = str(
            input_state.get("event_id")
            or input_state.get("planning_run_id")
            or ""
        ).strip()
        if not worker_id or not run_id:
            raise ValueError("Planning General Worker requires worker_id and event_id.")

        layout = initialize_run_workspace(
            run_id,
            storage_root=self.run_storage_root,
        )
        worker_graph = self.worker_factory(
            self.model,
            **({"summary_model": self.summary_model} if self.summary_model is not None else {}),
            tools=self.tools,
            progress_every_tool_calls=self.progress_every_tool_calls,
            finalization_model_rounds=self.finalization_model_rounds,
            schema_repair_max_rounds=self.schema_repair_max_rounds,
            middleware=list(self.middleware_factory()),
            backend=create_run_worker_backend(layout),
            permissions=handoff_read_only_permissions(),
            checkpointer=self.checkpointer,
            store=self.store,
        )
        controlled = WorkerLeadershipBridge(
            worker_graph,
            self.event_store,
            wake_policy=self.wake_policy,
            # General reports once on completion. Retain checkpoint/pause
            # handling without model-based leadership evaluation mid-step.
            decision_handler=None,
            coordination_state=self.coordination_state,
        )
        worker_input = dict(input_state)
        worker_input["run_storage_root"] = str(self.run_storage_root.resolve())
        with export_general_scope():
            return await controlled.ainvoke(
                worker_input,
                config=config,
                **kwargs,
            )


__all__ = ["GeneralStepRuntime"]
