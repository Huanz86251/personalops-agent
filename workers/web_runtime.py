"""Resource-scoped runtime for one isolated Web Worker invocation."""
from runtime_tracing import operation

from collections.abc import Callable, Sequence
from typing import Any
from pathlib import Path
from file_limits import TASK_FILE_MAX_MIB
from config import WORKER_FINALIZATION_DEFAULT_MODEL_ROUNDS

from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from eventing.store import AsyncEventStore
from mcp_runtime import PlaywrightMCPPool
from workers.leadership import (
    LeadershipCoordinationState,
    LeadershipDecisionHandler,
    WorkerLeadershipBridge,
)
from workers.wake_policy import LeadershipWakePolicy
from workers.web_worker import create_web_worker
from run_workspace import (
    RUN_WORKSPACE_ROOT,
    create_run_worker_backend,
    handoff_read_only_permissions,
    initialize_run_workspace,
)


class WebStepRuntime:
    """Lease one Playwright runtime for the complete lifetime of a Worker.

    A lease covers all model/tool rounds, progress-gate interrupts, and leader
    guidance for this invocation.  Leaving the context closes the MCP process
    and returns capacity even when the Worker raises or is cancelled.
    """

    def __init__(
        self,
        model: str | BaseChatModel,
        *,
        summary_model=None,
        playwright_pool: PlaywrightMCPPool,
        event_store: AsyncEventStore,
        tools: Sequence[BaseTool],
        wake_policy: LeadershipWakePolicy | None = None,
        decision_handler: LeadershipDecisionHandler | None = None,
        progress_every_tool_calls: int,
        finalization_model_rounds: int = WORKER_FINALIZATION_DEFAULT_MODEL_ROUNDS,
        schema_repair_max_rounds: int = 3,
        download_max_file_mib: int = TASK_FILE_MAX_MIB,
        middleware_factory: Callable[[], Sequence[AgentMiddleware]] | None = None,
        checkpointer=None,
        store=None,
        run_storage_root: Path = RUN_WORKSPACE_ROOT,
        worker_factory: Callable[..., Any] = create_web_worker,
    ) -> None:
        self.model = model
        self.summary_model = summary_model
        self.playwright_pool = playwright_pool
        self.event_store = event_store
        self.tools = tuple(tools)
        self.wake_policy = wake_policy or LeadershipWakePolicy()
        self.decision_handler = decision_handler
        self.progress_every_tool_calls = progress_every_tool_calls
        self.finalization_model_rounds = finalization_model_rounds
        self.schema_repair_max_rounds = schema_repair_max_rounds
        self.download_max_file_mib = download_max_file_mib
        self.middleware_factory = middleware_factory or (lambda: ())
        self.checkpointer = checkpointer
        self.store = store
        self.run_storage_root = Path(run_storage_root)
        self.worker_factory = worker_factory
        self.coordination_state = LeadershipCoordinationState()

    @staticmethod
    def _worker_id(input_state: Any) -> str:
        if not isinstance(input_state, dict):
            raise ValueError("Web Worker invocation requires a state mapping.")
        worker_id = str(input_state.get("worker_id") or "").strip()
        if not worker_id:
            raise ValueError("Web Worker invocation requires worker_id.")
        return worker_id

    @operation('Web Runtime / Browser Session', fields=('input_state',))
    async def ainvoke(self, input_state, config=None, **kwargs):
        """Run one Worker inside its own leased browser Session."""

        worker_id = self._worker_id(input_state)
        run_id = str(
            input_state.get("event_id")
            or input_state.get("planning_run_id")
            or ""
        ).strip()
        if not run_id:
            raise ValueError("Planning Web Worker requires event_id.")
        layout = initialize_run_workspace(
            run_id,
            storage_root=self.run_storage_root,
        )
        async with self.playwright_pool.lease(worker_id) as browser_runtime:
            worker_graph = self.worker_factory(
                self.model,
                **({"summary_model": self.summary_model} if self.summary_model is not None else {}),
                tools=[*self.tools, *browser_runtime.tools],
                progress_every_tool_calls=self.progress_every_tool_calls,
                finalization_model_rounds=self.finalization_model_rounds,
                schema_repair_max_rounds=self.schema_repair_max_rounds,
                download_max_file_mib=self.download_max_file_mib,
                backend=create_run_worker_backend(layout),
                permissions=handoff_read_only_permissions(),
                middleware=list(self.middleware_factory()),
                checkpointer=self.checkpointer,
                store=self.store,
            )
            controlled_worker = WorkerLeadershipBridge(
                worker_graph,
                self.event_store,
                wake_policy=self.wake_policy,
                decision_handler=self.decision_handler,
                coordination_state=self.coordination_state,
            )
            worker_input = dict(input_state)
            worker_input["run_storage_root"] = str(
                self.run_storage_root.resolve()
            )
            # A paused Web Worker would keep its Playwright lease. If a normal
            # parallel group already owns every browser slot, an inserted Web
            # task could then wait forever. Therefore WEB consumes INSERT only
            # at the outer Planning node boundary, after this lease exits.
            worker_config = dict(config or {})
            worker_configurable = dict(worker_config.get("configurable", {}))
            worker_configurable.pop("event_pause_control", None)
            worker_config["configurable"] = worker_configurable
            return await controlled_worker.ainvoke(
                worker_input,
                config=worker_config,
                **kwargs,
            )


__all__ = ["WebStepRuntime"]
