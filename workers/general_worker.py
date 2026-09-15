"""Lightweight LangChain worker with task files and self-report completion."""

import json
from collections.abc import Sequence

from deepagents import FilesystemMiddleware, FilesystemPermission
from deepagents.backends import StateBackend
from deepagents.backends.protocol import BackendProtocol
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from config import (
    WORKER_FINALIZATION_DEFAULT_MODEL_ROUNDS,
    WORKER_PROGRESS_DEFAULT_TOOL_CALLS,
)
from workers.compaction import WorkerCompactionMiddleware
from workers.file_access import TaskFileAccessMiddleware
from knowledge_rag.runtime import with_knowledge_tool
from prompt_loader import load_prompt
from skill_runtime.middleware import RoleSkillsMiddleware
from tools import ALL_TOOLS
from feishu_exports import export_tool_available
from workers.context import WorkerRuntimeContextMiddleware
from workers.evidence_refs import EvidenceReferenceMiddleware
from workers.general_completion import (
    GENERAL_REPORT_NAME,
    GeneralCompletionMiddleware,
    GeneralBudgetMiddleware,
    report_general_result,
)
from workers.progress import (
    PUBLISH_WORKER_PROGRESS_NAME,
    WorkerProgressState,
)
from workers.submission import (
    SUBMIT_FOR_REVIEW_NAME,
)

GENERAL_WORKER_NAME = "general_worker"


# These tools belong to the legacy single-agent control or filesystem layer.
# Deep Agents supplies its own state-backed filesystem tools, while toolset
# routing remains the responsibility of the current main runtime.
LEGACY_TOOL_NAMES = frozenset(
    {
        "read_file",
        "write_file",
        "list_directory",
        "grep_files",
        "replace_in_file",
        "find_files",
        "show_all_toolsets",
        # Web capabilities belong to the dedicated resource-scoped Worker.
        "web_search",
        "find_github_mirror",
        "fetch_webpage",
    }
)


DEEP_AGENT_FILE_TOOLS = [
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "grep",
]


from knowledge_rag.middleware import KnowledgeMiddleware
from workers.history_archive import ExecutionHistoryMiddleware
from workers.execution_state import ExecutionStateMiddleware

def _tool_name(tool: BaseTool) -> str:
    """Return a validated LangChain tool name."""

    name = str(getattr(tool, "name", "")).strip()

    if not name:
        raise ValueError("general_worker received a tool without a name.")

    return name


def select_general_worker_tools(
    tools: Sequence[BaseTool] | None = None,
    *,
    allow_file_export: bool = False,
) -> list[BaseTool]:
    """Select business tools while excluding legacy control/file tools.

    Passing ``None`` uses the application's current ``ALL_TOOLS`` registry.
    The returned list is a copy, so building a worker cannot mutate the legacy
    runtime's tool list.
    """

    source_tools = ALL_TOOLS if tools is None else tools
    selected_tools: list[BaseTool] = []
    seen_names: set[str] = set()

    for tool in source_tools:
        name = _tool_name(tool)

        if name in seen_names:
            raise ValueError(f"general_worker received duplicate tool name: {name}")

        seen_names.add(name)

        if name == "send_local_file_to_feishu" and not (allow_file_export and export_tool_available()):
            continue

        if name not in LEGACY_TOOL_NAMES:
            selected_tools.append(tool)

    return selected_tools


def create_general_worker(
    model: str | BaseChatModel,
    *,
    summary_model=None,
    tools: Sequence[BaseTool] | None = None,
    progress_every_tool_calls: int = (WORKER_PROGRESS_DEFAULT_TOOL_CALLS),
    finalization_model_rounds: int = (WORKER_FINALIZATION_DEFAULT_MODEL_ROUNDS),
    schema_repair_max_rounds: int = 3,
    middleware: Sequence[AgentMiddleware] | None = None,
    backend: BackendProtocol | None = None,
    permissions: Sequence[FilesystemPermission] | None = None,
    checkpointer=None,
    store=None,
) -> CompiledStateGraph:
    """Build the General graph without delegation or independent review.

    The compiled graph can be wrapped by ``WorkerLeadershipBridge`` for
    checkpoint recovery and event pause points. Web/browser work and executable code are routed
    to dedicated runtimes rather than exposed here.
    """

    business_tools = with_knowledge_tool(select_general_worker_tools(tools, allow_file_export=True))

    if any(
        _tool_name(current_tool)
        in {
            PUBLISH_WORKER_PROGRESS_NAME,
            SUBMIT_FOR_REVIEW_NAME,
            GENERAL_REPORT_NAME,
        }
        for current_tool in business_tools
    ):
        raise ValueError("Worker runtime control tool was supplied as a business tool.")

    completion = GeneralCompletionMiddleware()
    from middlewares import DynamicExecutionBudgetMiddleware
    policies = [
        GeneralBudgetMiddleware(
            finalization_model_rounds=finalization_model_rounds,
            schema_repair_max_rounds=schema_repair_max_rounds,
        )
        if type(m) is DynamicExecutionBudgetMiddleware
        else m
        for m in (middleware or ())
    ]

    resolved_backend = backend or StateBackend()
    resolved_permissions = list(permissions or ())
    filesystem = FilesystemMiddleware(
        custom_tool_descriptions=json.loads(load_prompt("tools/filesystem")),
        backend=resolved_backend,
        tools=list(DEEP_AGENT_FILE_TOOLS),
        _permissions=resolved_permissions,
    )

    graph = create_agent(
        model=model,
        tools=[
            *business_tools,
            report_general_result,
        ],
        system_prompt=load_prompt("workers/general_worker"),
        middleware=[
            *policies,
            KnowledgeMiddleware('General Agent'),
            ExecutionHistoryMiddleware(),
            ExecutionStateMiddleware(),
            WorkerCompactionMiddleware(summary_model if summary_model is not None else model),
            RoleSkillsMiddleware(model, "general", [*DEEP_AGENT_FILE_TOOLS, *(_tool_name(t) for t in business_tools)],
                                 policies=middleware or (), backend=resolved_backend),
            WorkerRuntimeContextMiddleware(execution_reserve=1),
            TaskFileAccessMiddleware(),
            filesystem,
            completion,
            EvidenceReferenceMiddleware(),
        ],
        state_schema=WorkerProgressState,
        checkpointer=checkpointer,
        store=store,
        name=GENERAL_WORKER_NAME,
    )
    from trace_callbacks import configure_graph
    return configure_graph(graph, "general")
