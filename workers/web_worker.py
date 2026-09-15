from workers.compaction import WorkerCompactionMiddleware
"""Dedicated Deep Agent for Web research and browser interaction."""

import json
from collections.abc import Sequence

from deepagents import FilesystemMiddleware, FilesystemPermission, create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.protocol import BackendProtocol
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from config import (
    WEB_DOWNLOAD_DEFAULT_MAX_FILE_MIB,
    WORKER_FINALIZATION_DEFAULT_MODEL_ROUNDS,
    WORKER_PROGRESS_DEFAULT_TOOL_CALLS,
)
from knowledge_rag.runtime import with_knowledge_tool
from prompt_loader import load_prompt
from skill_runtime.middleware import RoleSkillsMiddleware
from workers.context import WorkerRuntimeContextMiddleware
from workers.evidence_refs import EvidenceReferenceMiddleware
from workers.file_access import TaskFileAccessMiddleware
from workers.general_worker import DEEP_AGENT_FILE_TOOLS
from workers.progress import (
    PUBLISH_WORKER_PROGRESS_NAME,
    WorkerProgressMiddleware,
    WorkerProgressState,
    publish_worker_progress,
)
from workers.submission import SUBMIT_FOR_REVIEW_NAME, submit_for_review
from tools.web_artifacts import (
    DOWNLOAD_WEB_ARTIFACT_NAME,
    create_download_web_artifact_tool,
)


WEB_WORKER_NAME = "web_worker"
WEB_BUSINESS_TOOL_NAMES = frozenset(
    {"get_current_time", "web_search", "find_github_mirror", "fetch_webpage",
     "attachment_to_text", "ocr_image"}
)
BROWSER_TOOL_PREFIX = "browser_"


from knowledge_rag.middleware import KnowledgeMiddleware
from workers.history_archive import ExecutionHistoryMiddleware
from workers.execution_state import ExecutionStateMiddleware

def _tool_name(tool: BaseTool) -> str:
    name = str(getattr(tool, "name", "")).strip()
    if not name:
        raise ValueError("web_worker received a tool without a name.")
    return name


def select_web_worker_tools(
    tools: Sequence[BaseTool],
) -> list[BaseTool]:
    """Keep only Web-domain and leased Playwright tools."""

    selected: list[BaseTool] = []
    seen_names: set[str] = set()
    for current_tool in tools:
        name = _tool_name(current_tool)
        if name in seen_names:
            raise ValueError(f"web_worker received duplicate tool name: {name}")
        seen_names.add(name)
        if name in WEB_BUSINESS_TOOL_NAMES or name.startswith(
            BROWSER_TOOL_PREFIX
        ):
            selected.append(current_tool)

    selected_names = {_tool_name(tool) for tool in selected}
    if "web_search" not in selected_names:
        raise ValueError("web_worker requires the web_search tool.")
    return selected


def create_web_worker(
    model: str | BaseChatModel,
    *,
    summary_model=None,
    tools: Sequence[BaseTool],
    progress_every_tool_calls: int = WORKER_PROGRESS_DEFAULT_TOOL_CALLS,
    finalization_model_rounds: int = WORKER_FINALIZATION_DEFAULT_MODEL_ROUNDS,
    schema_repair_max_rounds: int = 3,
    download_max_file_mib: int = WEB_DOWNLOAD_DEFAULT_MAX_FILE_MIB,
    middleware: Sequence[AgentMiddleware] | None = None,
    backend: BackendProtocol | None = None,
    permissions: Sequence[FilesystemPermission] | None = None,
    checkpointer=None,
    store=None,
) -> CompiledStateGraph:
    """Build one Web Worker graph for one browser-session lease."""

    web_tools = with_knowledge_tool(select_web_worker_tools(tools))
    if any(
        _tool_name(tool)
        in {PUBLISH_WORKER_PROGRESS_NAME, SUBMIT_FOR_REVIEW_NAME}
        for tool in web_tools
    ):
        raise ValueError("Web control tool was supplied as a business tool.")

    download_tool = create_download_web_artifact_tool(
        max_file_mib=download_max_file_mib,
    )
    if DOWNLOAD_WEB_ARTIFACT_NAME in {
        _tool_name(current_tool) for current_tool in web_tools
    }:
        raise ValueError("Web download tool was supplied twice.")

    resolved_backend = backend or StateBackend()
    resolved_permissions = list(permissions or ())
    graph = create_deep_agent(
        model=model,
        tools=[
            *web_tools,
            download_tool,
            publish_worker_progress,
            submit_for_review,
        ],
        system_prompt=load_prompt("workers/web_worker"),
        backend=resolved_backend,
        permissions=resolved_permissions,
        middleware=[
            *(middleware or ()),
            KnowledgeMiddleware('Web Agent'),
            ExecutionHistoryMiddleware(),
            ExecutionStateMiddleware(),
            WorkerCompactionMiddleware(summary_model if summary_model is not None else model),
            RoleSkillsMiddleware(model, "web", [*DEEP_AGENT_FILE_TOOLS, *(_tool_name(t) for t in web_tools)],
                                 policies=middleware or (), backend=resolved_backend),
            WorkerRuntimeContextMiddleware(),
            TaskFileAccessMiddleware(),
            FilesystemMiddleware(
                custom_tool_descriptions=json.loads(load_prompt("tools/filesystem")),
                backend=resolved_backend,
                tools=list(DEEP_AGENT_FILE_TOOLS),
                _permissions=resolved_permissions,
            ),
            WorkerProgressMiddleware(
                every_tool_calls=progress_every_tool_calls,
                finalization_model_rounds=finalization_model_rounds,
                schema_repair_max_rounds=schema_repair_max_rounds,
            ),
            EvidenceReferenceMiddleware(),
        ],
        state_schema=WorkerProgressState,
        checkpointer=checkpointer,
        store=store,
        name=WEB_WORKER_NAME,
    )
    from trace_callbacks import configure_graph
    return configure_graph(graph, "web")


__all__ = [
    "BROWSER_TOOL_PREFIX",
    "WEB_BUSINESS_TOOL_NAMES",
    "WEB_WORKER_NAME",
    "create_web_worker",
    "select_web_worker_tools",
]
