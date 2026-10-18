from workers.compaction import WorkerCompactionMiddleware
from workers.default_middleware import DisabledDefaultSummarizationMiddleware
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
from workers.plan_objection import objection_tools, WorkerObjectionMiddleware
from workers.general_worker import DEEP_AGENT_FILE_TOOLS
from workers.progress import (
    PUBLISH_WORKER_PROGRESS_NAME,
    WorkerProgressMiddleware,
    WorkerProgressState,
    publish_worker_progress,
)
from workers.submission import COMPLETE_WEB_STEP_NAME, complete_web_step
from tools.web_artifacts import (
    DOWNLOAD_WEB_ARTIFACT_NAME,
    create_download_web_artifact_tool,
)


WEB_WORKER_NAME = "web_worker"
WEB_BUSINESS_TOOL_NAMES = frozenset(
    {"get_current_time", "web_search", "find_github_mirror", "fetch_webpage",
     "attachment_to_text", "ocr_image", "search_1688_products",
     "list_cinemas", "list_city_movies", "get_cinema_showtimes", "get_movie_showtimes",
     "host_list_files", "host_search_files", "host_import_file",
     "send_local_file_to_feishu"}
)
from feishu_exports import export_tool_available
from tools.capability_expansion import (
    REQUEST_ADDITIONAL_CAPABILITIES_NAME,
    request_additional_capabilities,
)
from step_user_question import ASK_USER_NAME, create_ask_user_tool
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
        if name == "send_local_file_to_feishu" and not export_tool_available():
            continue
        if (name in WEB_BUSINESS_TOOL_NAMES
                or name.startswith(BROWSER_TOOL_PREFIX)
                or name.startswith("maps_")):
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
        in {
            "raise_plan_objection",
            PUBLISH_WORKER_PROGRESS_NAME,
            COMPLETE_WEB_STEP_NAME,
            REQUEST_ADDITIONAL_CAPABILITIES_NAME,
            ASK_USER_NAME,
        }
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
    resolved_middleware = list(middleware or ())
    progress_checkpoint_resolver = None
    for current_middleware in resolved_middleware:
        if not getattr(
            current_middleware,
            "select_web_progress_checkpoint",
            False,
        ):
            continue
        candidate = getattr(
            current_middleware,
            "progress_checkpoint_required",
            None,
        )
        if callable(candidate):
            progress_checkpoint_resolver = candidate
            break
    ask_user = create_ask_user_tool("WEB_WORKER")
    graph = create_deep_agent(
        model=model,
        tools=[
            *objection_tools(),
            *web_tools,
            download_tool,
            ask_user,
            request_additional_capabilities,
            publish_worker_progress,
            complete_web_step,
        ],
        system_prompt=load_prompt("workers/web_worker"),
        backend=resolved_backend,
        permissions=resolved_permissions,
        middleware=[
            WorkerObjectionMiddleware(),
            DisabledDefaultSummarizationMiddleware(),
            *resolved_middleware,
            KnowledgeMiddleware('Web Agent'),
            ExecutionHistoryMiddleware(),
            ExecutionStateMiddleware(),
            RoleSkillsMiddleware(model, "web", [*DEEP_AGENT_FILE_TOOLS, ASK_USER_NAME, *(_tool_name(t) for t in web_tools)],
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
                progress_checkpoint_resolver=progress_checkpoint_resolver,
            ),
            EvidenceReferenceMiddleware(),
            WorkerCompactionMiddleware(summary_model if summary_model is not None else model, request_aware=True),
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
