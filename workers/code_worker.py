"""Dedicated Deep Agents runtime for implementation work."""

from __future__ import annotations
from workers.compaction import WorkerCompactionMiddleware

import json
from collections.abc import Sequence

from deepagents import FilesystemMiddleware, create_deep_agent
from workers.file_access import TaskFileAccessMiddleware
from deepagents.backends import StateBackend
from deepagents.backends.protocol import BackendProtocol
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from config import WORKER_PROGRESS_DEFAULT_TOOL_CALLS
from knowledge_rag.runtime import with_knowledge_tool
from prompt_loader import load_prompt
from skill_runtime.middleware import RoleSkillsMiddleware
from workers.code_state import CodeAgentState, CodeRuntimeContextMiddleware
from workers.code_submission import (
    RESPOND_TO_CODE_REVIEW_NAME,
    SUBMIT_CONTINUED_CODE_FOR_REVIEW_NAME,
    SUBMIT_CODE_FOR_REVIEW_NAME,
    respond_to_code_review,
    submit_continued_code_for_review,
    submit_code_for_review,
)
from workers.context import WorkerRuntimeContextMiddleware
from workers.evidence_refs import EvidenceReferenceMiddleware
from workers.general_worker import _tool_name, select_general_worker_tools
from workers.progress import (
    PUBLISH_WORKER_PROGRESS_NAME,
    WorkerProgressMiddleware,
    publish_worker_progress,
)
from workers.tool_access import WorkerToolAllowlistMiddleware


CODE_WORKER_NAME = "code_worker"
CODE_WORKER_FILESYSTEM_TOOLS = [
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "delete",
    "glob",
    "grep",
    "execute",
]


from knowledge_rag.middleware import KnowledgeMiddleware
from workers.history_archive import ExecutionHistoryMiddleware
from workers.execution_state import ExecutionStateMiddleware

def create_code_worker(
    model: str | BaseChatModel,
    *,
    summary_model=None,
    tools: Sequence[BaseTool] | None = None,
    backend: BackendProtocol | None = None,
    progress_every_tool_calls: int = WORKER_PROGRESS_DEFAULT_TOOL_CALLS,
    schema_repair_max_rounds: int = 3,
    middleware: Sequence[AgentMiddleware] | None = None,
    checkpointer=None,
    store=None,
) -> CompiledStateGraph:
    """Build a checkpointable single-writer implementation Agent."""

    business_tools = with_knowledge_tool(select_general_worker_tools(tools))
    reserved = {
        PUBLISH_WORKER_PROGRESS_NAME,
        SUBMIT_CODE_FOR_REVIEW_NAME,
        RESPOND_TO_CODE_REVIEW_NAME,
        SUBMIT_CONTINUED_CODE_FOR_REVIEW_NAME,
    }
    if any(_tool_name(item) in reserved for item in business_tools):
        raise ValueError(
            "Code Worker control tool was supplied as a business tool."
        )

    resolved_backend = backend or StateBackend()
    allowed_names = {
        *CODE_WORKER_FILESYSTEM_TOOLS,
        *(_tool_name(item) for item in business_tools),
        *reserved,
    }
    filesystem = FilesystemMiddleware(
        custom_tool_descriptions=json.loads(load_prompt("tools/filesystem")),
        backend=resolved_backend,
        tools=CODE_WORKER_FILESYSTEM_TOOLS,
    )
    from middlewares import DynamicExecutionBudgetMiddleware
    from workers.code_finalization import CodeWorkerBudgetMiddleware, CodeWorkerProgressMiddleware
    policies = [
        CodeWorkerBudgetMiddleware(
            finalization_model_rounds=m.finalization_model_rounds,
            schema_repair_max_rounds=schema_repair_max_rounds,
        )
        if type(m) is DynamicExecutionBudgetMiddleware
        else m
        for m in (middleware or ())
    ]
    progress = CodeWorkerProgressMiddleware(
        every_tool_calls=progress_every_tool_calls,
        schema_repair_max_rounds=schema_repair_max_rounds,
        finalization_tools=(SUBMIT_CODE_FOR_REVIEW_NAME, RESPOND_TO_CODE_REVIEW_NAME, SUBMIT_CONTINUED_CODE_FOR_REVIEW_NAME),
    )

    graph = create_deep_agent(
        model=model,
        tools=[
            *business_tools,
            publish_worker_progress,
            submit_code_for_review,
            respond_to_code_review,
            submit_continued_code_for_review,
        ],
        system_prompt=load_prompt("workers/code_worker"),
        backend=resolved_backend,
        middleware=[
            *policies,
            KnowledgeMiddleware('Code Worker'),
            ExecutionHistoryMiddleware(),
            ExecutionStateMiddleware(),
            WorkerCompactionMiddleware(summary_model if summary_model is not None else model),
            RoleSkillsMiddleware(model, "code", allowed_names,
                                 policies=middleware or (), backend=resolved_backend),
            WorkerRuntimeContextMiddleware(execution_reserve=1),
            CodeRuntimeContextMiddleware("WORKER"),
            filesystem,
            TaskFileAccessMiddleware(sandbox=resolved_backend),
            progress,
            WorkerToolAllowlistMiddleware(allowed_names),
            EvidenceReferenceMiddleware(),
        ],
        state_schema=CodeAgentState,
        checkpointer=checkpointer,
        store=store,
        name=CODE_WORKER_NAME,
    )
    from trace_callbacks import configure_graph
    return configure_graph(graph, "code")


__all__ = [
    "CODE_WORKER_FILESYSTEM_TOOLS",
    "CODE_WORKER_NAME",
    "create_code_worker",
]
