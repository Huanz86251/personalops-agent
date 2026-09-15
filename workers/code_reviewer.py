"""Dedicated Deep Agents runtime for independent code verification."""

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

from knowledge_rag.runtime import with_knowledge_tool
from prompt_loader import load_prompt
from skill_runtime.middleware import RoleSkillsMiddleware
from workers.code_state import (
    CodeAgentCompletionMiddleware,
    CodeAgentState,
    CodeRuntimeContextMiddleware,
)
from workers.code_submission import (
    REQUEST_CODE_WORKER_REPAIR_NAME,
    SUBMIT_CODE_REVIEW_NAME,
    request_code_worker_repair,
    submit_code_review,
)
from workers.code_publisher import (
    PUBLISH_REVIEWED_CANDIDATE_NAME,
    publish_reviewed_candidate,
)
from workers.context import WorkerRuntimeContextMiddleware
from workers.evidence_refs import EvidenceReferenceMiddleware
from workers.general_worker import _tool_name, select_general_worker_tools
from workers.tool_access import WorkerToolAllowlistMiddleware


CODE_REVIEWER_NAME = "code_reviewer"
CODE_REVIEWER_FILESYSTEM_TOOLS = [
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

def create_code_reviewer(
    model: str | BaseChatModel,
    *,
    summary_model=None,
    tools: Sequence[BaseTool] | None = None,
    backend: BackendProtocol | None = None,
    schema_repair_max_rounds: int = 3,
    middleware: Sequence[AgentMiddleware] | None = None,
    checkpointer=None,
    store=None,
) -> CompiledStateGraph:
    """Build an independent Reviewer with its own checkpoint and tools."""

    business_tools = with_knowledge_tool(select_general_worker_tools(tools))
    reserved = {
        REQUEST_CODE_WORKER_REPAIR_NAME,
        PUBLISH_REVIEWED_CANDIDATE_NAME,
        SUBMIT_CODE_REVIEW_NAME,
    }
    if any(_tool_name(item) in reserved for item in business_tools):
        raise ValueError(
            "Code Reviewer control tool was supplied as a business tool."
        )

    resolved_backend = backend or StateBackend()
    allowed_names = {
        *CODE_REVIEWER_FILESYSTEM_TOOLS,
        *(_tool_name(item) for item in business_tools),
        *reserved,
    }
    filesystem = FilesystemMiddleware(
        custom_tool_descriptions=json.loads(load_prompt("tools/filesystem")),
        backend=resolved_backend,
        tools=CODE_REVIEWER_FILESYSTEM_TOOLS,
    )

    from middlewares import DynamicExecutionBudgetMiddleware
    from workers.code_finalization import CodeReviewerBudgetMiddleware, CodeReviewerProgressMiddleware
    policies = [
        CodeReviewerBudgetMiddleware(
            finalization_model_rounds=m.finalization_model_rounds,
            schema_repair_max_rounds=schema_repair_max_rounds,
        )
        if type(m) is DynamicExecutionBudgetMiddleware
        else m
        for m in (middleware or ())
    ]
    progress = CodeReviewerProgressMiddleware(
        schema_repair_max_rounds=schema_repair_max_rounds,
        finalization_tools=(SUBMIT_CODE_REVIEW_NAME,))
    graph = create_deep_agent(
        model=model,
        tools=[
            *business_tools,
            request_code_worker_repair,
            publish_reviewed_candidate,
            submit_code_review,
        ],
        system_prompt=load_prompt("reviewers/code"),
        backend=resolved_backend,
        middleware=[
            *policies,
            KnowledgeMiddleware('Code Reviewer'),
            ExecutionHistoryMiddleware(),
            ExecutionStateMiddleware(),
            WorkerCompactionMiddleware(summary_model if summary_model is not None else model),
            RoleSkillsMiddleware(model, "reviewer", allowed_names,
                                 policies=middleware or (), backend=resolved_backend),
            WorkerRuntimeContextMiddleware(execution_reserve=1),
            CodeRuntimeContextMiddleware("REVIEWER"),
            filesystem,
            TaskFileAccessMiddleware(sandbox=resolved_backend),
            CodeAgentCompletionMiddleware(),
            progress,
            WorkerToolAllowlistMiddleware(allowed_names),
            EvidenceReferenceMiddleware(),
        ],
        state_schema=CodeAgentState,
        checkpointer=checkpointer,
        store=store,
        name=CODE_REVIEWER_NAME,
    )
    from trace_callbacks import configure_graph
    return configure_graph(graph, "code_reviewer")


__all__ = [
    "CODE_REVIEWER_FILESYSTEM_TOOLS",
    "CODE_REVIEWER_NAME",
    "create_code_reviewer",
]
