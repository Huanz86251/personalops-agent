from skill_runtime.preparation import skill_selector_scope
from knowledge_rag.runtime import RetrievalHub, retrieval_scope
import asyncio
from trace_presentation import conversation_trace_name, run_name
from trace_overview import conversation_overview
import logging
import os
from contextlib import nullcontext
import json
from pathlib import Path
from contextvars import (
    Context,
)

from dataclasses import (
    dataclass,
)
from datetime import (
    datetime,
    timezone,
)
from typing import (
    Any,
    AsyncContextManager,
)
from middlewares import (
    DynamicExecutionBudgetMiddleware,
    ToolsetRouterMiddleware,
)
from mcp_runtime import (
    EmailMCPRuntime,
    PlaywrightMCPPool,
)
from amap_mcp_runtime import AmapMapsMCPRuntime
from uuid import uuid4

from dotenv import dotenv_values

from langgraph.checkpoint.sqlite.aio import (
    AsyncSqliteSaver,
)
from memory import (
    MemoryService,
    RetrievedMemory,
)
from memory_write_gate import MemOperatorWriteGate
from agent import (
    build_role_model,
    generate_conversation_title,
)
from config import Settings
from prompt_injection_guard import (
    PromptInjectionGuard,
    PromptInjectionGuardConfig,
    PromptInjectionGuardMiddleware,
)
from delivery.models import PromotionStatus, WorkspacePromotion
from delivery.service import WorkspacePromotionService, format_promotion_summary
from tools.web_tools import (
    configure_web_search_parallelism,
)
from step_user_question import (
    ask_user_runtime_scope,
)

from path import AGENT_DATA_ROOT, WORKSPACE_ROOT
from run_workspace import (
    cleanup_expired_run_workspaces,
    inherit_replacement_workspace,
    initialize_conversation_workspace,
    initialize_run_workspace,
    read_replacement_inheritance_receipt,
    read_run_supersession_receipt,
    write_run_cancellation_receipt,
    write_run_supersession_receipt,
)
from integration_repository import IntegrationRepository
from observability import (
    set_span_attributes,
    set_span_output,
    trace_context,
    trace_span,
)
from langgraph.store.base import (
    IndexConfig,
)
from langgraph.store.sqlite import (
    AsyncSqliteStore,
)

from retrieval_models import (
    GTE_EMBEDDING_DIMENSIONS,
    RetrievalModelManager,
)

from planning_graph import (
    build_planning_graph,
)
from hard_planning import run_hard_worker_leader
from eventing.models import (
    EventAction,
    EventRun,
    EventStatus,
    RunStatus,
    build_planning_thread_id_from_parts,
)
from eventing.store import AsyncEventStore
from eventing.run_pump import EventPauseControl
from progress_events import (
    ProgressCallback,
)
from tools.time_tools import (
    get_current_time,
)
from scheduling import ScheduleService
from tools.schedule_tools import configure_schedule_service
from host_file_access import configure_host_file_judge

from toolsets import (
    DEFAULT_TOOLSET_REGISTRY,
    NO_TOOL_ROUTE,
)
from toolset_router import ToolsetRouter
from conversation_compaction import (
    SummaryCheckpoint, compress_dialogue, dialogue_token_estimate, read_summary_checkpoint,
)

from planning_models import (
    DialogueMessage,
    PlanningContextPack,
    PlanningReplacementContext,
)
from workers import (
    WorkerAgentRegistry,
    WorkerGroupCoordinator,
    WorkerLeadershipBridge,
    CodeSandboxManager,
    CodeSandboxPolicy,
    CodeStepRuntime,
    GeneralStepRuntime,
    WebStepRuntime,
    create_general_worker,
)
from workers.leadership_models import LeadershipDecisionResult
from workers.wake_policy import LeadershipWakePolicy
from workers.code_submission import (
    REQUEST_CODE_WORKER_REPAIR_NAME,
    RESPOND_TO_CODE_REVIEW_NAME,
    SUBMIT_CODE_FOR_REVIEW_NAME,
    SUBMIT_CODE_REVIEW_NAME,

)
from workers.code_publisher import PUBLISH_REVIEWED_CANDIDATE_NAME
from workers.general_completion import GENERAL_REPORT_NAME
from workers.submission import COMPLETE_WEB_STEP_NAME
logger = logging.getLogger(
    "agent"
)


# Toolset routing may rank and hide business capabilities, but must never hide
# the control-plane tools that a role's runtime can require on a later model
# round. Keep these role baselines named and import canonical terminal-tool
# constants so production composition and regression tests cannot silently
# drift apart.
GENERAL_TOOLSET_BASELINE_TOOL_NAMES = (
    "raise_plan_objection",
    "ask_user",
    "search_knowledge",
    "read_knowledge",
    "read_execution_history",
    GENERAL_REPORT_NAME,
    "request_additional_capabilities",
)

WEB_TOOLSET_BASELINE_TOOL_NAMES = (
    "raise_plan_objection",
    "ask_user",
    "search_knowledge",
    "read_knowledge",
    "read_execution_history",
    "download_web_artifact",
    "publish_worker_progress",
    COMPLETE_WEB_STEP_NAME,
    "request_additional_capabilities",
)

CODE_TOOLSET_BASELINE_TOOL_NAMES = (
    "raise_plan_objection",
    "ask_user",
    "search_knowledge",
    "read_knowledge",
    "read_execution_history",
    "publish_worker_progress",
    SUBMIT_CODE_FOR_REVIEW_NAME,
    RESPOND_TO_CODE_REVIEW_NAME,

    REQUEST_CODE_WORKER_REPAIR_NAME,
    PUBLISH_REVIEWED_CANDIDATE_NAME,
    SUBMIT_CODE_REVIEW_NAME,
    "request_additional_capabilities",
)


CHECKPOINT_PATH = (
    AGENT_DATA_ROOT
    / "checkpoints.sqlite3"
)

MEMORY_STORE_PATH = (
    AGENT_DATA_ROOT
    / "memories.sqlite3"
)


@dataclass(frozen=True)
class ConversationInfo:
    """提供给飞书入口使用的Conversation信息。"""

    conversation_id: str
    thread_id: str
    short_id: str

    title: str
    channel_key: str

    created_at: str
    last_active_at: str

    title_generated: bool

def _memory_to_trace_item(
    memory: RetrievedMemory,
) -> dict[
    str,
    object,
]:
    """把召回记忆转换成便于Phoenix展示的结构。"""

    dense_score = (
        round(
            memory.dense_score,
            6,
        )

        if memory.dense_score
        is not None

        else None
    )

    rerank_score = (
        round(
            memory.rerank_score,
            6,
        )

        if memory.rerank_score
        is not None

        else None
    )

    lexical_score = (
        round(memory.lexical_score, 6)
        if memory.lexical_score is not None
        else None
    )

    retrieval_score = (
        round(memory.retrieval_score, 6)
        if memory.retrieval_score is not None
        else None
    )

    return {
        "memory_id": (
            memory.memory_id
        ),

        "content": (
            memory.content
        ),

        "memory_type": (
            memory.memory_type
        ),

        "importance": (
            memory.importance
        ),

        "confidence": (
            memory.confidence
        ),

        "dense_score": (
            dense_score
        ),

        "lexical_score": (
            lexical_score
        ),

        "rerank_score": (
            rerank_score
        ),

        "retrieval_score": (
            retrieval_score
        ),

        "graph_distance": (
            memory.graph_distance
        ),

        "valid_from": (
            memory.valid_from
        ),

        "expires_at": (
            memory.expires_at
        ),
    }



def _message_content_to_text(
    content: Any,
) -> str:
    """把LangChain消息内容转换成普通文字。"""

    if isinstance(
        content,
        str,
    ):
        return content.strip()

    if isinstance(
        content,
        list,
    ):
        parts: list[str] = []

        for item in content:
            if isinstance(
                item,
                str,
            ):
                parts.append(
                    item
                )

                continue

            if not isinstance(
                item,
                dict,
            ):
                continue

            text = item.get(
                "text"
            )

            if isinstance(
                text,
                str,
            ):
                parts.append(
                    text
                )

        return "".join(
            parts
        ).strip()

    if content is None:
        return ""

    return str(
        content
    ).strip()


def _read_message_role(
    message: Any,
) -> str:
    """统一读取Conversation消息角色。"""

    if isinstance(
        message,
        dict,
    ):
        role = (
            message.get(
                "role"
            )
            or message.get(
                "type"
            )
            or ""
        )

    else:
        role = (
            getattr(
                message,
                "type",
                "",
            )
            or getattr(
                message,
                "role",
                "",
            )
            or ""
        )

    return str(
        role
    ).strip().lower()


def _extract_dialogue_turns(
    messages: list[
        Any
    ],
) -> list[
    list[
        DialogueMessage
    ]
]:
    """只提取用户消息和最终助手回答，并按用户Turn分组。"""

    turns: list[
        list[
            DialogueMessage
        ]
    ] = []

    current_turn: list[
        DialogueMessage
    ] = []

    for message in messages:
        role = _read_message_role(
            message
        )

        if isinstance(
            message,
            dict,
        ):
            content = message.get(
                "content",
                "",
            )

        else:
            content = getattr(
                message,
                "content",
                "",
            )

        message_text = content if isinstance(content, str) else _message_content_to_text(content)

        if not message_text:
            continue

        if role in {
            "user",
            "human",
        }:
            if current_turn:
                turns.append(
                    current_turn
                )

            current_turn = [
                DialogueMessage(
                    role="user",
                    content=message_text,
                )
            ]

            continue

        if role not in {
            "assistant",
            "ai",
        }:
            continue

        # 没有前置用户消息的孤立助手消息，
        # 不作为Conversation Turn提供给Hard节点。
        if not current_turn:
            continue

        current_turn.append(
            DialogueMessage(
                role="assistant",
                content=message_text,
            )
        )

    if current_turn:
        turns.append(
            current_turn
        )

    return turns


def _format_dialogue_messages(
    messages: list[
        DialogueMessage
    ],
) -> str:
    """把尚未批量摘要的较早对话转换成可读文字。"""

    return "\n\n".join(
        (
            "用户"
            if message.role == "user"
            else "助手"
        )
        + "："
        + message.content
        for message in messages
    )


def _compact_context_text(
    text: str,
    *,
    max_chars: int,
) -> str:
    """确定性限制Hard摘要区域的总字符数。"""

    normalized_text = text.strip()

    if len(
        normalized_text
    ) <= max_chars:
        return normalized_text

    marker = (
        "\n\n"
        "……较早上下文中部已省略……"
        "\n\n"
    )

    if max_chars <= len(
        marker
    ):
        return normalized_text[
            :max_chars
        ]

    available_chars = (
        max_chars
        - len(
            marker
        )
    )

    head_chars = (
        available_chars
        // 2
    )

    tail_chars = (
        available_chars
        - head_chars
    )

    return (
        normalized_text[
            :head_chars
        ]
        + marker
        + normalized_text[
            -tail_chars:
        ]
    )


def _planning_run_handoff_material(result: dict[str, Any]) -> str:
    """Project a completed run into bounded evidence, not its raw trace."""
    def data(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        return {}

    def clipped(value: Any, limit: int = 500) -> str:
        return str(value or "")[:limit]

    reports = []
    for raw_report in list(result.get("completed_step_reports") or [])[-12:]:
        report = data(raw_report)
        reports.append({
            "step_id": report.get("step_id"),
            "status": clipped(report.get("status"), 60),
            "summary": clipped(report.get("summary")),
            "stop_reason": clipped(report.get("stop_reason"), 180),
            "criterion_results": [
                {
                    "criterion_id": item.get("criterion_id"),
                    "criterion": clipped(item.get("criterion"), 160),
                    "status": clipped(item.get("status"), 60),
                    "evidence": clipped(item.get("evidence"), 220),
                }
                for item in (data(raw) for raw in report.get("criterion_results") or [])
            ][:8],
        })
    final_decision = data(result.get("final_review_decision"))
    final_decision_projection = {
        "action": clipped(final_decision.get("action"), 40),
        "status": clipped(final_decision.get("status"), 60),
        "review_reason": clipped(final_decision.get("review_reason"), 600),
        "criterion_reviews": [
            {
                "criterion_id": item.get("criterion_id"),
                "status": clipped(item.get("status"), 60),
                "observed_result": clipped(item.get("observed_result"), 300),
                "missing_requirement": clipped(item.get("missing_requirement"), 240),
            }
            for item in (
                data(raw)
                for raw in final_decision.get("criterion_reviews") or []
            )
        ][:12],
        "unmet_success_criteria": [
            clipped(item, 200)
            for item in final_decision.get("unmet_success_criteria") or []
        ][:8],
    } if final_decision else {}
    material = {
        "plan_objective": clipped(result.get("plan_objective")),
        "final_status": clipped(result.get("final_status"), 60),
        "overall_stop_reason": clipped(result.get("overall_stop_reason"), 300),
        "unmet_success_criteria": [clipped(item, 200) for item in
                                   list(result.get("unmet_success_criteria") or [])[:8]],
        "completed_step_reports": reports,
        "replan_history": [
            clipped(data(item).get("reason") or data(item).get("summary"), 200)
            for item in list(result.get("replan_history") or [])[-3:]
        ],
        # The final decision is authoritative when it corrected an earlier
        # Step Reporter.  Keep its bounded verdict, never its full answer or
        # raw evidence bodies, so the next turn does not inherit stale UNKNOWNs.
        "final_review_decision": final_decision_projection,
    }
    return _compact_context_text(
        json.dumps(material, ensure_ascii=False, default=str), max_chars=12000
    )


class ConversationRuntime:
    """管理模型、Agent、Conversation和Checkpoint。"""

    def __init__(
        self,
        settings: Settings,
        tools: list,
    ) -> None:
        self.settings = settings
        configure_web_search_parallelism(
            settings
            .runtime_concurrency
            .web_search_max_parallelism
        )
        self._base_tools = tuple(tools)
        self.tools = list(self._base_tools)
        self.email_mcp = EmailMCPRuntime(settings.email_mcp)
        # A personal AMap Web-service key is isolated in a Git-ignored local
        # file.  It takes precedence over the shared Bailian trial service.
        amap_local = dotenv_values(
            Path(__file__).resolve().parent / ".env.amap.local"
        )
        personal_amap_key = (
            os.getenv("AMAP_MAPS_API_KEY", "")
            or str(amap_local.get("AMAP_MAPS_API_KEY") or "")
        ).strip()
        self.amap_mcp = AmapMapsMCPRuntime(
            os.getenv("DASHSCOPE_API_KEY", ""),
            personal_api_key=personal_amap_key,
        )
        self.playwright_mcp = (
            PlaywrightMCPPool(
                max_sessions=(
                    settings
                    .runtime_concurrency
                    .playwright_max_sessions
                )
            )
        )
        self.model = None
        self.summary_model = None
        self.role_models = {}
        self.hard_model = None
        # The Deep Agent Worker is the single execution and conversation-state
        # graph. WorkerLeadershipBridge wraps this graph; there is no parallel
        # legacy Simple Executor runtime.
        self.general_worker: WorkerLeadershipBridge | None = None
        self.web_worker: WebStepRuntime | None = None
        self.code_runtime: CodeStepRuntime | None = None
        self.event_store = AsyncEventStore()
        # External schedule targets stay gated until main.py has both the
        # Feishu connection and EventRunPump ready.  Windows-only unit/service
        # instances retain ScheduleService's enabled-by-default behaviour.
        self.schedule_service = ScheduleService(
            dispatch_enabled=False,
            context_resolver=self._resolve_schedule_context,
        )
        self.workspace_promotions = WorkspacePromotionService(
            event_store=self.event_store,
            approval_mode=settings.delivery.approval_mode,
        )

        # 编译后的外层Hard Planning Graph。
        self.planning_graph = None

        # 提供给Hard Supervisor查看的能力目录。
        self.toolset_catalog: list[
            dict[
                str,
                str,
            ]
        ] = []
        self.memory_service: (
            MemoryService
            | None
        ) = None
        self._checkpointer: (
            AsyncSqliteSaver
            | None
        ) = None

        self._checkpointer_context: (
            AsyncContextManager[
                AsyncSqliteSaver
            ]
            | None
        ) = None
        self.retrieval_models: (
            RetrievalModelManager
            | None
        ) = None
        self.toolset_router: ToolsetRouter | None = None

        self._memory_store: (
            AsyncSqliteStore
            | None
        ) = None

        self._memory_store_context: (
            AsyncContextManager[
                AsyncSqliteStore
            ]
            | None
        ) = None
        self._thread_locks: dict[
            str,
            asyncio.Lock,
        ] = {}
        # 保存尚未完成的后台任务。
        # 防止任务对象被垃圾回收，
        # 并在程序关闭时统一等待或取消。
        self._background_tasks: set[
            asyncio.Task
        ] = set()

        # 同一个Conversation中的记忆写入必须保持顺序。
        # 不同Conversation之间仍然可以并发执行。
        self._memory_postprocess_locks: dict[
            str,
            asyncio.Lock,
        ] = {}

    async def _resolve_schedule_context(self, event_id: str) -> tuple[str, str]:
        """Resolve recipient IDs from a persisted trusted Event, never model input."""

        event = await self.event_store.require_event(event_id)
        reply_target_id = str(event.reply_target_id or "").strip()
        if not reply_target_id:
            raise RuntimeError("当前Event没有飞书回复地址，不能创建飞书日程。")
        return event.conversation_id, reply_target_id

    async def _recover_code_startup(
        self,
        code_runtime: CodeStepRuntime,
    ) -> list[EventRun]:
        """Recover live CODE ownership without reviving historical Event runs."""

        terminal_runs = await self.event_store.list_runs(
            statuses=[
                RunStatus.COMPLETED,
                RunStatus.CANCELLED,
                RunStatus.SUPERSEDED,
                RunStatus.FAILED,
            ]
        )
        code_recovery = await asyncio.to_thread(
            code_runtime.recover_startup_sessions,
            excluded_run_ids={run.event_id for run in terminal_runs},
        )
        if code_recovery:
            logger.info("CODE启动恢复扫描完成 | outcomes=%s", code_recovery)

        # CANCEL/REPLACE may have committed before their resource callback.
        # Reconcile only that run's resources, never execute its old task again.
        applied_events = await self.event_store.list_events(
            statuses=[EventStatus.APPLIED]
        )
        cancel_by_target = {
            event.target_event_id: event
            for event in applied_events
            if event.action is EventAction.CANCEL and event.target_event_id
        }
        for cancelled_run in terminal_runs:
            if cancelled_run.status is not RunStatus.CANCELLED:
                continue
            cancel_event = cancel_by_target.get(cancelled_run.event_id)
            if cancel_event is None:
                continue
            try:
                code_records = await asyncio.to_thread(
                    code_runtime.cancel_run,
                    cancelled_run.event_id,
                    reason=(
                        "Startup reconciled a user-cancelled Event before "
                        "accepting new work."
                    ),
                )
            except Exception as error:
                logger.warning(
                    "CODE取消启动归档未完成，记录已保留 | run_id=%s | error_type=%s",
                    cancelled_run.event_id,
                    type(error).__name__,
                )
                continue
            layout = initialize_run_workspace(cancelled_run.event_id)
            await asyncio.to_thread(
                write_run_cancellation_receipt,
                layout=layout,
                cancel_event_id=cancel_event.event_id,
                target_event_id=cancelled_run.event_id,
                code_final_records=code_records,
            )

        all_events = await self.event_store.list_events()
        replace_by_target = {
            event.target_event_id: event
            for event in all_events
            if event.action is EventAction.REPLACE and event.target_event_id
        }
        for superseded_run in terminal_runs:
            if superseded_run.status is not RunStatus.SUPERSEDED:
                continue
            replacement_event = replace_by_target.get(superseded_run.event_id)
            if replacement_event is None:
                continue
            try:
                code_records = await asyncio.to_thread(
                    code_runtime.supersede_run,
                    superseded_run.event_id,
                    reason=(
                        "Startup reconciled a user-superseded Event before "
                        "accepting new work."
                    ),
                )
            except Exception as error:
                logger.warning(
                    "CODE替换启动归档未完成，记录已保留 | run_id=%s | error_type=%s",
                    superseded_run.event_id,
                    type(error).__name__,
                )
                continue
            source_layout = initialize_run_workspace(superseded_run.event_id)
            replacement_layout = initialize_run_workspace(replacement_event.event_id)
            await asyncio.to_thread(
                write_run_supersession_receipt,
                layout=source_layout,
                replacement_event_id=replacement_event.event_id,
                target_event_id=superseded_run.event_id,
                code_final_records=code_records,
            )
            await asyncio.to_thread(
                inherit_replacement_workspace,
                source_layout=source_layout,
                replacement_layout=replacement_layout,
                replacement_event_id=replacement_event.event_id,
            )
        return terminal_runs

    async def start(self) -> None:
        """启动Checkpoint、长期Store、本地模型和Agent。"""

        if self.general_worker is not None:
            return

        CHECKPOINT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        MEMORY_STORE_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint_context = (
            AsyncSqliteSaver
            .from_conn_string(
                str(CHECKPOINT_PATH)
            )
        )

        memory_context = None
        checkpointer = None
        memory_store = None
        retrieval_models = None
        planning_graph = None

        toolset_catalog: list[
            dict[
                str,
                str,
            ]
        ] = []

        try:
            await self.event_store.start()
            await self.schedule_service.start()
            configure_schedule_service(self.schedule_service)
            self.tools = list(self._base_tools)
            await self.email_mcp.start()
            self.tools.extend(self.email_mcp.tools)
            await self.amap_mcp.start()
            self.tools.extend(self.amap_mcp.tools)

            checkpointer = await (
                checkpoint_context
                .__aenter__()
            )

            retrieval_models = (
                RetrievalModelManager(
                    embedding_model_name=(
                        self.settings
                        .memory_embedding_model
                    ),

                    reranker_model_name=(
                        self.settings
                        .memory_reranker_model
                    ),

                    cache_dir=(
                        self.settings
                        .memory_model_cache_dir
                    ),

                    device=(
                        self.settings
                        .memory_model_device
                    ),

                    router_enabled=(
                        self.settings
                        .memory_router_enabled
                    ),

                    router_model_repo=(
                        self.settings
                        .memory_router_model_repo
                    ),

                    router_model_filename=(
                        self.settings
                        .memory_router_model_filename
                    ),

                    router_context_length=(
                        self.settings
                        .memory_router_context_length
                    ),

                    router_max_tokens=(
                        self.settings
                        .memory_router_max_tokens
                    ),
                )
            )

            await retrieval_models.aload()
            toolset_router = ToolsetRouter(
                retrieval_models=retrieval_models,
            )

            memory_context = (
                AsyncSqliteStore
                .from_conn_string(
                    str(
                        MEMORY_STORE_PATH
                    ),

                    index=IndexConfig(
                        dims=(
                            GTE_EMBEDDING_DIMENSIONS
                        ),

                        embed=(
                            retrieval_models
                            .langchain_embeddings
                        ),

                        fields=[
                            "content",
                        ],
                    ),
                )
            )

            memory_store = await (
                memory_context
                .__aenter__()
            )

            await memory_store.setup()
            role_models = {
                role: build_role_model(self.settings, role)
                for role, config in self.settings.role_models.items()
                if role != "extraction" or config.api_key
            }
            configure_host_file_judge(role_models["host_file_judge"])
            model = role_models["general"]
            hard_model = role_models["scheduler"]
            summary_model = role_models["summary"]
            memory_model = role_models.get("extraction")
            guard_settings = self.settings.prompt_injection_guard
            prompt_injection_guard = PromptInjectionGuard(
                PromptInjectionGuardConfig(
                    enabled=guard_settings.enabled,
                    cache_dir=self.settings.memory_model_cache_dir,
                    primary_model=guard_settings.primary_model,
                    primary_onnx_file=guard_settings.primary_onnx_file,
                    primary_threshold=guard_settings.primary_threshold,
                    primary_window_tokens=guard_settings.primary_window_tokens,
                    primary_overlap_tokens=guard_settings.primary_overlap_tokens,
                    primary_batch_size=guard_settings.primary_batch_size,
                    secondary_model=guard_settings.secondary_model,
                    secondary_window_tokens=guard_settings.secondary_window_tokens,
                    secondary_overlap_tokens=guard_settings.secondary_overlap_tokens,
                    secondary_batch_size=guard_settings.secondary_batch_size,
                    secondary_device=guard_settings.secondary_device,
                )
            )

            memory_service = (
                MemoryService(
                    store=memory_store,

                    retrieval_models=(
                        retrieval_models
                    ),

                    model=memory_model,

                    timezone_name=(
                        "Asia/Shanghai"
                    ),

                    reranker_threshold=(
                        self.settings
                        .memory_reranker_threshold
                    ),

                    lexical_limit=(
                        self.settings
                        .memory_bm25_limit
                    ),

                    write_gate=(
                        MemOperatorWriteGate(
                            model_name=(
                                self.settings.memory_write_gate_model
                            ),
                            cache_dir=(
                                self.settings.memory_model_cache_dir
                                / "memory_write_gate"
                            ),
                            device=(
                                self.settings.memory_write_gate_device
                            ),
                            max_length=(
                                self.settings.memory_write_gate_max_length
                            ),
                            threshold=(
                                self.settings.memory_write_gate_threshold
                            ),
                        )
                        if self.settings.memory_write_gate_enabled
                        else None
                    ),

                    write_gate_batch_size=(
                        self.settings.memory_write_gate_batch_size
                    ),

                    extraction_enabled=(
                        self.settings.memory_extraction_enabled
                    ),

                    extraction_batch_size=(
                        self.settings.memory_extraction_batch_size
                    ),
                )
            )

            # 先把已经到期的active记忆
            # 统一改成retired。
            await (
                memory_service
                .retire_expired_memories()
            )

            # 再使用剩余active记忆重建图与BM25辅助索引。
            await (
                memory_service
                .rebuild_graph_index()
            )
            await memory_service.rebuild_conflict_groups()

            # 动态Worker每次attempt都会重建Graph；路由缓存必须由Conversation
            # Runtime持有，否则Reviewer返工会无谓地再次调用能力选择器。
            general_toolset_route_cache = {}
            web_toolset_route_cache = {}
            code_toolset_route_cache = {}

            worker_graph = create_general_worker(
                role_models["general"],
                summary_model=role_models["general_summary"],
                tools=self.tools,
                progress_every_tool_calls=(
                    self.settings
                    .worker_runtime
                    .progress_every_tool_calls
                ),
                middleware=[
                    PromptInjectionGuardMiddleware(prompt_injection_guard),
                    ToolsetRouterMiddleware(
                        retrieval_models,
                        minimum_score=self.settings.toolset_routing_threshold,
                        selector_model=role_models["capability_selector"],
                        route_cache=general_toolset_route_cache,
                        baseline_tool_names=GENERAL_TOOLSET_BASELINE_TOOL_NAMES,
                    ),
                    DynamicExecutionBudgetMiddleware(
                        enable_worker_finalization=False,
                        finalization_model_rounds=(
                            self.settings.worker_runtime
                            .finalization_model_rounds
                        ),
                        schema_repair_max_rounds=(
                            self.settings.worker_runtime
                            .schema_repair_max_rounds
                        ),
                    ),
                ],
                finalization_model_rounds=(
                    self.settings.worker_runtime.finalization_model_rounds
                ),
                schema_repair_max_rounds=(
                    self.settings.worker_runtime.schema_repair_max_rounds
                ),
                checkpointer=checkpointer,
                store=memory_store,
            )

            async def decide_worker_progress(request):
                leader_result = await run_hard_worker_leader(
                    role_models["worker_leader"],
                    request=request,
                )
                return LeadershipDecisionResult(
                    decision=leader_result.output,
                    model_rounds_used=leader_result.model_rounds_used,
                    used_fallback=leader_result.used_fallback,
                )

            general_worker = WorkerLeadershipBridge(
                worker_graph,
                self.event_store,
                wake_policy=LeadershipWakePolicy(
                    single_worker_reports=(
                        self.settings
                        .worker_runtime
                        .leadership_single_worker_reports
                    ),
                    multi_worker_reports=(
                        self.settings
                        .worker_runtime
                        .leadership_multi_worker_reports
                    ),
                ),
                decision_handler=None,
            )

            planning_general_runtime = GeneralStepRuntime(
                role_models["general"],
                summary_model=role_models["general_summary"],
                event_store=self.event_store,
                tools=self.tools,
                wake_policy=LeadershipWakePolicy(
                    single_worker_reports=(
                        self.settings.worker_runtime
                        .leadership_single_worker_reports
                    ),
                    multi_worker_reports=(
                        self.settings.worker_runtime
                        .leadership_multi_worker_reports
                    ),
                ),
                decision_handler=None,
                progress_every_tool_calls=(
                    self.settings.worker_runtime.progress_every_tool_calls
                ),
                finalization_model_rounds=(
                    self.settings.worker_runtime.finalization_model_rounds
                ),
                schema_repair_max_rounds=(
                    self.settings.worker_runtime.schema_repair_max_rounds
                ),
                middleware_factory=lambda: [
                    PromptInjectionGuardMiddleware(prompt_injection_guard),
                    ToolsetRouterMiddleware(
                        retrieval_models,
                        minimum_score=self.settings.toolset_routing_threshold,
                        selector_model=role_models["capability_selector"],
                        route_cache=general_toolset_route_cache,
                        baseline_tool_names=GENERAL_TOOLSET_BASELINE_TOOL_NAMES,
                    ),
                    DynamicExecutionBudgetMiddleware(
                        enable_worker_finalization=False,
                        finalization_model_rounds=(
                            self.settings.worker_runtime
                            .finalization_model_rounds
                        ),
                        schema_repair_max_rounds=(
                            self.settings.worker_runtime
                            .schema_repair_max_rounds
                        ),
                    ),
                ],
                checkpointer=checkpointer,
                store=memory_store,
            )

            # WEB按Worker生命周期租用浏览器。不同worker_id得到不同的
            # Playwright MCP进程；完成、异常或取消都会退出租约并释放容量。
            web_runtime = WebStepRuntime(
                role_models["web"],
                summary_model=role_models["web_summary"],
                playwright_pool=self.playwright_mcp,
                event_store=self.event_store,
                tools=self.tools,
                wake_policy=LeadershipWakePolicy(
                    single_worker_reports=(
                        self.settings.worker_runtime
                        .leadership_single_worker_reports
                    ),
                    multi_worker_reports=(
                        self.settings.worker_runtime
                        .leadership_multi_worker_reports
                    ),
                ),
                decision_handler=decide_worker_progress,
                progress_every_tool_calls=(
                    self.settings.worker_runtime.progress_every_tool_calls
                ),
                finalization_model_rounds=(
                    self.settings.worker_runtime.finalization_model_rounds
                ),
                schema_repair_max_rounds=(
                    self.settings.worker_runtime.schema_repair_max_rounds
                ),
                download_max_file_mib=(
                    self.settings.worker_runtime.web_download_max_file_mib
                ),
                middleware_factory=lambda: [
                    PromptInjectionGuardMiddleware(prompt_injection_guard),
                    ToolsetRouterMiddleware(
                        retrieval_models,
                        minimum_score=self.settings.toolset_routing_threshold,
                        selector_model=role_models["capability_selector"],
                        route_cache=web_toolset_route_cache,
                        baseline_tool_names=WEB_TOOLSET_BASELINE_TOOL_NAMES,
                        select_web_progress_checkpoint=True,
                    ),
                    DynamicExecutionBudgetMiddleware(
                        enable_worker_finalization=True,
                        finalization_model_rounds=(
                            self.settings.worker_runtime
                            .finalization_model_rounds
                        ),
                        schema_repair_max_rounds=(
                            self.settings.worker_runtime
                            .schema_repair_max_rounds
                        ),
                    ),
                ],
                checkpointer=checkpointer,
                store=memory_store,
            )

            # CODE is a lazy runtime: Docker is not touched during ordinary
            # chat startup.  A CODE PlanStep creates one serial Worker/
            # Reviewer pair, exports its candidate into the private archive,
            # and lets the Reviewer publish only approved files into the
            # run-local integration tree. Final user delivery is a separate
            # promotion boundary.
            code_runtime = CodeStepRuntime(
                role_models["code"],
                summary_model=role_models["code_summary"],
                reviewer_model=role_models["code_reviewer"],
                reviewer_summary_model=role_models["code_reviewer_summary"],
                sandbox_manager=CodeSandboxManager(
                    CodeSandboxPolicy.from_settings(
                        self.settings.code_sandbox
                    )
                ),
                source_root=WORKSPACE_ROOT,
                target_root=WORKSPACE_ROOT,
                archive_root=AGENT_DATA_ROOT / "code_attempts",
                # The CODE pair receives Docker-backed filesystem and shell
                # tools from Deep Agents.  Host-writing legacy tools are not
                # forwarded across the sandbox boundary.
                tools=(),
                max_writers=(
                    self.settings.runtime_concurrency.code_max_writers
                ),
                max_repair_rounds=(
                    self.settings.worker_runtime.code_review_max_repair_rounds
                ),
                archive_retention_minutes=(
                    self.settings.worker_runtime.workspace_retention_minutes
                ),
                progress_every_tool_calls=(
                    self.settings.worker_runtime.progress_every_tool_calls
                ),
                schema_repair_max_rounds=(
                    self.settings.worker_runtime.schema_repair_max_rounds
                ),
                middleware_factory=lambda: [
                    PromptInjectionGuardMiddleware(prompt_injection_guard),
                    ToolsetRouterMiddleware(
                        retrieval_models,
                        minimum_score=self.settings.toolset_routing_threshold,
                        selector_model=role_models["capability_selector"],
                        route_cache=code_toolset_route_cache,
                        baseline_tool_names=CODE_TOOLSET_BASELINE_TOOL_NAMES,
                    ),
                    DynamicExecutionBudgetMiddleware(
                        schema_repair_max_rounds=(
                            self.settings.worker_runtime
                            .schema_repair_max_rounds
                        ),
                    ),
                ],
                checkpointer=checkpointer,
                store=memory_store,
            )
            terminal_runs = await self._recover_code_startup(code_runtime)

            try:
                recovered_promotions = await (
                    self.workspace_promotions.reconcile_promoting()
                )
                if recovered_promotions:
                    logger.info(
                        "Workspace交付启动核对完成 | outcomes=%s",
                        recovered_promotions,
                    )
            except Exception:
                logger.exception(
                    "Workspace交付恢复遇到冲突；记录已保留，未猜测覆盖用户文件"
                )

            protected_promotions = await self.event_store.list_workspace_promotions(
                statuses=(
                    PromotionStatus.AWAITING_APPROVAL,
                    PromotionStatus.APPROVED,
                    PromotionStatus.PROMOTING,
                )
            )
            cleanup_outcomes = await asyncio.to_thread(
                cleanup_expired_run_workspaces,
                terminal_runs={
                    run.event_id: run.status_changed_at for run in terminal_runs
                },
                protected_run_ids={item.run_id for item in protected_promotions},
                retention_minutes=(
                    self.settings.worker_runtime.workspace_retention_minutes
                ),
            )
            cleaned = [
                item for item in cleanup_outcomes if item.get("status") == "CLEANED"
            ]
            if cleaned:
                logger.info("Run Workspace过期清理完成 | outcomes=%s", cleaned)
            archive_cleanup = await asyncio.to_thread(
                code_runtime.cleanup_expired_archives,
                protected_run_ids={item.run_id for item in protected_promotions},
            )
            cleaned_archives = [
                item for item in archive_cleanup if item.get("status") == "CLEANED"
            ]
            if cleaned_archives:
                logger.info(
                    "CODE过期Docker导出清理完成 | outcomes=%s",
                    cleaned_archives,
                )

            toolset_catalog = (
                DEFAULT_TOOLSET_REGISTRY
                .build_router_metadata(
                    self.tools,
                )
            )

            planning_graph = (
                build_planning_graph(
                    role_models=role_models,
                    reporter_output_limits={
                        "reporter": self.settings.role_models["reporter"].max_tokens
                    },
                    simple_model=(
                        model
                    ),

                    hard_model=(
                        hard_model
                    ),

                    worker_registry=(
                        WorkerAgentRegistry.with_specialized_workers(
                            planning_general_runtime,
                            web_runtime,
                            code_runtime,
                        )
                    ),

                    worker_group_coordinator=WorkerGroupCoordinator(
                        self.event_store,
                        workspace_retention_minutes=(
                            self.settings
                            .worker_runtime
                            .workspace_retention_minutes
                        ),
                    ),

                    planning=(
                        self.settings.planning
                    ),

                    # Step Reporter使用这个值计算：
                    #
                    # - 模型最大输出；
                    # - Reporter内部总预算；
                    # - 为Thinking和JSON输出预留的空间；
                    # - 最终允许注入多少执行轨迹。
                    model_output_max_tokens=(
                        self.settings
                        .cloud_llm_max_tokens
                    ),

                    web_max_parallelism=(
                        self.settings
                        .runtime_concurrency
                        .web_search_max_parallelism
                    ),
                    review_evidence_mode=(
                        self.settings.worker_runtime.review_evidence_mode
                    ),

                    # 外层Planning Graph与子Agent共用同一个
                    # SQLite checkpointer，但通过不同thread_id隔离快照。
                    checkpointer=checkpointer,
                )
            )

        except Exception:
            configure_schedule_service(None)
            configure_host_file_judge(None)
            await self.schedule_service.stop()
            await self.amap_mcp.stop()
            await self.email_mcp.stop()
            self.tools = list(self._base_tools)
            await (
                self.playwright_mcp
                .stop()
            )
            if memory_store is not None:
                await memory_context.__aexit__(
                    None,
                    None,
                    None,
                )

            if retrieval_models is not None:
                await (
                    retrieval_models
                    .aclose()
                )

            if checkpointer is not None:
                await (
                    checkpoint_context
                    .__aexit__(
                        None,
                        None,
                        None,
                    )
                )

            await self.event_store.close()

            raise

        self._checkpointer_context = (
            checkpoint_context
        )

        self._checkpointer = checkpointer

        self._memory_store_context = (
            memory_context
        )
        self.memory_service = (
            memory_service
        )
        self._memory_store = memory_store

        self.retrieval_models = (
            retrieval_models
        )
        self.retrieval_hub = RetrievalHub(memory_service, retrieval_models)
        with trace_span("RAG / Startup Sync") as rag_sync_span:
            try:
                set_span_output(rag_sync_span, await self.retrieval_hub.sync("owner"))
            except Exception as error:
                set_span_output(rag_sync_span, {"status": "failed", "error": str(error)})
        self.toolset_router = toolset_router

        self.model = model
        self.summary_model = summary_model
        self.role_models = role_models
        self.hard_model = hard_model
        self.general_worker = general_worker
        self.web_worker = web_runtime
        self.code_runtime = code_runtime

        self.planning_graph = (
            planning_graph
        )

        self.toolset_catalog = (
            toolset_catalog
        )

    async def new_conversation(
        self,
        channel: str,
        external_chat_id: int | str,
        title: str | None = None,
    ) -> ConversationInfo:
        """创建新Conversation，并使它成为最近使用的对话。"""

        agent = self._require_general_worker()
        await (
            self.playwright_mcp
            .reset_page()
        )
        conversation_id = (
            f"conv_{uuid4().hex}"
        )

        thread_id = (
            f"thread_{uuid4().hex}"
        )

        channel_key = (
            self._build_channel_key(
                channel,
                external_chat_id,
            )
        )

        now = self._now()

        resolved_title = (
            self._normalize_title(
                title
            )
            if title
            else "新对话"
        )

        title_generated = (
            bool(title)
        )

        config = self._build_config(
            thread_id
        )

        await agent.aupdate_state(
            config,
            {
                "messages": [],

                "conversation_id": (
                    conversation_id
                ),

                "conversation_title": (
                    resolved_title
                ),

                "channel_key": (
                    channel_key
                ),

                "created_at": now,

                "last_active_at": now,

                "title_generated": (
                    title_generated
                ),
            },

            as_node="__start__",
        )

        initialize_conversation_workspace(conversation_id)

        return ConversationInfo(
            conversation_id=(
                conversation_id
            ),

            thread_id=thread_id,

            short_id=(
                self._build_short_id(
                    conversation_id
                )
            ),

            title=resolved_title,

            channel_key=channel_key,

            created_at=now,

            last_active_at=now,

            title_generated=(
                title_generated
            ),
        )

    async def list_conversations(
        self,
        channel: str,
        external_chat_id: int | str,
    ) -> list[ConversationInfo]:
        """从LangGraph Checkpoint中读取Conversation列表。"""

        checkpointer = (
            self._require_checkpointer()
        )

        channel_key = (
            self._build_channel_key(
                channel,
                external_chat_id,
            )
        )

        conversations: list[
            ConversationInfo
        ] = []

        seen_threads: set[
            str
        ] = set()

        async for item in (
            checkpointer.alist(
                None
            )
        ):
            configurable = (
                item.config.get(
                    "configurable",
                    {},
                )
            )

            thread_id = configurable.get(
                "thread_id"
            )

            if not isinstance(
                thread_id,
                str,
            ):
                continue

            # alist按照新到旧返回。
            # 一个Thread只读取最新的Checkpoint。
            if thread_id in seen_threads:
                continue

            seen_threads.add(
                thread_id
            )

            values = (
                item.checkpoint.get(
                    "channel_values",
                    {},
                )
            )

            if (
                values.get(
                    "channel_key"
                )
                != channel_key
            ):
                continue

            conversation_id = (
                values.get(
                    "conversation_id"
                )
            )

            if not isinstance(
                conversation_id,
                str,
            ):
                # 忽略接入Conversation之前产生的旧Thread。
                continue

            created_at = values.get(
                "created_at"
            )

            if not isinstance(
                created_at,
                str,
            ):
                created_at = (
                    item.checkpoint.get(
                        "ts",
                        "",
                    )
                )

            last_active_at = (
                values.get(
                    "last_active_at"
                )
            )

            if not isinstance(
                last_active_at,
                str,
            ):
                last_active_at = (
                    created_at
                )

            title = values.get(
                "conversation_title"
            )

            if not isinstance(
                title,
                str,
            ):
                title = "新对话"

            conversations.append(
                ConversationInfo(
                    conversation_id=(
                        conversation_id
                    ),

                    thread_id=(
                        thread_id
                    ),

                    short_id=(
                        self._build_short_id(
                            conversation_id
                        )
                    ),

                    title=title,

                    channel_key=(
                        channel_key
                    ),

                    created_at=(
                        created_at
                    ),

                    last_active_at=(
                        last_active_at
                    ),

                    title_generated=bool(
                        values.get(
                            "title_generated",
                            False,
                        )
                    ),
                )
            )

        conversations.sort(
            key=lambda item: (
                item.last_active_at
            ),
            reverse=True,
        )

        return conversations

    async def get_active_conversation(
            self,
            channel: str,
            external_chat_id: int | str,
    ) -> ConversationInfo:
        """快速读取最近使用的Conversation；不存在时自动创建。

        不再调用checkpointer.alist(None)。

        alist(None)会按照checkpoint_id对整个Checkpoint表排序，
        即使Python只读取第一条，也可能在第一条产生之前长时间阻塞。

        当前项目使用SQLite Checkpointer，因此这里先按照SQLite rowid
        读取最近写入的少量根Checkpoint，再按thread_id读取该Thread
        的最新完整状态。
        """

        checkpointer = (
            self._require_checkpointer()
        )

        channel_key = (
            self._build_channel_key(
                channel,
                external_chat_id,
            )
        )

        # 确保LangGraph Checkpoint表已经创建。
        await checkpointer.setup()

        # 这里只读取thread_id，不读取和反序列化庞大的Checkpoint Blob。
        #
        # rowid代表SQLite中的写入顺序。
        # 从最后写入的根Checkpoint开始检查，
        # 可以快速找到最近活跃的Conversation。
        async with checkpointer.lock:
            async with checkpointer.conn.execute(
                    """
                    SELECT thread_id
                    FROM checkpoints
                    WHERE checkpoint_ns = ''
                    ORDER BY rowid DESC LIMIT 200
                    """
            ) as cursor:
                rows = await cursor.fetchall()

        seen_thread_ids: set[
            str
        ] = set()

        for row in rows:
            if not row:
                continue

            thread_id = row[0]

            if not isinstance(
                    thread_id,
                    str,
            ):
                continue

            if thread_id in seen_thread_ids:
                continue

            seen_thread_ids.add(
                thread_id
            )

            # 已经知道thread_id后，
            # aget_tuple会使用主键前缀进行定向查询，
            # 不再扫描所有Conversation。
            item = await checkpointer.aget_tuple(
                self._build_config(
                    thread_id
                )
            )

            if item is None:
                continue

            values = (
                item.checkpoint.get(
                    "channel_values",
                    {},
                )
            )

            if not isinstance(
                    values,
                    dict,
            ):
                continue

            if (
                    values.get(
                        "channel_key"
                    )
                    != channel_key
            ):
                continue

            conversation_id = (
                values.get(
                    "conversation_id"
                )
            )

            if not isinstance(
                    conversation_id,
                    str,
            ):
                # 忽略接入Conversation功能之前
                # 产生的旧Checkpoint。
                continue

            created_at = (
                values.get(
                    "created_at"
                )
            )

            if not isinstance(
                    created_at,
                    str,
            ):
                created_at = (
                    item.checkpoint.get(
                        "ts",
                        "",
                    )
                )

            last_active_at = (
                values.get(
                    "last_active_at"
                )
            )

            if not isinstance(
                    last_active_at,
                    str,
            ):
                last_active_at = (
                    created_at
                )

            title = (
                values.get(
                    "conversation_title"
                )
            )

            if not isinstance(
                    title,
                    str,
            ):
                title = "新对话"

            return ConversationInfo(
                conversation_id=(
                    conversation_id
                ),

                thread_id=(
                    thread_id
                ),

                short_id=(
                    self._build_short_id(
                        conversation_id
                    )
                ),

                title=title,

                channel_key=(
                    channel_key
                ),

                created_at=(
                    created_at
                ),

                last_active_at=(
                    last_active_at
                ),

                title_generated=bool(
                    values.get(
                        "title_generated",
                        False,
                    )
                ),
            )

        # 没有找到属于当前入口的Conversation时，
        # 创建新的Conversation。
        return await self.new_conversation(
            channel=channel,

            external_chat_id=(
                external_chat_id
            ),
        )
    async def switch_conversation(
        self,
        channel: str,
        external_chat_id: int | str,
        selector: str,
    ) -> ConversationInfo | None:
        """按列表编号或短ID切换Conversation。"""

        agent = self._require_general_worker()
        await (
            self.playwright_mcp
            .reset_page()
        )
        conversations = await (
            self.list_conversations(
                channel=channel,

                external_chat_id=(
                    external_chat_id
                ),
            )
        )

        selected: (
            ConversationInfo
            | None
        ) = None

        normalized_selector = (
            selector
            .strip()
            .lower()
        )

        if normalized_selector.isdigit():
            position = int(
                normalized_selector
            )

            if (
                1
                <= position
                <= len(conversations)
            ):
                selected = (
                    conversations[
                        position - 1
                    ]
                )

        else:
            for conversation in conversations:
                if normalized_selector in {
                    conversation.short_id.lower(),

                    conversation.conversation_id.lower(),

                    conversation.thread_id.lower(),
                }:
                    selected = (
                        conversation
                    )
                    break

        if selected is None:
            return None

        now = self._now()

        await agent.aupdate_state(
            self._build_config(
                selected.thread_id
            ),

            {
                "last_active_at": (
                    now
                ),
            },

            # 当前Thread已经存在。
            # 这是从ConversationRuntime外部
            # 写入Conversation元数据，
            # 明确归因于create_agent的model节点。
            as_node="model",
        )

        return ConversationInfo(
            conversation_id=(
                selected.conversation_id
            ),

            thread_id=(
                selected.thread_id
            ),

            short_id=(
                selected.short_id
            ),

            title=selected.title,

            channel_key=(
                selected.channel_key
            ),

            created_at=(
                selected.created_at
            ),

            last_active_at=now,

            title_generated=(
                selected.title_generated
            ),
        )

    async def get_conversation_by_id(
        self,
        *,
        channel: str,
        external_chat_id: int | str,
        conversation_id: str,
    ) -> ConversationInfo | None:
        """Resolve the conversation captured when an Event was enqueued."""

        normalized_id = str(conversation_id).strip()
        if not normalized_id:
            raise ValueError("conversation_id不能为空。")
        conversations = await self.list_conversations(
            channel=channel,
            external_chat_id=external_chat_id,
        )
        for conversation in conversations:
            if conversation.conversation_id == normalized_id:
                return conversation
        return None

    async def _prepare_planning_context(
        self,
        *,
        agent,
        hard_model,
        conversation: ConversationInfo,
        state_values: dict[
            str,
            Any,
        ],
        user_request: str,
        memory_context: str,
        replacement_context: PlanningReplacementContext | None = None,
    ) -> PlanningContextPack:
        """近期原文与按 token 门槛更新的较早对话 JSON 摘要。"""
        planning = self.settings.planning
        dialogue_turns = _extract_dialogue_turns(list(state_values.get("messages", [])))
        keep_turns = planning.hard_recent_dialogue_turns
        older_end = max(0, len(dialogue_turns) - keep_turns)
        compaction_enabled = getattr(planning, "conversation_compaction_enabled", False)
        checkpoint = read_summary_checkpoint(
            state_values.get("conversation_summary"), len(dialogue_turns)
        ) if compaction_enabled else None
        covered = checkpoint.covered_turns if checkpoint is not None else 0
        pending_turns = dialogue_turns[covered:older_end]
        threshold = planning.conversation_summary_trigger_tokens
        if compaction_enabled and pending_turns and dialogue_token_estimate(pending_turns) >= threshold:
            try:
                summary = await compress_dialogue(
                    self.summary_model,
                    previous=checkpoint.summary if checkpoint is not None else None,
                    turns=pending_turns,
                    first_turn=covered + 1,
                    max_chars=planning.conversation_summary_max_chars,
                )
            except Exception:
                # Keep both the previous checkpoint and ALL pending originals.
                # A failed/invalid summary never advances coverage or clips text.
                logger.warning("Conversation JSON摘要失败；保留旧摘要与未摘要原文。")
            else:
                checkpoint = SummaryCheckpoint(covered_turns=older_end, summary=summary)
                await agent.aupdate_state(
                    self._build_config(conversation.thread_id),
                    {"conversation_summary": checkpoint.model_dump_json(),
                     "conversation_summary_turn_count": older_end},
                    as_node="model",
                )
                covered = older_end
        # A larger retention setting can expose turns covered by a prior summary.
        # In that case, still restore the requested recent originals exactly.
        raw_start = min(covered, older_end)
        recent_dialogue = [item for turn in dialogue_turns[raw_start:] for item in turn]
        conversation_summary = checkpoint.model_dump_json() if checkpoint is not None else ""

        selected_toolset_catalog = self.toolset_catalog
        toolset_router = getattr(self, "toolset_router", None)
        if toolset_router is not None and self.toolset_catalog:
            routing_context = _format_dialogue_messages(recent_dialogue)
            routing_text = f"当前请求：{user_request}"
            if routing_context:
                routing_text += f"\n相关最近对话：\n{routing_context}"
            route_decision = await toolset_router.route(
                task_text=routing_text,
                available_tools=self.tools,
                # 是否需要工具由Scheduler在每个PlanStep中声明；这里仅在
                # 真实工具组之间压缩提供给Scheduler的能力目录。
                allow_no_tool=False,
            )
            if route_decision is not None:
                selected_names = set(route_decision.selected_toolset_names)
                if selected_names == {NO_TOOL_ROUTE}:
                    selected_toolset_catalog = []
                else:
                    filtered_catalog = [
                        item
                        for item in self.toolset_catalog
                        if item.get("name") in selected_names
                    ]
                    # Registry和实时目录短暂不一致时也不能静默隐藏全部能力。
                    selected_toolset_catalog = filtered_catalog or self.toolset_catalog

        return PlanningContextPack(
            current_time=(
                get_current_time(
                    "Asia/Shanghai"
                )
            ),

            user_request=(
                user_request
            ),

            # Originals already occur in recent_dialogue. Do not reinject
            # archived users or internal execution summaries from old checkpoints.
            user_instruction_history=[],
            conversation_summary=conversation_summary,
            previous_run_summary="",

            recent_dialogue=(
                recent_dialogue
            ),

            memory_context=(
                memory_context
            ),

            replacement_context=(
                replacement_context
            ),

            toolset_catalog=(
                selected_toolset_catalog
            ),
        )

    async def _prepare_replacement_context(
        self,
        *,
        conversation: ConversationInfo,
        replacement_event_id: str,
        target_event_id: str,
        replacement_instruction: str,
    ) -> PlanningReplacementContext:
        """Read only bounded, committed state from the superseded generation."""

        target_event = await self.event_store.require_event(target_event_id)
        if target_event.conversation_id != conversation.conversation_id:
            raise ValueError("REPLACE target belongs to another conversation")

        old_thread_id = build_planning_thread_id_from_parts(
            conversation_id=conversation.thread_id,
            event_id=target_event_id,
        )
        old_snapshot = await self._require_planning_graph().aget_state(
            {"configurable": {"thread_id": old_thread_id}}
        )
        values = getattr(old_snapshot, "values", {}) or {}
        if not isinstance(values, dict):
            values = {}

        raw_old_context = values.get("context")
        old_context = None
        if raw_old_context is not None:
            try:
                old_context = PlanningContextPack.model_validate(raw_old_context)
            except Exception:
                logger.exception(
                    "旧Planning Context无法读取，REPLACE退回Event原始请求 | "
                    "target_event_id=%s",
                    target_event_id,
                )

        def json_record(value: Any) -> dict[str, Any] | None:
            if hasattr(value, "model_dump"):
                dumped = value.model_dump(mode="json")
                return dict(dumped) if isinstance(dumped, dict) else None
            return dict(value) if isinstance(value, dict) else None

        completed_reports = [
            record
            for item in list(values.get("completed_step_reports", []))[-12:]
            if (record := json_record(item)) is not None
        ]
        handoff_receipts = [
            record
            for item in list(values.get("handoff_publication_receipts", []))[-24:]
            if (record := json_record(item)) is not None
        ]

        layout = initialize_run_workspace(target_event_id)
        supersession_receipt = read_run_supersession_receipt(
            layout=layout,
            replacement_event_id=replacement_event_id,
        ) or {}
        replacement_layout = initialize_run_workspace(replacement_event_id)
        inheritance_receipt = read_replacement_inheritance_receipt(
            layout=replacement_layout,
        ) or {}
        integration_status = None
        if (
            (replacement_layout.integration_root / ".git").is_dir()
            and (
                replacement_layout.receipts_root / "integration-baseline.json"
            ).is_file()
        ):
            integration_status = await asyncio.to_thread(
                lambda: IntegrationRepository(
                    run_id=replacement_event_id,
                    root=replacement_layout.integration_root,
                    receipts_root=replacement_layout.receipts_root,
                ).status().model_dump(mode="json")
            )

        return PlanningReplacementContext(
            target_event_id=target_event_id,
            replacement_event_id=replacement_event_id,
            original_user_request=(
                old_context.user_request
                if old_context is not None
                else target_event.payload_text
            ),
            replacement_instruction=replacement_instruction,
            previous_plan_objective=str(
                values.get("plan_objective") or ""
            ).strip(),
            completed_step_reports=completed_reports,
            handoff_publication_receipts=handoff_receipts,
            accepted_integration_status=integration_status,
            previous_usage={
                "model_rounds_used": int(values.get("model_rounds_used", 0) or 0),
                "tool_calls_used": int(values.get("tool_calls_used", 0) or 0),
            },
            supersession_receipt=supersession_receipt,
            workspace_inheritance=inheritance_receipt,
        )

    async def ask(
            self,
            *,
            user_text: str,
            channel: str,
            external_chat_id: int | str,
            reply_target_id: int | str | None = None,
            progress_callback: ProgressCallback,
            event_id: str | None = None,
            target_conversation_id: str | None = None,
            pause_control: EventPauseControl | None = None,
            bypass_conversation_lock: bool = False,
            resume_from_checkpoint: bool = False,
            replacement_target_event_id: str | None = None,
    ) -> str:
        """在当前Conversation中完成一轮用户请求。"""

        normalized_user_text = (
            user_text.strip()
        )

        if not normalized_user_text:
            raise ValueError(
                "user_text不能为空。"
            )

        agent = self._require_general_worker()

        model = self._require_model()

        hard_model = (
            self._require_hard_model()
        )

        planning_graph = (
            self._require_planning_graph()
        )

        memory_service = (
            self._require_memory_service()
        )

        if target_conversation_id is None:
            conversation = await self.get_active_conversation(
                channel=channel,
                external_chat_id=external_chat_id,
            )
        else:
            conversation = await self.get_conversation_by_id(
                channel=channel,
                external_chat_id=external_chat_id,
                conversation_id=target_conversation_id,
            )
            if conversation is None:
                raise RuntimeError(
                    "Event绑定的Conversation不存在："
                    f"{target_conversation_id}"
                )

        resolved_event_id = (
            event_id.strip()
            if event_id is not None
            else f"evt_direct_{uuid4().hex}"
        )
        if not resolved_event_id:
            raise ValueError("event_id不能为空。")

        planning_thread_id = (
            build_planning_thread_id_from_parts(
                conversation_id=conversation.thread_id,
                event_id=resolved_event_id,
            )
        )

        planning_config = {
            "configurable": {
                "thread_id": planning_thread_id,
                "progress_callback": progress_callback,
                "event_pause_control": pause_control,
                "resume_from_checkpoint": resume_from_checkpoint,
            },
        }
        resume_checkpoint_available = False
        checkpoint_state = None
        trace_query_text = normalized_user_text
        if resume_from_checkpoint:
            checkpoint_state = await planning_graph.aget_state(planning_config)
            resume_checkpoint_available = bool(
                getattr(checkpoint_state, "created_at", None)
                or getattr(checkpoint_state, "values", None)
                or getattr(checkpoint_state, "next", ())
            )
            if not resume_checkpoint_available:
                logger.warning(
                    "Event被标记为续跑，但Planning checkpoint不存在；"
                    "本轮退回首次执行 | event_id=%s | thread_id=%s",
                    resolved_event_id,
                    planning_thread_id,
                )
            else:
                checkpoint_values = getattr(checkpoint_state, "values", None) or {}
                checkpoint_context = checkpoint_values.get("context")
                if isinstance(checkpoint_context, dict):
                    checkpoint_request = checkpoint_context.get("user_request")
                else:
                    checkpoint_request = getattr(checkpoint_context, "user_request", None)
                if str(checkpoint_request or "").strip():
                    trace_query_text = str(checkpoint_request).strip()

        thread_lock = (
            self._thread_locks
            .setdefault(
                conversation.thread_id,

                asyncio.Lock(),
            )
        )

        lock_context = nullcontext() if bypass_conversation_lock else thread_lock
        async with lock_context:
            now = self._now()

            # Session上下文必须放在根Span外面。
            # conversation_turn及其所有子Span
            # 会继承相同的session.id。
            with trace_context(
                    session_id=(
                            conversation.thread_id
                    ),

                    metadata={
                        "conversation_id": (
                                conversation
                                        .conversation_id
                        ),

                        "thread_id": (
                                conversation
                                        .thread_id
                        ),

                        "conversation_title": (
                                conversation.title
                        ),

                        "channel": (
                                channel
                        ),
                    },

                    tags=[
                        "conversation-turn",
                        channel,
                    ],
            ):
                turn_overview_result: dict[str, Any] = {}
                with skill_selector_scope(self.role_models.get("skill_selector")), retrieval_scope(getattr(self, "retrieval_hub", None), "owner", resolved_event_id), trace_span(
                        (
                            run_name("Conversation")
                            if channel == "appworld"
                            else conversation_trace_name(
                                trace_query_text,
                                resumed=resume_checkpoint_available,
                            )
                        ),

                        kind="chain",

                        input_value={
                            "conversation_id": (
                                    conversation
                                            .conversation_id
                            ),

                            "thread_id": (
                                    conversation
                                            .thread_id
                            ),

                            "conversation_title": (
                                    conversation.title
                            ),

                            "channel": (
                                    channel
                            ),

                            "user_message": (
                                    normalized_user_text
                            ),
                        },

                        attributes={
                            "conversation.id": (
                                    conversation
                                            .conversation_id
                            ),

                            "conversation.thread_id": (
                                    conversation
                                            .thread_id
                            ),

                            "conversation.channel": (
                                    channel
                            ),

                            "conversation.title_generated": (
                                    conversation
                                            .title_generated
                            ),

                            "conversation.user_message_chars": (
                                    len(
                                        normalized_user_text
                                    )
                            ),
                        },
                ) as turn_span, conversation_overview(
                        turn_span,
                        {"instruction": normalized_user_text, "channel": channel,
                         "title": conversation.title},
                        turn_overview_result,
                        enabled=channel != "appworld",
                ):

                    memories: list[
                        RetrievedMemory
                    ] = []

                    memory_context = ""

                    memory_retrieval_status = (
                        "success"
                    )

                    try:
                        with trace_span(
                                "memory_retrieval",

                                kind="retriever",

                                input_value={
                                    "query": (
                                            normalized_user_text
                                    ),

                                    "dense_limit": (
                                            memory_service
                                                    .dense_limit
                                    ),

                                    "final_limit": (
                                            memory_service
                                                    .final_limit
                                    ),

                                    "graph_hops": (
                                            memory_service
                                                    .graph_hops
                                    ),
                                },

                                attributes={
                                    "retrieval.query_chars": (
                                            len(
                                                normalized_user_text
                                            )
                                    ),

                                    "retrieval.final_limit": (
                                            memory_service
                                                    .final_limit
                                    ),

                                    "retrieval.graph_hops": (
                                            memory_service
                                                    .graph_hops
                                    ),
                                },
                        ) as retrieval_span:

                            memories = (
                                []
                                if resume_checkpoint_available
                                else await memory_service.retrieve_for_turn(
                                    normalized_user_text
                                )
                            )

                            memory_context = (
                                memory_service
                                .format_context(
                                    memories
                                )
                            )

                            selected_memory_items = [
                                _memory_to_trace_item(
                                    memory
                                )

                                for memory
                                in memories
                            ]

                            set_span_attributes(
                                retrieval_span,

                                **{
                                    "retrieval.selected_count": (
                                        len(
                                            memories
                                        )
                                    ),

                                    "memory.context_chars": (
                                        len(
                                            memory_context
                                        )
                                    ),
                                },
                            )

                            set_span_output(
                                retrieval_span,

                                {
                                    "selected_count": (
                                        len(
                                            memories
                                        )
                                    ),

                                    "selected_memories": (
                                        selected_memory_items
                                    ),

                                    "rendered_memory_context": (
                                        memory_context
                                    ),

                                    "graph_stats": (
                                        memory_service
                                        .graph_index
                                        .stats()
                                    ),
                                },
                            )

                    except Exception:
                        memory_retrieval_status = (
                            "failed"
                        )

                        memories = []

                        memory_context = ""

                        logger.exception(
                            "长期记忆召回失败，"
                            "本轮将不注入长期记忆"
                        )

                    if resume_checkpoint_available:
                        # Planning Graph已经保存了首次输入和Rolling Context。
                        # 恢复时不重新召回记忆、不重新生成摘要，也不再次
                        # 让Hard Supervisor准备同一份规划上下文。
                        planning_context = None
                    else:
                        # Hard Supervisor不在create_agent内部，因此显式读取
                        # 原Conversation中的历史，准备Rolling Summary和最近对话。
                        conversation_state = await agent.aget_state(
                            self._build_config(conversation.thread_id)
                        )
                        state_values = (
                            getattr(conversation_state, "values", {}) or {}
                        )
                        replacement_context = None
                        if replacement_target_event_id is not None:
                            replacement_context = await self._prepare_replacement_context(
                                conversation=conversation,
                                replacement_event_id=resolved_event_id,
                                target_event_id=replacement_target_event_id,
                                replacement_instruction=normalized_user_text,
                            )
                        planning_context = await self._prepare_planning_context(
                            agent=agent,
                            hard_model=hard_model,
                            conversation=conversation,
                            state_values=state_values,
                            user_request=normalized_user_text,
                            memory_context=memory_context,
                            replacement_context=replacement_context,
                        )

                    planning_status = "success"

                    planning_result: dict[
                        str,
                        Any,
                    ] = {}

                    planning_input = None if resume_checkpoint_available else {
                                    "context": (
                                        planning_context
                                    ),

                                    "event_id": (
                                        resolved_event_id
                                    ),

                                    "conversation_thread_id": (
                                        conversation
                                        .thread_id
                                    ),

                                    "planning_run_id": (
                                        resolved_event_id
                                    ),

                                    "conversation_workspace_root": str(
                                        initialize_conversation_workspace(
                                            conversation.conversation_id
                                        )
                                    ),

                                    "model_rounds_used": 0,

                                    "tool_calls_used": 0,

                                    "completed_step_reports": [],

                                    "step_user_interactions": [],

                                    "remaining_steps": [],

                                    "replan_history": [],

                                    "replans_used": 0,

                                    "retry_current_step": False,
                                }
                    with trace_span("🎛️ Scheduler", kind="agent", input_value={"event_id": resolved_event_id}) as scheduler_span:
                        if planning_context is not None:
                            # Scheduler plans outcomes; document retrieval belongs to workers.
                            planning_context.rag_context = ""
                        try:
                            with ask_user_runtime_scope(
                                event_id=resolved_event_id,
                                chat_id=str(
                                    reply_target_id
                                    if reply_target_id is not None
                                    else external_chat_id
                                ),
                            ):
                                if pause_control is None:
                                    planning_result = await planning_graph.ainvoke(
                                        planning_input,
                                        config=planning_config,
                                        # 用户可见结果返回前，确保本轮所有节点状态
                                        # 已经同步写入SQLite checkpoint。
                                        durability="sync",
                                    )
                                else:
                                    async for planning_snapshot in planning_graph.astream(
                                        planning_input,
                                        config=planning_config,
                                        stream_mode="values",
                                        durability="sync",
                                    ):
                                        if isinstance(planning_snapshot, dict):
                                            planning_result = planning_snapshot
                                        # astream只会在一个完整节点提交之后产出
                                        # values；因此这里不会切断正在执行的模型、
                                        # 工具或Reviewer。
                                        await pause_control.pause_point()
                            if not planning_result and resume_checkpoint_available:
                                # 如果进程恰好在最后一个图节点已提交、但Event
                                # 状态尚未来得及落盘时退出，续跑不会再产生新的
                                # stream chunk。此时直接读取最终checkpoint即可，
                                # 不能把已经完成的任务误报成空回答。
                                completed_state = await planning_graph.aget_state(
                                    planning_config
                                )
                                completed_values = (
                                    getattr(completed_state, "values", {}) or {}
                                )
                                if isinstance(completed_values, dict):
                                    planning_result = dict(completed_values)
                        except Exception:
                            # 规划模块中的预期模型或Schema错误
                            # 已经在各自节点中降级。
                            # 到这里通常表示意外的编排或程序错误。
                            # 为了飞书演示不直接中断，返回稳定文字，
                            # 同时在Terminal和Phoenix保留完整异常。
                            planning_status = (
                                "failed_after_exception"
                            )

                            logger.exception(
                                "Planning Graph执行失败，"
                                "本轮使用安全错误回复"
                            )

                            reply = (
                                "本轮任务执行过程中出现了异常，"
                                "详细错误已经记录。"
                            )

                        else:
                            reply = str(
                                planning_result.get(
                                    "final_answer",
                                    "",
                                )
                            ).strip()

                            if not reply:
                                planning_status = (
                                    "empty_final_answer"
                                )

                                reply = (
                                    "本轮任务已经结束，"
                                    "但没有生成可发送的最终回答。"
                                )
                        set_span_output(scheduler_span, {"status": planning_result.get("final_status"), "answer": planning_result.get("final_answer"), "stop_reason": planning_result.get("overall_stop_reason")})
                        set_span_attributes(scheduler_span, **{"business.status": planning_result.get("final_status") or planning_status})

                    try:
                        promotion = await self.workspace_promotions.prepare_from_run(
                            layout=initialize_run_workspace(resolved_event_id),
                            conversation_id=conversation.conversation_id,
                        )
                        if promotion is not None:
                            promotion = await (
                                self.workspace_promotions.apply_auto_policy(promotion)
                            )
                            if promotion.status is PromotionStatus.AWAITING_APPROVAL:
                                reply = reply + "\n\n" + format_promotion_summary(promotion)
                            elif promotion.status is PromotionStatus.DELIVERED:
                                reply = (
                                    reply
                                    + "\n\n已自动交付到 Conversation Workspace。\n"
                                    + f"目录：{promotion.target_root}\n"
                                    + f"Delivery commit：{promotion.delivered_commit}"
                                )
                            elif promotion.status is PromotionStatus.FAILED:
                                reply = (
                                    reply
                                    + "\n\n文件已经通过内部验收，但最终 Workspace "
                                    + "交付失败，正式目录未被继续覆盖。\n"
                                    + f"交付编号：{promotion.promotion_id}"
                                )
                    except Exception:
                        logger.exception(
                            "本轮Workspace交付准备失败 | event_id=%s",
                            resolved_event_id,
                        )
                        reply = (
                            reply
                            + "\n\n内部验收记录已经保留，但最终 Workspace "
                            + "交付准备失败；没有继续覆盖用户文件。"
                        )

                    # Conversation history contains only what the user sees.
                    # Internal Step/Review evidence stays in the planning run.
                    previous_run_summary = ""

                    # 无论上游DIRECT直接回答，
                    # 还是Worker在独立Step Thread中执行并完成审核，
                    # 最终都把本轮用户消息和回答写回原Conversation。
                    await agent.aupdate_state(
                        self._build_config(
                            conversation
                            .thread_id
                        ),

                        {
                            "messages": [
                                {
                                    "role": "user",

                                    "content": (
                                        normalized_user_text
                                    ),
                                },

                                {
                                    "role": "assistant",

                                    "content": (
                                        reply
                                    ),
                                },
                            ],

                            "last_active_at": (
                                now
                            ),

                            "previous_run_summary": previous_run_summary,
                        },

                        # 这里保存的是已经由Planning Graph生成好的
                        # 最终用户消息和最终助手回答。
                        #
                        # 不应重新让General Worker执行这些消息，
                        # 只把它们视为model已经产生的最终状态。
                        as_node="model",
                    )
                    title_generation_status = (
                        "not_needed"
                    )

                    final_title = (
                        conversation.title
                    )

                    if not (
                            conversation
                                    .title_generated
                    ):
                        if planning_result.get("overall_stop_reason") == "direct_answer":
                            final_title = (
                                self._fallback_title(
                                    normalized_user_text
                                )
                            )
                            title_generation_status = (
                                "deterministic_direct"
                            )
                        else:
                            try:
                                final_title = await (
                                    generate_conversation_title(
                                        self.role_models.get("title", model),

                                        normalized_user_text,

                                        reply,
                                    )
                                )

                            except Exception:
                                logger.exception(
                                    "生成Conversation标题失败"
                                )

                                final_title = (
                                    self._fallback_title(
                                        normalized_user_text
                                    )
                                )

                                title_generation_status = (
                                    "fallback_after_error"
                                )

                            else:
                                title_generation_status = (
                                    "generated"
                                )

                        await agent.aupdate_state(
                            self._build_config(
                                conversation
                                .thread_id
                            ),

                            {
                                "conversation_title": (
                                    final_title
                                ),

                                "title_generated": (
                                    True
                                ),
                            },

                            # 标题生成发生在最终回答已经写入之后。
                            #
                            # 此处只是更新已有Conversation元数据，
                            # 明确归因于model节点，
                            # 避免LangGraph根据上一轮多节点历史推断失败。
                            as_node="model",
                        )

                    self._schedule_memory_consolidation(
                        memory_service=(
                            memory_service
                        ),

                        user_text=(
                            normalized_user_text
                        ),

                        source_platform=(
                            channel
                        ),

                        source_conversation_id=(
                            conversation
                            .conversation_id
                        ),

                        source_thread_id=(
                            conversation
                            .thread_id
                        ),
                    )

                    selected_memory_items = [
                        _memory_to_trace_item(
                            memory
                        )

                        for memory
                        in memories
                    ]

                    set_span_attributes(
                        turn_span,

                        **{
                            "conversation.memory_retrieval_status": (
                                memory_retrieval_status
                            ),

                            "conversation.injected_memory_count": (
                                len(
                                    memories
                                )
                            ),

                            "conversation.planning_status": (
                                planning_status
                            ),

                            "conversation.planning_final_status": (
                                str(
                                    planning_result.get(
                                        "final_status",
                                        "",
                                    )
                                )
                            ),

                            "conversation.planning_model_rounds": (
                                int(
                                    planning_result.get(
                                        "model_rounds_used",
                                        0,
                                    )
                                    or 0
                                )
                            ),

                            "conversation.planning_tool_calls": (
                                int(
                                    planning_result.get(
                                        "tool_calls_used",
                                        0,
                                    )
                                    or 0
                                )
                            ),

                            "conversation.title_generation_status": (
                                title_generation_status
                            ),

                            "conversation.assistant_reply_chars": (
                                len(
                                    reply
                                )
                            ),

                            "conversation.memory_background_scheduled": (
                                True
                            ),
                        },
                    )

                    set_span_output(
                        turn_span,

                        {
                            "assistant_reply": (
                                reply
                            ),

                            "planning": {
                                "status": (
                                    planning_status
                                ),

                                "final_status": (
                                    planning_result.get(
                                        "final_status"
                                    )
                                ),

                                "overall_stop_reason": (
                                    planning_result.get(
                                        "overall_stop_reason"
                                    )
                                ),

                                "model_rounds_used": (
                                    planning_result.get(
                                        "model_rounds_used",
                                        0,
                                    )
                                ),

                                "tool_calls_used": (
                                    planning_result.get(
                                        "tool_calls_used",
                                        0,
                                    )
                                ),

                                "step_reports": (
                                    planning_result.get(
                                        "completed_step_reports",
                                        [],
                                    )
                                ),

                                "replan_history": (
                                    planning_result.get(
                                        "replan_history",
                                        [],
                                    )
                                ),
                            },

                            "memory_retrieval": {
                                "status": (
                                    memory_retrieval_status
                                ),

                                "selected_count": (
                                    len(
                                        memories
                                    )
                                ),

                                "selected_memories": (
                                    selected_memory_items
                                ),

                                "rendered_context_chars": (
                                    len(
                                        memory_context
                                    )
                                ),
                            },

                            "title_generation": {
                                "status": (
                                    title_generation_status
                                ),

                                "final_title": (
                                    final_title
                                ),
                            },

                            "memory_consolidation": {
                                "scheduled": True,

                                "execution_mode": (
                                    "background"
                                ),
                            },
                        },
                    )

                    turn_overview_result.update({
                        "answer": reply,
                        "status": planning_result.get("final_status") or planning_status,
                        "planning": {
                            "planned_steps": planning_result.get("planned_steps", []),
                            "completed_step_reports": planning_result.get("completed_step_reports", []),
                            "final_status": planning_result.get("final_status"),
                            "overall_stop_reason": planning_result.get("overall_stop_reason"),
                        },
                    })

                    if channel != "appworld":
                        turn_span.update_name(
                            conversation_trace_name(
                                trace_query_text,
                                status=(
                                    planning_result.get("final_status")
                                    or planning_status
                                ),
                            )
                        )

                    return reply

    async def approve_workspace_promotion(
        self,
        *,
        conversation_id: str,
        promotion_id: str,
    ) -> WorkspacePromotion:
        promotion = await self.event_store.get_workspace_promotion(promotion_id)
        if promotion is None:
            raise LookupError(f"没有找到交付请求：{promotion_id}")
        if promotion.conversation_id != conversation_id:
            raise ValueError("交付请求不属于当前Conversation。")
        return await self.workspace_promotions.approve(
            promotion_id,
            actor="human:feishu",
            reason="User approved reviewed files for final delivery.",
        )

    async def reject_workspace_promotion(
        self,
        *,
        conversation_id: str,
        promotion_id: str,
        reason: str | None = None,
    ) -> WorkspacePromotion:
        promotion = await self.event_store.get_workspace_promotion(promotion_id)
        if promotion is None:
            raise LookupError(f"没有找到交付请求：{promotion_id}")
        if promotion.conversation_id != conversation_id:
            raise ValueError("交付请求不属于当前Conversation。")
        return await self.workspace_promotions.reject(
            promotion_id,
            actor="human:feishu",
            reason=reason,
        )

    async def finalize_cancelled_run(
        self,
        *,
        target_event_id: str,
        cancel_event_id: str,
    ) -> dict[str, Any]:
        """Archive run-owned resources after cooperative Event cancellation."""

        normalized_target = str(target_event_id).strip()
        normalized_cancel = str(cancel_event_id).strip()
        if not normalized_target or not normalized_cancel:
            raise ValueError("Cancellation finalization requires Event identities")

        code_records: tuple[dict[str, Any], ...] = ()
        code_runtime = self.code_runtime
        if code_runtime is not None:
            code_records = await asyncio.to_thread(
                code_runtime.cancel_run,
                normalized_target,
                reason=(
                    "The owning user Event was cancelled at a committed safe "
                    "point. Unreviewed files remain archive-only."
                ),
            )

        layout = initialize_run_workspace(normalized_target)
        receipt_path = await asyncio.to_thread(
            write_run_cancellation_receipt,
            layout=layout,
            cancel_event_id=normalized_cancel,
            target_event_id=normalized_target,
            code_final_records=code_records,
        )
        logger.info(
            "CANCEL归档完成 | target_event_id=%s | receipt=%s | code_records=%s",
            normalized_target,
            receipt_path,
            len(code_records),
        )
        return {
            "receipt_path": str(receipt_path),
            "code_final_records": list(code_records),
        }

    async def finalize_superseded_run(
        self,
        *,
        target_event_id: str,
        replacement_event_id: str,
    ) -> dict[str, Any]:
        """Archive the old generation before its replacement starts planning."""

        normalized_target = str(target_event_id).strip()
        normalized_replacement = str(replacement_event_id).strip()
        if not normalized_target or not normalized_replacement:
            raise ValueError("Supersession finalization requires Event identities")

        code_records: tuple[dict[str, Any], ...] = ()
        code_runtime = self.code_runtime
        if code_runtime is not None:
            code_records = await asyncio.to_thread(
                code_runtime.supersede_run,
                normalized_target,
                reason=(
                    "A newer user Event superseded this run at a committed "
                    "safe point. Unreviewed files remain archive-only."
                ),
            )

        layout = initialize_run_workspace(normalized_target)
        receipt_path = await asyncio.to_thread(
            write_run_supersession_receipt,
            layout=layout,
            replacement_event_id=normalized_replacement,
            target_event_id=normalized_target,
            code_final_records=code_records,
        )
        replacement_layout = initialize_run_workspace(normalized_replacement)
        inheritance_receipt_path = await asyncio.to_thread(
            inherit_replacement_workspace,
            source_layout=layout,
            replacement_layout=replacement_layout,
            replacement_event_id=normalized_replacement,
        )
        logger.info(
            "REPLACE归档完成 | target_event_id=%s | replacement_event_id=%s | "
            "receipt=%s | code_records=%s",
            normalized_target,
            normalized_replacement,
            receipt_path,
            len(code_records),
        )
        return {
            "receipt_path": str(receipt_path),
            "inheritance_receipt_path": str(inheritance_receipt_path),
            "code_final_records": list(code_records),
        }

    def _schedule_memory_consolidation(
            self,
            *,
            memory_service: MemoryService,
            user_text: str,
            source_platform: str,
            source_conversation_id: str,
            source_thread_id: str,
    ) -> None:
        """创建不继承当前Trace父节点的后台Write Gate任务。"""

        coroutine = (
            self._consolidate_memory_background(
                memory_service=(
                    memory_service
                ),

                user_text=(
                    user_text
                ),

                source_platform=(
                    source_platform
                ),

                source_conversation_id=(
                    source_conversation_id
                ),

                source_thread_id=(
                    source_thread_id
                ),
            )
        )

        # asyncio.create_task默认会复制当前Context。
        #
        # 当前函数是在conversation_turn Span内部调用的。
        # 如果直接create_task，后台任务可能继承
        # conversation_turn作为父Span。
        #
        # 使用全新的空Context创建任务，
        # 让后台记忆巩固成为独立Trace。
        background_context = (
            Context()
        )

        task = background_context.run(
            asyncio.create_task,

            coroutine,

            name=(
                "memory-write-gate-"
                f"{source_thread_id}"
            ),
        )

        self._background_tasks.add(
            task
        )

        task.add_done_callback(
            self._on_background_task_done
        )

    def _on_background_task_done(
            self,
            task: asyncio.Task,
    ) -> None:
        """移除完成的后台任务，并记录未处理异常。"""

        self._background_tasks.discard(
            task
        )

        if task.cancelled():
            return

        error = task.exception()

        if error is not None:
            logger.error(
                "后台记忆任务异常退出 | "
                "task=%s",

                task.get_name(),

                exc_info=(
                    type(
                        error
                    ),

                    error,

                    error.__traceback__,
                ),
            )

    async def _consolidate_memory_background(
            self,
            *,
            memory_service: MemoryService,
            user_text: str,
            source_platform: str,
            source_conversation_id: str,
            source_thread_id: str,
    ) -> None:
        """在独立Trace中把用户原话送入持久化Write Gate队列。"""

        memory_lock = (
            self._memory_postprocess_locks
            .setdefault(
                source_thread_id,

                asyncio.Lock(),
            )
        )

        # 同一Conversation的记忆更新必须串行。
        #
        # 先等待Lock，再开始Span，
        # 这样Phoenix中的Duration主要表示
        # 实际巩固耗时，而不是排队等待时间。
        async with memory_lock:
            with trace_context(
                    session_id=(
                            source_thread_id
                    ),

                    metadata={
                        "conversation_id": (
                                source_conversation_id
                        ),

                        "thread_id": (
                                source_thread_id
                        ),

                        "channel": (
                                source_platform
                        ),

                        "execution_mode": (
                                "background"
                        ),
                    },

                    tags=[
                        "memory-write-gate",
                        "background",
                        source_platform,
                    ],
            ):
                try:
                    with trace_span(
                            "memory_write_buffer",

                            # 这是一个固定的记忆处理工作流，
                            # 并不自主选择工具，
                            # 因此使用chain。
                            kind="chain",

                            input_value={
                                "source_platform": (
                                        source_platform
                                ),

                                "source_conversation_id": (
                                        source_conversation_id
                                ),

                                "source_thread_id": (
                                        source_thread_id
                                ),

                                "user_message": (
                                        user_text
                                ),

                            },

                            attributes={
                                "memory.execution_mode": (
                                        "background"
                                ),

                                "memory.user_message_chars": (
                                        len(
                                            user_text
                                        )
                                ),

                                "conversation.id": (
                                        source_conversation_id
                                ),

                                "conversation.thread_id": (
                                        source_thread_id
                                ),
                            },
                    ) as span:

                        buffered_candidate_ids = await (
                            memory_service
                            .consolidate_turn(
                                user_text=(
                                    user_text
                                ),

                                source_platform=(
                                    source_platform
                                ),

                                source_conversation_id=(
                                    source_conversation_id
                                ),

                                source_thread_id=(
                                    source_thread_id
                                ),
                            )
                        )

                        set_span_attributes(
                            span,

                            **{
                                "memory.buffered_candidate_count": (
                                    len(
                                        buffered_candidate_ids
                                    )
                                ),
                            },
                        )

                        set_span_output(
                            span,

                            {
                                "buffered_candidate_count": (
                                    len(
                                        buffered_candidate_ids
                                    )
                                ),

                                "buffered_candidate_ids": (
                                    buffered_candidate_ids
                                ),
                            },
                        )

                except Exception:
                    # memory_write_buffer Span已经记录
                    # 完整异常和ERROR状态。
                    logger.exception(
                        "后台Write Gate入队或批处理失败，"
                        "但不影响已经生成的Agent回答"
                    )
    async def stop(self) -> None:
        """有界等待后台任务，再关闭Store、模型和Checkpoint。"""

        background_tasks = list(
            self._background_tasks
        )

        if background_tasks:
            (
                _,
                pending_tasks,
            ) = await asyncio.wait(
                background_tasks,
                timeout=10.0,
            )

            if pending_tasks:
                logger.warning(
                    "后台长期记忆任务退出超时，"
                    "将取消剩余任务 | pending=%s",

                    len(
                        pending_tasks
                    ),
                )

                for task in pending_tasks:
                    task.cancel()

            results = await asyncio.gather(
                *background_tasks,
                return_exceptions=True,
            )

            failed_count = sum(
                1
                for result in results
                if (
                    isinstance(
                        result,
                        BaseException,
                    )
                    and not isinstance(
                        result,
                        asyncio.CancelledError,
                    )
                )
            )

            cancelled_count = sum(
                1
                for result in results
                if isinstance(
                    result,
                    asyncio.CancelledError,
                )
            )

            if failed_count:
                logger.warning(
                    "部分后台长期记忆任务异常结束 | "
                    "failed_count=%s",

                    failed_count,
                )

            if cancelled_count:
                logger.info(
                    "后台长期记忆任务已在退出时取消 | "
                    "cancelled_count=%s",

                    cancelled_count,
                )

        self._background_tasks.clear()

        self._memory_postprocess_locks.clear()

        configure_schedule_service(None)
        configure_host_file_judge(None)
        try:
            await self.schedule_service.stop()
        except Exception:
            logger.exception("关闭本地提醒服务时发生异常")

        try:
            await self.amap_mcp.stop()
        except Exception:
            logger.exception("关闭高德地图MCP时发生异常")

        try:
            await self.email_mcp.stop()
        except Exception:
            logger.exception("关闭只读邮箱MCP时发生异常")
        self.tools = list(self._base_tools)

        # 所有消息与记忆任务退出后，
        # 才开始关闭它们可能使用的底层资源。
        try:
            await self.playwright_mcp.stop()

        except Exception:
            logger.exception(
                "关闭Playwright MCP时发生异常"
            )

        memory_context = (
            self._memory_store_context
        )

        checkpoint_context = (
            self._checkpointer_context
        )

        retrieval_models = (
            self.retrieval_models
        )

        code_runtime = self.code_runtime

        self._memory_store_context = None
        self._memory_store = None

        self._checkpointer_context = None
        self._checkpointer = None

        self.retrieval_hub = None
        self.retrieval_models = None
        self.toolset_router = None
        self.memory_service = None

        self.planning_graph = None
        self.toolset_catalog = []

        self.hard_model = None
        self.model = None
        self.summary_model = None
        self.role_models = {}
        self.general_worker = None
        self.web_worker = None
        self.code_runtime = None

        self._thread_locks.clear()

        if code_runtime is not None:
            try:
                code_runtime.close()
            except Exception:
                logger.exception("关闭CODE Docker控制面时发生异常")

        if memory_context is not None:
            try:
                await memory_context.__aexit__(
                    None,
                    None,
                    None,
                )

            except Exception:
                logger.exception(
                    "关闭Memory Store时发生异常"
                )

        if retrieval_models is not None:
            try:
                await retrieval_models.aclose()

            except Exception:
                logger.exception(
                    "关闭本地检索模型时发生异常"
                )

        if checkpoint_context is not None:
            try:
                await checkpoint_context.__aexit__(
                    None,
                    None,
                    None,
                )

            except Exception:
                logger.exception(
                    "关闭Checkpoint Store时发生异常"
                )

        try:
            await self.event_store.close()
        except Exception:
            logger.exception(
                "关闭Event Store时发生异常"
            )

        logger.info(
            "ConversationRuntime关闭流程完成。"
        )

    def _require_general_worker(
        self,
    ) -> WorkerLeadershipBridge:
        if self.general_worker is None:
            raise RuntimeError(
                "ConversationRuntime尚未启动。"
            )

        return self.general_worker

    def _require_planning_graph(
        self,
    ):
        """返回已经编译的Planning Graph。"""

        if self.planning_graph is None:
            raise RuntimeError(
                "Planning Graph尚未创建。"
            )

        return self.planning_graph

    def _require_memory_service(
        self,
    ) -> MemoryService:
        if self.memory_service is None:
            raise RuntimeError(
                "MemoryService尚未启动。"
            )

        return self.memory_service

    async def memory_conflict_summary(self) -> dict[str, int]:
        """Expose deterministic conflict state to the authenticated UI layer."""

        return await self._require_memory_service().conflict_summary()

    async def next_memory_conflict(
        self,
        *,
        excluded_group_ids: set[str] | None = None,
    ):
        return await self._require_memory_service().next_conflict_pair(
            excluded_group_ids=excluded_group_ids,
        )

    async def resolve_memory_conflict(
        self,
        *,
        group_id: str,
        keep_memory_id: str,
        retire_memory_id: str,
    ) -> dict[str, Any]:
        return await self._require_memory_service().resolve_conflict_pair(
            group_id=group_id,
            keep_memory_id=keep_memory_id,
            retire_memory_id=retire_memory_id,
        )

    def _require_model(
        self,
    ):
        if self.model is None:
            raise RuntimeError(
                "模型尚未创建。"
            )

        return self.model

    def _require_hard_model(
            self,
    ):
        """返回已经初始化的Hard模型。"""

        if self.hard_model is None:
            raise RuntimeError(
                "Hard模型尚未创建。"
            )

        return self.hard_model

    def _require_checkpointer(
        self,
    ) -> AsyncSqliteSaver:
        if self._checkpointer is None:
            raise RuntimeError(
                "Checkpointer尚未启动。"
            )

        return self._checkpointer

    @staticmethod
    def _build_config(
        thread_id: str,
    ) -> dict:
        return {
            "configurable": {
                "thread_id": (
                    thread_id
                ),
            }
        }

    @staticmethod
    def _build_channel_key(
        channel: str,
        external_chat_id: int | str,
    ) -> str:
        normalized_channel = (
            channel
            .strip()
            .lower()
        )

        normalized_chat_id = (
            str(external_chat_id)
            .strip()
        )

        if not normalized_channel:
            raise ValueError(
                "channel不能为空。"
            )

        if not normalized_chat_id:
            raise ValueError(
                "external_chat_id不能为空。"
            )

        return (
            f"{normalized_channel}:"
            f"{normalized_chat_id}"
        )

    @staticmethod
    def _build_short_id(
        conversation_id: str,
    ) -> str:
        return (
            conversation_id
            .removeprefix(
                "conv_"
            )[:6]
        )

    @staticmethod
    def _normalize_title(
        title: str,
    ) -> str:
        normalized = (
            " ".join(
                title.split()
            )
        )

        if not normalized:
            return "新对话"

        return normalized[:30]

    @staticmethod
    def _fallback_title(
        user_text: str,
    ) -> str:
        normalized = "".join(
            user_text
            .strip()
            .split()
        )

        if not normalized:
            return "新对话"

        return normalized[:10]

    @staticmethod
    def _now() -> str:
        return datetime.now(
            timezone.utc
        ).isoformat()
