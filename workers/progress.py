"""Checkpoint-ready progress reporting shared by Web and Code workers."""

from __future__ import annotations

import operator
from collections.abc import (
    Callable,
)
from datetime import (
    datetime,
    timezone,
)
from typing import (
    Annotated,
    Any,
    Literal,
)

from deepagents.graph import (
    DeepAgentState,
)
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
    hook_config,
)
from langchain.messages import (
    SystemMessage,
    ToolMessage,
)
from langchain.tools import (
    ToolRuntime,
    tool,
)
from langgraph.types import (
    Command,
    interrupt,
)
from pydantic import (
    BaseModel,
    Field,
    field_validator,
)
from typing_extensions import (
    NotRequired,
)
from workers.leadership_models import LeadershipDecisionResult
from prompt_loader import load_prompt, render_prompt

from config import (
    WORKER_FINALIZATION_DEFAULT_MODEL_ROUNDS,
    WORKER_PROGRESS_DEFAULT_TOOL_CALLS,
    WORKER_PROGRESS_HARD_MAX_TOOL_CALLS,
    WORKER_PROGRESS_MIN_TOOL_CALLS,
)
from workers.submission import SUBMIT_FOR_REVIEW_NAME


PUBLISH_WORKER_PROGRESS_NAME = (
    "publish_worker_progress"
)


WorkerCompletionClaim = Literal[
    "not_ready",
    "possibly_ready",
    "blocked",
]


class WorkerProgressPayload(
    BaseModel
):
    """Model-authored progress shared by Web and Code workers."""

    phase: str = Field(
        min_length=1,
        max_length=80,
        description=(
            "Current concise phase, such as searching, "
            "verifying_sources, implementing, or testing."
        ),
    )
    summary: str = Field(
        min_length=1,
        max_length=1200,
        description=(
            "What has been accomplished since the previous report."
        ),
    )
    findings: list[str] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Important evidence, decisions, or completed changes."
        ),
    )
    difficulties: list[str] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Current blockers, repeated failures, or uncertainty."
        ),
    )
    evidence_refs: list[str] = Field(
        default_factory=list,
        description=(
            "Relevant references to sources, files, test output, or other saved artifacts."
        ),
    )
    next_action: str = Field(
        min_length=1,
        max_length=600,
        description=(
            "The next concrete action the worker plans to take."
        ),
    )
    completion_claim: WorkerCompletionClaim = Field(
        description=(
            "not_ready, possibly_ready, or blocked. "
            "This is a worker claim, not an authoritative runtime status."
        ),
    )

    @field_validator(
        "phase",
        "summary",
        "next_action",
    )
    @classmethod
    def _normalize_required_text(
        cls,
        value: str,
    ) -> str:
        normalized = " ".join(
            value.strip().split()
        )

        if not normalized:
            raise ValueError(
                "progress text cannot be empty"
            )

        return normalized

    @field_validator(
        "findings",
        "difficulties",
        "evidence_refs",
    )
    @classmethod
    def _normalize_string_lists(
        cls,
        values: list[str],
    ) -> list[str]:
        normalized_values: list[str] = []
        seen: set[str] = set()

        for value in values:
            normalized = " ".join(
                str(value).strip().split()
            )

            if (
                normalized
                and normalized not in seen
            ):
                seen.add(
                    normalized
                )
                normalized_values.append(
                    normalized
                )

        return normalized_values


class WorkerProgressRecord(
    BaseModel
):
    """Harness-authored envelope around one model progress payload."""

    sequence: int = Field(
        ge=1
    )
    total_tool_calls: int = Field(
        ge=1
    )
    published_at: datetime
    worker_id: str | None = None
    event_id: str | None = None
    step_id: str | None = None
    progress: WorkerProgressPayload


class WorkerProgressState(
    DeepAgentState
):
    """Unified Conversation, execution-budget, and Worker progress state."""

    conversation_id: NotRequired[str]
    worker_evidence_refs: NotRequired[dict[str, str]]
    worker_criterion_refs: NotRequired[dict[str, str]]
    worker_target_selection: NotRequired[dict[str, Any] | None]
    worker_invalid_target_receipts: NotRequired[list[str]]
    # Scheduler为当前Step生成的精确检索query；工具组路由优先复用它。
    # 为空时Planning Graph会写入由objective和success_criteria构成的
    # 确定性回退文本。
    toolset_route_query: NotRequired[str]
    # Scheduler决定当前Step是否开放业务工具。缺失时Harness按ENABLED处理。
    step_tool_access: NotRequired[str | bool]
    # 仅当第一轮本地路由低于绝对阈值时使用；由Harness从用户原话生成头尾窗口。
    toolset_route_fallback_query: NotRequired[str]
    # 仅供低置信工具组模型选择使用；不自动拼入普通Worker上下文。
    toolset_route_full_user_request: NotRequired[str]
    conversation_title: NotRequired[str]
    channel_key: NotRequired[str]
    created_at: NotRequired[str]
    last_active_at: NotRequired[str]
    title_generated: NotRequired[bool]
    memory_context: NotRequired[str]
    execution_instructions: NotRequired[str]
    skill_catalog: NotRequired[list[dict[str, Any]]]
    skill_topics: NotRequired[list[str]]
    skill_mode: NotRequired[str]
    skill_fixed_ids: NotRequired[dict[str, list[str]]]
    role_skill_snapshot: NotRequired[dict[str, Any]]
    skill_preparation_calls_used: NotRequired[int]
    worker_compaction_calls_used: NotRequired[int]
    worker_archived_messages: NotRequired[list[Any]]
    worker_control_fingerprint: NotRequired[str]
    runtime_context_fingerprint: NotRequired[str]
    code_context_fingerprint: NotRequired[str]
    conversation_summary: NotRequired[str]
    conversation_summary_message_count: NotRequired[int]

    executor_model_run_limit: NotRequired[int]
    executor_tool_run_limit: NotRequired[int]
    show_all_toolsets_run_limit: NotRequired[int]
    executor_model_calls_used: NotRequired[int]
    executor_tool_calls_used: NotRequired[int]
    show_all_toolsets_calls_used: NotRequired[int]

    worker_id: NotRequired[str]
    event_id: NotRequired[str]
    step_id: NotRequired[str]
    run_storage_root: NotRequired[str]

    worker_total_tool_calls: NotRequired[
        int
    ]
    worker_last_progress_tool_call: NotRequired[
        int
    ]
    worker_progress_reports: NotRequired[
        Annotated[
            list[
                dict[
                    str,
                    Any,
                ]
            ],
            operator.add,
        ]
    ]
    worker_control_enabled: NotRequired[bool]
    worker_terminal_action: NotRequired[str]
    worker_leadership_model_rounds_used: NotRequired[int]
    worker_leadership_decisions: NotRequired[
        Annotated[list[dict[str, Any]], operator.add]
    ]
    worker_review_requested: NotRequired[bool]
    worker_submission: NotRequired[dict[str, Any]]
    general_result: NotRequired[dict[str, Any]]
    worker_finalize_requested: NotRequired[bool]
    worker_finalize_reason: NotRequired[str]
    worker_finalization_model_run_limit: NotRequired[int]
    worker_finalization_model_calls_used: NotRequired[int]
    worker_finalization_tool_calls_used: NotRequired[int]
    worker_schema_repair_model_run_limit: NotRequired[int]
    worker_schema_repair_model_calls_used: NotRequired[int]
    worker_cancellation_record: NotRequired[dict[str, Any]]
    worker_downloaded_artifacts: NotRequired[
        Annotated[list[dict[str, Any]], operator.add]
    ]


def _read_non_negative_int(
    state: WorkerProgressState,
    field_name: str,
) -> int:
    value = int(
        state.get(
            field_name,
            0,
        )
    )

    return max(
        value,
        0,
    )


def count_tool_calls_since_progress(
    state: WorkerProgressState,
) -> int:
    """Return tool calls made after the latest published checkpoint."""

    total = _read_non_negative_int(
        state,
        "worker_total_tool_calls",
    )
    last_report = _read_non_negative_int(
        state,
        "worker_last_progress_tool_call",
    )

    return max(
        total - last_report,
        0,
    )


@tool(
    PUBLISH_WORKER_PROGRESS_NAME,
    description=load_prompt(
        "workers/publish_progress_tool"
    ),
)
def publish_worker_progress(
    progress: WorkerProgressPayload,
    runtime: ToolRuntime[
        None,
        WorkerProgressState,
    ],
) -> Command:
    """Publish a structured checkpoint before continuing worker tool use.

    Use this tool when runtime control requires a progress report. Report only
    evidence already observed, clearly describe difficulties, and state the
    next concrete action. A possibly_ready claim is advisory: the supervisor
    decides whether the overall task is complete.
    """

    state = runtime.state
    total_tool_calls = (
        _read_non_negative_int(
            state,
            "worker_total_tool_calls",
        )
    )
    previous_reports = list(
        state.get(
            "worker_progress_reports",
            [],
        )
    )
    record = WorkerProgressRecord(
        sequence=(
            len(previous_reports)
            + 1
        ),
        total_tool_calls=(
            total_tool_calls
        ),
        published_at=datetime.now(
            timezone.utc
        ),
        worker_id=state.get(
            "worker_id"
        ),
        event_id=state.get(
            "event_id"
        ),
        step_id=state.get(
            "step_id"
        ),
        progress=progress,
    )
    record_value = record.model_dump(
        mode="json"
    )

    runtime.stream_writer(
        {
            "type": (
                "worker_progress"
            ),
            "record": record_value,
        }
    )

    leadership_result: LeadershipDecisionResult | None = None
    if bool(state.get("worker_control_enabled", False)):
        leadership_result = LeadershipDecisionResult.model_validate(
            interrupt(
                {
                    "type": "worker_progress_gate",
                    "record": record_value,
                }
            )
        )

    message = (
        "Worker progress checkpoint "
        f"{record.sequence} published after {total_tool_calls} total tool calls."
    )
    update: dict[str, Any] = {
        "worker_last_progress_tool_call": total_tool_calls,
        "worker_progress_reports": [record_value],
    }

    if leadership_result is not None:
        decision = leadership_result.decision
        update["worker_leadership_model_rounds_used"] = (
            _read_non_negative_int(state, "worker_leadership_model_rounds_used")
            + leadership_result.model_rounds_used
        )
        update["worker_leadership_decisions"] = [
            leadership_result.model_dump(mode="json")
        ]
        if decision.action == "GUIDE":
            message += f" Leadership guidance: {decision.guidance}"
        elif decision.action in {"ACCEPT", "CANCEL", "REPLACE"}:
            update["worker_terminal_action"] = decision.action
            message += (
                f" Leadership selected {decision.action}: {decision.reason}"
            )
            if decision.action == "REPLACE":
                message += (
                    " Replacement assignment: "
                    f"{decision.replacement_assignment}"
                )
            if decision.action in {"ACCEPT", "REPLACE"}:
                update["worker_finalize_requested"] = True
                update["worker_finalize_reason"] = (
                    f"LEADERSHIP_{decision.action}"
                )
        else:
            message += " Leadership selected CONTINUE."

    return Command(
        update={
            **update,
            "messages": [
                ToolMessage(
                    content=message,
                    tool_call_id=(
                        runtime.tool_call_id
                        or PUBLISH_WORKER_PROGRESS_NAME
                    ),
                )
            ],
        }
    )


class WorkerProgressMiddleware(
    AgentMiddleware[
        WorkerProgressState
    ]
):
    """Count every tool call and require periodic structured checkpoints."""

    state_schema = WorkerProgressState

    def __init__(
        self,
        every_tool_calls: int = (
            WORKER_PROGRESS_DEFAULT_TOOL_CALLS
        ),
        finalization_model_rounds: int = (
            WORKER_FINALIZATION_DEFAULT_MODEL_ROUNDS
        ),
        schema_repair_max_rounds: int = 3,
        finalization_tools: tuple[str, ...] = (SUBMIT_FOR_REVIEW_NAME,),
    ) -> None:
        if not (
            WORKER_PROGRESS_MIN_TOOL_CALLS
            <= every_tool_calls
            <= WORKER_PROGRESS_HARD_MAX_TOOL_CALLS
        ):
            raise ValueError(
                "Worker progress interval must be between "
                f"{WORKER_PROGRESS_MIN_TOOL_CALLS} and "
                f"{WORKER_PROGRESS_HARD_MAX_TOOL_CALLS} tool calls."
            )

        self.every_tool_calls = (
            every_tool_calls
        )
        if finalization_model_rounds < 1:
            raise ValueError("Worker finalization rounds must be positive.")
        self.finalization_model_rounds = finalization_model_rounds
        if schema_repair_max_rounds < 1:
            raise ValueError("Worker schema repair rounds must be positive.")
        self.schema_repair_max_rounds = schema_repair_max_rounds
        if not finalization_tools or len(set(finalization_tools)) != len(finalization_tools):
            raise ValueError("Finalization tools must be a non-empty unique tuple")
        self.finalization_tools = finalization_tools

    @hook_config(can_jump_to=["end"])
    def before_model(
        self,
        state: WorkerProgressState,
        runtime,
    ) -> dict[str, Any] | None:
        """Stop cooperatively after ACCEPT, CANCEL, or REPLACE is checkpointed."""

        if state.get("worker_review_requested"):
            return {"jump_to": "end"}

        if state.get("worker_terminal_action") == "CANCEL":
            return {"jump_to": "end"}
        if (
            state.get("worker_finalize_requested")
            and state.get("worker_finalize_reason") == "SCHEMA_REPAIR"
        ):
            used = _read_non_negative_int(
                state,
                "worker_schema_repair_model_calls_used",
            )
            limit = max(
                int(
                    state.get(
                        "worker_schema_repair_model_run_limit",
                        self.schema_repair_max_rounds,
                    )
                    or self.schema_repair_max_rounds
                ),
                1,
            )
            if used >= limit:
                return {"jump_to": "end"}
        elif state.get("worker_finalize_requested"):
            used = _read_non_negative_int(
                state,
                "worker_finalization_model_calls_used",
            )
            limit = max(
                int(
                    state.get(
                        "worker_finalization_model_run_limit",
                        self.finalization_model_rounds,
                    )
                    or self.finalization_model_rounds
                ),
                1,
            )
            if used >= limit:
                return {"jump_to": "end"}
        from workers.runtime_events import context_event
        if state.get("worker_finalize_requested"):
            reminder = render_prompt("workers/finalize_submission", reason=state.get("worker_finalize_reason") or "结束", finalization_tools=" / ".join(self.finalization_tools))
        elif self._progress_is_due(state):
            reminder = render_prompt("workers/progress_checkpoint", tool_calls_since_progress=count_tool_calls_since_progress(state))
        else:
            return {"worker_control_fingerprint": ""} if state.get("worker_control_fingerprint") else None
        return context_event(state, reminder, "worker_control_fingerprint")

    @hook_config(can_jump_to=["model"])
    def after_model(
        self,
        state: WorkerProgressState,
        runtime,
    ) -> dict[
        str,
        Any,
    ] | None:
        """Count every tool call requested by the latest model response."""

        messages = list(
            state.get(
                "messages",
                [],
            )
        )

        if not messages:
            return None

        tool_calls = list(
            getattr(
                messages[-1],
                "tool_calls",
                [],
            )
            or []
        )

        if state.get("worker_finalize_requested"):
            updates: dict[str, Any] = {
                "worker_finalization_model_calls_used": (
                    _read_non_negative_int(
                        state,
                        "worker_finalization_model_calls_used",
                    )
                    + 1
                ),
                "worker_finalization_tool_calls_used": (
                    _read_non_negative_int(
                        state,
                        "worker_finalization_tool_calls_used",
                    )
                    + sum(
                        1
                        for tool_call in tool_calls
                        if self._tool_name(tool_call)
                        in self.finalization_tools
                    )
                ),
            }
            if tool_calls:
                updates["worker_total_tool_calls"] = (
                    _read_non_negative_int(
                        state,
                        "worker_total_tool_calls",
                    )
                    + len(tool_calls)
                )
            return updates

        if not tool_calls:
            if (
                not state.get("worker_finalize_requested")
                and not state.get("worker_review_requested")
                and state.get("worker_terminal_action") != "CANCEL"
            ):
                return {
                    "worker_finalize_requested": True,
                    "worker_finalize_reason": "NATURAL_EXIT",
                    "jump_to": "model",
                }
            return None

        total_before = (
            _read_non_negative_int(
                state,
                "worker_total_tool_calls",
            )
        )

        return {
            "worker_total_tool_calls": (
                total_before
                + len(tool_calls)
            )
        }

    def _progress_is_due(
        self,
        state: WorkerProgressState,
    ) -> bool:
        return (
            count_tool_calls_since_progress(
                state
            )
            >= self.every_tool_calls
        )

    @staticmethod
    def _tool_name(
        current_tool,
    ) -> str:
        if isinstance(
            current_tool,
            dict,
        ):
            return str(
                current_tool.get(
                    "name",
                    current_tool.get(
                        "function",
                        {},
                    ).get(
                        "name",
                        "",
                    ),
                )
            )

        return str(
            getattr(
                current_tool,
                "name",
                "",
            )
        )

    def _require_control_request(
        self,
        request: ModelRequest,
    ) -> ModelRequest:
        if request.state.get("worker_finalize_requested") and not (
            request.state.get("worker_review_requested")
        ):
            submit_tools = [
                current_tool
                for current_tool in request.tools
                if self._tool_name(current_tool) in self.finalization_tools
            ]
            if len(submit_tools) != len(self.finalization_tools):
                raise RuntimeError(
                    "Worker finalization requires its registered terminal tools: "
                    + ", ".join(self.finalization_tools)
                )
            return request.override(tools=submit_tools, tool_choice=self.finalization_tools[0] if len(submit_tools) == 1 else "required")

        if not self._progress_is_due(
            request.state
        ):
            return request

        progress_tools = [
            current_tool
            for current_tool
            in request.tools
            if (
                self._tool_name(
                    current_tool
                )
                == PUBLISH_WORKER_PROGRESS_NAME
            )
        ]

        if len(progress_tools) != 1:
            raise RuntimeError(
                "Worker runtime requires exactly one "
                "publish_worker_progress tool."
            )

        return request.override(tools=progress_tools, tool_choice=PUBLISH_WORKER_PROGRESS_NAME)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[
            [ModelRequest],
            ModelResponse,
        ],
    ) -> ModelResponse:
        """Temporarily force the progress tool on a synchronous model call."""

        return handler(
            self._require_progress_request(
                request
            )
        )

    def _require_progress_request(self, request: ModelRequest) -> ModelRequest:
        """Backward-compatible alias for the unified control selector."""

        return self._require_control_request(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable,
    ) -> ModelResponse:
        """Force a due progress report or terminal submission."""

        return await handler(
            self._require_control_request(
                request
            )
        )


__all__ = [
    "PUBLISH_WORKER_PROGRESS_NAME",
    "WorkerCompletionClaim",
    "WorkerProgressMiddleware",
    "WorkerProgressPayload",
    "WorkerProgressRecord",
    "WorkerProgressState",
    "count_tool_calls_since_progress",
    "publish_worker_progress",
]
