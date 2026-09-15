from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from typing_extensions import (
    NotRequired,
)

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    HostExecutionPolicy,
    ModelRequest,
    ShellToolMiddleware,
    ToolCallLimitMiddleware,
    hook_config,
)

from langchain.messages import (
    RemoveMessage,
)

from langgraph.graph.message import (
    REMOVE_ALL_MESSAGES,
)

from path import WORKSPACE_ROOT
from retrieval_models import (
    RetrievalModelManager,
)
from toolset_router import (
    ToolsetRouter,
)
from observability import (
    set_span_attributes,
    set_span_output,
    trace_span,
)
from dataclasses import (
    dataclass,
)
from tools.show_all_toolsets import (
    SHOW_ALL_TOOLSETS_NAME,
    SHOW_ALL_TOOLSETS_PREFIX,
)

logger = logging.getLogger("agent")

GIT_INSTALL_COMMAND = (
    "winget install --id Git.Git -e --source winget"
)

# Cross-Encoder的max_length是按每个
# “用户请求 + 单个工具描述”分别计算的。
#
# 这里仍然主动压缩文本，
# 避免极长用户消息或极长MCP描述
# 占满单个Pair的上下文。
TOOL_SELECTOR_QUERY_MAX_CHARS = 1600

# 单用户应用只保存少量Conversation的
# 最近一次工具组路由结果。
TOOLSET_ROUTE_CACHE_MAX_CONVERSATIONS = 32
TOOLSET_ROUTE_STEP_WEIGHT = 0.4
TOOLSET_ROUTE_USER_WEIGHT = 0.6

# Scheduler关闭工具访问时仍需允许角色提交结果或进入既有审核协议。
# 检索、历史、文件和业务工具都不在此集合中。
NO_ACCESS_CONTROL_TOOL_NAMES = frozenset({
    "report_general_result",
    "publish_worker_progress",
    "submit_for_review",
    "submit_code_for_review",
    "respond_to_code_review",
    "submit_continued_code_for_review",
    "request_code_worker_repair",
    "publish_reviewed_candidate",
    "submit_code_review",
})

@dataclass(
    frozen=True,
)
class ToolsetRouteCacheEntry:
    """保存一个Conversation最近一次工具组路由结果。"""

    user_turn_number: int

    routing_task: str

    fallback_routing_task: str

    tool_access_enabled: bool

    tool_signature: tuple[
        str,
        ...,
    ]

    selected_toolset_names: tuple[
        str,
        ...,
    ]

    selected_tool_names: tuple[
        str,
        ...,
    ]

    # 本地加权路由和同Worker模型都失败时缓存控制面结果，避免每轮
    # 重复付费；不会把全部业务工具暴露给模型。
    routing_failed: bool = False

    fallback_reason: str | None = None

def _message_content_to_text(
    content: Any,
) -> str:
    """把LangChain消息内容转换成普通文本。"""

    if isinstance(
        content,
        str,
    ):
        return content

    if isinstance(
        content,
        list,
    ):
        text_parts: list[str] = []

        for item in content:
            if isinstance(
                item,
                str,
            ):
                text_parts.append(
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
                text_parts.append(
                    text
                )

        return "".join(
            text_parts
        )

    if isinstance(
        content,
        dict,
    ):
        text = content.get(
            "text"
        )

        if isinstance(
            text,
            str,
        ):
            return text

    if content is None:
        return ""

    return str(
        content
    )
def _compact_tool_selector_text(
    text: str,
    max_chars: int,
) -> str:
    """压缩Tool Selector使用的文本。"""

    normalized_text = (
        " ".join(
            text.split()
        )
    )

    if len(
        normalized_text
    ) <= max_chars:
        return normalized_text

    head_length = (
        max_chars
        // 2
    )

    tail_length = (
        max_chars
        - head_length
    )

    return (
        normalized_text[
            :head_length
        ]
        + "\n...\n"
        + normalized_text[
            -tail_length:
        ]
    )

def _read_message_fields(
    message: Any,
) -> tuple[
    str,
    str,
    str,
]:
    """统一读取消息的角色、工具名称和文字内容。"""

    if isinstance(
        message,
        dict,
    ):
        role = str(
            message.get(
                "role"
            )
            or message.get(
                "type"
            )
            or ""
        ).strip().lower()

        name = str(
            message.get(
                "name"
            )
            or ""
        ).strip()

        content = (
            _message_content_to_text(
                message.get(
                    "content",
                    "",
                )
            )
        )

    else:
        role = str(
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
        ).strip().lower()

        name = str(
            getattr(
                message,
                "name",
                "",
            )
            or ""
        ).strip()

        content = (
            _message_content_to_text(
                getattr(
                    message,
                    "content",
                    "",
                )
            )
        )

    return (
        "system" if (message.get("additional_kwargs", {}) if isinstance(message, dict) else getattr(message, "additional_kwargs", {})).get("personalops_runtime_event") else role,
        name,
        content,
    )

def _get_current_toolset_route(
    messages: list,
) -> tuple[
    int,
    str,
    str,
]:
    """读取当前用户Turn和最新工具组路由请求。"""

    latest_user_index: (
        int
        | None
    ) = None

    latest_user_text = ""

    user_turn_number = 0

    for (
        message_index,
        message,
    ) in enumerate(
        messages
    ):
        (
            role,
            _name,
            content,
        ) = _read_message_fields(
            message
        )

        if role not in {
            "user",
            "human",
        }:
            continue

        user_turn_number += 1

        latest_user_index = (
            message_index
        )

        latest_user_text = (
            _compact_tool_selector_text(
                content,

                max_chars=(
                    TOOL_SELECTOR_QUERY_MAX_CHARS
                ),
            )
        )

    if latest_user_index is None:
        return (
            0,
            "",
            "none",
        )

    # 只检查最近用户消息之后的ToolMessage。
    #
    # 旧Turn中的show_all_toolsets结果
    # 不能影响当前用户请求。
    for message in reversed(
        messages[
            latest_user_index + 1:
        ]
    ):
        (
            role,
            name,
            content,
        ) = _read_message_fields(
            message
        )

        if (
            role != "tool"

            or name
            != SHOW_ALL_TOOLSETS_NAME
        ):
            continue

        normalized_content = (
            content.strip()
        )

        if not (
            normalized_content
            .startswith(
                SHOW_ALL_TOOLSETS_PREFIX
            )
        ):
            continue

        requested_task = (
            normalized_content[
                len(
                    SHOW_ALL_TOOLSETS_PREFIX
                ):
            ]
            .strip()
        )

        if not requested_task:
            continue

        return (
            user_turn_number,

            _compact_tool_selector_text(
                requested_task,

                max_chars=(
                    TOOL_SELECTOR_QUERY_MAX_CHARS
                ),
            ),

            "show_all_toolsets",
        )

    return (
        user_turn_number,
        latest_user_text,
        "user_message",
    )
def _infer_model_round(
    messages: list,
) -> int:
    """判断当前用户Turn中的模型调用轮次。

    一个用户Turn中可能多次调用主模型：

    round_1:
        模型决定调用工具。

    round_2:
        模型读取工具结果后继续调用工具。

    round_3:
        模型生成最终回答。
    """

    latest_user_index: (
        int
        | None
    ) = None

    for (
        message_index,
        message,
    ) in enumerate(
        messages
    ):
        (
            role,
            _name,
            _content,
        ) = _read_message_fields(
            message
        )

        if role in {
            "user",
            "human",
        }:
            latest_user_index = (
                message_index
            )

    if latest_user_index is None:
        return 1

    previous_model_calls = 0

    for message in messages[
        latest_user_index + 1:
    ]:
        (
            role,
            _name,
            _content,
        ) = _read_message_fields(
            message
        )

        if role in {
            "assistant",
            "ai",
        }:
            previous_model_calls += 1

    return (
        previous_model_calls
        + 1
    )

def _get_conversation_key(
    request: ModelRequest,
) -> str:
    """读取当前Conversation的缓存Key。"""

    state = (
        request.state
        or {}
    )

    conversation_id = state.get(
        "conversation_id",
        "",
    )

    if (
        isinstance(
            conversation_id,
            str,
        )
        and conversation_id.strip()
    ):
        return (
            conversation_id.strip()
        )

    # 理论上正常Conversation都有ID。
    #
    # 这里作为单用户模式下的安全回退。
    return "__single_user__"


def _read_tool_name(
    tool: Any,
) -> str:
    """统一读取LangChain Tool名称。"""

    return str(
        getattr(
            tool,
            "name",
            "",
        )
    ).strip()

def _tool_to_trace_item(
    tool,
) -> dict[
    str,
    str,
]:
    """把一个工具转换成Phoenix中的简洁结构。"""

    tool_name = str(
        getattr(
            tool,
            "name",
            "",
        )
    ).strip()

    raw_description = getattr(
        tool,
        "description",
        "",
    )

    tool_description = (
        _message_content_to_text(
            raw_description
        )
    )

    tool_description = (
        _compact_tool_selector_text(
            tool_description,

            max_chars=320,
        )
    )

    return {
        "tool_name": (
            tool_name
        ),

        "description": (
            tool_description
        ),
    }



def _read_dynamic_limit(
    state: AgentState,
    field_name: str,
) -> int:
    """读取Planning Graph写入的非负整数预算。"""

    raw_value = state.get(
        field_name
    )

    if (
        isinstance(
            raw_value,
            bool,
        )
        or not isinstance(
            raw_value,
            int,
        )
    ):
        raise RuntimeError(
            f"Agent State缺少合法预算字段："
            f"{field_name}。"
        )

    if raw_value < 0:
        raise RuntimeError(
            f"Agent State预算不能为负数："
            f"{field_name}={raw_value}。"
        )

    return raw_value


def _current_turn_messages(
    messages: list[
        Any
    ],
) -> list[
    Any
]:
    """只保留最近一条用户消息之后的当前Turn。"""

    latest_user_index: (
        int
        | None
    ) = None

    for (
        message_index,
        message,
    ) in enumerate(
        messages
    ):
        (
            role,
            _name,
            _content,
        ) = _read_message_fields(
            message
        )

        if role in {
            "user",
            "human",
        }:
            latest_user_index = (
                message_index
            )

    if latest_user_index is None:
        return list(
            messages
        )

    return list(
        messages[
            latest_user_index:
        ]
    )


def _read_ai_tool_calls(
    message: Any,
) -> list[
    Any
]:
    """读取一条AIMessage声明的工具调用。"""

    (
        role,
        _name,
        _content,
    ) = _read_message_fields(
        message
    )

    if role not in {
        "assistant",
        "ai",
    }:
        return []

    if isinstance(
        message,
        dict,
    ):
        raw_tool_calls = (
            message.get(
                "tool_calls"
            )
            or []
        )

    else:
        raw_tool_calls = (
            getattr(
                message,
                "tool_calls",
                None,
            )
            or []
        )

    if not isinstance(
        raw_tool_calls,
        list,
    ):
        return []

    return list(
        raw_tool_calls
    )


def _read_tool_call_name(
    tool_call: Any,
) -> str:
    """读取ToolCall名称。"""

    if isinstance(
        tool_call,
        dict,
    ):
        function_data = (
            tool_call.get(
                "function"
            )
        )

        if not isinstance(
            function_data,
            dict,
        ):
            function_data = {}

        tool_name = (
            tool_call.get(
                "name"
            )
            or function_data.get(
                "name"
            )
            or ""
        )

    else:
        tool_name = (
            getattr(
                tool_call,
                "name",
                "",
            )
            or ""
        )

    return str(
        tool_name
    ).strip()


def _replace_ai_tool_calls(
    message: Any,
    *,
    tool_calls: list[
        Any
    ],
    fallback_content: str,
) -> Any:
    """复制AIMessage并替换允许进入Tools节点的调用。"""

    if isinstance(
        message,
        dict,
    ):
        updated_message = dict(
            message
        )

        updated_message[
            "tool_calls"
        ] = tool_calls

        existing_content = (
            _message_content_to_text(
                updated_message.get(
                    "content",
                    "",
                )
            )
            .strip()
        )

        if (
            not tool_calls
            and not existing_content
        ):
            updated_message[
                "content"
            ] = fallback_content

        return updated_message

    update_values: dict[
        str,
        Any,
    ] = {
        "tool_calls": (
            tool_calls
        ),
    }

    existing_content = (
        _message_content_to_text(
            getattr(
                message,
                "content",
                "",
            )
        )
        .strip()
    )

    if (
        not tool_calls
        and not existing_content
    ):
        update_values[
            "content"
        ] = fallback_content

    model_copy = getattr(
        message,
        "model_copy",
        None,
    )

    if callable(
        model_copy
    ):
        return model_copy(
            update=(
                update_values
            )
        )

    legacy_copy = getattr(
        message,
        "copy",
        None,
    )

    if callable(
        legacy_copy
    ):
        return legacy_copy(
            update=(
                update_values
            )
        )

    raise RuntimeError(
        "无法复制当前AIMessage，"
        "因此不能安全裁剪工具调用。"
    )


class ExecutionBudgetState(
    AgentState
):
    """动态预算Middleware自己的持久计数。"""

    executor_model_calls_used: NotRequired[
        int
    ]

    executor_tool_calls_used: NotRequired[
        int
    ]

    show_all_toolsets_calls_used: NotRequired[
        int
    ]
    skill_preparation_calls_used: NotRequired[int]
    worker_compaction_calls_used: NotRequired[int]
    worker_archived_messages: NotRequired[list[Any]]
    worker_control_fingerprint: NotRequired[str]
    worker_finalize_requested: NotRequired[bool]
    worker_finalize_reason: NotRequired[str]
    worker_review_requested: NotRequired[bool]
    worker_finalization_model_run_limit: NotRequired[int]
    worker_finalization_model_calls_used: NotRequired[int]
    worker_finalization_tool_calls_used: NotRequired[int]
    worker_schema_repair_model_run_limit: NotRequired[int]
    worker_schema_repair_model_calls_used: NotRequired[int]


class DynamicExecutionBudgetMiddleware(
    AgentMiddleware[
        ExecutionBudgetState
    ]
):
    """执行Planning Graph为当前Attempt计算的动态预算。

    模型预算：
        在每次模型调用前检查Middleware State中的真实调用计数。
        达到上限后直接跳到Agent结束节点。

    工具预算：
        在模型返回后、进入Tools节点前，裁剪本轮新AIMessage中的
        tool_calls。这样并行工具调用也不会发生竞争式越界。
    """

    state_schema = (
        ExecutionBudgetState
    )

    def __init__(
        self,
        *,
        enable_worker_finalization: bool = False,
        finalization_model_rounds: int = 1,
        schema_repair_max_rounds: int = 3,
    ) -> None:
        if finalization_model_rounds < 1:
            raise ValueError("finalization_model_rounds must be positive.")
        if schema_repair_max_rounds < 1:
            raise ValueError("schema_repair_max_rounds must be positive.")
        self.enable_worker_finalization = enable_worker_finalization
        self.finalization_model_rounds = finalization_model_rounds
        self.schema_repair_max_rounds = schema_repair_max_rounds

    @hook_config(can_jump_to=["end"])
    def before_model(
        self,
        state: ExecutionBudgetState,
        runtime,
    ) -> dict[
        str,
        Any,
    ] | None:
        """达到当前Attempt模型额度后结束Simple Agent。"""

        model_limit = (
            _read_dynamic_limit(
                state,
                "executor_model_run_limit",
            )
        )

        model_calls_used = int(
            state.get(
                "executor_model_calls_used",
                0,
            )
        )

        finalizing = bool(state.get("worker_finalize_requested")) and not bool(
            state.get("worker_review_requested")
        )
        model_calls_used += int(state.get("skill_preparation_calls_used", 0) or 0)
        model_calls_used += int(state.get("worker_compaction_calls_used", 0) or 0)
        finalization_used = max(
            int(state.get("worker_finalization_model_calls_used", 0) or 0),
            0,
        )
        finalization_limit = max(
            int(
                state.get(
                    "worker_finalization_model_run_limit",
                    self.finalization_model_rounds,
                )
                or self.finalization_model_rounds
            ),
            1,
        )

        schema_repairing = (
            finalizing and state.get("worker_finalize_reason") == "SCHEMA_REPAIR"
        )
        if schema_repairing:
            repair_used = max(
                int(state.get("worker_schema_repair_model_calls_used", 0) or 0),
                0,
            )
            repair_limit = max(
                int(
                    state.get(
                        "worker_schema_repair_model_run_limit",
                        self.schema_repair_max_rounds,
                    )
                    or self.schema_repair_max_rounds
                ),
                1,
            )
            if repair_used < repair_limit:
                return None
            return {"jump_to": "end"}

        if self.enable_worker_finalization and finalizing:
            if finalization_used < finalization_limit:
                return None
            return {"jump_to": "end"}

        if model_calls_used < model_limit:
            return None

        with trace_span(
            "execution_budget.model_limit",

            kind="chain",

            input_value={
                "model_calls_used": (
                    model_calls_used
                ),

                "model_call_limit": (
                    model_limit
                ),
            },

            attributes={
                "budget.model.used": (
                    model_calls_used
                ),

                "budget.model.limit": (
                    model_limit
                ),

                "budget.model.exhausted": True,
            },
        ) as span:
            set_span_output(
                span,

                {
                    "action": (
                        "jump_to_end"
                    ),

                    "reason": (
                        "executor_model_run_limit"
                    ),
                },
            )

        if self.enable_worker_finalization:
            return {
                "worker_finalize_requested": True,
                "worker_finalize_reason": "MODEL_BUDGET_EXHAUSTED",
            }
        return {"jump_to": "end"}

    @hook_config(can_jump_to=["model"])
    def after_model(
        self,
        state: ExecutionBudgetState,
        runtime,
    ) -> dict[
        str,
        Any,
    ] | None:
        """在Tools节点前裁剪超过动态预算的工具调用。"""

        tool_limit = (
            _read_dynamic_limit(
                state,
                "executor_tool_run_limit",
            )
        )

        show_all_toolsets_limit = (
            _read_dynamic_limit(
                state,
                "show_all_toolsets_run_limit",
            )
        )

        messages = list(
            state.get(
                "messages",
                [],
            )
        )

        if not messages:
            return None

        current_turn = (
            _current_turn_messages(
                messages
            )
        )

        if not current_turn:
            return None

        last_message = (
            current_turn[-1]
        )

        model_calls_used_before = int(
            state.get(
                "executor_model_calls_used",
                0,
            )
        )

        total_used_before = int(
            state.get(
                "executor_tool_calls_used",
                0,
            )
        )

        show_all_toolsets_used_before = int(
            state.get(
                "show_all_toolsets_calls_used",
                0,
            )
        )

        current_tool_calls = (
            _read_ai_tool_calls(
                last_message
            )
        )

        finalizing = bool(state.get("worker_finalize_requested")) and not bool(
            state.get("worker_review_requested")
        )
        if self.enable_worker_finalization and finalizing:
            allowed_tool_calls = [
                tool_call
                for tool_call in current_tool_calls
                if _read_tool_call_name(tool_call) == "submit_for_review"
            ][:1]
            blocked_tool_calls = [
                tool_call
                for tool_call in current_tool_calls
                if tool_call not in allowed_tool_calls
            ]
            updates: dict[str, Any] = {}
            if blocked_tool_calls:
                updates["messages"] = [
                    RemoveMessage(id=REMOVE_ALL_MESSAGES),
                    *messages[:-1],
                    _replace_ai_tool_calls(
                        last_message,
                        tool_calls=allowed_tool_calls,
                        fallback_content=(
                            "Worker finalization only permits submit_for_review."
                        ),
                    ),
                ]
            if state.get("worker_finalize_reason") == "SCHEMA_REPAIR":
                updates["worker_schema_repair_model_calls_used"] = (
                    int(state.get("worker_schema_repair_model_calls_used", 0) or 0)
                    + 1
                )
            return updates or None

        if not current_tool_calls:
            return {
                "executor_model_calls_used": (
                    model_calls_used_before
                    + 1
                ),
            }

        total_remaining = max(
            0,
            tool_limit
            - total_used_before,
        )

        show_all_toolsets_remaining = max(
            0,
            show_all_toolsets_limit
            - show_all_toolsets_used_before,
        )

        allowed_tool_calls: list[
            Any
        ] = []

        blocked_tool_calls: list[
            dict[
                str,
                Any,
            ]
        ] = []

        for tool_call in (
            current_tool_calls
        ):
            tool_name = (
                _read_tool_call_name(
                    tool_call
                )
            )

            if total_remaining <= 0:
                blocked_tool_calls.append(
                    {
                        "tool_name": (
                            tool_name
                        ),

                        "reason": (
                            "executor_tool_run_limit"
                        ),
                    }
                )

                continue

            if (
                tool_name
                == SHOW_ALL_TOOLSETS_NAME
                and show_all_toolsets_remaining
                <= 0
            ):
                blocked_tool_calls.append(
                    {
                        "tool_name": (
                            tool_name
                        ),

                        "reason": (
                            "show_all_toolsets_run_limit"
                        ),
                    }
                )

                continue

            allowed_tool_calls.append(
                tool_call
            )

            total_remaining -= 1

            if (
                tool_name
                == SHOW_ALL_TOOLSETS_NAME
            ):
                show_all_toolsets_remaining -= 1

        allowed_show_all_toolsets_calls = sum(
            1

            for tool_call
            in allowed_tool_calls

            if _read_tool_call_name(
                tool_call
            )
            == SHOW_ALL_TOOLSETS_NAME
        )

        counter_updates = {
            "executor_model_calls_used": (
                model_calls_used_before
                + 1
            ),

            "executor_tool_calls_used": (
                total_used_before
                + len(
                    allowed_tool_calls
                )
            ),

            "show_all_toolsets_calls_used": (
                show_all_toolsets_used_before
                + allowed_show_all_toolsets_calls
            ),
        }

        if not blocked_tool_calls:
            return counter_updates

        fallback_content = (
            "当前Step的工具调用预算已耗尽，"
            "停止继续调用工具。"
        )

        updated_last_message = (
            _replace_ai_tool_calls(
                last_message,

                tool_calls=(
                    allowed_tool_calls
                ),

                fallback_content=(
                    fallback_content
                ),
            )
        )

        # 用REMOVE_ALL_MESSAGES重建消息列表，
        # 不依赖供应商是否为AIMessage分配了id。
        updated_messages = [
            RemoveMessage(
                id=(
                    REMOVE_ALL_MESSAGES
                )
            ),

            *messages[:-1],

            updated_last_message,
        ]

        with trace_span(
            "execution_budget.tool_filter",

            kind="chain",

            input_value={
                "tool_call_limit": (
                    tool_limit
                ),

                "show_all_toolsets_limit": (
                    show_all_toolsets_limit
                ),

                "used_before": (
                    total_used_before
                ),

                "show_all_toolsets_used_before": (
                    show_all_toolsets_used_before
                ),

                "requested_tool_names": [
                    _read_tool_call_name(
                        tool_call
                    )

                    for tool_call
                    in current_tool_calls
                ],
            },

            attributes={
                "budget.tools.limit": (
                    tool_limit
                ),

                "budget.tools.used_before": (
                    total_used_before
                ),

                "budget.tools.allowed_current": len(
                    allowed_tool_calls
                ),

                "budget.tools.blocked_current": len(
                    blocked_tool_calls
                ),

                "budget.show_all_toolsets.limit": (
                    show_all_toolsets_limit
                ),
            },
        ) as span:
            set_span_output(
                span,

                {
                    "allowed_tool_names": [
                        _read_tool_call_name(
                            tool_call
                        )

                        for tool_call
                        in allowed_tool_calls
                    ],

                    "blocked_tool_calls": (
                        blocked_tool_calls
                    ),

                    "all_current_calls_blocked": (
                        not allowed_tool_calls
                    ),
                },
            )

        result = {
            **counter_updates,

            "messages": (
                updated_messages
            ),
        }
        if self.enable_worker_finalization and not allowed_tool_calls:
            result.update(
                {
                    "worker_finalize_requested": True,
                    "worker_finalize_reason": "TOOL_BUDGET_EXHAUSTED",
                    "jump_to": "model",
                }
            )
        return result


class ToolsetRouterMiddleware(
    AgentMiddleware
):
    """按用户Turn缓存工具组路由，并动态限制主模型可见工具。"""

    def __init__(
        self,
        retrieval_models: (
            RetrievalModelManager
        ),
        *,
        baseline_tool_names: Sequence[str] = (),
        minimum_score: float = 0.4,
    ) -> None:
        super().__init__()

        self.toolset_router = (
            ToolsetRouter(
                retrieval_models=retrieval_models,
                minimum_score=minimum_score,
            )
        )

        # 控制面、报告和按需读取工具不参与业务能力竞争。它们先从
        # Cross-Encoder候选池移除，再在结果后确定性加回；后续各工具
        # Middleware仍可执行自己的条件门控。
        self.baseline_tool_names = frozenset(
            str(name).strip()
            for name in baseline_tool_names
            if str(name).strip()
        )

        # 单用户应用：
        # 每个Conversation只保存最近一次路由结果。
        self._route_cache: dict[
            str,
            ToolsetRouteCacheEntry,
        ] = {}

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler,
    ):
        """选择工具组、展开去重工具，再继续调用主模型。"""

        available_tools = list(
            request.tools
        )

        request_state = (
            request.state
            or {}
        )

        tool_limit = (
            _read_dynamic_limit(
                request_state,
                "executor_tool_run_limit",
            )
        )

        tool_calls_used = int(
            request_state.get(
                "executor_tool_calls_used",
                0,
            )
            or 0
        )

        tool_calls_remaining = max(
            0,

            tool_limit
            - tool_calls_used,
        )

        current_messages = list(
            request.messages
        )

        model_round = (
            _infer_model_round(
                current_messages
            )
        )

        (
            user_turn_number,
            routing_task,
            query_source,
        ) = _get_current_toolset_route(
            current_messages
        )

        explicit_step_query = str(
            request_state.get("toolset_route_query") or ""
        ).strip()
        raw_tool_access = request_state.get("step_tool_access", "ENABLED")
        if isinstance(raw_tool_access, bool):
            tool_access_enabled = raw_tool_access
        else:
            normalized_tool_access = str(raw_tool_access or "").strip().upper()
            tool_access_enabled = normalized_tool_access not in {
                "DISABLED", "FALSE", "NO", "N", "0", "OFF",
            }
        fallback_route_query = _compact_tool_selector_text(
            str(request_state.get("toolset_route_fallback_query") or "").strip(),
            max_chars=TOOL_SELECTOR_QUERY_MAX_CHARS,
        )
        full_user_request = str(
            request_state.get("toolset_route_full_user_request")
            or request_state.get("toolset_route_fallback_query")
            or ""
        ).strip()
        if explicit_step_query:
            routing_task = _compact_tool_selector_text(
                explicit_step_query,
                max_chars=TOOL_SELECTOR_QUERY_MAX_CHARS,
            )
            query_source = "scheduler_step_query"

        # Worker在同一个Step里会因为工具结果、运行时提示等新增
        # HumanMessage；这些消息不代表业务路由目标改变。显式的
        # Scheduler Step query已经是稳定路由键，因此不再让消息计数
        # 把同一Step的Cross Encoder缓存击穿。
        route_cache_turn_number = (
            0 if explicit_step_query else user_turn_number
        )

        conversation_key = (
            _get_conversation_key(
                request
            )
        )

        # show_all_toolsets属于控制面工具，
        # 不交给本地Router参与业务工具组展开。
        show_all_toolsets_tool = next(
            (
                current_tool

                for current_tool
                in available_tools

                if (
                    _read_tool_name(
                        current_tool
                    )
                    == SHOW_ALL_TOOLSETS_NAME
                )
            ),

            None,
        )

        always_visible_tools = [
            current_tool
            for current_tool in available_tools
            if _read_tool_name(current_tool) in self.baseline_tool_names
        ]

        business_tools = [
            current_tool

            for current_tool
            in available_tools

            if (
                _read_tool_name(
                    current_tool
                )
                != SHOW_ALL_TOOLSETS_NAME
                and _read_tool_name(current_tool)
                not in self.baseline_tool_names
            )
        ]

        available_tool_items = [
            _tool_to_trace_item(
                current_tool
            )

            for current_tool
            in available_tools
        ]

        available_business_tool_items = [
            _tool_to_trace_item(
                current_tool
            )

            for current_tool
            in business_tools
        ]

        selection_mode = (
            "no_tools"
        )

        cache_hit = False

        cache_reset = False

        router_called = False

        fallback_reason: (
            str
            | None
        ) = None

        selected_toolset_names: list[
            str
        ] = []

        selected_business_tools: list[
            Any
        ] = []

        selected_tools: list[
            Any
        ] = []

        final_visible_tool_names: list[
            str
        ] = []

        selected_request = request

        with trace_span(
            (
                "model_call."
                f"round_{model_round}"
            ),

            kind="chain",

            input_value={
                "conversation_key": (
                    conversation_key
                ),

                "user_turn_number": (
                    user_turn_number
                ),

                "model_round": (
                    model_round
                ),

                "query_source": (
                    query_source
                ),

                "routing_task": (
                    routing_task
                ),
                "step_tool_access": "ENABLED" if tool_access_enabled else "DISABLED",
                "fallback_routing_task": fallback_route_query,

                "available_tools": (
                    available_tool_items
                ),
            },

            attributes={
                "agent.model_round": (
                    model_round
                ),

                "agent.user_turn_number": (
                    user_turn_number
                ),

                "tools.available_count": len(
                    available_tools
                ),

                "tools.query_source": (
                    query_source
                ),
            },
        ) as model_call_span:

            with trace_span(
                "toolset_selection",

                kind="chain",

                input_value={
                    "routing_task": (
                        routing_task
                    ),

                    "query_source": (
                        query_source
                    ),

                    "available_business_tools": (
                        available_business_tool_items
                    ),

                    "control_tool": (
                        _tool_to_trace_item(
                            show_all_toolsets_tool
                        )

                        if show_all_toolsets_tool
                        is not None

                        else None
                    ),
                },
            ) as selection_span:

                if tool_calls_remaining <= 0:
                    # 工具预算耗尽时，不再把任何工具暴露给模型。
                    # 模型仍然正常调用，但只能基于已有信息分析和收口。
                    selection_mode = (
                        "tool_budget_exhausted"
                    )

                    selected_tools = []

                elif not tool_access_enabled:
                    # Scheduler已经声明该Step只依赖当前上下文。报告工具仍需保留，
                    # 使Worker能够提交结果；业务工具和show_all均不挂载。
                    selection_mode = "scheduler_tools_disabled"
                    selected_tools = [
                        current_tool
                        for current_tool in always_visible_tools
                        if _read_tool_name(current_tool) in NO_ACCESS_CONTROL_TOOL_NAMES
                    ]

                elif not available_tools:
                    selection_mode = (
                        "no_tools"
                    )

                    selected_tools = []

                elif not routing_task:
                    selection_mode = "routing_context_missing"
                    fallback_reason = "没有找到有效的工具组路由请求"
                    selected_tools = [
                        current_tool
                        for current_tool in always_visible_tools
                        if _read_tool_name(current_tool) in NO_ACCESS_CONTROL_TOOL_NAMES
                    ]
                    logger.warning(
                        "工具组路由没有找到有效请求，本次只保留提交控制工具。"
                    )

                else:
                    # 工具池发生变化时，
                    # 旧缓存不能继续使用。
                    tool_signature = tuple(
                        sorted(
                            _read_tool_name(
                                current_tool
                            )

                            for current_tool
                            in business_tools
                        )
                    )

                    business_tools_by_name = {
                        _read_tool_name(
                            current_tool
                        ): current_tool

                        for current_tool
                        in business_tools
                    }

                    cached_entry = (
                        self._route_cache
                        .get(
                            conversation_key
                        )
                    )

                    cache_is_usable = (
                        cached_entry is not None

                        and (
                            cached_entry
                            .user_turn_number
                            == route_cache_turn_number
                        )

                        and (
                            cached_entry
                            .routing_task
                            == routing_task
                        )

                        and (
                            cached_entry
                            .fallback_routing_task
                            == fallback_route_query
                        )

                        and cached_entry.tool_access_enabled == tool_access_enabled

                        and (
                            cached_entry
                            .tool_signature
                            == tool_signature
                        )
                    )

                    if cache_is_usable:
                        selected_business_tools = [
                            business_tools_by_name[
                                tool_name
                            ]

                            for tool_name
                            in (
                                cached_entry
                                .selected_tool_names
                            )

                            if tool_name
                            in business_tools_by_name
                        ]

                        # 缓存中的某个工具已经不存在时，
                        # 放弃缓存并重新路由。
                        cache_is_usable = (
                            len(
                                selected_business_tools
                            )
                            == len(
                                cached_entry
                                .selected_tool_names
                            )
                        )

                    if cache_is_usable:
                        cache_hit = True

                        if cached_entry.routing_failed:
                            selection_mode = "turn_cache_model_routing_failed"
                            fallback_reason = (
                                cached_entry.fallback_reason
                                or "本地工具组路由失败（缓存）"
                            )
                            selected_tools = [
                                current_tool
                                for current_tool in always_visible_tools
                                if _read_tool_name(current_tool) in NO_ACCESS_CONTROL_TOOL_NAMES
                            ]
                        else:
                            selection_mode = (
                                "turn_cache"
                            )

                            selected_toolset_names = list(
                                cached_entry
                                .selected_toolset_names
                            )

                    else:
                        router_called = True

                        route_decision = await self.toolset_router.route(
                            task_text=routing_task,
                            available_tools=business_tools,
                            allow_no_tool=False,
                            user_request_text=fallback_route_query,
                            step_weight=TOOLSET_ROUTE_STEP_WEIGHT,
                            user_weight=TOOLSET_ROUTE_USER_WEIGHT,
                        )
                        route_stage = "weighted_cross_encoder"

                        if route_decision is None:
                            route_decision = await self.toolset_router.route_with_model(
                                getattr(request, "model", None),
                                primary_task=routing_task,
                                user_request_window=full_user_request,
                                available_tools=business_tools,
                                allow_no_tool=False,
                            )
                            route_stage = "same_worker_model"

                        if route_decision is None:
                            selection_mode = "model_routing_failed"
                            fallback_reason = "加权本地路由和同Worker模型均未形成合法工具组"

                            selected_tools = [
                                current_tool
                                for current_tool in always_visible_tools
                                if _read_tool_name(current_tool) in NO_ACCESS_CONTROL_TOOL_NAMES
                            ]

                            logger.warning(
                                "本地工具组路由失败，"
                                "本次只保留提交控制工具。"
                            )

                            if (
                                conversation_key not in self._route_cache
                                and len(self._route_cache)
                                >= TOOLSET_ROUTE_CACHE_MAX_CONVERSATIONS
                            ):
                                self._route_cache.clear()
                                cache_reset = True

                            self._route_cache[conversation_key] = (
                                ToolsetRouteCacheEntry(
                                    user_turn_number=route_cache_turn_number,
                                    routing_task=routing_task,
                                    fallback_routing_task=fallback_route_query,
                                    tool_access_enabled=tool_access_enabled,
                                    tool_signature=tool_signature,
                                    selected_toolset_names=(),
                                    # 保存完整业务工具签名，使缓存只在真实工具池
                                    # 完全一致时命中。
                                    selected_tool_names=(),
                                    routing_failed=True,
                                    fallback_reason=fallback_reason,
                                )
                            )

                        else:
                            selection_mode = route_stage

                            selected_toolset_names = list(
                                route_decision
                                .selected_toolset_names
                            )

                            # route_decision.tools已经在
                            # toolset_router.py中按tool.name去重。
                            selected_business_tools = list(
                                route_decision.tools
                            )

                            # 单用户项目不需要复杂LRU。
                            #
                            # Conversation太多时直接清空，
                            # 后续重新积累即可。
                            if (
                                conversation_key
                                not in self._route_cache

                                and len(
                                    self._route_cache
                                )
                                >= (
                                    TOOLSET_ROUTE_CACHE_MAX_CONVERSATIONS
                                )
                            ):
                                self._route_cache.clear()

                                cache_reset = True

                            self._route_cache[
                                conversation_key
                            ] = (
                                ToolsetRouteCacheEntry(
                                    user_turn_number=(
                                        route_cache_turn_number
                                    ),

                                    routing_task=(
                                        routing_task
                                    ),

                                    fallback_routing_task=(
                                        fallback_route_query
                                    ),

                                    tool_access_enabled=tool_access_enabled,

                                    tool_signature=(
                                        tool_signature
                                    ),

                                    selected_toolset_names=tuple(
                                        selected_toolset_names
                                    ),

                                    selected_tool_names=tuple(
                                        _read_tool_name(
                                            current_tool
                                        )

                                        for current_tool
                                        in selected_business_tools
                                    ),
                                )
                            )

                    if fallback_reason is None:
                        selected_tools = list(
                            selected_business_tools
                        )

                        selected_name_set = {
                            _read_tool_name(current_tool)
                            for current_tool in selected_tools
                        }
                        for current_tool in always_visible_tools:
                            current_name = _read_tool_name(current_tool)
                            if current_name not in selected_name_set:
                                selected_tools.append(current_tool)
                                selected_name_set.add(current_name)

                        # 工具组一经选中会由Step级缓存持续复用。Worker不再
                        # 获得动态展开全部工具的入口，避免工具Schema来回变化。

                final_visible_tool_names = [
                    _read_tool_name(
                        current_tool
                    )

                    for current_tool
                    in selected_tools
                ]

                set_span_attributes(
                    selection_span,

                    **{
                        "toolset.selection_mode": (
                            selection_mode
                        ),

                        "toolset.cache_hit": (
                            cache_hit
                        ),

                        "toolset.cache_reset": (
                            cache_reset
                        ),

                        "toolset.router_called": (
                            router_called
                        ),

                        "toolset.cache_turn_number": (
                            route_cache_turn_number
                        ),

                        "toolset.selected_count": len(
                            selected_toolset_names
                        ),

                        "tools.available_business_count": len(
                            business_tools
                        ),

                        "tools.visible_count": len(
                            selected_tools
                        ),

                        "tools.show_all_toolsets_visible": (
                            show_all_toolsets_tool
                            is not None

                            and SHOW_ALL_TOOLSETS_NAME
                            in final_visible_tool_names
                        ),
                    },
                )

                set_span_output(
                    selection_span,

                    {
                        "selection_mode": (
                            selection_mode
                        ),

                        "cache_hit": (
                            cache_hit
                        ),

                        "cache_reset": (
                            cache_reset
                        ),

                        "router_called": (
                            router_called
                        ),

                        "fallback_reason": (
                            fallback_reason
                        ),

                        "selected_toolsets": (
                            selected_toolset_names
                        ),

                        "selected_business_tool_names": [
                            _read_tool_name(
                                current_tool
                            )

                            for current_tool
                            in selected_business_tools
                        ],

                        "show_all_toolsets_always_visible": (
                            show_all_toolsets_tool
                            is not None

                            and SHOW_ALL_TOOLSETS_NAME
                            in final_visible_tool_names
                        ),

                        "final_visible_tool_names": (
                            final_visible_tool_names
                        ),

                        "visibility_stage": (
                            "post_route_pre_tool_gates"
                        ),
                    },
                )

            # 工具列表未变化时避免创建多余request对象。
            if (
                selected_tools
                == available_tools
            ):
                selected_request = request

            else:
                selected_request = (
                    request.override(
                        tools=(
                            selected_tools
                        )
                    )
                )

            set_span_attributes(
                model_call_span,

                **{
                    "toolset.selection_mode": (
                        selection_mode
                    ),

                    "toolset.cache_hit": (
                        cache_hit
                    ),

                    "toolset.router_called": (
                        router_called
                    ),

                    "tools.visible_count": len(
                        selected_tools
                    ),
                },
            )

            # handler才是真正调用主模型。
            #
            # LangChain生成的ChatModel Span
            # 会成为当前model_call节点的子节点。
            model_response = await handler(
                selected_request
            )

            set_span_output(
                model_call_span,

                {
                    "model_round": (
                        model_round
                    ),

                    "routing_task": (
                        routing_task
                    ),

                    "query_source": (
                        query_source
                    ),

                    "selection_mode": (
                        selection_mode
                    ),

                    "selected_toolsets": (
                        selected_toolset_names
                    ),

                    "final_visible_tool_names": (
                        final_visible_tool_names
                    ),

                    "model_response_type": (
                        type(
                            model_response
                        ).__name__
                    ),
                },
            )

            return model_response
def _is_bash_usable(
    candidate: Path,
) -> bool:
    """确认目标文件能够作为 Bash 正常执行。"""

    if not candidate.is_file():
        return False

    try:
        result = subprocess.run(
            [
                str(candidate),
                "--noprofile",
                "--norc",
                "-lc",
                "printf agent-bash-ok",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

    except (
        OSError,
        subprocess.SubprocessError,
    ):
        return False

    return (
        result.returncode == 0
        and "agent-bash-ok" in result.stdout
    )


def _find_git_bash() -> str | None:
    """查找可用的 Git Bash；找不到时返回 None。"""

    candidates: list[Path] = []

    # 用户在 .env 中填写的路径拥有最高优先级。
    configured_path = os.getenv(
        "BASH_PATH",
        "",
    ).strip().strip('"')

    if configured_path:
        candidates.append(
            Path(configured_path)
        )

    # Linux 和 macOS 通常已经安装了 Bash。
    if os.name != "nt":
        bash_command = shutil.which(
            "bash"
        )

        if bash_command:
            candidates.append(
                Path(bash_command)
            )

    # 如果 Git 已经加入 PATH，
    # 尝试根据 git.exe 的位置推导 Git Bash。
    git_command = shutil.which(
        "git"
    )

    if git_command:
        git_path = Path(
            git_command
        ).resolve()

        # 常见结构：
        # C:\Program Files\Git\cmd\git.exe
        # C:\Program Files\Git\bin\bash.exe
        git_root = git_path.parent.parent

        candidates.extend(
            [
                git_root / "bin" / "bash.exe",
                git_root / "usr" / "bin" / "bash.exe",
            ]
        )

    # Git for Windows 常见安装位置。
    candidates.extend(
        [
            Path(
                r"C:\Program Files\Git\bin\bash.exe"
            ),
            Path(
                r"C:\Program Files\Git\usr\bin\bash.exe"
            ),
            Path(
                r"C:\Program Files (x86)\Git\bin\bash.exe"
            ),
        ]
    )

    # Git 也可能只为当前 Windows 用户安装。
    local_app_data = os.getenv(
        "LOCALAPPDATA",
        "",
    ).strip()

    if local_app_data:
        candidates.append(
            Path(local_app_data)
            / "Programs"
            / "Git"
            / "bin"
            / "bash.exe"
        )

    # 避免重复检查同一个路径。
    checked_paths: set[Path] = set()

    for candidate in candidates:
        candidate = candidate.expanduser()

        if candidate in checked_paths:
            continue

        checked_paths.add(
            candidate
        )

        if _is_bash_usable(candidate):
            return str(
                candidate.resolve()
            )

    return None

def is_shell_tool_available() -> bool:
    """判断当前启动环境是否会注册Shell工具。

    Hard能力目录和Simple Agent使用同一个检测逻辑，
    避免一边认为Shell存在、另一边实际没有注册。
    """

    return (
        _find_git_bash()
        is not None
    )
def _build_shell_environment() -> dict[str, str]:
    """创建 Shell 所需环境，但不直接暴露全部环境变量。"""

    allowed_names = (
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "LOCALAPPDATA",
        "APPDATA",
    )

    environment = {
        name: os.environ[name]
        for name in allowed_names
        if os.environ.get(name)
    }

    # 确保 shell 中的 python 指向当前 Agent 使用的
    # Anaconda 环境，而不是其他 Python。
    python_directory = str(
        Path(sys.executable)
        .resolve()
        .parent
    )

    scripts_directory = str(
        Path(sys.prefix)
        .resolve()
        / "Scripts"
    )

    original_path = environment.get(
        "PATH",
        "",
    )

    environment["PATH"] = os.pathsep.join(
        path
        for path in (
            python_directory,
            scripts_directory,
            original_path,
        )
        if path
    )

    user_profile = environment.get(
        "USERPROFILE"
    )

    if user_profile:
        environment["HOME"] = user_profile

    environment.update(
        {
            # Python 输出统一使用 UTF-8。
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",

            # 防止 Git 等命令打开分页器后一直等待输入。
            "GIT_PAGER": "cat",
            "PAGER": "cat",
        }
    )

    return environment


def build_middlewares(
    retrieval_models: (
        RetrievalModelManager
        | None
    ) = None,
) -> list:
    """创建当前Agent使用的执行层middleware。"""

    # 动态预算必须始终存在，
    # 并放在执行层Middleware的最外侧。
    resolved_middlewares = [
        DynamicExecutionBudgetMiddleware(),
    ]

    git_bash_path = (
        _find_git_bash()
    )

    if git_bash_path is None:
        logger.warning(
            "未检测到 Git for Windows / Git Bash，"
            "Shell 工具暂时无法使用；"
            "其他功能不受影响。\n"
            "请在 PowerShell 中执行以下命令安装 Git：\n%s\n"
            "安装完成后，请重新启动本项目。",
            GIT_INSTALL_COMMAND,
        )

    else:
        logger.info(
            "已启用 Shell 工具 | "
            "Git Bash=%s",
            git_bash_path,
        )

        shell_command = [
            git_bash_path,
            "--noprofile",
            "--norc",
        ]

        resolved_middlewares.extend(
            [
                ShellToolMiddleware(
                    workspace_root=(
                        WORKSPACE_ROOT
                    ),

                    shell_command=(
                        shell_command
                    ),

                    env=(
                        _build_shell_environment()
                    ),

                    execution_policy=(
                        HostExecutionPolicy()
                    ),

                    tool_description=(
                        "在本机持续存在的"
                        "Git Bash会话中执行命令。"
                        "默认工作目录是项目workspace。"
                        "适合运行Python、Git、pytest"
                        "和其他开发命令。"
                        "必须根据命令的真实输出"
                        "判断是否成功。"
                        "执行具有明显副作用的操作前，"
                        "应先向用户说明。"
                    ),
                ),

                ToolCallLimitMiddleware(
                    tool_name="shell",

                    run_limit=5,

                    exit_behavior=(
                        "continue"
                    ),
                ),
            ]
        )

    if retrieval_models is None:
        logger.warning(
            "没有向build_middlewares传入"
            "RetrievalModelManager，"
            "本次不会启用动态工具组路由。"
        )

    else:
        resolved_middlewares.append(
            ToolsetRouterMiddleware(
                retrieval_models=(
                    retrieval_models
                ),
            )
        )
    return resolved_middlewares
