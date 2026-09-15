import logging

from typing import (
    Any,
)

from langchain.agents import (
    AgentState,
)

from langchain.agents.middleware import (
    before_model,
)

from langchain.messages import (
    RemoveMessage,
)

from langgraph.graph.message import (
    REMOVE_ALL_MESSAGES,
)
logger = logging.getLogger(
    "agent"
)
from observability import (
    set_span_output,
    trace_span,
)
from prompt_loader import (
    render_prompt,
)
def _read_message_role(
    message: Any,
) -> str:
    """统一读取LangChain消息的角色。"""

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

def _read_ai_tool_call_ids(
    message: Any,
) -> list[str]:
    """读取AIMessage声明的全部tool_call_id。"""

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

    tool_call_ids: list[
        str
    ] = []

    for tool_call in raw_tool_calls:
        if isinstance(
            tool_call,
            dict,
        ):
            tool_call_id = (
                tool_call.get(
                    "id"
                )
                or tool_call.get(
                    "tool_call_id"
                )
                or ""
            )

        else:
            tool_call_id = (
                getattr(
                    tool_call,
                    "id",
                    "",
                )
                or getattr(
                    tool_call,
                    "tool_call_id",
                    "",
                )
                or ""
            )

        normalized_id = str(
            tool_call_id
        ).strip()

        if normalized_id:
            tool_call_ids.append(
                normalized_id
            )

    return tool_call_ids


def _read_tool_message_call_id(
    message: Any,
) -> str:
    """读取ToolMessage响应的tool_call_id。"""

    if isinstance(
        message,
        dict,
    ):
        tool_call_id = (
            message.get(
                "tool_call_id"
            )
            or ""
        )

    else:
        tool_call_id = (
            getattr(
                message,
                "tool_call_id",
                "",
            )
            or ""
        )

    return str(
        tool_call_id
    ).strip()


def _repair_tool_message_history(
    messages: list[
        Any
    ],
) -> tuple[
    list[Any],
    list[dict[str, Any]],
]:
    """删除不完整的工具调用消息块。

    合法消息块必须是：

    AIMessage(tool_calls=[A, B])
        ↓
    ToolMessage(tool_call_id=A)
    ToolMessage(tool_call_id=B)

    如果缺少结果、ID不匹配、重复响应，
    就删除整组AI工具调用和紧随其后的ToolMessage。

    不伪造工具执行结果。
    后面的用户消息和正常对话仍然保留。
    """

    repaired_messages: list[
        Any
    ] = []

    repair_issues: list[
        dict[
            str,
            Any,
        ]
    ] = []

    message_index = 0

    while (
        message_index
        < len(
            messages
        )
    ):
        current_message = (
            messages[
                message_index
            ]
        )

        current_role = (
            _read_message_role(
                current_message
            )
        )

        expected_tool_call_ids = (
            _read_ai_tool_call_ids(
                current_message
            )
        )

        # AI消息声明了工具调用。
        if (
            current_role
            in {
                "assistant",
                "ai",
            }

            and expected_tool_call_ids
        ):
            next_index = (
                message_index
                + 1
            )

            tool_messages: list[
                Any
            ] = []

            actual_tool_call_ids: list[
                str
            ] = []

            # 工具结果必须紧跟在AI工具调用消息后。
            while (
                next_index
                < len(
                    messages
                )

                and _read_message_role(
                    messages[
                        next_index
                    ]
                )
                == "tool"
            ):
                tool_message = (
                    messages[
                        next_index
                    ]
                )

                tool_messages.append(
                    tool_message
                )

                actual_tool_call_ids.append(
                    _read_tool_message_call_id(
                        tool_message
                    )
                )

                next_index += 1

            # sorted同时检查：
            #
            # - 数量；
            # - ID；
            # - 重复响应。
            tool_block_is_valid = (
                bool(
                    actual_tool_call_ids
                )

                and all(
                    actual_tool_call_ids
                )

                and sorted(
                    actual_tool_call_ids
                )
                == sorted(
                    expected_tool_call_ids
                )
            )

            if tool_block_is_valid:
                repaired_messages.append(
                    current_message
                )

                repaired_messages.extend(
                    tool_messages
                )

            else:
                repair_issues.append(
                    {
                        "message_index": (
                            message_index
                        ),

                        "expected_tool_call_ids": (
                            expected_tool_call_ids
                        ),

                        "actual_tool_call_ids": (
                            actual_tool_call_ids
                        ),

                        "removed_message_count": (
                            1
                            + len(
                                tool_messages
                            )
                        ),
                    }
                )

            message_index = (
                next_index
            )

            continue

        # 没有前置AI工具调用的ToolMessage
        # 本身也是非法的孤立结果。
        if current_role == "tool":
            repair_issues.append(
                {
                    "message_index": (
                        message_index
                    ),

                    "expected_tool_call_ids": [],

                    "actual_tool_call_ids": [
                        _read_tool_message_call_id(
                            current_message
                        )
                    ],

                    "removed_message_count": 1,

                    "reason": (
                        "orphan_tool_message"
                    ),
                }
            )

            message_index += 1

            continue

        repaired_messages.append(
            current_message
        )

        message_index += 1

    return (
        repaired_messages,
        repair_issues,
    )

@before_model
def repair_incomplete_tool_history(
    state: AgentState,
    runtime,
) -> dict[
    str,
    Any,
] | None:
    """在主模型调用前修复不完整工具消息链。"""

    original_messages = list(
        state.get(
            "messages",
            [],
        )
    )

    if not original_messages:
        return None

    (
        repaired_messages,
        repair_issues,
    ) = _repair_tool_message_history(
        original_messages
    )

    if not repair_issues:
        return None

    removed_message_count = (
        len(
            original_messages
        )
        - len(
            repaired_messages
        )
    )

    logger.warning(
        "检测到不完整工具消息链，"
        "已在调用主模型前自动修复 | "
        "original=%s | repaired=%s | "
        "removed=%s | issue_count=%s",

        len(
            original_messages
        ),

        len(
            repaired_messages
        ),

        removed_message_count,

        len(
            repair_issues
        ),
    )

    with trace_span(
        "message_history.repair",

        kind="chain",

        input_value={
            "original_message_count": (
                len(
                    original_messages
                )
            ),

            "issues": (
                repair_issues
            ),
        },

        attributes={
            "history.original_count": (
                len(
                    original_messages
                )
            ),

            "history.repaired_count": (
                len(
                    repaired_messages
                )
            ),

            "history.removed_count": (
                removed_message_count
            ),

            "history.issue_count": (
                len(
                    repair_issues
                )
            ),
        },
    ) as span:

        set_span_output(
            span,

            {
                "status": (
                    "repaired"
                ),

                "removed_message_count": (
                    removed_message_count
                ),

                "issues": (
                    repair_issues
                ),
            },
        )

    # 使用REMOVE_ALL_MESSAGES后重新加入合法消息。
    #
    # 这不只是临时修改本次ModelRequest，
    # 还会通过messages reducer写回当前Thread状态，
    # 因此下次恢复Checkpoint时也是干净的。
    return {
        "messages": [
            RemoveMessage(
                id=(
                    REMOVE_ALL_MESSAGES
                )
            ),

            *repaired_messages,
        ]
    }

def _infer_model_round(
    messages: list,
) -> int:
    """判断当前用户Turn中的主模型调用轮次。

    例如：

    用户消息
        ↓
    模型第一次调用工具
        ↓
    工具返回结果
        ↓
    模型第二次调用工具
        ↓
    工具返回结果
        ↓
    模型第三次生成最终回答

    对应返回：
    1、2、3。
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
        role = _read_message_role(
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

    previous_model_calls = sum(
        1
        for message
        in messages[
            latest_user_index + 1:
        ]
        if _read_message_role(
            message
        )
        in {
            "assistant",
            "ai",
        }
    )

    return (
        previous_model_calls
        + 1
    )

def _message_content_to_text(
    content: Any,
) -> str:
    """把不同模型消息内容转换成普通文字。"""

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


def _format_messages_for_summary(
    messages: list[
        Any
    ],
) -> str:
    """把Conversation消息转换成摘要模型可读文字。

    原Conversation理论上只包含用户消息和最终助手回答。
    这里仍然忽略ToolMessage，避免把内部工具轨迹重复写入
    跨Turn摘要。
    """

    lines: list[str] = []

    for message in messages:
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

        role = _read_message_role(
            message
        )

        if role in {
            "user",
            "human",
        }:
            role_name = "用户"

        elif role in {
            "assistant",
            "ai",
        }:
            role_name = "助手"

        else:
            continue

        message_text = (
            _message_content_to_text(
                content
            )
        )

        if not message_text:
            continue

        lines.append(
            f"{role_name}：{message_text}"
        )

    return "\n\n".join(
        lines
    )


def _compact_summary_text(
    text: str,
    max_chars: int,
) -> str:
    """在模型未遵守长度要求时确定性压缩摘要。"""

    normalized_text = text.strip()

    if len(
        normalized_text
    ) <= max_chars:
        return normalized_text

    marker = (
        "\n\n"
        "……摘要中部因长度限制已省略……"
        "\n\n"
    )

    available_chars = max(
        0,
        max_chars
        - len(
            marker
        ),
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


async def summarize_conversation_history(
    model,
    *,
    previous_summary: str,
    messages: list[
        Any
    ],
    max_chars: int,
) -> str:
    """更新Hard节点共用的Conversation Rolling Summary。

    这个函数不是Agent Middleware：

    - 由ConversationRuntime在一次Planning Run开始前调用；
    - 使用conversation/summary提示词整理历史进展；
    - 把旧摘要和新进入摘要区的消息合并压缩；
    - 失败时保留旧摘要，并附带确定性压缩的新消息，
      避免因为一次模型异常丢失Conversation上下文。
    """

    if max_chars < 1:
        raise ValueError(
            "max_chars不能小于1。"
        )

    normalized_previous_summary = (
        previous_summary.strip()
    )

    new_history_text = (
        _format_messages_for_summary(
            messages
        )
    )

    if not new_history_text:
        return _compact_summary_text(
            normalized_previous_summary,
            max_chars,
        )

    source_blocks: list[str] = []

    if normalized_previous_summary:
        source_blocks.append(
            "[已有Rolling Summary]\n"
            f"{normalized_previous_summary}"
        )

    source_blocks.append(
        "[本次新增的较早对话]\n"
        f"{new_history_text}"
    )

    source_text = "\n\n".join(
        source_blocks
    )

    summary_prompt = (
        render_prompt("conversation/summary")
        + "\n\n"
        + (
            "最终摘要总长度不得超过"
            f"{max_chars}个字符。"
        )
    )

    with trace_span(
        "conversation_summary.update",

        kind="chain",

        input_value={
            "previous_summary_chars": len(
                normalized_previous_summary
            ),

            "new_message_count": len(
                messages
            ),

            "new_history_chars": len(
                new_history_text
            ),

            "max_summary_chars": (
                max_chars
            ),
        },
    ) as span:
        try:
            response = await model.ainvoke(
                [
                    {
                        "role": "system",
                        "content": summary_prompt,
                    },
                    {"role": "user", "content": source_text},
                ]
            )

            summary_text = (
                _message_content_to_text(
                    getattr(
                        response,
                        "content",
                        "",
                    )
                )
            )

            if not summary_text:
                raise RuntimeError(
                    "摘要模型没有返回文字内容。"
                )

            final_summary = (
                _compact_summary_text(
                    summary_text,
                    max_chars,
                )
            )

            set_span_output(
                span,

                {
                    "status": "success",
                    "summary": final_summary,
                    "summary_chars": len(
                        final_summary
                    ),
                },
            )

            return final_summary

        except Exception as error:
            logger.warning(
                "Conversation摘要更新失败，"
                "已保留可恢复的确定性摘要 | "
                "error=%s: %s",

                type(error).__name__,
                error,
            )

            fallback_source = "\n\n".join(
                source_blocks
            )

            fallback_summary = (
                _compact_summary_text(
                    fallback_source,
                    max_chars,
                )
            )

            set_span_output(
                span,

                {
                    "status": "fallback",
                    "error": (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                    "summary": (
                        fallback_summary
                    ),
                    "summary_chars": len(
                        fallback_summary
                    ),
                },
            )

            return fallback_summary
