
from typing import Any

from langchain.agents import (
    AgentState,
    create_agent,
)

from langchain.chat_models import (
    init_chat_model,
)
from typing_extensions import (
    NotRequired,
)
from config import (
    PlanningSettings,
    Settings,
)
from context_middlewares import (
    build_context_middlewares,
)
from middlewares import (
    build_middlewares,
)
from prompt_loader import (
    load_prompt,
)
from observability import (
    set_span_attributes,
    set_span_output,
    trace_span,
)

TRACE_MESSAGE_PREVIEW_MAX_CHARS = 1200

class ConversationState(
    AgentState
):
    """单条Conversation需要持久化的额外状态。"""

    conversation_id: NotRequired[
        str
    ]

    conversation_title: NotRequired[
        str
    ]

    channel_key: NotRequired[
        str
    ]

    created_at: NotRequired[
        str
    ]

    last_active_at: NotRequired[
        str
    ]

    title_generated: NotRequired[
        bool
    ]
    memory_context: NotRequired[
        str
    ]
    conversation_summary: NotRequired[
        str
    ]

    conversation_summary_message_count: NotRequired[
        int
    ]
    # Planning Graph为当前Step Attempt计算的动态预算。
    #
    # 这些字段由执行层Middleware读取，
    # 不由模型直接决定。
    executor_model_run_limit: NotRequired[
        int
    ]

    executor_tool_run_limit: NotRequired[
        int
    ]

    request_toolset_run_limit: NotRequired[
        int
    ]


def _build_chat_model(
    *,
    model_provider: str,
    model_name: str,
    max_tokens: int,
    timeout_seconds: int,
    thinking_enabled: bool,
):
    """统一构造普通模型和Hard模型。

    当前只有DeepSeek需要通过extra_body
    显式控制thinking模式。

    其他供应商暂时只复用：
    - model
    - model_provider
    - max_tokens
    - timeout
    - max_retries

    后续如果OpenAI或Anthropic需要额外的
    reasoning参数，应继续在这个函数中集中适配，
    不要分别散落到Supervisor和Executor中。
    """

    normalized_provider = (
        model_provider
        .strip()
        .lower()
    )

    normalized_model_name = (
        model_name
        .strip()
    )

    if not normalized_provider:
        raise ValueError(
            "model_provider不能为空。"
        )

    if not normalized_model_name:
        raise ValueError(
            "model_name不能为空。"
        )

    model_options: dict[
        str,
        Any,
    ] = {
        "model": (
            normalized_model_name
        ),

        "model_provider": (
            normalized_provider
        ),

        "max_tokens": (
            max_tokens
        ),

        "timeout": (
            timeout_seconds
        ),

        # 这里属于供应商请求失败后的
        # 传输层重试。
        #
        # 它不属于新的业务模型轮次，
        # 后续不计入Supervisor或Executor预算。
        "max_retries": 2,
    }

    if normalized_provider == "deepseek":
        model_options[
            "extra_body"
        ] = {
            "thinking": {
                "type": (
                    "enabled"

                    if thinking_enabled

                    else "disabled"
                ),
            }
        }

    return init_chat_model(
        **model_options,
    )

def build_model(
    settings: Settings,
):
    """构造Simple Executor使用的普通云端模型。

    当前演示项目中的以下调用共用这个模型：

    - Simple Executor；
    - Step Reporter；
    - Conversation Summary；
    - Conversation Title；
    - 云端记忆提取和判断。

    普通云端模型默认开启Thinking。

    Thinking不会被当成额外业务模型轮次：
    一次模型请求无论内部思考多长，
    仍然只计算一次逻辑模型调用。
    """

    return _build_chat_model(
        model_provider=(
            settings.llm_provider
        ),

        model_name=(
            settings.llm_model
        ),

        max_tokens=(
            settings
            .cloud_llm_max_tokens
        ),

        # 输出上限扩大后，
        # 给Thinking和较长结构化输出更多时间。
        timeout_seconds=120,

        thinking_enabled=True,
    )

def build_hard_model(
    settings: Settings,
):
    """构造Hard Supervisor系列节点使用的模型。

    这个模型负责：

    - 初始Hard Supervisor；
    - 全局唯一一次Hard Replanner；
    - Hard Final Reviewer。

    当前演示项目不再为三个Hard职责
    分别创建不同的模型配置，
    它们统一使用全局云端输出上限。
    """

    return _build_chat_model(
        model_provider=(
            settings.hard_llm_provider
        ),

        model_name=(
            settings.hard_llm_model
        ),

        max_tokens=(
            settings
            .cloud_llm_max_tokens
        ),

        # Final Reviewer可能生成较完整的最终回答，
        # Hard Thinking也可能需要更长时间。
        timeout_seconds=180,

        thinking_enabled=True,
    )

def build_agent(
    model,
    *,
    planning: PlanningSettings,
    tools: list | None = None,
    middleware: list | None = None,
    checkpointer=None,
    store=None,
    retrieval_models=None,
):
    """组装唯一的Simple Executor Agent。

    所有用户请求先经过Hard Supervisor。

    Hard Supervisor输出FINAL时，
    不会调用这个Agent。

    只有输出PLAN时，
    Planning Graph才会使用这个Agent
    执行当前Step和调用业务工具。
    """

    context_middlewares = (
        build_context_middlewares(
            model,
            planning,
        )
    )

    execution_middlewares = (
        build_middlewares(
            retrieval_models=(
                retrieval_models
            )
        )

        if middleware is None

        else list(
            middleware
        )
    )

    # 动态模型/工具预算不再在这里写死。
    #
    # Planning Graph会把当前Attempt的剩余额度写入
    # ConversationState；执行层Middleware再按本次State
    # 实际限制模型和工具调用。
    resolved_middlewares = [
        *context_middlewares,
        *execution_middlewares,
    ]

    return create_agent(
        model=model,

        tools=list(
            tools
            or []
        ),

        middleware=(
            resolved_middlewares
        ),

        system_prompt=load_prompt(
            "assistant"
        ),

        state_schema=(
            ConversationState
        ),

        checkpointer=(
            checkpointer
        ),

        store=store,
    )

def _content_to_text(
    content,
) -> str:
    """把不同模型的消息内容统一转换成字符串。"""

    if isinstance(
        content,
        str,
    ):
        return content

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

            elif isinstance(
                item,
                dict,
            ):
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
        )

    return str(
        content
    )

def _compact_trace_text(
    text: str,
) -> str:
    """压缩Agent根节点中的消息预览。

    完整模型输入、工具参数和工具结果
    由Phoenix中的自动子Span负责保存。

    Agent根节点只保存便于快速查看的时间线摘要。
    """

    normalized_text = (
        text.strip()
    )

    if (
        len(
            normalized_text
        )
        <= TRACE_MESSAGE_PREVIEW_MAX_CHARS
    ):
        return normalized_text

    removed_chars = (
        len(
            normalized_text
        )
        - TRACE_MESSAGE_PREVIEW_MAX_CHARS
    )

    return (
        normalized_text[
            :TRACE_MESSAGE_PREVIEW_MAX_CHARS
        ]
        + "\n"
        + (
            f"... 根节点预览已省略"
            f"{removed_chars}个字符；"
            "完整内容请展开对应子Span。"
        )
    )


def _read_message_role(
    message: Any,
) -> str:
    """统一读取消息角色。"""

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


def _read_message_name(
    message: Any,
) -> str:
    """统一读取ToolMessage中的工具名称。"""

    if isinstance(
        message,
        dict,
    ):
        name = (
            message.get(
                "name"
            )
            or ""
        )

    else:
        name = (
            getattr(
                message,
                "name",
                "",
            )
            or ""
        )

    return str(
        name
    ).strip()


def _read_tool_call_id(
    message: Any,
) -> str:
    """统一读取ToolMessage关联的tool_call_id。"""

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


def _normalize_tool_call(
    tool_call: Any,
) -> dict[
    str,
    Any,
]:
    """把不同格式的工具调用转换成统一结构。"""

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

        tool_arguments = (
            tool_call.get(
                "args"
            )
        )

        if tool_arguments is None:
            tool_arguments = (
                tool_call.get(
                    "arguments"
                )
            )

        if tool_arguments is None:
            tool_arguments = (
                function_data.get(
                    "arguments"
                )
            )

        tool_call_id = (
            tool_call.get(
                "id"
            )
            or tool_call.get(
                "tool_call_id"
            )
            or ""
        )

        tool_call_type = (
            tool_call.get(
                "type"
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

        tool_arguments = (
            getattr(
                tool_call,
                "args",
                None,
            )
        )

        if tool_arguments is None:
            tool_arguments = (
                getattr(
                    tool_call,
                    "arguments",
                    None,
                )
            )

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

        tool_call_type = (
            getattr(
                tool_call,
                "type",
                "",
            )
            or ""
        )

    return {
        "tool_call_id": str(
            tool_call_id
        ),

        "tool_name": str(
            tool_name
        ),

        "arguments": (
            tool_arguments
        ),

        "type": str(
            tool_call_type
        ),
    }


def _message_to_trace_item(
    message: Any,
) -> dict[
    str,
    Any,
]:
    """把一条LangChain消息转换成简洁的Trace结构。"""

    if isinstance(
        message,
        dict,
    ):
        raw_content = (
            message.get(
                "content",
                "",
            )
        )

        raw_tool_calls = (
            message.get(
                "tool_calls"
            )
            or []
        )

    else:
        raw_content = (
            getattr(
                message,
                "content",
                "",
            )
        )

        raw_tool_calls = (
            getattr(
                message,
                "tool_calls",
                None,
            )
            or []
        )

    content_text = (
        _content_to_text(
            raw_content
        )
    )

    item: dict[
        str,
        Any,
    ] = {
        "role": (
            _read_message_role(
                message
            )
        ),

        "content_preview": (
            _compact_trace_text(
                content_text
            )
        ),

        "content_chars": len(
            content_text
        ),
    }

    message_name = (
        _read_message_name(
            message
        )
    )

    if message_name:
        item[
            "tool_name"
        ] = message_name

    tool_call_id = (
        _read_tool_call_id(
            message
        )
    )

    if tool_call_id:
        item[
            "tool_call_id"
        ] = tool_call_id

    if raw_tool_calls:
        item[
            "tool_calls"
        ] = [
            _normalize_tool_call(
                tool_call
            )

            for tool_call
            in raw_tool_calls
        ]

    return item


def _extract_current_turn_messages(
    messages: list,
) -> list:
    """只截取当前用户Turn产生的消息。

    agent.ainvoke返回的messages通常包含
    当前Conversation之前积累的全部历史。

    Phoenix根节点不应每轮重复保存全部旧历史，
    因此从最后一条用户消息开始截取。
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
        return list(
            messages
        )

    return list(
        messages[
            latest_user_index:
        ]
    )


def _build_agent_execution_summary(
    messages: list,
) -> dict[
    str,
    Any,
]:
    """生成当前Agent Turn的结构化执行摘要。"""

    current_turn_messages = (
        _extract_current_turn_messages(
            messages
        )
    )

    timeline: list[
        dict[
            str,
            Any,
        ]
    ] = []

    model_round = 0
    tool_result_count = 0
    tool_call_count = 0

    for (
        sequence_number,
        message,
    ) in enumerate(
        current_turn_messages,
        start=1,
    ):
        trace_item = (
            _message_to_trace_item(
                message
            )
        )

        role = trace_item.get(
            "role",
            "",
        )

        trace_item[
            "sequence"
        ] = sequence_number

        if role in {
            "assistant",
            "ai",
        }:
            model_round += 1

            trace_item[
                "model_round"
            ] = model_round

            tool_calls = (
                trace_item.get(
                    "tool_calls",
                    [],
                )
            )

            if isinstance(
                tool_calls,
                list,
            ):
                tool_call_count += len(
                    tool_calls
                )

        elif role == "tool":
            tool_result_count += 1

            trace_item[
                "tool_result_number"
            ] = tool_result_count

        timeline.append(
            trace_item
        )

    return {
        "current_turn_message_count": len(
            current_turn_messages
        ),

        "model_call_count": (
            model_round
        ),

        "tool_call_count": (
            tool_call_count
        ),

        "tool_result_count": (
            tool_result_count
        ),

        "timeline": (
            timeline
        ),
    }
def _fallback_title(
    user_text: str,
) -> str:
    """标题模型失败时，从用户首条消息生成备用标题。"""

    normalized = "".join(
        user_text
        .strip()
        .split()
    )

    if not normalized:
        return "新对话"

    return normalized[:10]


async def generate_conversation_title(
    model,
    user_text: str,
    assistant_text: str,
) -> str:
    """使用普通模型生成十字以内的Conversation标题。"""

    model_user_content = (
        "用户首条消息：\n"
        f"{user_text[:600]}"
        "\n\n"
        "助手首轮回答：\n"
        f"{assistant_text[:600]}"
    )

    with trace_span(
        "conversation_title_generation",

        # 这一层负责准备标题任务、
        # 调用模型并规范化标题。
        #
        # 真正的模型调用会由自动Instrumentation
        # 生成一个LLM子Span。
        kind="chain",

        input_value={
            "user_message": (
                user_text[:600]
            ),

            "assistant_reply": (
                assistant_text[:600]
            ),

            "title_max_chars": 10,
        },

        attributes={
            "agent.operation": (
                "conversation_title"
            ),

            "conversation.title_max_chars": (
                10
            ),
        },
    ) as span:

        response = await model.ainvoke(
            [
                {
                    "role": "system",

                    "content": load_prompt(
                        "conversation_title"
                    ),
                },
                {
                    "role": "user",

                    "content": (
                        model_user_content
                    ),
                },
            ]
        )

        raw_title = (
            _content_to_text(
                response.content
            )
            .strip()
        )

        normalized_title = (
            raw_title
        )

        if normalized_title:
            normalized_title = (
                normalized_title
                .splitlines()[0]
            )

        for prefix in (
            "标题：",
            "标题:",
        ):
            if normalized_title.startswith(
                prefix
            ):
                normalized_title = (
                    normalized_title[
                        len(prefix):
                    ]
                )

        normalized_title = (
            normalized_title.strip(
                " \t\r\n"
                "\"'“”‘’"
                "《》【】"
                "。."
            )
        )

        normalized_title = "".join(
            normalized_title.split()
        )

        fallback_used = (
            not bool(
                normalized_title
            )
        )

        if fallback_used:
            final_title = (
                _fallback_title(
                    user_text
                )
            )

        else:
            final_title = (
                normalized_title[:10]
            )

        set_span_attributes(
            span,

            **{
                "conversation.title_fallback_used": (
                    fallback_used
                ),

                "conversation.title_chars": len(
                    final_title
                ),
            },
        )

        set_span_output(
            span,

            {
                "raw_model_output": (
                    raw_title
                ),

                "normalized_title": (
                    normalized_title
                ),

                "fallback_used": (
                    fallback_used
                ),

                "final_title": (
                    final_title
                ),
            },
        )

        return final_title


async def ask_agent(
    agent,
    user_text: str,
    *,
    thread_id: str,
    state_update: dict[
        str,
        Any,
    ] | None = None,
    return_details: bool = False,
) -> str | dict[
    str,
    Any,
]:
    """调用主Agent，并记录完整执行根节点。

    默认只返回最终文字，保持现有调用兼容。

    return_details为True时，
    额外返回当前Turn消息和执行统计，
    供后续StepReport使用。
    """

    input_state: dict[
        str,
        Any,
    ] = {
        "messages": [
            {
                "role": "user",

                "content": (
                    user_text
                ),
            }
        ]
    }

    resolved_state_update = (
        state_update
        or {}
    )

    if resolved_state_update:
        input_state.update(
            resolved_state_update
        )

    memory_context = (
        resolved_state_update.get(
            "memory_context",
            "",
        )
    )

    if not isinstance(
        memory_context,
        str,
    ):
        memory_context = ""

    with trace_span(
        "main_agent.run",

        # AGENT表示一个由LLM驱动、
        # 可以循环调用工具的推理单元。
        kind="agent",

        input_value={
            "thread_id": (
                thread_id
            ),

            "user_message": (
                user_text
            ),

            "state_update_summary": {
                "keys": sorted(
                    resolved_state_update
                    .keys()
                ),

                "memory_context_present": bool(
                    memory_context
                    .strip()
                ),

                "memory_context_chars": len(
                    memory_context
                ),

                "executor_model_run_limit": (
                    resolved_state_update.get(
                        "executor_model_run_limit"
                    )
                ),

                "executor_tool_run_limit": (
                    resolved_state_update.get(
                        "executor_tool_run_limit"
                    )
                ),

                "request_toolset_run_limit": (
                    resolved_state_update.get(
                        "request_toolset_run_limit"
                    )
                ),
            },
        },

        attributes={
            "agent.name": (
                "main_agent"
            ),

            "conversation.thread_id": (
                thread_id
            ),

            "agent.input_chars": len(
                user_text
            ),

            "memory.context_present": bool(
                memory_context
                .strip()
            ),

            "memory.context_chars": len(
                memory_context
            ),

            "agent.executor_model_run_limit": (
                resolved_state_update.get(
                    "executor_model_run_limit",
                    0,
                )
            ),

            "agent.executor_tool_run_limit": (
                resolved_state_update.get(
                    "executor_tool_run_limit",
                    0,
                )
            ),

            "agent.request_toolset_run_limit": (
                resolved_state_update.get(
                    "request_toolset_run_limit",
                    0,
                )
            ),
        },
    ) as span:

        result = await agent.ainvoke(
            input_state,

            config={
                "configurable": {
                    "thread_id": (
                        thread_id
                    ),
                }
            },
        )

        if isinstance(
            result,
            dict,
        ):
            result_messages = list(
                result.get(
                    "messages",
                    [],
                )
            )

        else:
            result_messages = []

        if result_messages:
            final_message = (
                result_messages[-1]
            )

            if isinstance(
                final_message,
                dict,
            ):
                final_content = (
                    final_message.get(
                        "content",
                        "",
                    )
                )

            else:
                final_content = (
                    getattr(
                        final_message,
                        "content",
                        "",
                    )
                )

            answer = (
                _content_to_text(
                    final_content
                )
                .strip()
            )

        else:
            answer = ""

        if not answer:
            answer = (
                "模型已返回结果，"
                "但没有可显示的文字内容。"
            )
        execution_summary = (
            _build_agent_execution_summary(
                result_messages
            )
        )

        if isinstance(
            result,
            dict,
        ):
            model_calls_used = result.get(
                "executor_model_calls_used"
            )

            tool_calls_used = result.get(
                "executor_tool_calls_used"
            )

            request_toolset_calls_used = (
                result.get(
                    "request_toolset_calls_used"
                )
            )

            if (
                isinstance(
                    model_calls_used,
                    int,
                )
                and not isinstance(
                    model_calls_used,
                    bool,
                )
            ):
                execution_summary[
                    "model_call_count"
                ] = model_calls_used

            if (
                isinstance(
                    tool_calls_used,
                    int,
                )
                and not isinstance(
                    tool_calls_used,
                    bool,
                )
            ):
                execution_summary[
                    "tool_call_count"
                ] = tool_calls_used

            if (
                isinstance(
                    request_toolset_calls_used,
                    int,
                )
                and not isinstance(
                    request_toolset_calls_used,
                    bool,
                )
            ):
                execution_summary[
                    "request_toolset_call_count"
                ] = (
                    request_toolset_calls_used
                )

        set_span_attributes(
            span,

            **{
                "agent.model_call_count": (
                    execution_summary[
                        "model_call_count"
                    ]
                ),

                "agent.tool_call_count": (
                    execution_summary[
                        "tool_call_count"
                    ]
                ),

                "agent.tool_result_count": (
                    execution_summary[
                        "tool_result_count"
                    ]
                ),

                "agent.output_chars": len(
                    answer
                ),
            },
        )

        set_span_output(
            span,

            {
                "final_answer": (
                    answer
                ),

                "execution_summary": (
                    execution_summary
                ),
            },
        )

        if return_details:
            return {
                "final_answer": (
                    answer
                ),

                "current_turn_messages": (
                    _extract_current_turn_messages(
                        result_messages
                    )
                ),

                "execution_summary": (
                    execution_summary
                ),
            }

        return answer