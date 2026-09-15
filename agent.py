import json

from typing import Any

from langchain.chat_models import (
    init_chat_model,
)
from config import (
    Settings,
)
from prompt_loader import (
    load_prompt,
)
from observability import (
    set_span_attributes,
    set_span_output,
    trace_span,
)
from worker_termination import build_worker_cancellation_record

TRACE_MESSAGE_PREVIEW_MAX_CHARS = 1200

def _build_chat_model(
    *,
    model_provider: str,
    model_name: str,
    max_tokens: int,
    timeout_seconds: int,
    thinking_enabled: bool,
    api_key: str | None = None,
    base_url: str | None = None,
    reasoning_effort: str | None = None,
    extra_body: dict | None = None,
    max_retries: int = 2,
    token_limit_parameter: str = "max_completion_tokens",
    trace_role: str = "model",
):
    """Construct role models with provider-specific reasoning controls.

    DeepSeek supports disabled thinking; GPT-5 nano uses minimal effort
    when thinking_enabled is false, since it cannot disable reasoning.
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
        "max_retries": max_retries,
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

    if normalized_provider == "openai" and normalized_model_name.startswith("gpt-5-nano"):
        model_options["reasoning_effort"] = "low" if thinking_enabled else "minimal"
        model_options["max_completion_tokens"] = model_options.pop("max_tokens")

    if normalized_provider in {"compatible", "qwen"}:
        if not base_url:
            raise ValueError("compatible/qwen models require an explicit base_url")
        model_options["model_provider"] = "openai"
        # Keep compatible providers on Chat Completions, including unknown model IDs.
        model_options["use_responses_api"] = False
    if normalized_provider == "qwen":
        model_options["extra_body"] = {"enable_thinking": thinking_enabled}
    if api_key is not None:
        model_options["api_key"] = api_key
    if base_url:
        model_options["base_url"] = base_url
    if reasoning_effort is not None:
        if normalized_provider == "qwen":
            model_options["extra_body"]["reasoning_effort"] = reasoning_effort
        else:
            model_options["reasoning_effort"] = reasoning_effort
    if extra_body:
        model_options["extra_body"] = {**model_options.get("extra_body", {}), **extra_body}

    if normalized_provider in {"qwen", "compatible"}:
        from model_clients import CompatibleChatModel
        model_options.pop("model_provider")
        model = CompatibleChatModel(**model_options, token_limit_parameter=token_limit_parameter)
    else:
        model = init_chat_model(**model_options)
    profile = dict(getattr(model, "profile", None) or {})
    if normalized_provider in {"qwen", "compatible"}:
        model.profile = {**profile, "pdf_inputs": False, "pdf_tool_message": False}
    elif profile.get("attachment") is False:
        model.profile = {**profile, "pdf_inputs": False, "pdf_tool_message": False}
    from langchain_core.language_models.chat_models import BaseChatModel
    if isinstance(model, BaseChatModel):
        from trace_callbacks import callbacks
        model.callbacks = callbacks(model.callbacks)
        model.metadata = {**(model.metadata or {}), "runtime.model_role": trace_role, "trace.owner": "personalops"}
    return model

def build_role_model(settings: Settings, role: str):
    """Construct a role with its own credentials, endpoint, and request settings."""
    config = settings.role_models[role]
    return _build_chat_model(
        model_provider=config.provider, model_name=config.model,
        max_tokens=config.max_tokens, timeout_seconds=config.timeout_seconds,
        thinking_enabled=config.thinking_enabled, api_key=config.api_key,
        base_url=config.base_url or None, reasoning_effort=config.reasoning_effort,
        extra_body=config.extra_body, max_retries=config.max_retries,
        token_limit_parameter=config.token_limit_parameter,
        trace_role=role,
    )


def build_model(
    settings: Settings,
):
    """Build the General model; old settings objects retain their original fallback."""
    if getattr(settings, "role_models", None):
        return build_role_model(settings, "general")

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

        thinking_enabled=settings.llm_thinking_enabled,
    )


def build_summary_model(settings: Settings):
    """Build the dedicated conversation/Worker summarization model."""
    if getattr(settings, "role_models", None):
        return build_role_model(settings, "summary")
    return _build_chat_model(
        model_provider=settings.summary_llm_provider,
        model_name=settings.summary_llm_model,
        max_tokens=settings.cloud_llm_max_tokens,
        timeout_seconds=120,
        thinking_enabled=False,
    )


def build_memory_model(settings: Settings):
    """Build the dedicated low-cost typed memory extraction model."""

    if getattr(settings, "role_models", None):
        return build_role_model(settings, "extraction")
    return _build_chat_model(
        model_provider=settings.extraction_llm_provider,
        model_name=settings.extraction_llm_model,
        max_tokens=settings.memory_extraction_max_tokens,
        timeout_seconds=120,
        thinking_enabled=False,
    )

def build_hard_model(
    settings: Settings,
):
    """Build the Scheduler model; other production roles have independent settings."""
    if getattr(settings, "role_models", None):
        return build_role_model(settings, "scheduler")

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

        thinking_enabled=settings.scheduler_thinking_enabled,
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
    metadata = message.get("additional_kwargs", {}) if isinstance(message, dict) else getattr(message, "additional_kwargs", {})
    if metadata.get("personalops_runtime_event"):
        return "system"

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


def _extract_handoff_source_messages(result: Any, messages: list) -> list:
    """Keep only Tool results explicitly referenced by structured API handoff.

    Compaction may move early documentation results out of the current turn.
    The planning trace needs these few records to validate handoff entries, but
    it does not need the Worker's full archived conversation.
    """
    if not isinstance(result, dict):
        return []

    source_ids: set[str] = set()

    def visit(value: Any, *, inside_apis: bool = False) -> None:
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        if isinstance(value, dict):
            for key, item in value.items():
                active = inside_apis or key == "handoff_apis"
                if active and key == "tool_call_id" and isinstance(item, str):
                    source_ids.add(item)
                else:
                    visit(item, inside_apis=active)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item, inside_apis=inside_apis)

    for key in ("general_result", "worker_submission", "code_worker_submission"):
        visit(result.get(key))
    if not source_ids:
        return []

    candidates = [*result.get("worker_archived_messages", []), *messages]
    selected = []
    seen: set[str] = set()
    for message in candidates:
        call_id = _read_tool_call_id(message)
        if call_id in source_ids and call_id not in seen:
            selected.append(message)
            seen.add(call_id)
    return selected


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
                        "conversation/title"
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


async def ask_worker(
    agent,
    user_text: str,
    *,
    thread_id: str,
    state_update: dict[
        str,
        Any,
    ] | None = None,
    runtime_configurable: dict[str, Any] | None = None,
    return_details: bool = False,
    trace_role: str = "general",
) -> str | dict[
    str,
    Any,
]:
    """调用 General Worker，并记录完整执行根节点。

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

    # The outer runtime grants a fresh per-invocation budget, including on
    # Code repair. Keep the role's skill snapshot/history, not old counters.
    for counter in (
        "executor_model_calls_used", "executor_tool_calls_used",
        "show_all_toolsets_calls_used", "skill_preparation_calls_used", "worker_compaction_calls_used",
    ):
        input_state.setdefault(counter, 0)

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

    from runtime_tracing import ROLE_NAMES, TRACE_IDS, identities
    trace_name = "Code Agent" if trace_role == "code_agent" else ROLE_NAMES.get(trace_role, trace_role)
    trace_identity = {**TRACE_IDS.get(), **identities(resolved_state_update)}
    from trace_presentation import role_badge
    display_name = role_badge(trace_name)
    if trace_identity.get("step_id") is not None:
        display_name += f" / Step {trace_identity['step_id']}"
    if trace_identity.get("candidate_revision") is not None:
        display_name += f" / Revision {trace_identity['candidate_revision']}"
    with trace_span(
        display_name,

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

                "show_all_toolsets_run_limit": (
                    resolved_state_update.get(
                        "show_all_toolsets_run_limit"
                    )
                ),
            },
        },

        attributes={
            "agent.name": (
                trace_name
            ),

            "conversation.thread_id": (
                thread_id
            ),

            "agent.input_chars": len(
                user_text
            ),
            **{"runtime." + key: value for key, value in {**TRACE_IDS.get(), **identities(resolved_state_update)}.items()},

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

            "agent.show_all_toolsets_run_limit": (
                resolved_state_update.get(
                    "show_all_toolsets_run_limit",
                    0,
                )
            ),
        },
    ) as span:

        from knowledge_rag.runtime import automatic_rag, shared_code_scope, SHARED_CODE
        from knowledge_rag.query import retrieval_query as task_query
        shared_owner = SHARED_CODE.get() if trace_role in {"code", "reviewer", "code_reviewer"} else None
        if trace_role == "code_agent":
            shared_owner = "code-task:" + thread_id
            query = task_query(user_text, input_state.get("code_task"))
            await automatic_rag(query, "Code Agent", key=shared_owner, agent=shared_owner)

        worker_configurable = dict(runtime_configurable or {})
        # The stable Worker identity is owned by this boundary. Callers may
        # forward pause/recovery controls, but cannot redirect the checkpoint.
        worker_configurable["thread_id"] = thread_id
        from trace_callbacks import callbacks
        from workers.history_archive import history_scope
        with shared_code_scope(shared_owner), history_scope(thread_id.split(':step_', 1)[0], thread_id):
            result = await agent.ainvoke(
                input_state,

                config={
                    "configurable": worker_configurable,
                    "callbacks": callbacks(),
                    "metadata": {"runtime.model_role": trace_role, "trace.owner": "personalops",
                                 **{"runtime." + k: v for k, v in identities(resolved_state_update).items()}},
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

        if isinstance(result, dict) and result.get("general_result"):
            answer = str(result["general_result"].get("summary") or answer)
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

            show_all_toolsets_calls_used = (
                result.get(
                    "show_all_toolsets_calls_used"
                )
            )

            leadership_model_rounds = result.get(
                "worker_leadership_model_rounds_used",
                0,
            )
            finalization_model_rounds = result.get(
                "worker_finalization_model_calls_used",
                0,
            )
            finalization_tool_calls = result.get(
                "worker_finalization_tool_calls_used",
                0,
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
                ] = (
                    model_calls_used
                    + (
                        leadership_model_rounds
                        if isinstance(leadership_model_rounds, int)
                        and not isinstance(leadership_model_rounds, bool)
                        else 0
                    )
                )

            preparation_calls = int(result.get("skill_preparation_calls_used", 0) or 0)
            execution_summary["skill_preparation_call_count"] = preparation_calls
            execution_summary["model_call_count"] += preparation_calls
            compression_calls = int(result.get("worker_compaction_calls_used", 0) or 0)
            execution_summary["compaction_call_count"] = compression_calls
            execution_summary["model_call_count"] += compression_calls

            execution_summary["leadership_model_call_count"] = (
                leadership_model_rounds
                if isinstance(leadership_model_rounds, int)
                and not isinstance(leadership_model_rounds, bool)
                else 0
            )
            execution_summary["finalization_model_call_count"] = (
                finalization_model_rounds
                if isinstance(finalization_model_rounds, int)
                and not isinstance(finalization_model_rounds, bool)
                else 0
            )
            execution_summary["finalization_tool_call_count"] = (
                finalization_tool_calls
                if isinstance(finalization_tool_calls, int)
                and not isinstance(finalization_tool_calls, bool)
                else 0
            )

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
                    show_all_toolsets_calls_used,
                    int,
                )
                and not isinstance(
                    show_all_toolsets_calls_used,
                    bool,
                )
            ):
                execution_summary[
                    "show_all_toolsets_call_count"
                ] = (
                    show_all_toolsets_calls_used
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
            details = {
                "general_result": result.get("general_result") if isinstance(result, dict) else None,
                "final_answer": (
                    answer
                ),

                "current_turn_messages": (
                    _extract_current_turn_messages(
                        result_messages
                    )
                ),

                "handoff_source_messages": (
                    _extract_handoff_source_messages(result, result_messages)
                ),

                "execution_summary": (
                    execution_summary
                ),

                "worker_terminal_action": (
                    result.get("worker_terminal_action")
                    if isinstance(result, dict)
                    else None
                ),

                "worker_leadership_decisions": (
                    list(result.get("worker_leadership_decisions", []))
                    if isinstance(result, dict)
                    else []
                ),

                "worker_submission": (
                    result.get("worker_submission")
                    if isinstance(result, dict)
                    else None
                ),
            }

            if (
                isinstance(result, dict)
                and result.get("worker_terminal_action") == "CANCEL"
            ):
                decisions = list(result.get("worker_leadership_decisions", []))
                reason = "Worker was cancelled by leadership."
                if decisions:
                    decision = decisions[-1].get("decision", {})
                    if isinstance(decision, dict):
                        reason = str(decision.get("reason") or reason)
                details["worker_cancellation_record"] = (
                    build_worker_cancellation_record(
                        result,
                        reason=reason,
                    ).model_dump(mode="json")
                )

            # Dedicated CODE runtimes return control-plane records in
            # addition to ordinary messages.  Keep this adapter generic, but
            # explicitly preserve the bounded records that the Planning Graph
            # is allowed to consume.  Hidden model reasoning and the complete
            # checkpoint are deliberately not forwarded.
            if isinstance(result, dict):
                for key in (
                    "code_worker_submission",
                    "code_review_loop",
                    "code_review_report",
                    "code_artifact_manifest",
                    "code_publication_receipt",
                    "code_handoff_publication_receipts",
                    "code_integration_commit",
                    "code_integration_status",
                    "code_attempt_archive",
                    "code_attempt_final_record",
                    "code_runtime_session_id",
                    "code_scheduler_decision_applied",
                    "role_skill_snapshot",
                    "skill_preparation_calls_used",
                    "code_superseded_attempt_records",
                ):
                    if key in result:
                        details[key] = result[key]

            return details

        return answer
