

import asyncio
import logging
import os
from dataclasses import dataclass, field

from concurrent.futures import (
    CancelledError,
    Future,
)
from pathlib import Path
from uuid import uuid4

from feishu_attachments import AttachmentIngressError, FeishuAttachmentStore, PreparedInbound
from knowledge_rag.ingress import RagUploadInbox
from feishu_exports import ExportService, configure_exports, export_event_scope
from path import AGENT_DATA_ROOT

from dotenv import load_dotenv
# 必须先读取.env。
#
# observability.py中的Phoenix开关，
# 可能在其他业务模块导入时立即读取环境变量。
PROJECT_DIRECTORY = (
    Path(__file__)
    .resolve()
    .parent
)

ENV_PATH = (
    PROJECT_DIRECTORY
    / ".env"
)

load_dotenv(
    dotenv_path=(
        ENV_PATH
    ),
    override=True,
)


# Terminal只保留少量程序生命周期日志、
# 用户输入、最终回答、Warning和Error。
logging.basicConfig(
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),

    level=logging.INFO,
)


# 第三方库的内部调用细节统一放到WARNING以上。
#
# LangChain和LangGraph的详细调用过程，
# 以后由Phoenix Trace负责展示，
# 不再依赖它们的Terminal日志。
NOISY_LOGGER_NAMES = (
    "httpx",
    "httpcore",

    "Lark",
    "lark_channel",
    "lark_oapi",
    "websockets",

    "openai",
    "anthropic",

    "langchain",
    "langgraph",

    "mcp",

    "ddgs",

    "sentence_transformers",
    "transformers",
    "huggingface_hub",

    "urllib3",

    "opentelemetry",
    "phoenix",
)

for noisy_logger_name in (
    NOISY_LOGGER_NAMES
):
    logging.getLogger(
        noisy_logger_name
    ).setLevel(
        logging.WARNING
    )
logging.getLogger(
    "Lark"
).propagate = False

logger = logging.getLogger(
    "agent"
)
from phoenix_runtime import (
    PhoenixServerRuntime,
)


phoenix_server_runtime = (
    PhoenixServerRuntime()
)

os.environ[
    "PHOENIX_UI_URL"
] = phoenix_server_runtime.ui_url

os.environ[
    "PHOENIX_COLLECTOR_ENDPOINT"
] = (
    f"{phoenix_server_runtime.ui_url}"
    "/v1/traces"
)
try:
    phoenix_server_runtime.start()

except Exception:
    # Phoenix属于观测增强能力。
    #
    # 启动失败时关闭本轮Tracing，
    # 但飞书入口和Agent仍然继续启动。
    logger.exception(
        "Phoenix本地服务启动失败，"
        "本次运行将关闭Trace"
    )

    os.environ[
        "PHOENIX_TRACING_ENABLED"
    ] = "false"

# Phoenix必须在导入tools、agent、
# conversation_runtime等业务模块前注册。
#
# 这些模块会继续导入LangChain和LangGraph。
# 如果注册过晚，部分自动Instrumentation
# 可能无法完整覆盖。
from observability import (
    setup_observability,
    shutdown_observability,
)


setup_observability()


# 下面这些导入故意放在Phoenix初始化之后。

from typing import (
    Any,
)

from lark_channel import (
    FeishuChannel,
    new_card,
)
from lark_channel.core.enum import (
    LogLevel,
)
from config import (
    load_settings,
)
from conversation_runtime import (
    ConversationRuntime,
)
from progress_events import (
    ProgressCallback,
    ProgressEvent,
    render_progress_event,
)
from eventing import (
    AgentEvent,
    EventAction,
    EventConflictError,
    EventOrigin,
    EventRunPump,
    RunStatus,
    create_agent_event,
    create_event_run,
)
from scheduling import DeliveryTarget, ReminderRun, ReminderSchedule
from tools import (
    ALL_TOOLS,
)

FEISHU_CHANNEL = (
    "feishu"
)


# 当前项目是单用户个人Agent。
#
# ConversationRuntime目前仍然要求
# 提供external_chat_id。
#
# 这里故意不使用真实的飞书chat_id
# 作为Conversation所有者，
# 而是固定使用owner。
#
# 这样即使以后从不同设备进入飞书，
# 仍然共享同一套Conversation。
OWNER_EXTERNAL_CHAT_ID = (
    "owner"
)


TERMINAL_MESSAGE_MAX_CHARS = 1200


# 这里只串行化极短的入口状态操作：确定当前Conversation、写入Event，
# 或执行/new、/switch等Conversation命令。模型和工具执行不持有此锁；
# 顶层任务的串行性由EventRunPump单消费者保证。
OWNER_MESSAGE_LOCK = (
    asyncio.Lock()
)

# Event和飞书回复地址持久化在SQLite；进度回调本身属于当前进程，
# 重启恢复时会用持久化地址重新创建。最终发送的事务型Outbox仍是后续能力。
FEISHU_EVENT_CONTEXTS: dict[
    str,
    tuple[str, ProgressCallback],
] = {}
FEISHU_EVENT_PUMP: EventRunPump | None = None
FEISHU_ATTACHMENTS = FeishuAttachmentStore(AGENT_DATA_ROOT / "feishu-attachments")
RAG_UPLOADS = RagUploadInbox(AGENT_DATA_ROOT / "rag-upload")
# ConversationRuntime、SQLite、LangGraph、
# Memory和MCP实际所属的主事件循环。
#
# 飞书SDK会在自己的工作线程和事件循环中
# 调用消息Handler，因此不能在那里直接运行Agent。
APPLICATION_LOOP: (
    asyncio.AbstractEventLoop
    | None
) = None


# 保存从飞书SDK线程提交到
# Agent主事件循环中的消息任务。
#
# 避免Future过早失去引用，
# 同时便于记录任务中的未处理异常。
FEISHU_MESSAGE_FUTURES: set[
    Future
] = set()


@dataclass
class FeishuInteractionState:
    """Short-lived UI state; durable memory choices are revalidated in SQLite."""

    mode: str
    token: str = field(default_factory=lambda: uuid4().hex)
    group_id: str = ""
    left_memory_id: str = ""
    right_memory_id: str = ""
    skipped_group_ids: set[str] = field(default_factory=set)


FEISHU_INTERACTIONS: dict[str, FeishuInteractionState] = {}

settings = load_settings()


conversation_runtime = (
    ConversationRuntime(
        settings=settings,
        tools=ALL_TOOLS,
    )
)


feishu_channel = (
    FeishuChannel(
        app_id=(
            settings.feishu_app_id
        ),

        app_secret=(
            settings.feishu_app_secret
        ),

        # 飞书SDK的连接、身份解析等
        # 正常INFO日志不再显示。
        #
        # Warning和Error仍然保留。
        log_level=(
            LogLevel.WARNING
        ),
    )
)

FEISHU_EXPORTS = ExportService(AGENT_DATA_ROOT / "feishu-exports", settings.feishu_export, feishu_channel)
configure_exports(FEISHU_EXPORTS)


logger.info(
    "Agent配置已加载 | "
    "platform=feishu | "
    "role_models=%s | tool_count=%s",

    {role: config.provider + ":" + config.model for role, config in settings.role_models.items()},

    len(
        ALL_TOOLS
    ),
)


def _compact_terminal_text(
    text: str,
) -> str:
    """压缩Terminal中显示的用户消息和最终回答。"""

    normalized_text = (
        text.strip()
    )

    if (
        len(
            normalized_text
        )
        <= TERMINAL_MESSAGE_MAX_CHARS
    ):
        return normalized_text

    removed_chars = (
        len(
            normalized_text
        )
        - TERMINAL_MESSAGE_MAX_CHARS
    )

    return (
        normalized_text[
            :TERMINAL_MESSAGE_MAX_CHARS
        ]
        + "\n"
        + (
            f"... Terminal已省略"
            f"{removed_chars}个字符，"
            "完整内容请在Phoenix中查看。"
        )
    )


def _parse_command(
    text: str,
) -> tuple[
    str | None,
    list[str],
]:
    """解析飞书文字中的斜杠命令。"""

    normalized_text = (
        text.strip()
    )

    if not normalized_text.startswith(
        "/"
    ):
        return (
            None,
            [],
        )

    parts = (
        normalized_text.split()
    )

    command = (
        parts[0][1:]
        .strip()
        .lower()
    )

    if not command:
        return (
            None,
            [],
        )

    return (
        command,
        parts[1:],
    )


async def _send_text(
    chat_id: str,
    text: str,
) -> None:
    """向当前飞书会话发送纯文本。"""

    normalized_text = (
        text.strip()
    )

    if not normalized_text:
        normalized_text = (
            "Agent没有生成可发送的文字。"
        )

    await feishu_channel.send(
        chat_id,

        {
            "text": (
                normalized_text
            ),
        },
    )


async def _send_card(
    chat_id: str,
    card: dict[str, Any],
    *,
    fallback_text: str,
) -> None:
    """Send an interactive card and retain a text-only fallback path."""

    try:
        result = await feishu_channel.send(chat_id, {"card": card})
        if not result.success:
            await _send_text(chat_id, fallback_text)
    except Exception:
        logger.exception("飞书消息卡片发送失败，已降级为纯文本")
        await _send_text(chat_id, fallback_text)


def _control_buttons() -> list[dict[str, Any]]:
    return [
        {"label": "提交 RAG", "action": {"kind": "personalops", "action": "rag_upload"}, "style": "primary"},
        {"label": "帮助", "action": {"kind": "personalops", "action": "help"}},
        {"label": "插入任务", "action": {"kind": "personalops", "action": "insert"}},
        {"label": "替换任务", "action": {"kind": "personalops", "action": "replace"}},
        {"label": "取消任务", "action": {"kind": "personalops", "action": "cancel"}, "style": "danger"},
        {"label": "清理记忆", "action": {"kind": "personalops", "action": "clean"}, "style": "primary"},
    ]


async def _send_control_panel(chat_id: str) -> None:
    buttons = _control_buttons()
    card = (
        new_card()
        .header("PersonalOps", subtitle="快捷操作", template="blue")
        .buttons(buttons[:3])
        .buttons(buttons[3:6])
        .buttons(buttons[6:])
        .build()
    )
    await _send_card(
        chat_id,
        card,
        fallback_text="快捷操作：提交 RAG · /help · /insert 内容 · /replace 内容 · /cancel · /clean",
    )


async def _send_input_prompt(chat_id: str, mode: str) -> None:
    label = "插入任务" if mode == "insert" else "替换任务"
    state = FeishuInteractionState(mode=mode)
    FEISHU_INTERACTIONS[chat_id] = state
    card = (
        new_card()
        .header(label, subtitle="等待下一条文字", template="orange")
        .text(
            "请直接发送具体内容。下一条普通文字会作为"
            f"{label}提交；发送“退出”或点击按钮可取消输入。"
        )
        .button(
            "退出输入",
            action={
                "kind": "personalops",
                "action": "exit",
                "token": state.token,
            },
        )
        .build()
    )
    await _send_card(
        chat_id,
        card,
        fallback_text=f"已进入{label}输入。请发送内容；发送“退出”可离开。",
    )

def _build_progress_callback(
    chat_id: str,
) -> ProgressCallback:
    """为当前飞书请求建立独立的进度更新回调。"""

    normalized_chat_id = (
        chat_id.strip()
    )

    if not normalized_chat_id:
        raise ValueError(
            "chat_id不能为空。"
        )

    sent_event_keys: set[
        tuple[
            str,
            int | None,
            str | None,
        ]
    ] = set()

    # 保存本轮任务唯一的进度消息ID。
    #
    # 第一次进度事件发送新消息，
    # 后面的进度事件全部修改这条消息。
    progress_message_id: (
        str
        | None
    ) = None

    async def progress_callback(
        event: ProgressEvent,
    ) -> None:
        """首次发送进度消息，后续原地更新同一条消息。"""

        nonlocal progress_message_id

        if not isinstance(
            event,
            ProgressEvent,
        ):
            logger.warning(
                "收到非法进度事件，"
                "本次已经忽略"
            )

            return

        event_key = (
            event.dedup_key
        )

        if event_key in sent_event_keys:
            return

        progress_text = (
            render_progress_event(
                event
            )
        )

        if not progress_text:
            return

        progress_payload = {
            "text": (
                progress_text
            ),
        }

        try:
            # 第一次：创建进度消息，
            # 并保存飞书返回的message_id。
            if progress_message_id is None:
                result = await (
                    feishu_channel.send(
                        normalized_chat_id,
                        progress_payload,
                    )
                )

                message_id = str(
                    getattr(
                        result,
                        "message_id",
                        "",
                    )
                    or ""
                ).strip()

                if not (
                    getattr(
                        result,
                        "success",
                        False,
                    )
                    and message_id
                ):
                    logger.warning(
                        "飞书进度消息首次发送失败，"
                        "Agent主任务将继续执行"
                    )

                    return

                progress_message_id = (
                    message_id
                )

            # 后续：修改之前创建的同一条消息。
            else:
                result = await (
                    feishu_channel
                    .edit_message(
                        progress_message_id,
                        progress_payload,
                    )
                )

                if not getattr(
                    result,
                    "success",
                    False,
                ):
                    logger.warning(
                        "飞书进度消息更新失败，"
                        "Agent主任务将继续执行"
                    )

                    return

        except Exception:
            # 进度展示失败不能影响Agent主任务。
            logger.warning(
                "飞书进度消息发送或更新失败，"
                "Agent主任务将继续执行",
                exc_info=False,
            )

            return

        sent_event_keys.add(
            event_key
        )

    return progress_callback
async def _start_command(
    chat_id: str,
) -> None:
    """显示个人Agent使用说明。"""

    try:
        conflicts = await conversation_runtime.memory_conflict_summary()
    except RuntimeError:
        conflicts = {"open_groups": 0, "open_pairs": 0, "high_priority_groups": 0}
    help_text = (
        "**对话**\n"
        "`/new [标题]` 新对话 · `/list` 对话列表 · `/switch 编号` 切换 · "
        "`/current` 当前对话\n\n"
        "**运行控制**\n"
        "`/insert 内容` 优先插入后恢复 · `/replace 内容` 安全替换 · "
        "`/cancel` 安全取消\n\n"
        "**记忆**\n"
        f"`/clean` 进入冲突清理（当前 {conflicts['open_groups']} 组 / "
        f"{conflicts['open_pairs']} 对，高置信 {conflicts['high_priority_groups']} 组） · "
        "清理中可用 `A`、`B`、`跳过`、`退出`\n\n"
        "**文件交付**\n"
        "`/approve 编号`、`/reject 编号 [原因]`；飞书文件回传使用 "
        "`/file_whoami`、`/file_status 编号`、`/file_approve 编号`、"
        "`/file_reject 编号`\n\n"
        "任何输入状态都可发送 `/exit` 或“退出”离开。"
    )
    buttons = _control_buttons()
    card = (
        new_card()
        .header("PersonalOps 帮助", subtitle="命令与快捷操作", template="blue")
        .markdown(help_text)
        .divider()
        .buttons(buttons[:3])
        .buttons(buttons[3:])
        .build()
    )
    await _send_card(chat_id, card, fallback_text=help_text.replace("**", "").replace("`", ""))


async def _new_conversation_command(
    chat_id: str,
    args: list[str],
) -> None:
    """创建并切换到新Conversation。"""

    title = (
        " ".join(
            args
        ).strip()
        or None
    )

    conversation = await (
        conversation_runtime
        .new_conversation(
            channel=(
                FEISHU_CHANNEL
            ),

            external_chat_id=(
                OWNER_EXTERNAL_CHAT_ID
            ),

            title=title,
        )
    )

    if title:
        description = (
            f"标题：{conversation.title}"
        )

    else:
        description = (
            "发送第一条消息后，"
            "模型会自动生成标题。"
        )

    await _send_text(
        chat_id,

        (
            "已创建并切换到新对话。\n\n"
            f"短ID：{conversation.short_id}\n"
            f"{description}"
        ),
    )


async def _list_conversations_command(
    chat_id: str,
) -> None:
    """列出个人Agent中的Conversation。"""

    active = await (
        conversation_runtime
        .get_active_conversation(
            channel=(
                FEISHU_CHANNEL
            ),

            external_chat_id=(
                OWNER_EXTERNAL_CHAT_ID
            ),
        )
    )

    conversations = await (
        conversation_runtime
        .list_conversations(
            channel=(
                FEISHU_CHANNEL
            ),

            external_chat_id=(
                OWNER_EXTERNAL_CHAT_ID
            ),
        )
    )

    lines = [
        "最近对话：",
        "",
    ]

    for index, conversation in enumerate(
        conversations,
        start=1,
    ):
        marker = (
            "▶"

            if (
                conversation.thread_id
                == active.thread_id
            )

            else " "
        )

        lines.append(
            f"{marker} {index}. "
            f"{conversation.title} "
            f"[{conversation.short_id}]"
        )

    if conversations:
        lines.extend(
            [
                "",
                "切换示例：",
                "/switch 2",
                (
                    "/switch "
                    f"{conversations[0].short_id}"
                ),
            ]
        )

    await _send_text(
        chat_id,

        "\n".join(
            lines
        ),
    )


async def _switch_conversation_command(
    chat_id: str,
    args: list[str],
) -> None:
    """按列表编号或短ID切换Conversation。"""

    if len(args) != 1:
        await _send_text(
            chat_id,

            (
                "用法：\n"
                "/switch 2\n"
                "或者：/switch a1b2c3"
            ),
        )

        return

    conversation = await (
        conversation_runtime
        .switch_conversation(
            channel=(
                FEISHU_CHANNEL
            ),

            external_chat_id=(
                OWNER_EXTERNAL_CHAT_ID
            ),

            selector=(
                args[0]
            ),
        )
    )

    if conversation is None:
        await _send_text(
            chat_id,

            (
                "没有找到该对话。\n"
                "请先使用 /list 查看。"
            ),
        )

        return

    await _send_text(
        chat_id,

        (
            "已切换对话。\n\n"
            f"标题：{conversation.title}\n"
            f"短ID：{conversation.short_id}"
        ),
    )


async def _current_conversation_command(
    chat_id: str,
) -> None:
    """查看当前Conversation。"""

    conversation = await (
        conversation_runtime
        .get_active_conversation(
            channel=(
                FEISHU_CHANNEL
            ),

            external_chat_id=(
                OWNER_EXTERNAL_CHAT_ID
            ),
        )
    )

    await _send_text(
        chat_id,

        (
            "当前对话：\n\n"
            f"标题：{conversation.title}\n"
            f"短ID：{conversation.short_id}"
        ),
    )


async def _approve_workspace_command(
    chat_id: str,
    args: list[str],
) -> None:
    """Approve one reviewed, immutable promotion for the active conversation."""

    if len(args) != 1:
        await _send_text(chat_id, "用法：/approve promotion-xxxx")
        return
    conversation = await conversation_runtime.get_active_conversation(
        channel=FEISHU_CHANNEL,
        external_chat_id=OWNER_EXTERNAL_CHAT_ID,
    )
    try:
        promotion = await conversation_runtime.approve_workspace_promotion(
            conversation_id=conversation.conversation_id,
            promotion_id=args[0],
        )
    except LookupError:
        await _send_text(chat_id, "没有找到这条交付请求。")
        return
    except ValueError as error:
        await _send_text(chat_id, str(error))
        return
    except RuntimeError as error:
        await _send_text(chat_id, f"这条交付请求现在不能批准：{error}")
        return
    except Exception:
        logger.exception("Workspace人工批准失败 | promotion_id=%s", args[0])
        await _send_text(
            chat_id,
            "交付过程中出现错误，正式工作区没有被继续覆盖。",
        )
        return
    await _send_text(
        chat_id,
        (
            "交付完成。\n\n"
            f"目录：{promotion.target_root}\n"
            f"Delivery commit：{promotion.delivered_commit}"
        ),
    )


async def _reject_workspace_command(
    chat_id: str,
    args: list[str],
) -> None:
    """Reject a pending promotion without changing user-visible files."""

    if not args:
        await _send_text(chat_id, "用法：/reject promotion-xxxx [原因]")
        return
    conversation = await conversation_runtime.get_active_conversation(
        channel=FEISHU_CHANNEL,
        external_chat_id=OWNER_EXTERNAL_CHAT_ID,
    )
    reason = " ".join(args[1:]).strip() or None
    try:
        promotion = await conversation_runtime.reject_workspace_promotion(
            conversation_id=conversation.conversation_id,
            promotion_id=args[0],
            reason=reason,
        )
    except LookupError:
        await _send_text(chat_id, "没有找到这条交付请求。")
        return
    except ValueError as error:
        await _send_text(chat_id, str(error))
        return
    except RuntimeError as error:
        await _send_text(chat_id, f"这条交付请求现在不能拒绝：{error}")
        return
    await _send_text(
        chat_id,
        (
            "已拒绝本次文件交付，正式工作区没有变化。\n\n"
            f"交付编号：{promotion.promotion_id}"
        ),
    )


def _format_conflict_choice(label: str, value: dict[str, Any]) -> str:
    confidence = {1: "低", 2: "中", 3: "高", 4: "高"}.get(value["confidence"], "中")
    importance = {1: "低", 2: "中", 3: "高", 4: "紧急"}.get(value["importance"], "中")
    valid_from = value.get("valid_from") or "未注明"
    expires_at = value.get("expires_at") or "无限期"
    return (
        f"**{label}**　置信度：{confidence}｜重要性：{importance}\n"
        f"{value['content']}\n"
        f"<font color='grey'>有效期：{valid_from} → {expires_at}</font>"
    )


async def _show_next_memory_conflict(chat_id: str) -> None:
    state = FEISHU_INTERACTIONS.get(chat_id)
    skipped = state.skipped_group_ids if state and state.mode == "clean" else set()
    summary = await conversation_runtime.memory_conflict_summary()
    pair = await conversation_runtime.next_memory_conflict(
        excluded_group_ids=skipped,
    )
    if pair is None:
        FEISHU_INTERACTIONS.pop(chat_id, None)
        text = (
            "本轮没有更多需要处理的冲突。"
            if summary["open_groups"]
            else "当前没有未解决的冲突记忆。"
        )
        await _send_card(
            chat_id,
            new_card()
            .header("记忆清理", subtitle="已退出清理模式", template="green")
            .text(text)
            .button("返回帮助", action={"kind": "personalops", "action": "help"})
            .build(),
            fallback_text=text,
        )
        return

    token = uuid4().hex
    FEISHU_INTERACTIONS[chat_id] = FeishuInteractionState(
        mode="clean",
        token=token,
        group_id=pair.group_id,
        left_memory_id=pair.left["memory_id"],
        right_memory_id=pair.right["memory_id"],
        skipped_group_ids=set(skipped),
    )
    action_base = {
        "kind": "personalops_memory_clean",
        "token": token,
        "group_id": pair.group_id,
        "left_memory_id": pair.left["memory_id"],
        "right_memory_id": pair.right["memory_id"],
    }
    card = (
        new_card()
        .header(
            "选择要保留的记忆",
            subtitle=(
                f"{summary['open_groups']} 组 / {summary['open_pairs']} 对待处理 · "
                f"本组还含 {pair.remaining_pair_count} 对"
            ),
            template="orange",
        )
        .markdown(_format_conflict_choice("A", pair.left))
        .divider()
        .markdown(_format_conflict_choice("B", pair.right))
        .divider()
        .buttons([
            {"label": "保留 A", "action": {**action_base, "action": "choose_a"}, "style": "primary"},
            {"label": "保留 B", "action": {**action_base, "action": "choose_b"}, "style": "primary"},
            {"label": "跳过本组", "action": {**action_base, "action": "skip"}},
            {"label": "退出", "action": {**action_base, "action": "exit"}},
        ])
        .build()
    )
    fallback = (
        f"记忆冲突：\nA. {pair.left['content']}\nB. {pair.right['content']}\n"
        "回复 A 或 B 保留对应记忆；也可回复“跳过”或“退出”。"
    )
    await _send_card(chat_id, card, fallback_text=fallback)


async def _handle_clean_action(
    chat_id: str,
    action: str,
    *,
    token: str | None = None,
    group_id: str | None = None,
    left_memory_id: str | None = None,
    right_memory_id: str | None = None,
) -> None:
    state = FEISHU_INTERACTIONS.get(chat_id)
    if state is None or state.mode != "clean":
        await _send_text(chat_id, "清理会话已失效，请重新使用 /clean。")
        return
    if token is not None and (
        token != state.token
        or group_id != state.group_id
        or left_memory_id != state.left_memory_id
        or right_memory_id != state.right_memory_id
    ):
        await _send_text(chat_id, "这是旧的清理卡片，请使用最新一张卡片。")
        return
    if action == "exit":
        FEISHU_INTERACTIONS.pop(chat_id, None)
        await _send_text(chat_id, "已退出记忆清理；未处理的冲突仍会保留。")
        return
    if action == "skip":
        state.skipped_group_ids.add(state.group_id)
        await _show_next_memory_conflict(chat_id)
        return
    if action not in {"choose_a", "choose_b"}:
        await _send_text(chat_id, "可选择 A、B、跳过或退出。")
        return
    keep_id = state.left_memory_id if action == "choose_a" else state.right_memory_id
    retire_id = state.right_memory_id if action == "choose_a" else state.left_memory_id
    try:
        await conversation_runtime.resolve_memory_conflict(
            group_id=state.group_id,
            keep_memory_id=keep_id,
            retire_memory_id=retire_id,
        )
    except ValueError as error:
        FEISHU_INTERACTIONS.pop(chat_id, None)
        await _send_text(chat_id, str(error))
        return
    await _show_next_memory_conflict(chat_id)


async def _clean_command(chat_id: str, args: list[str]) -> None:
    if not args:
        FEISHU_INTERACTIONS[chat_id] = FeishuInteractionState(mode="clean")
        await _show_next_memory_conflict(chat_id)
        return
    action = args[0].casefold()
    action = {
        "a": "choose_a",
        "b": "choose_b",
        "跳过": "skip",
        "skip": "skip",
        "退出": "exit",
        "exit": "exit",
    }.get(action, action)
    await _handle_clean_action(chat_id, action)


async def _cancel_running_task(chat_id: str) -> None:
    try:
        cancel_event = await _enqueue_feishu_cancel(chat_id=chat_id)
    except EventConflictError:
        await _send_text(chat_id, "这个任务已经有一条取消请求在等待安全节点，无需重复发送。")
        return
    except Exception:
        logger.exception("飞书CANCEL事件入队失败")
        await _send_text(chat_id, "取消请求没有成功保存，请稍后重试。")
        return
    if cancel_event is None:
        await _send_text(chat_id, "当前对话没有正在运行的任务。")


async def _handle_command(
    chat_id: str,
    command: str,
    args: list[str],
) -> bool:
    """处理命令。

    Returns:
        True表示消息已经作为命令处理；
        False表示应继续交给主Agent。
    """

    if command in {
        "start",
        "help",
    }:
        await _start_command(
            chat_id
        )

        return True

    if command == "new":
        await _new_conversation_command(
            chat_id,
            args,
        )

        return True

    if command in {
        "list",
        "conversations",
    }:
        await _list_conversations_command(
            chat_id
        )

        return True

    if command == "switch":
        await _switch_conversation_command(
            chat_id,
            args,
        )

        return True

    if command == "current":
        await _current_conversation_command(
            chat_id
        )

        return True

    if command == "approve":
        await _approve_workspace_command(chat_id, args)
        return True

    if command == "reject":
        await _reject_workspace_command(chat_id, args)
        return True

    if command == "clean":
        await _clean_command(chat_id, args)
        return True

    if command == "exit":
        if FEISHU_INTERACTIONS.pop(chat_id, None) is None:
            await _send_text(chat_id, "当前没有需要退出的交互模式。")
        else:
            await _send_text(chat_id, "已退出当前交互模式。")
        return True

    return False


def _require_event_pump() -> EventRunPump:
    pump = FEISHU_EVENT_PUMP
    if pump is None or not pump.running:
        raise RuntimeError("飞书Event Run Pump尚未启动。")
    return pump


async def _deliver_scheduled_feishu_text(
    schedule: ReminderSchedule,
    run: ReminderRun,
) -> dict[str, Any]:
    """Deliver literal reminder text to the trusted chat captured at creation."""

    reply_target_id = str(schedule.reply_target_id or "").strip()
    if not reply_target_id:
        raise RuntimeError("飞书日程缺少持久化回复地址。")
    await _send_text(reply_target_id, schedule.message)
    return {
        "provider": "feishu",
        "reply_target_id": reply_target_id,
        "run_id": run.run_id,
    }


async def _enqueue_scheduled_agent_event(
    schedule: ReminderSchedule,
    run: ReminderRun,
) -> dict[str, Any]:
    """Idempotently turn one due occurrence into a normal queued Agent Event."""

    pump = _require_event_pump()
    conversation_id = str(schedule.conversation_id or "").strip()
    reply_target_id = str(schedule.reply_target_id or "").strip()
    if not conversation_id or not reply_target_id:
        raise RuntimeError("定时Agent任务缺少Conversation或飞书回复地址。")

    event_id = f"evt_schedule_{run.run_id.removeprefix('srun_')}"
    event = await conversation_runtime.event_store.get_event(event_id)
    if event is None:
        event = create_agent_event(
            conversation_id=conversation_id,
            action=EventAction.QUEUE,
            payload_text=schedule.message,
            event_id=event_id,
            origin=EventOrigin.SYSTEM,
            reply_target_id=reply_target_id,
        )
        await conversation_runtime.event_store.add_event(
            event,
            run=create_event_run(event),
        )
    elif (
        event.conversation_id != conversation_id
        or event.reply_target_id != reply_target_id
        or event.payload_text != schedule.message
        or event.action is not EventAction.QUEUE
    ):
        raise RuntimeError("定时Event幂等键已存在，但内容不一致。")

    FEISHU_EVENT_CONTEXTS[event.event_id] = (
        reply_target_id,
        _build_progress_callback(reply_target_id),
    )
    pump.notify(event)
    return {
        "event_id": event.event_id,
        "action": event.action.value,
        "status": event.status.value,
        "run_id": run.run_id,
    }


async def _run_feishu_event(
    event: AgentEvent,
    pause_control,
    resume_from_checkpoint: bool,
) -> str:
    context = FEISHU_EVENT_CONTEXTS.get(event.event_id)
    if context is None:
        reply_target_id = str(event.reply_target_id or "").strip()
        if not reply_target_id:
            raise RuntimeError(
                "Event缺少持久化的飞书回复地址，无法安全恢复投递。"
            )
        context = (
            reply_target_id,
            _build_progress_callback(reply_target_id),
        )
        FEISHU_EVENT_CONTEXTS[event.event_id] = context
    _, progress_callback = context
    await asyncio.to_thread(FEISHU_ATTACHMENTS.materialize, event)
    logger.info(
        "EVENT START | event_id=%s | action=%s",
        event.event_id,
        event.action.value,
    )
    with export_event_scope(await asyncio.to_thread(FEISHU_ATTACHMENTS.export_identity, event)):
        return await conversation_runtime.ask(
            user_text=event.payload_text,
            channel=FEISHU_CHANNEL,
            external_chat_id=OWNER_EXTERNAL_CHAT_ID,
            progress_callback=progress_callback,
            event_id=event.event_id,
            target_conversation_id=event.conversation_id,
            pause_control=pause_control,
            bypass_conversation_lock=(event.action is EventAction.INSERT),
            resume_from_checkpoint=resume_from_checkpoint,
            replacement_target_event_id=(
                event.target_event_id
                if event.action is EventAction.REPLACE
                else None
            ),
        )


async def _deliver_feishu_event_result(event: AgentEvent, result: str) -> None:
    context = FEISHU_EVENT_CONTEXTS.pop(event.event_id, None)
    if event.action in {EventAction.CANCEL, EventAction.REPLACE} and event.target_event_id:
        # The cancelled/superseded task will not reach its ordinary callback.
        # Release only its in-memory channel callback; the durable Event,
        # checkpoint, and reply target remain in SQLite for audit/recovery.
        FEISHU_EVENT_CONTEXTS.pop(event.target_event_id, None)
    if context is None:
        logger.error(
            "Agent事件完成但缺少飞书回复上下文 | event_id=%s",
            event.event_id,
        )
        return
    chat_id, _ = context
    await _send_text(chat_id, result)
    if event.action is not EventAction.CANCEL:
        await _send_control_panel(chat_id)
    logger.info(
        "EVENT COMPLETE | event_id=%s | %s",
        event.event_id,
        _compact_terminal_text(result),
    )


async def _deliver_feishu_event_failure(
    event: AgentEvent,
    error: BaseException,
) -> None:
    context = FEISHU_EVENT_CONTEXTS.pop(event.event_id, None)
    if event.action is EventAction.REPLACE and event.target_event_id:
        FEISHU_EVENT_CONTEXTS.pop(event.target_event_id, None)
    logger.error(
        "Agent事件执行失败 | event_id=%s | error=%s: %s",
        event.event_id,
        type(error).__name__,
        error,
    )
    if context is None:
        return
    chat_id, _ = context
    await _send_text(
        chat_id,
        "Agent处理这条排队任务时出现了问题，详细错误已经记录。",
    )


async def _finalize_feishu_cancelled_event(
    target_event: AgentEvent,
    cancel_event: AgentEvent,
) -> None:
    """Archive run-owned resources without asking another model to summarize."""

    await conversation_runtime.finalize_cancelled_run(
        target_event_id=target_event.event_id,
        cancel_event_id=cancel_event.event_id,
    )


async def _finalize_feishu_superseded_event(
    target_event: AgentEvent,
    replacement_event: AgentEvent,
) -> None:
    """Archive the old generation before the new Scheduler is invoked."""

    await conversation_runtime.finalize_superseded_run(
        target_event_id=target_event.event_id,
        replacement_event_id=replacement_event.event_id,
    )


async def _enqueue_feishu_event(
    *,
    chat_id: str,
    payload_text: str,
    action: EventAction,
    inbound: PreparedInbound | None = None,
) -> AgentEvent:
    """Bind one Feishu message to the current conversation and persist it."""

    pump = _require_event_pump()
    normalized_payload = payload_text.strip()
    if not normalized_payload:
        raise ValueError("排队任务内容不能为空。")

    async with OWNER_MESSAGE_LOCK:
        if inbound is not None:
            existing = await conversation_runtime.event_store.get_event(inbound.event_id)
            if existing is not None:
                return existing
        conversation = await conversation_runtime.get_active_conversation(
            channel=FEISHU_CHANNEL,
            external_chat_id=OWNER_EXTERNAL_CHAT_ID,
        )
        conversation_id = inbound.conversation_id if inbound else conversation.conversation_id
        target_event_id = pump.active_event_id if action is EventAction.INSERT else None
        if target_event_id is not None:
            target = await conversation_runtime.event_store.require_event(target_event_id)
            if target.conversation_id != conversation_id:
                target_event_id = None
        event = create_agent_event(
            conversation_id=conversation_id,
            action=action,
            payload_text=inbound.payload(normalized_payload) if inbound else normalized_payload,
            event_id=inbound.event_id if inbound else None,
            target_event_id=target_event_id,
            origin=EventOrigin.FEISHU,
            reply_target_id=chat_id,
        )
        run = create_event_run(event)
        FEISHU_EVENT_CONTEXTS[event.event_id] = (
            chat_id,
            _build_progress_callback(chat_id),
        )
        try:
            if inbound is not None:
                await asyncio.to_thread(FEISHU_ATTACHMENTS.bind, inbound, event.event_id)
            await conversation_runtime.event_store.add_event(event, run=run)
        except BaseException:
            FEISHU_EVENT_CONTEXTS.pop(event.event_id, None)
            raise

    if action is EventAction.INSERT and event.target_event_id:
        acknowledgement = (
            "已收到插入任务；当前任务会在下一个安全节点暂停，"
            "插入任务完成后再继续。"
        )
    elif action is EventAction.INSERT:
        acknowledgement = "当前没有运行中的任务；已进入优先队列。"
    else:
        acknowledgement = "已收到，任务已经进入处理队列。"
    try:
        await _send_text(chat_id, acknowledgement)
    finally:
        pump.notify(event)
    return event


async def _enqueue_feishu_cancel(*, chat_id: str) -> AgentEvent | None:
    """Persist `/cancel` against the task currently running in this chat."""

    pump = _require_event_pump()
    async with OWNER_MESSAGE_LOCK:
        target_event_id = pump.active_event_id
        if target_event_id is None:
            return None
        conversation = await conversation_runtime.get_active_conversation(
            channel=FEISHU_CHANNEL,
            external_chat_id=OWNER_EXTERNAL_CHAT_ID,
        )
        target = await conversation_runtime.event_store.require_event(
            target_event_id
        )
        if target.conversation_id != conversation.conversation_id:
            return None
        target_run = await conversation_runtime.event_store.require_run(
            target_event_id
        )
        if target_run.status not in {RunStatus.RUNNING, RunStatus.PAUSED}:
            return None
        event = create_agent_event(
            conversation_id=conversation.conversation_id,
            action=EventAction.CANCEL,
            payload_text="",
            target_event_id=target_event_id,
            origin=EventOrigin.FEISHU,
            reply_target_id=chat_id,
        )
        FEISHU_EVENT_CONTEXTS[event.event_id] = (
            chat_id,
            _build_progress_callback(chat_id),
        )
        try:
            # CANCEL is a control event. It intentionally owns no Agent run.
            await conversation_runtime.event_store.add_event(event)
        except BaseException:
            FEISHU_EVENT_CONTEXTS.pop(event.event_id, None)
            raise

    try:
        await _send_text(
            chat_id,
            "已收到取消请求；当前任务会在下一个安全节点停止，"
            "执行记录和已落盘归档会保留。",
        )
    finally:
        pump.notify(event)
    return event


async def _enqueue_feishu_replace(
    *,
    chat_id: str,
    payload_text: str,
    inbound: PreparedInbound | None = None,
) -> AgentEvent | None:
    """Persist `/replace` as a new run targeting the current execution."""

    pump = _require_event_pump()
    normalized_payload = payload_text.strip()
    if not normalized_payload:
        raise ValueError("替换任务内容不能为空。")
    async with OWNER_MESSAGE_LOCK:
        if inbound is not None:
            existing = await conversation_runtime.event_store.get_event(inbound.event_id)
            if existing is not None:
                return existing
        target_event_id = pump.active_event_id
        if target_event_id is None:
            return None
        conversation = await conversation_runtime.get_active_conversation(
            channel=FEISHU_CHANNEL,
            external_chat_id=OWNER_EXTERNAL_CHAT_ID,
        )
        target = await conversation_runtime.event_store.require_event(
            target_event_id
        )
        if target.conversation_id != conversation.conversation_id:
            return None
        if inbound is not None and inbound.conversation_id != conversation.conversation_id:
            return None
        target_run = await conversation_runtime.event_store.require_run(
            target_event_id
        )
        if target_run.status not in {RunStatus.RUNNING, RunStatus.PAUSED}:
            return None
        event = create_agent_event(
            conversation_id=conversation.conversation_id,
            action=EventAction.REPLACE,
            payload_text=inbound.payload(normalized_payload) if inbound else normalized_payload,
            event_id=inbound.event_id if inbound else None,
            target_event_id=target_event_id,
            origin=EventOrigin.FEISHU,
            reply_target_id=chat_id,
        )
        FEISHU_EVENT_CONTEXTS[event.event_id] = (
            chat_id,
            _build_progress_callback(chat_id),
        )
        try:
            if inbound is not None:
                await asyncio.to_thread(FEISHU_ATTACHMENTS.bind, inbound, event.event_id)
            await conversation_runtime.event_store.add_event(
                event,
                run=create_event_run(event),
            )
        except BaseException:
            FEISHU_EVENT_CONTEXTS.pop(event.event_id, None)
            raise

    try:
        await _send_text(
            chat_id,
            "已收到替换要求；当前任务会在下一个安全节点停止。"
            "系统将保留已验收成果，并由 Scheduler 生成一份全新的计划。",
        )
    finally:
        pump.notify(event)
    return event


async def _handle_pending_interaction(
    *,
    chat_id: str,
    user_text: str,
    inbound: PreparedInbound | None,
) -> bool:
    """Consume the next plain-text answer for a button-started interaction."""

    state = FEISHU_INTERACTIONS.get(chat_id)
    if state is None:
        return False
    command, _ = _parse_command(user_text)
    if command is not None:
        return False
    normalized = user_text.strip()
    if normalized.casefold() in {"退出", "exit", "cancel"}:
        FEISHU_INTERACTIONS.pop(chat_id, None)
        await _send_text(chat_id, "已退出当前交互模式。")
        return True
    if state.mode == "clean":
        action = {
            "a": "choose_a",
            "b": "choose_b",
            "跳过": "skip",
            "skip": "skip",
        }.get(normalized.casefold())
        if action is None:
            await _send_text(chat_id, "清理模式中请回复 A、B、跳过或退出。")
        else:
            await _handle_clean_action(chat_id, action)
        return True
    if state.mode not in {"insert", "replace"}:
        FEISHU_INTERACTIONS.pop(chat_id, None)
        return False
    if inbound is not None and inbound.attachments:
        await _send_text(
            chat_id,
            "快捷输入目前只接收文字。若要携带附件，请使用 /insert 内容 或 /replace 内容。",
        )
        return True
    try:
        if state.mode == "insert":
            await _enqueue_feishu_event(
                chat_id=chat_id,
                payload_text=normalized,
                action=EventAction.INSERT,
                inbound=None,
            )
        else:
            replacement = await _enqueue_feishu_replace(
                chat_id=chat_id,
                payload_text=normalized,
                inbound=None,
            )
            if replacement is None:
                FEISHU_INTERACTIONS.pop(chat_id, None)
                await _send_text(chat_id, "当前没有正在运行、可以替换的任务。")
                return True
    except EventConflictError:
        await _send_text(chat_id, "当前任务已经有一条取消或替换请求在等待安全节点。")
        return True
    except Exception:
        logger.exception("飞书快捷输入入队失败 | mode=%s", state.mode)
        await _send_text(chat_id, "任务没有成功进入队列，请稍后重试。")
        return True
    FEISHU_INTERACTIONS.pop(chat_id, None)
    return True

def _on_feishu_message_future_done(
    future: Future,
) -> None:
    """清理已经完成的飞书消息任务，并记录异常。"""

    FEISHU_MESSAGE_FUTURES.discard(
        future
    )

    try:
        # done_callback只会在Future完成后执行，
        # 所以这里不会阻塞任何线程。
        future.result()

    except CancelledError:
        # 程序关闭时主动取消任务，
        # 不属于业务异常。
        return

    except Exception:
        logger.exception(
            "飞书消息在Agent主事件循环中"
            "处理失败"
        )


def dispatch_feishu_message(
    message: Any,
) -> None:
    """把飞书SDK线程中的消息转交给Agent主事件循环。

    FeishuChannel的WebSocket Handler
    运行在SDK自己的线程和Event Loop中。

    ConversationRuntime、AsyncSqliteSaver、
    AsyncSqliteStore、LangGraph和MCP
    都是在应用主Event Loop中初始化的。

    因此飞书回调不能直接await
    handle_feishu_message，而应通过
    run_coroutine_threadsafe跨线程提交。
    """

    application_loop = (
        APPLICATION_LOOP
    )

    if application_loop is None:
        logger.error(
            "收到飞书消息时，"
            "Agent主事件循环尚未初始化"
        )

        return

    if application_loop.is_closed():
        logger.error(
            "收到飞书消息时，"
            "Agent主事件循环已经关闭"
        )

        return

    future = (
        asyncio.run_coroutine_threadsafe(
            handle_feishu_message(
                message
            ),

            application_loop,
        )
    )

    FEISHU_MESSAGE_FUTURES.add(
        future
    )

    future.add_done_callback(
        _on_feishu_message_future_done
    )


async def handle_feishu_card_action(event: Any) -> None:
    """Handle only host-authored card actions; never forward values to a model."""

    chat_id = str(getattr(event, "chat_id", "") or "").strip()
    operator_id = str(getattr(getattr(event, "operator", None), "open_id", "") or "")
    if not chat_id:
        return
    policy = FEISHU_EXPORTS.policy
    if policy.owners and (
        operator_id not in policy.owners
        or chat_id not in policy.chats
    ):
        await _send_text(chat_id, "当前个人助手仅对本机配置的授权用户和会话开放。")
        return
    value = getattr(getattr(event, "action", None), "value", None)
    if not isinstance(value, dict):
        await _send_text(chat_id, "无法识别这个按钮，请使用 /help。")
        return
    kind = str(value.get("kind", ""))
    action = str(value.get("action", ""))
    if kind == "personalops_memory_clean":
        await _handle_clean_action(
            chat_id,
            action,
            token=str(value.get("token", "")),
            group_id=str(value.get("group_id", "")),
            left_memory_id=str(value.get("left_memory_id", "")),
            right_memory_id=str(value.get("right_memory_id", "")),
        )
        return
    if kind != "personalops":
        await _send_text(chat_id, "无法识别这个按钮，请使用 /help。")
        return
    if action == "rag_upload":
        RAG_UPLOADS.arm(chat_id, operator_id)
        await _send_text(chat_id, "请发送下一条资料（文本、PDF、TXT、Markdown、JSON/JSONL、CSV、HTML、DOCX、XLSX、PPTX）；只存入 RAG，不执行任务。扫描版 PDF 会尝试 OCR，语音需有转写文本。")
    elif action == "help":
        await _start_command(chat_id)
    elif action in {"insert", "replace"}:
        await _send_input_prompt(chat_id, action)
    elif action == "cancel":
        await _cancel_running_task(chat_id)
    elif action == "clean":
        await _clean_command(chat_id, [])
    elif action == "exit":
        state = FEISHU_INTERACTIONS.get(chat_id)
        if state is not None and value.get("token") == state.token:
            FEISHU_INTERACTIONS.pop(chat_id, None)
            await _send_text(chat_id, "已退出当前输入模式。")
        else:
            await _send_text(chat_id, "这是旧按钮，当前输入状态没有改变。")
    else:
        await _send_text(chat_id, "无法识别这个按钮，请使用 /help。")


def dispatch_feishu_card_action(event: Any) -> None:
    application_loop = APPLICATION_LOOP
    if application_loop is None or application_loop.is_closed():
        logger.error("收到飞书卡片操作时，Agent主事件循环不可用")
        return
    future = asyncio.run_coroutine_threadsafe(
        handle_feishu_card_action(event),
        application_loop,
    )
    FEISHU_MESSAGE_FUTURES.add(future)
    future.add_done_callback(_on_feishu_message_future_done)


async def handle_feishu_message(
    message: Any,
) -> None:
    """接收飞书消息，并交给Agent处理。

    Channel SDK会把飞书消息规范化为统一对象。

    图片和文件先持久化，再将限定于该任务的附件清单传入队列。
    """

    chat_id = str(
        getattr(
            message,
            "chat_id",
            "",
        )
        or ""
    ).strip()

    if not chat_id:
        logger.warning(
            "收到缺少chat_id的飞书消息，"
            "本次已经忽略"
        )

        return

    # Control-only messages must not wait behind an attachment download.
    inbound = None
    user_text = FEISHU_ATTACHMENTS._clean_text(message)
    if not FEISHU_EXPORTS.admit_message(message):
        await _send_text(chat_id, "当前个人助手仅对本机配置的授权用户和会话开放。")
        return
    conversation = await conversation_runtime.get_active_conversation(
        channel=FEISHU_CHANNEL, external_chat_id=OWNER_EXTERNAL_CHAT_ID,
    )
    upload_reply = await RAG_UPLOADS.consume(
        message, FEISHU_ATTACHMENTS, feishu_channel,
        getattr(conversation_runtime, "retrieval_hub", None), conversation.conversation_id,
    )
    if upload_reply is not None:
        await _send_text(chat_id, upload_reply)
        await _send_control_panel(chat_id)
        return
    if await FEISHU_EXPORTS.handle_control(message, user_text):
        return
    command, _ = _parse_command(user_text)
    control_only = (
        command in {
            "start", "help", "new", "list", "switch", "current",
            "approve", "reject", "cancel", "insert", "replace", "clean", "exit",
        }
        and not getattr(message, "resources", ())
        and not getattr(message, "batched_sources", None)
    )
    if not control_only:
        try:
            async with OWNER_MESSAGE_LOCK:
                conversation = await conversation_runtime.get_active_conversation(
                    channel=FEISHU_CHANNEL,
                    external_chat_id=OWNER_EXTERNAL_CHAT_ID,
                )
            inbound = await FEISHU_ATTACHMENTS.prepare(
                message, conversation.conversation_id, feishu_channel,
            )
        except AttachmentIngressError as error:
            await _send_text(chat_id, f"消息未进入任务队列：{error}")
            return
        except Exception:
            logger.exception("飞书消息附件接收失败")
            await _send_text(
                chat_id,
                "消息未进入任务队列：附件可能无法下载、格式不支持、超出大小限制，"
                "或引用不属于当前会话。支持图片、PDF 和常见文档；每个文件最多 20 MiB，"
                "每条消息最多 8 个、合计 40 MiB。请检查机器人资源权限或重新发送。",
            )
            return
        user_text = inbound.instruction
    if not user_text:
        await _send_text(
            chat_id,
            inbound.acknowledgement() if inbound and inbound.attachments
            else "已收到消息。请发送文字指令，或上传图片、PDF 等文档附件。",
        )
        return

    if await _handle_pending_interaction(
        chat_id=chat_id,
        user_text=user_text,
        inbound=inbound,
    ):
        return

    logger.info("USER | %s", _compact_terminal_text(user_text))

    command, args = _parse_command(user_text)
    if command == "cancel":
        if args:
            await _send_text(chat_id, "用法：/cancel（取消当前运行任务）")
            return
        await _cancel_running_task(chat_id)
        return

    if command == "insert":
        if not args:
            await _send_input_prompt(chat_id, "insert")
            return
        try:
            await _enqueue_feishu_event(
                chat_id=chat_id,
                payload_text=" ".join(args),
                action=EventAction.INSERT,
                inbound=inbound,
            )
        except Exception:
            logger.exception("飞书INSERT事件入队失败")
            await _send_text(chat_id, "任务没有成功进入队列，请稍后重试。")
        return

    if command == "replace":
        if not args:
            await _send_input_prompt(chat_id, "replace")
            return
        try:
            replacement_event = await _enqueue_feishu_replace(
                chat_id=chat_id,
                payload_text=" ".join(args),
                inbound=inbound,
            )
        except EventConflictError:
            await _send_text(
                chat_id,
                "当前任务已经有一条取消或替换请求在等待安全节点。",
            )
            return
        except Exception:
            logger.exception("飞书REPLACE事件入队失败")
            await _send_text(chat_id, "替换请求没有成功保存，请稍后重试。")
            return
        if replacement_event is None:
            await _send_text(chat_id, "当前对话没有正在运行、可以替换的任务。")
        return

    if command is not None:
        async with OWNER_MESSAGE_LOCK:
            command_handled = await _handle_command(chat_id, command, args)
        if command_handled:
            return

    try:
        await _enqueue_feishu_event(
            chat_id=chat_id,
            payload_text=user_text,
            action=EventAction.QUEUE,
            inbound=inbound,
        )
    except Exception:
        logger.exception("飞书QUEUE事件入队失败")
        await _send_text(chat_id, "任务没有成功进入队列，请稍后重试。")


# 收到飞书消息后，
# Channel SDK先调用同步桥接函数，
# 再把Agent任务提交到应用主事件循环。
feishu_channel.on(
    "message",
    dispatch_feishu_message,
)
feishu_channel.on(
    "cardAction",
    dispatch_feishu_card_action,
)


async def run_application() -> None:
    """启动Agent Runtime和飞书后台长连接。

    Agent Runtime中的所有异步资源
    都归属于当前主事件循环。

    飞书SDK收到消息后，
    会通过dispatch_feishu_message
    把任务转交回这个主循环。
    """

    global APPLICATION_LOOP, FEISHU_EVENT_PUMP

    APPLICATION_LOOP = (
        asyncio.get_running_loop()
    )

    await conversation_runtime.start()

    FEISHU_EVENT_PUMP = EventRunPump(
        conversation_runtime.event_store,
        handler=_run_feishu_event,
        on_result=_deliver_feishu_event_result,
        on_failure=_deliver_feishu_event_failure,
        on_cancel=_finalize_feishu_cancelled_event,
        on_replace=_finalize_feishu_superseded_event,
    )
    FEISHU_EVENT_PUMP.start()

    conversation_runtime.schedule_service.configure_delivery_handler(
        DeliveryTarget.FEISHU,
        _deliver_scheduled_feishu_text,
    )
    conversation_runtime.schedule_service.configure_delivery_handler(
        DeliveryTarget.AGENT_EVENT,
        _enqueue_scheduled_agent_event,
    )

    try:
        await (
            feishu_channel
            .connect_until_ready(
                timeout=30.0
            )
        )

        logger.info(
            "飞书长连接已经就绪"
        )

        conversation_runtime.schedule_service.enable_dispatch()

        await asyncio.Event().wait()

    finally:
        conversation_runtime.schedule_service.disable_dispatch()
        # 第一步：阻止飞书SDK继续提交新的Agent任务。
        APPLICATION_LOOP = None

        # 第二步：取消并等待已经提交到主循环的消息任务。
        #
        # concurrent.futures.Future.cancel()
        # 只负责发出取消请求。
        #
        # 必须等待对应协程真正退出，
        # 才能安全关闭SQLite、Memory Store和MCP。
        pending_futures = list(
            FEISHU_MESSAGE_FUTURES
        )

        for future in pending_futures:
            if not future.done():
                future.cancel()

        if pending_futures:
            wrapped_futures = [
                asyncio.wrap_future(
                    future
                )

                for future in pending_futures
            ]

            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *wrapped_futures,
                        return_exceptions=True,
                    ),

                    timeout=10.0,
                )

            except TimeoutError:
                logger.warning(
                    "部分飞书消息任务未能在退出期限内结束，"
                    "将继续关闭Runtime | pending=%s",

                    sum(
                        1
                        for future in pending_futures
                        if not future.done()
                    ),
                )

            finally:
                for future in pending_futures:
                    if not future.done():
                        future.cancel()

        FEISHU_MESSAGE_FUTURES.clear()

        event_pump = FEISHU_EVENT_PUMP
        FEISHU_EVENT_PUMP = None
        if event_pump is not None:
            await event_pump.stop()
        FEISHU_EVENT_CONTEXTS.clear()
        FEISHU_INTERACTIONS.clear()

        # 第三步：关闭飞书连接，
        # 停止SDK后台线程继续收消息。
        try:
            await feishu_channel.disconnect()

        except Exception:
            logger.exception(
                "关闭飞书Channel时发生异常"
            )

        # 第四步：消息任务退出后，
        # 再关闭Agent底层资源。
        try:
            await conversation_runtime.stop()

        finally:
            shutdown_observability()

def main() -> None:
    """启动飞书个人Agent。"""

    try:
        asyncio.run(
            run_application()
        )

    except KeyboardInterrupt:
        logger.info(
            "收到退出信号，程序已停止"
        )


if __name__ == "__main__":
    main()
