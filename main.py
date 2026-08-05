

import asyncio
import logging
import os

from concurrent.futures import (
    CancelledError,
    Future,
)
from pathlib import Path

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


# 即使飞书在短时间内推送两条消息，
# 单用户Agent也严格按顺序处理。
#
# 这不是多用户并发设计，
# 只是防止同一个用户连续发送消息时，
# 两轮Agent执行互相覆盖状态。
OWNER_MESSAGE_LOCK = (
    asyncio.Lock()
)
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


logger.info(
    "Agent配置已加载 | "
    "platform=feishu | "
    "model=%s | tool_count=%s",

    settings.llm_model,

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

    await _send_text(
        chat_id,

        (
            "嗨～ ฅ^•ﻌ•^ฅ\n\n"

            "这是你的飞书个人Agent。\n\n"

            "对话命令：\n"
            "/new [标题] - 创建新对话\n"
            "/list - 查看对话列表\n"
            "/switch 编号 - 切换对话\n"
            "/current - 查看当前对话"
        ),
    )


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

    return False

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
async def handle_feishu_message(
    message: Any,
) -> None:
    """接收飞书消息，并交给Agent处理。

    Channel SDK会把飞书消息规范化为统一对象。

    当前使用：
    - message.chat_id：回复目标；
    - message.content_text：消息文字。
    """

    chat_id = str(
        getattr(
            message,
            "chat_id",
            "",
        )
        or ""
    ).strip()

    user_text = str(
        getattr(
            message,
            "content_text",
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

    if not user_text:
        await _send_text(
            chat_id,

            (
                "我已经收到这条消息啦 👀\n\n"
                "不过暂时只支持文字输入。"
            ),
        )

        return

    async with OWNER_MESSAGE_LOCK:
        logger.info(
            "USER | %s",

            _compact_terminal_text(
                user_text
            ),
        )

        (
            command,
            args,
        ) = _parse_command(
            user_text
        )

        if command is not None:
            command_handled = await (
                _handle_command(
                    chat_id,
                    command,
                    args,
                )
            )

            if command_handled:
                return

        progress_callback = (
            _build_progress_callback(
                chat_id
            )
        )

        try:
            agent_reply = await (
                conversation_runtime.ask(
                    user_text=user_text,

                    channel=(
                        FEISHU_CHANNEL
                    ),

                    external_chat_id=(
                        OWNER_EXTERNAL_CHAT_ID
                    ),

                    progress_callback=(
                        progress_callback
                    ),
                )
            )

        except Exception:
            logger.exception(
                "调用Agent时发生异常"
            )

            await _send_text(
                chat_id,

                (
                    "Agent处理消息时出现了一点问题 "
                    "(｡•́︿•̀｡)\n"
                    "详细错误已经记录在"
                    "PyCharm控制台中。"
                ),
            )

            return

        await _send_text(
            chat_id,
            agent_reply,
        )

        logger.info(
            "ASSISTANT | %s",

            _compact_terminal_text(
                agent_reply
            ),
        )


# 收到飞书消息后，
# Channel SDK先调用同步桥接函数，
# 再把Agent任务提交到应用主事件循环。
feishu_channel.on(
    "message",
    dispatch_feishu_message,
)


async def run_application() -> None:
    """启动Agent Runtime和飞书后台长连接。

    Agent Runtime中的所有异步资源
    都归属于当前主事件循环。

    飞书SDK收到消息后，
    会通过dispatch_feishu_message
    把任务转交回这个主循环。
    """

    global APPLICATION_LOOP

    APPLICATION_LOOP = (
        asyncio.get_running_loop()
    )

    await conversation_runtime.start()

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

        await asyncio.Event().wait()

    finally:
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