import asyncio
import logging
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
    is_shell_tool_available,
)
from mcp_runtime import (
    PlaywrightMCPRuntime,
)
from uuid import uuid4

from langgraph.checkpoint.sqlite.aio import (
    AsyncSqliteSaver,
)
from memory import (
    MemoryService,
    RetrievedMemory,
)
from agent import (
    build_agent,
    build_hard_model,
    build_model,
    generate_conversation_title,
)
from config import Settings

from path import AGENT_DATA_ROOT
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
from progress_events import (
    ProgressCallback,
)
from tools.time_tools import (
    get_current_time,
)

from toolsets import (
    DEFAULT_TOOLSET_REGISTRY,
)
from context_middlewares import (
    summarize_conversation_history,
)

from planning_models import (
    DialogueMessage,
    PlanningContextPack,
)
logger = logging.getLogger(
    "agent"
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

        "dense_score": (
            dense_score
        ),

        "rerank_score": (
            rerank_score
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

        message_text = (
            _message_content_to_text(
                content
            )
        )

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


def _trim_recent_dialogue(
    messages: list[
        DialogueMessage
    ],
    *,
    max_chars: int,
) -> list[
    DialogueMessage
]:
    """保留近期Turn结构，同时限制提供给Hard节点的字符数。"""

    if not messages or max_chars < 1:
        return []

    total_chars = sum(
        len(
            message.content
        )
        for message in messages
    )

    if total_chars <= max_chars:
        return list(
            messages
        )

    message_count = len(
        messages
    )

    base_limit, extra_chars = divmod(
        max_chars,
        message_count,
    )

    trimmed_messages: list[
        DialogueMessage
    ] = []

    for index, message in enumerate(
        messages
    ):
        message_limit = (
            base_limit
            + (
                1
                if index < extra_chars
                else 0
            )
        )

        content = message.content

        if len(
            content
        ) > message_limit:
            marker = "……"

            if message_limit <= len(
                marker
            ):
                content = content[
                    :message_limit
                ]

            else:
                available_chars = (
                    message_limit
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

                content = (
                    content[
                        :head_chars
                    ]
                    + marker
                    + content[
                        -tail_chars:
                    ]
                )

        trimmed_messages.append(
            message.model_copy(
                update={
                    "content": content,
                }
            )
        )

    return trimmed_messages


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
class ConversationRuntime:
    """管理模型、Agent、Conversation和Checkpoint。"""

    def __init__(
        self,
        settings: Settings,
        tools: list,
    ) -> None:
        self.settings = settings
        self.tools = list(
            tools
        )
        self.playwright_mcp = (
            PlaywrightMCPRuntime()
        )
        self.model = None
        self.hard_model = None
        self.agent = None

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

    async def start(self) -> None:
        """启动Checkpoint、长期Store、本地模型和Agent。"""

        if self.agent is not None:
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
            model = build_model(
                self.settings
            )
            hard_model = build_hard_model(
                self.settings
            )

            memory_service = (
                MemoryService(
                    store=memory_store,

                    retrieval_models=(
                        retrieval_models
                    ),

                    model=model,

                    timezone_name=(
                        "Asia/Shanghai"
                    ),
                )
            )

            # 先把已经到期的active记忆
            # 统一改成retired。
            await (
                memory_service
                .retire_expired_memories()
            )

            # 再使用剩余active记忆重建图索引。
            await (
                memory_service
                .rebuild_graph_index()
            )

            playwright_tools = []

            try:
                await (
                    self.playwright_mcp
                    .start()
                )

                playwright_tools = (
                    self.playwright_mcp
                    .tools
                )

            except Exception:
                # 浏览器属于增强能力。
                #
                # MCP启动失败时，
                # 飞书、长期记忆、文件工具和Web Search
                # 仍然可以继续工作。
                logger.exception(
                    "Playwright MCP启动失败，"
                    "本次运行将回退为无浏览器模式。"
                )

            runtime_tools = [
                *self.tools,
                *playwright_tools,
            ]

            agent = build_agent(
                model=model,

                tools=(
                    runtime_tools
                ),

                checkpointer=(
                    checkpointer
                ),

                store=(
                    memory_store
                ),

                retrieval_models=(
                    retrieval_models
                ),
                planning=self.settings.planning
            )

            middleware_tool_names: tuple[
                str,
                ...
            ] = ()

            if is_shell_tool_available():
                middleware_tool_names = (
                    "shell",
                )

            toolset_catalog = (
                DEFAULT_TOOLSET_REGISTRY
                .build_router_metadata(
                    runtime_tools,

                    extra_available_tool_names=(
                        middleware_tool_names
                    ),
                )
            )

            planning_graph = (
                build_planning_graph(
                    simple_model=(
                        model
                    ),

                    hard_model=(
                        hard_model
                    ),

                    # 实际传入的是唯一的Simple Executor Agent。
                    step_agent=(
                        agent
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
                )
            )

        except Exception:
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

        self.model = model
        self.hard_model = hard_model
        self.agent = agent

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

        agent = self._require_agent()
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

        agent = self._require_agent()
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
    ) -> PlanningContextPack:
        """准备Hard节点共用的Conversation上下文。

        较早历史进入Rolling Summary；
        最近若干Turn保留用户与最终助手原文。
        所有阈值均来自PlanningSettings。
        """

        planning = (
            self.settings.planning
        )

        dialogue_turns = (
            _extract_dialogue_turns(
                list(
                    state_values.get(
                        "messages",
                        [],
                    )
                )
            )
        )

        recent_turn_count = min(
            planning
            .hard_recent_dialogue_turns,

            len(
                dialogue_turns
            ),
        )

        if recent_turn_count:
            summary_turns = (
                dialogue_turns[
                    :-recent_turn_count
                ]
            )

            recent_turns = (
                dialogue_turns[
                    -recent_turn_count:
                ]
            )

        else:
            summary_turns = (
                dialogue_turns
            )

            recent_turns = []

        summary_messages = [
            message
            for turn in summary_turns
            for message in turn
        ]

        recent_dialogue = [
            message
            for turn in recent_turns
            for message in turn
        ]

        recent_dialogue = (
            _trim_recent_dialogue(
                recent_dialogue,

                max_chars=(
                    planning
                    .hard_recent_dialogue_max_chars
                ),
            )
        )

        stored_summary = (
            state_values.get(
                "conversation_summary",
                "",
            )
        )

        if not isinstance(
            stored_summary,
            str,
        ):
            stored_summary = ""

        summarized_message_count = (
            state_values.get(
                "conversation_summary_message_count",
                0,
            )
        )

        if (
            isinstance(
                summarized_message_count,
                bool,
            )
            or not isinstance(
                summarized_message_count,
                int,
            )
            or summarized_message_count < 0
        ):
            summarized_message_count = 0

        summarized_message_count = min(
            summarized_message_count,
            len(
                summary_messages
            ),
        )

        pending_messages = (
            summary_messages[
                summarized_message_count:
            ]
        )

        pending_turn_count = sum(
            1
            for message in pending_messages
            if message.role == "user"
        )

        if (
            pending_messages
            and pending_turn_count
            >= planning
            .conversation_summary_trigger_turns
        ):
            stored_summary = await (
                summarize_conversation_history(
                    hard_model,

                    previous_summary=(
                        stored_summary
                    ),

                    messages=(
                        pending_messages
                    ),

                    max_chars=(
                        planning
                        .conversation_summary_max_chars
                    ),
                )
            )

            summarized_message_count = len(
                summary_messages
            )

            pending_messages = []

            await agent.aupdate_state(
                self._build_config(
                    conversation.thread_id
                ),

                {
                    "conversation_summary": (
                        stored_summary
                    ),

                    "conversation_summary_message_count": (
                        summarized_message_count
                    ),
                },

                # 摘要是ConversationRuntime对已有Conversation
                # 进行的外部元数据更新。
                #
                # 必须显式指定来源节点，
                # 防止Planning前的状态写入因多节点历史而歧义。
                as_node="model",
            )

        summary_blocks: list[str] = []

        if stored_summary.strip():
            summary_blocks.append(
                stored_summary.strip()
            )

        pending_history_text = (
            _format_dialogue_messages(
                pending_messages
            )
        )

        if pending_history_text:
            summary_blocks.append(
                (
                    "[尚未达到批量摘要阈值的较早对话]\n"
                    f"{pending_history_text}"
                )
            )

        conversation_summary = (
            _compact_context_text(
                "\n\n".join(
                    summary_blocks
                ),

                max_chars=(
                    planning
                    .conversation_summary_max_chars
                ),
            )

            if summary_blocks

            else ""
        )

        return PlanningContextPack(
            current_time=(
                get_current_time(
                    "Asia/Shanghai"
                )
            ),

            user_request=(
                user_request
            ),

            conversation_summary=(
                conversation_summary
            ),

            recent_dialogue=(
                recent_dialogue
            ),

            memory_context=(
                memory_context
            ),

            toolset_catalog=(
                self.toolset_catalog
            ),
        )
    async def ask(
            self,
            *,
            user_text: str,
            channel: str,
            external_chat_id: int | str,
            progress_callback: ProgressCallback,
    ) -> str:
        """在当前Conversation中完成一轮用户请求。"""

        normalized_user_text = (
            user_text.strip()
        )

        if not normalized_user_text:
            raise ValueError(
                "user_text不能为空。"
            )

        agent = self._require_agent()

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

        conversation = await (
            self.get_active_conversation(
                channel=channel,

                external_chat_id=(
                    external_chat_id
                ),
            )
        )

        thread_lock = (
            self._thread_locks
            .setdefault(
                conversation.thread_id,

                asyncio.Lock(),
            )
        )

        async with thread_lock:
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
                with trace_span(
                        "conversation_turn",

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
                ) as turn_span:

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

                            memories = await (
                                memory_service
                                .retrieve_for_turn(
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

                    # Hard Supervisor不在create_agent内部，
                    # 因此显式读取原Conversation中的历史，
                    # 准备Rolling Summary和最近对话。
                    conversation_state = await (
                        agent.aget_state(
                            self._build_config(
                                conversation
                                .thread_id
                            )
                        )
                    )

                    state_values = (
                        getattr(
                            conversation_state,
                            "values",
                            {},
                        )
                        or {}
                    )

                    planning_context = await (
                        self._prepare_planning_context(
                            agent=agent,

                            hard_model=(
                                hard_model
                            ),

                            conversation=(
                                conversation
                            ),

                            state_values=(
                                state_values
                            ),

                            user_request=(
                                normalized_user_text
                            ),

                            memory_context=(
                                memory_context
                            ),
                        )
                    )

                    planning_status = "success"

                    planning_result: dict[
                        str,
                        Any,
                    ] = {}

                    try:
                        planning_result = await (
                            planning_graph.ainvoke(
                                {
                                    "context": (
                                        planning_context
                                    ),

                                    "conversation_thread_id": (
                                        conversation
                                        .thread_id
                                    ),

                                    "planning_run_id": (
                                        uuid4().hex[:12]
                                    ),

                                    "model_rounds_used": 0,

                                    "tool_calls_used": 0,

                                    "completed_step_reports": [],

                                    "remaining_steps": [],

                                    "replan_history": [],

                                    "replans_used": 0,

                                    "retry_current_step": False,
                                },

                                config={
                                    "configurable": {
                                        "progress_callback": (
                                            progress_callback
                                        ),
                                    },
                                },
                            )
                        )
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

                    # 无论Supervisor直接FINAL，
                    # 还是Simple Agent在独立Step Thread中执行，
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
                        },

                        # 这里保存的是已经由Planning Graph生成好的
                        # 最终用户消息和最终助手回答。
                        #
                        # 不应重新让Simple Agent执行这些消息，
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
                        try:
                            final_title = await (
                                generate_conversation_title(
                                    model,

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

                        assistant_text=(
                            reply
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

                    return reply

    def _schedule_memory_consolidation(
            self,
            *,
            memory_service: MemoryService,
            user_text: str,
            assistant_text: str,
            source_platform: str,
            source_conversation_id: str,
            source_thread_id: str,
    ) -> None:
        """创建不继承当前Trace父节点的后台记忆任务。"""

        coroutine = (
            self._consolidate_memory_background(
                memory_service=(
                    memory_service
                ),

                user_text=(
                    user_text
                ),

                assistant_text=(
                    assistant_text
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
                "memory-consolidation-"
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
            assistant_text: str,
            source_platform: str,
            source_conversation_id: str,
            source_thread_id: str,
    ) -> None:
        """在独立Trace中按Conversation顺序巩固长期记忆。"""

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
                        "memory-consolidation",
                        "background",
                        source_platform,
                    ],
            ):
                try:
                    with trace_span(
                            "memory_consolidation",

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

                                "assistant_reply": (
                                        assistant_text
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

                                "memory.assistant_reply_chars": (
                                        len(
                                            assistant_text
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

                        stored_memory_ids = await (
                            memory_service
                            .consolidate_turn(
                                user_text=(
                                    user_text
                                ),

                                assistant_text=(
                                    assistant_text
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
                                "memory.stored_count": (
                                    len(
                                        stored_memory_ids
                                    )
                                ),
                            },
                        )

                        set_span_output(
                            span,

                            {
                                "stored_count": (
                                    len(
                                        stored_memory_ids
                                    )
                                ),

                                "stored_memory_ids": (
                                    stored_memory_ids
                                ),
                            },
                        )

                except Exception:
                    # memory_consolidation Span已经记录
                    # 完整异常和ERROR状态。
                    logger.exception(
                        "后台长期记忆巩固失败，"
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

        self._memory_store_context = None
        self._memory_store = None

        self._checkpointer_context = None
        self._checkpointer = None

        self.retrieval_models = None
        self.memory_service = None

        self.planning_graph = None
        self.toolset_catalog = []

        self.hard_model = None
        self.model = None
        self.agent = None

        self._thread_locks.clear()

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

        logger.info(
            "ConversationRuntime关闭流程完成。"
        )

    def _require_agent(
        self,
    ):
        if self.agent is None:
            raise RuntimeError(
                "ConversationRuntime尚未启动。"
            )

        return self.agent

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