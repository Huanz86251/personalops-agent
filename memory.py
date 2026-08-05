from dataclasses import (
    dataclass,
)

from langgraph.store.base import (
    BaseStore,
)

from retrieval_models import (
    RetrievalModelManager,
)
import asyncio

from memory_graph import (
    GraphMemoryHit,
    MemoryGraphIndex,
)
from prompt_loader import (
    load_prompt,
    render_prompt,
)
import logging
import json
import re
from datetime import (
    datetime,
    timezone,
)
from typing import (
    Any,
    Literal,
)
from observability import (
    set_span_attributes,
    set_span_output,
    trace_span,
)

from uuid import uuid4

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from zoneinfo import (
    ZoneInfo,
    ZoneInfoNotFoundError,
)
MEMORY_NAMESPACE = (
    "memories",
)
MemoryType = Literal[
    "profile",
    "preference",
    "routine",
    "project",
    "episode",
]
logger = logging.getLogger(
    "agent"
)


MemoryAction = Literal[
    "ADD",
    "COEXIST",
    "MERGE",
    "SUPERSEDE",
    "IGNORE",
]
MEMORY_WRITE_GATE_LABELS = (

    "RECORD",
    "NOT_RECORD",
)

MEMORY_RESOLUTION_GATE_LABELS = (
    "RELATED",
    "NOT_RELATED",
)


MEMORY_READ_GATE_LABELS = (
    "RELEVANT",
    "NOT_RELEVANT",
)


MEMORY_WRITE_GATE_MAX_CHARS = 400
MEMORY_WRITE_GATE_THINK_MAX_TOKENS = 600

MEMORY_RESOLUTION_CANDIDATE_MAX_CHARS = 120
MEMORY_RESOLUTION_EXISTING_MAX_CHARS = 120

MEMORY_READ_QUERY_MAX_CHARS = 220
MEMORY_READ_ITEM_MAX_CHARS = 220


# 只有完全匹配这些简短消息时才直接跳过。
# 不使用contains，避免误伤：
# “谢谢，不过以后请……”这样的消息。
MEMORY_WRITE_SKIP_TEXTS = frozenset(
    {
        "好",
        "好的",
        "行",
        "可以",
        "明白",
        "明白了",
        "知道了",
        "收到",
        "谢谢",
        "感谢",
        "继续",
        "继续吧",
        "没问题",
        "你确定吗",
        "再解释一下",
    }
)


# 明确的长期记忆信号。
# 匹配后只是强制进入云端提取，
# 并不会直接把原话写进数据库。
MEMORY_EXPLICIT_WRITE_PATTERNS = tuple(
    re.compile(
        pattern,
        flags=re.IGNORECASE,
    )
    for pattern in (
        # 明确要求写入记忆。
        (
            r"(?:请|帮我|给我)?"
            r"(?:记住|记下|记录下|"
            r"保存下来|存下来)"
        ),

        # 明确要求删除或修改记忆。
        (
            r"(?:请|帮我|给我)?"
            r"(?:忘记|忘掉|删除)"
        ),

        r"以后请",
        r"今后请",
        r"从现在开始",
        r"不要再",

        # 用户主动陈述长期偏好。
        r"喜欢",
        r"热爱",
        r"偏爱",
        r"讨厌",

        r"我的目标是",
        r"我的项目(?:改成|变成|现在是)",
    )
)
MEMORY_EXPLICIT_WRITE_PATTERNS = tuple(
    MEMORY_EXPLICIT_WRITE_PATTERNS
)
def _message_content_to_text(
    content: Any,
) -> str:
    """把不同模型的消息内容转换成普通字符串。"""

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
def _extract_json_object(
    text: str,
) -> str:
    """从模型文字中提取最外层JSON对象。"""

    normalized = (
        text
        .strip()
    )

    normalized = re.sub(
        r"^```(?:json)?\s*",
        "",
        normalized,
        flags=re.IGNORECASE,
    )

    normalized = re.sub(
        r"\s*```$",
        "",
        normalized,
    )

    start_index = (
        normalized.find(
            "{"
        )
    )

    end_index = (
        normalized.rfind(
            "}"
        )
    )

    if (
        start_index < 0
        or end_index < start_index
    ):
        raise RuntimeError(
            "模型没有返回可识别的JSON对象。"
        )

    return normalized[
        start_index:end_index + 1
    ]
def _parse_memory_timestamp(
    value: Any,
) -> datetime | None:
    """把记忆时间解析成UTC datetime。

    None表示没有明确的时间边界。
    非法时间返回None，主要用于兼容旧数据库。
    """

    if value is None:
        return None

    if isinstance(
        value,
        datetime,
    ):
        parsed_time = value

    elif isinstance(
        value,
        str,
    ):
        normalized_value = (
            value.strip()
        )

        if not normalized_value:
            return None

        # 兼容ISO 8601中的Z写法。
        if normalized_value.endswith(
            "Z"
        ):
            normalized_value = (
                normalized_value[:-1]
                + "+00:00"
            )

        try:
            parsed_time = (
                datetime.fromisoformat(
                    normalized_value
                )
            )

        except ValueError:
            return None

    else:
        return None

    # 不接受没有时区的信息，
    # 防止不同机器产生不同解释。
    if parsed_time.tzinfo is None:
        return None

    return parsed_time.astimezone(
        timezone.utc
    )


def _normalize_memory_timestamp(
    value: Any,
) -> str | None:
    """验证时间并统一保存为UTC ISO 8601字符串。"""

    if value is None:
        return None

    parsed_time = (
        _parse_memory_timestamp(
            value
        )
    )

    if parsed_time is None:
        raise ValueError(
            "时间必须是带时区的ISO 8601格式，"
            "或者使用null。"
        )

    return parsed_time.isoformat()
class MemoryTripleCandidate(
    BaseModel
):
    """表示从一条记忆中提取出的图关系。"""

    subject: str = Field(
        min_length=1,
        max_length=120,
    )

    subject_type: str = Field(
        min_length=1,
        max_length=50,
    )

    relation: str = Field(
        min_length=1,
        max_length=80,
    )

    object: str = Field(
        min_length=1,
        max_length=120,
    )

    object_type: str = Field(
        min_length=1,
        max_length=50,
    )
class MemoryCandidate(
    BaseModel
):
    """表示模型提取出的一条新记忆候选。"""

    content: str = Field(
        min_length=1,
        max_length=1200,
    )

    memory_type: MemoryType

    importance: int = Field(
        ge=1,
        le=4,
    )

    confidence: int = Field(
        ge=1,
        le=4,
    )

    # 事实从什么时候开始成立。
    #
    # 不知道时必须输出null，
    # 不能把created_at当成valid_from。
    valid_from: str | None

    # 事实从什么时候起不再作为当前事实使用。
    #
    # null表示没有明确截止时间，
    # 即无限期有效。
    expires_at: str | None

    triples: list[
        MemoryTripleCandidate
    ] = Field(
        default_factory=list,
    )

    @field_validator(
        "valid_from",
        "expires_at",
        mode="before",
    )
    @classmethod
    def normalize_memory_timestamps(
        cls,
        value: Any,
    ) -> str | None:
        return _normalize_memory_timestamp(
            value
        )

    @model_validator(
        mode="after",
    )
    def validate_memory_time_range(
        self,
    ) -> "MemoryCandidate":
        valid_from_time = (
            _parse_memory_timestamp(
                self.valid_from
            )
        )

        expires_at_time = (
            _parse_memory_timestamp(
                self.expires_at
            )
        )

        if (
            valid_from_time is not None
            and expires_at_time is not None
            and expires_at_time
            < valid_from_time
        ):
            raise ValueError(
                "expires_at不能早于valid_from。"
            )

        return self
class MemoryExtractionResult(
    BaseModel
):
    """表示一次对话轮次的记忆提取结果。"""

    memories: list[
        MemoryCandidate
    ] = Field(
        default_factory=list,
    )
class MemoryResolution(
    BaseModel
):
    """表示新旧记忆之间的处理决定。"""

    action: MemoryAction

    target_memory_ids: list[
        str
    ] = Field(
        default_factory=list,
    )

    final_content: (
        str
        | None
    ) = None

    final_memory_type: (
        MemoryType
        | None
    ) = None
    final_valid_from: (
        str
        | None
    ) = None

    final_expires_at: (
        str
        | None
    ) = None
    final_triples: list[
        MemoryTripleCandidate
    ] = Field(
        default_factory=list,
    )

    reason: str = ""
    @field_validator(
        "final_valid_from",
        "final_expires_at",
        mode="before",
    )
    @classmethod
    def normalize_final_timestamps(
        cls,
        value: Any,
    ) -> str | None:
        return _normalize_memory_timestamp(
            value
        )
@dataclass(frozen=True)
class RetrievedMemory:
    """表示经过召回或重排的一条长期记忆。"""

    memory_id: str
    content: str

    memory_type: str
    importance: int

    valid_from: (
        str
        | None
    ) = None

    expires_at: (
        str
        | None
    ) = None

    dense_score: (
        float
        | None
    ) = None

    rerank_score: (
        float
        | None
    ) = None

    # None表示不是通过图扩展召回。
    # 1或2表示距离种子记忆的图跳数。
    graph_distance: (
        int
        | None
    ) = None
def _retrieved_memory_to_trace_item(
    memory: RetrievedMemory,
) -> dict[
    str,
    Any,
]:
    """把一条召回记忆转换成Phoenix结构。"""

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

        "valid_from": (
            memory.valid_from
        ),

        "expires_at": (
            memory.expires_at
        ),

        "dense_score": (
            round(
                memory.dense_score,
                6,
            )

            if memory.dense_score
            is not None

            else None
        ),

        "rerank_score": (
            round(
                memory.rerank_score,
                6,
            )

            if memory.rerank_score
            is not None

            else None
        ),

        "graph_distance": (
            memory.graph_distance
        ),
    }


def _retrieved_memories_to_trace_items(
    memories: list[
        RetrievedMemory
    ],
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """批量转换召回记忆。"""

    return [
        _retrieved_memory_to_trace_item(
            memory
        )

        for memory
        in memories
    ]
def _memory_candidate_to_trace_item(
    candidate: MemoryCandidate,
) -> dict[
    str,
    Any,
]:
    """把记忆候选转换成稳定的Phoenix结构。"""

    return candidate.model_dump(
        mode="json"
    )


def _memory_resolution_to_trace_item(
    resolution: MemoryResolution,
) -> dict[
    str,
    Any,
]:
    """把记忆处理决策转换成稳定的Phoenix结构。"""

    return resolution.model_dump(
        mode="json"
    )
class MemoryService:
    """管理长期记忆的召回、重排和上下文生成。"""

    def __init__(
            self,
            store: BaseStore,
            retrieval_models: (
                    RetrievalModelManager
            ),
            model: Any,
            timezone_name: str = (
                    "Asia/Shanghai"
            ),
            dense_limit: int = 8,
            final_limit: int = 2,
            resolution_limit: int = 3,
            graph_hops: int = 2,
            graph_limit: int = 6,
    ) -> None:
        self.store = store

        self.retrieval_models = (
            retrieval_models
        )

        self.model = model
        try:
            self.timezone = (
                ZoneInfo(
                    timezone_name
                )
            )

        except ZoneInfoNotFoundError as error:
            raise ValueError(
                "无法识别记忆时区："
                f"{timezone_name}"
            ) from error

        self.timezone_name = (
            timezone_name
        )
        self.dense_limit = (
            dense_limit
        )

        self.final_limit = (
            final_limit
        )

        self.resolution_limit = (
            resolution_limit
        )

        self.graph_hops = (
            graph_hops
        )

        self.graph_limit = (
            graph_limit
        )

        self.graph_index = (
            MemoryGraphIndex()
        )
    async def retire_expired_memories(
        self,
        page_size: int = 200,
    ) -> list[str]:
        """扫描active记忆，把已经到期的记忆改成retired。"""

        now = datetime.now(
            timezone.utc
        )

        offset = 0

        expired_memory_ids: list[
            str
        ] = []

        invalid_items: list[
            dict[str, Any]
        ] = []

        while True:
            items = await self.store.asearch(
                MEMORY_NAMESPACE,

                filter={
                    "status": "active",
                },

                limit=(
                    page_size
                ),

                offset=(
                    offset
                ),
            )

            if not items:
                break

            for item in items:
                memory_id = getattr(
                    item,
                    "key",
                    "",
                )

                value = getattr(
                    item,
                    "value",
                    None,
                )

                if not isinstance(
                    memory_id,
                    str,
                ):
                    continue

                if not isinstance(
                    value,
                    dict,
                ):
                    continue

                raw_expires_at = value.get(
                    "expires_at"
                )

                # 旧记忆没有该字段，
                # 或者expires_at为null，
                # 都视为无限期有效。
                if raw_expires_at is None:
                    continue

                expires_at = (
                    _parse_memory_timestamp(
                        raw_expires_at
                    )
                )

                if expires_at is None:
                    invalid_items.append(
                        {
                            "memory_id": (
                                memory_id
                            ),

                            "content": (
                                value.get(
                                    "content",
                                    "",
                                )
                            ),

                            "expires_at": (
                                raw_expires_at
                            ),
                        }
                    )

                    continue

                if expires_at <= now:
                    expired_memory_ids.append(
                        memory_id
                    )

            if len(items) < page_size:
                break

            offset += len(
                items
            )

        # 必须全部扫描结束后再修改状态。
        #
        # 如果边分页边修改active结果，
        # offset可能跳过部分记录。
        if expired_memory_ids:
            await self._retire_memories(
                memory_ids=(
                    expired_memory_ids
                ),

                replaced_by=None,

                reason=(
                    "expired"
                ),
            )
        if invalid_items:
            logger.warning(
                "长期记忆中存在无法解析的过期时间 | "
                "count=%s",

                len(
                    invalid_items
                ),
            )

        return expired_memory_ids


        return expired_memory_ids
    async def rebuild_graph_index(
            self,
            page_size: int = 200,
    ) -> None:
        """从SQLite Store中的active记忆重建NetworkX图。"""

        self.graph_index.clear()

        offset = 0

        scanned_count = 0
        indexed_memory_count = 0
        indexed_edge_count = 0

        while True:
            items = await self.store.asearch(
                MEMORY_NAMESPACE,

                filter={
                    "status": "active",
                },

                limit=(
                    page_size
                ),

                offset=(
                    offset
                ),
            )

            if not items:
                break

            for item in items:
                memory_id = getattr(
                    item,
                    "key",
                    "",
                )

                value = getattr(
                    item,
                    "value",
                    None,
                )

                scanned_count += 1

                if not isinstance(
                        memory_id,
                        str,
                ):
                    continue

                if not isinstance(
                        value,
                        dict,
                ):
                    continue

                edge_count = (
                    self.graph_index
                    .add_memory(
                        memory_id=(
                            memory_id
                        ),

                        value=(
                            value
                        ),
                    )
                )

                if edge_count:
                    indexed_memory_count += 1

                    indexed_edge_count += (
                        edge_count
                    )

            if len(items) < page_size:
                break

            offset += len(
                items
            )



    @staticmethod
    def _value_to_memory(
            memory_id: str,
            value: dict[
                str,
                Any,
            ],
            *,
            dense_score: (
                    float
                    | None
            ) = None,
            graph_distance: (
                    int
                    | None
            ) = None,
    ) -> RetrievedMemory | None:
        """把Store中的value转换成统一记忆对象。"""

        if value.get(
                "status"
        ) != "active":
            return None

        expires_at = value.get(
            "expires_at"
        )

        expires_at_time = (
            _parse_memory_timestamp(
                expires_at
            )
        )

        # 即使程序启动时的扫描因为某种原因
        # 没有及时执行，已经过期的记忆
        # 也不能继续注入主模型。
        if (
            expires_at_time is not None
            and expires_at_time
            <= datetime.now(
                timezone.utc
            )
        ):
            return None

        content = value.get(
            "content"
        )

        if not isinstance(
            content,
            str,
        ):
            return None

        content = content.strip()

        if not content:
            return None

        memory_type = value.get(
            "memory_type",
            "general",
        )

        if not isinstance(
            memory_type,
            str,
        ):
            memory_type = "general"

        importance = value.get(
            "importance",
            2,
        )

        if not isinstance(
            importance,
            int,
        ):
            importance = 2

        valid_from = value.get(
            "valid_from"
        )

        if not isinstance(
            valid_from,
            str,
        ):
            valid_from = None

        if not isinstance(
            expires_at,
            str,
        ):
            expires_at = None

        return RetrievedMemory(
            memory_id=(
                memory_id
            ),

            content=(
                content
            ),

            memory_type=(
                memory_type
            ),

            importance=(
                importance
            ),

            valid_from=(
                valid_from
            ),

            expires_at=(
                expires_at
            ),

            dense_score=(
                dense_score
            ),

            graph_distance=(
                graph_distance
            ),
        )
    def _format_timestamp_for_model(
        self,
        value: str | None,
        *,
        missing_text: str,
    ) -> str:
        """把UTC时间转换成模型易读的本地时间。"""

        parsed_time = (
            _parse_memory_timestamp(
                value
            )
        )

        if parsed_time is None:
            return missing_text

        local_time = (
            parsed_time.astimezone(
                self.timezone
            )
        )

        return local_time.isoformat(
            timespec="minutes"
        )

    def _memory_to_model_text(
        self,
        memory: RetrievedMemory,
    ) -> str:
        """组合记忆正文和时间范围。"""

        valid_from_text = (
            self._format_timestamp_for_model(
                memory.valid_from,

                missing_text=(
                    "未注明"
                ),
            )
        )

        expires_at_text = (
            self._format_timestamp_for_model(
                memory.expires_at,

                missing_text=(
                    "无限期"
                ),
            )
        )

        return (
            "[有效开始："
            f"{valid_from_text}"
            "；有效截止："
            f"{expires_at_text}"
            "] "
            f"{memory.content}"
        )
    @staticmethod
    def _compact_router_text(
            text: str,
            max_chars: int = (
                    MEMORY_WRITE_GATE_MAX_CHARS
            ),
    ) -> str:
        """压缩交给本地小模型的文本长度。"""

        normalized_text = (
            " ".join(
                text.split()
            )
        )

        if len(normalized_text) <= max_chars:
            return normalized_text

        # 同时保留开头和结尾。
        # 用户经常在开头描述背景，
        # 在结尾给出真正的决定或要求。
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

    @staticmethod
    def _route_memory_write_by_rule(
            user_text: str,
    ) -> str | None:
        """用确定性规则处理最明确的情况。"""

        normalized_text = re.sub(
            r"[\s。.!！?？,，、]+",
            "",
            user_text,
        ).casefold()

        if not normalized_text:
            return "NOT_RECORD"

        if normalized_text in (
                MEMORY_WRITE_SKIP_TEXTS
        ):
            return "NOT_RECORD"

        for pattern in (
                MEMORY_EXPLICIT_WRITE_PATTERNS
        ):
            if pattern.search(
                    user_text
            ):
                return "RECORD"

        # 规则无法判断时，
        # 再交给本地小模型。
        return None

    async def classify_memory_write_level(
            self,
            user_text: str,
    ) -> str | None:
        """判断本轮是否值得进入云端记忆提取。"""

        with trace_span(
                "memory.write_gate",

                # 这一阶段的本质是：
                # 判断本轮是否值得写入长期记忆。
                kind="evaluator",

                input_value={
                    "user_message": (
                            user_text
                    ),

                    "allowed_labels": list(
                        MEMORY_WRITE_GATE_LABELS
                    ),

                    "fallback_policy": (
                            "router失败时继续进入云端提取"
                    ),
                },

                attributes={
                    "memory.write_gate.input_chars": len(
                        user_text
                    ),
                },
        ) as span:

            rule_label = (
                self._route_memory_write_by_rule(
                    user_text
                )
            )

            if rule_label is not None:
                should_extract = (
                        rule_label
                        != "NOT_RECORD"
                )

                set_span_attributes(
                    span,

                    **{
                        "memory.write_gate.source": (
                            "deterministic_rule"
                        ),

                        "memory.write_gate.label": (
                            rule_label
                        ),

                        "memory.write_gate.should_extract": (
                            should_extract
                        ),

                        "memory.write_gate.fallback_used": (
                            False
                        ),
                    },
                )

                set_span_output(
                    span,

                    {
                        "decision_source": (
                            "deterministic_rule"
                        ),

                        "selected_label": (
                            rule_label
                        ),

                        "should_enter_extraction": (
                            should_extract
                        ),

                        "fallback_used": (
                            False
                        ),
                    },
                )

                return rule_label

            compact_text = (
                self._compact_router_text(
                    user_text
                )
            )

            prompt = render_prompt(
                "memory_write_gate",

                user_message=(
                    compact_text
                ),
            )

            try:
                router_label = await (
                    self.retrieval_models
                    .aclassify_with_router(
                        prompt=prompt,

                        allowed_labels=(
                            MEMORY_WRITE_GATE_LABELS
                        ),

                        trace_name=(
                            "记忆写入门控"
                        ),

                        thinking=True,

                        max_tokens=(
                            MEMORY_WRITE_GATE_THINK_MAX_TOKENS
                        ),
                    )
                )

            except Exception as error:
                # 这里不重新抛出。
                #
                # 写入门控失败时，
                # 当前业务策略是保守进入云端提取，
                # 而不是让整个后台任务失败。
                logger.exception(
                    "本地记忆写入门控失败，"
                    "本轮将保守进入云端提取"
                )

                set_span_attributes(
                    span,

                    **{
                        "memory.write_gate.source": (
                            "local_router"
                        ),

                        "memory.write_gate.label": (
                            "NONE"
                        ),

                        "memory.write_gate.should_extract": (
                            True
                        ),

                        "memory.write_gate.fallback_used": (
                            True
                        ),

                        "memory.write_gate.status": (
                            "degraded"
                        ),
                    },
                )

                set_span_output(
                    span,

                    {
                        "decision_source": (
                            "local_router"
                        ),

                        "selected_label": None,

                        "should_enter_extraction": (
                            True
                        ),

                        "fallback_used": (
                            True
                        ),

                        "fallback_reason": (
                            "router_exception"
                        ),

                        "error": (
                            f"{type(error).__name__}: "
                            f"{error}"
                        ),
                    },
                )

                return None

            if router_label is None:
                logger.warning(
                    "本地记忆写入门控没有返回合法标签，"
                    "本轮将保守进入云端提取"
                )

                set_span_attributes(
                    span,

                    **{
                        "memory.write_gate.source": (
                            "local_router"
                        ),

                        "memory.write_gate.label": (
                            "NONE"
                        ),

                        "memory.write_gate.should_extract": (
                            True
                        ),

                        "memory.write_gate.fallback_used": (
                            True
                        ),

                        "memory.write_gate.status": (
                            "invalid_output"
                        ),
                    },
                )

                set_span_output(
                    span,

                    {
                        "decision_source": (
                            "local_router"
                        ),

                        "selected_label": None,

                        "should_enter_extraction": (
                            True
                        ),

                        "fallback_used": (
                            True
                        ),

                        "fallback_reason": (
                            "router_returned_no_valid_label"
                        ),
                    },
                )

                return None

            should_extract = (
                    router_label
                    != "NOT_RECORD"
            )

            set_span_attributes(
                span,

                **{
                    "memory.write_gate.source": (
                        "local_router"
                    ),

                    "memory.write_gate.label": (
                        router_label
                    ),

                    "memory.write_gate.should_extract": (
                        should_extract
                    ),

                    "memory.write_gate.fallback_used": (
                        False
                    ),
                },
            )

            set_span_output(
                span,

                {
                    "decision_source": (
                        "local_router"
                    ),

                    "selected_label": (
                        router_label
                    ),

                    "should_enter_extraction": (
                        should_extract
                    ),

                    "fallback_used": (
                        False
                    ),
                },
            )

            return router_label

    async def classify_memory_resolution_route(
            self,
            candidate: MemoryCandidate,
            related_memories: list[
                RetrievedMemory
            ],
    ) -> str | None:
        """判断候选与旧记忆是否属于同一事实维度。"""

        with trace_span(
                "memory.relation_gate",

                kind="evaluator",

                input_value={
                    "candidate": (
                            _memory_candidate_to_trace_item(
                                candidate
                            )
                    ),

                    "related_memories": (
                            _retrieved_memories_to_trace_items(
                                related_memories
                            )
                    ),

                    "allowed_labels": list(
                        MEMORY_RESOLUTION_GATE_LABELS
                    ),

                    "fallback_policy": (
                            "失败时升级到云端Resolver"
                    ),
                },

                attributes={
                    "memory.relation_gate.existing_count": (
                            len(
                                related_memories
                            )
                    ),
                },
        ) as span:

            if not related_memories:
                route_label = (
                    "NOT_RELATED"
                )

                set_span_attributes(
                    span,

                    **{
                        "memory.relation_gate.source": (
                            "deterministic_no_existing_memory"
                        ),

                        "memory.relation_gate.label": (
                            route_label
                        ),

                        "memory.relation_gate.cloud_required": (
                            False
                        ),
                    },
                )

                set_span_output(
                    span,

                    {
                        "decision_source": (
                            "deterministic_no_existing_memory"
                        ),

                        "selected_label": (
                            route_label
                        ),

                        "cloud_resolver_required": (
                            False
                        ),
                    },
                )

                return route_label

            compact_candidate = (
                self._compact_router_text(
                    candidate.content,

                    max_chars=(
                        MEMORY_RESOLUTION_CANDIDATE_MAX_CHARS
                    ),
                )
            )

            existing_memory_items = []

            for (
                    index,
                    memory,
            ) in enumerate(
                related_memories,
                start=1,
            ):
                compact_memory = (
                    self._compact_router_text(
                        memory.content,

                        max_chars=(
                            MEMORY_RESOLUTION_EXISTING_MAX_CHARS
                        ),
                    )
                )

                existing_memory_items.append(
                    f"{index}. "
                    f"[{memory.memory_type}] "
                    f"{compact_memory}"
                )

            existing_memories_text = (
                "\n".join(
                    existing_memory_items
                )
            )

            prompt = render_prompt(
                "memory_resolution_gate",

                candidate_memory=(
                    compact_candidate
                ),

                existing_memories=(
                    existing_memories_text
                ),
            )

            try:
                route_label = await (
                    self.retrieval_models
                    .aclassify_with_router(
                        prompt=prompt,

                        allowed_labels=(
                            MEMORY_RESOLUTION_GATE_LABELS
                        ),

                        trace_name=(
                            "记忆关系门控"
                        ),
                    )
                )

            except Exception as error:
                logger.exception(
                    "本地记忆关系门控失败，"
                    "本候选将保守升级到云端Resolver"
                )

                set_span_attributes(
                    span,

                    **{
                        "memory.relation_gate.source": (
                            "local_router"
                        ),

                        "memory.relation_gate.label": (
                            "NONE"
                        ),

                        "memory.relation_gate.cloud_required": (
                            True
                        ),

                        "memory.relation_gate.status": (
                            "degraded"
                        ),
                    },
                )

                set_span_output(
                    span,

                    {
                        "decision_source": (
                            "local_router"
                        ),

                        "selected_label": None,

                        "cloud_resolver_required": (
                            True
                        ),

                        "fallback_reason": (
                            "router_exception"
                        ),

                        "error": (
                            f"{type(error).__name__}: "
                            f"{error}"
                        ),
                    },
                )

                return None

            if route_label is None:
                logger.warning(
                    "本地记忆关系门控没有返回合法标签，"
                    "本候选将保守升级到云端Resolver"
                )

                set_span_attributes(
                    span,

                    **{
                        "memory.relation_gate.source": (
                            "local_router"
                        ),

                        "memory.relation_gate.label": (
                            "NONE"
                        ),

                        "memory.relation_gate.cloud_required": (
                            True
                        ),

                        "memory.relation_gate.status": (
                            "invalid_output"
                        ),
                    },
                )

                set_span_output(
                    span,

                    {
                        "decision_source": (
                            "local_router"
                        ),

                        "selected_label": None,

                        "cloud_resolver_required": (
                            True
                        ),

                        "fallback_reason": (
                            "router_returned_no_valid_label"
                        ),
                    },
                )

                return None

            cloud_required = (
                    route_label
                    == "RELATED"
            )

            set_span_attributes(
                span,

                **{
                    "memory.relation_gate.source": (
                        "local_router"
                    ),

                    "memory.relation_gate.label": (
                        route_label
                    ),

                    "memory.relation_gate.cloud_required": (
                        cloud_required
                    ),
                },
            )

            set_span_output(
                span,

                {
                    "decision_source": (
                        "local_router"
                    ),

                    "selected_label": (
                        route_label
                    ),

                    "cloud_resolver_required": (
                        cloud_required
                    ),
                },
            )

            return route_label

    async def warmup_router_cache(
            self,
    ) -> None:
        """启动时预热三个长期记忆Gate的Prompt缓存。"""

        if not (
                self.retrieval_models
                        .router_enabled
        ):
            logger.info(
                "Memory Router未启用，"
                "跳过Gate Prompt缓存预热"
            )

            return

        warmup_requests = [
            (
                "记忆写入门控",

                render_prompt(
                    "memory_write_gate",

                    user_message=(
                        "我最近开始学习游泳。"
                    ),
                ),

                MEMORY_WRITE_GATE_LABELS,
            ),

            (
                "记忆关系门控",

                render_prompt(
                    "memory_resolution_gate",

                    candidate_memory=(
                        "用户最近开始学习游泳。"
                    ),

                    existing_memories=(
                        "1. [preference] "
                        "用户喜欢进行体育活动。"
                    ),
                ),

                MEMORY_RESOLUTION_GATE_LABELS,
            ),

            (
                "记忆读取门控",

                render_prompt(
                    "memory_read_gate",

                    user_message=(
                        "给我推荐适合初学者的运动。"
                    ),

                    memory_text=(
                        "用户最近开始学习游泳。"
                    ),
                ),

                MEMORY_READ_GATE_LABELS,
            ),
        ]

        logger.info(
            "开始预热Memory Router Gate缓存 | "
            "count=%s",
            len(
                warmup_requests
            ),
        )

        for (
                gate_name,
                prompt,
                allowed_labels,
        ) in warmup_requests:

            try:
                selected_label = await (
                    self.retrieval_models
                    .aclassify_with_router(
                        prompt=prompt,

                        allowed_labels=(
                            allowed_labels
                        ),

                        trace_name=(
                            f"{gate_name}预热"
                        ),
                    )
                )

            except Exception:
                logger.exception(
                    "Memory Router Gate缓存预热失败 | "
                    "gate=%s",
                    gate_name,
                )

                # 一个Gate预热失败，
                # 不应阻止其他模型和Telegram启动。
                continue

            logger.info(
                "Memory Router Gate缓存预热完成 | "
                "gate=%s | label=%s",
                gate_name,
                selected_label,
            )

        logger.info(
            "Memory Router Gate缓存预热阶段完成"
        )

    @classmethod
    def _item_to_memory(
            cls,
            item,
    ) -> RetrievedMemory | None:
        """把LangGraph Store结果转换成统一记忆对象。"""

        value = getattr(
            item,
            "value",
            None,
        )

        if not isinstance(
                value,
                dict,
        ):
            return None

        memory_id = getattr(
            item,
            "key",
            "",
        )

        if not isinstance(
                memory_id,
                str,
        ):
            memory_id = str(
                memory_id
            )

        dense_score = getattr(
            item,
            "score",
            None,
        )

        if not isinstance(
                dense_score,
                (
                        int,
                        float,
                ),
        ):
            dense_score = None

        return cls._value_to_memory(
            memory_id=(
                memory_id
            ),

            value=(
                value
            ),

            dense_score=(
                float(
                    dense_score
                )
                if dense_score
                   is not None
                else None
            ),
        )

    async def extract_candidates(
            self,
            user_text: str,
            assistant_text: str,
    ) -> list[MemoryCandidate]:
        """从一轮成功对话中提取长期记忆候选。"""

        current_time = datetime.now(
            self.timezone
        )

        payload = {
            "current_time": (
                current_time.isoformat(
                    timespec="seconds"
                )
            ),

            "timezone": (
                self.timezone_name
            ),

            "user_message": (
                user_text
            ),

            "assistant_reply": (
                assistant_text
            ),
        }

        system_prompt = load_prompt(
            "memory_extraction"
        )

        with trace_span(
                "memory.extraction",

                # 外层还负责JSON提取、Pydantic校验、
                # 最多保留三条候选等业务处理。
                #
                # 真正的模型调用由自动Instrumentation
                # 生成LLM子节点。
                kind="chain",

                input_value={
                    "system_prompt": (
                            system_prompt
                    ),

                    "user_payload": (
                            payload
                    ),

                    "candidate_limit": 3,
                },

                attributes={
                    "memory.extraction.user_chars": len(
                        user_text
                    ),

                    "memory.extraction.assistant_chars": len(
                        assistant_text
                    ),

                    "memory.extraction.candidate_limit": 3,
                },
        ) as span:

            response = await self.model.ainvoke(
                [
                    {
                        "role": "system",

                        "content": (
                            system_prompt
                        ),
                    },
                    {
                        "role": "user",

                        "content": json.dumps(
                            payload,
                            ensure_ascii=False,
                        ),
                    },
                ]
            )

            response_text = (
                _message_content_to_text(
                    response.content
                )
            )

            json_text: (
                    str
                    | None
            ) = None

            try:
                json_text = (
                    _extract_json_object(
                        response_text
                    )
                )

                result = (
                    MemoryExtractionResult
                    .model_validate_json(
                        json_text
                    )
                )

            except (
                    RuntimeError,
                    ValidationError,
            ) as error:
                set_span_attributes(
                    span,

                    **{
                        "memory.extraction.parse_status": (
                            "failed"
                        ),
                    },
                )

                set_span_output(
                    span,

                    {
                        "parse_status": (
                            "failed"
                        ),

                        "raw_model_output": (
                            response_text
                        ),

                        "extracted_json": (
                            json_text
                        ),

                        "error": (
                            f"{type(error).__name__}: "
                            f"{error}"
                        ),
                    },
                )

                raise RuntimeError(
                    "记忆提取结果不符合结构要求。"
                ) from error

            candidates = (
                result.memories[:3]
            )

            candidate_items = [
                _memory_candidate_to_trace_item(
                    candidate
                )

                for candidate
                in candidates
            ]

            set_span_attributes(
                span,

                **{
                    "memory.extraction.parse_status": (
                        "success"
                    ),

                    "memory.extraction.candidate_count": (
                        len(
                            candidates
                        )
                    ),
                },
            )

            set_span_output(
                span,

                {
                    "parse_status": (
                        "success"
                    ),

                    "raw_model_output": (
                        response_text
                    ),

                    "extracted_json": (
                        json_text
                    ),

                    "candidate_count": (
                        len(
                            candidates
                        )
                    ),

                    "parsed_candidates": (
                        candidate_items
                    ),
                },
            )

            return candidates
    async def _dense_retrieve(
        self,
        query: str,
    ) -> list[RetrievedMemory]:
        """通过LangGraph Store执行Dense语义召回。"""

        items = await self.store.asearch(
            MEMORY_NAMESPACE,

            query=query,

            filter={
                "status": "active",
            },

            limit=(
                self.dense_limit
            ),
        )

        memories: list[
            RetrievedMemory
        ] = []

        for item in items:
            memory = (
                self._item_to_memory(
                    item
                )
            )

            if memory is not None:
                memories.append(
                    memory
                )

        return memories

    async def _retrieve_graph_memories(
            self,
            seed_memories: list[
                RetrievedMemory
            ],
    ) -> list[RetrievedMemory]:
        """从向量种子记忆出发执行多跳图扩展。"""

        seed_memory_items = (
            _retrieved_memories_to_trace_items(
                seed_memories
            )
        )

        with trace_span(
                "memory.graph_expansion",

                kind="retriever",

                input_value={
                    "seed_memories": (
                            seed_memory_items
                    ),

                    "max_hops": (
                            self.graph_hops
                    ),

                    "result_limit": (
                            self.graph_limit
                    ),

                    "graph_stats_before": (
                            self.graph_index
                                    .stats()
                    ),
                },

                attributes={
                    "memory.graph.seed_count": (
                            len(
                                seed_memories
                            )
                    ),

                    "memory.graph.max_hops": (
                            self.graph_hops
                    ),

                    "memory.graph.result_limit": (
                            self.graph_limit
                    ),
                },
        ) as span:

            if not seed_memories:
                set_span_attributes(
                    span,

                    **{
                        "memory.graph.status": (
                            "skipped_no_seed"
                        ),

                        "memory.graph.hit_count": 0,

                        "memory.graph.result_count": 0,
                    },
                )

                set_span_output(
                    span,

                    {
                        "status": (
                            "skipped_no_seed"
                        ),

                        "raw_graph_hits": [],

                        "graph_memories": [],

                        "removed_stale_memory_ids": [],

                        "graph_stats_after": (
                            self.graph_index
                            .stats()
                        ),
                    },
                )

                return []

            graph_hits = (
                self.graph_index
                .expand_from_memory_ids(
                    seed_memory_ids=[
                        memory.memory_id
                        for memory
                        in seed_memories
                    ],

                    max_hops=(
                        self.graph_hops
                    ),

                    limit=(
                        self.graph_limit
                    ),
                )
            )

            graph_hit_items = [
                {
                    "memory_id": (
                        hit.memory_id
                    ),

                    "distance": (
                        hit.distance
                    ),
                }

                for hit
                in graph_hits
            ]

            if not graph_hits:
                set_span_attributes(
                    span,

                    **{
                        "memory.graph.status": (
                            "no_hits"
                        ),

                        "memory.graph.hit_count": 0,

                        "memory.graph.result_count": 0,
                    },
                )

                set_span_output(
                    span,

                    {
                        "status": (
                            "no_hits"
                        ),

                        "raw_graph_hits": [],

                        "graph_memories": [],

                        "removed_stale_memory_ids": [],

                        "graph_stats_after": (
                            self.graph_index
                            .stats()
                        ),
                    },
                )

                return []

            items = await asyncio.gather(
                *[
                    self.store.aget(
                        MEMORY_NAMESPACE,

                        hit.memory_id,
                    )

                    for hit
                    in graph_hits
                ]
            )

            graph_memories: list[
                RetrievedMemory
            ] = []

            stale_memory_ids: list[
                str
            ] = []

            for (
                    hit,
                    item,
            ) in zip(
                graph_hits,
                items,
            ):
                if item is None:
                    stale_memory_ids.append(
                        hit.memory_id
                    )

                    continue

                value = getattr(
                    item,
                    "value",
                    None,
                )

                if not isinstance(
                        value,
                        dict,
                ):
                    stale_memory_ids.append(
                        hit.memory_id
                    )

                    continue

                memory = (
                    self._value_to_memory(
                        memory_id=(
                            hit.memory_id
                        ),

                        value=(
                            value
                        ),

                        graph_distance=(
                            hit.distance
                        ),
                    )
                )

                if memory is None:
                    # 图中可能暂时残留：
                    #
                    # - 已经retired的记忆；
                    # - 已经过期的记忆；
                    # - 内容无效的旧记录。
                    #
                    # 它们不能参与当前召回，
                    # 同时从辅助图索引中移除。
                    stale_memory_ids.append(
                        hit.memory_id
                    )

                    continue

                graph_memories.append(
                    memory
                )

            removed_stale_edge_count = 0

            for memory_id in (
                    stale_memory_ids
            ):
                removed_stale_edge_count += (
                    self.graph_index
                    .remove_memory(
                        memory_id
                    )
                )

            set_span_attributes(
                span,

                **{
                    "memory.graph.status": (
                        "completed"
                    ),

                    "memory.graph.hit_count": (
                        len(
                            graph_hits
                        )
                    ),

                    "memory.graph.result_count": (
                        len(
                            graph_memories
                        )
                    ),

                    "memory.graph.stale_count": (
                        len(
                            stale_memory_ids
                        )
                    ),

                    "memory.graph.removed_stale_edge_count": (
                        removed_stale_edge_count
                    ),
                },
            )

            set_span_output(
                span,

                {
                    "status": (
                        "completed"
                    ),

                    "raw_graph_hits": (
                        graph_hit_items
                    ),

                    "graph_memories": (
                        _retrieved_memories_to_trace_items(
                            graph_memories
                        )
                    ),

                    "removed_stale_memory_ids": (
                        stale_memory_ids
                    ),

                    "removed_stale_edge_count": (
                        removed_stale_edge_count
                    ),

                    "graph_stats_after": (
                        self.graph_index
                        .stats()
                    ),
                },
            )

            return graph_memories

    @staticmethod
    def _merge_retrieval_candidates(
            *memory_groups: list[
                RetrievedMemory
            ],
    ) -> list[RetrievedMemory]:
        """按memory_id合并向量候选和图候选。"""

        merged: dict[
            str,
            RetrievedMemory,
        ] = {}

        order: list[
            str
        ] = []

        for memory_group in memory_groups:
            for memory in memory_group:
                existing = merged.get(
                    memory.memory_id
                )

                if existing is None:
                    merged[
                        memory.memory_id
                    ] = memory

                    order.append(
                        memory.memory_id
                    )

                    continue

                dense_score = (
                    existing.dense_score
                    if existing.dense_score
                       is not None
                    else memory.dense_score
                )

                graph_distances = [
                    distance
                    for distance in (
                        existing.graph_distance,
                        memory.graph_distance,
                    )
                    if distance is not None
                ]

                graph_distance = (
                    min(
                        graph_distances
                    )
                    if graph_distances
                    else None
                )

                rerank_scores = [
                    score
                    for score in (
                        existing.rerank_score,
                        memory.rerank_score,
                    )
                    if score is not None
                ]

                rerank_score = (
                    max(
                        rerank_scores
                    )
                    if rerank_scores
                    else None
                )

                merged[
                    memory.memory_id
                ] = RetrievedMemory(
                    memory_id=(
                        existing.memory_id
                    ),

                    content=(
                        existing.content
                    ),

                    memory_type=(
                        existing.memory_type
                    ),

                    importance=max(
                        existing.importance,
                        memory.importance,
                    ),
                    valid_from=(
                        existing.valid_from
                    ),

                    expires_at=(
                        existing.expires_at
                    ),
                    dense_score=(
                        dense_score
                    ),

                    rerank_score=(
                        rerank_score
                    ),

                    graph_distance=(
                        graph_distance
                    ),
                )

        return [
            merged[
                memory_id
            ]
            for memory_id
            in order
        ]

    async def _rerank(
        self,
        query: str,
        memories: list[
            RetrievedMemory
        ],
        top_k: int | None = None,
    ) -> list[RetrievedMemory]:
        """使用本地Cross-Encoder精排候选记忆。"""

        if not memories:
            return []

        resolved_limit = (
            self.final_limit
            if top_k is None
            else top_k
        )

        documents = [
            self._memory_to_model_text(
                memory
            )
            for memory in memories
        ]

        ranked_results = await (
            self.retrieval_models
            .arerank(
                query=query,
                documents=documents,

                top_k=min(
                    resolved_limit,
                    len(memories),
                ),
            )
        )

        reranked_memories: list[
            RetrievedMemory
        ] = []

        for result in ranked_results:
            original = memories[
                result.index
            ]

            reranked_memories.append(
                RetrievedMemory(
                    memory_id=(
                        original.memory_id
                    ),

                    content=(
                        original.content
                    ),

                    memory_type=(
                        original.memory_type
                    ),

                    importance=(
                        original.importance
                    ),
                    valid_from=(
                        original.valid_from
                    ),

                    expires_at=(
                        original.expires_at
                    ),
                    dense_score=(
                        original.dense_score
                    ),

                    rerank_score=(
                        result.score
                    ),
                    graph_distance=(
                        original.graph_distance
                    ),
                )

            )

        return reranked_memories

    async def resolve_candidate(
            self,
            candidate: MemoryCandidate,
    ) -> MemoryResolution:
        """判断候选应新增、合并、替代、共存或忽略。"""

        candidate_item = (
            _memory_candidate_to_trace_item(
                candidate
            )
        )

        candidate_triples = [
            triple.model_dump(
                mode="json"
            )

            for triple
            in candidate.triples
        ]

        # 第一阶段：
        # 使用NetworkX图索引执行确定性完全去重。
        with trace_span(
                "memory.duplicate_check",

                kind="evaluator",

                input_value={
                    "candidate": (
                            candidate_item
                    ),

                    "matching_rule": (
                            "content + triples + time range完全相同"
                    ),
                },
        ) as duplicate_span:

            duplicate_memory_ids = (
                self.graph_index
                .find_duplicate_memory_ids(
                    content=(
                        candidate.content
                    ),

                    triples=(
                        candidate_triples
                    ),

                    valid_from=(
                        candidate.valid_from
                    ),

                    expires_at=(
                        candidate.expires_at
                    ),
                )
            )

            is_duplicate = bool(
                duplicate_memory_ids
            )

            set_span_attributes(
                duplicate_span,

                **{
                    "memory.duplicate_check.is_duplicate": (
                        is_duplicate
                    ),

                    "memory.duplicate_check.match_count": (
                        len(
                            duplicate_memory_ids
                        )
                    ),
                },
            )

            set_span_output(
                duplicate_span,

                {
                    "is_duplicate": (
                        is_duplicate
                    ),

                    "duplicate_memory_ids": (
                        duplicate_memory_ids
                    ),
                },
            )

        if duplicate_memory_ids:
            return MemoryResolution(
                action="IGNORE",

                target_memory_ids=(
                    duplicate_memory_ids
                ),

                reason=(
                    "图索引发现内容、三元组和时间范围"
                    "完全相同的active记忆，"
                    "因此确定性忽略重复候选。"
                ),
            )

        # 第二阶段：
        # 召回语义相近的旧记忆并执行CE精排。
        with trace_span(
                "memory.resolution_retrieval",

                kind="retriever",

                input_value={
                    "query": (
                            candidate.content
                    ),

                    "dense_limit": (
                            self.dense_limit
                    ),

                    "resolution_limit": (
                            self.resolution_limit
                    ),
                },

                attributes={
                    "retrieval.stage": (
                            "memory_resolution"
                    ),

                    "retrieval.limit": (
                            self.resolution_limit
                    ),
                },
        ) as retrieval_span:

            dense_memories = await (
                self._dense_retrieve(
                    candidate.content
                )
            )

            if dense_memories:
                with trace_span(
                        (
                                "memory.resolution_retrieval."
                                "cross_encoder"
                        ),

                        kind="reranker",

                        input_value={
                            "query": (
                                    candidate.content
                            ),

                            "top_k": (
                                    self.resolution_limit
                            ),

                            "documents": (
                                    _retrieved_memories_to_trace_items(
                                        dense_memories
                                    )
                            ),
                        },

                        attributes={
                            "reranker.stage": (
                                    "memory_resolution"
                            ),

                            "reranker.input_count": (
                                    len(
                                        dense_memories
                                    )
                            ),

                            "reranker.top_k": (
                                    self.resolution_limit
                            ),
                        },
                ) as reranker_span:

                    related_memories = await (
                        self._rerank(
                            query=(
                                candidate.content
                            ),

                            memories=(
                                dense_memories
                            ),

                            top_k=(
                                self.resolution_limit
                            ),
                        )
                    )

                    set_span_output(
                        reranker_span,

                        {
                            "ranked_memories": (
                                _retrieved_memories_to_trace_items(
                                    related_memories
                                )
                            )
                        },
                    )

                retrieval_mode = (
                    "dense_then_cross_encoder"
                )

            else:
                related_memories = []

                retrieval_mode = (
                    "no_dense_candidates"
                )

            set_span_attributes(
                retrieval_span,

                **{
                    "retrieval.dense_count": (
                        len(
                            dense_memories
                        )
                    ),

                    "retrieval.selected_count": (
                        len(
                            related_memories
                        )
                    ),

                    "retrieval.mode": (
                        retrieval_mode
                    ),
                },
            )

            set_span_output(
                retrieval_span,

                {
                    "mode": (
                        retrieval_mode
                    ),

                    "dense_candidates": (
                        _retrieved_memories_to_trace_items(
                            dense_memories
                        )
                    ),

                    "related_memories": (
                        _retrieved_memories_to_trace_items(
                            related_memories
                        )
                    ),
                },
            )

        # 第三阶段：
        # 没有旧记忆时也经过同一个关系门控接口，
        # 但走确定性NOT_RELATED，不会调用本地模型。
        route_label = await (
            self.classify_memory_resolution_route(
                candidate=(
                    candidate
                ),

                related_memories=(
                    related_memories
                ),
            )
        )

        if route_label == "NOT_RELATED":
            return MemoryResolution(
                action="ADD",

                reason=(
                    "本地关系门控判断候选记忆"
                    "与召回旧记忆不属于同一事实维度。"
                ),
            )

        # RELATED或Router失败返回None时，
        # 都升级到云端Resolver。
        payload = {
            "candidate": (
                candidate_item
            ),

            "existing_memories": [
                {
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

                    "valid_from": (
                        memory.valid_from
                    ),

                    "expires_at": (
                        memory.expires_at
                    ),
                }

                for memory
                in related_memories
            ],
        }

        system_prompt = load_prompt(
            "memory_resolution"
        )

        with trace_span(
                "memory.cloud_resolution",

                kind="chain",

                input_value={
                    "local_router_label": (
                            route_label
                    ),

                    "fallback_from_router_failure": (
                            route_label is None
                    ),

                    "system_prompt": (
                            system_prompt
                    ),

                    "user_payload": (
                            payload
                    ),
                },

                attributes={
                    "memory.cloud_resolution.existing_count": (
                            len(
                                related_memories
                            )
                    ),

                    "memory.cloud_resolution.router_fallback": (
                            route_label is None
                    ),
                },
        ) as resolver_span:

            response = await self.model.ainvoke(
                [
                    {
                        "role": "system",

                        "content": (
                            system_prompt
                        ),
                    },
                    {
                        "role": "user",

                        "content": json.dumps(
                            payload,
                            ensure_ascii=False,
                        ),
                    },
                ]
            )

            response_text = (
                _message_content_to_text(
                    response.content
                )
            )

            json_text: (
                    str
                    | None
            ) = None

            try:
                json_text = (
                    _extract_json_object(
                        response_text
                    )
                )

                resolution = (
                    MemoryResolution
                    .model_validate_json(
                        json_text
                    )
                )

            except (
                    RuntimeError,
                    ValidationError,
            ) as error:
                set_span_attributes(
                    resolver_span,

                    **{
                        "memory.cloud_resolution.parse_status": (
                            "failed"
                        ),
                    },
                )

                set_span_output(
                    resolver_span,

                    {
                        "parse_status": (
                            "failed"
                        ),

                        "raw_model_output": (
                            response_text
                        ),

                        "extracted_json": (
                            json_text
                        ),

                        "error": (
                            f"{type(error).__name__}: "
                            f"{error}"
                        ),
                    },
                )

                raise RuntimeError(
                    "记忆判断结果不符合结构要求。"
                ) from error

            allowed_ids = {
                memory.memory_id

                for memory
                in related_memories
            }

            safe_target_ids = [
                memory_id

                for memory_id
                in resolution.target_memory_ids

                if memory_id
                   in allowed_ids
            ]

            dropped_target_ids = [
                memory_id

                for memory_id
                in resolution.target_memory_ids

                if memory_id
                   not in allowed_ids
            ]

            safe_resolution = (
                resolution.model_copy(
                    update={
                        "target_memory_ids": (
                            safe_target_ids
                        ),
                    }
                )
            )

            set_span_attributes(
                resolver_span,

                **{
                    "memory.cloud_resolution.parse_status": (
                        "success"
                    ),

                    "memory.cloud_resolution.action": (
                        safe_resolution.action
                    ),

                    "memory.cloud_resolution.target_count": (
                        len(
                            safe_target_ids
                        )
                    ),

                    "memory.cloud_resolution.dropped_target_count": (
                        len(
                            dropped_target_ids
                        )
                    ),
                },
            )

            set_span_output(
                resolver_span,

                {
                    "parse_status": (
                        "success"
                    ),

                    "raw_model_output": (
                        response_text
                    ),

                    "extracted_json": (
                        json_text
                    ),

                    "parsed_resolution": (
                        _memory_resolution_to_trace_item(
                            safe_resolution
                        )
                    ),

                    "dropped_unsafe_target_ids": (
                        dropped_target_ids
                    ),
                },
            )

            return safe_resolution
    async def classify_memory_read_relevance(
            self,
            user_text: str,
            memory: RetrievedMemory,
    ) -> str | None:
        """判断一条记忆是否应该注入当前主模型请求。"""

        compact_user_text = (
            self._compact_router_text(
                user_text,

                max_chars=(
                    MEMORY_READ_QUERY_MAX_CHARS
                ),
            )
        )

        memory_text = (
            self._memory_to_model_text(
                memory
            )
        )

        compact_memory = (
            self._compact_router_text(
                memory_text,

                max_chars=(
                    MEMORY_READ_ITEM_MAX_CHARS
                ),
            )
        )

        try:
            prompt = render_prompt(
                "memory_read_gate",

                user_message=(
                    compact_user_text
                ),

                memory_text=(
                    compact_memory
                ),
            )

            relevance_label = await (
                self.retrieval_models
                .aclassify_with_router(
                    prompt=prompt,

                    allowed_labels=(
                        MEMORY_READ_GATE_LABELS
                    ),

                    trace_name=(
                        "记忆读取门控"
                    ),
                )
            )

        except Exception:
            logger.exception(
                "本地记忆读取门控失败，"
                "本轮将不注入该记忆"
            )

            return None


        return relevance_label

    async def filter_memories_for_turn(
            self,
            user_text: str,
            memories: list[
                RetrievedMemory
            ],
    ) -> list[RetrievedMemory]:
        """逐条判断哪些长期记忆可以注入主模型。"""

        with trace_span(
                "memory.read_gate",

                # 这一层负责组织多条候选的逐条判断。
                #
                # 每条候选的本地模型判断
                # 会放在对应item子节点中。
                kind="chain",

                input_value={
                    "user_message": (
                            user_text
                    ),

                    "candidate_memories": (
                            _retrieved_memories_to_trace_items(
                                memories
                            )
                    ),

                    "allowed_labels": list(
                        MEMORY_READ_GATE_LABELS
                    ),

                    "failure_policy": (
                            "fail_closed"
                    ),
                },

                attributes={
                    "memory.read_gate.candidate_count": (
                            len(
                                memories
                            )
                    ),

                    "memory.read_gate.failure_policy": (
                            "fail_closed"
                    ),
                },
        ) as gate_span:

            if not memories:
                set_span_attributes(
                    gate_span,

                    **{
                        "memory.read_gate.selected_count": 0,

                        "memory.read_gate.rejected_count": 0,

                        "memory.read_gate.failed_count": 0,
                    },
                )

                set_span_output(
                    gate_span,

                    {
                        "decisions": [],

                        "selected_memories": [],
                    },
                )

                return []

            selected_memories: list[
                RetrievedMemory
            ] = []

            decisions: list[
                dict[
                    str,
                    Any,
                ]
            ] = []

            rejected_count = 0
            failed_count = 0

            for (
                    memory_index,
                    memory,
            ) in enumerate(
                memories,
                start=1,
            ):
                with trace_span(
                        (
                                "memory.read_gate."
                                f"item_{memory_index}"
                        ),

                        # 这一层表示：
                        # 对单条候选进行相关性判定。
                        kind="evaluator",

                        input_value={
                            "user_message": (
                                    user_text
                            ),

                            "candidate_memory": (
                                    _retrieved_memory_to_trace_item(
                                        memory
                                    )
                            ),
                        },

                        attributes={
                            "memory.read_gate.item_index": (
                                    memory_index
                            ),

                            "memory.id": (
                                    memory.memory_id
                            ),
                        },
                ) as item_span:

                    relevance_label = await (
                        self.classify_memory_read_relevance(
                            user_text=(
                                user_text
                            ),

                            memory=(
                                memory
                            ),
                        )
                    )

                    should_include = (
                            relevance_label
                            == "RELEVANT"
                    )

                    if relevance_label is None:
                        decision_status = (
                            "router_failed_or_invalid"
                        )

                        failed_count += 1

                    elif should_include:
                        decision_status = (
                            "accepted"
                        )

                    else:
                        decision_status = (
                            "rejected"
                        )

                        rejected_count += 1

                    decision = {
                        "memory_id": (
                            memory.memory_id
                        ),

                        "content": (
                            memory.content
                        ),

                        "dense_score": (
                            round(
                                memory.dense_score,
                                6,
                            )

                            if memory.dense_score
                               is not None

                            else None
                        ),

                        "rerank_score": (
                            round(
                                memory.rerank_score,
                                6,
                            )

                            if memory.rerank_score
                               is not None

                            else None
                        ),

                        "graph_distance": (
                            memory.graph_distance
                        ),

                        "router_label": (
                            relevance_label
                        ),

                        "decision_status": (
                            decision_status
                        ),

                        "will_be_injected": (
                            should_include
                        ),
                    }

                    decisions.append(
                        decision
                    )

                    set_span_attributes(
                        item_span,

                        **{
                            "memory.read_gate.label": (
                                    relevance_label
                                    or "NONE"
                            ),

                            "memory.read_gate.accepted": (
                                should_include
                            ),

                            "memory.read_gate.decision_status": (
                                decision_status
                            ),
                        },
                    )

                    set_span_output(
                        item_span,

                        decision,
                    )

                    # 只允许明确RELEVANT的记忆通过。
                    #
                    # NOT_RELEVANT和Router失败返回的None
                    # 都不会注入主模型。
                    if should_include:
                        selected_memories.append(
                            memory
                        )

            set_span_attributes(
                gate_span,

                **{
                    "memory.read_gate.selected_count": (
                        len(
                            selected_memories
                        )
                    ),

                    "memory.read_gate.rejected_count": (
                        rejected_count
                    ),

                    "memory.read_gate.failed_count": (
                        failed_count
                    ),
                },
            )

            set_span_output(
                gate_span,

                {
                    "decisions": (
                        decisions
                    ),

                    "selected_memories": (
                        _retrieved_memories_to_trace_items(
                            selected_memories
                        )
                    ),
                },
            )

            return selected_memories

    async def _store_new_memory(
            self,
            candidate: MemoryCandidate,
            resolution: MemoryResolution,
            source_platform: str,
            source_conversation_id: str,
            source_thread_id: str,
    ) -> str:
        """把最终确定的记忆写入Store并同步图索引。"""

        with trace_span(
                "memory.store_new",

                kind="chain",

                input_value={
                    "candidate": (
                            _memory_candidate_to_trace_item(
                                candidate
                            )
                    ),

                    "resolution": (
                            _memory_resolution_to_trace_item(
                                resolution
                            )
                    ),

                    "source_platform": (
                            source_platform
                    ),

                    "source_conversation_id": (
                            source_conversation_id
                    ),

                    "source_thread_id": (
                            source_thread_id
                    ),
                },
        ) as span:

            memory_id = (
                f"mem_{uuid4().hex}"
            )

            now = datetime.now(
                timezone.utc
            ).isoformat()

            final_content = (
                    resolution.final_content
                    or candidate.content
            ).strip()

            final_memory_type = (
                    resolution.final_memory_type
                    or candidate.memory_type
            )

            final_triples = (
                    resolution.final_triples
                    or candidate.triples
            )

            final_valid_from = (
                candidate.valid_from
            )

            final_expires_at = (
                candidate.expires_at
            )

            if (
                    "final_valid_from"
                    in resolution.model_fields_set
            ):
                final_valid_from = (
                    resolution.final_valid_from
                )

            if (
                    "final_expires_at"
                    in resolution.model_fields_set
            ):
                final_expires_at = (
                    resolution.final_expires_at
                )

            replaces = (
                resolution.target_memory_ids

                if resolution.action
                   in {
                       "MERGE",
                       "SUPERSEDE",
                   }

                else []
            )

            value = {
                "content": (
                    final_content
                ),

                "memory_type": (
                    final_memory_type
                ),

                "status": "active",

                "valid_from": (
                    final_valid_from
                ),

                "expires_at": (
                    final_expires_at
                ),

                "importance": (
                    candidate.importance
                ),

                "confidence": (
                    candidate.confidence
                ),

                "triples": [
                    triple.model_dump(
                        mode="json"
                    )

                    for triple
                    in final_triples
                ],

                "created_at": (
                    now
                ),

                "updated_at": (
                    now
                ),

                "source_platform": (
                    source_platform
                ),

                "source_conversation_id": (
                    source_conversation_id
                ),

                "source_thread_id": (
                    source_thread_id
                ),

                "replaces": (
                    replaces
                ),

                "replaced_by": None,

                "resolution_action": (
                    resolution.action
                ),

                "resolution_reason": (
                    resolution.reason
                ),
            }

            await self.store.aput(
                MEMORY_NAMESPACE,

                memory_id,

                value,
            )

            graph_sync_status = (
                "success"
            )

            graph_edge_count = 0

            graph_sync_error: (
                    str
                    | None
            ) = None

            try:
                graph_edge_count = (
                    self.graph_index
                    .add_memory(
                        memory_id=(
                            memory_id
                        ),

                        value=(
                            value
                        ),
                    )
                )

            except Exception as error:
                # SQLite Store是真相源。
                #
                # 图同步失败不回滚已经成功的持久化。
                graph_sync_status = (
                    "failed"
                )

                graph_sync_error = (
                    f"{type(error).__name__}: "
                    f"{error}"
                )

                logger.exception(
                    "长期记忆已写入Store，"
                    "但同步到NetworkX图索引失败 | "
                    "memory_id=%s",

                    memory_id,
                )

            set_span_attributes(
                span,

                **{
                    "memory.id": (
                        memory_id
                    ),

                    "memory.type": (
                        final_memory_type
                    ),

                    "memory.resolution_action": (
                        resolution.action
                    ),

                    "memory.graph_sync_status": (
                        graph_sync_status
                    ),

                    "memory.graph_edge_count": (
                        graph_edge_count
                    ),
                },
            )

            set_span_output(
                span,

                {
                    "memory_id": (
                        memory_id
                    ),

                    "stored_value": (
                        value
                    ),

                    "graph_sync": {
                        "status": (
                            graph_sync_status
                        ),

                        "edge_count": (
                            graph_edge_count
                        ),

                        "error": (
                            graph_sync_error
                        ),

                        "graph_stats": (
                            self.graph_index
                            .stats()
                        ),
                    },
                },
            )

            return memory_id

    async def _retire_memories(
            self,
            memory_ids: list[str],
            replaced_by: str | None,
            reason: str = "",
    ) -> list[
        dict[
            str,
            Any,
        ]
    ]:
        """把旧记忆标记为retired并返回处理结果。"""

        now = datetime.now(
            timezone.utc
        ).isoformat()

        results: list[
            dict[
                str,
                Any,
            ]
        ] = []

        for memory_id in (
                memory_ids
        ):
            item = await self.store.aget(
                MEMORY_NAMESPACE,

                memory_id,
            )

            if item is None:
                results.append(
                    {
                        "memory_id": (
                            memory_id
                        ),

                        "status": (
                            "not_found"
                        ),

                        "reason": (
                            reason
                        ),

                        "replaced_by": (
                            replaced_by
                        ),
                    }
                )

                continue

            value = getattr(
                item,
                "value",
                None,
            )

            if not isinstance(
                    value,
                    dict,
            ):
                results.append(
                    {
                        "memory_id": (
                            memory_id
                        ),

                        "status": (
                            "invalid_store_value"
                        ),

                        "reason": (
                            reason
                        ),

                        "replaced_by": (
                            replaced_by
                        ),
                    }
                )

                continue

            if value.get(
                    "status"
            ) == "retired":
                results.append(
                    {
                        "memory_id": (
                            memory_id
                        ),

                        "content": (
                            value.get(
                                "content",
                                "",
                            )
                        ),

                        "status": (
                            "already_retired"
                        ),

                        "reason": (
                            reason
                        ),

                        "replaced_by": (
                            value.get(
                                "replaced_by"
                            )
                        ),
                    }
                )

                continue

            old_content = value.get(
                "content",
                "",
            )

            updated_value = dict(
                value
            )

            updated_value[
                "status"
            ] = "retired"

            updated_value[
                "replaced_by"
            ] = replaced_by

            updated_value[
                "updated_at"
            ] = now

            await self.store.aput(
                MEMORY_NAMESPACE,

                memory_id,

                updated_value,
            )

            graph_sync_status = (
                "success"
            )

            removed_graph_edges = 0

            graph_sync_error: (
                    str
                    | None
            ) = None

            try:
                removed_graph_edges = (
                    self.graph_index
                    .remove_memory(
                        memory_id
                    )
                )

            except Exception as error:
                graph_sync_status = (
                    "failed"
                )

                graph_sync_error = (
                    f"{type(error).__name__}: "
                    f"{error}"
                )

                logger.exception(
                    "旧记忆已经在Store中废弃，"
                    "但从NetworkX图索引删除失败 | "
                    "memory_id=%s",

                    memory_id,
                )

            results.append(
                {
                    "memory_id": (
                        memory_id
                    ),

                    "content": (
                        old_content
                    ),

                    "status": (
                        "retired"
                    ),

                    "reason": (
                        reason
                    ),

                    "replaced_by": (
                        replaced_by
                    ),

                    "graph_sync": {
                        "status": (
                            graph_sync_status
                        ),

                        "removed_edge_count": (
                            removed_graph_edges
                        ),

                        "error": (
                            graph_sync_error
                        ),
                    },
                }
            )

        return results

    async def apply_resolution(
            self,
            candidate: MemoryCandidate,
            resolution: MemoryResolution,
            source_platform: str,
            source_conversation_id: str,
            source_thread_id: str,
    ) -> str | None:
        """根据最终Resolution更新长期记忆。"""

        with trace_span(
                "memory.apply_resolution",

                kind="chain",

                input_value={
                    "candidate": (
                            _memory_candidate_to_trace_item(
                                candidate
                            )
                    ),

                    "resolution": (
                            _memory_resolution_to_trace_item(
                                resolution
                            )
                    ),
                },

                attributes={
                    "memory.resolution_action": (
                            resolution.action
                    ),

                    "memory.candidate_confidence": (
                            candidate.confidence
                    ),
                },
        ) as span:

            if candidate.confidence < 3:
                set_span_attributes(
                    span,

                    **{
                        "memory.apply.status": (
                            "skipped_low_confidence"
                        ),
                    },
                )

                set_span_output(
                    span,

                    {
                        "status": (
                            "skipped_low_confidence"
                        ),

                        "stored_memory_id": None,

                        "retirement_results": [],
                    },
                )

                return None

            if resolution.action == "IGNORE":
                set_span_attributes(
                    span,

                    **{
                        "memory.apply.status": (
                            "ignored"
                        ),
                    },
                )

                set_span_output(
                    span,

                    {
                        "status": (
                            "ignored"
                        ),

                        "stored_memory_id": None,

                        "retirement_results": [],
                    },
                )

                return None

            memory_id = await (
                self._store_new_memory(
                    candidate=(
                        candidate
                    ),

                    resolution=(
                        resolution
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

            retirement_results: list[
                dict[
                    str,
                    Any,
                ]
            ] = []

            if (
                    resolution.action
                    in {
                "MERGE",
                "SUPERSEDE",
            }
                    and resolution.target_memory_ids
            ):
                with trace_span(
                        "memory.retire_old",

                        kind="chain",

                        input_value={
                            "memory_ids": (
                                    resolution
                                            .target_memory_ids
                            ),

                            "replaced_by": (
                                    memory_id
                            ),

                            "reason": (
                                    "superseded"
                            ),
                        },
                ) as retire_span:
                    retirement_results = await (
                        self._retire_memories(
                            memory_ids=(
                                resolution
                                .target_memory_ids
                            ),

                            replaced_by=(
                                memory_id
                            ),

                            reason=(
                                "superseded"
                            ),
                        )
                    )

                    retired_count = sum(
                        1

                        for result
                        in retirement_results

                        if result.get(
                            "status"
                        )
                        == "retired"
                    )

                    set_span_attributes(
                        retire_span,

                        **{
                            "memory.retire.requested_count": (
                                len(
                                    resolution
                                    .target_memory_ids
                                )
                            ),

                            "memory.retire.completed_count": (
                                retired_count
                            ),
                        },
                    )

                    set_span_output(
                        retire_span,

                        {
                            "requested_memory_ids": (
                                resolution
                                .target_memory_ids
                            ),

                            "results": (
                                retirement_results
                            ),
                        },
                    )

            set_span_attributes(
                span,

                **{
                    "memory.apply.status": (
                        "stored"
                    ),

                    "memory.apply.retirement_requested": (
                        bool(
                            retirement_results
                        )
                    ),
                },
            )

            set_span_output(
                span,

                {
                    "status": (
                        "stored"
                    ),

                    "stored_memory_id": (
                        memory_id
                    ),

                    "resolution_action": (
                        resolution.action
                    ),

                    "retirement_results": (
                        retirement_results
                    ),
                },
            )

            return memory_id

    async def consolidate_turn(
            self,
            user_text: str,
            assistant_text: str,
            source_platform: str,
            source_conversation_id: str,
            source_thread_id: str,
    ) -> list[str]:
        """从一轮成功对话中巩固跨Conversation长期记忆。"""

        # memory_consolidation根Span已经由
        # ConversationRuntime创建。
        #
        # 这里不再创建重复根节点。
        write_level = await (
            self.classify_memory_write_level(
                user_text
            )
        )

        if write_level == "NOT_RECORD":
            return []

        # write_level为None表示本地Gate失败，
        # 按当前策略仍然进入云端提取。
        candidates = await (
            self.extract_candidates(
                user_text=(
                    user_text
                ),

                assistant_text=(
                    assistant_text
                ),
            )
        )

        if not candidates:
            return []

        stored_memory_ids: list[
            str
        ] = []

        for (
                candidate_index,
                candidate,
        ) in enumerate(
            candidates,
            start=1,
        ):
            with trace_span(
                    (
                            "memory.candidate_"
                            f"{candidate_index}"
                    ),

                    kind="chain",

                    input_value={
                        "candidate_index": (
                                candidate_index
                        ),

                        "candidate": (
                                _memory_candidate_to_trace_item(
                                    candidate
                                )
                        ),
                    },

                    attributes={
                        "memory.candidate.index": (
                                candidate_index
                        ),

                        "memory.candidate.type": (
                                candidate.memory_type
                        ),

                        "memory.candidate.importance": (
                                candidate.importance
                        ),

                        "memory.candidate.confidence": (
                                candidate.confidence
                        ),
                    },
            ) as candidate_span:

                if candidate.confidence < 3:
                    set_span_attributes(
                        candidate_span,

                        **{
                            "memory.candidate.status": (
                                "skipped_low_confidence"
                            ),
                        },
                    )

                    set_span_output(
                        candidate_span,

                        {
                            "status": (
                                "skipped_low_confidence"
                            ),

                            "candidate": (
                                _memory_candidate_to_trace_item(
                                    candidate
                                )
                            ),

                            "resolution": None,

                            "stored_memory_id": None,
                        },
                    )

                    continue

                resolution = await (
                    self.resolve_candidate(
                        candidate
                    )
                )

                memory_id = await (
                    self.apply_resolution(
                        candidate=(
                            candidate
                        ),

                        resolution=(
                            resolution
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

                if memory_id is not None:
                    stored_memory_ids.append(
                        memory_id
                    )

                    candidate_status = (
                        "stored"
                    )

                elif resolution.action == "IGNORE":
                    candidate_status = (
                        "ignored"
                    )

                else:
                    candidate_status = (
                        "not_stored"
                    )

                set_span_attributes(
                    candidate_span,

                    **{
                        "memory.candidate.status": (
                            candidate_status
                        ),

                        "memory.candidate.action": (
                            resolution.action
                        ),

                        "memory.candidate.stored": (
                                memory_id
                                is not None
                        ),
                    },
                )

                set_span_output(
                    candidate_span,

                    {
                        "status": (
                            candidate_status
                        ),

                        "candidate": (
                            _memory_candidate_to_trace_item(
                                candidate
                            )
                        ),

                        "resolution": (
                            _memory_resolution_to_trace_item(
                                resolution
                            )
                        ),

                        "stored_memory_id": (
                            memory_id
                        ),
                    },
                )

        return stored_memory_ids

    async def retrieve_for_turn(
            self,
            user_text: str,
    ) -> list[RetrievedMemory]:
        """使用向量、图扩展、重排和读取门控召回记忆。"""

        query = (
            user_text.strip()
        )

        if not query:
            return []

        # 第一阶段：
        # Dense向量召回较宽的一批候选。
        with trace_span(
                "memory.dense_retrieval",

                kind="retriever",

                input_value={
                    "query": (
                            query
                    ),

                    "namespace": (
                            MEMORY_NAMESPACE
                    ),

                    "filter": {
                        "status": "active",
                    },

                    "limit": (
                            self.dense_limit
                    ),
                },

                attributes={
                    "retrieval.stage": (
                            "dense"
                    ),

                    "retrieval.limit": (
                            self.dense_limit
                    ),
                },
        ) as dense_span:

            dense_memories = await (
                self._dense_retrieve(
                    query
                )
            )

            set_span_attributes(
                dense_span,

                **{
                    "retrieval.result_count": (
                        len(
                            dense_memories
                        )
                    ),
                },
            )

            set_span_output(
                dense_span,

                {
                    "result_count": (
                        len(
                            dense_memories
                        )
                    ),

                    "candidates": (
                        _retrieved_memories_to_trace_items(
                            dense_memories
                        )
                    ),
                },
            )

        # 第二阶段：
        # 从Dense候选中选择可靠的图扩展种子。
        with trace_span(
                "memory.seed_ranking",

                kind="chain",

                input_value={
                    "query": (
                            query
                    ),

                    "candidate_count": (
                            len(
                                dense_memories
                            )
                    ),

                    "candidates": (
                            _retrieved_memories_to_trace_items(
                                dense_memories
                            )
                    ),
                },
        ) as seed_stage_span:

            if dense_memories:
                seed_top_k = min(
                    self.final_limit,

                    len(
                        dense_memories
                    ),
                )

                with trace_span(
                        (
                                "memory.seed_ranking."
                                "cross_encoder"
                        ),

                        kind="reranker",

                        input_value={
                            "query": (
                                    query
                            ),

                            "top_k": (
                                    seed_top_k
                            ),

                            "documents": (
                                    _retrieved_memories_to_trace_items(
                                        dense_memories
                                    )
                            ),
                        },

                        attributes={
                            "reranker.stage": (
                                    "memory_seed"
                            ),

                            "reranker.input_count": (
                                    len(
                                        dense_memories
                                    )
                            ),

                            "reranker.top_k": (
                                    seed_top_k
                            ),
                        },
                ) as seed_reranker_span:

                    seed_memories = await (
                        self._rerank(
                            query=(
                                query
                            ),

                            memories=(
                                dense_memories
                            ),

                            top_k=(
                                seed_top_k
                            ),
                        )
                    )

                    set_span_output(
                        seed_reranker_span,

                        {
                            "ranked_memories": (
                                _retrieved_memories_to_trace_items(
                                    seed_memories
                                )
                            )
                        },
                    )

                seed_status = (
                    "completed"
                )

            else:
                seed_top_k = 0

                seed_memories = []

                seed_status = (
                    "skipped_no_dense_candidates"
                )

            set_span_attributes(
                seed_stage_span,

                **{
                    "memory.seed_ranking.status": (
                        seed_status
                    ),

                    "memory.seed_ranking.selected_count": (
                        len(
                            seed_memories
                        )
                    ),
                },
            )

            set_span_output(
                seed_stage_span,

                {
                    "status": (
                        seed_status
                    ),

                    "top_k": (
                        seed_top_k
                    ),

                    "selected_seed_memories": (
                        _retrieved_memories_to_trace_items(
                            seed_memories
                        )
                    ),
                },
            )

        # 第三阶段：
        # 从种子记忆中的实体节点向外多跳扩展。
        graph_memories = await (
            self._retrieve_graph_memories(
                seed_memories
            )
        )

        # 第四阶段：
        # 图扩展只扩大候选集，
        # 最终候选仍由Cross-Encoder控制精度。
        with trace_span(
                "memory.final_ranking",

                kind="chain",

                input_value={
                    "query": (
                            query
                    ),

                    "dense_candidates": (
                            _retrieved_memories_to_trace_items(
                                dense_memories
                            )
                    ),

                    "graph_candidates": (
                            _retrieved_memories_to_trace_items(
                                graph_memories
                            )
                    ),
                },
        ) as final_stage_span:

            if graph_memories:
                combined_candidates = (
                    self
                    ._merge_retrieval_candidates(
                        dense_memories,

                        graph_memories,
                    )
                )

                with trace_span(
                        (
                                "memory.final_ranking."
                                "cross_encoder"
                        ),

                        kind="reranker",

                        input_value={
                            "query": (
                                    query
                            ),

                            "top_k": (
                                    self.final_limit
                            ),

                            "documents": (
                                    _retrieved_memories_to_trace_items(
                                        combined_candidates
                                    )
                            ),
                        },

                        attributes={
                            "reranker.stage": (
                                    "memory_final"
                            ),

                            "reranker.input_count": (
                                    len(
                                        combined_candidates
                                    )
                            ),

                            "reranker.top_k": (
                                    self.final_limit
                            ),
                        },
                ) as final_reranker_span:

                    final_reranked_memories = await (
                        self._rerank(
                            query=(
                                query
                            ),

                            memories=(
                                combined_candidates
                            ),

                            top_k=(
                                self.final_limit
                            ),
                        )
                    )

                    set_span_output(
                        final_reranker_span,

                        {
                            "ranked_memories": (
                                _retrieved_memories_to_trace_items(
                                    final_reranked_memories
                                )
                            )
                        },
                    )

                final_ranking_mode = (
                    "cross_encoder_after_graph_expansion"
                )

            else:
                # 没有图候选时，Dense候选集没有发生变化。
                #
                # 直接复用Seed Reranker结果，
                # 避免对同一批候选重复运行CE。
                combined_candidates = (
                    dense_memories
                )

                final_reranked_memories = (
                    seed_memories
                )

                final_ranking_mode = (
                    "reuse_seed_ranking"
                )

            set_span_attributes(
                final_stage_span,

                **{
                    "memory.final_ranking.mode": (
                        final_ranking_mode
                    ),

                    "memory.final_ranking.combined_count": (
                        len(
                            combined_candidates
                        )
                    ),

                    "memory.final_ranking.selected_count": (
                        len(
                            final_reranked_memories
                        )
                    ),
                },
            )

            set_span_output(
                final_stage_span,

                {
                    "mode": (
                        final_ranking_mode
                    ),

                    "combined_candidates": (
                        _retrieved_memories_to_trace_items(
                            combined_candidates
                        )
                    ),

                    "selected_memories": (
                        _retrieved_memories_to_trace_items(
                            final_reranked_memories
                        )
                    ),
                },
            )

        # 第五阶段：
        # 本地Memory Router进行最终逐条读取门控。
        filtered_memories = await (
            self.filter_memories_for_turn(
                user_text=(
                    query
                ),

                memories=(
                    final_reranked_memories
                ),
            )
        )

        return filtered_memories

    def format_context(
            self,
            memories: list[
                RetrievedMemory
            ],
    ) -> str:
        """把召回结果渲染成带时间的长期记忆上下文。"""

        if not memories:
            return ""

        memory_items = "\n".join(
            (
                f"{index}. "
                f"{self._memory_to_model_text(memory)}"
            )

            for (
                index,
                memory,
            ) in enumerate(
                memories,
                start=1,
            )
        )

        return render_prompt(
            "memory_context",

            memory_items=(
                memory_items
            ),
        )