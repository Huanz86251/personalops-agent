from __future__ import annotations

import logging

from dataclasses import (
    dataclass,
)

from typing import (
    Any,
    Sequence,
)

from observability import (
    set_span_attributes,
    set_span_output,
    trace_span,
)

from prompt_loader import (
    render_prompt,
)

from retrieval_models import (
    RetrievalModelManager,
)

from toolsets import (
    DEFAULT_TOOLSET_REGISTRY,
    NO_TOOL_ROUTE,
    ToolsetRegistry,
)


logger = logging.getLogger(
    "agent"
)


TOOLSET_ROUTING_TASK_MAX_CHARS = 400

TOOLSET_ROUTER_MAX_TOOLSETS = 3

TOOLSET_ROUTER_MAX_TOKENS = 64


@dataclass(frozen=True)
class RoutedToolset:
    """表示一个被Router选中的工具组。"""

    name: str

    tool_names: tuple[
        str,
        ...,
    ]

    instructions: str

    missing_optional_tool_names: tuple[
        str,
        ...,
    ]


@dataclass(frozen=True)
class ToolsetRouteDecision:
    """表示一次完整工具组路由结果。"""

    routing_task: str

    selected_toolset_names: tuple[
        str,
        ...,
    ]

    tools: tuple[
        Any,
        ...,
    ]

    routed_toolsets: tuple[
        RoutedToolset,
        ...,
    ]

    @property
    def tool_names(
        self,
    ) -> tuple[
        str,
        ...,
    ]:
        """返回最终去重后的工具名称。"""

        return tuple(
            _read_tool_name(
                tool
            )

            for tool in self.tools
        )

    @property
    def combined_instructions(
        self,
    ) -> str:
        """合并所有已选工具组的工作规则。"""

        instruction_blocks: list[
            str
        ] = []

        for routed_toolset in (
            self.routed_toolsets
        ):
            normalized_instructions = (
                routed_toolset
                .instructions
                .strip()
            )

            if not normalized_instructions:
                continue

            instruction_blocks.append(
                (
                    f"[{routed_toolset.name}]\n"
                    f"{normalized_instructions}"
                )
            )

        return "\n\n".join(
            instruction_blocks
        )


class ToolsetRouter:
    """使用本地Router选择工具组并展开真实工具。"""

    def __init__(
        self,
        retrieval_models: (
            RetrievalModelManager
        ),
        registry: ToolsetRegistry = (
            DEFAULT_TOOLSET_REGISTRY
        ),
        max_toolsets: int = (
            TOOLSET_ROUTER_MAX_TOOLSETS
        ),
        max_tokens: int = (
            TOOLSET_ROUTER_MAX_TOKENS
        ),
    ) -> None:
        if max_toolsets < 1:
            raise ValueError(
                "max_toolsets不能小于1。"
            )

        if max_tokens < 1:
            raise ValueError(
                "max_tokens不能小于1。"
            )

        self.retrieval_models = (
            retrieval_models
        )

        self.registry = registry

        self.max_toolsets = (
            max_toolsets
        )

        self.max_tokens = (
            max_tokens
        )

    async def route(
        self,
        task_text: str,
        available_tools: Sequence[
            Any
        ],
    ) -> ToolsetRouteDecision | None:
        """为当前请求选择工具组，并生成去重工具集合。"""

        routing_task = (
            _compact_routing_task(
                task_text
            )
        )

        if not routing_task:
            return None

        available_tools = tuple(
            available_tools
        )

        router_metadata = (
            self.registry
            .build_router_metadata(
                available_tools
            )
        )

        # 一个真实工具组都无法激活时，
        # 交给上层Middleware执行回退。
        if not router_metadata:
            logger.warning(
                "当前没有可以完整激活的工具组。"
            )

            return None

        allowed_labels = (
            self.registry
            .available_route_labels(
                available_tools
            )
        )

        toolset_metadata_text = (
            _format_toolset_metadata(
                router_metadata
            )
        )

        prompt = render_prompt(
            "toolset_router",

            routing_task=(
                routing_task
            ),

            toolset_metadata=(
                toolset_metadata_text
            ),
        )

        available_tool_names = tuple(
            _read_tool_name(
                tool
            )

            for tool in available_tools
        )

        with trace_span(
            "toolset_routing",

            kind="chain",

            input_value={
                "routing_task": (
                    routing_task
                ),

                "available_tool_names": (
                    available_tool_names
                ),

                "available_toolsets": (
                    router_metadata
                ),

                "allowed_labels": (
                    allowed_labels
                ),

                "max_toolsets": (
                    self.max_toolsets
                ),

                "thinking_enabled": False,

                "max_tokens": (
                    self.max_tokens
                ),
            },

            attributes={
                "toolset.available_count": len(
                    router_metadata
                ),

                "toolset.max_selected": (
                    self.max_toolsets
                ),

                "tools.available_count": len(
                    available_tools
                ),
            },
        ) as span:

            selected_names = await (
                self.retrieval_models
                .aclassify_many_with_router(
                    prompt=prompt,

                    allowed_labels=(
                        allowed_labels
                    ),

                    trace_name=(
                        "toolset_route"
                    ),

                    thinking=False,

                    max_tokens=(
                        self.max_tokens
                    ),

                    max_labels=(
                        self.max_toolsets
                    ),
                )
            )

            selected_names = (
                _normalize_selected_toolsets(
                    selected_names=(
                        selected_names
                    ),

                    allowed_labels=(
                        allowed_labels
                    ),

                    max_toolsets=(
                        self.max_toolsets
                    ),
                )
            )

            if not selected_names:
                set_span_attributes(
                    span,

                    **{
                        "toolset.route_valid": False,
                        "toolset.fallback_required": True,
                    },
                )

                set_span_output(
                    span,

                    {
                        "status": (
                            "router_failed"
                        ),

                        "selected_toolsets": [],

                        "fallback_required": True,
                    },
                )

                return None

            # NO_TOOL不是一个真实工具组，
            # 因此不需要执行工具展开。
            if selected_names == [
                NO_TOOL_ROUTE
            ]:
                decision = (
                    ToolsetRouteDecision(
                        routing_task=(
                            routing_task
                        ),

                        selected_toolset_names=(
                            NO_TOOL_ROUTE,
                        ),

                        tools=(),

                        routed_toolsets=(),
                    )
                )

                set_span_attributes(
                    span,

                    **{
                        "toolset.route_valid": True,
                        "toolset.selected_count": 1,
                        "tools.final_count": 0,
                        "tools.duplicate_count": 0,
                    },
                )

                set_span_output(
                    span,

                    {
                        "status": "success",

                        "selected_toolsets": [
                            NO_TOOL_ROUTE
                        ],

                        "expanded_tool_names": [],

                        "duplicate_tool_names": [],

                        "final_visible_tool_names": [],
                    },
                )

                return decision

            routed_toolsets: list[
                RoutedToolset
            ] = []

            expanded_tool_names: list[
                str
            ] = []

            duplicate_tool_names: list[
                str
            ] = []

            # 字典保持插入顺序。
            #
            # 第一个工具组中首次出现的工具
            # 会保留在最终列表中的原始位置。
            unique_tools_by_name: dict[
                str,
                Any,
            ] = {}

            for toolset_name in (
                selected_names
            ):
                resolution = (
                    self.registry
                    .resolve(
                        toolset_name,
                        available_tools,
                    )
                )

                # 正常情况下不会发生，
                # 因为不可用工具组不会提供给Router。
                #
                # 如果真实工具池在中间发生异常变化，
                # 则交给上层进行安全回退。
                if not resolution.is_available:
                    logger.warning(
                        "工具组缺少必要工具 | "
                        "toolset=%s | missing=%s",

                        toolset_name,

                        (
                            resolution
                            .missing_required_tool_names
                        ),
                    )

                    set_span_attributes(
                        span,

                        **{
                            "toolset.route_valid": False,
                            "toolset.fallback_required": True,
                        },
                    )

                    set_span_output(
                        span,

                        {
                            "status": (
                                "missing_required_tools"
                            ),

                            "selected_toolsets": (
                                selected_names
                            ),

                            "failed_toolset": (
                                toolset_name
                            ),

                            "missing_required_tools": (
                                resolution
                                .missing_required_tool_names
                            ),

                            "fallback_required": True,
                        },
                    )

                    return None

                routed_toolsets.append(
                    RoutedToolset(
                        name=(
                            resolution.spec.name
                        ),

                        tool_names=(
                            resolution.tool_names
                        ),

                        instructions=(
                            resolution
                            .spec
                            .instructions
                        ),

                        missing_optional_tool_names=(
                            resolution
                            .missing_optional_tool_names
                        ),
                    )
                )

                for tool in resolution.tools:
                    tool_name = (
                        _read_tool_name(
                            tool
                        )
                    )

                    expanded_tool_names.append(
                        tool_name
                    )

                    if (
                        tool_name
                        in unique_tools_by_name
                    ):
                        duplicate_tool_names.append(
                            tool_name
                        )

                        continue

                    unique_tools_by_name[
                        tool_name
                    ] = tool

            final_tools = tuple(
                unique_tools_by_name
                .values()
            )

            decision = ToolsetRouteDecision(
                routing_task=(
                    routing_task
                ),

                selected_toolset_names=tuple(
                    selected_names
                ),

                tools=(
                    final_tools
                ),

                routed_toolsets=tuple(
                    routed_toolsets
                ),
            )

            set_span_attributes(
                span,

                **{
                    "toolset.route_valid": True,

                    "toolset.selected_count": len(
                        selected_names
                    ),

                    "tools.expanded_count": len(
                        expanded_tool_names
                    ),

                    "tools.final_count": len(
                        final_tools
                    ),

                    "tools.duplicate_count": len(
                        duplicate_tool_names
                    ),

                    "toolset.fallback_required": False,
                },
            )

            set_span_output(
                span,

                {
                    "status": "success",

                    "selected_toolsets": (
                        selected_names
                    ),

                    "resolved_toolsets": [
                        {
                            "name": (
                                routed_toolset.name
                            ),

                            "tool_names": (
                                routed_toolset
                                .tool_names
                            ),

                            "missing_optional_tools": (
                                routed_toolset
                                .missing_optional_tool_names
                            ),

                            "instructions": (
                                routed_toolset
                                .instructions
                            ),
                        }

                        for routed_toolset
                        in routed_toolsets
                    ],

                    # 去重前的完整展开结果。
                    "expanded_tool_names": (
                        expanded_tool_names
                    ),

                    # 这些名称在后面的工具组中
                    # 再次出现，因此被跳过。
                    "duplicate_tool_names": (
                        duplicate_tool_names
                    ),

                    # 主模型最终真正能够看到的工具。
                    "final_visible_tool_names": (
                        decision.tool_names
                    ),

                    "combined_instructions": (
                        decision
                        .combined_instructions
                    ),

                    "fallback_required": False,
                },
            )

            return decision


def _compact_routing_task(
    text: str,
) -> str:
    """压缩并限制提供给本地Router的请求长度。"""

    normalized_text = (
        " ".join(
            text
            .strip()
            .split()
        )
    )

    if (
        len(
            normalized_text
        )
        <= TOOLSET_ROUTING_TASK_MAX_CHARS
    ):
        return normalized_text

    head_length = (
        TOOLSET_ROUTING_TASK_MAX_CHARS
        // 2
    )

    tail_length = (
        TOOLSET_ROUTING_TASK_MAX_CHARS
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


def _format_toolset_metadata(
    metadata: list[
        dict[
            str,
            str,
        ]
    ],
) -> str:
    """把工具组名称和描述转换成短文本。"""

    lines: list[
        str
    ] = []

    for item in metadata:
        toolset_name = str(
            item.get(
                "name",
                "",
            )
        ).strip()

        description = str(
            item.get(
                "description",
                "",
            )
        ).strip()

        if not toolset_name:
            continue

        lines.append(
            f"- {toolset_name}: "
            f"{description}"
        )

    return "\n".join(
        lines
    )


def _normalize_selected_toolsets(
    selected_names: Sequence[
        str
    ] | None,
    allowed_labels: Sequence[
        str
    ],
    max_toolsets: int,
) -> list[str]:
    """再次验证、去重并截断Router结果。"""

    if not selected_names:
        return []

    allowed_name_set = {
        label.strip().upper()

        for label in allowed_labels

        if label.strip()
    }

    normalized_names: list[
        str
    ] = []

    seen_names: set[
        str
    ] = set()

    for selected_name in selected_names:
        normalized_name = (
            str(
                selected_name
            )
            .strip()
            .upper()
        )

        if (
            normalized_name
            not in allowed_name_set
        ):
            continue

        if normalized_name in seen_names:
            continue

        seen_names.add(
            normalized_name
        )

        normalized_names.append(
            normalized_name
        )

    # 只要存在真实工具组，
    # NO_TOOL就失去意义。
    if (
        NO_TOOL_ROUTE
        in normalized_names

        and len(
            normalized_names
        ) > 1
    ):
        normalized_names = [
            name

            for name in normalized_names

            if name != NO_TOOL_ROUTE
        ]

    return normalized_names[
        :max_toolsets
    ]


def _read_tool_name(
    tool: Any,
) -> str:
    """读取真实LangChain Tool名称。"""

    return str(
        getattr(
            tool,
            "name",
            "",
        )
    ).strip()