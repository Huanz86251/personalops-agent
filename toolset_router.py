from __future__ import annotations

import json
import logging

from dataclasses import (
    dataclass,
)

from typing import (
    Any,
    Mapping,
    Sequence,
)

from pydantic import BaseModel, ConfigDict, Field

from observability import (
    set_span_attributes,
    set_span_output,
    trace_span,
)

from prompt_loader import load_prompt

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


TOOLSET_ROUTING_TASK_MAX_CHARS = 1600

TOOLSET_ROUTER_MAX_TOOLSETS = 2
TOOLSET_ROUTING_MIN_SCORE = 0.4
TOOLSET_NO_TOOL_THRESHOLD = 0.4
TOOLSET_NO_TOOL_MARGIN = 0.0

# 更宽能力组已包含较窄组的执行能力时，不重复向下游暴露同一批工具。
TOOLSET_DOMINANCE = {
    "FILE_EDITING": frozenset({"FILE_INSPECTION"}),
    "SOFTWARE_DEVELOPMENT": frozenset({"FILE_INSPECTION", "FILE_EDITING"}),
    "APPWORLD": frozenset({"WEB_RESEARCH", "BROWSER_AUTOMATION"}),
}


class ModelToolsetSelection(BaseModel):
    """低置信本地路由后的同Worker模型选择。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    reason: str = Field(
        min_length=1,
        max_length=500,
        description="先说明当前Step为什么需要所选能力组。",
    )
    selected_toolsets: list[str] = Field(
        min_length=1,
        max_length=2,
        description="从本次提供的真实可用能力组名称中选择一到两个。",
    )


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
class ToolsetRouteScore:
    """保存一个候选组的原始相关性分数与决策阈值。"""

    name: str
    score: float
    threshold: float
    selected: bool


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

    route_scores: tuple[
        ToolsetRouteScore,
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
    """使用常驻Cross-Encoder选择工具组并展开真实工具。"""

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
        minimum_score: float = TOOLSET_ROUTING_MIN_SCORE,
        no_tool_threshold: float = TOOLSET_NO_TOOL_THRESHOLD,
        no_tool_margin: float = TOOLSET_NO_TOOL_MARGIN,
    ) -> None:
        if max_toolsets < 1:
            raise ValueError(
                "max_toolsets不能小于1。"
            )

        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError(
                "minimum_score必须在0到1之间。"
            )

        if not 0.0 <= no_tool_threshold <= 1.0:
            raise ValueError(
                "no_tool_threshold必须在0到1之间。"
            )

        if not 0.0 <= no_tool_margin <= 1.0:
            raise ValueError(
                "no_tool_margin必须在0到1之间。"
            )

        self.retrieval_models = (
            retrieval_models
        )

        self.registry = registry

        self.max_toolsets = (
            max_toolsets
        )

        self.minimum_score = minimum_score

        self.no_tool_threshold = (
            no_tool_threshold
        )

        self.no_tool_margin = (
            no_tool_margin
        )

    async def route(
        self,
        task_text: str,
        available_tools: Sequence[
            Any
        ],
        *,
        allow_no_tool: bool = True,
        user_request_text: str = "",
        step_weight: float = 0.4,
        user_weight: float = 0.6,
    ) -> ToolsetRouteDecision | None:
        """为当前请求选择工具组，并生成去重工具集合。"""

        routing_task = (
            _compact_routing_task(
                task_text
            )
        )

        user_routing_task = _compact_routing_task(user_request_text)
        use_weighted_scores = bool(user_routing_task)
        if use_weighted_scores:
            if step_weight < 0 or user_weight < 0 or step_weight + user_weight <= 0:
                raise ValueError("工具路由权重必须为非负数且总和大于0。")
            weight_total = step_weight + user_weight
            step_weight /= weight_total
            user_weight /= weight_total

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

        candidate_specs = [
            self.registry.get(item["name"])
            for item in router_metadata
        ]
        candidate_specs = [
            spec for spec in candidate_specs
            if spec is not None
        ]
        no_tool_profile = load_prompt("routing/toolsets/no_tool") if allow_no_tool else ""
        candidate_names = [
            *([NO_TOOL_ROUTE] if allow_no_tool else []),
            *(spec.name for spec in candidate_specs),
        ]
        candidate_documents = [
            *(
                [
                    _build_routing_document(
                        name=NO_TOOL_ROUTE,
                        description="无需读取外部状态或执行动作，可以直接用语言回答。",
                        profile=no_tool_profile,
                    )
                ]
                if allow_no_tool else []
            ),
            *(
                _build_routing_document(
                    name=spec.name,
                    description=spec.description,
                    profile=spec.routing_profile,
                )
                for spec in candidate_specs
            ),
        ]

        available_tool_names = tuple(
            _read_tool_name(
                tool
            )

            for tool in available_tools
        )

        scoring_task = routing_task
        scoring_user_task = user_routing_task
        fit_query = getattr(self.retrieval_models, "fit_rerank_query", None)
        if callable(fit_query):
            try:
                scoring_task = fit_query(routing_task, candidate_documents)
                if use_weighted_scores:
                    scoring_user_task = fit_query(user_routing_task, candidate_documents)
            except Exception:
                logger.exception("无法按真实Tokenizer约束工具路由query，将使用原始有界query")
                scoring_task = routing_task

        with trace_span(
            "toolset_routing",

            kind="chain",

            input_value={
                "routing_task": (
                    routing_task
                ),

                "effective_scoring_task": scoring_task,
                "user_request_routing_task": user_routing_task,
                "effective_user_scoring_task": scoring_user_task,

                "available_tool_names": (
                    available_tool_names
                ),

                "available_toolsets": (
                    router_metadata
                ),

                "candidate_names": (
                    candidate_names
                ),

                "max_toolsets": (
                    self.max_toolsets
                ),

                "scoring_input_kind": "task_text_vs_toolset_capability_document",
                "scoring_explanation": "任务文本与每个候选的名称、简介、正向场景和例子配对评分；不是工具参数 Schema，也不是仅按名称匹配。完整能力卡仅供审计。",
                "candidate_documents": [
                    {"name": name, "scoring_text": document,
                     "full_profile_for_audit_only": profile}
                    for name, document, profile in zip(
                        candidate_names, candidate_documents,
                        [*([no_tool_profile] if allow_no_tool else []), *(spec.routing_profile for spec in candidate_specs)],
                    )
                ],
                "scorer_implementation": type(self.retrieval_models).__module__ + "." + type(self.retrieval_models).__qualname__,
                "selection_method": "weighted_cross_encoder" if use_weighted_scores else "cross_encoder",
                "step_weight": step_weight if use_weighted_scores else 1.0,
                "user_weight": user_weight if use_weighted_scores else 0.0,
                "minimum_score": self.minimum_score,
                "no_tool_threshold": self.minimum_score,
                "no_tool_margin": 0.0,
                "allow_no_tool": allow_no_tool,
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

            try:
                step_ranked_results = await self.retrieval_models.arerank(
                    query=scoring_task,
                    documents=candidate_documents,
                    top_k=len(candidate_documents),
                )
                user_ranked_results = (
                    await self.retrieval_models.arerank(
                        query=scoring_user_task,
                        documents=candidate_documents,
                        top_k=len(candidate_documents),
                    )
                    if use_weighted_scores
                    else []
                )
            except Exception as error:
                logger.exception(
                    "Cross-Encoder工具组路由失败，本次交给上层保守回退"
                )
                set_span_attributes(
                    span,
                    **{
                        "toolset.route_valid": False,
                        "toolset.fallback_required": True,
                        "toolset.router_error": type(error).__name__,
                    },
                )
                set_span_output(
                    span,
                    {
                        "status": "cross_encoder_failed",
                        "error_type": type(error).__name__,
                        "selected_toolsets": [],
                        "fallback_required": True,
                    },
                )
                return None

            # Preserve actual returned evidence; missing scores below use a decision
            # default of zero and must not be mistaken for observed model scores.
            step_ranked_results = list(step_ranked_results)
            user_ranked_results = list(user_ranked_results)
            step_scores_by_name = {
                candidate_names[result.index]: float(result.score)
                for result in step_ranked_results
                if 0 <= result.index < len(candidate_names)
            }
            user_scores_by_name = {
                candidate_names[result.index]: float(result.score)
                for result in user_ranked_results
                if 0 <= result.index < len(candidate_names)
            }
            scores_by_name = {
                name: (
                    step_weight * step_scores_by_name.get(name, 0.0)
                    + user_weight * user_scores_by_name.get(name, 0.0)
                    if use_weighted_scores
                    else step_scores_by_name.get(name, 0.0)
                )
                for name in candidate_names
            }
            set_span_attributes(span, **{
                "toolset.scorer_returned_count": len(step_ranked_results) + len(user_ranked_results),
                "toolset.scorer_returned_scores": [
                    f"step:{result.index}:{float(result.score)}" for result in step_ranked_results
                ] + [
                    f"user:{result.index}:{float(result.score)}" for result in user_ranked_results
                ],
                "toolset.missing_score_policy": "decision_default_zero_not_observed_score",
            })
            score_items = [
                {
                    "name": name,
                    "score": round(scores_by_name.get(name, 0.0), 8),
                    "step_score": round(step_scores_by_name.get(name, 0.0), 8),
                    "user_score": round(user_scores_by_name.get(name, 0.0), 8) if use_weighted_scores else None,
                    "threshold": (
                        self.minimum_score
                    ),
                }
                for name in candidate_names
            ]
            ranked_names = sorted(
                candidate_names,
                key=lambda name: scores_by_name.get(name, 0.0),
                reverse=True,
            )
            best_name = ranked_names[0] if ranked_names else ""
            best_score = scores_by_name.get(best_name, 0.0) if best_name else 0.0

            # 只使用绝对阈值。加权分数仍不足时，上层直接调用同Worker
            # 模型做一次短结构化选择，不再展开全部工具。
            selected_names = (
                [best_name]
                if best_name and best_score > self.minimum_score
                else []
            )

            route_scores = tuple(
                ToolsetRouteScore(
                    name=item["name"],
                    score=item["score"],
                    threshold=item["threshold"],
                    selected=item["name"] in selected_names,
                )
                for item in score_items
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
                            "uncertain_scores"
                        ),

                        "scores": score_items,
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
                        route_scores=route_scores,
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

                        "selection_method": "weighted_cross_encoder" if use_weighted_scores else "cross_encoder",
                        "scores": score_items,
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
                route_scores=route_scores,
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

                    "selection_method": "weighted_cross_encoder" if use_weighted_scores else "cross_encoder",
                    "scores": score_items,
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


    def resolve_selected_toolsets(
        self,
        selected_names: Sequence[str],
        available_tools: Sequence[Any],
        *,
        routing_task: str,
        allow_no_tool: bool = True,
    ) -> ToolsetRouteDecision | None:
        """校验模型选择并展开当前真实可用工具组。"""
        available_tools = tuple(available_tools)
        metadata = self.registry.build_router_metadata(available_tools)
        available_names = {
            *(str(item.get("name") or "") for item in metadata),
        }
        if allow_no_tool:
            available_names.add(NO_TOOL_ROUTE)
        normalized: list[str] = []
        for raw_name in selected_names:
            name = str(raw_name or "").strip().upper()
            if name in available_names and name not in normalized:
                normalized.append(name)
        if not normalized:
            return None
        if allow_no_tool and NO_TOOL_ROUTE in normalized:
            return ToolsetRouteDecision(
                routing_task=_compact_routing_task(routing_task),
                selected_toolset_names=(NO_TOOL_ROUTE,),
                tools=(),
                routed_toolsets=(),
                route_scores=(),
            )
        normalized = _apply_toolset_dominance(normalized)[: self.max_toolsets]
        exclusive = [
            name for name in normalized
            if (self.registry.get(name) is not None and self.registry.get(name).exclusive)
        ]
        if exclusive:
            normalized = [exclusive[0]]

        routed: list[RoutedToolset] = []
        unique_tools: dict[str, Any] = {}
        for name in normalized:
            resolution = self.registry.resolve(name, available_tools)
            if not resolution.is_available:
                return None
            routed.append(RoutedToolset(
                name=resolution.spec.name,
                tool_names=resolution.tool_names,
                instructions=resolution.spec.instructions,
                missing_optional_tool_names=resolution.missing_optional_tool_names,
            ))
            for current_tool in resolution.tools:
                unique_tools.setdefault(_read_tool_name(current_tool), current_tool)
        return ToolsetRouteDecision(
            routing_task=_compact_routing_task(routing_task),
            selected_toolset_names=tuple(normalized),
            tools=tuple(unique_tools.values()),
            routed_toolsets=tuple(routed),
            route_scores=(),
        )

    async def route_with_model(
        self,
        model: Any,
        *,
        primary_task: str,
        user_request_window: str,
        available_tools: Sequence[Any],
        allow_no_tool: bool = True,
    ) -> ToolsetRouteDecision | None:
        """两轮本地分数都不足时，复用当前角色模型选择一到两个组。"""
        metadata = self.registry.build_router_metadata(tuple(available_tools))
        if not metadata or model is None or not hasattr(model, "with_structured_output"):
            return None
        catalog = [
            *(
                [{
                    "name": NO_TOOL_ROUTE,
                    "description": "当前Step无需读取外部状态或执行动作，可以直接回答。",
                }]
                if allow_no_tool else []
            ),
            *(
                {"name": item["name"], "description": item["description"]}
                for item in metadata
            ),
        ]
        messages = [
            {
                "role": "system",
                "content": load_prompt("routing/toolset_model_fallback")
                + "\n输出Schema："
                + json.dumps(ModelToolsetSelection.model_json_schema(), ensure_ascii=False, separators=(",", ":")),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "current_step": _compact_routing_task(primary_task),
                        "user_request": str(user_request_window or "").strip(),
                        "available_toolsets": catalog,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]
        try:
            structured = model.with_structured_output(
                ModelToolsetSelection, method="json_mode", include_raw=True,
            )
            response = await structured.ainvoke(messages)
            if isinstance(response, ModelToolsetSelection):
                parsed = response
            elif isinstance(response, Mapping):
                if response.get("parsing_error") is not None:
                    return None
                parsed = ModelToolsetSelection.model_validate(response.get("parsed"))
            else:
                parsed = ModelToolsetSelection.model_validate(response)
        except Exception:
            logger.exception("同Worker模型工具组回退失败，交给上层展示全部工具")
            return None
        return self.resolve_selected_toolsets(
            parsed.selected_toolsets,
            available_tools,
            routing_task=primary_task,
            allow_no_tool=allow_no_tool,
        )


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


def _build_routing_document(
    *,
    name: str,
    description: str,
    profile: str,
) -> str:
    """只把正向定义和正例交给相关性模型。

    完整能力卡仍保留反例与边界供人审查；通用Reranker并不可靠理解
    Markdown中的否定关系，把反例原文输入模型反而会制造关键词假阳性。
    """

    selected_lines: list[str] = []
    include_section = False
    positive_sections = {"选择它", "典型命令"}

    for raw_line in profile.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            include_section = line[3:].strip() in positive_sections
            continue
        if include_section and line:
            selected_lines.append(line)

    positive_text = "\n".join(selected_lines).strip()
    if not positive_text:
        raise ValueError(f"工具组{name}的能力卡缺少正向路由章节。")

    return (
        f"能力组：{name}\n"
        f"核心能力：{description.strip()}\n"
        f"适用场景与例子：\n{positive_text}"
    )


def _apply_toolset_dominance(selected_names: Sequence[str]) -> list[str]:
    """去掉已被更宽工具组完整覆盖的窄组，同时保留相关性顺序。"""

    selected_set = set(selected_names)
    suppressed = {
        covered_name
        for selected_name in selected_names
        for covered_name in TOOLSET_DOMINANCE.get(selected_name, ())
        if covered_name in selected_set
    }
    return [name for name in selected_names if name not in suppressed]


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
