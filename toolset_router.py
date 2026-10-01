from __future__ import annotations

import json
import logging

from dataclasses import (
    dataclass,
    replace,
)

from typing import (
    Any,
    Mapping,
    Sequence,
)

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from schema_normalization import coerce_text_sequence, drop_invalid_auxiliary_fields

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

TOOLSET_ROUTER_MAX_TOOLSETS = 3
TOOLSET_ROUTING_MIN_SCORE = 0.4
TOOLSET_NO_TOOL_THRESHOLD = 0.4
TOOLSET_NO_TOOL_MARGIN = 0.0
TOOLSET_BM25_CANDIDATES = 3
TOOLSET_CROSS_ENCODER_CANDIDATES = 5
TOOLSET_HYBRID_CANDIDATES = 8

# 更宽能力组已包含较窄组的执行能力时，不重复向下游暴露同一批工具。
TOOLSET_DOMINANCE = {
    "FILE_EDITING": frozenset({"FILE_INSPECTION"}),
    "SOFTWARE_DEVELOPMENT": frozenset({"FILE_INSPECTION", "FILE_EDITING"}),
    "APPWORLD": frozenset({"WEB_RESEARCH", "BROWSER_AUTOMATION"}),
}


class ModelToolsetSelection(BaseModel):
    """专用轻量模型对混合召回候选进行组合选择。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    reason: str = Field(
        default="",
        max_length=500,
        description="可选选择依据；格式错误时Harness会省略，不影响selected_toolsets。",
    )
    selected_toolsets: list[str] = Field(
        min_length=1,
        max_length=3,
        description="从本次提供的真实可用能力组名称中选择一到三个。",
    )

    @field_validator("selected_toolsets", mode="before")
    @classmethod
    def normalize_selected_toolsets(cls, value):
        return coerce_text_sequence(value)

    @model_validator(mode="before")
    @classmethod
    def omit_invalid_auxiliary_fields(cls, value):
        return drop_invalid_auxiliary_fields(cls, value, ("reason",))


class WebModelToolsetSelection(ModelToolsetSelection):
    """Web专用选择结果：同时判断是否需要周期性强制进度检查。"""

    require_progress_checkpoint: bool = Field(
        default=False,
        description=(
            "仅判断当前Web Step是否属于多阶段依赖、学术/深度研究或高阻塞风险任务，"
            "因而需要Harness在达到工具调用阈值后强制要求一次进度检查。"
            "普通事实查询、可并发完成的多关键词搜索均填false；缺失或非法值也按false。"
        ),
    )

    @field_validator("require_progress_checkpoint", mode="before")
    @classmethod
    def normalize_progress_checkpoint(cls, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in {0, 1}:
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "y", "1", "on"}:
                return True
            if normalized in {"false", "no", "n", "0", "off", ""}:
                return False
        return False


class CapabilityExpansionSelection(BaseModel):
    """专用轻量模型从完整剩余目录补选能力组。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    reason: str = Field(
        default="",
        max_length=800,
        description="可选能力层选择依据；格式错误时Harness会省略。",
    )
    selected_toolsets: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="只从remaining_toolsets选择零到三个；空列表表示没有合适的新增能力。",
    )

    @field_validator("selected_toolsets", mode="before")
    @classmethod
    def normalize_selected_toolsets(cls, value):
        return coerce_text_sequence(value)

    @model_validator(mode="before")
    @classmethod
    def omit_invalid_auxiliary_fields(cls, value):
        return drop_invalid_auxiliary_fields(cls, value, ("reason",))


@dataclass(frozen=True)
class HybridToolsetCandidates:
    """BM25与Cross Encoder经RRF合并后的有限候选。"""

    names: tuple[str, ...]
    bm25_names: tuple[str, ...]
    cross_encoder_names: tuple[str, ...]
    rrf_scores: tuple[tuple[str, float], ...]


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

    # 只有Web初始能力选择器会设置。其他角色和所有降级路径保持False。
    require_progress_checkpoint: bool = False

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

            # 这条旧的单路Cross Encoder路由仅保留为离线降级；
            # 生产路由使用BM25 + Cross Encoder + RRF召回，
            # 并由独立轻量选择模型做最终组合判断。
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

    async def hybrid_candidates(
        self,
        *,
        primary_task: str,
        user_request_window: str,
        available_tools: Sequence[Any],
    ) -> HybridToolsetCandidates | None:
        """BM25取前三、Cross Encoder取前五，再以RRF合并为最多八项。"""

        metadata = self.registry.build_router_metadata(tuple(available_tools))
        if not metadata:
            return None
        specs = [self.registry.get(str(item["name"])) for item in metadata]
        specs = [spec for spec in specs if spec is not None]
        if not specs:
            return None
        names = [spec.name for spec in specs]
        documents = [
            _build_routing_document(
                name=spec.name,
                description=spec.description,
                profile=spec.routing_profile,
            )
            for spec in specs
        ]
        query = _compact_routing_task(
            "\n".join(
                part for part in (
                    str(primary_task or "").strip(),
                    str(user_request_window or "").strip(),
                ) if part
            )
        )
        if not query:
            return None

        bm25_names: list[str] = []
        try:
            from langchain_core.documents import Document
            from knowledge_rag.service import MatchingBM25Retriever, terms

            lexical = MatchingBM25Retriever.from_documents(
                [
                    Document(page_content=document, metadata={"name": name})
                    for name, document in zip(names, documents)
                ],
                preprocess_func=terms,
                k=min(TOOLSET_BM25_CANDIDATES, len(documents)),
            )
            lexical_hits = await lexical.ainvoke(query)
            bm25_names = [
                str(document.metadata.get("name") or "")
                for document in lexical_hits
                if str(document.metadata.get("name") or "") in names
            ][:TOOLSET_BM25_CANDIDATES]
        except Exception:
            logger.exception("BM25工具组候选召回失败；继续使用Cross Encoder候选")

        cross_encoder_names: list[str] = []
        try:
            fit_query = getattr(self.retrieval_models, "fit_rerank_query", None)
            scoring_query = (
                fit_query(query, documents) if callable(fit_query) else query
            )
            ranked = await self.retrieval_models.arerank(
                query=scoring_query,
                documents=documents,
                top_k=min(TOOLSET_CROSS_ENCODER_CANDIDATES, len(documents)),
            )
            cross_encoder_names = [
                names[result.index]
                for result in ranked
                if 0 <= result.index < len(names)
            ][:TOOLSET_CROSS_ENCODER_CANDIDATES]
        except Exception:
            logger.exception("Cross Encoder工具组候选召回失败；继续使用BM25候选")

        if not bm25_names and not cross_encoder_names:
            return None

        # RRF只融合两个排序，不把不可比较的BM25与CE原始分数直接相加。
        fused: dict[str, float] = {}
        for ranking in (bm25_names, cross_encoder_names):
            for rank, name in enumerate(ranking, start=1):
                fused[name] = fused.get(name, 0.0) + 1.0 / (60.0 + rank)
        ordered = sorted(
            fused,
            key=lambda name: (-fused[name], names.index(name)),
        )[:TOOLSET_HYBRID_CANDIDATES]
        return HybridToolsetCandidates(
            names=tuple(ordered),
            bm25_names=tuple(bm25_names),
            cross_encoder_names=tuple(cross_encoder_names),
            rrf_scores=tuple((name, fused[name]) for name in ordered),
        )

    async def route_hybrid_with_model(
        self,
        model: Any,
        *,
        primary_task: str,
        user_request_window: str,
        available_tools: Sequence[Any],
        select_progress_checkpoint: bool = False,
    ) -> ToolsetRouteDecision | None:
        """混合召回后只让专用轻量模型选择，不再调用Worker模型路由。"""

        candidates = await self.hybrid_candidates(
            primary_task=primary_task,
            user_request_window=user_request_window,
            available_tools=available_tools,
        )
        if candidates is None or model is None or not hasattr(model, "with_structured_output"):
            return None
        metadata = {
            str(item["name"]): item
            for item in self.registry.build_router_metadata(tuple(available_tools))
        }
        catalog = [metadata[name] for name in candidates.names if name in metadata]
        selection_schema = (
            WebModelToolsetSelection
            if select_progress_checkpoint
            else ModelToolsetSelection
        )
        selector_prompt = load_prompt("routing/capability_selector")
        if select_progress_checkpoint:
            selector_prompt += "\n\n" + load_prompt(
                "routing/web_progress_checkpoint_selector"
            )
        from structured_output_policy import textual_schema_fallback
        schema_guidance = textual_schema_fallback(
            model,
            selection_schema,
            heading="只输出符合下列Schema的 JSON object，不要输出其他文字：",
        )
        messages = [
            {
                "role": "system",
                "content": selector_prompt
                + (("\n" + schema_guidance) if schema_guidance else ""),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "mode": "INITIAL_SELECTION",
                        "current_step": _compact_routing_task(primary_task),
                        "user_request": str(user_request_window or "").strip(),
                        "candidate_toolsets": catalog,
                        "retrieval": {
                            "bm25_top3": candidates.bm25_names,
                            "cross_encoder_top5": candidates.cross_encoder_names,
                            "rrf_union": candidates.names,
                        },
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]
        try:
            from structured_output_policy import with_provider_structured_output
            structured = with_provider_structured_output(
                model,
                selection_schema,
                include_raw=True,
                fallback_method="json_mode",
            )
            response = await structured.ainvoke(messages)
            if isinstance(response, selection_schema):
                parsed = response
            elif isinstance(response, Mapping):
                if response.get("parsing_error") is not None:
                    return None
                parsed = selection_schema.model_validate(response.get("parsed"))
            else:
                parsed = selection_schema.model_validate(response)
        except Exception:
            logger.exception("专用能力选择模型调用失败")
            return None
        selected = [name for name in parsed.selected_toolsets if name in candidates.names]
        if not selected:
            return None
        decision = self.resolve_selected_toolsets(
            selected,
            available_tools,
            routing_task=primary_task,
            allow_no_tool=False,
        )
        if decision is None:
            return None
        return replace(
            decision,
            require_progress_checkpoint=(
                bool(getattr(parsed, "require_progress_checkpoint", False))
                if select_progress_checkpoint
                else False
            ),
        )

    async def expand_with_model(
        self,
        model: Any,
        *,
        primary_task: str,
        user_request_window: str,
        available_tools: Sequence[Any],
        mounted_toolsets: Sequence[str],
        capability_request: Mapping[str, Any],
    ) -> ToolsetRouteDecision | None:
        """从完整目录补选最多三组；不经过BM25/CE候选截断。"""

        if model is None or not hasattr(model, "with_structured_output"):
            return None
        metadata = self.registry.build_router_metadata(tuple(available_tools))
        available_names = {str(item["name"]) for item in metadata}
        mounted = {str(name).strip().upper() for name in mounted_toolsets}
        mounted_specs = [self.registry.get(name) for name in mounted]
        if any(spec is not None and spec.exclusive for spec in mounted_specs):
            return None
        remaining = []
        for item in metadata:
            name = str(item["name"])
            spec = self.registry.get(name)
            if (
                name not in mounted
                and spec is not None
                and spec.allow_runtime_expansion
            ):
                remaining.append(item)
        if not remaining:
            return None
        # 完整目录置于静态前缀，有利于选择器自己的上下文缓存。
        # 选择器可以看见全部已定义能力以正确判断缺口，
        # 但remaining_toolsets仍只包含当前Worker真实可挂载的组。
        full_catalog = []
        for name in self.registry.toolset_names:
            spec = self.registry.get(name)
            if spec is None or not spec.allow_runtime_expansion:
                continue
            full_catalog.append({
                "name": spec.name,
                "description": spec.description,
                "routing_profile": spec.routing_profile,
                "exclusive": bool(spec.exclusive),
                "available_to_current_worker": spec.name in available_names,
            })
        from structured_output_policy import textual_schema_fallback
        schema_guidance = textual_schema_fallback(
            model,
            CapabilityExpansionSelection,
            heading="只输出符合下列Schema的 JSON object，不要输出其他文字：",
        )
        messages = [
            {
                "role": "system",
                "content": load_prompt("routing/capability_expansion_selector")
                + "\n完整能力目录（固定顺序）："
                + json.dumps(full_catalog, ensure_ascii=False, separators=(",", ":"))
                + (("\n" + schema_guidance) if schema_guidance else ""),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "mode": "ONE_SHOT_EXPANSION",
                        "current_step": _compact_routing_task(primary_task),
                        "user_request": str(user_request_window or "").strip(),
                        "mounted_toolsets": sorted(mounted),
                        "remaining_toolsets": [str(item["name"]) for item in remaining],
                        "capability_gap": dict(capability_request),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                ),
            },
        ]
        try:
            from structured_output_policy import with_provider_structured_output
            structured = with_provider_structured_output(
                model,
                CapabilityExpansionSelection,
                include_raw=True,
                fallback_method="json_mode",
            )
            response = await structured.ainvoke(messages)
            if isinstance(response, CapabilityExpansionSelection):
                parsed = response
            elif isinstance(response, Mapping):
                if response.get("parsing_error") is not None:
                    return None
                parsed = CapabilityExpansionSelection.model_validate(response.get("parsed"))
            else:
                parsed = CapabilityExpansionSelection.model_validate(response)
        except Exception:
            logger.exception("专用能力补选模型调用失败")
            return None
        remaining_names = {str(item["name"]) for item in remaining}
        requested = []
        for name in parsed.selected_toolsets:
            normalized = str(name or "").strip().upper()
            if normalized in remaining_names and normalized not in requested:
                requested.append(normalized)
        if not requested:
            return None
        # 独占组不能追加到已经挂载业务组的Worker。
        requested = [
            name for name in requested
            if not bool(self.registry.get(name).exclusive)
        ][:3]
        if not requested:
            return None
        return self.resolve_selected_toolsets(
            requested,
            available_tools,
            routing_task=primary_task,
            allow_no_tool=False,
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
