from __future__ import annotations

from typing import (
    Any,
    Literal,
)
from pathlib import PurePosixPath, PureWindowsPath

from pydantic.json_schema import SkipJsonSchema
from api_handoff import ApiHandoffList, ApiHandoffReceipt, api_field
from handoff_knowledge import HandoffKnowledge, knowledge_field
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


SupervisorAction = Literal[
    "FINAL",
    "PLAN",
]


ReplanAction = Literal[
    "CONTINUE",
    "RETURN_TO_WORKER",
    "FINISH",
]


FinalReviewAction = Literal[
    "FINAL",
    "RETURN_TO_WORKER",
    "REPLAN",
]


StepStatus = Literal[
    "COMPLETED",
    "PARTIAL",
    "BLOCKED",
    "FAILED",
]


StepExecutionMode = Literal[
    "SINGLE",
    "PARALLEL",
]


WorkerKind = Literal[
    "GENERAL",
    "WEB",
    "CODE",
]

ToolAccessMode = Literal[
    "ENABLED",
    "DISABLED",
]


CodeRequirementPriority = Literal[
    "MUST",
    "SHOULD",
]


CodeDeliveryMode = Literal[
    "PATCH",
    "ARTIFACT",
    "EVALUATION",
]


CodeInterfaceKind = Literal[
    "PYTHON_SYMBOL",
    "HTTP_API",
    "CLI",
    "WEB_PAGE",
    "FILE_ARTIFACT",
    "OTHER",
]


StepJoinPolicy = Literal[
    "ALL_TERMINAL",
]


ArtifactDisposition = Literal[
    "INTERNAL_HANDOFF",
    "USER_DELIVERABLE",
]


CriterionStatus = Literal[
    "MET",
    "PARTIAL",
    "NOT_MET",
    "UNKNOWN",
]


FinalReviewStatus = Literal[
    "COMPLETED",
    "PARTIAL",
    "FAILED",
]


class PlanningModel(
    BaseModel
):
    """规划模块结构化输出的公共基类。"""

    model_config = ConfigDict(
        extra="forbid",

        str_strip_whitespace=True,
    )


_NULLISH_STRINGS = {"", "null", "none", "nil", "false", "n/a"}


def _normalize_optional_value(value: Any) -> Any:
    """Accept common model sentinels for optional schema fields."""
    if value is None or value is False:
        return None
    if isinstance(value, str) and value.strip().lower() in _NULLISH_STRINGS:
        return None
    return value


def _normalize_optional_list(value: Any) -> Any:
    normalized = _normalize_optional_value(value)
    return [] if normalized is None else normalized


class DialogueMessage(
    PlanningModel
):
    """提供给Hard节点的用户对话消息。

    原Conversation只保存和提供：
    - 用户消息；
    - 最终助手回答。

    不包含Step内部工具轨迹。
    """

    role: Literal[
        "user",
        "assistant",
    ]

    content: str = Field(
        min_length=1,
    )


class PlanningReplacementContext(PlanningModel):
    """Bounded, validated carry-over from a superseded execution generation."""

    target_event_id: str = Field(min_length=1)
    replacement_event_id: str = Field(min_length=1)
    original_user_request: str = Field(min_length=1)
    replacement_instruction: str = Field(min_length=1)
    previous_plan_objective: str = ""
    completed_step_reports: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=12,
    )
    handoff_publication_receipts: list[dict[str, Any]] = Field(
        default_factory=list,
        max_length=24,
    )
    accepted_integration_status: dict[str, Any] | None = None
    previous_usage: dict[str, int] = Field(default_factory=dict)
    supersession_receipt: dict[str, Any] = Field(default_factory=dict)
    workspace_inheritance: dict[str, Any] = Field(default_factory=dict)


class PlanningContextPack(
    PlanningModel
):
    """一次Planning运行共用的基础上下文。

    Supervisor、Replanner和Final Reviewer
    共用同一个Context Pack。

    Simple Executor和Step Reporter不会直接读取
    conversation_summary与recent_dialogue。
    """

    current_time: str = Field(
        min_length=1,
    )

    user_request: str = Field(
        min_length=1,
    )

    # Harness-owned scope routing result. It is frozen for one planning run,
    # so checkpoint resume never repeats the paid resolver.
    scope_router: dict[str, Any] | None = None
    scope_contract: "ScopeContract | None" = None

    # 较早Conversation的滚动摘要。
    conversation_summary: str = ""

    # 兼容旧Checkpoint的历史指令字段。新Run不再把无界用户原文账本
    # 注入Scheduler；连续性由上一轮Pair和高相关历史Pair提供。
    user_instruction_history: list[str] = Field(default_factory=list)

    # Persisted Scheduler message stream; never an LLM-generated field.
    scheduler_session: dict[str, Any] = Field(default_factory=dict)

    # 最近若干轮用户与最终助手原文。
    recent_dialogue: list[
        DialogueMessage
    ] = Field(
        default_factory=list,
    )

    # 根据当前用户请求召回的长期记忆。
    memory_context: str = ""
    rag_context: str = ""

    # Application-supplied environment contract and optional skill for this run.
    execution_instructions: str = ""
    # Harness-owned, frozen once per planning run. These are never model output.
    skill_catalog: list[dict[str, Any]] | None = None
    role_skill_snapshots: dict[str, dict[str, Any]] = Field(default_factory=dict)
    skill_mode: Literal["off", "fixed", "dynamic"] | None = None
    skill_fixed_ids: dict[str, list[str]] = Field(default_factory=dict)

    # Present only when a new Event replaces an older execution generation.
    replacement_context: PlanningReplacementContext | None = None

    # 当前实际可用的业务能力目录。
    toolset_catalog: list[
        dict[
            str,
            str,
        ]
    ] = Field(
        default_factory=list,
    )


class WorkerAssignment(
    PlanningModel
):
    """One independent branch of a parallel Step.

    ``assignment_key`` identifies the logical slot. A later runtime REPLACE
    keeps this key and creates a new attempt generation instead of rewriting
    the original attempt.
    """

    assignment_key: str = Field(
        min_length=1,
        max_length=32,
        pattern=r"^[a-z][a-z0-9_-]*$",
    )

    objective: str = Field(
        min_length=1,
    )


class StepArtifactOutput(PlanningModel):
    """Scheduler-owned intent for a non-CODE Step artifact.

    Runtime artifact IDs do not exist when the plan is created. The Scheduler
    therefore names a semantic output slot; a Worker later binds one candidate
    to that slot and the Reporter decides whether to approve the binding.
    """

    output_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_-]*$",
    )
    description: str = Field(min_length=1, max_length=600)
    disposition: ArtifactDisposition = Field(default="INTERNAL_HANDOFF", description="INTERNAL_HANDOFF is runtime-managed internal evidence: target_path must be null. USER_DELIVERABLE is a user file: target_path must be a safe relative path.")
    target_path: str | None = Field(default=None, description="Must be null for INTERNAL_HANDOFF. Required for USER_DELIVERABLE: non-empty relative file path, no absolute path or '..' components.")
    required: bool = True

    @model_validator(mode="after")
    def validate_delivery_target(self) -> "StepArtifactOutput":
        raw_target = str(self.target_path or "").strip().replace("\\", "/")
        if self.disposition == "INTERNAL_HANDOFF":
            if raw_target:
                raise ValueError(
                    "INTERNAL_HANDOFF artifact不得提供target_path。"
                )
            self.target_path = None
            return self

        posix = PurePosixPath(raw_target)
        windows = PureWindowsPath(raw_target)
        if (
            not raw_target
            or raw_target == "."
            or posix.is_absolute()
            or windows.is_absolute()
            or ".." in posix.parts
            or any(":" in part for part in posix.parts)
            or any(part in {".agent", ".git"} for part in posix.parts)
        ):
            raise ValueError(
                "USER_DELIVERABLE artifact必须提供安全的相对target_path。"
            )
        self.target_path = posix.as_posix()
        return self


class CodeRequirement(
    PlanningModel
):
    """One stable, reviewable requirement for a CODE Step."""

    requirement_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_-]*$",
    )

    priority: CodeRequirementPriority = "MUST"

    statement: str = Field(
        min_length=1,
    )


class CodeInterfaceRequirement(
    PlanningModel
):
    """Scheduler-visible interface expectation without a rigid DSL.

    ``details`` deliberately remains an open JSON object. The Scheduler can
    describe an HTTP route, Python callable, CLI command, page behavior, or
    produced file without the framework pretending every project shares one
    universal interface schema.
    """

    interface_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_-]*$",
    )

    kind: CodeInterfaceKind

    description: str = Field(
        min_length=1,
    )

    details: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )


class CodeTaskContract(
    PlanningModel
):
    """The frozen Scheduler contract handed to Worker and Reviewer.

    MUST/SHOULD separate acceptance gates from preferences. The remaining
    fields give the Scheduler room to be specific while keeping the runtime
    contract small enough to validate and persist.
    """

    delivery_mode: CodeDeliveryMode = "PATCH"

    requirements: list[
        CodeRequirement
    ] = Field(
        min_length=1,
        max_length=20,
    )

    interfaces: list[
        CodeInterfaceRequirement
    ] = Field(
        default_factory=list,
        max_length=12,
    )

    validation_expectations: list[
        str
    ] = Field(
        default_factory=list,
        max_length=20,
    )

    implementation_guidance: list[
        str
    ] = Field(
        default_factory=list,
        max_length=20,
    )

    non_goals: list[
        str
    ] = Field(
        default_factory=list,
        max_length=12,
    )

    @model_validator(
        mode="after",
    )
    def validate_unique_ids(
        self,
    ) -> "CodeTaskContract":
        requirement_ids = [
            item.requirement_id
            for item in self.requirements
        ]
        interface_ids = [
            item.interface_id
            for item in self.interfaces
        ]

        if len(set(requirement_ids)) != len(requirement_ids):
            raise ValueError(
                "CODE requirements中的requirement_id必须唯一。"
            )

        if len(set(interface_ids)) != len(interface_ids):
            raise ValueError(
                "CODE interfaces中的interface_id必须唯一。"
            )

        return self


SetOperation = Literal[
    "DIRECT",
    "INTERSECTION",
    "UNION",
    "DIFFERENCE",
]

EffectMode = Literal["READ_ONLY", "MUTATION"]


class ScopeConstraint(PlanningModel):
    """One exact qualifier copied from the user's request."""

    source_text: str = Field(min_length=1, max_length=300, description="用户原话中的限定词或关系短语。")
    applies_to: str = Field(min_length=1, max_length=80, description="该限定条件直接约束的对象类型，不能按词语距离猜。")
    meaning: str = Field(min_length=1, max_length=300, description="该条件如何改变最终对象范围。")


class ScopeSetSpec(PlanningModel):
    """One logical object set, independent of any concrete API."""

    set_id: str = Field(
        min_length=1,
        max_length=24,
        pattern=r"^[A-Z][A-Z0-9_]*$",
    )
    definition: str = Field(min_length=1, max_length=500, description="一个可独立取得的对象集合。不要把两个需要分别读取再求交的集合藏进一句definition。")
    result_entity: str = Field(min_length=1, max_length=80, description="该集合展开后产出的实体类型，必须与target_entity一致。")


class RequiredContextSpec(PlanningModel):
    """Small cross-entity fact required before resolving the target set."""

    read: str = Field(min_length=1, max_length=300, description="必须先读取的对象或信息，不写API名。")
    source_system: str | None = Field(
        default=None,
        max_length=80,
        description=(
            "该前置信息由哪个业务系统或权威数据源定义，例如phone contact book；"
            "写系统语义名，不写API名。无法由用户语义确定时填null。"
        ),
    )
    relationship: str | None = Field(
        default=None,
        max_length=80,
        description="需要读取的关系标签，例如friend、roommate；非关系型前置信息填null。",
    )
    because: str = Field(min_length=1, max_length=300, description="为什么缺少这项信息就不能正确确定最终结果。")
    used_for: str = Field(min_length=1, max_length=300, description="该信息将用于筛选、关联或生成哪一部分最终结果。")

    @field_validator("source_system", "relationship", mode="before")
    @classmethod
    def normalize_optional_source_fields(cls, value: Any) -> Any:
        return _normalize_optional_value(value)


class ResolvedTimeRangeSpec(PlanningModel):
    """A user time phrase normalized to explicit inclusive boundaries."""

    source_text: str = Field(
        min_length=1,
        max_length=300,
        description="用户原话中的时间短语，例如今年3月、本财年或过去7天。",
    )
    start_at: str | None = Field(
        default=None,
        max_length=40,
        description=(
            "根据任务开始时间换算出的含首端绝对ISO-8601日期或时间；"
            "无法仅凭现有上下文确定时填null，并在required_context说明缺项。"
        ),
    )
    end_at: str | None = Field(
        default=None,
        max_length=40,
        description=(
            "根据任务开始时间换算出的含尾端绝对ISO-8601日期或时间；"
            "无法仅凭现有上下文确定时填null，并在required_context说明缺项。"
        ),
    )

    @field_validator("start_at", "end_at", mode="before")
    @classmethod
    def normalize_null_time_boundary(cls, value: Any) -> Any:
        return _normalize_optional_value(value)


class ScopeContract(PlanningModel):
    """Small provider-independent semantic contract for Scheduler input."""

    target_entity: str = Field(min_length=1, max_length=80, description="最终被操作的实体类型。")
    effect_mode: EffectMode = Field(
        default="MUTATION",
        description="READ_ONLY只读取、筛选、计算或回答；MUTATION会改变外部应用中的业务状态。生成回答本身不算写入。",
    )
    constraints: list[ScopeConstraint] = Field(default_factory=list, max_length=8, description="用户原文中的限定条件及其归属。")
    sets: list[ScopeSetSpec] = Field(min_length=1, max_length=4, description="用于计算最终对象范围的集合。")
    operation: SetOperation = Field(description="单集合DIRECT；同时满足INTERSECTION；任一满足UNION；排除DIFFERENCE。")
    operands: list[str] = Field(min_length=1, max_length=4, description="参与运算的全部set_id，每个恰好一次。")
    join_key: str | None = Field(default=None, max_length=80, description="多集合运算必须填写稳定实体ID字段，如bookmark_id；DIRECT必须为null。")
    ambiguity: bool = Field(default=False, description="两种合理解释会改变最终对象集合时为true。")
    alternatives: list[str] = Field(default_factory=list, max_length=3, description="ambiguity=true时列出可能范围，否则为空。")
    required_context: list[RequiredContextSpec] = Field(
        default_factory=list,
        max_length=4,
        description="确定最终对象前必须读取的跨实体信息。没有前置信息依赖时填[]；不写API、执行步骤或登录方式。",
    )
    resolved_time_ranges: list[ResolvedTimeRangeSpec] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "用户范围中的每个时间短语及其绝对含首尾边界。任务开始时间足以换算时必须填写绝对值；"
            "没有时间条件时填[]。"
        ),
    )

    @field_validator("join_key", mode="before")
    @classmethod
    def normalize_null_join_key(cls, value: Any) -> Any:
        return _normalize_optional_value(value)

    @field_validator("alternatives", "required_context", "resolved_time_ranges", mode="before")
    @classmethod
    def normalize_optional_collections(cls, value: Any) -> Any:
        return _normalize_optional_list(value)

    @model_validator(mode="after")
    def validate_scope_expression(self) -> "ScopeContract":
        set_ids = [item.set_id for item in self.sets]
        if len(set_ids) != len(set(set_ids)):
            raise ValueError("ScopeContract set_id必须唯一。")
        if set(set_ids) != set(self.operands) or len(self.operands) != len(set(self.operands)):
            raise ValueError("ScopeContract每个集合必须且只能在operands中出现一次。")
        if {item.result_entity for item in self.sets} != {self.target_entity}:
            raise ValueError("ScopeContract每个集合必须产出target_entity。")
        if self.operation == "DIRECT":
            if len(self.operands) != 1:
                raise ValueError("DIRECT必须且只能有一个集合。")
            self.join_key = None
        elif len(self.operands) < 2 or not self.join_key:
            raise ValueError("多集合运算至少需要两个集合和稳定join_key。")
        if self.ambiguity and not self.alternatives:
            raise ValueError("ambiguity=true时必须给出alternatives。")
        if not self.ambiguity:
            self.alternatives = []
        return self


class TargetSetSpec(PlanningModel):
    """One user-defined object set used to calculate a Step's write scope."""

    set_id: str = Field(
        min_length=1,
        max_length=24,
        pattern=r"^[A-Z][A-Z0-9_]*$",
        description="集合短编号，例如A、B、EXCLUDED；只用于本target_selection。",
    )
    definition: str = Field(
        min_length=1,
        max_length=500,
        description="忠实保留用户对这个集合的范围描述，不写API名。",
    )
    condition_owner: str = Field(
        min_length=1,
        max_length=80,
        description="该条件直接限制的对象类型，例如folder、document、playlist或song。",
    )
    result_entity: str = Field(
        min_length=1,
        max_length=80,
        description="读取并展开该条件后参与集合运算的对象类型；参与同一运算的集合必须一致。",
    )


class TargetSelection(PlanningModel):
    """Declarative set algebra for the exact objects a Step may read or modify."""

    target_entity: str = Field(
        min_length=1,
        max_length=80,
        description="最终允许读取，以及在MUTATION时允许修改和验收的对象类型。",
    )
    effect_mode: EffectMode = Field(
        default="MUTATION",
        description="复制已采用ScopeContract的外部效果：READ_ONLY不进入写入链；MUTATION必须写入并回读验收。",
    )
    sets: list[TargetSetSpec] = Field(
        min_length=1,
        max_length=4,
        description="从用户原请求拆出的对象集合；不得增加用户未给出的条件。",
    )
    operation: SetOperation = Field(
        description="DIRECT为单集合；INTERSECTION对应同时/其中；UNION对应或；DIFFERENCE按operands顺序做差集。",
    )
    operands: list[str] = Field(
        min_length=1,
        max_length=4,
        description="参与运算的set_id；DIFFERENCE中顺序有意义。",
    )
    join_key: str | None = Field(
        default=None,
        max_length=80,
        description="多集合运算使用的稳定对象ID，例如song_id；DIRECT可为null。",
    )
    write_scope: Literal["NONE", "RESULT"] = Field(
        default="RESULT",
        description="Harness按effect_mode规范化：READ_ONLY为NONE；MUTATION为RESULT且写入只能作用于集合运算结果。",
    )
    verify_scope: Literal["READ_RESULT", "RESULT"] = Field(
        default="RESULT",
        description="Harness按effect_mode规范化：READ_ONLY为READ_RESULT；MUTATION为RESULT并要求写后回读同一集合。",
    )
    required_context: list[RequiredContextSpec] = Field(
        default_factory=list,
        max_length=4,
        description="执行者在计算目标集合前必须先取得的跨实体信息；从ScopeContract原样保留，不猜API。",
    )
    resolved_time_ranges: list[ResolvedTimeRangeSpec] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "从ScopeContract原样复制的绝对时间边界；Scheduler的Step目标、执行说明和验收条件"
            "必须使用这些绝对值，不得退化回只看月份或其他不完整条件。"
        ),
    )

    @field_validator("join_key", mode="before")
    @classmethod
    def normalize_null_join_key(cls, value: Any) -> Any:
        return _normalize_optional_value(value)

    @field_validator("required_context", "resolved_time_ranges", mode="before")
    @classmethod
    def normalize_optional_collections(cls, value: Any) -> Any:
        return _normalize_optional_list(value)

    @model_validator(mode="after")
    def validate_set_expression(self) -> "TargetSelection":
        if self.effect_mode == "READ_ONLY":
            self.write_scope = "NONE"
            self.verify_scope = "READ_RESULT"
        else:
            self.write_scope = "RESULT"
            self.verify_scope = "RESULT"
        set_ids = [item.set_id for item in self.sets]
        if len(set_ids) != len(set(set_ids)):
            raise ValueError("target_selection中的set_id必须唯一。")
        if len(self.operands) != len(set(self.operands)):
            raise ValueError("target_selection.operands不得重复。")
        if set(set_ids) != set(self.operands):
            raise ValueError("每个集合必须且只能在operands中引用一次。")

        entities = {item.result_entity for item in self.sets}
        if entities != {self.target_entity}:
            raise ValueError("参与运算的每个集合都必须产出target_entity。")

        if self.operation == "DIRECT":
            if len(self.operands) != 1:
                raise ValueError("DIRECT必须且只能引用一个集合。")
            self.join_key = None
        else:
            if len(self.operands) < 2:
                raise ValueError("交集、并集或差集至少需要两个集合。")
            if not self.join_key:
                raise ValueError("多集合运算必须提供稳定join_key。")
        return self


class PlanStep(
    PlanningModel
):
    """Supervisor或Replanner生成的一个高层Step。

    Step数量和最大编号不再由这个Schema写死。

    后续统一由PlanningSettings和Planning Graph
    根据环境变量执行限制与重新编号。
    """

    step_id: int = Field(
        ge=1,
    )

    objective: str = Field(
        min_length=1,
        description="当前步骤目标，保留与该步骤有关的原请求范围、筛选条件及例外，不扩大对象集合。",
    )

    success_criteria: list[
        str
    ] = Field(
        min_length=1,
        description="可核验条件，包含该步骤的对象范围及用户要求保留不变的内容。",
    )

    tool_access: ToolAccessMode = Field(
        default="ENABLED",
        description=(
            "当前Step是否开放工具访问。需要读取消息之外的状态、查询外部数据、"
            "处理文件、执行代码、写入或回读时填ENABLED；只有完全依据已给上下文"
            "即可直接作答时填DISABLED。这里只判断是否开放，不填写或猜测工具名；"
            "省略、空值或无法识别时Harness按ENABLED处理。"
        ),
    )

    @field_validator("tool_access", mode="before")
    @classmethod
    def normalize_tool_access(cls, value: Any) -> str:
        """兼容模型常见布尔写法，并对未知值采取开放工具的保守回退。"""
        if isinstance(value, bool):
            return "ENABLED" if value else "DISABLED"
        normalized = str(value or "").strip().upper()
        if normalized in {"DISABLED", "FALSE", "NO", "N", "0", "OFF"}:
            return "DISABLED"
        if normalized in {"ENABLED", "TRUE", "YES", "Y", "1", "ON"}:
            return "ENABLED"
        return "ENABLED"

    # Hard模型提供给Simple Executor的
    # 少量高价值执行建议。
    rag_query: str | None = Field(default=None, max_length=300,
        description="当前Step的独立文档检索query：写明待查信息、对象、完整范围，以及在什么场景需要哪类读取、写入或回读能力；不混入预算、报告格式或无关收尾，不猜接口名。无需检索填null；Harness仅用本字段召回和重排，不从其他字段补写。")

    execution_guidance: (
        str
        | None
    ) = None

    target_selection: TargetSelection | None = Field(
        default=None,
        description="复杂范围任务的结构化对象集合合同。出现同时、其中、或、排除、only/all/except等组合条件时填写；简单单集合任务可为null。不得写未经确认的API名。",
    )

    # The whole Step uses one Worker capability class. SINGLE may select any
    # available kind; V1 PARALLEL is deliberately limited to WEB.
    worker_kind: WorkerKind = Field(default="GENERAL", description="CODE requires code_task and artifact_outputs=[]; WEB/GENERAL require code_task=null and may declare artifact_outputs.")
    skill_topics: list[str] = Field(
        default_factory=list, max_length=3,
        description="Optional method topics: documents, research, implementation, verification, appworld. These do not grant tools or change acceptance criteria.",
    )

    # CODE Step的Scheduler契约。它会原样交给Code Worker和Reviewer，
    # 避免Reviewer根据Worker自报内容重新发明验收标准。
    code_task: (
        CodeTaskContract
        | None
    ) = Field(default=None, description="Required only for CODE: shared Worker/Reviewer acceptance and delivery contract. Must be null for WEB/GENERAL. CODE delivery belongs here, never in artifact_outputs.")

    # Scheduler declares whether a Web/General artifact is only an internal
    # handoff or a user-visible deliverable. Runtime candidate IDs are bound
    # later by the Worker and independently approved by the Step Reporter.
    artifact_outputs: list[
        StepArtifactOutput
    ] = Field(
        default_factory=list,
        max_length=12,
        description="Only WEB/GENERAL output slots. CODE must use an empty list and declare delivery in code_task. INTERNAL_HANDOFF slots must have target_path=null; USER_DELIVERABLE slots require a safe relative target_path.",
    )

    # SINGLE directly executes this Step's objective. PARALLEL requires
    # explicit, complementary Web assignments so the harness never clones the
    # same prompt into several Workers by accident.
    execution_mode: StepExecutionMode = "SINGLE"

    worker_assignments: list[
        WorkerAssignment
    ] = Field(
        default_factory=list,
        max_length=3,
    )

    # V1 intentionally supports one deterministic barrier: every Worker slot
    # must reach a terminal runtime state before the Reporter is claimed.
    join_policy: StepJoinPolicy = "ALL_TERMINAL"

    @field_validator(
        "rag_query", "execution_guidance", "target_selection", "code_task",
        mode="before",
    )
    @classmethod
    def normalize_optional_fields(cls, value: Any) -> Any:
        return _normalize_optional_value(value)

    @model_validator(
        mode="after",
    )
    def validate_execution_shape(
        self,
    ) -> "PlanStep":
        """Keep single execution cheap and parallel execution intentional."""

        if self.worker_kind == "CODE":
            if self.code_task is None:
                raise ValueError(
                    "CODE Step必须提供code_task验收契约。"
                )
        elif self.code_task is not None:
            raise ValueError(
                "只有CODE Step可以提供code_task。"
            )

        if self.worker_kind == "CODE" and self.artifact_outputs:
            raise ValueError(
                "CODE Step使用code_task交付契约，不得再提供artifact_outputs。"
            )

        output_ids = [item.output_id for item in self.artifact_outputs]
        if len(output_ids) != len(set(output_ids)):
            raise ValueError("同一个Step内的artifact output_id必须唯一。")
        delivery_paths = [
            item.target_path
            for item in self.artifact_outputs
            if item.disposition == "USER_DELIVERABLE"
        ]
        if len(delivery_paths) != len(set(delivery_paths)):
            raise ValueError("同一个Step内的USER_DELIVERABLE target_path必须唯一。")

        if self.execution_mode == "SINGLE":
            if self.worker_assignments:
                raise ValueError(
                    "SINGLE Step不得提供worker_assignments；"
                    "它直接使用Step objective。"
                )
            return self

        if len(self.worker_assignments) < 2:
            raise ValueError(
                "PARALLEL Step必须提供2到3个worker_assignments。"
            )

        assignment_keys = [
            assignment.assignment_key
            for assignment in self.worker_assignments
        ]
        if len(set(assignment_keys)) != len(assignment_keys):
            raise ValueError(
                "同一个PARALLEL Step内的assignment_key必须唯一。"
            )

        if self.worker_kind != "WEB":
            raise ValueError(
                "V1只允许WEB Worker并行；"
                "CODE和GENERAL任务必须使用SINGLE。"
            )

        return self


class SupervisorDecision(
    PlanningModel
):
    """初始Hard Supervisor的决定。"""

    action: SupervisorAction

    # FINAL时使用。
    final_answer: (
        str
        | None
    ) = None

    # PLAN时使用。
    plan_objective: (
        str
        | None
    ) = Field(default=None, description="PLAN的完整目标：保留用户原请求的对象范围、筛选条件、条件间关系与例外，不扩大操作范围。")

    plan_success_criteria: list[
        str
    ] = Field(
        default_factory=list,
        description="逐项覆盖原请求的范围、筛选条件及例外；按原关系组合条件，不只验证动作已执行。",
    )

    steps: list[
        PlanStep
    ] = Field(
        default_factory=list,
    )

    @model_validator(
        mode="after",
    )
    def validate_action_fields(
        self,
    ) -> "SupervisorDecision":
        """检查FINAL和PLAN的字段组合。"""

        if self.action == "FINAL":
            if not self.final_answer:
                raise ValueError(
                    "FINAL必须提供final_answer。"
                )

            if (
                self.plan_objective
                or self.plan_success_criteria
                or self.steps
            ):
                raise ValueError(
                    "FINAL不能同时提供Plan字段。"
                )

            return self

        if self.final_answer:
            raise ValueError(
                "PLAN不能提供final_answer。"
            )

        if not self.plan_objective:
            raise ValueError(
                "PLAN必须提供plan_objective。"
            )

        if not self.plan_success_criteria:
            raise ValueError(
                "PLAN必须提供"
                "plan_success_criteria。"
            )

        if not self.steps:
            raise ValueError(
                "PLAN必须提供至少一个Step。"
            )

        return self


class StepCriterionResult(
    PlanningModel
):
    """一项Step成功标准的审核结果。"""

    criterion_id: str | None = Field(default=None, description="Harness-owned current Step criterion reference.")

    criterion: str = Field(
        min_length=1,
    )

    evidence: list[
        str
    ] = Field(
        default_factory=list,
        description="先列出支持或限制本项判断的短证据编号与观测，再填写status。",
    )

    status: CriterionStatus


class StepArtifactReport(
    PlanningModel
):
    """A final artifact the main planner may pass to later Steps."""

    path: str = Field(
        min_length=1,
    )

    description: str = Field(
        min_length=1,
    )


class WorkerContribution(
    PlanningModel
):
    """One Worker's concise contribution to a shared Step."""

    worker_id: str = Field(
        min_length=1,
    )

    contribution: str = Field(
        min_length=1,
    )


class StepReport(
    PlanningModel
):
    """Simple Step Reporter生成的压缩报告。"""

    handoff_apis: ApiHandoffList = api_field()
    handoff_api_receipts: SkipJsonSchema[list[ApiHandoffReceipt]] = Field(
        default_factory=list,
        description="Harness-authored per-API validation receipts, including rejected leads and reasons.",
    )
    handoff_knowledge: SkipJsonSchema[list[HandoffKnowledge]] = Field(default_factory=list, description="Worker-authored reusable knowledge preserved by the Harness; not a separate verification verdict.")


    step_id: int = Field(
        ge=1,
    )

    summary: str = Field(
        min_length=1,
        description="先概括实际完成与未完成情况，再填写整体status。",
    )

    stop_reason: str = Field(
        min_length=1,
    )

    status: StepStatus

    criterion_results: list[
        StepCriterionResult
    ] = Field(
        default_factory=list,
    )

    @field_validator("criterion_results")
    @classmethod
    def assign_legacy_criterion_ids(cls, values):
        # Canonical reports include IDs even on deterministic fallback/CODE paths.
        return [item.model_copy(update={"criterion_id": item.criterion_id or f"C{i}"})
                for i, item in enumerate(values, 1)]

    confirmed_results: list[
        str
    ] = Field(
        default_factory=list,
    )

    assessment_source: Literal["INDEPENDENT_REVIEW", "GENERAL_SELF_REPORT", "RUNTIME_FAILURE"] = "INDEPENDENT_REVIEW"

    completed_work: list[
        str
    ] = Field(
        default_factory=list,
    )

    artifacts: list[
        StepArtifactReport
    ] = Field(
        default_factory=list,
    )

    # Generic Reporter selects immutable candidate references. The Harness
    # Publisher, not the model, converts these into shared handoff paths.
    approved_artifact_refs: list[str] = Field(
        default_factory=list,
        max_length=12,
    )

    @field_validator("approved_artifact_refs")
    @classmethod
    def normalize_approved_artifact_refs(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            item = str(value).strip()
            if item and item not in normalized:
                normalized.append(item)
        return normalized

    worker_contributions: list[
        WorkerContribution
    ] = Field(
        default_factory=list,
    )

    evidence: list[
        str
    ] = Field(
        default_factory=list,
    )

    errors: list[
        str
    ] = Field(
        default_factory=list,
    )

    unresolved_items: list[
        str
    ] = Field(
        default_factory=list,
    )

    # Reporter只能建议Replan，
    # 不能自己生成新Plan。
    replan_reason: (
        str
        | None
    ) = Field(
        default=None,
        description="先说明原计划哪项前提失效及已有依据，再填写request_replan。无需重规划时填null。",
    )

    request_replan: bool = False

    next_action: (
        str
        | None
    ) = None

    @model_validator(
        mode="after",
    )
    def validate_replan_fields(
        self,
    ) -> "StepReport":
        """请求Replan时必须提供明确原因。"""

        if (
            self.request_replan
            and not self.replan_reason
        ):
            raise ValueError(
                "request_replan为True时，"
                "必须提供replan_reason。"
            )

        if (
            not self.request_replan
            and self.replan_reason
        ):
            raise ValueError(
                "request_replan为False时，"
                "不能提供replan_reason。"
            )

        return self


class ReplanDecision(
    PlanningModel
):
    """预算内Hard Replanner的决定。

    CONTINUE：
        使用remaining_steps继续执行。

    FINISH：
        停止继续执行Step，
        交给Final Reviewer根据已有报告收口。
    """

    reason: str = Field(
        min_length=1,
        description="先说明已有事实、剩余问题和选择该动作的依据，再填写action。",
    )

    action: ReplanAction

    worker_instruction: str | None = Field(
        default=None,
        description=(
            "仅RETURN_TO_WORKER填写：说明为什么驳回本次计划异议，以及原Worker"
            "继续执行时必须保留的范围边界。不猜API。"
        ),
    )

    remaining_steps: list[
        PlanStep
    ] = Field(
        default_factory=list,
    )

    @model_validator(
        mode="after",
    )
    def validate_action_fields(
        self,
    ) -> "ReplanDecision":
        """检查CONTINUE和FINISH的字段组合。"""

        if self.action == "CONTINUE":
            if not self.remaining_steps:
                raise ValueError(
                    "CONTINUE必须提供"
                    "remaining_steps。"
                )

            if self.worker_instruction is not None:
                raise ValueError("CONTINUE不能提供worker_instruction。")
            return self

        if self.action == "RETURN_TO_WORKER":
            if self.remaining_steps:
                raise ValueError("RETURN_TO_WORKER不能提供remaining_steps。")
            if not self.worker_instruction:
                raise ValueError("RETURN_TO_WORKER必须提供worker_instruction。")
            return self

        if self.remaining_steps:
            raise ValueError(
                "FINISH不能提供remaining_steps。"
            )
        if self.worker_instruction is not None:
            raise ValueError("FINISH不能提供worker_instruction。")

        return self


class FinalCriterionReview(PlanningModel):
    """Final Reviewer对一条冻结验收标准的证据判断。"""

    criterion_id: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)
    observed_result: str = Field(
        min_length=1,
        description="先写实际看到的结果；没有证据时明确写未观察到。",
    )
    missing_requirement: str | None = Field(
        default=None,
        description="仍缺少的要求；已满足时填null。不能在这里猜修复方法或API。",
    )
    status: CriterionStatus


class FinalWorkerRepairRequest(PlanningModel):
    """交回最后一个Worker的事实型返修单，不替Worker设计执行方案。"""

    step_id: int = Field(ge=1)
    worker_kind: WorkerKind
    failed_criterion_ids: list[str] = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)
    observed_problem: str = Field(
        min_length=1,
        description="说明已经观察到的缺口或矛盾，不提供API猜测或执行步骤。",
    )
    missing_requirement: str = Field(
        min_length=1,
        description="说明验收还缺什么可观察结果，不提供直接操作建议。",
    )


class FinalReviewDecision(
    PlanningModel
):
    """Hard Final Reviewer的决定。

    FINAL：
        生成可直接发送给用户的最终回答。

    RETURN_TO_WORKER：
        将事实型缺口交回最后一个Worker，在原checkpoint继续返修。

    REPLAN：
        请求使用剩余Replan机会，上限由运行配置控制。

    是否还允许Replan由Planning Graph中的
    replan_used状态确定，不由模型自行决定。
    """

    review_reason: str = Field(
        default="依据当前可见证据完成最终审核。",
        min_length=1,
        description="先概括原始要求、已确认事实和主要缺口，再填写任何决定。",
    )

    criterion_reviews: list[FinalCriterionReview] = Field(
        default_factory=list,
        description="逐项写证据、观测、缺口和状态；证据不足不能补猜。",
    )

    repair_request: FinalWorkerRepairRequest | None = Field(
        default=None,
        description="仅RETURN_TO_WORKER填写；只陈述未通过事实，不提供API或执行建议。",
    )

    # REPLAN的原因独立保留，避免返修单与Scheduler重规划混在一起。
    replan_reason: (
        str
        | None
    ) = Field(default=None, description="先填写判断依据：REPLAN写已有进展、剩余阻塞和可执行补救；FINAL填null。仅replan_available=true可选REPLAN。")

    action: FinalReviewAction = Field(description="完成证据判断后，只选FINAL、RETURN_TO_WORKER或REPLAN；此阶段不输出PLAN或steps。")

    # FINAL时使用。
    status: (
        FinalReviewStatus
        | None
    ) = Field(default=None, description="FINAL必填完成状态；REPLAN必须省略或为null。")

    final_answer: (
        str
        | None
    ) = Field(default=None, description="FINAL必填用户答复；REPLAN必须省略或为null。")

    unmet_success_criteria: list[
        str
    ] = Field(
        default_factory=list,
        description="仅FINAL列出未满足条件；REPLAN必须省略或为[]，未完成项写入replan_reason。",
    )

    @model_validator(
        mode="after",
    )
    def validate_action_fields(
        self,
    ) -> "FinalReviewDecision":
        """检查FINAL和REPLAN的字段组合。"""

        if self.action == "FINAL":
            if self.status is None:
                raise ValueError(
                    "FINAL必须提供status。"
                )

            if not self.final_answer:
                raise ValueError(
                    "FINAL必须提供final_answer。"
                )

            if self.replan_reason:
                raise ValueError(
                    "FINAL不能提供replan_reason。"
                )

            if self.repair_request is not None:
                raise ValueError(
                    "FINAL不能提供repair_request。"
                )

            return self

        if self.action == "RETURN_TO_WORKER":
            if self.repair_request is None:
                raise ValueError(
                    "RETURN_TO_WORKER必须提供repair_request。"
                )
            if self.replan_reason:
                raise ValueError(
                    "RETURN_TO_WORKER不能提供replan_reason。"
                )
            if self.status is not None or self.final_answer or self.unmet_success_criteria:
                raise ValueError(
                    "RETURN_TO_WORKER要求status=null、final_answer=null、"
                    "unmet_success_criteria=[]。"
                )
            return self

        if not self.replan_reason:
            raise ValueError(
                "REPLAN必须提供replan_reason。"
            )

        if (
            self.status is not None
            or self.final_answer
            or self.unmet_success_criteria
            or self.repair_request is not None
        ):
            raise ValueError(
                "REPLAN要求status=null、final_answer=null、unmet_success_criteria=[]；"
                "未完成项和补救方案写入replan_reason，不填写新计划。"
            )

        return self
