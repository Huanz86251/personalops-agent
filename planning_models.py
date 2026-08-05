from __future__ import annotations

from typing import (
    Literal,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


SupervisorAction = Literal[
    "FINAL",
    "PLAN",
]


ReplanAction = Literal[
    "CONTINUE",
    "FINISH",
]


FinalReviewAction = Literal[
    "FINAL",
    "REPLAN",
]


StepStatus = Literal[
    "COMPLETED",
    "PARTIAL",
    "BLOCKED",
    "FAILED",
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

    # 较早Conversation的滚动摘要。
    conversation_summary: str = ""

    # 最近若干轮用户与最终助手原文。
    recent_dialogue: list[
        DialogueMessage
    ] = Field(
        default_factory=list,
    )

    # 根据当前用户请求召回的长期记忆。
    memory_context: str = ""

    # 当前实际可用的业务能力目录。
    toolset_catalog: list[
        dict[
            str,
            str,
        ]
    ] = Field(
        default_factory=list,
    )


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
    )

    success_criteria: list[
        str
    ] = Field(
        min_length=1,
    )

    # Hard模型提供给Simple Executor的
    # 少量高价值执行建议。
    execution_guidance: (
        str
        | None
    ) = None


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
    ) = None

    plan_success_criteria: list[
        str
    ] = Field(
        default_factory=list,
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

    criterion: str = Field(
        min_length=1,
    )

    status: CriterionStatus

    evidence: list[
        str
    ] = Field(
        default_factory=list,
    )


class StepReport(
    PlanningModel
):
    """Simple Step Reporter生成的压缩报告。"""

    step_id: int = Field(
        ge=1,
    )

    status: StepStatus

    summary: str = Field(
        min_length=1,
    )

    stop_reason: str = Field(
        min_length=1,
    )

    criterion_results: list[
        StepCriterionResult
    ] = Field(
        default_factory=list,
    )

    confirmed_results: list[
        str
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

    next_action: (
        str
        | None
    ) = None

    # Reporter只能建议Replan，
    # 不能自己生成新Plan。
    request_replan: bool = False

    replan_reason: (
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
    """唯一一次Hard Replanner的决定。

    CONTINUE：
        使用remaining_steps继续执行。

    FINISH：
        停止继续执行Step，
        交给Final Reviewer根据已有报告收口。
    """

    action: ReplanAction

    reason: str = Field(
        min_length=1,
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

            return self

        if self.remaining_steps:
            raise ValueError(
                "FINISH不能提供remaining_steps。"
            )

        return self


class FinalReviewDecision(
    PlanningModel
):
    """Hard Final Reviewer的决定。

    FINAL：
        生成可直接发送给用户的最终回答。

    REPLAN：
        请求使用全局唯一一次Replan机会。

    是否还允许Replan由Planning Graph中的
    replan_used状态确定，不由模型自行决定。
    """

    action: FinalReviewAction

    # FINAL时使用。
    status: (
        FinalReviewStatus
        | None
    ) = None

    final_answer: (
        str
        | None
    ) = None

    unmet_success_criteria: list[
        str
    ] = Field(
        default_factory=list,
    )

    # REPLAN时使用。
    replan_reason: (
        str
        | None
    ) = None

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

            return self

        if not self.replan_reason:
            raise ValueError(
                "REPLAN必须提供replan_reason。"
            )

        if (
            self.status is not None
            or self.final_answer
            or self.unmet_success_criteria
        ):
            raise ValueError(
                "REPLAN不能同时提供"
                "最终回答字段。"
            )

        return self