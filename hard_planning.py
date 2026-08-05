from __future__ import annotations

import json
import logging

from collections.abc import (
    Callable,
    Mapping,
)
from dataclasses import (
    dataclass,
)
from typing import (
    Any,
    Generic,
    TypeVar,
)

from pydantic import (
    BaseModel,
)

from observability import (
    set_span_attributes,
    set_span_output,
    trace_span,
)
from planning_models import (
    FinalReviewDecision,
    PlanStep,
    PlanningContextPack,
    ReplanDecision,
    StepReport,
    SupervisorDecision,
)
from prompt_loader import (
    render_prompt,
)


logger = logging.getLogger(
    "agent"
)


MAX_VALIDATION_RETRIES = 1
VALIDATION_ERROR_MAX_CHARS = 1000


SchemaT = TypeVar(
    "SchemaT",
    bound=BaseModel,
)


@dataclass(
    frozen=True,
)
class PlanningCallResult(
    Generic[
        SchemaT
    ]
):
    """一次Hard结构化模型调用的统一结果。"""

    output: SchemaT

    # 真正收到多少条模型输出。
    # 第一次格式错误、第二次修复成功时为2。
    model_rounds_used: int

    validation_retry_count: int

    # True表示最终输出来自Python确定性兜底。
    used_fallback: bool


def _require_text(
    value: str,
    field_name: str,
) -> str:
    """检查必填文本。"""

    normalized_value = value.strip()

    if not normalized_value:
        raise ValueError(
            f"{field_name}不能为空。"
        )

    return normalized_value


def _json_default(
    value: Any,
) -> Any:
    """让json.dumps支持Pydantic对象。"""

    if isinstance(
        value,
        BaseModel,
    ):
        return value.model_dump(
            mode="json",
        )

    return str(
        value
    )


def _prompt_text(
    value: Any,
    empty_text: str,
) -> str:
    """把普通对象转换成Prompt中的稳定文字。"""

    if value is None:
        return empty_text

    if isinstance(
        value,
        str,
    ):
        return (
            value.strip()
            or empty_text
        )

    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        default=_json_default,
    )


def _format_hard_context(
    context: PlanningContextPack,
) -> str:
    """生成所有Hard节点共用的基础上下文视图。

    Supervisor、Replanner和Final Reviewer调用同一个函数，
    不分别管理时间、历史摘要、近期对话和长期记忆。
    """

    validated_context = (
        PlanningContextPack
        .model_validate(
            context
        )
    )

    return (
        "[当前时间]\n"
        f"{validated_context.current_time}\n\n"

        "[当前用户请求]\n"
        f"{validated_context.user_request}\n\n"

        "[较早Conversation摘要]\n"
        f"{_prompt_text(validated_context.conversation_summary, '没有较早对话摘要。')}\n\n"

        "[最近用户与最终助手对话]\n"
        f"{_prompt_text(validated_context.recent_dialogue, '没有近期对话。')}\n\n"

        "[本轮相关长期记忆]\n"
        f"{_prompt_text(validated_context.memory_context, '没有召回相关长期记忆。')}\n\n"

        "[当前可用能力目录]\n"
        f"{_prompt_text(validated_context.toolset_catalog, '当前没有可用业务能力。')}"
    )


def _build_validation_repair_message(
    *,
    schema_name: str,
    error: Any,
) -> str:
    """要求模型只修复结构化输出。"""

    error_text = str(
        error
        or "结构化输出没有通过校验。"
    ).strip()

    error_text = error_text[
        :VALIDATION_ERROR_MAX_CHARS
    ]

    return (
        "上一次输出没有通过结构化校验。\n"
        f"目标Schema：{schema_name}\n"
        "请保持原任务和业务判断不变，"
        "只重新输出符合Schema的结果。\n\n"
        "校验问题：\n"
        f"{error_text}"
    )


async def _invoke_structured(
    model,
    *,
    prompt: str,
    output_schema: type[
        SchemaT
    ],
    trace_name: str,
    fallback_factory: Callable[
        [str],
        SchemaT,
    ],
) -> PlanningCallResult[
    SchemaT
]:
    """执行结构化调用、一次格式修复和安全降级。

    每一次真正发起的模型请求都会计入模型轮次。

    模型截断、超时或其他调用异常时，
    如果仍有固定修复机会，就继续下一轮。

    最终仍然失败时，
    使用当前节点对应的Python确定性兜底。
    """

    normalized_prompt = _require_text(
        prompt,
        "prompt",
    )

    schema_name = (
        output_schema.__name__
    )

    # Thinking模式下统一使用JSON Mode，
    # 不依赖function_calling中的强制tool_choice。
    structured_prompt = (
        f"{normalized_prompt}\n\n"

        "[结构化输出要求]\n"
        "必须只输出一个JSON对象；"
        "不要输出Markdown代码块或额外说明。\n"
        "JSON Schema：\n"

        + json.dumps(
            output_schema.model_json_schema(),

            ensure_ascii=False,

            indent=2,
        )
    )

    structured_model = (
        model.with_structured_output(
            output_schema,

            method="json_mode",

            include_raw=True,
        )
    )

    messages: list[
        dict[
            str,
            str,
        ]
    ] = [
        {
            "role": "system",

            "content": (
                structured_prompt
            ),
        }
    ]

    model_rounds_used = 0
    validation_retry_count = 0

    last_error = (
        "未知结构化输出错误。"
    )

    with trace_span(
        trace_name,

        kind="chain",

        input_value={
            "schema": (
                schema_name
            ),

            "max_validation_retries": (
                MAX_VALIDATION_RETRIES
            ),
        },
    ) as span:

        for attempt_index in range(
            MAX_VALIDATION_RETRIES
            + 1
        ):
            if attempt_index > 0:
                validation_retry_count += 1

                messages.append(
                    {
                        "role": "user",

                        "content": (
                            _build_validation_repair_message(
                                schema_name=(
                                    schema_name
                                ),

                                error=(
                                    last_error
                                ),
                            )
                        ),
                    }
                )

            # 只要真实发起一次模型请求，
            # 无论成功、截断、超时还是解析失败，
            # 都必须计入Planning模型预算。
            model_rounds_used += 1

            try:
                response = await (
                    structured_model
                    .ainvoke(
                        messages
                    )
                )

            except Exception as error:
                last_error = (
                    f"{type(error).__name__}: "
                    f"{error}"
                )

                if (
                    attempt_index
                    < MAX_VALIDATION_RETRIES
                ):
                    continue

                break

            if not isinstance(
                response,
                Mapping,
            ):
                last_error = (
                    "with_structured_output"
                    "没有返回Mapping结果。"
                )

                continue

            parsing_error = response.get(
                "parsing_error"
            )

            if parsing_error is not None:
                last_error = str(
                    parsing_error
                )

                continue

            try:
                validated_output = (
                    output_schema
                    .model_validate(
                        response.get(
                            "parsed"
                        )
                    )
                )

            except Exception as error:
                last_error = str(
                    error
                )

                continue

            result = PlanningCallResult(
                output=(
                    validated_output
                ),

                model_rounds_used=(
                    model_rounds_used
                ),

                validation_retry_count=(
                    validation_retry_count
                ),

                used_fallback=False,
            )

            set_span_attributes(
                span,

                **{
                    "planning.success": True,

                    "planning.used_fallback": False,

                    "planning.model_rounds_used": (
                        model_rounds_used
                    ),

                    "planning.validation_retry_count": (
                        validation_retry_count
                    ),
                },
            )

            set_span_output(
                span,

                {
                    "status": "success",

                    "output": (
                        validated_output
                    ),

                    "model_rounds_used": (
                        model_rounds_used
                    ),

                    "validation_retry_count": (
                        validation_retry_count
                    ),
                },
            )

            return result

        fallback_output = (
            fallback_factory(
                last_error
            )
        )

        logger.warning(
            "%s结构化调用失败，"
            "已使用安全兜底 | "
            "schema=%s | rounds=%s | "
            "error=%s",

            trace_name,

            schema_name,

            model_rounds_used,

            last_error,
        )

        set_span_attributes(
            span,

            **{
                "planning.success": False,

                "planning.used_fallback": True,

                "planning.model_rounds_used": (
                    model_rounds_used
                ),

                "planning.validation_retry_count": (
                    validation_retry_count
                ),
            },
        )

        set_span_output(
            span,

            {
                "status": "fallback",

                "error": (
                    last_error
                ),

                "fallback_output": (
                    fallback_output
                ),

                "model_rounds_used": (
                    model_rounds_used
                ),

                "validation_retry_count": (
                    validation_retry_count
                ),
            },
        )

        return PlanningCallResult(
            output=(
                fallback_output
            ),

            model_rounds_used=(
                model_rounds_used
            ),

            validation_retry_count=(
                validation_retry_count
            ),

            used_fallback=True,
        )

def _build_supervisor_fallback(
    _error: str,
) -> SupervisorDecision:
    """Supervisor失败时退化为一个通用Step。"""

    return SupervisorDecision(
        action="PLAN",
        final_answer=None,
        plan_objective=(
            "在当前可用能力范围内"
            "完成用户请求。"
        ),
        plan_success_criteria=[
            "产出可以直接发送给用户的可靠回答。",
            "不伪造未经执行或验证的结果。",
        ],
        steps=[
            PlanStep(
                step_id=1,
                objective=(
                    "使用当前可用能力处理"
                    "用户请求并形成可靠结果。"
                ),
                success_criteria=[
                    "得到与用户请求直接相关的结果。",
                    "明确说明无法确认或未完成的部分。",
                ],
                execution_guidance=(
                    "优先直接完成；需要工具时"
                    "选择最相关的能力，"
                    "不要伪造工具结果。"
                ),
            )
        ],
    )


def _build_replanner_fallback(
    _error: str,
) -> ReplanDecision:
    """Replanner失败时停止扩展计划。"""

    return ReplanDecision(
        action="FINISH",
        reason=(
            "Replanner未能生成可靠的"
            "结构化结果，因此不再扩展计划，"
            "交给Final Reviewer根据现有报告收口。"
        ),
        remaining_steps=[],
    )


def _build_final_reviewer_fallback(
    *,
    step_reports: list[
        StepReport
    ],
    plan_success_criteria: list[
        str
    ],
) -> FinalReviewDecision:
    """Final Reviewer失败时确定性生成最终回答。"""

    confirmed_results: list[str] = []
    unresolved_items: list[str] = []
    errors: list[str] = []

    for report in step_reports:
        confirmed_results.extend(
            report.confirmed_results
        )
        unresolved_items.extend(
            report.unresolved_items
        )
        errors.extend(
            report.errors
        )

    all_completed = bool(
        step_reports
    ) and all(
        report.status == "COMPLETED"
        for report in step_reports
    )

    if all_completed and not unresolved_items:
        status = "COMPLETED"
        unmet_success_criteria: list[str] = []

    elif confirmed_results:
        status = "PARTIAL"
        unmet_success_criteria = (
            unresolved_items[:4]
            or plan_success_criteria[:4]
        )

    else:
        status = "FAILED"
        unmet_success_criteria = (
            plan_success_criteria[:4]
            or [
                "没有获得足够可靠的执行结果。"
            ]
        )

    lines = [
        "本轮任务已完成安全收尾。"
    ]

    if confirmed_results:
        lines.extend(
            [
                "",
                "已确认结果：",
                *[
                    f"- {item}"
                    for item in confirmed_results[:8]
                ],
            ]
        )

    if unresolved_items:
        lines.extend(
            [
                "",
                "尚未解决：",
                *[
                    f"- {item}"
                    for item in unresolved_items[:4]
                ],
            ]
        )

    if errors:
        lines.extend(
            [
                "",
                "执行中遇到的问题：",
                *[
                    f"- {item}"
                    for item in errors[:4]
                ],
            ]
        )

    if not confirmed_results:
        lines.extend(
            [
                "",
                (
                    "最终审核模型未能生成可靠结果，"
                    "因此本轮没有伪造结论。"
                ),
            ]
        )

    return FinalReviewDecision(
        action="FINAL",
        status=status,
        final_answer="\n".join(
            lines
        ),
        unmet_success_criteria=(
            unmet_success_criteria
        ),
        replan_reason=None,
    )


async def run_hard_supervisor(
    hard_model,
    *,
    context: PlanningContextPack,
    max_steps_per_plan: int,
) -> PlanningCallResult[
    SupervisorDecision
]:
    """运行初始Hard Supervisor。"""

    if max_steps_per_plan < 1:
        raise ValueError(
            "max_steps_per_plan不能小于1。"
        )

    prompt = render_prompt(
        "hard_supervisor",
        hard_context=(
            _format_hard_context(
                context
            )
        ),
        max_steps_per_plan=(
            max_steps_per_plan
        ),
    )

    return await _invoke_structured(
        hard_model,
        prompt=prompt,
        output_schema=SupervisorDecision,
        trace_name="hard_supervisor",
        fallback_factory=(
            _build_supervisor_fallback
        ),
    )


async def run_hard_replanner(
    hard_model,
    *,
    context: PlanningContextPack,
    plan_objective: str,
    plan_success_criteria: list[
        str
    ],
    completed_step_reports: list[
        StepReport
    ],
    replan_context: str,
    remaining_steps: list[
        PlanStep
    ],
    remaining_budget: Any,
    next_step_id: int,
    max_remaining_steps: int,
) -> PlanningCallResult[
    ReplanDecision
]:
    """运行全局唯一的Hard Replanner。

    Replanner可以：
    - CONTINUE：替换尚未执行的步骤；
    - FINISH：停止继续执行，交给Reviewer收口。
    """

    if next_step_id < 1:
        raise ValueError(
            "next_step_id不能小于1。"
        )

    if max_remaining_steps < 0:
        raise ValueError(
            "max_remaining_steps不能小于0。"
        )

    prompt = render_prompt(
        "hard_replanner",
        hard_context=(
            _format_hard_context(
                context
            )
        ),
        plan_objective=(
            _require_text(
                plan_objective,
                "plan_objective",
            )
        ),
        plan_success_criteria=(
            _prompt_text(
                plan_success_criteria,
                "没有整体成功标准。",
            )
        ),
        completed_step_reports=(
            _prompt_text(
                completed_step_reports,
                "尚无已完成StepReport。",
            )
        ),
        replan_context=(
            _require_text(
                replan_context,
                "replan_context",
            )
        ),
        remaining_steps=(
            _prompt_text(
                remaining_steps,
                "原计划没有剩余步骤。",
            )
        ),
        remaining_budget=(
            _prompt_text(
                remaining_budget,
                "没有剩余预算信息。",
            )
        ),
        next_step_id=next_step_id,
        max_remaining_steps=(
            max_remaining_steps
        ),
    )

    return await _invoke_structured(
        hard_model,
        prompt=prompt,
        output_schema=ReplanDecision,
        trace_name="hard_replanner",
        fallback_factory=(
            _build_replanner_fallback
        ),
    )


async def run_hard_final_reviewer(
    hard_model,
    *,
    context: PlanningContextPack,
    plan_objective: str,
    plan_success_criteria: list[
        str
    ],
    step_reports: list[
        StepReport
    ],
    replan_history: Any,
    overall_stop_reason: str,
    replan_available: bool,
) -> PlanningCallResult[
    FinalReviewDecision
]:
    """运行Hard Final Reviewer。

    Reviewer可以：
    - FINAL：生成最终回答；
    - REPLAN：仅在replan_available为True时请求唯一一次Replan。

    是否真正允许跳转由Planning Graph执行。
    """

    prompt = render_prompt(
        "hard_final_reviewer",
        hard_context=(
            _format_hard_context(
                context
            )
        ),
        plan_objective=(
            _require_text(
                plan_objective,
                "plan_objective",
            )
        ),
        plan_success_criteria=(
            _prompt_text(
                plan_success_criteria,
                "没有整体成功标准。",
            )
        ),
        step_reports=(
            _prompt_text(
                step_reports,
                "没有可用StepReport。",
            )
        ),
        replan_history=(
            _prompt_text(
                replan_history,
                "本次任务没有执行Replan。",
            )
        ),
        overall_stop_reason=(
            overall_stop_reason.strip()
            or "正常进入最终审核。"
        ),
        replan_available=(
            "是"
            if replan_available
            else "否"
        ),
    )

    return await _invoke_structured(
        hard_model,
        prompt=prompt,
        output_schema=(
            FinalReviewDecision
        ),
        trace_name=(
            "hard_final_reviewer"
        ),
        fallback_factory=(
            lambda _error: (
                _build_final_reviewer_fallback(
                    step_reports=step_reports,
                    plan_success_criteria=(
                        plan_success_criteria
                    ),
                )
            )
        ),
    )
