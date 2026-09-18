from __future__ import annotations

import json
import logging
import hashlib

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
    ValidationError,
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
from workers.leadership_models import (
    LeadershipDecision,
    LeadershipWakeRequest,
)
from workers.code_review_models import (
    CodeReviewLoopState,
    CodeReviewReport,
    SchedulerCodeDecision,
)
from prompt_loader import (
    render_prompt,
    load_prompt,
    split_prompt,
)
from skill_runtime import skill_prompt
from schema_utils import is_schema_repairable_error, schema_repair_feedback


logger = logging.getLogger(
    "agent"
)


MAX_VALIDATION_RETRIES = 3
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
    validation_errors: tuple[str, ...] = ()


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

    Supervisor和Replanner调用同一个函数，
    不分别管理时间、历史摘要、近期对话和长期记忆。
    """

    validated_context = (
        PlanningContextPack
        .model_validate(
            context
        )
    )

    blocks: list[str] = []
    if validated_context.execution_instructions:
        blocks.append(
            "[本轮执行环境与技能]\n"
            f"{validated_context.execution_instructions}"
        )
    blocks.extend(
        [
            validated_context.current_time_context(),
            "[当前用户请求]\n"
            f"{validated_context.user_request}",
        ]
    )
    if validated_context.replacement_context is not None:
        blocks.append(
            "[用户替换任务上下文]\n"
            f"{_prompt_text(validated_context.replacement_context, '没有替换上下文。')}"
        )
    blocks.extend(
        [
            "[历史用户原文（按时间顺序，未经摘要）]\n"
            f"{_prompt_text(validated_context.user_instruction_history, '没有历史用户原文。')}",
            "[较早Conversation摘要]\n"
            f"{_prompt_text(validated_context.conversation_summary, '没有较早对话摘要。')}",
            "[上一轮内部执行摘要（仅供参考，不能覆盖用户原话）]\n"
            f"{_prompt_text(validated_context.previous_run_summary, '没有上一轮内部执行摘要。')}",
            "[最近用户与最终助手对话]\n"
            f"{_prompt_text(validated_context.recent_dialogue, '没有近期对话。')}",
            "[本轮相关长期记忆]\n"
            f"{_prompt_text(validated_context.memory_context, '没有召回相关长期记忆。')}",
            "[当前可用能力目录]\n"
            f"{_prompt_text(validated_context.toolset_catalog, '当前没有可用业务能力。')}",
        ]
    )
    return "\n\n".join(blocks)


def _validation_error_text(error: Any) -> str:
    """Keep field diagnostics, not the failed JSON copied by SDK wrappers."""
    pending, seen = [error], set()
    while pending:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, ValidationError):
            errors = current.errors(include_url=False, include_context=False, include_input=False)
            shown = errors[:10]
            width = (VALIDATION_ERROR_MAX_CHARS - 80) // max(1, len(shown))
            lines = []
            for item in shown:
                path = ".".join(map(str, item["loc"])) or "$"
                line = f"{path}: {item['msg']} ({item['type']})"
                lines.append(line if len(line) <= width else line[:width - 1] + "…")
            if len(errors) > len(shown):
                lines.append(f"另有 {len(errors) - len(shown)} 项错误未展开。")
            return "\n".join(lines)
        for name in ("__cause__", "__context__"):
            nested = getattr(current, name, None)
            if nested is not None:
                pending.append(nested)
    text = str(error or "结构化输出没有通过校验。").strip()
    if len(text) > VALIDATION_ERROR_MAX_CHARS:
        return "[错误详情过长，仅保留末尾]\n" + text[-(VALIDATION_ERROR_MAX_CHARS - 40):]
    return text


def _build_validation_repair_message(
    *,
    schema_name: str,
    output_schema: type[BaseModel],
    error: Any,
) -> str:
    """要求同一个角色保留业务判断，只修复结构化表单。"""

    return schema_repair_feedback(
        schema_name=schema_name,
        schema=output_schema.model_json_schema(),
        error_text=_validation_error_text(error),
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
    skill_snapshot: dict[str, Any] | None = None,
    scheduler_session=None,
    scheduler_messages=None,
    scheduler_decision_key: str | None = None,
    scheduler_decision_protected: bool = True,
    candidate_validator: Callable[[SchemaT], SchemaT] | None = None,
) -> PlanningCallResult[
    SchemaT
]:
    """执行结构化调用、最多三次格式修复和安全降级。

    每一次真正发起的模型请求都会计入模型轮次。

    仅结构校验、解析或输出截断错误使用格式修复轮次；
    超时、连接、限流或服务异常直接走当前节点的安全降级。

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

    structured_model = model.with_structured_output(
        output_schema, method="json_mode", include_raw=True,
    )
    if scheduler_messages is not None:
        messages = list(scheduler_messages)
    else:
        from scheduler_runtime import compact_json, compact_schema
        fixed_prompt, runtime_context = split_prompt(normalized_prompt)
        methods = skill_prompt(skill_snapshot)
        messages = [{"role": "system", "content": fixed_prompt + "\nSchema:"
                     + compact_json(compact_schema(output_schema.model_json_schema()))
                     + ("\n" + methods if methods else "")}]
        if runtime_context:
            messages.append({"role": "user", "content": runtime_context})

    model_rounds_used = 0
    validation_retry_count = 0
    failed_plan = None
    validation_errors = []

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
                validation_errors.append(_validation_error_text(last_error))
                validation_retry_count += 1

                if failed_plan is not None:
                    messages.append({"role": "assistant", "content": failed_plan})

                messages.append(
                    {
                        "role": "user",

                        "content": (
                            _build_validation_repair_message(
                                schema_name=schema_name,
                                output_schema=output_schema,
                                error=last_error,
                            )
                        ),
                    }
                )

            # 只要真实发起一次模型请求，
            # 无论成功、截断、超时还是解析失败，
            # 都必须计入Planning模型预算。
            model_rounds_used += 1
            failed_plan = None

            try:
                response = await (
                    structured_model
                    .ainvoke(
                        messages
                    )
                )

            except Exception as error:
                last_error = error
                raw_output = getattr(error, "llm_output", None)
                if isinstance(raw_output, str) and raw_output:
                    failed_plan = raw_output

                if (
                    attempt_index < MAX_VALIDATION_RETRIES
                    and is_schema_repairable_error(error)
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

            raw = response.get("raw")
            raw_content = raw.get("content") if isinstance(raw, Mapping) else getattr(raw, "content", None)
            if isinstance(raw_content, str) and raw_content:
                failed_plan = raw_content

            if parsing_error is not None:
                last_error = parsing_error
                if failed_plan is not None:
                    try:
                        output_schema.model_validate_json(failed_plan)
                    except ValidationError as error:
                        last_error = error

                continue

            try:
                candidate = response.get("parsed")
                validated_output = (
                    output_schema
                    .model_validate(
                        candidate
                    )
                )
                if candidate_validator is not None:
                    validated_output = candidate_validator(validated_output)

            except Exception as error:
                last_error = error

                continue

            if scheduler_session is not None:
                scheduler_session.add(
                    "decision",
                    validated_output.model_dump_json(),
                    role="assistant",
                    protected=scheduler_decision_protected,
                    key=scheduler_decision_key,
                )

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

        validation_errors.append(_validation_error_text(last_error))
        fallback_output = (
            fallback_factory(
                _validation_error_text(last_error)
            )
        )

        if scheduler_session is not None:
            scheduler_session.add(
                "decision",
                fallback_output.model_dump_json(),
                role="assistant",
                protected=scheduler_decision_protected,
                key=scheduler_decision_key,
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
                "planning.validation_errors": validation_errors,
                "planning.business_status": "FAILED" if output_schema is SupervisorDecision else "FALLBACK",

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
                "validation_errors": validation_errors,

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
            validation_errors=tuple(validation_errors),
        )

def _build_supervisor_fallback(
    error: str,
) -> SupervisorDecision:
    """Return a deterministic failure; never invent a replacement task."""

    return SupervisorDecision(
        action="FINAL",
        final_answer=(
            "本次任务失败：首次计划生成及一次修复均未获得有效计划。\n"
            f"原因：{error}\n"
            "尚未启动任务执行，已停止本次任务。"
        ),
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
    error: str = "",
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

    budget_exhausted = "budget exhausted" in error.lower()
    all_completed = bool(
        step_reports
    ) and all(
        report.status == "COMPLETED"
        for report in step_reports
    )

    if all_completed and not unresolved_items and not budget_exhausted:
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

    lines = [(
        "整题模型调用预算已耗尽，Harness 已停止继续调用模型并生成这份确定性终止报告。"
        if budget_exhausted
        else "本轮任务已完成安全收尾。"
    )]

    if budget_exhausted:
        lines.extend([
            "",
            "这不是 Final Reviewer 的模型结论；以下只汇总预算耗尽前已经落盘的报告，未确认内容不会被当作完成。",
        ])

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
        review_reason=(
            "模型调用预算耗尽；Harness 仅依据已落盘 StepReport 确定性收尾。"
            if budget_exhausted
            else "Final Reviewer 未能生成合法结果；Harness 仅依据已落盘 StepReport 安全收尾。"
        ),
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


def _validate_supervisor_scope_contract(
    decision: SupervisorDecision,
    contract,
) -> SupervisorDecision:
    """Require one planned target contract to preserve the frozen scope tail."""
    if contract is None or decision.action != "PLAN":
        return decision
    expected_context = [
        item.model_dump(mode="json") for item in contract.required_context
    ]
    expected_time_ranges = [
        item.model_dump(mode="json") for item in contract.resolved_time_ranges
    ]
    candidates = [
        step.target_selection
        for step in decision.steps
        if step.target_selection is not None
        and step.target_selection.target_entity == contract.target_entity
    ]
    if not candidates:
        raise ValueError(
            "ScopeContract exists but no Step.target_selection preserves its target_entity."
        )
    matching = [
        selection for selection in candidates
        if selection.effect_mode == contract.effect_mode
        and [
            item.model_dump(mode="json")
            for item in selection.required_context
        ] == expected_context
        and [
            item.model_dump(mode="json")
            for item in selection.resolved_time_ranges
        ] == expected_time_ranges
    ]
    if not matching:
        raise ValueError(
            "Step.target_selection must copy ScopeContract.effect_mode and "
            "required_context and resolved_time_ranges exactly; preserve field values "
            "and list order."
        )
    return decision


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

    event = {"max_steps_per_plan": max_steps_per_plan}

    return await _invoke_scheduler_stage(
        hard_model,
        context=context, stage="supervisor", event=event,
        output_schema=SupervisorDecision,
        trace_name="hard_supervisor",
        fallback_factory=(
            _build_supervisor_fallback
        ),
        candidate_validator=lambda decision: _validate_supervisor_scope_contract(
            decision, context.scope_contract
        ),
    )


def _build_worker_leadership_fallback(error: str) -> LeadershipDecision:
    """Fail open so a malformed leader response does not kill useful work."""

    return LeadershipDecision(
        action="CONTINUE",
        reason=(
            "Leadership output could not be validated; the deterministic "
            f"safe fallback is CONTINUE. Error: {error[:500]}"
        ),
    )


async def run_hard_worker_leader(
    hard_model,
    *,
    request: LeadershipWakeRequest,
) -> PlanningCallResult[LeadershipDecision]:
    """Ask the main model for one bounded, structured Worker-control action."""

    from scheduler_runtime import CURRENT_SCHEDULER
    session = CURRENT_SCHEDULER.get()
    if session is None:
        # Standalone bridge compatibility; normal graph calls always bind a session.
        context = PlanningContextPack(current_time=str(request.created_at), user_request="处理 Worker 进度")
    else:
        context = session.context
    return await _invoke_scheduler_stage(
        hard_model, context=context, stage="worker_leader",
        event=request.model_dump(mode="json"), output_schema=LeadershipDecision,
        trace_name="hard_worker_leader", fallback_factory=_build_worker_leadership_fallback,
    )


def _build_code_scheduler_fallback(error: str) -> SchedulerCodeDecision:
    """Stop safely when the CODE control decision cannot be validated."""

    return SchedulerCodeDecision(
        action="STOP",
        reason=(
            "Scheduler could not produce a reliable structured CODE recovery "
            f"decision, so the safe fallback is STOP. Error: {error[:500]}"
        ),
    )


async def run_hard_code_scheduler(
    hard_model,
    *,
    context: PlanningContextPack,
    plan_objective: str,
    current_step: PlanStep,
    code_review_loop: CodeReviewLoopState,
    code_review_report: CodeReviewReport,
    code_control_history: Any,
    remaining_budget: Any,
) -> PlanningCallResult[SchedulerCodeDecision]:
    """Choose CONTINUE, RESTART, or STOP for one escalated CODE attempt."""

    if current_step.worker_kind != "CODE" or current_step.code_task is None:
        raise ValueError("CODE Scheduler control requires a frozen CODE Step.")
    if code_review_loop.status != "ESCALATED_TO_SCHEDULER":
        raise ValueError("CODE Scheduler control requires an escalated review loop.")
    if code_review_report.candidate != code_review_loop.candidate:
        raise ValueError("CODE Scheduler received a stale Reviewer report.")
    if code_review_report.verdict == "PASSED":
        raise ValueError("An applied CODE result does not need Scheduler recovery.")

    event = {"current_step": current_step, "review_loop": code_review_loop, "review_report": code_review_report, "remaining_budget": remaining_budget}

    return await _invoke_scheduler_stage(
        hard_model,
        context=context, stage="code_controller", event=event,
        output_schema=SchedulerCodeDecision,
        trace_name="hard_code_scheduler",
        fallback_factory=_build_code_scheduler_fallback,
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
    - CONTINUE：替换争议Step及尚未执行的步骤；
    - RETURN_TO_WORKER：驳回计划异议并让原Worker继续；
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

    from scheduler_runtime import conversation
    session = conversation(context)
    session.initialize()
    session.fact("验收目标", {"objective": plan_objective, "success_criteria": plan_success_criteria})
    for report in completed_step_reports:
        session.fact("StepReport", report)
    event = {
        "reason": replan_context,
        "completed_step_ids": [report.step_id for report in completed_step_reports],
        "remaining_steps": remaining_steps,
        "remaining_budget": remaining_budget,
        "next_step_id": next_step_id,
        "max_remaining_steps": max_remaining_steps,
    }

    return await _invoke_scheduler_stage(
        hard_model,
        context=context, stage="replanner", event=event,
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
    plan_steps: list[PlanStep] | None = None,
    latest_worker_evidence: dict[str, Any] | None = None,
    repair_history: list[dict[str, Any]] | None = None,
    repair_round: int = 0,
    max_repair_rounds: int = 0,
) -> PlanningCallResult[
    FinalReviewDecision
]:
    """运行Hard Final Reviewer。

    Reviewer可以：
    - FINAL：生成最终回答；
    - REPLAN：仅在replan_available为True时请求预算内Replan。

    是否真正允许跳转由Planning Graph执行。
    """

    # Final review is an independent evidence check. It does not inherit
    # Scheduler capability catalogs, planning protocols, or process history.
    # It does receive the same Harness-owned clock label as every other role.
    from scheduler_runtime import compact_json
    from schema_utils import compact_schema

    reviewer_prompt = load_prompt("planning/final_reviewer")
    reviewer_skill = skill_prompt(
        context.role_skill_snapshots.get("final_reviewer")
    )
    if reviewer_skill:
        reviewer_prompt += "\n\n" + reviewer_skill
    reviewer_messages = [{
        "role": "system",
        "content": reviewer_prompt + "\nSchema:"
        + compact_json(compact_schema(FinalReviewDecision.model_json_schema())),
    }, {
        "role": "user",
        "content": compact_json({
            "当前时间": context.current_time_context(),
            "用户原始请求": context.user_request,
            "已校验范围合同": context.scope_contract,
            "计划验收目标": {
                "objective": plan_objective,
                "success_criteria": plan_success_criteria,
                "steps": plan_steps or [],
            },
            "独立审核材料": {
                "step_reports": step_reports,
                "latest_worker_evidence": latest_worker_evidence or {},
                "final_worker_repair_history": repair_history or [],
                "stop_reason": overall_stop_reason,
                "return_to_worker_available": (
                    repair_round < max_repair_rounds
                    and bool(latest_worker_evidence)
                ),
                "repair_round": repair_round,
                "max_repair_rounds": max_repair_rounds,
                "replan_available": replan_available,
                "replan_history": replan_history,
            },
        }),
    }]

    if context.completion_api_contract:
        reviewer_messages.append({
            "role": "user",
            "content": context.completion_api_contract,
        })

    return await _invoke_structured(
        hard_model,
        prompt=reviewer_prompt,
        output_schema=FinalReviewDecision,
        trace_name="hard_final_reviewer",
        scheduler_messages=reviewer_messages,
        fallback_factory=lambda _error: _build_final_reviewer_fallback(
            step_reports=step_reports,
            plan_success_criteria=plan_success_criteria,
            error=_error,
        ),
    )


async def _invoke_scheduler_stage(
    model, *, context, stage, event, output_schema, trace_name,
    fallback_factory, candidate_validator=None,
):
    from scheduler_runtime import compact_json, conversation
    session = conversation(context)
    async with session.lock:
        session.initialize()
        protocol = session.disclose(stage, output_schema.model_json_schema())
        # Stage inputs and structured outputs are request-local UI blocks.
        # Canonical state is retained separately as graph facts/projections,
        # so old plans, Web progress packets, and final-answer drafts do not
        # accumulate in the reusable Scheduler prefix.
        transient = True
        event_key = (
            "transient:event:" + hashlib.sha256(compact_json(event).encode()).hexdigest()
            if transient
            else None
        )
        decision_key = (
            "transient:decision:" + hashlib.sha256(compact_json(event).encode()).hexdigest()
            if transient
            else None
        )
        session.add(
            "event",
            {"使用协议": protocol, "事件": event},
            protected=not transient,
            key=event_key,
        )
        try:
            result = await _invoke_structured(
                model, prompt=load_prompt("planning/scheduler"), output_schema=output_schema,
                trace_name=trace_name, fallback_factory=fallback_factory,
                scheduler_session=session, scheduler_messages=session.wire(),
                scheduler_decision_key=decision_key,
                scheduler_decision_protected=not transient,
                candidate_validator=candidate_validator,
            )
            if stage == "worker_leader" and result.output.action in {"GUIDE", "ACCEPT", "CANCEL", "REPLACE"}:
                session.set_active(
                    "当前Worker控制",
                    result.output.model_dump(mode="json"),
                    role="assistant",
                )
            elif stage == "supervisor" and result.output.action == "PLAN":
                session.set_active(
                    "活动计划",
                    {
                        "plan_objective": result.output.plan_objective,
                        "plan_success_criteria": result.output.plan_success_criteria,
                        "remaining_steps": result.output.steps,
                    },
                )
            elif stage == "replanner":
                session.set_active(
                    "活动计划",
                    {
                        "status": result.output.action,
                        "reason": result.output.reason,
                        "reviewed_step_ids": event.get("completed_step_ids", []),
                        "remaining_steps": result.output.remaining_steps,
                    },
                )
            return result
        finally:
            session.remove_key(event_key)
            session.remove_key(decision_key)
