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
from langchain_core.messages import ToolMessage

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
            "[较早用户可见对话的JSON摘要]\n"
            f"{_prompt_text(validated_context.conversation_summary, '没有较早对话摘要。')}",
            "[尚未摘要的用户与最终助手原文（含固定保留的近期轮次）]\n"
            f"{_prompt_text(validated_context.recent_dialogue, '没有历史对话。')}",
            "[本轮相关长期记忆]\n"
            f"{_prompt_text(validated_context.memory_context, '没有召回相关长期记忆。')}",
            "[当前可用能力目录]\n"
            f"{_prompt_text(validated_context.toolset_catalog, '当前没有可用业务能力。')}",
        ]
    )
    if validated_context.user_instruction_history:
        blocks.append("[兼容旧调用方的历史用户原文]\n" + _prompt_text(validated_context.user_instruction_history, ""))
    if validated_context.previous_run_summary:
        blocks.append("[兼容旧调用方的内部执行摘要]\n" + validated_context.previous_run_summary)
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


def _is_output_truncation_error(error: Any) -> bool:
    """Recognize provider/SDK output-length failures through wrapped causes."""

    current = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if "lengthfinishreason" in type(current).__name__.lower():
            return True
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    text = f"{type(error).__name__}: {error}".lower()
    return "finish_reason" in text and "length" in text


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
    retry_output_truncation: bool = True,
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

    from step_user_question import ASK_USER_NAME, ask_user_available, create_ask_user_tool

    ask_user_tool = None
    tool_structured_model = None
    if ask_user_available() and hasattr(model, "bind_tools"):
        ask_user_tool = create_ask_user_tool(trace_name)
        from structured_output_policy import strict_tool_schema_options
        strict_options = strict_tool_schema_options(model)
        try:
            tool_structured_model = model.bind_tools(
                [ask_user_tool, output_schema],
                tool_choice="any",
                parallel_tool_calls=False,
                **strict_options,
            )
        except TypeError:
            tool_structured_model = model.bind_tools(
                [ask_user_tool, output_schema],
                tool_choice="any",
                **strict_options,
            )

    from structured_output_policy import (
        textual_schema_fallback,
        with_provider_structured_output,
    )

    structured_model = (
        None
        if tool_structured_model is not None
        else with_provider_structured_output(
            model,
            output_schema,
            include_raw=True,
            fallback_method="json_mode",
        )
    )
    if scheduler_messages is not None:
        messages = list(scheduler_messages)
    else:
        fixed_prompt, runtime_context = split_prompt(normalized_prompt)
        methods = skill_prompt(skill_snapshot)
        schema_guidance = textual_schema_fallback(model, output_schema)
        messages = [{
            "role": "system",
            "content": fixed_prompt
            + (("\n" + schema_guidance) if schema_guidance else "")
            + (("\n" + methods) if methods else ""),
        }]
        if runtime_context:
            messages.append({"role": "user", "content": runtime_context})

    model_rounds_used = 0
    validation_retry_count = 0
    failed_plan = None
    validation_errors = []
    human_questions_used = 0

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
                while True:
                    if tool_structured_model is None:
                        response = await structured_model.ainvoke(messages)
                        break

                    raw_response = await tool_structured_model.ainvoke(messages)
                    calls = list(getattr(raw_response, "tool_calls", []) or [])
                    ask_calls = [
                        call for call in calls
                        if str(call.get("name") or "") == ask_user_tool.name
                    ]
                    schema_calls = [
                        call for call in calls
                        if str(call.get("name") or "") == schema_name
                    ]

                    if ask_calls:
                        if human_questions_used >= 3:
                            raise ValueError(
                                "当前角色连续询问用户已达到3次；请依据已有回答提交结果。"
                            )
                        call = ask_calls[0]
                        # A Human Loop request is exclusive.  If the model also
                        # emitted a premature decision, only the question is
                        # preserved and the decision is regenerated after the
                        # answer returns.
                        additional_kwargs = dict(
                            getattr(raw_response, "additional_kwargs", {}) or {}
                        )
                        raw_calls = additional_kwargs.get("tool_calls")
                        if isinstance(raw_calls, list):
                            additional_kwargs["tool_calls"] = [
                                item for item in raw_calls
                                if isinstance(item, dict)
                                and str(item.get("id") or "") == str(call.get("id") or "")
                            ]
                        sanitized = raw_response.model_copy(
                            update={
                                "tool_calls": [call],
                                "additional_kwargs": additional_kwargs,
                            }
                        )
                        messages.append(sanitized)
                        answer = await ask_user_tool.ainvoke(call.get("args") or {})
                        messages.append(
                            ToolMessage(
                                content=str(answer),
                                tool_call_id=str(call.get("id") or ASK_USER_NAME),
                                name=ask_user_tool.name,
                            )
                        )
                        human_questions_used += 1
                        model_rounds_used += 1
                        continue

                    if len(schema_calls) == 1 and len(calls) == 1:
                        response = {
                            "parsed": schema_calls[0].get("args") or {},
                            "raw": raw_response,
                            "parsing_error": None,
                        }
                        break

                    raw_content = getattr(raw_response, "content", "")
                    try:
                        parsed_content = output_schema.model_validate_json(
                            raw_content
                        )
                    except Exception as parse_error:
                        response = {
                            "parsed": None,
                            "raw": raw_response,
                            "parsing_error": parse_error,
                        }
                    else:
                        response = {
                            "parsed": parsed_content,
                            "raw": raw_response,
                            "parsing_error": None,
                        }
                    break

            except Exception as error:
                last_error = error
                raw_output = getattr(error, "llm_output", None)
                if isinstance(raw_output, str) and raw_output:
                    failed_plan = raw_output

                if (
                    attempt_index < MAX_VALIDATION_RETRIES
                    and is_schema_repairable_error(error)
                    and (
                        retry_output_truncation
                        or not _is_output_truncation_error(error)
                    )
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

                if (
                    not retry_output_truncation
                    and _is_output_truncation_error(last_error)
                ):
                    break
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
    """Return a valid inert plan record; the graph stops it before execution."""

    return SupervisorDecision(
        plan_objective="规划生成失败后的不可执行占位记录。",
        plan_success_criteria=["不得启动该占位步骤。"],
        steps=[PlanStep(
            step_id=1,
            objective=f"不可执行：规划生成失败。{error[:300]}",
            success_criteria=["Harness 必须在启动 Worker 前终止。"],
            tool_access="DISABLED",
            worker_kind="GENERAL",
        )],
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
    if contract is None:
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
        additional_protocol_prompts=(
            ("planning/scheduler_scope_contract",)
            if context.scope_contract is not None
            else ()
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


async def run_hard_worker_objection(hard_model, *, context, event):
    """Brief Scheduler adjudication; a REPLAN decision delegates plan generation."""
    from schemas.worker_control import WorkerObjectionDecision
    return await _invoke_scheduler_stage(
        hard_model, context=context, stage="worker_objection", event=event,
        output_schema=WorkerObjectionDecision, trace_name="hard_worker_objection",
        fallback_factory=lambda error: WorkerObjectionDecision(
            action="CONTINUE", reason=f"Scheduler adjudication unavailable; continue without changing the plan. {str(error)[:400]}"
        ),
    )






def _approved_recovery_proposal(
    reports: list[StepReport],
    *,
    origin_step_id: int | None,
    origin_worker_kind: str | None,
) -> dict[str, Any] | None:
    """Return the exact downgrade contract explicitly approved by the user."""

    expected_kind = str(origin_worker_kind or "").upper()
    for report in reversed(reports):
        if origin_step_id is not None and report.step_id != origin_step_id:
            continue
        proposal = report.recovery_proposal
        if not isinstance(proposal, dict):
            continue
        if str(proposal.get("proposal_kind") or "").upper() != expected_kind:
            continue
        decision = proposal.get("human_decision") or {}
        if (
            proposal.get("downgrade_recommended") is True
            and isinstance(decision, dict)
            and decision.get("status") == "APPROVED"
        ):
            return proposal
    return None


def _validate_approved_downgrade_replan(
    decision: ReplanDecision,
    proposal: dict[str, Any] | None,
) -> ReplanDecision:
    """An approved contract change is mandatory, not Scheduler advice."""

    if proposal is None:
        return decision
    if decision.action != "CONTINUE":
        raise ValueError(
            "The user approved the exact downgrade; Replanner must choose CONTINUE "
            "and execute that approved contract."
        )
    outcome = str(proposal.get("proposed_degraded_outcome") or "").strip()
    disclosure = str(proposal.get("user_visible_disclosure") or "").strip()
    serialized = json.dumps(
        [step.model_dump(mode="json") for step in decision.remaining_steps],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if not outcome or outcome not in serialized:
        raise ValueError(
            "The first replacement Step must copy proposed_degraded_outcome "
            "verbatim into its executable contract."
        )
    if disclosure and disclosure not in serialized:
        raise ValueError(
            "The replacement plan must preserve user_visible_disclosure verbatim."
        )
    return decision


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
    replan_source: str = "unknown",
    origin_worker_kind: str | None = None,
    origin_step_id: int | None = None,
) -> PlanningCallResult[
    ReplanDecision
]:
    """运行全局唯一的Hard Replanner。

    Replanner可以：
    - CONTINUE：生成尚未执行的新步骤；
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

    from scheduler_runtime import conversation, scheduler_report_projection
    session = conversation(context)
    session.initialize()
    session.fact("验收目标", {"objective": plan_objective, "success_criteria": plan_success_criteria})
    for report in completed_step_reports:
        session.fact("StepReport", scheduler_report_projection(report))
    event = {
        "replan_source": replan_source,
        "origin_worker_kind": origin_worker_kind,
        "origin_step_id": origin_step_id,
        "reason": replan_context,
        "completed_step_ids": [report.step_id for report in completed_step_reports],
        "remaining_steps": remaining_steps,
        "remaining_budget": remaining_budget,
        "next_step_id": next_step_id,
        "max_remaining_steps": max_remaining_steps,
    }

    recovery_prompt = {
        "WEB": "planning/replan_web_recovery",
        "GENERAL": "planning/replan_general_recovery",
        "CODE": "planning/replan_code_recovery",
    }.get(str(origin_worker_kind or "").upper())

    approved_proposal = _approved_recovery_proposal(
        completed_step_reports,
        origin_step_id=origin_step_id,
        origin_worker_kind=origin_worker_kind,
    )
    protocol_prompts = tuple(
        prompt
        for prompt in (
            "planning/scheduler_scope_contract" if context.scope_contract is not None else None,
            recovery_prompt,
            "planning/approved_downgrade" if approved_proposal is not None else None,
        )
        if prompt is not None
    )

    def validate_replan(decision):
        return _validate_approved_downgrade_replan(decision, approved_proposal)

    return await _invoke_scheduler_stage(
        hard_model,
        context=context, stage="replanner", event=event,
        output_schema=ReplanDecision,
        trace_name="hard_replanner",
        fallback_factory=(
            _build_replanner_fallback
        ),
        additional_protocol_prompts=protocol_prompts,
        candidate_validator=validate_replan,
    )


def _review_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return {}


def _review_pick(value: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    source = _review_mapping(value)
    return {field: source[field] for field in fields if field in source}


def _final_review_step_projection(value: Any) -> dict[str, Any]:
    return _review_pick(
        value,
        (
            "step_id",
            "worker_kind",
            "objective",
            "success_criteria",
            "code_task",
            "artifact_outputs",
        ),
    )


def _final_review_report_projection(value: Any) -> dict[str, Any]:
    """Keep adjudicated facts and deliverables, not Reporter internals."""

    return _review_pick(
        value,
        (
            "step_id",
            "status",
            "summary",
            "answer_text",
            "stop_reason",
            "criterion_results",
            "source_links",
            "confirmed_results",
            "artifacts",
            "approved_artifact_refs",
            "errors",
            "unresolved_items",
            "assessment_source",
            "request_replan",
            "replan_reason",
            "next_action",
        ),
    )


def _final_review_history_projection(value: Any, *, limit: int = 6) -> Any:
    if not isinstance(value, (list, tuple)):
        return value
    rows = []
    for item in value[-limit:]:
        if not isinstance(item, Mapping):
            rows.append(item)
            continue
        rows.append(_review_pick(
            item,
            (
                "round",
                "attempt",
                "source",
                "action",
                "status",
                "reason",
                "replan_reason",
                "repair_request",
                "decision",
                "outcome",
                "human_decision",
            ),
        ))
    return rows


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
    return_to_worker_available: bool = False,
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
    from structured_output_policy import textual_schema_fallback

    reviewer_prompt = load_prompt("planning/final_reviewer")
    reviewer_skill = skill_prompt(
        context.role_skill_snapshots.get("final_reviewer")
    )
    if reviewer_skill:
        reviewer_prompt += "\n\n" + reviewer_skill
    reviewer_schema_guidance = textual_schema_fallback(
        hard_model,
        FinalReviewDecision,
    )
    reviewer_messages = [{
        "role": "system",
        "content": reviewer_prompt
        + (("\n" + reviewer_schema_guidance) if reviewer_schema_guidance else ""),
    }, {
        "role": "user",
        "content": compact_json({
            "当前时间": context.current_time_context(),
            "用户原始请求": context.user_request,
            "已校验范围合同": context.scope_contract,
            "计划验收目标": {
                "objective": plan_objective,
                "success_criteria": plan_success_criteria,
                "steps": [
                    _final_review_step_projection(step)
                    for step in (plan_steps or [])
                ],
            },
            "独立审核材料": {
                "step_reports": [
                    _final_review_report_projection(report)
                    for report in step_reports
                ],
                "final_worker_repair_history": _final_review_history_projection(
                    repair_history or []
                ),
                "stop_reason": overall_stop_reason,
                "return_to_worker_available": (
                    repair_round < max_repair_rounds
                    and return_to_worker_available
                ),
                "repair_round": repair_round,
                "max_repair_rounds": max_repair_rounds,
                "replan_available": replan_available,
                "replan_history": _final_review_history_projection(replan_history),
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
        retry_output_truncation=False,
    )


async def _invoke_scheduler_stage(
    model, *, context, stage, event, output_schema, trace_name,
    fallback_factory, candidate_validator=None, additional_protocol_prompts=(),
):
    from scheduler_runtime import compact_json, conversation
    from structured_output_policy import supports_strict_json_schema
    session = conversation(context)
    async with session.lock:
        session.initialize()
        protocol = session.disclose(
            stage,
            output_schema.model_json_schema(),
            additional_prompts=additional_protocol_prompts,
            include_schema=not supports_strict_json_schema(model),
        )
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
                session.set_worker_control(result.output.model_dump(mode="json"), event)
            elif stage == "supervisor" and not result.used_fallback:
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
