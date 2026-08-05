from __future__ import annotations

import json
from typing import (
    Any,
    Literal,
    TypedDict,
    cast,
)

from langchain_core.runnables import (
    RunnableConfig,
)
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agent import ask_agent
from config import PlanningSettings
from hard_planning import (
    MAX_VALIDATION_RETRIES,
    run_hard_final_reviewer,
    run_hard_replanner,
    run_hard_supervisor,
)
from planning_models import (
    FinalReviewDecision,
    PlanStep,
    PlanningContextPack,
    StepCriterionResult,
    StepReport,
)
from step_execution import run_step_reporter
from progress_events import (
    ProgressCallback,
    ProgressEvent,
    sanitize_progress_text,
)

HARD_CALL_MAX_ROUNDS = MAX_VALIDATION_RETRIES + 1


class PlanningState(TypedDict, total=False):
    """一次用户请求在Planning Graph中的共享状态。"""

    context: PlanningContextPack
    conversation_thread_id: str
    planning_run_id: str

    plan_objective: str
    plan_success_criteria: list[str]
    remaining_steps: list[PlanStep]
    completed_step_reports: list[StepReport]

    current_step: PlanStep | None
    current_step_attempt: int
    current_step_executor_rounds: int
    current_step_report_rounds: int
    current_step_model_rounds: int
    current_step_tool_calls: int
    current_step_request_toolset_calls: int
    current_step_trace: dict[str, Any]
    current_step_stop_reason: str
    retry_current_step: bool

    replans_used: int
    replan_history: list[dict[str, Any]]
    pending_replan_reason: str
    pending_replan_source: str

    model_rounds_used: int

    # 整个用户请求累计使用的真实工具调用总数。
    #
    # 初始Plan和Replan阶段共同累计，
    # Replan后不能重置。
    #
    # 主要用于：
    # - Phoenix观测；
    # - 最终执行统计；
    # - 审计实际工具调用数量。
    tool_calls_used: int

    # 当前Plan阶段已经使用的工具调用数。
    #
    # 初始Plan阶段从0开始。
    #
    # Replan真正生成新步骤后，
    # 重新重置为0。
    #
    # planning.max_plan_tool_calls
    # 限制的是这个字段，而不是上面的总计数。
    phase_tool_calls_used: int
    final_status: str
    final_answer: str
    unmet_success_criteria: list[str]
    overall_stop_reason: str

def _read_progress_callback(
    config: RunnableConfig,
) -> ProgressCallback:
    """读取本次Planning Graph运行的进度回调。

    progress_callback是新架构的必需运行时依赖。

    它不属于PlanningState，也不允许通过旧接口省略。
    """

    configurable = config.get(
        "configurable"
    )

    if not isinstance(
        configurable,
        dict,
    ):
        raise RuntimeError(
            "Planning Graph缺少"
            "configurable运行配置。"
        )

    progress_callback = (
        configurable.get(
            "progress_callback"
        )
    )

    if not callable(
        progress_callback
    ):
        raise RuntimeError(
            "Planning Graph缺少合法的"
            "progress_callback。"
        )

    return cast(
        ProgressCallback,
        progress_callback,
    )


async def _emit_progress(
    config: RunnableConfig,
    event: ProgressEvent,
) -> None:
    """向当前请求的飞书进度通道发送事件。

    这里只负责把结构化事件交给回调。

    文字安全清理、事件去重和飞书发送失败隔离，
    由main.py建立的progress_callback负责。
    """

    progress_callback = (
        _read_progress_callback(
            config
        )
    )

    await progress_callback(
        event
    )


def _build_plan_created_message(
    steps: list[
        PlanStep
    ],
) -> str:
    """生成PLAN_CREATED事件的高层摘要。"""

    lines = [
        (
            "已生成执行计划，"
            f"共{len(steps)}步。"
        )
    ]

    for step in steps:
        objective = (
            sanitize_progress_text(
                step.objective,
                max_chars=80,
            )
        )

        if not objective:
            objective = (
                "执行当前计划步骤"
            )

        lines.append(
            f"{step.step_id}. {objective}"
        )

    return "\n".join(
        lines
    )
def _context(state: PlanningState) -> PlanningContextPack:
    return PlanningContextPack.model_validate(state["context"])


def _json_default(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    return model_dump(mode="json") if callable(model_dump) else str(value)


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        default=_json_default,
    )


def _normalize_steps(
    steps: list[PlanStep],
    *,
    start_step_id: int,
    limit: int,
) -> list[PlanStep]:
    if limit <= 0:
        return []

    return [
        step.model_copy(update={"step_id": start_step_id + offset})
        for offset, step in enumerate(steps[:limit])
    ]


def _replace_step_report(
    reports: list[StepReport],
    new_report: StepReport,
) -> list[StepReport]:
    updated = [
        report
        for report in reports
        if report.step_id != new_report.step_id
    ]
    updated.append(new_report)
    updated.sort(key=lambda report: report.step_id)
    return updated


def _report_for_step(
    reports: list[StepReport],
    step_id: int,
) -> StepReport | None:
    return next(
        (
            report
            for report in reversed(reports)
            if report.step_id == step_id
        ),
        None,
    )


def _next_step_id(state: PlanningState) -> int:
    reports = state.get("completed_step_reports", [])
    return max((report.step_id for report in reports), default=0) + 1


def _remaining_step_capacity(
    state: PlanningState,
    planning: PlanningSettings,
) -> int:
    return max(
        0,
        planning.max_total_steps
        - len(state.get("completed_step_reports", [])),
    )


def _plan_model_rounds_remaining(
    state: PlanningState,
    planning: PlanningSettings,
) -> int:
    return max(
        0,
        planning.max_plan_model_rounds
        - state.get("model_rounds_used", 0),
    )


def _plan_tool_calls_remaining(
    state: PlanningState,
    planning: PlanningSettings,
) -> int:
    """返回当前Plan阶段剩余的工具调用额度。

    tool_calls_used：
        保存整个用户请求的真实工具调用总数，
        Replan后不会重置。

    phase_tool_calls_used：
        只保存当前初始Plan或Replan阶段
        已经使用的工具调用数。

    planning.max_plan_tool_calls：
        限制的是当前Plan阶段，
        而不是整个用户请求的累计总量。
    """

    return max(
        0,

        (
            planning.max_plan_tool_calls

            - state.get(
                "phase_tool_calls_used",
                0,
            )
        ),
    )
def _remaining_budget(
    state: PlanningState,
    planning: PlanningSettings,
) -> dict[str, int]:
    """返回当前执行状态中的剩余预算。"""

    return {
        # 模型预算仍然属于整个用户请求，
        # Replan不会重置。
        "plan_model_rounds": (
            _plan_model_rounds_remaining(
                state,
                planning,
            )
        ),

        # 当前Plan阶段还剩多少工具额度。
        "plan_tool_calls": (
            _plan_tool_calls_remaining(
                state,
                planning,
            )
        ),

        # 当前Plan阶段已经使用的工具数。
        "phase_tool_calls_used": (
            state.get(
                "phase_tool_calls_used",
                0,
            )
        ),

        # 整轮用户请求累计使用的真实工具数。
        "tool_calls_used_total": (
            state.get(
                "tool_calls_used",
                0,
            )
        ),

        "step_model_rounds": max(
            0,

            (
                planning.max_step_model_rounds

                - state.get(
                    "current_step_model_rounds",
                    0,
                )
            ),
        ),

        "step_executor_rounds": max(
            0,

            (
                planning.max_step_executor_rounds

                - state.get(
                    "current_step_executor_rounds",
                    0,
                )
            ),
        ),

        "step_report_rounds": max(
            0,

            (
                planning.max_step_report_rounds

                - state.get(
                    "current_step_report_rounds",
                    0,
                )
            ),
        ),

        "step_tool_calls": max(
            0,

            (
                planning.max_step_tool_calls

                - state.get(
                    "current_step_tool_calls",
                    0,
                )
            ),
        ),

        "step_request_toolset_calls": max(
            0,

            (
                1

                - state.get(
                    "current_step_request_toolset_calls",
                    0,
                )
            ),
        ),

        "step_attempts": max(
            0,

            (
                planning.max_step_attempts

                - state.get(
                    "current_step_attempt",
                    0,
                )
            ),
        ),

        "replans": max(
            0,

            (
                planning.max_replans

                - state.get(
                    "replans_used",
                    0,
                )
            ),
        ),

        "total_step_capacity": (
            _remaining_step_capacity(
                state,
                planning,
            )
        ),
    }

def _execution_budget_exhausted(
    state: PlanningState,
    planning: PlanningSettings,
) -> bool:
    """判断执行链是否已经无法继续调用模型。

    工具预算为0时，Executor仍然可以进行无工具分析、
    整理已有结果和生成文本；只有模型预算为0时，
    才必须停止进入新的Executor模型调用。
    """

    return (
        _plan_model_rounds_remaining(
            state,
            planning,
        )
        <= 0
    )

def _replan_available(
    state: PlanningState,
    planning: PlanningSettings,
    *,
    from_reviewer: bool,
) -> bool:
    """判断当前是否仍然允许执行唯一一次Hard Replan。

    当前Plan阶段的工具预算即使已经耗尽，
    也不能阻止Replan。

    原因是Replan真正生成新步骤后，
    新Plan阶段会重新获得完整的工具预算。

    模型总预算不会重置，因此仍然必须为：
    - Hard Replanner；
    - 至少一次Simple Executor；
    - Replan后的Final Reviewer；
    预留足够的模型轮次。
    """

    if (
        state.get(
            "replans_used",
            0,
        )
        >= planning.max_replans
    ):
        return False

    if (
        _remaining_step_capacity(
            state,
            planning,
        )
        <= 0
    ):
        return False

    # 不再检查：
    #
    # _plan_tool_calls_remaining(...) <= 0
    #
    # 旧Plan阶段工具耗尽，
    # 不代表新Plan阶段没有工具预算。

    if from_reviewer:
        # 当前Final Reviewer尚未执行。
        #
        # 需要预留：
        # 1. 当前Final Reviewer；
        # 2. Hard Replanner；
        # 3. 至少一次Simple Executor；
        # 4. Replan后的Final Reviewer。
        required_rounds = (
            HARD_CALL_MAX_ROUNDS * 3
            + 1
        )

    else:
        # 当前Step Reporter或第一次Reviewer
        # 已经完成模型调用。
        #
        # 需要预留：
        # 1. Hard Replanner；
        # 2. 至少一次Simple Executor；
        # 3. Replan后的Final Reviewer。
        required_rounds = (
            HARD_CALL_MAX_ROUNDS * 2
            + 1
        )

    return (
        _plan_model_rounds_remaining(
            state,
            planning,
        )
        >= required_rounds
    )

def _executor_limits(
    state: PlanningState,
    planning: PlanningSettings,
) -> tuple[int, int]:
    model_limit = min(
        planning.max_step_executor_rounds
        - state.get("current_step_executor_rounds", 0),
        planning.max_step_model_rounds
        - state.get("current_step_model_rounds", 0),
        _plan_model_rounds_remaining(state, planning),
    )
    tool_limit = min(
        planning.max_step_tool_calls
        - state.get("current_step_tool_calls", 0),
        _plan_tool_calls_remaining(state, planning),
    )
    return max(0, model_limit), max(0, tool_limit)


def _reporter_limit(
    state: PlanningState,
    planning: PlanningSettings,
) -> int:
    return max(
        0,
        min(
            planning.max_step_report_rounds
            - state.get("current_step_report_rounds", 0),
            planning.max_step_model_rounds
            - state.get("current_step_model_rounds", 0),
            _plan_model_rounds_remaining(state, planning),
        ),
    )


def _build_step_instruction(
    state: PlanningState,
    current_step: PlanStep,
    attempt: int,
    *,
    model_limit: int,
    tool_limit: int,
    planning: PlanningSettings,
) -> str:
    reports = [
        report.model_dump(
            mode="json"
        )

        for report in state.get(
            "completed_step_reports",
            [],
        )
    ]

    if tool_limit > 0:
        tool_budget_instruction = (
            "本次Attempt可以按需调用工具，"
            f"但最多只能调用{tool_limit}次。"
        )

    else:
        tool_budget_instruction = (
            "本次Attempt的工具额度为0。"
            "不要尝试调用任何工具，也不要调用request_toolset；"
            "请只依据用户请求、此前StepReport、长期记忆和"
            "已经确认的执行结果完成分析、整理或收口。"
            "无法确认的内容必须明确说明，不得编造。"
        )

    return (
        "你正在执行多步骤任务中的一个高层Step。\n"
        "只执行当前Step；不要修改整体Plan，不要生成StepReport，"
        "也不要直接回答整个用户请求。\n"
        "必须依据真实且已经获得的结果工作；"
        "满足成功标准后立即停止。\n\n"

        f"用户原始请求：\n"
        f"{_context(state).user_request}\n\n"

        f"整体目标：\n"
        f"{state['plan_objective']}\n\n"

        f"当前Step（第{attempt}次尝试）：\n"
        f"{_json_text(current_step.model_dump(mode='json'))}\n\n"

        f"此前StepReport：\n"
        f"{_json_text(reports)}\n\n"

        "本次Attempt预算：\n"
        f"- 最多模型轮次：{model_limit}\n"
        f"- 最多工具调用：{tool_limit}\n"
        f"- 执行要求：{tool_budget_instruction}\n\n"

        f"整轮剩余预算：\n"
        f"{_json_text(_remaining_budget(state, planning))}"
    )


def _build_forced_step_report(
    state: PlanningState,
    *,
    reason: str,
) -> StepReport:
    current_step = state.get("current_step")
    if current_step is None:
        raise RuntimeError("当前没有可以生成报告的Step。")

    previous = _report_for_step(
        state.get("completed_step_reports", []),
        current_step.step_id,
    )
    confirmed = list(previous.confirmed_results) if previous else []
    evidence = list(previous.evidence) if previous else []
    errors = list(previous.errors) if previous else []
    errors.append(reason)
    unresolved = (
        list(previous.unresolved_items)
        if previous
        else list(current_step.success_criteria)
    )

    return StepReport(
        step_id=current_step.step_id,
        status="PARTIAL" if confirmed or evidence else "FAILED",
        summary=(
            "当前Step已经结束，但没有足够模型预算"
            "生成新的结构化StepReport。"
        ),
        stop_reason=reason,
        criterion_results=[
            StepCriterionResult(
                criterion=criterion,
                status="UNKNOWN",
                evidence=[],
            )
            for criterion in current_step.success_criteria
        ],
        confirmed_results=confirmed,
        evidence=evidence,
        errors=errors,
        unresolved_items=unresolved,
        next_action=None,
        request_replan=False,
        replan_reason=None,
    )


def _build_forced_final_decision(
    state: PlanningState,
) -> FinalReviewDecision:
    reports = state.get("completed_step_reports", [])
    confirmed: list[str] = []
    unresolved: list[str] = []
    errors: list[str] = []

    for report in reports:
        confirmed.extend(report.confirmed_results)
        unresolved.extend(report.unresolved_items)
        errors.extend(report.errors)

    all_completed = bool(reports) and all(
        report.status == "COMPLETED" for report in reports
    )

    if all_completed and not unresolved:
        status = "COMPLETED"
        unmet: list[str] = []
    elif confirmed:
        status = "PARTIAL"
        unmet = unresolved[:6] or state.get(
            "plan_success_criteria", []
        )[:6]
    else:
        status = "FAILED"
        unmet = state.get("plan_success_criteria", [])[:6] or [
            "没有获得足够可靠的执行结果。"
        ]

    lines = ["本轮任务已结束。"]
    if confirmed:
        lines += ["", "已确认结果:", *[f"- {item}" for item in confirmed[:10]]]
    if unresolved:
        lines += ["", "尚未解决:", *[f"- {item}" for item in unresolved[:6]]]
    if errors:
        lines += ["", "执行中遇到的问题:", *[f"- {item}" for item in errors[:4]]]
    if not confirmed:
        lines += ["", "当前没有足够可靠的信息支持进一步结论。"]

    return FinalReviewDecision(
        action="FINAL",
        status=status,
        final_answer="\n".join(lines),
        unmet_success_criteria=unmet,
        replan_reason=None,
    )


def build_planning_graph(
    *,
    simple_model,
    hard_model,
    step_agent,
    planning: PlanningSettings,
    model_output_max_tokens: int,
):
    """创建Hard规划、Simple执行和最终审核图。"""

    async def supervisor_node(
        state: PlanningState,
        config: RunnableConfig,
    ) -> Command[
        Literal[
            "step_executor",
            "final_reviewer",
            "__end__",
        ]
    ]:
        if _plan_model_rounds_remaining(state, planning) < HARD_CALL_MAX_ROUNDS:
            decision = _build_forced_final_decision(state)
            return Command(
                update={
                    "final_status": decision.status or "FAILED",
                    "final_answer": decision.final_answer or "",
                    "unmet_success_criteria": decision.unmet_success_criteria,
                    "overall_stop_reason": "supervisor_budget_unavailable",
                },
                goto=END,
            )

        result = await run_hard_supervisor(
            hard_model,
            context=_context(state),
            max_steps_per_plan=planning.max_steps_per_plan,
        )
        decision = result.output
        model_rounds_used = (
            state.get("model_rounds_used", 0)
            + result.model_rounds_used
        )

        if decision.action == "FINAL":
            return Command(
                update={
                    "model_rounds_used": model_rounds_used,
                    "final_status": "COMPLETED",
                    "final_answer": decision.final_answer or "",
                    "unmet_success_criteria": [],
                    "overall_stop_reason": "supervisor_final",
                },
                goto=END,
            )

        steps = _normalize_steps(
            decision.steps,
            start_step_id=1,
            limit=min(
                planning.max_steps_per_plan,
                planning.max_total_steps,
            ),
        )

        if steps:
            await _emit_progress(
                config,

                ProgressEvent(
                    stage="PLAN_CREATED",

                    message=(
                        _build_plan_created_message(
                            steps
                        )
                    ),

                    total_steps=(
                        len(
                            steps
                        )
                    ),
                ),
            )

        update: dict[
            str,
            Any,
        ] = {
            "plan_objective": decision.plan_objective or _context(state).user_request,
            "plan_success_criteria": list(decision.plan_success_criteria),
            "remaining_steps": steps,
            "completed_step_reports": [],
            "current_step": None,
            "current_step_attempt": 0,
            "current_step_executor_rounds": 0,
            "current_step_report_rounds": 0,
            "current_step_model_rounds": 0,
            "current_step_tool_calls": 0,
            "current_step_request_toolset_calls": 0,
            "current_step_trace": {},
            "current_step_stop_reason": "",
            "retry_current_step": False,
            "replans_used": 0,
            "replan_history": [],
            "pending_replan_reason": "",
            "pending_replan_source": "",
            "model_rounds_used": (
                model_rounds_used
            ),

            # 整个用户请求的真实工具调用总数。
            # 新Planning运行正常情况下从0开始。
            "tool_calls_used": (
                state.get(
                    "tool_calls_used",
                    0,
                )
            ),

            # 初始Plan阶段获得一套新的工具预算。
            "phase_tool_calls_used": 0,
        }

        if not steps:
            update["overall_stop_reason"] = "supervisor_returned_no_steps"
            return Command(update=update, goto="final_reviewer")

        if _execution_budget_exhausted(update, planning):
            update["overall_stop_reason"] = "budget_exhausted_after_supervisor"
            return Command(update=update, goto="final_reviewer")

        return Command(update=update, goto="step_executor")

    async def step_executor_node(
        state: PlanningState,
        config: RunnableConfig,
    ) -> Command[
        Literal[
            "step_reporter",
            "final_reviewer",
        ]
    ]:
        """执行当前Step的一次Attempt。

        普通模型、工具和request_toolset额度
        都由Planning Graph计算后写入Agent State。

        request_toolset的额度属于整个Step，
        不会因为创建新的Attempt Thread而重置。
        """

        if _execution_budget_exhausted(
            state,
            planning,
        ):
            return Command(
                update={
                    "overall_stop_reason": (
                        "execution_budget_exhausted"
                    )
                },

                goto="final_reviewer",
            )

        retry = state.get(
            "retry_current_step",
            False,
        )

        remaining_steps = list(
            state.get(
                "remaining_steps",
                [],
            )
        )

        if retry:
            current_step = state.get(
                "current_step"
            )

            if current_step is None:
                return Command(
                    update={
                        "overall_stop_reason": (
                            "retry_without_current_step"
                        )
                    },

                    goto="final_reviewer",
                )

            attempt = (
                state.get(
                    "current_step_attempt",
                    1,
                )
                + 1
            )

            executor_rounds = state.get(
                "current_step_executor_rounds",
                0,
            )

            report_rounds = state.get(
                "current_step_report_rounds",
                0,
            )

            step_model_rounds = state.get(
                "current_step_model_rounds",
                0,
            )

            step_tool_calls = state.get(
                "current_step_tool_calls",
                0,
            )

            step_request_toolset_calls = (
                state.get(
                    "current_step_request_toolset_calls",
                    0,
                )
            )

        else:
            if not remaining_steps:
                return Command(
                    update={
                        "overall_stop_reason": (
                            "no_remaining_steps"
                        )
                    },

                    goto="final_reviewer",
                )

            current_step = (
                remaining_steps.pop(
                    0
                )
            )

            attempt = 1

            executor_rounds = 0
            report_rounds = 0
            step_model_rounds = 0
            step_tool_calls = 0

            # 新Step重新获得一次工具组刷新机会。
            step_request_toolset_calls = 0

        provisional: PlanningState = dict(
            state
        )

        provisional.update(
            {
                "current_step": (
                    current_step
                ),

                "current_step_attempt": (
                    attempt
                ),

                "current_step_executor_rounds": (
                    executor_rounds
                ),

                "current_step_report_rounds": (
                    report_rounds
                ),

                "current_step_model_rounds": (
                    step_model_rounds
                ),

                "current_step_tool_calls": (
                    step_tool_calls
                ),

                "current_step_request_toolset_calls": (
                    step_request_toolset_calls
                ),

                "remaining_steps": (
                    remaining_steps
                ),
            }
        )

        (
            model_limit,
            tool_limit,
        ) = _executor_limits(
            provisional,
            planning,
        )

        if model_limit <= 0:
            reason = (
                "当前Step已经没有可用的"
                "Executor模型预算。"
            )

            return Command(
                update={
                    "current_step": (
                        current_step
                    ),

                    "current_step_attempt": (
                        attempt
                    ),

                    "current_step_executor_rounds": (
                        executor_rounds
                    ),

                    "current_step_report_rounds": (
                        report_rounds
                    ),

                    "current_step_model_rounds": (
                        step_model_rounds
                    ),

                    "current_step_tool_calls": (
                        step_tool_calls
                    ),

                    "current_step_request_toolset_calls": (
                        step_request_toolset_calls
                    ),

                    "current_step_trace": {
                        "status": (
                            "not_started"
                        ),

                        "reason": (
                            reason
                        ),
                    },

                    "current_step_stop_reason": (
                        reason
                    ),

                    "remaining_steps": (
                        remaining_steps
                    ),

                    "retry_current_step": False,
                },

                goto="step_reporter",
            )
        visible_step_ids = {
            current_step.step_id,
        }

        visible_step_ids.update(
            report.step_id

            for report
            in state.get(
                "completed_step_reports",
                [],
            )
        )

        visible_step_ids.update(
            step.step_id

            for step
            in remaining_steps
        )

        total_steps = len(
            visible_step_ids
        )

        step_objective = (
            sanitize_progress_text(
                current_step.objective,
                max_chars=80,
            )
            or "执行当前计划步骤"
        )

        if attempt == 1:
            progress_message = (
                f"正在执行"
                f"{current_step.step_id}/"
                f"{total_steps}："
                f"{step_objective}。"
            )

        else:
            progress_message = (
                f"第{current_step.step_id}步"
                f"正在进行第{attempt}次尝试："
                f"{step_objective}。"
            )

        await _emit_progress(
            config,

            ProgressEvent(
                stage="STEP_STARTED",

                message=(
                    progress_message
                ),

                step_id=(
                    current_step.step_id
                ),

                total_steps=(
                    total_steps
                ),

                status=(
                    f"ATTEMPT_{attempt}"
                ),
            ),
        )

        # request_toolset属于Step级额度。

        #
        # 第一次Attempt尚未使用时为1；
        # 任意Attempt真正调用过一次后，
        # 后续Attempt全部为0。
        request_toolset_limit = (
            0

            if tool_limit <= 0

            else max(
                0,

                1
                - step_request_toolset_calls,
            )
        )

        thread_id = (
            f"{state.get('conversation_thread_id', 'conversation')}:"
            f"{state.get('planning_run_id', 'planning')}:"
            f"step_{current_step.step_id}:"
            f"attempt_{attempt}"
        )

        agent_state = {
            "conversation_id": (
                thread_id
            ),

            "conversation_title": (
                "Planning Step "
                f"{current_step.step_id}"
            ),

            "memory_context": (
                _context(
                    state
                ).memory_context
            ),

            "executor_model_run_limit": (
                model_limit
            ),

            "executor_tool_run_limit": (
                tool_limit
            ),

            "request_toolset_run_limit": (
                request_toolset_limit
            ),
        }

        instruction = (
            _build_step_instruction(
                provisional,
                current_step,
                attempt,

                model_limit=(
                    model_limit
                ),

                tool_limit=(
                    tool_limit
                ),

                planning=planning,
            )
        )

        stop_reason = (
            "Simple Executor正常结束。"
        )

        try:
            details = await (
                ask_agent(
                    step_agent,
                    instruction,

                    thread_id=(
                        thread_id
                    ),

                    state_update=(
                        agent_state
                    ),

                    return_details=True,
                )
            )

            if not isinstance(
                details,
                dict,
            ):
                raise RuntimeError(
                    "Step Agent没有返回"
                    "详细执行结果。"
                )

            summary = details.get(
                "execution_summary",
                {},
            )

            attempt_model_rounds = max(
                0,

                int(
                    summary.get(
                        "model_call_count",
                        0,
                    )
                    or 0
                ),
            )

            attempt_tool_calls = max(
                0,

                int(
                    summary.get(
                        "tool_call_count",
                        0,
                    )
                    or 0
                ),
            )

            attempt_request_toolset_calls = max(
                0,

                int(
                    summary.get(
                        "request_toolset_call_count",
                        0,
                    )
                    or 0
                ),
            )

            trace = {
                "attempt": (
                    attempt
                ),

                "final_answer": (
                    details.get(
                        "final_answer",
                        "",
                    )
                ),

                "messages": (
                    details.get(
                        "current_turn_messages",
                        [],
                    )
                ),

                "execution_summary": (
                    summary
                ),

                "applied_limits": {
                    "model_rounds": (
                        model_limit
                    ),

                    "tool_calls": (
                        tool_limit
                    ),

                    "request_toolset_calls": (
                        request_toolset_limit
                    ),
                },
            }

        except Exception as error:
            attempt_model_rounds = 0
            attempt_tool_calls = 0
            attempt_request_toolset_calls = 0

            stop_reason = (
                "Simple Executor发生异常。"
            )

            trace = {
                "attempt": (
                    attempt
                ),

                "error": (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),

                "applied_limits": {
                    "model_rounds": (
                        model_limit
                    ),

                    "tool_calls": (
                        tool_limit
                    ),

                    "request_toolset_calls": (
                        request_toolset_limit
                    ),
                },
            }

        executor_rounds += (
            attempt_model_rounds
        )

        step_model_rounds += (
            attempt_model_rounds
        )

        step_tool_calls += (
            attempt_tool_calls
        )

        # 理论上Middleware已经保证不会超过1。
        # 这里再夹到1，维护Step State的不变量。
        step_request_toolset_calls = min(
            1,

            (
                step_request_toolset_calls
                + attempt_request_toolset_calls
            ),
        )

        model_rounds_used = (
            state.get(
                "model_rounds_used",
                0,
            )
            + attempt_model_rounds
        )

        # 整个用户请求累计使用的工具数。
        #
        # Replan后不会重置。
        tool_calls_used = (
            state.get(
                "tool_calls_used",
                0,
            )
            + attempt_tool_calls
        )

        # 当前Plan阶段累计使用的工具数。
        #
        # Replan真正开始新阶段后会重置。
        phase_tool_calls_used = (
            state.get(
                "phase_tool_calls_used",
                0,
            )
            + attempt_tool_calls
        )

        if (
            executor_rounds
            >= planning
            .max_step_executor_rounds
        ):
            stop_reason = (
                "当前Step已达到"
                "Executor模型轮次上限。"
            )

        if (
            step_model_rounds
            >= planning
            .max_step_model_rounds
        ):
            stop_reason = (
                "当前Step已达到"
                "总模型轮次上限。"
            )

        if (
            attempt_tool_calls > 0
            and step_tool_calls
            >= planning
            .max_step_tool_calls
        ):
            stop_reason = (
                "当前Step已达到工具调用上限；"
                "后续只能进行无工具分析。"
            )

        if (
            model_rounds_used
            >= planning
            .max_plan_model_rounds
        ):
            stop_reason = (
                "整体Plan模型预算已经耗尽。"
            )

        elif (
            attempt_tool_calls > 0
            and phase_tool_calls_used
            >= planning.max_plan_tool_calls
        ):
            stop_reason = (
                "当前Plan阶段的工具预算已经耗尽；"
                "后续只能进行无工具分析，"
                "或者在满足条件时进入Replan。"
            )

        return Command(
            update={
                "current_step": (
                    current_step
                ),

                "current_step_attempt": (
                    attempt
                ),

                "current_step_executor_rounds": (
                    executor_rounds
                ),

                "current_step_report_rounds": (
                    report_rounds
                ),

                "current_step_model_rounds": (
                    step_model_rounds
                ),

                "current_step_tool_calls": (
                    step_tool_calls
                ),

                "current_step_request_toolset_calls": (
                    step_request_toolset_calls
                ),

                "current_step_trace": (
                    trace
                ),

                "current_step_stop_reason": (
                    stop_reason
                ),

                "remaining_steps": (
                    remaining_steps
                ),

                "model_rounds_used": (
                    model_rounds_used
                ),

                # 整个用户请求的工具调用总数。
                "tool_calls_used": (
                    tool_calls_used
                ),

                # 当前Plan阶段的工具调用数。
                "phase_tool_calls_used": (
                    phase_tool_calls_used
                ),

                "retry_current_step": False,
            },

            goto="step_reporter",
        )

    async def step_reporter_node(
        state: PlanningState,
        config: RunnableConfig,
    ) -> Command[
        Literal[
            "step_executor",
            "replanner",
            "final_reviewer",
        ]
    ]:
        """生成StepReport并发送当前Attempt的结束状态。"""

        current_step = state.get(
            "current_step"
        )

        if current_step is None:
            return Command(
                update={
                    "overall_stop_reason": (
                        "reporter_without_current_step"
                    )
                },

                goto="final_reviewer",
            )

        limit = _reporter_limit(
            state,
            planning,
        )

        previous = _report_for_step(
            state.get(
                "completed_step_reports",
                [],
            ),

            current_step.step_id,
        )

        if limit <= 0:
            report = (
                _build_forced_step_report(
                    state,

                    reason=(
                        "剩余模型预算不足以调用"
                        "Step Reporter。"
                    ),
                )
            )

            reporter_rounds = 0

        else:
            result = await (
                run_step_reporter(
                    simple_model,

                    user_request=(
                        _context(
                            state
                        ).user_request
                    ),

                    plan_objective=(
                        state[
                            "plan_objective"
                        ]
                    ),

                    current_step=(
                        current_step
                    ),

                    step_execution_trace={
                        "current_attempt": (
                            state.get(
                                "current_step_trace",
                                {},
                            )
                        ),

                        "previous_step_report": (
                            previous.model_dump(
                                mode="json"
                            )

                            if previous

                            else None
                        ),
                    },

                    stop_reason=(
                        state.get(
                            "current_step_stop_reason",

                            (
                                "Simple Executor"
                                "正常结束。"
                            ),
                        )
                    ),

                    remaining_budget=(
                        _remaining_budget(
                            state,
                            planning,
                        )
                    ),

                    max_model_rounds=(
                        limit
                    ),
                    model_output_max_tokens=(
                        model_output_max_tokens
                    ),
                )
            )

            report = result.report

            reporter_rounds = (
                result.model_rounds_used
            )

        reports = (
            _replace_step_report(
                state.get(
                    "completed_step_reports",
                    [],
                ),

                report,
            )
        )

        update: dict[
            str,
            Any,
        ] = {
            "completed_step_reports": (
                reports
            ),

            "model_rounds_used": (
                state.get(
                    "model_rounds_used",
                    0,
                )
                + reporter_rounds
            ),

            "current_step_report_rounds": (
                state.get(
                    "current_step_report_rounds",
                    0,
                )
                + reporter_rounds
            ),

            "current_step_model_rounds": (
                state.get(
                    "current_step_model_rounds",
                    0,
                )
                + reporter_rounds
            ),

            "retry_current_step": False,
        }

        current: PlanningState = dict(
            state
        )

        current.update(
            update
        )

        budget_exhausted = (
            _execution_budget_exhausted(
                current,
                planning,
            )
        )

        replan_available = (
            not budget_exhausted

            and report.request_replan

            and _replan_available(
                current,
                planning,

                from_reviewer=False,
            )
        )

        can_retry = (
            not budget_exhausted

            and not report.request_replan

            and report.status
            in {
                "BLOCKED",
                "FAILED",
            }

            and state.get(
                "current_step_attempt",
                1,
            )
            < planning.max_step_attempts

            and current.get(
                "current_step_model_rounds",
                0,
            )
            < planning.max_step_model_rounds

            and state.get(
                "current_step_executor_rounds",
                0,
            )
            < planning.max_step_executor_rounds

            and _plan_model_rounds_remaining(
                current,
                planning,
            )
            > 0
        )

        step_id = (
            current_step.step_id
        )

        attempt = state.get(
            "current_step_attempt",
            1,
        )

        # ProgressEvent.stage表示审核结果。
        #
        # status使用Attempt编号参与去重，
        # 这样同一个Step的第二次受阻或失败
        # 不会被第一次事件错误吞掉。
        if report.status == "COMPLETED":
            progress_event = (
                ProgressEvent(
                    stage=(
                        "STEP_COMPLETED"
                    ),

                    message=(
                        f"第{step_id}步已完成。"
                    ),

                    step_id=(
                        step_id
                    ),

                    status=(
                        f"ATTEMPT_{attempt}"
                    ),
                )
            )

        elif report.status == "PARTIAL":
            progress_event = (
                ProgressEvent(
                    stage=(
                        "STEP_PARTIAL"
                    ),

                    message=(
                        f"第{step_id}步"
                        "已部分完成，"
                        "仍有内容尚未确认。"
                    ),

                    step_id=(
                        step_id
                    ),

                    status=(
                        f"ATTEMPT_{attempt}"
                    ),
                )
            )

        elif report.status == "BLOCKED":
            if replan_available:
                progress_message = (
                    f"第{step_id}步受到阻塞，"
                    "正在判断新的执行路径。"
                )

            elif can_retry:
                progress_message = (
                    f"第{step_id}步受到阻塞，"
                    "正在准备重试。"
                )

            else:
                progress_message = (
                    f"第{step_id}步受到阻塞，"
                    "将根据已有结果继续收口。"
                )

            progress_event = (
                ProgressEvent(
                    stage=(
                        "STEP_BLOCKED"
                    ),

                    message=(
                        progress_message
                    ),

                    step_id=(
                        step_id
                    ),

                    status=(
                        f"ATTEMPT_{attempt}"
                    ),
                )
            )

        else:
            # StepStatus已经被Schema限制为：
            #
            # COMPLETED
            # PARTIAL
            # BLOCKED
            # FAILED
            #
            # 因此前三个分支以外只能是FAILED。
            if replan_available:
                progress_message = (
                    f"第{step_id}步"
                    "未能获得可靠结果，"
                    "正在判断新的执行路径。"
                )

            elif can_retry:
                progress_message = (
                    f"第{step_id}步"
                    "未能获得可靠结果，"
                    "正在准备重试。"
                )

            else:
                progress_message = (
                    f"第{step_id}步"
                    "未能获得可靠结果，"
                    "将根据已有结果继续收口。"
                )

            progress_event = (
                ProgressEvent(
                    stage=(
                        "STEP_FAILED"
                    ),

                    message=(
                        progress_message
                    ),

                    step_id=(
                        step_id
                    ),

                    status=(
                        f"ATTEMPT_{attempt}"
                    ),
                )
            )

        await _emit_progress(
            config,
            progress_event,
        )

        if budget_exhausted:
            update[
                "overall_stop_reason"
            ] = (
                "budget_exhausted_after_report"
            )

            return Command(
                update=update,
                goto="final_reviewer",
            )

        if report.request_replan:
            if replan_available:
                update.update(
                    {
                        "pending_replan_reason": (
                            report.replan_reason
                            or report.summary
                        ),

                        "pending_replan_source": (
                            "step_reporter"
                        ),

                        "overall_stop_reason": (
                            "step_report_requested_replan"
                        ),
                    }
                )

                return Command(
                    update=update,
                    goto="replanner",
                )

            update[
                "overall_stop_reason"
            ] = (
                "requested_replan_unavailable"
            )

            return Command(
                update=update,
                goto="final_reviewer",
            )

        if can_retry:
            update[
                "retry_current_step"
            ] = True

            return Command(
                update=update,
                goto="step_executor",
            )

        if report.status in {
            "BLOCKED",
            "FAILED",
        }:
            update[
                "overall_stop_reason"
            ] = (
                "step_failed_after_attempts"
            )

            return Command(
                update=update,
                goto="final_reviewer",
            )

        if state.get(
            "remaining_steps"
        ):
            return Command(
                update=update,
                goto="step_executor",
            )

        update[
            "overall_stop_reason"
        ] = (
            "all_steps_reported"
        )

        return Command(
            update=update,
            goto="final_reviewer",
        )

    async def replanner_node(
        state: PlanningState,
        config: RunnableConfig,
    ) -> Command[
        Literal[
            "step_executor",
            "final_reviewer",
        ]
    ]:
        """执行全局唯一一次Hard Replan。"""

        if not _replan_available(
            state,
            planning,
            from_reviewer=False,
        ):
            return Command(
                update={
                    "overall_stop_reason": (
                        "replan_unavailable"
                    ),

                    "pending_replan_reason": "",

                    "pending_replan_source": "",
                },

                goto="final_reviewer",
            )

        max_steps = min(
            planning.max_steps_per_plan,

            _remaining_step_capacity(
                state,
                planning,
            ),
        )

        next_step_id = (
            _next_step_id(
                state
            )
        )

        reason = (
            state.get(
                "pending_replan_reason",
                "",
            )
            .strip()
        )

        if not reason:
            reports = state.get(
                "completed_step_reports",
                [],
            )

            reason = (
                reports[-1].summary

                if reports

                else (
                    "需要重新规划"
                    "剩余步骤。"
                )
            )

        await _emit_progress(
            config,

            ProgressEvent(
                stage=(
                    "REPLAN_STARTED"
                ),

                message=(
                    "原执行路径需要调整，"
                    "正在重新规划后续步骤。"
                ),

                status=(
                    "RUNNING"
                ),
            ),
        )
        replan_budget = (
            _remaining_budget(
                state,
                planning,
            )
        )

        # Hard Replanner规划的是下一个Plan阶段。
        #
        # 新阶段会重新获得完整的工具额度，
        # 因此不能把旧阶段剩余的0
        # 当成新阶段预算传给模型。
        replan_budget[
            "plan_tool_calls"
        ] = planning.max_plan_tool_calls

        replan_budget[
            "phase_tool_calls_used"
        ] = 0
        result = await (
            run_hard_replanner(
                hard_model,

                context=(
                    _context(
                        state
                    )
                ),

                plan_objective=(
                    state[
                        "plan_objective"
                    ]
                ),

                plan_success_criteria=(
                    state.get(
                        "plan_success_criteria",
                        [],
                    )
                ),

                completed_step_reports=(
                    state.get(
                        "completed_step_reports",
                        [],
                    )
                ),

                replan_context=(
                    reason
                ),

                remaining_steps=(
                    state.get(
                        "remaining_steps",
                        [],
                    )
                ),

                remaining_budget=(
                    replan_budget
                ),

                next_step_id=(
                    next_step_id
                ),

                max_remaining_steps=(
                    max_steps
                ),
            )
        )

        decision = (
            result.output
        )

        new_steps = (
            _normalize_steps(
                decision.remaining_steps,

                start_step_id=(
                    next_step_id
                ),

                limit=(
                    max_steps
                ),
            )

            if decision.action
            == "CONTINUE"

            else []
        )

        if new_steps:
            progress_lines = [
                (
                    "后续执行计划已调整，"
                    f"共{len(new_steps)}步。"
                )
            ]

            for step in new_steps:
                objective = (
                    sanitize_progress_text(
                        step.objective,
                        max_chars=80,
                    )
                    or "执行后续计划步骤"
                )

                progress_lines.append(
                    (
                        f"{step.step_id}. "
                        f"{objective}"
                    )
                )

            replan_finished_message = (
                "\n".join(
                    progress_lines
                )
            )

            replan_finished_status = (
                "CONTINUE"
            )

        else:
            replan_finished_message = (
                "执行计划调整完成，"
                "不再增加新的步骤，"
                "将根据已有结果进行最终汇总。"
            )

            replan_finished_status = (
                "FINISH"
            )

        await _emit_progress(
            config,

            ProgressEvent(
                stage=(
                    "REPLAN_FINISHED"
                ),

                message=(
                    replan_finished_message
                ),

                total_steps=(
                    len(
                        new_steps
                    )

                    if new_steps

                    else None
                ),

                status=(
                    replan_finished_status
                ),
            ),
        )

        history = list(
            state.get(
                "replan_history",
                [],
            )
        )

        history.append(
            {
                "source": (
                    state.get(
                        "pending_replan_source",
                        "unknown",
                    )
                ),

                "request_reason": (
                    reason
                ),

                "decision": (
                    decision.action
                ),

                "decision_reason": (
                    decision.reason
                ),

                "remaining_steps": [
                    step.model_dump(
                        mode="json"
                    )

                    for step
                    in new_steps
                ],

                "used_fallback": (
                    result.used_fallback
                ),

                "model_rounds_used": (
                    result.model_rounds_used
                ),
            }
        )

        update: dict[
            str,
            Any,
        ] = {
            "remaining_steps": (
                new_steps
            ),

            "current_step": None,

            "current_step_attempt": 0,

            "current_step_executor_rounds": 0,

            "current_step_report_rounds": 0,

            "current_step_model_rounds": 0,

            "current_step_tool_calls": 0,

            "current_step_request_toolset_calls": 0,

            "current_step_trace": {},

            "current_step_stop_reason": "",

            "retry_current_step": False,

            "replans_used": (
                state.get(
                    "replans_used",
                    0,
                )
                + 1
            ),

            "replan_history": (
                history
            ),

            "pending_replan_reason": "",

            "pending_replan_source": "",

            "model_rounds_used": (
                    state.get(
                        "model_rounds_used",
                        0,
                    )
                    + result.model_rounds_used
            ),

            # 只有真正生成新步骤时，
            # 才代表一个新的Plan阶段已经开始。
            #
            # 新阶段重新获得完整工具预算。
            #
            # 如果Replanner决定FINISH，
            # 则不存在新阶段，保留原计数。
            "phase_tool_calls_used": (
                0

                if new_steps

                else state.get(
                    "phase_tool_calls_used",
                    0,
                )
            ),
        }

        if (
            decision.action
            == "FINISH"

            or not new_steps
        ):
            update[
                "overall_stop_reason"
            ] = (
                "replanner_finish: "
                f"{decision.reason}"
            )

            return Command(
                update=update,
                goto="final_reviewer",
            )

        current: PlanningState = dict(
            state
        )

        current.update(
            update
        )

        if _execution_budget_exhausted(
            current,
            planning,
        ):
            update[
                "overall_stop_reason"
            ] = (
                "budget_exhausted_after_replan"
            )

            return Command(
                update=update,
                goto="final_reviewer",
            )

        return Command(
            update=update,
            goto="step_executor",
        )

    async def final_reviewer_node(
        state: PlanningState,
        config: RunnableConfig,
    ) -> Command[
        Literal[
            "replanner",
            "__end__",
        ]
    ]:
        """执行Hard Final Reviewer并发送最终汇总进度。"""

        if (
            _plan_model_rounds_remaining(
                state,
                planning,
            )
            < HARD_CALL_MAX_ROUNDS
        ):
            decision = (
                _build_forced_final_decision(
                    state
                )
            )

            return Command(
                update={
                    "final_status": (
                        decision.status
                        or "FAILED"
                    ),

                    "final_answer": (
                        decision.final_answer
                        or ""
                    ),

                    "unmet_success_criteria": (
                        decision
                        .unmet_success_criteria
                    ),

                    "overall_stop_reason": (
                        state.get(
                            "overall_stop_reason",
                            "reviewer_budget_unavailable",
                        )
                    ),
                },

                goto=END,
            )

        can_replan = (
            _replan_available(
                state,
                planning,

                from_reviewer=True,
            )
        )

        reviewed_after_replan = (
            state.get(
                "replans_used",
                0,
            )
            > 0
        )

        if reviewed_after_replan:
            progress_message = (
                "调整后的步骤已执行完成，"
                "正在重新汇总最终结果。"
            )

            progress_status = (
                "AFTER_REPLAN"
            )

        else:
            progress_message = (
                "正在汇总已确认结果并生成最终回答。"
            )

            progress_status = (
                "INITIAL"
            )

        await _emit_progress(
            config,

            ProgressEvent(
                stage=(
                    "FINAL_REVIEW_STARTED"
                ),

                message=(
                    progress_message
                ),

                status=(
                    progress_status
                ),
            ),
        )

        result = await (
            run_hard_final_reviewer(
                hard_model,

                context=(
                    _context(
                        state
                    )
                ),

                plan_objective=(
                    state.get(
                        "plan_objective",

                        _context(
                            state
                        ).user_request,
                    )
                ),

                plan_success_criteria=(
                    state.get(
                        "plan_success_criteria",
                        [],
                    )
                ),

                step_reports=(
                    state.get(
                        "completed_step_reports",
                        [],
                    )
                ),

                replan_history=(
                    state.get(
                        "replan_history",
                        [],
                    )
                ),

                overall_stop_reason=(
                    state.get(
                        "overall_stop_reason",
                        "正常进入最终审核。",
                    )
                ),

                replan_available=(
                    can_replan
                ),
            )
        )

        model_rounds_used = (
            state.get(
                "model_rounds_used",
                0,
            )
            + result.model_rounds_used
        )

        decision = (
            result.output
        )
        if decision.action == "REPLAN":
            current: PlanningState = dict(
                state
            )

            current[
                "model_rounds_used"
            ] = model_rounds_used

            if (
                    can_replan

                    and _replan_available(
                current,
                planning,

                from_reviewer=False,
            )
            ):
                return Command(
                    update={
                        "model_rounds_used": (
                            model_rounds_used
                        ),

                        "pending_replan_reason": (
                                decision.replan_reason
                                or ""
                        ),

                        "pending_replan_source": (
                            "final_reviewer"
                        ),

                        "overall_stop_reason": (
                            "reviewer_requested_replan"
                        ),
                    },

                    goto="replanner",
                )

            decision = (
                _build_forced_final_decision(
                    current
                )
            )

        return Command(
            update={
                "model_rounds_used": (
                    model_rounds_used
                ),

                "final_status": (
                        decision.status
                        or "FAILED"
                ),

                "final_answer": (
                        decision.final_answer
                        or ""
                ),

                "unmet_success_criteria": (
                    decision
                    .unmet_success_criteria
                ),

                "overall_stop_reason": (
                    state.get(
                        "overall_stop_reason",
                        "final_reviewer_completed",
                    )
                ),
            },

            goto=END,
        )

    workflow = StateGraph(
        PlanningState
    )

    workflow.add_node(
        "supervisor",
        supervisor_node,
    )

    workflow.add_node(
        "step_executor",
        step_executor_node,
    )

    workflow.add_node(
        "step_reporter",
        step_reporter_node,
    )

    workflow.add_node(
        "replanner",
        replanner_node,
    )

    workflow.add_node(
        "final_reviewer",
        final_reviewer_node,
    )

    workflow.add_edge(
        START,
        "supervisor",
    )

    recursion_limit = (
            8
            + planning.max_total_steps
            * planning.max_step_attempts
            * 2
    )

    return (
        workflow
        .compile()
        .with_config(
            {
                "recursion_limit": (
                    recursion_limit
                )
            }
        )
    )