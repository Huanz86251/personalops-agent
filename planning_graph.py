from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Literal,
    TypedDict,
    cast,
)

from langchain_core.runnables import (
    RunnableConfig,
)
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agent import ask_worker
from artifact_publisher import (
    ArtifactPublicationReceipt,
    publish_artifact_to_handoff,
)
from integration_repository import IntegrationCommitReceipt
from config import PlanningSettings
from hard_planning import (
    MAX_VALIDATION_RETRIES,
    run_hard_code_scheduler,
    run_hard_final_reviewer,
    run_hard_replanner,
    run_hard_supervisor,
)
from planning_models import (
    FinalReviewDecision,
    PlanStep,
    PlanningContextPack,
    StepArtifactReport,
    StepCriterionResult,
    StepReport,
    WorkerContribution,
)
from scope_resolution import prepare_scope_context
from reporting import (
    StepReviewPacket,
    build_step_review_packet,
    materialize_worker_review_trace,
)
from run_workspace import RUN_WORKSPACE_ROOT, initialize_run_workspace
from step_execution import run_step_reporter
from workers.coordinator import WorkerGroupCoordinator
from workers.group import (
    ACTIVE_ATTEMPT_STATUSES,
    StepWorkerGroup,
    WorkerAttempt,
    WorkerAttemptStatus,
)
from workers.registry import WorkerAgentRegistry
from workers.code_review_models import (
    CodeReviewLoopState,
    CodeReviewReport,
    SchedulerCodeDecision,
)
from progress_events import (
    ProgressCallback,
    ProgressEvent,
    sanitize_progress_text,
)
from skill_runtime import load_catalog, prepare_skills
from scheduler_runtime import scheduler_node

HARD_CALL_MAX_ROUNDS = MAX_VALIDATION_RETRIES + 1


def _record_skill_traces(context, traces):
    snapshots = dict(context.role_skill_snapshots)
    for trace in traces:
        snapshot = trace.get("role_skill_snapshot")
        if snapshot:
            snapshots[str(trace.get("worker_id") or "worker")] = snapshot
    return context.model_copy(update={"role_skill_snapshots": snapshots})


def _final_worker_evidence(state: "PlanningState") -> dict[str, Any]:
    """Build a bounded evidence packet; preserve reports and recent tool outcomes."""

    step = state.get("current_step")
    trace = dict(state.get("current_step_trace") or {})
    if step is None or not trace:
        return {}
    recent_tools = []
    for message in trace.get("messages", [])[-24:]:
        if isinstance(message, dict):
            kind = message.get("type", message.get("role"))
            if kind != "tool":
                continue
            recent_tools.append({
                "tool_call_id": message.get("tool_call_id"),
                "name": message.get("name"),
                "status": message.get("status"),
                "content": str(message.get("content", ""))[:1600],
            })
    return {
        "step": step.model_dump(mode="json"),
        "attempt": trace.get("attempt"),
        "worker_id": trace.get("worker_id"),
        "finish_reason": trace.get("finish_reason"),
        "stop_reason": trace.get("stop_reason"),
        "execution_summary": trace.get("execution_summary", {}),
        "general_result": trace.get("general_result"),
        "worker_submission": trace.get("worker_submission"),
        "code_review_report": trace.get("code_review_report"),
        "code_publication_receipt": trace.get("code_publication_receipt"),
        "recent_tool_results": recent_tools,
    }


def _trace_has_artifact_candidate(trace: dict[str, Any]) -> bool:
    result = trace.get("general_result") or {}
    submission = trace.get("worker_submission") or {}
    return bool(
        result.get("files")
        or submission.get("artifact_candidates")
        or submission.get("files")
        or trace.get("code_artifact_manifest")
        or trace.get("code_publication_receipt")
    )


class PlanningState(TypedDict, total=False):
    """一次用户请求在Planning Graph中的共享状态。"""

    context: PlanningContextPack
    event_id: str
    conversation_thread_id: str
    planning_run_id: str
    planning_failure: dict[str, Any]
    conversation_workspace_root: str

    plan_objective: str
    plan_success_criteria: list[str]
    remaining_steps: list[PlanStep]
    completed_step_reports: list[StepReport]
    planned_steps: list[PlanStep]
    handoff_publication_receipts: list[dict[str, Any]]

    current_step: PlanStep | None
    current_step_attempt: int
    current_step_executor_rounds: int
    current_step_report_rounds: int
    current_step_model_rounds: int
    current_step_tool_calls: int
    current_step_show_all_toolsets_calls: int
    current_step_trace: dict[str, Any]
    current_step_worker_traces: list[dict[str, Any]]
    current_worker_group_id: str
    current_step_leadership_override: str
    current_step_replacement_history: list[dict[str, Any]]
    current_step_stop_reason: str
    retry_current_step: bool
    current_code_runtime_session_id: str
    current_code_scheduler_decision: dict[str, Any] | None
    current_code_control_rounds: int
    code_control_history: list[dict[str, Any]]
    code_superseded_attempt_records: list[dict[str, Any]]

    replans_used: int
    replan_history: list[dict[str, Any]]
    pending_replan_reason: str
    pending_replan_source: str
    plan_challenge_resume_active: bool
    plan_challenge_scheduler_instruction: str

    final_review_rounds: int
    final_worker_repair_round: int
    final_worker_repair_request: dict[str, Any] | None
    final_worker_repair_history: list[dict[str, Any]]
    final_worker_repair_active: bool
    final_worker_role_review_pending: bool
    final_repair_model_rounds_used: int

    model_rounds_used: int
    scope_resolver_model_rounds: int

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
    # Explicit provenance: a real Scheduler Final Review issued a terminal decision.
    scheduler_final_decision: bool
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
        - (
            state.get("model_rounds_used", 0)
            - state.get("final_repair_model_rounds_used", 0)
        ),
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

        "step_show_all_toolsets_calls": max(
            0,

            (
                1

                - state.get(
                    "current_step_show_all_toolsets_calls",
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
    """判断当前是否仍然允许执行预算内Hard Replan。

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
    step = state.get("current_step")
    review_reserve = max(0, planning.max_step_report_rounds - state.get("current_step_report_rounds", 0)) if step is not None and step.worker_kind == "GENERAL" else 0
    model_limit = min(
        planning.max_step_executor_rounds
        - state.get("current_step_executor_rounds", 0),
        planning.max_step_model_rounds
        - state.get("current_step_model_rounds", 0) - review_reserve,
        _plan_model_rounds_remaining(state, planning) - review_reserve,
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
    completed_reports = list(state.get("completed_step_reports", []))
    # Handoff contracts are injected once in a short, prominent block below.
    # Keep the full prior Step reports for outcome context without duplicating
    # their potentially verbose handoff fields.
    reports = []
    for report in completed_reports:
        value = report.model_dump(mode="json") if hasattr(report, "model_dump") else dict(report)
        value.pop("handoff_apis", None)
        value.pop("handoff_api_receipts", None)
        value.pop("handoff_knowledge", None)
        reports.append(value)

    from handoff_knowledge import compact_direct_handoff
    direct_handoff = compact_direct_handoff(completed_reports)
    handoff_block = (
        "前序Worker直接交接（无需先调用历史工具）：\n"
        + _json_text(direct_handoff)
        + "\n"
        "validated_apis已由Harness核对结构和真实来源，可直接复用其接口契约；"
        "不要重复查同一接口。rejected_api_leads保留了验证失败的原交接和具体原因，"
        "只能作为查证线索，不能直接执行。worker_notes是前序Worker留下的工作说明，不等于独立验收。"
        "review_failures直接列出前序Reviewer或自报未通过的状态与原因，必须继续处理；"
        "若需要原始返回细节，再调用read_execution_history补查。\n\n"
        if direct_handoff
        else ""
    )

    if tool_limit > 0:
        tool_budget_instruction = (
            "本次Attempt可以按需调用工具，"
            f"但最多只能调用{tool_limit}次。"
        )

    else:
        tool_budget_instruction = (
            "本次Attempt的工具额度为0。"
            "不要尝试调用任何工具，也不要调用show_all_toolsets；"
            "请只依据用户请求、此前StepReport、长期记忆和"
            "已经确认的执行结果完成分析、整理或收口。"
            "无法确认的内容必须明确说明，不得编造。"
        )

    leadership_override = str(
        state.get("current_step_leadership_override", "")
    ).strip()
    leadership_block = (
        "[Leadership replacement assignment]\n"
        f"{leadership_override}\n\n"
        if leadership_override
        else ""
    )

    # Worker-facing execution capacity is supplied after role allocation and
    # skill preparation. Do not advertise the shared plan/control reserve.
    worker_budget = "" if current_step.worker_kind in {"GENERAL", "CODE"} else (
        "本次Attempt预算：\n"
        f"- 最多模型轮次：{model_limit}\n"
        f"- 最多工具调用：{tool_limit}\n"
        f"- 执行要求：{tool_budget_instruction}\n\n"
        f"整轮剩余预算：\n{_json_text(_remaining_budget(state, planning))}"
    )

    from prompt_loader import load_prompt

    from reporting.criteria import criterion_registry

    return (
        load_prompt("workers/step_scope") + "\n\n"
        "以下是背景，不是本次执行清单：\n"


        f"用户原始请求：\n"
        f"{_context(state).user_request}\n\n"

        f"整体目标：\n"
        f"{state['plan_objective']}\n\n"

        "本次只执行下面的Step：\n"
        f"当前Step（第{attempt}次尝试）：\n"
        f"{_json_text(current_step.model_dump(mode='json'))}\n\n"

        "HARNESS_CRITERIA: " + json.dumps(criterion_registry(current_step.success_criteria), ensure_ascii=False) + "\n\n"

        f"{handoff_block}"
        f"此前StepReport：\n"
        f"{_json_text(reports)}\n\n"

        f"{worker_budget}\n{leadership_block}"
    )


def _split_parallel_budget(total: int, count: int) -> tuple[int, ...]:
    """Deterministically divide one Step budget without multiplying cost."""

    if total < 0:
        raise ValueError("Parallel budget cannot be negative.")
    if count < 1:
        raise ValueError("Parallel Worker count must be positive.")
    quotient, remainder = divmod(total, count)
    return tuple(
        quotient + (1 if index < remainder else 0)
        for index in range(count)
    )


def _parallel_group_id(
    state: PlanningState,
    current_step: PlanStep,
    outer_attempt: int,
) -> str:
    event_id = str(
        state.get("event_id")
        or state.get("planning_run_id")
        or "event"
    ).strip()
    return (
        f"group:{event_id}:step:{current_step.step_id}:"
        f"execution:{outer_attempt}"
    )


def _group_attempt(
    group: StepWorkerGroup,
    assignment_key: str,
) -> WorkerAttempt:
    for slot in group.slots:
        if slot.assignment_key == assignment_key:
            return slot.current_attempt
    raise KeyError(f"Unknown Worker assignment: {assignment_key}")


def _slot_review_traces(
    group: StepWorkerGroup,
    assignment_key: str,
) -> list[dict[str, Any]]:
    """Rebuild Reporter inputs from one persisted slot after a graph replay."""

    for slot in group.slots:
        if slot.assignment_key != assignment_key:
            continue
        traces: list[dict[str, Any]] = []
        for attempt in slot.attempts:
            if attempt.review_payload is not None:
                traces.append(dict(attempt.review_payload))
                continue
            traces.append(
                materialize_worker_review_trace(
                    {
                        "assignment_key": slot.assignment_key,
                        "assignment_objective": attempt.objective,
                        "attempt": attempt.attempt_no,
                        "worker_id": attempt.worker_id,
                        "workspace_id": attempt.workspace.workspace_id,
                        "checkpoint_thread_id": (
                            attempt.workspace.checkpoint_thread_id
                        ),
                        "finish_reason": attempt.status.value,
                        "stop_reason": (
                            attempt.terminal_reason
                            or "Worker attempt has no persisted review payload."
                        ),
                    }
                )
            )
        return traces
    raise KeyError(f"Unknown Worker assignment: {assignment_key}")


def _group_review_traces(group: StepWorkerGroup) -> list[dict[str, Any]]:
    return [
        trace
        for slot in group.slots
        for trace in _slot_review_traces(group, slot.assignment_key)
    ]


def _review_artifact_index(
    packet: StepReviewPacket,
) -> dict[str, Any]:
    """Index only Harness-resolved candidates visible to this Reporter."""

    indexed: dict[str, Any] = {}
    for attempt in packet.attempts:
        for candidate in attempt.resolved_artifacts:
            review_ref = str(candidate.review_ref or "").strip()
            if not review_ref:
                raise ValueError(
                    "resolved artifact is missing its Harness review reference"
                )
            if review_ref in indexed:
                raise ValueError(
                    f"duplicate artifact review reference: {review_ref}"
                )
            indexed[review_ref] = candidate
    return indexed


def _build_general_step_report(current_step, trace, packet):
    """Translate General's own report; do not spend a model call judging it."""
    from workers.general_completion import GeneralResult

    raw = trace.get("general_result")
    result = GeneralResult.model_validate(raw) if raw else None
    messages = trace.get("messages", [])
    last = messages[-1] if messages else None
    last_kind = last.get("type", last.get("role")) if isinstance(last, dict) else getattr(last, "type", None)
    missing_submission = not result and last_kind == "tool"
    errors = [str(trace["error"])[:1200]] if trace.get("error") else []
    for message in trace.get("messages", []):
        kind = message.get("type", message.get("role")) if isinstance(message, dict) else getattr(message, "type", None)
        content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
        status = message.get("status") if isinstance(message, dict) else getattr(message, "status", None)
        if kind == "tool" and (status == "error" or any(marker in str(content) for marker in ("Traceback", "Error:", '"status": "error"'))):
            errors.append(str(content)[:1200])
    errors = list(dict.fromkeys(errors))[-6:]
    interrupted = trace.get("finish_reason") in {"ERROR", "BUDGET_EXHAUSTED", "LEADERSHIP_CANCEL", "LEADERSHIP_REPLACE"}
    if result is not None and not interrupted:
        status, summary = result.status, result.summary
        unresolved = list(result.unresolved_items)
    else:
        status = "FAILED" if interrupted or missing_submission or not trace.get("final_answer") else "PARTIAL"
        natural_answer = trace.get("final_answer") if not interrupted and not missing_submission else None
        summary = str(natural_answer or "General 未提交完整总结，无法确认任务完成。停止原因：" + str(trace.get("stop_reason") or trace.get("finish_reason") or "UNKNOWN"))[:4000]
        unresolved = ["General did not provide a complete self-report; do not assume the task succeeded."]
    if status == "COMPLETED" and unresolved:
        status = "PARTIAL"
    return StepReport(
        step_id=current_step.step_id, status=status, summary=summary,
        criterion_results=[StepCriterionResult(criterion_id=f'C{i}',criterion=criterion,
            status='MET' if any(c.criterion==criterion for c in (result.criterion_claims if result else [])) and status=='COMPLETED' else 'UNKNOWN',
            evidence=[c.conclusion for c in (result.criterion_claims if result else []) if c.criterion==criterion])
            for i,criterion in enumerate(getattr(current_step, "success_criteria", []),1)],
        handoff_knowledge=result.handoff_knowledge if result else [],
        assessment_source="GENERAL_SELF_REPORT" if (result or trace.get("final_answer")) and not interrupted and not missing_submission else "RUNTIME_FAILURE",
        stop_reason=("General submitted its self-report; no independent Step review." if result and not interrupted
                     else str(trace.get("stop_reason") or trace.get("finish_reason") or "General ended without a structured self-report.")),
        errors=errors,
        completed_work=[summary] if result else [],
        unresolved_items=unresolved,
        evidence=[f"Tool call: {e.tool_call_id}" for a in packet.attempts for e in a.resolved_evidence],
        # Self-report never approves file publication.
        approved_artifact_refs=[],
    )


def _publish_step_handoff(
    *,
    report: StepReport,
    packet: StepReviewPacket,
    run_id: str,
    run_storage_root: Path,
) -> tuple[StepReport, list[ArtifactPublicationReceipt]]:
    """Turn Reporter selection into verified shared paths using Harness code."""

    approved = list(report.approved_artifact_refs)
    output_contracts = {
        output.output_id: output
        for output in packet.task_contract.artifact_outputs
    }
    # Reporter output paths are never trusted. Only deterministic Publisher
    # receipts become StepReport artifacts.
    required_output_check = report.status == "COMPLETED" and any(
        output.required for output in output_contracts.values()
    )
    if not approved and not required_output_check:
        return report.model_copy(update={"artifacts": []}), []
    receipts: list[ArtifactPublicationReceipt] = []
    published: list[StepArtifactReport] = []
    try:
        candidates = _review_artifact_index(packet)
        unknown = [reference for reference in approved if reference not in candidates]
        if unknown:
            raise ValueError(
                "Reporter approved unknown artifact references: "
                + ", ".join(unknown)
            )
        approved_candidates = [candidates[reference] for reference in approved]
        approved_output_ids = [
            candidate.output_id
            for candidate in approved_candidates
            if candidate.output_id is not None
        ]
        unknown_outputs = sorted(
            set(approved_output_ids) - set(output_contracts)
        )
        if unknown_outputs:
            raise ValueError(
                "Reporter approved candidates for unknown output IDs: "
                + ", ".join(unknown_outputs)
            )
        if len(approved_output_ids) != len(set(approved_output_ids)):
            raise ValueError(
                "Reporter approved multiple candidates for one output contract"
            )
        if report.status == "COMPLETED":
            missing_required = [
                output.output_id
                for output in output_contracts.values()
                if output.required and output.output_id not in approved_output_ids
            ]
            if missing_required:
                raise ValueError(
                    "Reporter completed the Step without required artifacts: "
                    + ", ".join(missing_required)
                )
        if not approved:
            return report.model_copy(update={"artifacts": []}), []
        layout = initialize_run_workspace(
            run_id,
            storage_root=run_storage_root,
        )
        for reference in approved:
            candidate = candidates[reference]
            receipt = publish_artifact_to_handoff(
                layout=layout,
                candidate=candidate,
                output=(
                    output_contracts.get(candidate.output_id)
                    if candidate.output_id is not None
                    else None
                ),
            )
            receipts.append(receipt)
            published.append(
                StepArtifactReport(
                    path=receipt.handoff_path,
                    description=candidate.description,
                )
            )
    except (OSError, TypeError, ValueError) as error:
        message = f"Shared handoff publication failed: {error}"
        unresolved = list(report.unresolved_items)
        if "Approved artifacts were not fully published." not in unresolved:
            unresolved.append("Approved artifacts were not fully published.")
        return (
            report.model_copy(
                update={
                    "status": "FAILED",
                    "artifacts": published,
                    "errors": [*report.errors, message],
                    "unresolved_items": unresolved,
                    "next_action": (
                        "Retry the Step or correct its artifact publication "
                        "references before continuing."
                    ),
                }
            ),
            receipts,
        )

    return report.model_copy(update={"artifacts": published}), receipts


def _merge_publication_receipts(
    existing: list[dict[str, Any]],
    additions: list[ArtifactPublicationReceipt],
) -> list[dict[str, Any]]:
    merged = list(existing)
    known = {
        str(item.get("publication_id") or "")
        for item in merged
        if isinstance(item, dict)
    }
    for receipt in additions:
        if receipt.publication_id in known:
            continue
        merged.append(receipt.model_dump(mode="json"))
        known.add(receipt.publication_id)
    return merged


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
    worker_registry: WorkerAgentRegistry,
    worker_group_coordinator: WorkerGroupCoordinator,
    planning: PlanningSettings,
    model_output_max_tokens: int,
    role_models: dict | None = None,
    reporter_output_limits: dict | None = None,
    web_max_parallelism: int = 3,
    checkpointer: BaseCheckpointSaver | None = None,
    run_storage_root: Path = RUN_WORKSPACE_ROOT,
):
    """创建Hard规划、Simple执行和最终审核图。"""

    if web_max_parallelism < 1 or web_max_parallelism > 3:
        raise ValueError("web_max_parallelism must be between 1 and 3.")

    models = role_models or {}
    def role_model(role, fallback):
        return models[role] if role in models else fallback

    reporter_output_limits = reporter_output_limits or {}

    async def prepare_role(state: PlanningState, config: RunnableConfig, role: str):
        context = _context(state)
        catalog = context.skill_catalog
        if catalog is None:
            catalog = load_catalog()
        saved = context.role_skill_snapshots.get(role)
        snapshot = await prepare_skills(
            role_model(role, hard_model), role=role,
            task={
                "user_request": context.user_request,
                "user_instruction_history": context.user_instruction_history,
                "conversation_summary": context.conversation_summary,
                "recent_dialogue": [m.model_dump(mode="json") for m in context.recent_dialogue],
                "memory_context": context.memory_context,
                "execution_instructions": context.execution_instructions,
                "replacement_context": context.replacement_context,
                "toolset_catalog": context.toolset_catalog,
                "step_reports": state.get("completed_step_reports", []),
            },
            catalog=catalog, saved=saved, config=config,
            mode=context.skill_mode, fixed_ids=context.skill_fixed_ids.get(role, []),
            allow_model=_plan_model_rounds_remaining(state, planning) > HARD_CALL_MAX_ROUNDS,
        )
        updated_context = context.model_copy(update={
            "skill_catalog": catalog,
            "role_skill_snapshots": {**context.role_skill_snapshots, role: snapshot.model_dump(mode="json")},
        })
        return {
            "context": updated_context,
            "model_rounds_used": state.get("model_rounds_used", 0) + (snapshot.model_calls if saved is None else 0),
        }

    async def prepare_scheduler_node(state: PlanningState, config: RunnableConfig):
        context, resolver_calls = await prepare_scope_context(
            _context(state), role_model("scope_resolver", hard_model),
        )
        prepared = await prepare_role({**state, "context": context}, config, "scheduler")
        return {
            **prepared,
            "scope_resolver_model_rounds": (
                state.get("scope_resolver_model_rounds", 0) + resolver_calls
            ),
        }


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
            role_model("scheduler", hard_model),
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
                    "final_status": "FAILED" if result.used_fallback else "COMPLETED",
                    "final_answer": decision.final_answer or "",
                    "unmet_success_criteria": [],
                    "overall_stop_reason": "supervisor_generation_failed" if result.used_fallback else "supervisor_final",
                    "planning_failure": ({
                        "stage": "supervisor",
                        "attempts": result.model_rounds_used,
                        "validation_retries": result.validation_retry_count,
                        "errors": list(result.validation_errors),
                        "workers_started": False,
                    } if result.used_fallback else {}),
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
            "planned_steps": steps,
            "handoff_publication_receipts": [],
            "current_step": None,
            "current_step_attempt": 0,
            "current_step_executor_rounds": 0,
            "current_step_report_rounds": 0,
            "current_step_model_rounds": 0,
            "current_step_tool_calls": 0,
            "current_step_show_all_toolsets_calls": 0,
            "current_step_trace": {},
            "current_step_worker_traces": [],
            "current_worker_group_id": "",
            "current_step_leadership_override": "",
            "current_step_replacement_history": [],
            "current_step_stop_reason": "",
            "retry_current_step": False,
            "current_code_runtime_session_id": "",
            "current_code_scheduler_decision": None,
            "current_code_control_rounds": 0,
            "code_control_history": [],
            "code_superseded_attempt_records": [],
            "replans_used": 0,
            "replan_history": [],
            "pending_replan_reason": "",
            "pending_replan_source": "",

            "plan_challenge_resume_active": False,

            "plan_challenge_scheduler_instruction": "",
            "final_review_rounds": 0,
            "final_worker_repair_round": 0,
            "final_worker_repair_request": None,
            "final_worker_repair_history": [],
            "final_worker_repair_active": False,
            "final_worker_role_review_pending": False,
            "final_repair_model_rounds_used": 0,
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

    async def run_parallel_worker_slot(
        *,
        provisional: PlanningState,
        current_step: PlanStep,
        group_id: str,
        assignment_key: str,
        outer_attempt: int,
        model_limit: int,
        tool_limit: int,
        show_all_toolsets_limit: int,
        runtime_configurable: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Run one logical slot, including bounded Leadership replacement."""

        stored = await worker_group_coordinator.event_store.require_worker_group(
            group_id
        )
        group = StepWorkerGroup.model_validate(stored.snapshot)
        worker_attempt = _group_attempt(group, assignment_key)
        if worker_attempt.status not in ACTIVE_ATTEMPT_STATUSES:
            return _slot_review_traces(group, assignment_key)
        remaining_model = model_limit
        remaining_tools = tool_limit
        remaining_show_all_toolsets = show_all_toolsets_limit
        traces: list[dict[str, Any]] = []
        worker_agent = worker_registry.require(current_step.worker_kind)

        while True:
            if remaining_model <= 0:
                stop_reason = (
                    "并行Step的剩余Executor模型预算不足以启动该Worker。"
                )
                trace = {
                    "assignment_key": assignment_key,
                    "assignment_objective": worker_attempt.objective,
                    "attempt": worker_attempt.attempt_no,
                    "worker_id": worker_attempt.worker_id,
                    "workspace_id": worker_attempt.workspace.workspace_id,
                    "checkpoint_thread_id": (
                        worker_attempt.workspace.checkpoint_thread_id
                    ),
                    "finish_reason": "BUDGET_EXHAUSTED",
                    "stop_reason": stop_reason,
                    "execution_summary": {
                        "model_call_count": 0,
                        "tool_call_count": 0,
                        "show_all_toolsets_call_count": 0,
                    },
                    "applied_limits": {
                        "model_rounds": 0,
                        "tool_calls": remaining_tools,
                        "show_all_toolsets_calls": remaining_show_all_toolsets,
                    },
                }
                trace = materialize_worker_review_trace(trace)
                await worker_group_coordinator.finish_attempt(
                    group_id=group_id,
                    assignment_key=assignment_key,
                    attempt_no=worker_attempt.attempt_no,
                    worker_id=worker_attempt.worker_id,
                    status=WorkerAttemptStatus.BUDGET_EXHAUSTED,
                    terminal_reason=stop_reason,
                    review_payload=trace,
                    occurred_at=datetime.now(timezone.utc),
                )
                traces.append(trace)
                return traces

            await worker_group_coordinator.start_attempt(
                group_id=group_id,
                assignment_key=assignment_key,
                attempt_no=worker_attempt.attempt_no,
                worker_id=worker_attempt.worker_id,
                occurred_at=datetime.now(timezone.utc),
            )

            thread_id = worker_attempt.workspace.checkpoint_thread_id
            agent_state = {
                "skill_catalog": _context(provisional).skill_catalog,
                "skill_mode": _context(provisional).skill_mode,
                "skill_fixed_ids": _context(provisional).skill_fixed_ids,
                "skill_topics": current_step.skill_topics,
                "toolset_route_query": json.dumps(
                    {
                        "current_step": current_step.objective,
                        "success_criteria": current_step.success_criteria,
                        "required_context": (
                            [
                                item.model_dump(mode="json")
                                for item in current_step.target_selection.required_context
                            ]
                            if current_step.target_selection is not None
                            else []
                        ),
                        "rag_query": current_step.rag_query or "",
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "step_tool_access": current_step.tool_access,
                "toolset_route_fallback_query": _context(provisional).user_request,
                "toolset_route_full_user_request": _context(provisional).user_request,
                "execution_instructions": (
                    _context(provisional).execution_instructions
                ),
                "conversation_id": thread_id,
                "conversation_title": (
                    f"Planning Step {current_step.step_id} / {assignment_key}"
                ),
                "memory_context": _context(provisional).memory_context,
                "executor_model_run_limit": remaining_model,
                "executor_tool_run_limit": remaining_tools,
                "show_all_toolsets_run_limit": remaining_show_all_toolsets,
                "worker_id": worker_attempt.worker_id,
                "event_id": str(
                    provisional.get("event_id")
                    or provisional.get("planning_run_id")
                    or "event"
                ),
                "step_id": str(current_step.step_id),
                "worker_total_tool_calls": 0,
                "worker_last_progress_tool_call": 0,
                "worker_progress_reports": [],
                "worker_plan_challenges_remaining": max(
                    0, planning.max_replans - provisional.get("replans_used", 0)
                ),
            }
            instruction = (
                _build_step_instruction(
                    provisional,
                    current_step,
                    outer_attempt,
                    model_limit=remaining_model,
                    tool_limit=remaining_tools,
                    planning=planning,
                )
                + "\n\n[并行Worker分工]\n"
                + f"assignment_key: {assignment_key}\n"
                + f"你只负责这个分支：{worker_attempt.objective}\n"
                + "不要代替其他并行Worker重复完成它们的分支。"
            )

            stop_reason = "General Worker正常结束。"
            terminal_action = None
            leadership_decisions: list[dict[str, Any]] = []
            replacement_assignment = ""
            worker_submission = None
            worker_error = False
            details: dict[str, Any] = {}

            try:
                raw_details = await ask_worker(
                    worker_agent,
                    instruction,
                    trace_role="web" if current_step.worker_kind == "WEB" else "general",
                    thread_id=thread_id,
                    state_update=agent_state,
                    runtime_configurable=runtime_configurable,
                    return_details=True,
                )
                if not isinstance(raw_details, dict):
                    raise RuntimeError(
                        "Parallel Worker did not return execution details."
                    )
                details = raw_details
                summary = details.get("execution_summary", {})
                if not isinstance(summary, dict):
                    summary = {}
                terminal_action = details.get("worker_terminal_action")
                leadership_decisions = list(
                    details.get("worker_leadership_decisions", [])
                )
                worker_submission = details.get("worker_submission")
                if leadership_decisions:
                    decision = leadership_decisions[-1].get("decision", {})
                    if isinstance(decision, dict):
                        replacement_assignment = str(
                            decision.get("replacement_assignment") or ""
                        ).strip()
                used_models = max(int(summary.get("model_call_count", 0) or 0), 0)
                used_tools = max(int(summary.get("tool_call_count", 0) or 0), 0)
                used_toolset = max(
                    int(summary.get("show_all_toolsets_call_count", 0) or 0),
                    0,
                )
            except Exception as error:
                worker_error = True
                summary = {}
                used_models = 0
                used_tools = 0
                used_toolset = 0
                stop_reason = "General Worker发生异常。"
                details = {
                    "error": f"{type(error).__name__}: {error}",
                }

            remaining_model = max(0, remaining_model - used_models)
            remaining_tools = max(0, remaining_tools - used_tools)
            remaining_show_all_toolsets = max(
                0,
                remaining_show_all_toolsets - used_toolset,
            )

            if terminal_action == "ACCEPT":
                stop_reason = "Leadership要求当前Worker立即提交独立验收。"
            elif terminal_action == "CANCEL":
                stop_reason = "Leadership已取消当前Worker。"

            can_replace = (
                terminal_action == "REPLACE"
                and bool(replacement_assignment)
                and remaining_model > 0
                and worker_attempt.attempt_no < planning.max_step_attempts
            )
            if terminal_action == "REPLACE":
                stop_reason = (
                    "Leadership要求用新的任务说明替换当前Worker。"
                    if can_replace
                    else "Leadership要求替换Worker，但该分支剩余预算不足。"
                )

            if worker_submission:
                finish_reason = "READY_FOR_REVIEW"
                terminal_status = WorkerAttemptStatus.SUBMITTED
            elif terminal_action == "CANCEL":
                finish_reason = "LEADERSHIP_CANCEL"
                terminal_status = WorkerAttemptStatus.CANCELLED
            elif terminal_action == "REPLACE":
                finish_reason = "LEADERSHIP_REPLACE"
                terminal_status = WorkerAttemptStatus.BUDGET_EXHAUSTED
            elif worker_error:
                finish_reason = "ERROR"
                terminal_status = WorkerAttemptStatus.FAILED
            elif remaining_model <= 0 or (
                used_tools > 0 and remaining_tools <= 0
            ):
                finish_reason = "BUDGET_EXHAUSTED"
                terminal_status = WorkerAttemptStatus.BUDGET_EXHAUSTED
            else:
                finish_reason = "NATURAL_EXIT"
                terminal_status = WorkerAttemptStatus.NATURAL_EXIT

            trace = {
                "assignment_key": assignment_key,
                "assignment_objective": worker_attempt.objective,
                "attempt": worker_attempt.attempt_no,
                "worker_id": worker_attempt.worker_id,
                "workspace_id": worker_attempt.workspace.workspace_id,
                "checkpoint_thread_id": thread_id,
                "role_skill_snapshot": details.get("role_skill_snapshot"),
                "final_answer": details.get("final_answer", ""),
                "execution_summary": summary,
                "worker_terminal_action": terminal_action,
                "worker_leadership_decisions": leadership_decisions,
                "worker_submission": worker_submission,
                "worker_cancellation_record": details.get(
                    "worker_cancellation_record"
                ),
                "applied_limits": {
                    "model_rounds": remaining_model + used_models,
                    "tool_calls": remaining_tools + used_tools,
                    "show_all_toolsets_calls": (
                        remaining_show_all_toolsets + used_toolset
                    ),
                },
                "finish_reason": finish_reason,
                "stop_reason": stop_reason,
            }
            if "error" in details:
                trace["error"] = details["error"]
            trace = materialize_worker_review_trace(trace)
            traces.append(trace)

            if can_replace:
                replacement = await worker_group_coordinator.replace_attempt(
                    group_id=group_id,
                    assignment_key=assignment_key,
                    attempt_no=worker_attempt.attempt_no,
                    worker_id=worker_attempt.worker_id,
                    replacement_objective=replacement_assignment,
                    reason=stop_reason,
                    review_payload=trace,
                    occurred_at=datetime.now(timezone.utc),
                )
                worker_attempt = _group_attempt(
                    replacement.group,
                    assignment_key,
                )
                remaining_show_all_toolsets = 0
                continue

            await worker_group_coordinator.finish_attempt(
                group_id=group_id,
                assignment_key=assignment_key,
                attempt_no=worker_attempt.attempt_no,
                worker_id=worker_attempt.worker_id,
                status=terminal_status,
                terminal_reason=stop_reason,
                review_payload=trace,
                occurred_at=datetime.now(timezone.utc),
            )
            return traces

    async def step_executor_node(
        state: PlanningState,
        config: RunnableConfig,
    ) -> Command[
        Literal[
            "step_executor",
            "code_controller",
            "step_reporter",
            "general_report",
            "final_reviewer",
        ]
    ]:
        """执行当前Step的一次Attempt。

        普通模型、工具和show_all_toolsets额度
        都由Planning Graph计算后写入Agent State。

        show_all_toolsets的额度属于整个Step，
        不会因为创建新的Attempt Thread而重置。
        """

        raw_pending_code_decision = state.get(
            "current_code_scheduler_decision"
        )
        pending_code_action = (
            str(raw_pending_code_decision.get("action") or "")
            if isinstance(raw_pending_code_decision, dict)
            else ""
        )
        if (
            _execution_budget_exhausted(
                state,
                planning,
            )
            and not state.get("final_worker_repair_active", False)
            # STOP consumes no model budget inside Code Runtime.  It must still
            # reach the frozen session so the containers are archived and
            # released instead of leaking when the plan budget reaches zero.
            and pending_code_action != "STOP"
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

            raw_code_decision = state.get("current_code_scheduler_decision")
            code_decision_action = (
                str(raw_code_decision.get("action") or "")
                if isinstance(raw_code_decision, dict)
                else ""
            )
            # Final Reviewer repairs resume the same checkpoint and receive one
            # fresh normal Worker budget. Other generic retries create a new attempt.
            final_repair_active = state.get("final_worker_repair_active", False)
            challenge_resume_active = state.get("plan_challenge_resume_active", False)
            same_worker_resume = final_repair_active or challenge_resume_active
            attempt = state.get("current_step_attempt", 1)
            if not same_worker_resume and not (
                current_step.worker_kind == "CODE"
                and code_decision_action in {"CONTINUE", "STOP"}
            ):
                attempt += 1

            executor_rounds = 0 if same_worker_resume else state.get(
                "current_step_executor_rounds",
                0,
            )

            report_rounds = 0 if same_worker_resume else state.get(
                "current_step_report_rounds",
                0,
            )

            step_model_rounds = 0 if same_worker_resume else state.get(
                "current_step_model_rounds",
                0,
            )

            step_tool_calls = 0 if same_worker_resume else state.get(
                "current_step_tool_calls",
                0,
            )

            step_show_all_toolsets_calls = (
                0
                if same_worker_resume
                else state.get(
                    "current_step_show_all_toolsets_calls",
                    0,
                )
            )
            leadership_override = str(
                state.get("current_step_leadership_override", "")
            )
            replacement_history = list(
                state.get("current_step_replacement_history", [])
            )
            code_runtime_session_id = str(
                state.get("current_code_runtime_session_id") or ""
            )
            code_scheduler_decision = raw_code_decision
            code_control_rounds = state.get("current_code_control_rounds", 0)
            code_control_history = list(state.get("code_control_history", []))
            code_superseded_attempt_records = list(
                state.get("code_superseded_attempt_records", [])
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
            step_show_all_toolsets_calls = 0
            leadership_override = ""
            replacement_history = []
            code_runtime_session_id = ""
            code_scheduler_decision = None
            code_control_rounds = 0
            code_control_history = []
            code_superseded_attempt_records = []
            final_repair_active = False
            challenge_resume_active = False
            same_worker_resume = False

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

                "current_step_show_all_toolsets_calls": (
                    step_show_all_toolsets_calls
                ),

                "remaining_steps": (
                    remaining_steps
                ),

                "current_step_leadership_override": leadership_override,

                "current_step_replacement_history": replacement_history,
                "current_code_runtime_session_id": code_runtime_session_id,
                "current_code_scheduler_decision": code_scheduler_decision,
                "current_code_control_rounds": code_control_rounds,
                "code_control_history": code_control_history,
                "code_superseded_attempt_records": (
                    code_superseded_attempt_records
                ),
            }
        )

        if final_repair_active:
            model_limit = planning.max_step_executor_rounds
            tool_limit = planning.max_step_tool_calls
        elif challenge_resume_active:
            model_limit = min(
                planning.max_step_executor_rounds,
                _plan_model_rounds_remaining(provisional, planning),
            )
            tool_limit = planning.max_step_tool_calls
        else:
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

                    "current_step_show_all_toolsets_calls": (
                        step_show_all_toolsets_calls
                    ),

                    "current_step_trace": {
                        "status": (
                            "not_started"
                        ),

                        "reason": (
                            reason
                        ),
                    },

                    "current_step_worker_traces": [],

                    "current_worker_group_id": "",

                    "current_step_stop_reason": (
                        reason
                    ),

                    "remaining_steps": (
                        remaining_steps
                    ),

                    "retry_current_step": False,
                },

                goto="general_report" if current_step.worker_kind == "GENERAL" else "step_reporter",
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

        # show_all_toolsets属于Step级额度。

        #
        # 第一次Attempt尚未使用时为1；
        # 任意Attempt真正调用过一次后，
        # 后续Attempt全部为0。
        show_all_toolsets_limit = (
            0

            if tool_limit <= 0

            else max(
                0,

                1
                - step_show_all_toolsets_calls,
            )
        )

        if current_step.execution_mode == "PARALLEL":
            group_id = _parallel_group_id(
                provisional,
                current_step,
                attempt,
            )
            stored_group = await worker_group_coordinator.create_group(
                current_step,
                event_id=str(
                    state.get("event_id")
                    or state.get("planning_run_id")
                    or "event"
                ),
                group_id=group_id,
            )
            group = StepWorkerGroup.model_validate(stored_group.snapshot)
            slot_count = len(group.slots)
            model_budgets = _split_parallel_budget(model_limit, slot_count)
            tool_budgets = _split_parallel_budget(tool_limit, slot_count)
            semaphore = asyncio.Semaphore(web_max_parallelism)

            async def run_bounded_slot(
                assignment_key: str,
                slot_index: int,
            ) -> list[dict[str, Any]]:
                async with semaphore:
                    return await run_parallel_worker_slot(
                        provisional=provisional,
                        current_step=current_step,
                        group_id=group_id,
                        assignment_key=assignment_key,
                        outer_attempt=attempt,
                        model_limit=model_budgets[slot_index],
                        tool_limit=tool_budgets[slot_index],
                        show_all_toolsets_limit=(
                            show_all_toolsets_limit
                            if slot_index == 0
                            else 0
                        ),
                        runtime_configurable={
                            key: value
                            for key, value in (
                                config.get("configurable") or {}
                            ).items()
                            if key in {
                                "event_pause_control",
                                "resume_from_checkpoint",
                            }
                        },
                    )

            await asyncio.gather(
                *(
                    run_bounded_slot(slot.assignment_key, index)
                    for index, slot in enumerate(group.slots)
                )
            )
            latest_stored = (
                await worker_group_coordinator.event_store.require_worker_group(
                    group_id
                )
            )
            latest_group = StepWorkerGroup.model_validate(
                latest_stored.snapshot
            )
            if latest_group.nonterminal_worker_ids:
                raise RuntimeError(
                    "Parallel Worker Group returned before ALL_TERMINAL join."
                )
            worker_traces = _group_review_traces(latest_group)

            parallel_model_rounds = sum(
                max(
                    int(
                        trace.get("execution_summary", {}).get(
                            "model_call_count",
                            0,
                        )
                        or 0
                    ),
                    0,
                )
                for trace in worker_traces
            )
            parallel_tool_calls = sum(
                max(
                    int(
                        trace.get("execution_summary", {}).get(
                            "tool_call_count",
                            0,
                        )
                        or 0
                    ),
                    0,
                )
                for trace in worker_traces
            )
            parallel_toolset_calls = min(
                1,
                sum(
                    max(
                        int(
                            trace.get("execution_summary", {}).get(
                                "show_all_toolsets_call_count",
                                0,
                            )
                            or 0
                        ),
                        0,
                    )
                    for trace in worker_traces
                ),
            )
            executor_rounds += parallel_model_rounds
            step_model_rounds += parallel_model_rounds
            step_tool_calls += parallel_tool_calls
            step_show_all_toolsets_calls = min(
                1,
                step_show_all_toolsets_calls + parallel_toolset_calls,
            )
            model_rounds_used = (
                state.get("model_rounds_used", 0)
                + parallel_model_rounds
            )
            tool_calls_used = (
                state.get("tool_calls_used", 0)
                + parallel_tool_calls
            )
            phase_tool_calls_used = (
                state.get("phase_tool_calls_used", 0)
                + parallel_tool_calls
            )
            stop_reason = (
                "并行Worker Group已满足ALL_TERMINAL屏障，等待统一验收。"
            )
            aggregate_trace = {
                "status": "join_ready",
                "group_id": group_id,
                "join_policy": current_step.join_policy,
                "revision": latest_stored.revision,
                "worker_count": slot_count,
                "worker_traces": worker_traces,
                "finish_reason": "ALL_TERMINAL",
                "stop_reason": stop_reason,
            }
            return Command(
                update={
                    "current_step": current_step,
                    "current_step_attempt": attempt,
                    "current_step_executor_rounds": executor_rounds,
                    "current_step_report_rounds": report_rounds,
                    "current_step_model_rounds": step_model_rounds,
                    "current_step_tool_calls": step_tool_calls,
                    "current_step_show_all_toolsets_calls": (
                        step_show_all_toolsets_calls
                    ),
                    "current_step_trace": aggregate_trace,
                    "current_step_worker_traces": worker_traces,
                    "context": _record_skill_traces(_context(state), worker_traces),
                    "current_worker_group_id": group_id,
                    "current_step_leadership_override": "",
                    "current_step_replacement_history": [],
                    "current_step_stop_reason": stop_reason,
                    "remaining_steps": remaining_steps,
                    "model_rounds_used": model_rounds_used,
                    "tool_calls_used": tool_calls_used,
                    "phase_tool_calls_used": phase_tool_calls_used,
                    "retry_current_step": False,
                },
                goto="general_report" if current_step.worker_kind == "GENERAL" else "step_reporter",
            )

        thread_id = (
            f"{state.get('conversation_thread_id', 'conversation')}:"
            f"{state.get('planning_run_id', 'planning')}:"
            f"step_{current_step.step_id}:"
            f"attempt_{attempt}"
        )

        worker_id = (
            f"worker:{state.get('event_id', 'event')}:"
            f"step:{current_step.step_id}:attempt:{attempt}"
        )

        agent_state = {
            "skill_catalog": _context(state).skill_catalog,
            "skill_mode": _context(state).skill_mode,
            "skill_fixed_ids": _context(state).skill_fixed_ids,
            "skill_topics": current_step.skill_topics,
            "toolset_route_query": json.dumps(
                {
                    "current_step": current_step.objective,
                    "success_criteria": current_step.success_criteria,
                    "required_context": (
                        [
                            item.model_dump(mode="json")
                            for item in current_step.target_selection.required_context
                        ]
                        if current_step.target_selection is not None
                        else []
                    ),
                    "rag_query": current_step.rag_query or "",
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "step_tool_access": current_step.tool_access,
            "toolset_route_fallback_query": _context(state).user_request,
            "toolset_route_full_user_request": _context(state).user_request,
            "execution_instructions": _context(state).execution_instructions,
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

            "show_all_toolsets_run_limit": (
                show_all_toolsets_limit
            ),

            # Stable control-plane identity. The Leadership Bridge requires
            # these fields before it will persist a progress report.
            "worker_id": (
                worker_id
            ),
            "event_id": state.get(
                "event_id",
                state.get("planning_run_id", ""),
            ),
            "step_id": str(current_step.step_id),
            "worker_criterion_refs": {f"C{i}": text for i, text in enumerate(current_step.success_criteria, 1)},
            "worker_target_selection": (
                current_step.target_selection.model_dump(mode="json")
                if current_step.target_selection is not None else None
            ),
            "worker_invalid_target_receipts": [],
            "worker_total_tool_calls": 0,
            "worker_last_progress_tool_call": 0,
            "worker_progress_reports": [],
            "worker_plan_challenges_remaining": max(
                0, planning.max_replans - state.get("replans_used", 0)
            ),
            "final_worker_repair_request": (
                state.get("final_worker_repair_request")
                if final_repair_active else None
            ),
            "conversation_workspace_root": state.get(
                "conversation_workspace_root",
                "",
            ),
        }
        if current_step.worker_kind == "CODE":
            if current_step.code_task is None:
                raise RuntimeError("CODE Step is missing its frozen contract")
            agent_state["code_task"] = current_step.code_task.model_dump(
                mode="json"
            )
            if code_runtime_session_id:
                agent_state["code_runtime_session_id"] = code_runtime_session_id
            if code_scheduler_decision is not None:
                agent_state["code_scheduler_decision"] = code_scheduler_decision
            if code_superseded_attempt_records:
                agent_state["code_superseded_attempt_records"] = (
                    code_superseded_attempt_records
                )

        instruction = (
            _build_step_instruction(
                provisional,
                current_step,
                attempt,
                model_limit=model_limit,
                tool_limit=tool_limit,
                planning=planning,
            )
        )
        if final_repair_active:
            instruction += (
                "\n\n[Final Reviewer事实型返修单]\n"
                + _json_text(state.get("final_worker_repair_request") or {})
                + "\n这是验收缺口，不是接口或执行步骤。沿用本Worker已有上下文和真实工具，"
                  "保留已完成结果，只补缺口并重新提交本角色原有的完成Schema。"
            )
        elif challenge_resume_active:
            instruction += (
                "\n\n[Scheduler对计划异议的裁决]\n"
                + str(state.get("plan_challenge_scheduler_instruction") or "")
                + "\n沿用本Worker已有上下文和真实工具结果。按裁决后的当前Step继续；"
                  "不要重复提交已经裁决的同一异议。"
            )

        stop_reason = (
            "General Worker正常结束。"
        )
        terminal_action = None
        leadership_decisions: list[dict[str, Any]] = []
        replacement_assignment = ""
        worker_submission = None
        code_worker_submission = None
        code_review_loop = None
        code_review_report = None
        code_artifact_manifest = None
        code_publication_receipt = None
        code_handoff_publication_receipts = None
        code_integration_commit = None
        code_integration_status = None
        code_attempt_archive = None
        code_attempt_final_record = None
        code_runtime_session_id_result = ""
        code_scheduler_decision_applied = None
        code_superseded_attempt_records_result = (
            code_superseded_attempt_records
        )
        worker_error = False

        try:
            details = await (
                ask_worker(
                    worker_registry.require(current_step.worker_kind),
                    instruction,
                    trace_role="code_agent" if current_step.worker_kind == "CODE" else current_step.worker_kind.lower(),

                    thread_id=(
                        thread_id
                    ),

                    state_update=(
                        agent_state
                    ),

                    runtime_configurable={
                        key: value
                        for key, value in (config.get("configurable") or {}).items()
                        if key in {
                            "event_pause_control",
                            "resume_from_checkpoint",
                        }
                    },

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
            terminal_action = details.get("worker_terminal_action")
            leadership_decisions = list(
                details.get("worker_leadership_decisions", [])
            )
            worker_submission = details.get(
                "worker_submission"
            )
            code_worker_submission = details.get(
                "code_worker_submission"
            )
            code_review_loop = details.get("code_review_loop")
            code_review_report = details.get("code_review_report")
            code_artifact_manifest = details.get(
                "code_artifact_manifest"
            )
            code_publication_receipt = details.get(
                "code_publication_receipt"
            )
            code_handoff_publication_receipts = details.get(
                "code_handoff_publication_receipts"
            )
            code_integration_commit = details.get("code_integration_commit")
            code_integration_status = details.get("code_integration_status")
            code_attempt_archive = details.get("code_attempt_archive")
            code_attempt_final_record = details.get(
                "code_attempt_final_record"
            )
            code_runtime_session_id_result = str(
                details.get("code_runtime_session_id") or ""
            )
            code_scheduler_decision_applied = details.get(
                "code_scheduler_decision_applied"
            )
            code_superseded_attempt_records_result = list(
                details.get("code_superseded_attempt_records")
                or code_superseded_attempt_records
            )
            if leadership_decisions:
                last_decision = leadership_decisions[-1].get("decision", {})
                if isinstance(last_decision, dict):
                    replacement_assignment = str(
                        last_decision.get("replacement_assignment") or ""
                    ).strip()

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

            attempt_show_all_toolsets_calls = max(
                0,

                int(
                    summary.get(
                        "show_all_toolsets_call_count",
                        0,
                    )
                    or 0
                ),
            )

            trace = {
                "attempt": (
                    attempt
                ),

                "worker_id": worker_id,
                "final_reviewer_request": (
                    state.get("final_worker_repair_request")
                    if final_repair_active else None
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

                "handoff_source_messages": (
                    details.get("handoff_source_messages", [])
                ),

                "execution_summary": (
                    summary
                ),

                "worker_terminal_action": terminal_action,

                "worker_leadership_decisions": leadership_decisions,

                "worker_submission": worker_submission,
                "general_result": details.get("general_result"),
                "role_skill_snapshot": details.get("role_skill_snapshot"),
                "worker_cancellation_record": details.get(
                    "worker_cancellation_record"
                ),

                "code_worker_submission": code_worker_submission,

                "code_review_loop": code_review_loop,

                "code_review_report": code_review_report,

                "code_artifact_manifest": code_artifact_manifest,

                "code_publication_receipt": code_publication_receipt,

                "code_handoff_publication_receipts": (
                    code_handoff_publication_receipts
                ),

                "code_integration_commit": code_integration_commit,

                "code_integration_status": code_integration_status,

                "code_attempt_archive": code_attempt_archive,

                "code_attempt_final_record": code_attempt_final_record,
                "code_runtime_session_id": code_runtime_session_id_result,
                "code_scheduler_decision_applied": (
                    code_scheduler_decision_applied
                ),
                "code_superseded_attempt_records": (
                    code_superseded_attempt_records_result
                ),

                "applied_limits": {
                    "model_rounds": (
                        model_limit
                    ),

                    "tool_calls": (
                        tool_limit
                    ),

                    "show_all_toolsets_calls": (
                        show_all_toolsets_limit
                    ),
                },
            }

        except Exception as error:
            if current_step.worker_kind == "CODE":
                detail = f"{type(error).__name__}: {error}"
                failed_trace = {
                    "attempt": attempt,
                    "worker_id": worker_id,
                    "error": detail,
                    "usage_status": "unknown_after_runtime_error",
                    "finish_reason": "ERROR",
                    "stop_reason": "Code Agent执行异常。",
                    "final_reviewer_request": (
                        state.get("final_worker_repair_request")
                        if final_repair_active else None
                    ),
                }
                if final_repair_active:
                    history = list(state.get("final_worker_repair_history", []))
                    history.append({
                        "round": state.get("final_worker_repair_round", 0),
                        "reviewer_request": state.get("final_worker_repair_request"),
                        "worker_evidence": failed_trace,
                    })
                    return Command(update={
                        "current_step_trace": failed_trace,
                        "current_step_worker_traces": [failed_trace],
                        "current_step_stop_reason": "Code Agent执行异常。",
                        "final_worker_repair_history": history,
                        "final_worker_repair_request": None,
                        "final_worker_repair_active": False,
                        "retry_current_step": False,
                        "overall_stop_reason": "final_code_worker_repair_error",
                    }, goto="final_reviewer")
                # An unhandled initial runtime failure is not a Reviewer verdict.
                return Command(update={
                    "final_status": "FAILED",
                    "final_answer": f"任务失败：Code Agent 第 {current_step.step_id} 步执行异常。\n原因：{detail}",
                    "overall_stop_reason": "code_runtime_error",
                    "current_step_stop_reason": "Code Agent执行异常，任务已停止。",
                    "current_step_trace": failed_trace,
                    "current_step_worker_error": True,
                    "retry_current_step": False,
                }, goto=END)
            worker_error = True
            attempt_model_rounds = 0
            attempt_tool_calls = 0
            attempt_show_all_toolsets_calls = 0

            stop_reason = (
                "General Worker发生异常。"
            )

            trace = {
                "attempt": (
                    attempt
                ),

                "worker_id": worker_id,
                "final_reviewer_request": (
                    state.get("final_worker_repair_request")
                    if final_repair_active else None
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

                    "show_all_toolsets_calls": (
                        show_all_toolsets_limit
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
        step_show_all_toolsets_calls = min(
            1,

            (
                step_show_all_toolsets_calls
                + attempt_show_all_toolsets_calls
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

        if terminal_action == "ACCEPT":
            stop_reason = "Leadership要求当前Worker立即提交独立验收。"
        elif terminal_action == "CANCEL":
            stop_reason = "Leadership已取消当前Worker。"

        can_replace = (
            terminal_action == "REPLACE"
            and current_step.worker_kind != "GENERAL"
            and bool(replacement_assignment)
            and attempt < planning.max_step_attempts
            and executor_rounds < planning.max_step_executor_rounds
            and step_model_rounds < planning.max_step_model_rounds
            and model_rounds_used < planning.max_plan_model_rounds
        )
        if terminal_action == "REPLACE":
            stop_reason = (
                "Leadership要求用新的任务说明替换当前Worker。"
                if can_replace
                else "Leadership要求替换Worker，但Step剩余预算不足。"
            )

        if code_review_report:
            code_report = CodeReviewReport.model_validate(code_review_report)
            finish_reason = (
                "CODE_APPLIED"
                if code_report.verdict == "PASSED"
                else "CODE_ESCALATED"
            )
            stop_reason = (
                "Code Reviewer已完成发布并提交APPLIED报告。"
                if code_report.verdict == "PASSED"
                else "Code Reviewer未批准候选并已上报Scheduler。"
            )
        elif current_step.worker_kind == "GENERAL" and trace.get("general_result"):
            finish_reason = "GENERAL_SELF_REPORT"
        elif worker_submission:
            finish_reason = "READY_FOR_REVIEW"
        elif terminal_action == "ACCEPT":
            finish_reason = "LEADERSHIP_ACCEPT"
        elif terminal_action == "CANCEL":
            finish_reason = "LEADERSHIP_CANCEL"
        elif terminal_action == "REPLACE":
            finish_reason = "LEADERSHIP_REPLACE"
        elif worker_error:
            finish_reason = "ERROR"
        elif "预算" in stop_reason or "上限" in stop_reason:
            finish_reason = "BUDGET_EXHAUSTED"
        else:
            finish_reason = "NATURAL_EXIT"

        trace["finish_reason"] = finish_reason
        trace["stop_reason"] = stop_reason

        if can_replace:
            replacement_history = [*replacement_history, trace]

        if final_repair_active:
            history = list(state.get("final_worker_repair_history", []))
            history.append({
                "round": state.get("final_worker_repair_round", 0),
                "reviewer_request": state.get("final_worker_repair_request"),
                "worker_evidence": {
                    "finish_reason": trace.get("finish_reason"),
                    "stop_reason": trace.get("stop_reason"),
                    "general_result": trace.get("general_result"),
                    "worker_submission": trace.get("worker_submission"),
                    "code_review_report": trace.get("code_review_report"),
                    "execution_summary": trace.get("execution_summary", {}),
                },
            })
            role_review = True
            update = {
                "current_step": current_step,
                "current_step_attempt": attempt,
                "current_step_executor_rounds": executor_rounds,
                "current_step_report_rounds": report_rounds,
                "current_step_model_rounds": step_model_rounds,
                "current_step_tool_calls": step_tool_calls,
                "current_step_show_all_toolsets_calls": step_show_all_toolsets_calls,
                "current_step_trace": trace,
                "current_step_worker_traces": [trace],
                "context": _record_skill_traces(_context(state), [trace]),
                "current_step_stop_reason": stop_reason,
                "current_code_runtime_session_id": code_runtime_session_id_result,
                "current_code_scheduler_decision": None,
                "current_code_control_rounds": code_control_rounds,
                "code_control_history": code_control_history,
                "code_superseded_attempt_records": code_superseded_attempt_records_result,
                "remaining_steps": remaining_steps,
                "model_rounds_used": model_rounds_used,
                "tool_calls_used": tool_calls_used,
                "phase_tool_calls_used": phase_tool_calls_used,
                "final_repair_model_rounds_used": (
                    state.get("final_repair_model_rounds_used", 0)
                    + attempt_model_rounds
                ),
                "final_worker_repair_history": history,
                "final_worker_repair_request": None,
                "final_worker_repair_active": False,
                "final_worker_role_review_pending": role_review,
                "retry_current_step": False,
                "overall_stop_reason": "final_worker_repair_completed",
            }
            return Command(
                update=update,
                goto=(
                    "general_report"
                    if current_step.worker_kind == "GENERAL"
                    else "step_reporter"
                ),
            )

        code_requires_scheduler = False
        if current_step.worker_kind == "CODE" and code_review_loop:
            code_requires_scheduler = (
                CodeReviewLoopState.model_validate(code_review_loop).status
                == "ESCALATED_TO_SCHEDULER"
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

                "current_step_show_all_toolsets_calls": (
                    step_show_all_toolsets_calls
                ),

                "current_step_trace": (
                    trace
                ),

                "current_step_worker_traces": [trace],
                "context": _record_skill_traces(_context(state), [trace]),

                "current_worker_group_id": "",

                "current_step_leadership_override": (
                    replacement_assignment
                    if can_replace
                    else leadership_override
                ),

                "current_step_replacement_history": replacement_history,

                "current_step_stop_reason": (
                    stop_reason
                ),

                "current_code_runtime_session_id": (
                    code_runtime_session_id_result
                ),

                # The directive is single-use. The runtime result records what
                # was applied; a later escalation must receive a fresh decision.
                "current_code_scheduler_decision": None,

                "current_code_control_rounds": code_control_rounds,

                "code_control_history": code_control_history,

                "code_superseded_attempt_records": (
                    code_superseded_attempt_records_result
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

                "retry_current_step": can_replace,
                "plan_challenge_resume_active": False,
                "plan_challenge_scheduler_instruction": "",
            },

            goto=(
                "step_executor"
                if can_replace
                else (
                    "code_controller"
                    if code_requires_scheduler
                    else "general_report" if current_step.worker_kind == "GENERAL" else "step_reporter"
                )
            ),
        )

    async def code_controller_node(
        state: PlanningState,
        config: RunnableConfig,
    ) -> Command[Literal["step_executor", "step_reporter"]]:
        """Turn one Reviewer escalation into an executable CODE directive.

        The Reviewer supplies evidence and a recommendation.  This node is the
        Scheduler-owned policy boundary: it independently chooses CONTINUE,
        RESTART, or STOP, persists that decision, then sends it through the
        normal Step Executor so Code Runtime can mutate the real session.
        """

        current_step = state.get("current_step")
        trace = dict(state.get("current_step_trace") or {})
        raw_loop = trace.get("code_review_loop")
        raw_report = trace.get("code_review_report")
        session_id = str(
            state.get("current_code_runtime_session_id") or ""
        ).strip()

        if (
            current_step is None
            or current_step.worker_kind != "CODE"
            or not raw_loop
            or not raw_report
            or not session_id
        ):
            trace["code_controller_error"] = (
                "Reviewer escalation is missing its CODE Step, report, "
                "review loop, or live runtime session."
            )
            return Command(
                update={
                    "current_step_trace": trace,
                    "current_step_stop_reason": trace["code_controller_error"],
                    "retry_current_step": False,
                },
                goto="step_reporter",
            )

        loop = CodeReviewLoopState.model_validate(raw_loop)
        report = CodeReviewReport.model_validate(raw_report)
        if loop.status != "ESCALATED_TO_SCHEDULER":
            raise ValueError(
                "Code Controller requires an ESCALATED_TO_SCHEDULER loop."
            )
        if report.candidate != loop.candidate:
            raise ValueError("Code Controller received a stale Reviewer report.")

        control_rounds = state.get("current_code_control_rounds", 0)
        model_remaining = _plan_model_rounds_remaining(state, planning)
        forced_reason = ""
        if report.confirmed_plan_challenge is not None:
            forced_reason = (
                "Code Reviewer confirmed a material plan conflict; stop this frozen "
                "CODE session so the Planning Graph can ask Scheduler to adjudicate it."
            )
        elif control_rounds >= planning.max_step_attempts:
            forced_reason = (
                "The CODE control-round limit was reached; stop the frozen "
                "attempt instead of creating an unbounded repair loop."
            )
        elif model_remaining < HARD_CALL_MAX_ROUNDS + 2:
            forced_reason = (
                "The remaining plan-model budget cannot safely fund both a "
                "Scheduler decision and another Worker/Reviewer exchange."
            )

        if forced_reason:
            decision = SchedulerCodeDecision(
                action="STOP",
                reason=forced_reason,
            )
            controller_rounds = 0
            used_fallback = True
        else:
            result = await run_hard_code_scheduler(
                role_model("code_scheduler", hard_model),
                context=_context(state),
                plan_objective=state["plan_objective"],
                current_step=current_step,
                code_review_loop=loop,
                code_review_report=report,
                code_control_history=state.get("code_control_history", []),
                remaining_budget=_remaining_budget(state, planning),
            )
            decision = result.output
            controller_rounds = result.model_rounds_used
            used_fallback = result.used_fallback

        history = list(state.get("code_control_history", []))
        history.append(
            {
                "session_id": session_id,
                "attempt_id": loop.candidate.attempt_id,
                "candidate_revision": loop.candidate.candidate_revision,
                "reviewer_verdict": report.verdict,
                "reviewer_recommendation": report.recommended_action,
                "scheduler_decision": decision.model_dump(mode="json"),
                "used_fallback": used_fallback,
                "model_rounds_used": controller_rounds,
                "decided_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        await _emit_progress(
            config,
            ProgressEvent(
                stage="STEP_STARTED",
                message=(
                    f"第{current_step.step_id}步的代码验收需要恢复决策；"
                    f"Scheduler已选择{decision.action}。"
                ),
                step_id=current_step.step_id,
                status=f"CONTROL_{control_rounds + 1}",
            ),
        )

        return Command(
            update={
                "current_code_scheduler_decision": decision.model_dump(
                    mode="json"
                ),
                "current_code_control_rounds": control_rounds + 1,
                "code_control_history": history,
                "model_rounds_used": (
                    state.get("model_rounds_used", 0) + controller_rounds
                ),
                "current_step_model_rounds": (
                    state.get("current_step_model_rounds", 0)
                    + controller_rounds
                ),
                "current_step_stop_reason": (
                    f"Scheduler chose {decision.action}: {decision.reason}"
                ),
                "retry_current_step": True,
            },
            goto="step_executor",
        )

    async def finish_step(
        state: PlanningState,
        config: RunnableConfig,
        *,
        general: bool = False,
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

        worker_group_id = str(
            state.get("current_worker_group_id") or ""
        ).strip()
        review_id = (
            f"review:{state.get('planning_run_id', 'planning')}:"
            f"step:{current_step.step_id}:"
            f"execution:{state.get('current_step_attempt', 1)}"
        )
        persisted_report: StepReport | None = None
        if worker_group_id:
            claim = await worker_group_coordinator.claim_reporter(
                group_id=worker_group_id,
                review_id=review_id,
                changed_at=datetime.now(timezone.utc),
            )
            if claim.record.report is not None:
                persisted_report = StepReport.model_validate(
                    claim.record.report
                )
            elif not claim.owned_by_caller:
                raise RuntimeError(
                    "Parallel Step Reporter is already owned by another review."
                )

        review_packet = build_step_review_packet(
            user_request=_context(state).user_request,
            plan_objective=state["plan_objective"],
            current_step=current_step,
            replaced_attempts=state.get(
                "current_step_replacement_history",
                [],
            ),
            current_attempt=state.get(
                "current_step_trace",
                {},
            ),
            worker_attempts=(
                state.get("current_step_worker_traces")
                if worker_group_id
                else None
            ),
            previous_step_report=previous,
            stop_reason=state.get(
                "current_step_stop_reason",
                "General Worker正常结束。",
            ),
            remaining_budget=_remaining_budget(
                state,
                planning,
            ),
        )

        reporter_context = _context(state)
        reporter_snapshot_key = f"step_reporter:{current_step.step_id}:{state.get('current_step_attempt', 0)}"
        from reporting.general_gate import general_review_reasons
        review_reasons = general_review_reasons(current_step, state.get('current_step_trace', {}), review_packet,
                                               state.get('current_step_stop_reason','')) if current_step.worker_kind=='GENERAL' else ['other_role']
        if persisted_report is not None:
            report = persisted_report
            reporter_rounds = 0

        elif current_step.worker_kind=='GENERAL' and not review_reasons and not worker_group_id:
            # Preserve the worker's checked result; no additional LLM review.
            report = _build_general_step_report(current_step,state.get('current_step_trace',{}),review_packet)
            reporter_rounds = 0

        elif (
            current_step.worker_kind == "CODE"
            and state.get("current_step_trace", {}).get(
                "code_review_report"
            )
        ):
            # The Code Reviewer already performed the independent model-based
            # verification and the deterministic Publisher already committed
            # the approved manifest.  Translate that result; do not pay for a
            # second generic Reporter to repeat the same judgment.
            report = _build_code_step_report(
                current_step,
                state.get("current_step_trace", {}),
            )
            reporter_rounds = 0

        elif limit <= 0:
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
                    role_model("web_reporter" if current_step.worker_kind == "WEB" else "reporter", simple_model),

                    current_step=(
                        current_step
                    ),

                    review_packet=review_packet,
                    skill_catalog=_context(state).skill_catalog,
                    skill_mode=_context(state).skill_mode,
                    skill_fixed_ids=_context(state).skill_fixed_ids.get("step_reporter", []),
                    saved_skill_snapshot=reporter_context.role_skill_snapshots.get(reporter_snapshot_key),

                    max_model_rounds=(
                        limit
                    ),
                    model_output_max_tokens=(
                        reporter_output_limits.get(
                            "web_reporter" if current_step.worker_kind == "WEB" else "reporter",
                            model_output_max_tokens,
                        )
                    ),
                )
            )

            report = result.report
            if result.skill_snapshot is not None:
                reporter_context = reporter_context.model_copy(update={
                    "role_skill_snapshots": {**reporter_context.role_skill_snapshots,
                                              reporter_snapshot_key: result.skill_snapshot},
                })

            reporter_rounds = (
                result.model_rounds_used
            )

        publication_receipts: list[ArtifactPublicationReceipt] = []
        if current_step.worker_kind != "CODE":
            run_id = str(
                state.get("event_id")
                or state.get("planning_run_id")
                or "planning"
            ).strip()
            report, publication_receipts = _publish_step_handoff(
                report=report,
                packet=review_packet,
                run_id=run_id,
                run_storage_root=run_storage_root,
            )
        else:
            publication_receipts = [
                ArtifactPublicationReceipt.model_validate(item)
                for item in (
                    state.get("current_step_trace", {}).get(
                        "code_handoff_publication_receipts"
                    )
                    or []
                )
            ]

        # Preserve accepted worker knowledge verbatim across the review boundary.
        # Review verdicts remain separate; this is not a certification of each fact.
        from handoff_knowledge import (
            collect_api_handoff_receipts,
            collect_api_handoffs,
            collect_handoff_knowledge,
        )
        handoff_traces = state.get("current_step_worker_traces") or [state.get("current_step_trace", {})]
        if persisted_report is None:
            report = report.model_copy(update={
                "handoff_knowledge": collect_handoff_knowledge(handoff_traces),
                "handoff_apis": collect_api_handoffs(handoff_traces),
                "handoff_api_receipts": collect_api_handoff_receipts(handoff_traces),
            })

        if worker_group_id and persisted_report is None:
            await worker_group_coordinator.complete_reporter(
                group_id=worker_group_id,
                review_id=review_id,
                report=report.model_dump(mode="json"),
                changed_at=datetime.now(timezone.utc),
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
            "context": reporter_context,

            "handoff_publication_receipts": _merge_publication_receipts(
                state.get("handoff_publication_receipts", []),
                publication_receipts,
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

        raw_code_report = state.get("current_step_trace", {}).get(
            "code_review_report"
        )
        raw_code_scheduler_decision = state.get("current_step_trace", {}).get(
            "code_scheduler_decision_applied"
        )
        code_effective_action = (
            SchedulerCodeDecision.model_validate(
                raw_code_scheduler_decision
            ).action
            if raw_code_scheduler_decision
            else (
                CodeReviewReport.model_validate(
                    raw_code_report
                ).recommended_action
                if raw_code_report
                else None
            )
        )
        can_retry = (
            not budget_exhausted
            # Web review is deliberately one-way: the Reporter compresses and
            # assesses the bounded evidence packet, then hands control back to
            # the Scheduler.  It never creates an automatic Web repair loop.
            # CODE keeps its specialized repair/escalation machinery.
            and current_step.worker_kind == "CODE"

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

            # Reviewer STOP is a deliberate terminal recommendation.  A
            # generic retry would silently override the CODE control plane.
            and code_effective_action != "STOP"
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

        if state.get("final_worker_role_review_pending", False):
            update["final_worker_role_review_pending"] = False
            update["overall_stop_reason"] = "final_worker_role_review_completed"
            return Command(update=update, goto="final_reviewer")

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
                has_worker_plan_challenge = any(
                    attempt.plan_challenge is not None
                    for attempt in review_packet.attempts
                ) or bool(
                    current_step.worker_kind == "CODE"
                    and raw_code_report
                    and CodeReviewReport.model_validate(raw_code_report).confirmed_plan_challenge
                )
                update.update(
                    {
                        "pending_replan_reason": (
                            report.replan_reason
                            or report.summary
                        ),

                        "pending_replan_source": (
                            "worker_plan_challenge"
                            if has_worker_plan_challenge
                            else "step_reporter"
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

    async def general_report_node(state: PlanningState, config: RunnableConfig):
        return await finish_step(state, config, general=True)

    async def step_reporter_node(state: PlanningState, config: RunnableConfig):
        # Also handles old checkpoints paused at this node before migration.
        step = state.get("current_step")
        return await finish_step(state, config, general=step is not None and step.worker_kind == "GENERAL")

    async def replanner_node(
        state: PlanningState,
        config: RunnableConfig,
    ) -> Command[
        Literal[
            "step_executor",
            "final_reviewer",
        ]
    ]:
        """在剩余全局预算内执行Hard Replan。"""

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

        challenge_flow = bool(
            state.get("pending_replan_source") == "worker_plan_challenge"
            and state.get("current_step") is not None
        )
        max_steps = min(
            planning.max_steps_per_plan,

            _remaining_step_capacity(state, planning) + (1 if challenge_flow else 0),
        )

        next_step_id = (
            state["current_step"].step_id
            if challenge_flow
            else _next_step_id(state)
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
                role_model("replanner", hard_model),

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
                    [state["current_step"], *state.get("remaining_steps", [])]
                    if challenge_flow
                    else state.get("remaining_steps", [])
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

        if challenge_flow and decision.action == "RETURN_TO_WORKER":
            current_step = state["current_step"]
            resume_same_worker = (
                current_step.worker_kind in {"GENERAL", "WEB", "CODE"}
                and current_step.execution_mode == "SINGLE"
            )
            instruction = decision.worker_instruction or decision.reason
            return Command(
                update={
                    "replans_used": state.get("replans_used", 0) + 1,
                    "replan_history": history,
                    "pending_replan_reason": "",
                    "pending_replan_source": "",
                    "model_rounds_used": state.get("model_rounds_used", 0) + result.model_rounds_used,
                    "phase_tool_calls_used": 0,
                    "retry_current_step": True,
                    "plan_challenge_resume_active": resume_same_worker,
                    "plan_challenge_scheduler_instruction": instruction,
                    "current_step_leadership_override": (
                        "Scheduler驳回计划异议：" + instruction
                    ),
                    # Code Reviewer在上送计划异议时会冻结当前执行。
                    # Scheduler驳回后沿用原线程与运行会话，但不能重放STOP。
                    "current_code_scheduler_decision": None,
                    "current_code_control_rounds": 0,
                    "code_control_history": [],
                    "overall_stop_reason": "scheduler_rejected_worker_plan_challenge",
                },
                goto="step_executor",
            )

        if challenge_flow and decision.action == "CONTINUE" and new_steps:
            previous_step = state["current_step"]
            revised_step = new_steps[0]
            resume_same_worker = (
                previous_step.worker_kind == revised_step.worker_kind
                and revised_step.worker_kind in {"GENERAL", "WEB", "CODE"}
                and previous_step.execution_mode == "SINGLE"
                and revised_step.execution_mode == "SINGLE"
            )
            instruction = (
                "Scheduler接受计划异议并替换了当前及后续步骤。"
                f"裁决依据：{decision.reason}"
            )
            return Command(
                update={
                    "remaining_steps": new_steps[1:],
                    "planned_steps": [*state.get("planned_steps", []), *new_steps],
                    "current_step": revised_step,
                    "current_step_trace": {},
                    "current_step_worker_traces": [],
                    "current_worker_group_id": "",
                    "current_step_stop_reason": "",
                    "current_step_executor_rounds": 0,
                    "current_step_report_rounds": 0,
                    "current_step_model_rounds": 0,
                    "current_step_tool_calls": 0,
                    "current_step_show_all_toolsets_calls": 0,
                    "current_step_replacement_history": [],
                    "current_code_runtime_session_id": "" if not resume_same_worker else state.get("current_code_runtime_session_id", ""),
                    "current_code_scheduler_decision": None,
                    "current_code_control_rounds": 0,
                    "code_control_history": [],
                    "code_superseded_attempt_records": [],
                    "retry_current_step": True,
                    "plan_challenge_resume_active": resume_same_worker,
                    "plan_challenge_scheduler_instruction": instruction,
                    "current_step_leadership_override": (
                        "" if resume_same_worker else instruction
                    ),
                    "replans_used": state.get("replans_used", 0) + 1,
                    "replan_history": history,
                    "pending_replan_reason": "",
                    "pending_replan_source": "",
                    "model_rounds_used": state.get("model_rounds_used", 0) + result.model_rounds_used,
                    "phase_tool_calls_used": 0,
                    "overall_stop_reason": "scheduler_accepted_worker_plan_challenge",
                },
                goto="step_executor",
            )

        update: dict[
            str,
            Any,
        ] = {
            "remaining_steps": (
                new_steps
            ),

            "planned_steps": [
                *state.get("planned_steps", []),
                *new_steps,
            ],

            "current_step": None,

            "current_step_attempt": 0,

            "current_step_executor_rounds": 0,

            "current_step_report_rounds": 0,

            "current_step_model_rounds": 0,

            "current_step_tool_calls": 0,

            "current_step_show_all_toolsets_calls": 0,

            "current_step_trace": {},

            "current_step_worker_traces": [],

            "current_worker_group_id": "",

            "current_step_leadership_override": "",

            "current_step_replacement_history": [],

            "current_code_runtime_session_id": "",

            "current_code_scheduler_decision": None,

            "current_code_control_rounds": 0,

            "code_control_history": [],

            "code_superseded_attempt_records": [],

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

            "plan_challenge_resume_active": False,

            "plan_challenge_scheduler_instruction": "",

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
    ) -> Command[Literal["step_executor", "replanner", "__end__"]]:
        """最终审核：优先交回最后Worker，三轮后才允许Scheduler重规划。"""

        max_final_reviews = planning.max_final_worker_repair_rounds + 1
        final_review_rounds = state.get("final_review_rounds", 0)
        if final_review_rounds >= max_final_reviews:
            decision = _build_forced_final_decision(state)
            return Command(
                update={
                    "final_status": decision.status or "FAILED",
                    "final_answer": decision.final_answer or "",
                    "unmet_success_criteria": decision.unmet_success_criteria,
                    "overall_stop_reason": "final_review_round_limit",
                },
                goto=END,
            )

        context = _context(state)
        skill_rounds = 0
        if "final_reviewer" not in context.role_skill_snapshots:
            catalog = context.skill_catalog
            if catalog is None:
                catalog = load_catalog()
            appworld = any(
                str(item.get("name", "")).upper() == "APPWORLD"
                for item in context.toolset_catalog
                if isinstance(item, dict)
            )
            snapshot = await prepare_skills(
                role_model("skill_selector", hard_model),
                role="final_reviewer",
                task={
                    "user_request": context.user_request,
                    "scope_contract": (
                        context.scope_contract.model_dump(mode="json")
                        if context.scope_contract is not None else None
                    ),
                    "plan_objective": state.get("plan_objective", ""),
                    "success_criteria": state.get("plan_success_criteria", []),
                    "review_type": "AppWorld" if appworld else "general",
                },
                catalog=catalog,
                config=config,
                mode=context.skill_mode,
                fixed_ids=context.skill_fixed_ids.get("final_reviewer", []),
                topics=("appworld", "verification") if appworld else ("verification",),
                tools=(),
                allow_model=True,
            )
            context = context.model_copy(update={
                "skill_catalog": catalog,
                "role_skill_snapshots": {
                    **context.role_skill_snapshots,
                    "final_reviewer": snapshot.model_dump(mode="json"),
                },
            })
            skill_rounds = snapshot.model_calls

        repair_round = state.get("final_worker_repair_round", 0)
        latest_worker_evidence = _final_worker_evidence(state)
        return_available = bool(
            latest_worker_evidence
            and state.get("current_step") is not None
            and state["current_step"].execution_mode == "SINGLE"
            and repair_round < planning.max_final_worker_repair_rounds
        )
        can_replan = _replan_available(state, planning, from_reviewer=True)

        await _emit_progress(
            config,
            ProgressEvent(
                stage="FINAL_REVIEW_STARTED",
                message=(
                    "正在复核最后一次返修结果。"
                    if repair_round
                    else "正在汇总已确认结果并生成最终回答。"
                ),
                status=f"ROUND_{final_review_rounds + 1}",
            ),
        )

        result = await run_hard_final_reviewer(
            role_model("final_reviewer", hard_model),
            context=context,
            plan_objective=state.get("plan_objective", context.user_request),
            plan_success_criteria=state.get("plan_success_criteria", []),
            step_reports=state.get("completed_step_reports", []),
            replan_history=state.get("replan_history", []),
            overall_stop_reason=state.get(
                "overall_stop_reason", "正常进入最终审核。"
            ),
            replan_available=can_replan,
            plan_steps=state.get("planned_steps", []),
            latest_worker_evidence=latest_worker_evidence,
            repair_history=state.get("final_worker_repair_history", []),
            repair_round=repair_round,
            max_repair_rounds=(
                planning.max_final_worker_repair_rounds
                if return_available else repair_round
            ),
        )

        reviewer_rounds = result.model_rounds_used + skill_rounds
        model_rounds_used = state.get("model_rounds_used", 0) + reviewer_rounds
        final_repair_model_rounds = state.get("final_repair_model_rounds_used", 0)
        if repair_round:
            final_repair_model_rounds += result.model_rounds_used

        decision = result.output
        current_step = state.get("current_step")
        if decision.action == "RETURN_TO_WORKER":
            request = decision.repair_request
            request_matches = bool(
                return_available
                and request is not None
                and current_step is not None
                and request.step_id == current_step.step_id
                and request.worker_kind == current_step.worker_kind
            )
            if request_matches:
                return Command(
                    update={
                        "context": context,
                        "model_rounds_used": model_rounds_used,
                        "final_repair_model_rounds_used": final_repair_model_rounds,
                        "final_review_rounds": final_review_rounds + 1,
                        "final_worker_repair_round": repair_round + 1,
                        "final_worker_repair_request": request.model_dump(mode="json"),
                        "final_worker_repair_active": True,
                        "retry_current_step": True,
                        "overall_stop_reason": "final_reviewer_returned_to_worker",
                    },
                    goto="step_executor",
                )
            # A stale or invented target must not be executed. Escalate only
            # through the normal Scheduler boundary when budget allows.
            if can_replan:
                decision = FinalReviewDecision(
                    review_reason=decision.review_reason,
                    criterion_reviews=decision.criterion_reviews,
                    repair_request=None,
                    replan_reason=(
                        "Final Reviewer返修单没有绑定当前最后Worker；"
                        "保留现有证据并由Scheduler重新判断执行路径。"
                    ),
                    action="REPLAN",
                    status=None,
                    final_answer=None,
                    unmet_success_criteria=[],
                )
            else:
                decision = _build_forced_final_decision(state)

        if decision.action == "REPLAN":
            current: PlanningState = dict(state)
            current["model_rounds_used"] = model_rounds_used
            if can_replan and _replan_available(
                current, planning, from_reviewer=False
            ):
                return Command(
                    update={
                        "context": context,
                        "model_rounds_used": model_rounds_used,
                        "final_repair_model_rounds_used": final_repair_model_rounds,
                        "final_review_rounds": final_review_rounds + 1,
                        "pending_replan_reason": decision.replan_reason or "",
                        "pending_replan_source": "final_reviewer",
                        "overall_stop_reason": "reviewer_requested_replan",
                    },
                    goto="replanner",
                )
            decision = _build_forced_final_decision(current)

        return Command(
            update={
                "context": context,
                "model_rounds_used": model_rounds_used,
                "final_repair_model_rounds_used": final_repair_model_rounds,
                "final_review_rounds": final_review_rounds + 1,
                "scheduler_final_decision": result.output.action == "FINAL",
                "final_status": decision.status or "FAILED",
                "final_answer": decision.final_answer or "",
                "unmet_success_criteria": decision.unmet_success_criteria,
                "overall_stop_reason": state.get(
                    "overall_stop_reason", "final_reviewer_completed"
                ),
            },
            goto=END,
        )

    workflow = StateGraph(
        PlanningState
    )
    workflow.add_node("prepare_scheduler", scheduler_node(prepare_scheduler_node, planning.scheduler_summary_trigger_tokens))
    workflow.add_edge("prepare_scheduler", "supervisor")

    workflow.add_node(
        "supervisor",
        scheduler_node(supervisor_node, planning.scheduler_summary_trigger_tokens),
    )

    workflow.add_node(
        "step_executor",
        scheduler_node(step_executor_node, planning.scheduler_summary_trigger_tokens),
    )

    workflow.add_node(
        "code_controller",
        scheduler_node(code_controller_node, planning.scheduler_summary_trigger_tokens),
    )

    workflow.add_node(
        "step_reporter",
        scheduler_node(step_reporter_node, planning.scheduler_summary_trigger_tokens),
    )
    workflow.add_node("general_report", scheduler_node(general_report_node, planning.scheduler_summary_trigger_tokens))

    workflow.add_node(
        "replanner",
        scheduler_node(replanner_node, planning.scheduler_summary_trigger_tokens),
    )

    workflow.add_node(
        "final_reviewer",
        scheduler_node(final_reviewer_node, planning.scheduler_summary_trigger_tokens),
    )

    workflow.add_edge(
        START,
        "prepare_scheduler",
    )

    recursion_limit = (
            8
            + planning.max_total_steps
            * planning.max_step_attempts
            * 4
            + planning.max_final_worker_repair_rounds * 4
    )

    return (
        workflow
        .compile(checkpointer=checkpointer)
        .with_config(
            {
                "recursion_limit": (
                    recursion_limit
                )
            }
        )
    )


def _build_code_step_report(
    current_step: PlanStep,
    trace: dict[str, Any],
) -> StepReport:
    """Translate the independent CODE review into Scheduler vocabulary.

    This is an adapter, not another review.  It performs no model call and
    makes no new quality judgment: PASSED is accepted only when the persisted
    loop is already APPLIED by the Reviewer-owned Publisher transaction.
    """

    report = CodeReviewReport.model_validate(trace.get("code_review_report"))
    loop = CodeReviewLoopState.model_validate(trace.get("code_review_loop"))
    if report.candidate != loop.candidate:
        raise ValueError("CODE report and review loop reference different candidates")
    if report.candidate.step_id != current_step.step_id:
        raise ValueError("CODE report references another Planning Step")

    applied = report.verdict == "PASSED" and loop.status == "APPLIED"
    if report.verdict == "PASSED" and not applied:
        raise ValueError("PASSED CODE report requires an APPLIED review loop")

    check_evidence = [
        f"{check.check_id}: {check.status} - {check.summary}"
        for check in report.check_results
    ]
    publication_evidence = []
    if report.publication_id:
        publication_evidence.append(
            f"publication_id={report.publication_id}"
        )
    if report.applied_revision:
        publication_evidence.append(
            f"applied_revision={report.applied_revision}"
        )
    evidence = [
        *report.evidence_refs,
        *check_evidence,
        *publication_evidence,
    ]
    raw_integration_commit = trace.get("code_integration_commit")
    integration_commit = (
        IntegrationCommitReceipt.model_validate(raw_integration_commit)
        if raw_integration_commit
        else None
    )
    if integration_commit is not None:
        evidence.append(
            f"integration_commit={integration_commit.accepted_commit}"
        )

    raw_scheduler_decision = trace.get("code_scheduler_decision_applied")
    scheduler_decision = (
        SchedulerCodeDecision.model_validate(raw_scheduler_decision)
        if raw_scheduler_decision
        else None
    )
    effective_action = (
        scheduler_decision.action
        if scheduler_decision is not None
        else report.recommended_action
    )

    if applied:
        status = "COMPLETED"
        criterion_status = "MET"
        unresolved: list[str] = []
        errors: list[str] = []
        request_replan = False
        replan_reason = None
        stop_reason = "Code Reviewer verified and published the candidate."
    else:
        status = "BLOCKED" if report.verdict == "ESCALATED" else "FAILED"
        criterion_status = "UNKNOWN"
        unresolved = list(current_step.success_criteria)
        errors = [*report.failed_test_summaries]
        if not errors:
            errors = [report.verification_summary]
        # RESTART is now executed by Code Controller before this adapter is
        # reached.  A terminal STOP must not accidentally fall back to the old
        # generic retry/replan path because the Reviewer had recommended a
        # different action one state transition earlier.
        request_replan = report.confirmed_plan_challenge is not None
        replan_reason = (
            report.confirmed_plan_challenge.reason
            if report.confirmed_plan_challenge is not None
            else None
        )
        stop_reason = (
            "Code Reviewer escalated the candidate and Scheduler applied "
            f"{effective_action}."
        )
        if scheduler_decision is not None:
            evidence.append(
                "scheduler_decision="
                f"{scheduler_decision.action}: {scheduler_decision.reason}"
            )

    raw_handoff_receipts = trace.get("code_handoff_publication_receipts") or []
    if raw_handoff_receipts:
        artifacts = [
            StepArtifactReport(
                path=ArtifactPublicationReceipt.model_validate(item).handoff_path,
                description=(
                    "Code Reviewer approved this file and the Harness "
                    "published it to the run handoff workspace."
                ),
            )
            for item in raw_handoff_receipts
        ]
    else:
        artifacts = [
            StepArtifactReport(
                path=path,
                description=(
                    "Reviewer-approved artifact published to "
                    f"{report.delivery_location}."
                ),
            )
            for path in report.published_artifact_paths
        ]
    completed_work = [
        f"{item.path}: {item.change_summary}"
        for item in report.changed_files
    ]
    if integration_commit is not None:
        completed_work.append(
            "Accepted integration commit: "
            f"{integration_commit.accepted_commit}"
        )
    confirmed_results = (
        [report.summary, report.verification_summary, *check_evidence]
        if applied
        else []
    )
    worker_id = str(trace.get("worker_id") or "code-worker")

    from handoff_knowledge import (
        collect_api_handoff_receipts,
        collect_api_handoffs,
        collect_handoff_knowledge,
    )
    return StepReport(
        step_id=current_step.step_id,
        handoff_knowledge=collect_handoff_knowledge([trace]),
        handoff_apis=collect_api_handoffs([trace]),
        handoff_api_receipts=collect_api_handoff_receipts([trace]),
        status=status,
        summary=report.summary,
        stop_reason=stop_reason,
        criterion_results=[
            StepCriterionResult(
                criterion=criterion,
                status=criterion_status,
                evidence=evidence if applied else [],
            )
            for criterion in current_step.success_criteria
        ],
        confirmed_results=confirmed_results,
        completed_work=completed_work,
        artifacts=artifacts,
        worker_contributions=[
            WorkerContribution(
                worker_id=worker_id,
                contribution=report.verification_summary,
            )
        ],
        evidence=evidence,
        errors=errors,
        unresolved_items=unresolved,
        next_action=(
            None
            if applied
            else f"Scheduler action: {effective_action}."
        ),
        request_replan=request_replan,
        replan_reason=replan_reason,
    )
