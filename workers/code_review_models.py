"""Persistent contracts for the Code Worker and Code Reviewer repair loop.

This module is intentionally model- and Docker-agnostic. It defines the
messages and legal state transitions first; later runtime adapters can persist
the state in LangGraph checkpoints and execute either side in containers.
"""

from __future__ import annotations
from runtime_tracing import operation

from typing import Literal

from pydantic.json_schema import SkipJsonSchema
from api_handoff import ApiHandoffList, api_field
from handoff_knowledge import HandoffKnowledge, knowledge_field
from pydantic import BaseModel, ConfigDict, Field, model_validator
from workers.plan_challenge import PlanChallenge


CodeReviewLoopStatus = Literal[
    "REVIEWING",
    "WAITING_FOR_WORKER",
    "PUBLISHED_PENDING_REPORT",
    "APPLIED",
    "ESCALATED_TO_SCHEDULER",
    "SUPERSEDED",
    "STOPPED",
]

CodeRepairResponseAction = Literal[
    "REVISION_READY",
    "SUBMISSION_UPDATED",
    "TEST_DISPUTE",
    "BLOCKED",
    "REQUEST_SCOPE_CHANGE",
]

CodeFindingCategory = Literal[
    "CANDIDATE_DEFECT",
    "TEST_DEFECT",
    "ENVIRONMENT",
    "REQUIREMENT_AMBIGUITY",
    "DELIVERY_MANIFEST",
]

CodeCheckStatus = Literal[
    "PASSED",
    "FAILED",
    "ERROR",
]

CodeReviewVerdict = Literal[
    "PASSED",
    "FAILED",
    "ESCALATED",
]

SchedulerCodeAction = Literal[
    "CONTINUE",
    "RESTART",
    "STOP",
]

CodeContinuationAction = Literal[
    "REVISION_READY",
    "SUBMISSION_UPDATED",
    "BLOCKED",
    "REQUEST_SCOPE_CHANGE",
]


class CodeReviewModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class CodeCandidateRef(CodeReviewModel):
    """Identity fence preventing the Reviewer from testing stale output."""

    event_id: str = Field(min_length=1)
    step_id: int = Field(ge=1)
    attempt_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    candidate_revision: int = Field(ge=1)


class CodeImplementedInterface(CodeReviewModel):
    interface_id: str = Field(min_length=1)
    actual_contract: dict = Field(default_factory=dict)
    implementation_location: str = Field(min_length=1)


class CodeChangedFile(CodeReviewModel):
    path: str = Field(min_length=1)
    change_summary: str = Field(min_length=1)


class CodeSelfCheck(CodeReviewModel):
    check: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    outcome: Literal["PASSED", "FAILED", "NOT_RUN"]


class CodeWorkerSubmission(CodeReviewModel):
    """The bounded implementation manifest; not the Worker''s hidden trace."""

    handoff_apis: ApiHandoffList = api_field()
    handoff_knowledge: SkipJsonSchema[list[HandoffKnowledge]] = knowledge_field()
    plan_challenge: PlanChallenge | None = Field(
        default=None,
        description=(
            "Only for a material conflict between the user request and the frozen code task; "
            "ordinary implementation difficulty or an unknown API is not a plan challenge."
        ),
    )


    candidate: CodeCandidateRef
    summary: str = Field(min_length=1)
    requirement_status: dict[str, Literal["MET", "PARTIAL", "NOT_MET"]]
    implemented_interfaces: tuple[CodeImplementedInterface, ...] = ()
    changed_files: tuple[CodeChangedFile, ...] = ()
    proposed_artifact_paths: tuple[str, ...] = ()
    self_checks: tuple[CodeSelfCheck, ...] = ()
    evidence_tool_call_ids: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


class CodeReviewFinding(CodeReviewModel):
    finding_id: str = Field(min_length=1)
    category: CodeFindingCategory
    summary: str = Field(min_length=1)
    affected_requirement_ids: tuple[str, ...] = ()
    related_paths: tuple[str, ...] = ()


class CodeRepairInstruction(CodeReviewModel):
    round_no: int = Field(ge=1)
    candidate: CodeCandidateRef
    summary: str = Field(min_length=1)
    required_changes: tuple[str, ...] = Field(min_length=1)
    preserve_behaviors: tuple[str, ...] = ()
    findings: tuple[CodeReviewFinding, ...] = Field(min_length=1)


class CodeWorkerRepairResponse(CodeReviewModel):
    round_no: int = Field(ge=1)
    reasons: tuple[str, ...] = ()
    action: CodeRepairResponseAction
    candidate: CodeCandidateRef
    summary: str = Field(min_length=1)
    changed_files: tuple[CodeChangedFile, ...] = ()
    evidence_tool_call_ids: tuple[str, ...] = ()


class CodeRepairExchange(CodeReviewModel):
    instruction: CodeRepairInstruction
    response: CodeWorkerRepairResponse


class CodeCheckResult(CodeReviewModel):
    check_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    status: CodeCheckStatus


class CodeReviewReport(CodeReviewModel):
    """Compressed CODE-specific StepReport consumed by the Scheduler."""

    handoff_apis: ApiHandoffList = api_field()
    handoff_knowledge: SkipJsonSchema[list[HandoffKnowledge]] = knowledge_field()
    confirmed_plan_challenge: PlanChallenge | None = Field(
        default=None,
        description=(
            "Independent confirmation of a Worker plan challenge. If present, verdict must be "
            "ESCALATED and recommended_action STOP so the Planning Graph can replan."
        ),
    )


    candidate: CodeCandidateRef
    summary: str = Field(min_length=1)
    verification_summary: str = Field(min_length=1, description="Actual independent checks and observed results; separate verified work, missing evidence, not attempted and actual failure. Identify scope violations without hiding completed changes.")
    verified_requirement_ids: tuple[str, ...] = Field(default=(), description="Unique IDs copied from frozen code_task.requirements that were actually verified. PASSED covers every requirement.")
    verified_interfaces: tuple[CodeImplementedInterface, ...] = ()
    check_results: tuple[CodeCheckResult, ...] = ()
    failed_test_summaries: tuple[str, ...] = Field(default=(), description="Observed failing check/error, affected requirement, impact and remaining prerequisite. Missing checks are unverified, not fabricated tool failures.")
    changed_files: tuple[CodeChangedFile, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    verdict: CodeReviewVerdict = Field(description="填写实际检查、失败摘要和evidence_refs后再作出整体结论。")
    approved_artifact_paths: tuple[str, ...] = ()
    published_artifact_paths: tuple[str, ...] = ()
    delivery_location: str | None = None
    publication_id: str | None = None
    applied_revision: str | None = None
    recommended_action: SchedulerCodeAction | None = None

    @model_validator(mode="after")
    def validate_verdict_action(self) -> "CodeReviewReport":
        if self.confirmed_plan_challenge is not None and (
            self.verdict != "ESCALATED" or self.recommended_action != "STOP"
        ):
            raise ValueError(
                "confirmed_plan_challenge要求verdict=ESCALATED且recommended_action=STOP。"
            )
        publication_fields = (
            self.delivery_location,
            self.publication_id,
            self.applied_revision,
        )
        if self.verdict == "PASSED":
            if self.recommended_action is not None:
                raise ValueError("已经APPLIED的报告不再建议Scheduler动作。")
            if any(value is None for value in publication_fields):
                raise ValueError("PASSED报告必须包含完整发布回执摘要。")
            if self.published_artifact_paths != self.approved_artifact_paths:
                raise ValueError("发布文件必须与Reviewer批准文件完全一致。")
            if not self.check_results:
                raise ValueError("PASSED报告必须逐项提供测试或检查结果。")
        else:
            if self.recommended_action is None:
                raise ValueError("未通过的报告必须建议Scheduler恢复动作。")
            if any(value is not None for value in publication_fields):
                raise ValueError("未通过的报告不能声称已经发布。")
            if self.published_artifact_paths:
                raise ValueError("未通过的报告不能包含已发布文件。")
        return self


class SchedulerCodeDecision(CodeReviewModel):
    reason: str = Field(min_length=1, description="先说明审核事实与选择依据，再填写action。")
    action: SchedulerCodeAction
    worker_instruction: str | None = None
    reviewer_instruction: str | None = None
    repair_rounds: int | None = Field(default=None, ge=1, le=3)

    @model_validator(mode="after")
    def validate_action_fields(self) -> "SchedulerCodeDecision":
        continuation_fields = (
            self.worker_instruction,
            self.reviewer_instruction,
            self.repair_rounds,
        )
        if self.action == "CONTINUE":
            if not self.worker_instruction or not self.reviewer_instruction:
                raise ValueError(
                    "CONTINUE必须同时给Worker和Reviewer新指令。"
                )
            if self.repair_rounds is None:
                raise ValueError("CONTINUE必须提供新的repair_rounds。")
        elif any(value is not None for value in continuation_fields):
            raise ValueError(
                "只有CONTINUE可以提供继续执行字段。"
            )
        return self


class SchedulerContinueDirective(CodeReviewModel):
    """One persisted Scheduler instruction for the existing Agent pair."""

    scheduler_epoch: int = Field(ge=2)
    candidate: CodeCandidateRef
    reason: str = Field(min_length=1)
    worker_instruction: str = Field(min_length=1)
    reviewer_instruction: str = Field(min_length=1)
    repair_rounds: int = Field(ge=1, le=3)


class CodeContinuationSubmission(CodeReviewModel):
    """Worker result sent to Reviewer after Scheduler continuation guidance."""

    scheduler_epoch: int = Field(ge=2)
    reasons: tuple[str, ...] = ()
    action: CodeContinuationAction
    candidate: CodeCandidateRef
    summary: str = Field(min_length=1)
    changed_files: tuple[CodeChangedFile, ...] = ()
    evidence_tool_call_ids: tuple[str, ...] = ()


class SchedulerContinuationRecord(CodeReviewModel):
    """Auditable directive/submission pair; it is not Scheduler dialogue."""

    directive: SchedulerContinueDirective
    submission: CodeContinuationSubmission


class CodeReviewLoopState(CodeReviewModel):
    """Checkpointable state for one Worker/Reviewer pair."""

    candidate: CodeCandidateRef
    worker_checkpoint_id: str = Field(min_length=1)
    reviewer_checkpoint_id: str = Field(min_length=1)
    status: CodeReviewLoopStatus = "REVIEWING"
    scheduler_epoch: int = Field(default=1, ge=1)
    repair_round: int = Field(default=0, ge=0)
    max_repair_rounds: int = Field(default=2, ge=1, le=3)
    pending_instruction: CodeRepairInstruction | None = None
    pending_scheduler_directive: SchedulerContinueDirective | None = None
    exchanges: tuple[CodeRepairExchange, ...] = ()
    scheduler_continuations: tuple[SchedulerContinuationRecord, ...] = ()
    terminal_summary: str | None = None

    @model_validator(mode="after")
    def validate_state_shape(self) -> "CodeReviewLoopState":
        if self.repair_round > self.max_repair_rounds:
            raise ValueError("repair_round不能超过max_repair_rounds。")
        waiting = self.status == "WAITING_FOR_WORKER"
        pending_count = sum(
            item is not None
            for item in (
                self.pending_instruction,
                self.pending_scheduler_directive,
            )
        )
        if waiting != (pending_count == 1):
            raise ValueError(
                "WAITING_FOR_WORKER必须且只能保存一种Worker指令。"
            )
        if self.pending_scheduler_directive is not None:
            if (
                self.pending_scheduler_directive.scheduler_epoch
                != self.scheduler_epoch
            ):
                raise ValueError("Scheduler指令epoch不是当前epoch。")
        if self.status in {
            "PUBLISHED_PENDING_REPORT",
            "APPLIED",
            "ESCALATED_TO_SCHEDULER",
            "SUPERSEDED",
            "STOPPED",
        } and self.status != "PUBLISHED_PENDING_REPORT" and not self.terminal_summary:
            raise ValueError("终态必须提供terminal_summary。")
        return self


@operation('Code Review / Open Review', fields=('candidate', 'max_repair_rounds'))
def create_code_review_loop(
    *,
    candidate: CodeCandidateRef,
    worker_checkpoint_id: str,
    reviewer_checkpoint_id: str,
    max_repair_rounds: int = 2,
) -> CodeReviewLoopState:
    return CodeReviewLoopState(
        candidate=candidate,
        worker_checkpoint_id=worker_checkpoint_id,
        reviewer_checkpoint_id=reviewer_checkpoint_id,
        max_repair_rounds=max_repair_rounds,
    )


@operation('Code Reviewer / Request Repair', fields=('summary', 'findings', 'required_changes'))
def request_code_repair(
    state: CodeReviewLoopState,
    *,
    summary: str,
    required_changes: tuple[str, ...],
    findings: tuple[CodeReviewFinding, ...],
    preserve_behaviors: tuple[str, ...] = (),
) -> CodeReviewLoopState:
    if state.status != "REVIEWING":
        raise ValueError("只有REVIEWING状态可以请求修复。")
    if state.repair_round >= state.max_repair_rounds:
        raise ValueError("本轮本地修复预算已经耗尽。")

    next_round = state.repair_round + 1
    instruction = CodeRepairInstruction(
        round_no=next_round,
        candidate=state.candidate,
        summary=summary,
        required_changes=required_changes,
        preserve_behaviors=preserve_behaviors,
        findings=findings,
    )
    return state.model_copy(
        update={
            "status": "WAITING_FOR_WORKER",
            "repair_round": next_round,
            "pending_instruction": instruction,
        }
    )


@operation('Code Worker / Repair Response', fields=('response',))
def receive_code_worker_response(
    state: CodeReviewLoopState,
    response: CodeWorkerRepairResponse,
) -> CodeReviewLoopState:
    if state.status != "WAITING_FOR_WORKER" or state.pending_instruction is None:
        raise ValueError("当前没有等待中的Code Worker修复请求。")
    if response.round_no != state.repair_round:
        raise ValueError("Worker响应的round_no与当前修复轮不一致。")

    old = state.candidate
    new = response.candidate
    same_attempt = (
        old.event_id,
        old.step_id,
        old.attempt_id,
        old.workspace_id,
    ) == (
        new.event_id,
        new.step_id,
        new.attempt_id,
        new.workspace_id,
    )
    if not same_attempt:
        raise ValueError("Worker响应不能悄悄切换attempt或workspace。")

    if response.action == "REVISION_READY":
        if new.candidate_revision <= old.candidate_revision:
            raise ValueError("REVISION_READY必须提高candidate_revision。")
        next_status: CodeReviewLoopStatus = "REVIEWING"
        terminal_summary = None
    elif response.action in {"SUBMISSION_UPDATED", "TEST_DISPUTE"}:
        if new.candidate_revision != old.candidate_revision:
            raise ValueError(
                f"{response.action}不能伪造新的candidate revision。"
            )
        next_status = "REVIEWING"
        terminal_summary = None
    else:
        if new.candidate_revision != old.candidate_revision:
            raise ValueError("未生成修订时不能伪造candidate revision。")
        next_status = "REVIEWING"
        terminal_summary = None

    exchange = CodeRepairExchange(
        instruction=state.pending_instruction,
        response=response,
    )
    return state.model_copy(
        update={
            "candidate": new,
            "status": next_status,
            "pending_instruction": None,
            "exchanges": (*state.exchanges, exchange),
            "terminal_summary": terminal_summary,
        }
    )


@operation('Code Worker / Continuation Submission', fields=('submission',))
def receive_code_continuation_submission(
    state: CodeReviewLoopState,
    submission: CodeContinuationSubmission,
) -> CodeReviewLoopState:
    """Route the same Worker's continuation result to independent review."""

    directive = state.pending_scheduler_directive
    if state.status != "WAITING_FOR_WORKER" or directive is None:
        raise ValueError("当前没有等待中的Scheduler CONTINUE指令。")
    if submission.scheduler_epoch != directive.scheduler_epoch:
        raise ValueError("Worker提交的scheduler_epoch已经过期。")

    old = state.candidate
    new = submission.candidate
    same_attempt = (
        old.event_id,
        old.step_id,
        old.attempt_id,
        old.workspace_id,
    ) == (
        new.event_id,
        new.step_id,
        new.attempt_id,
        new.workspace_id,
    )
    if not same_attempt:
        raise ValueError("CONTINUE响应不能切换attempt或workspace。")

    if submission.action == "REVISION_READY":
        if new.candidate_revision <= old.candidate_revision:
            raise ValueError("REVISION_READY必须提高candidate_revision。")
    else:
        if new.candidate_revision != old.candidate_revision:
            raise ValueError("未生成修订时不能伪造candidate revision。")

    record = SchedulerContinuationRecord(
        directive=directive,
        submission=submission,
    )
    return state.model_copy(
        update={
            "candidate": new,
            "status": "REVIEWING",
            "pending_scheduler_directive": None,
            "scheduler_continuations": (
                *state.scheduler_continuations,
                record,
            ),
            "terminal_summary": None,
        }
    )


@operation('Code Review / Record Publication', fields=('candidate',))
def record_code_publication(
    state: CodeReviewLoopState,
    *,
    candidate: CodeCandidateRef,
) -> CodeReviewLoopState:
    """Checkpoint a successful publish before the Reviewer writes its report."""

    if state.status != "REVIEWING":
        raise ValueError("只有REVIEWING状态可以发布candidate。")
    if candidate != state.candidate:
        raise ValueError("不能发布过期的candidate revision。")
    return state.model_copy(
        update={
            "status": "PUBLISHED_PENDING_REPORT",
            "terminal_summary": None,
        }
    )


@operation('Code Reviewer / Verdict', fields=('report',))
def finish_code_review(
    state: CodeReviewLoopState,
    report: CodeReviewReport,
) -> CodeReviewLoopState:
    if report.candidate != state.candidate:
        raise ValueError("Reviewer报告不是针对当前candidate revision。")

    if report.verdict == "PASSED":
        if state.status != "PUBLISHED_PENDING_REPORT":
            raise ValueError("Reviewer必须先成功发布才能提交PASSED报告。")
        status: CodeReviewLoopStatus = "APPLIED"
    else:
        if state.status != "REVIEWING":
            raise ValueError("只有REVIEWING状态可以提交未通过报告。")
        status = "ESCALATED_TO_SCHEDULER"

    return state.model_copy(
        update={
            "status": status,
            "terminal_summary": report.summary,
        }
    )


@operation('Code Runtime / Apply Scheduler Decision', fields=('decision',))
def apply_scheduler_code_decision(
    state: CodeReviewLoopState,
    decision: SchedulerCodeDecision,
) -> CodeReviewLoopState:
    if state.status != "ESCALATED_TO_SCHEDULER":
        raise ValueError("Scheduler只处理Reviewer未完成并上报的状态。")

    if decision.action == "CONTINUE":
        next_epoch = state.scheduler_epoch + 1
        directive = SchedulerContinueDirective(
            scheduler_epoch=next_epoch,
            candidate=state.candidate,
            reason=decision.reason,
            worker_instruction=decision.worker_instruction,
            reviewer_instruction=decision.reviewer_instruction,
            repair_rounds=decision.repair_rounds,
        )
        return state.model_copy(
            update={
                "status": "WAITING_FOR_WORKER",
                "scheduler_epoch": next_epoch,
                "repair_round": 0,
                "max_repair_rounds": decision.repair_rounds,
                "pending_instruction": None,
                "pending_scheduler_directive": directive,
                "terminal_summary": None,
            }
        )

    if decision.action == "RESTART":
        status: CodeReviewLoopStatus = "SUPERSEDED"
    elif decision.action == "STOP":
        status = "STOPPED"
    return state.model_copy(
        update={
            "status": status,
            "terminal_summary": decision.reason,
        }
    )
