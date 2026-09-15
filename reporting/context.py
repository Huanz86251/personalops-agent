"""Build a StepReviewPacket from Worker submissions, never raw messages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from planning_models import PlanStep, StepReport
from reporting.models import (
    ReviewAttempt,
    ReviewTaskContract,
    StepReviewPacket,
)
from workers.submission import (
    ResolvedToolEvidence,
    WorkerSubmissionRecord,
)
from worker_termination import WorkerCancellationRecord


EVIDENCE_RESULT_MAX_CHARS = 6000
EVIDENCE_TRUNCATION_MARKER = (
    "\n...[middle omitted by StepReviewPacket limit]...\n"
)
REVIEW_FALLBACK_TEXT_MAX_CHARS = 4000


def _bounded_result(value: str) -> str:
    if len(value) <= EVIDENCE_RESULT_MAX_CHARS:
        return value

    remaining = (
        EVIDENCE_RESULT_MAX_CHARS
        - len(EVIDENCE_TRUNCATION_MARKER)
    )
    head = remaining // 2
    tail = remaining - head
    return (
        value[:head]
        + EVIDENCE_TRUNCATION_MARKER
        + value[-tail:]
    )


def _bounded_evidence(
    evidence: Sequence[ResolvedToolEvidence],
) -> list[ResolvedToolEvidence]:
    return [
        item.model_copy(
            update={
                "result": _bounded_result(item.result),
            }
        )
        for item in evidence
    ]


def _bounded_text(value: Any, max_chars: int) -> str:
    normalized = str(value or "")
    if len(normalized) <= max_chars:
        return normalized
    marker = "\n...[truncated for durable review payload]...\n"
    remaining = max_chars - len(marker)
    head = remaining // 2
    return normalized[:head] + marker + normalized[-(remaining - head):]


def _tool_audit(trace):
    if "tool_audit" in trace:
        return trace["tool_audit"], bool(trace.get("tool_audit_available", False))
    if "messages" not in trace:
        return [], False
    rows = []
    by_id = {}
    for message in trace.get("messages") or []:
        def get(key, default=None):
            return message.get(key, default) if isinstance(message, dict) else getattr(message, key, default)
        for call in get("tool_calls", []) or []:
            if call.get("name") in {"report_general_result", "submit_worker_result"}:
                continue
            row = {"tool_call_id": call.get("id"), "tool_name": call.get("name"),
                   "arguments": _bounded_text(call.get("args", {}), 600), "status": "NO_RESULT", "result": ""}
            rows.append(row); by_id[call.get("id")] = row
        call_id = get("tool_call_id")
        if call_id in by_id:
            content = str(get("content", ""))
            import re
            match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))\b", content)
            fetch = re.search(r'fetch_status[\\]*["\']?\s*:\s*[\\]*["\'](EMPTY_CONTENT|ACCESS_DENIED|RATE_LIMITED|NETWORK_ERROR|SUCCESS)\b', content)
            by_id[call_id].update(status="ERROR" if get("status") == "error" or content.startswith("Execution failed") else "RETURNED",
                                  error_type=match.group(1) if match and (get("status") == "error" or content.startswith("Execution failed")) else None,
                                  fetch_status=fetch.group(1) if fetch else None,
                                  result_empty=not content.strip(),
                                  result_chars=len(content),
                                  result=_bounded_text(content, 240))

    return rows, True


def materialize_worker_review_trace(
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the bounded, durable Worker result used to rebuild review."""

    audit, available = _tool_audit(trace)
    durable: dict[str, Any] = {"tool_audit": audit, "tool_audit_available": available}
    for field_name in (
        "assignment_key",
        "assignment_objective",
        "attempt",
        "worker_id",
        "workspace_id",
        "checkpoint_thread_id",
        "finish_reason",
        "worker_terminal_action",
        "role_skill_snapshot",
        "final_reviewer_request",
    ):
        if field_name in trace:
            durable[field_name] = trace[field_name]

    for field_name, limit in (
        ("final_answer", REVIEW_FALLBACK_TEXT_MAX_CHARS),
        ("stop_reason", 2000),
        ("error", 2000),
    ):
        if field_name in trace:
            durable[field_name] = _bounded_text(trace[field_name], limit)

    execution_summary = trace.get("execution_summary")
    if isinstance(execution_summary, Mapping):
        durable["execution_summary"] = {
            key: execution_summary[key]
            for key in (
                "model_call_count",
                "skill_preparation_call_count",
                "tool_call_count",
                "tool_result_count",
                "leadership_model_call_count",
                "show_all_toolsets_call_count",
                "finalization_model_call_count",
                "finalization_tool_call_count",
            )
            if key in execution_summary
        }
    applied_limits = trace.get("applied_limits")
    if isinstance(applied_limits, Mapping):
        durable["applied_limits"] = dict(applied_limits)

    raw_submission = trace.get("worker_submission")
    if raw_submission:
        try:
            submission = WorkerSubmissionRecord.model_validate(raw_submission)
        except Exception:
            submission = None
        if submission is not None:
            durable["worker_submission"] = submission.model_copy(
                update={
                    "resolved_evidence": _bounded_evidence(
                        submission.resolved_evidence
                    )
                }
            ).model_dump(mode="json")
    raw_cancellation = trace.get("worker_cancellation_record")
    if raw_cancellation:
        try:
            cancellation = WorkerCancellationRecord.model_validate(
                raw_cancellation
            )
        except Exception:
            cancellation = None
        if cancellation is not None:
            durable["worker_cancellation_record"] = cancellation.model_dump(
                mode="json"
            )
    return durable


def _fallback_text(trace: Mapping[str, Any]) -> str:
    raw_cancellation = trace.get("worker_cancellation_record")
    if raw_cancellation:
        try:
            cancellation = WorkerCancellationRecord.model_validate(
                raw_cancellation
            )
        except Exception:
            cancellation = None
        if cancellation is not None:
            return (
                "Worker was cancelled without another model call. "
                f"Reason: {cancellation.reason}"
            )

    answer = str(trace.get("final_answer") or "").strip()
    if answer:
        return answer

    error = str(trace.get("error") or "").strip()
    if error:
        return f"Worker ended with an error: {error}"

    return "Worker ended without a structured review submission."


def _build_attempt(
    trace: Mapping[str, Any],
    *,
    current_step: PlanStep,
    default_stop_reason: str,
) -> ReviewAttempt:
    audit, audit_available = _tool_audit(trace)
    raw_record = trace.get("worker_submission")
    record: WorkerSubmissionRecord | None = None
    if raw_record:
        try:
            record = WorkerSubmissionRecord.model_validate(raw_record)
        except Exception:
            record = None

    attempt = max(int(trace.get("attempt", 1) or 1), 1)
    assignment_key = str(trace.get("assignment_key") or "primary").strip()
    assignment_objective = str(
        trace.get("assignment_objective") or current_step.objective
    ).strip()
    stop_reason = str(
        trace.get("stop_reason")
        or default_stop_reason
        or "Worker attempt ended."
    )
    finish_reason = str(
        trace.get("finish_reason")
        or (
            "READY_FOR_REVIEW"
            if record is not None
            else "NATURAL_EXIT"
        )
    )
    execution_summary = trace.get("execution_summary")
    if not isinstance(execution_summary, Mapping):
        execution_summary = {}
    execution_metrics = {
        key: execution_summary.get(key)
        for key in (
            "model_call_count",
            "tool_call_count",
            "tool_result_count",
            "leadership_model_call_count",
        )
        if key in execution_summary
    }
    applied_limits = trace.get("applied_limits")
    if not isinstance(applied_limits, Mapping):
        applied_limits = {}

    if record is not None:
        submission = record.submission
        artifact_owner = str(record.worker_id or assignment_key).strip()
        resolved_artifacts = [
            artifact.model_copy(
                update={
                    "review_ref": (
                        f"{assignment_key}/attempt-{attempt}/"
                        f"{artifact_owner}/{artifact.candidate_id}"
                    )
                }
            )
            for artifact in record.resolved_artifacts
        ]
        return ReviewAttempt(
            attempt=attempt,
            assignment_key=assignment_key,
            assignment_objective=assignment_objective,
            worker_id=record.worker_id,
            submission_source="WORKER",
            finish_reason=finish_reason,
            stop_reason=stop_reason,
            summary=submission.summary,
            final_conclusion=submission.final_conclusion,
            criterion_claims=submission.criterion_claims,
            resolved_evidence=_bounded_evidence(
                record.resolved_evidence
            ),
            resolved_artifacts=resolved_artifacts,
            unresolved_items=submission.unresolved_items,
            tool_audit=audit, tool_audit_available=audit_available,
            handoff_knowledge=[k.model_dump(mode="json") for k in submission.handoff_knowledge],
            handoff_apis=[k.model_dump(mode="json") for k in submission.handoff_apis],
            execution_metrics=execution_metrics,
            applied_limits=dict(applied_limits),
            leadership_terminal_action=trace.get(
                "worker_terminal_action"
            ),
            final_reviewer_request=trace.get("final_reviewer_request"),
            plan_challenge=submission.plan_challenge,
        )

    fallback = _fallback_text(trace)
    return ReviewAttempt(
        attempt=attempt,
        assignment_key=assignment_key,
        assignment_objective=assignment_objective,
        worker_id=trace.get("worker_id"),
        submission_source="HARNESS_FALLBACK",
        finish_reason=finish_reason,
        stop_reason=stop_reason,
        summary=fallback,
        final_conclusion=fallback,
        criterion_claims=[],
        resolved_evidence=[],
        resolved_artifacts=[],
        unresolved_items=list(current_step.success_criteria),
        tool_audit=audit, tool_audit_available=audit_available,
        execution_metrics=execution_metrics,
        applied_limits=dict(applied_limits),
        leadership_terminal_action=trace.get(
            "worker_terminal_action"
        ),
        final_reviewer_request=trace.get("final_reviewer_request"),
        plan_challenge=None,
    )


def build_step_review_packet(
    *,
    user_request: str,
    plan_objective: str,
    current_step: PlanStep,
    current_attempt: Mapping[str, Any],
    replaced_attempts: Sequence[Mapping[str, Any]] = (),
    worker_attempts: Sequence[Mapping[str, Any]] | None = None,
    previous_step_report: StepReport | None = None,
    stop_reason: str,
    remaining_budget: Mapping[str, Any] | None = None,
) -> StepReviewPacket:
    """Build the materialized review view consumed by the Reporter."""

    traces = (
        list(worker_attempts)
        if worker_attempts is not None
        else [*replaced_attempts, current_attempt]
    )
    if not traces:
        raise ValueError("StepReviewPacket requires at least one Worker attempt.")
    attempts = [
        _build_attempt(
            trace,
            current_step=current_step,
            default_stop_reason=stop_reason,
        )
        for trace in traces
    ]
    return StepReviewPacket(
        created_at=datetime.now(timezone.utc),
        task_contract=ReviewTaskContract(
            user_request=user_request,
            plan_objective=plan_objective,
            step_id=current_step.step_id,
            step_assignment=current_step.objective,
            execution_guidance=current_step.execution_guidance,
            success_criteria=list(current_step.success_criteria),
            artifact_outputs=list(current_step.artifact_outputs),
        ),
        attempts=attempts,
        previous_step_report=(
            previous_step_report.model_dump(mode="json")
            if previous_step_report is not None
            else None
        ),
        remaining_budget=dict(remaining_budget or {}),
    )


__all__ = [
    "EVIDENCE_RESULT_MAX_CHARS",
    "build_step_review_packet",
    "materialize_worker_review_trace",
]
