"""Structured handoff tools for Code Worker and Code Reviewer agents."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command
from pydantic import BaseModel, Field

from planning_models import CodeTaskContract
from prompt_loader import load_prompt
from workers.code_review_models import (
    CodeCandidateRef,
    CodeRepairInstruction,
    CodeReviewFinding,
    CodeReviewLoopState,
    CodeReviewReport,
    CodeWorkerRepairResponse,
    CodeWorkerSubmission,
    CodeContinuationSubmission,
    finish_code_review,
    receive_code_worker_response,
    receive_code_continuation_submission,
    request_code_repair as advance_repair_state,
)
from workers.code_attempt_models import (
    CodeArtifactManifest,
    CodePublicationReceipt,
)
from workers.code_publisher import PUBLISH_REVIEWED_CANDIDATE_NAME
from workers.submission import ResolvedToolEvidence, resolve_tool_evidence


SUBMIT_CODE_FOR_REVIEW_NAME = "submit_code_for_review"
RESPOND_TO_CODE_REVIEW_NAME = "respond_to_code_review"
SUBMIT_CONTINUED_CODE_FOR_REVIEW_NAME = "submit_continued_code_for_review"
REQUEST_CODE_WORKER_REPAIR_NAME = "request_code_worker_repair"
SUBMIT_CODE_REVIEW_NAME = "submit_code_review"

CODE_CONTROL_TOOL_NAMES = frozenset(
    {
        SUBMIT_CODE_FOR_REVIEW_NAME,
        RESPOND_TO_CODE_REVIEW_NAME,
        SUBMIT_CONTINUED_CODE_FOR_REVIEW_NAME,
        REQUEST_CODE_WORKER_REPAIR_NAME,
        SUBMIT_CODE_REVIEW_NAME,
        PUBLISH_REVIEWED_CANDIDATE_NAME,
        "publish_worker_progress",
    }
)


class CodeWorkerSubmissionRecord(BaseModel):
    submitted_at: datetime
    worker_id: str | None = None
    submission: CodeWorkerSubmission
    resolved_evidence: list[ResolvedToolEvidence] = Field(default_factory=list)


def _tool_message(
    runtime: ToolRuntime,
    content: str,
    fallback_id: str,
) -> ToolMessage:
    return ToolMessage(
        content=content,
        tool_call_id=runtime.tool_call_id or fallback_id,
    )


def _read_candidate(state: dict[str, Any]) -> CodeCandidateRef:
    raw = state.get("code_candidate")
    if not raw:
        raise ValueError("code_candidate is missing from the Agent state")
    return CodeCandidateRef.model_validate(raw)


def _read_loop(state: dict[str, Any]) -> CodeReviewLoopState:
    raw = state.get("code_review_loop")
    if not raw:
        raise ValueError("code_review_loop is missing from the Agent state")
    return CodeReviewLoopState.model_validate(raw)


def _validate_submission_contract(
    state: dict[str, Any],
    submission: CodeWorkerSubmission,
) -> None:
    expected_candidate = _read_candidate(state)
    if submission.candidate != expected_candidate:
        raise ValueError(
            "submission candidate does not match the assigned candidate"
        )

    contract = CodeTaskContract.model_validate(state.get("code_task"))
    expected_ids = {
        item.requirement_id
        for item in contract.requirements
    }
    submitted_ids = set(submission.requirement_status)
    if submitted_ids != expected_ids:
        missing = sorted(expected_ids - submitted_ids)
        extra = sorted(submitted_ids - expected_ids)
        raise ValueError(
            "requirement_status must cover the frozen contract exactly; "
            f"missing={missing}, extra={extra}"
        )


def _build_submission_record(
    state: dict[str, Any],
    submission: CodeWorkerSubmission,
) -> CodeWorkerSubmissionRecord:
    if submission.plan_challenge is not None and int(
        state.get("worker_plan_challenges_remaining", 0) or 0
    ) <= 0:
        raise ValueError(
            "the global plan-challenge budget is exhausted; continue the current code task "
            "or resubmit with plan_challenge=null"
        )
    _validate_submission_contract(state, submission)
    evidence = resolve_tool_evidence(
        list(state.get("worker_archived_messages", [])) + list(state.get("messages", [])),
        [
            *submission.evidence_tool_call_ids,
            *(
                submission.plan_challenge.evidence_tool_call_ids
                if submission.plan_challenge is not None
                else []
            ),
        ],
    )
    forbidden = [
        item.tool_call_id
        for item in evidence
        if item.tool_name in CODE_CONTROL_TOOL_NAMES
    ]
    if forbidden:
        raise ValueError(
            f"CODE control tools cannot be evidence: {forbidden}"
        )
    return CodeWorkerSubmissionRecord(
        submitted_at=datetime.now(timezone.utc),
        worker_id=state.get("worker_id"),
        submission=submission,
        resolved_evidence=evidence,
    )


def _validate_review_publication(
    state: dict[str, Any],
    report: CodeReviewReport,
) -> None:
    raw_receipt = state.get("code_publication_receipt")
    raw_manifest = state.get("code_artifact_manifest")
    if report.verdict != "PASSED":
        if raw_receipt is not None or raw_manifest is not None:
            raise ValueError(
                "a published candidate cannot be closed with a failed report"
            )
        return
    if raw_receipt is None or raw_manifest is None:
        raise ValueError(
            "PASSED review requires publish_reviewed_candidate first"
        )

    receipt = CodePublicationReceipt.model_validate(raw_receipt)
    manifest = CodeArtifactManifest.model_validate(raw_manifest)
    manifest_paths = tuple(item.path for item in manifest.files)
    if receipt.candidate != report.candidate:
        raise ValueError("publication receipt references a stale candidate")
    if manifest.candidate != report.candidate:
        raise ValueError("artifact manifest references a stale candidate")
    if receipt.manifest_id != manifest.manifest_id:
        raise ValueError("publication receipt and manifest do not match")
    if report.approved_artifact_paths != manifest_paths:
        raise ValueError("report approval differs from published manifest")
    if report.published_artifact_paths != manifest_paths:
        raise ValueError("report published files differ from manifest")
    if report.publication_id != receipt.publication_id:
        raise ValueError("report publication_id differs from receipt")
    if report.delivery_location != receipt.target_root:
        raise ValueError("report delivery_location differs from receipt")
    if report.applied_revision != receipt.applied_revision:
        raise ValueError("report applied_revision differs from receipt")


@tool(
    SUBMIT_CODE_FOR_REVIEW_NAME,
    description=load_prompt("workers/code_submit_tool"),
)
def submit_code_for_review(
    submission: CodeWorkerSubmission,
    runtime: ToolRuntime,
) -> Command:
    """Store the initial candidate manifest and stop the Code Worker turn."""

    if submission.plan_challenge is not None and int(
        runtime.state.get("worker_plan_challenges_remaining", 0) or 0
    ) <= 0:
        return Command(update={"messages": [_tool_message(
            runtime,
            "Plan challenge rejected by Harness: the global challenge budget is exhausted. "
            "Continue the current code task when possible or resubmit the real unresolved result "
            "with plan_challenge=null; no Reviewer or Scheduler was called.",
            SUBMIT_CODE_FOR_REVIEW_NAME,
        )]})
    try:
        record = _build_submission_record(runtime.state, submission)
    except (ValueError, TypeError) as error:
        return Command(
            update={
                "messages": [
                    _tool_message(
                        runtime,
                        (
                            "Code submission rejected: "
                            f"{error}. Correct it and submit again."
                        ),
                        SUBMIT_CODE_FOR_REVIEW_NAME,
                    )
                ]
            }
        )

    value = record.model_dump(mode="json")
    runtime.stream_writer(
        {"type": "code_worker_submission", "record": value}
    )
    return Command(
        update={
            "code_worker_submission": value,
            "code_agent_finished": True,
            "worker_review_requested": True,
            "messages": [
                _tool_message(
                    runtime,
                    "Code candidate manifest accepted for independent review.",
                    SUBMIT_CODE_FOR_REVIEW_NAME,
                )
            ],
        }
    )


@tool(
    RESPOND_TO_CODE_REVIEW_NAME,
    description=load_prompt("workers/code_repair_response_tool"),
)
def respond_to_code_review(
    response: CodeWorkerRepairResponse,
    updated_submission: CodeWorkerSubmission | None,
    runtime: ToolRuntime,
) -> Command:
    """Respond to one Reviewer repair request using the same Worker context."""

    try:
        loop = _read_loop(runtime.state)
        updated_loop = receive_code_worker_response(loop, response)
        record_value = None
        if response.action in {"REVISION_READY", "SUBMISSION_UPDATED"}:
            if updated_submission is None:
                raise ValueError(
                    f"{response.action} requires updated_submission"
                )
            if updated_submission.candidate != response.candidate:
                raise ValueError(
                    "updated_submission and response candidates differ"
                )
            record_value = _build_submission_record(
                {
                    **runtime.state,
                    "code_candidate": response.candidate.model_dump(
                        mode="json"
                    ),
                },
                updated_submission,
            ).model_dump(mode="json")
        elif updated_submission is not None:
            raise ValueError(
                "only REVISION_READY or SUBMISSION_UPDATED may include "
                "updated_submission"
            )
    except (ValueError, TypeError) as error:
        return Command(
            update={
                "messages": [
                    _tool_message(
                        runtime,
                        (
                            "Code repair response rejected: "
                            f"{error}. Correct it and respond again."
                        ),
                        RESPOND_TO_CODE_REVIEW_NAME,
                    )
                ]
            }
        )

    update: dict[str, Any] = {
        "code_candidate": response.candidate.model_dump(mode="json"),
        "code_review_loop": updated_loop.model_dump(mode="json"),
        "code_worker_repair_response": response.model_dump(mode="json"),
        "code_agent_finished": True,
        "worker_review_requested": True,
        "messages": [
            _tool_message(
                runtime,
                (
                    "Code repair response accepted with action "
                    f"{response.action}."
                ),
                RESPOND_TO_CODE_REVIEW_NAME,
            )
        ],
    }
    if record_value is not None:
        update["code_worker_submission"] = record_value
    runtime.stream_writer(
        {
            "type": "code_worker_repair_response",
            "response": response.model_dump(mode="json"),
        }
    )
    return Command(update=update)


@tool(
    SUBMIT_CONTINUED_CODE_FOR_REVIEW_NAME,
    description=load_prompt("workers/code_continuation_submit_tool"),
)
def submit_continued_code_for_review(
    submission: CodeContinuationSubmission,
    updated_submission: CodeWorkerSubmission | None,
    runtime: ToolRuntime,
) -> Command:
    """Send continuation output to Reviewer, never directly to Scheduler."""

    try:
        loop = _read_loop(runtime.state)
        updated_loop = receive_code_continuation_submission(loop, submission)
        record_value = None
        if submission.action in {"REVISION_READY", "SUBMISSION_UPDATED"}:
            if updated_submission is None:
                raise ValueError(
                    f"{submission.action} requires updated_submission"
                )
            if updated_submission.candidate != submission.candidate:
                raise ValueError(
                    "updated_submission and continuation candidates differ"
                )
            record_value = _build_submission_record(
                {
                    **runtime.state,
                    "code_candidate": submission.candidate.model_dump(
                        mode="json"
                    ),
                },
                updated_submission,
            ).model_dump(mode="json")
        elif updated_submission is not None:
            raise ValueError(
                "only REVISION_READY or SUBMISSION_UPDATED may include "
                "updated_submission"
            )
    except (ValueError, TypeError) as error:
        return Command(
            update={
                "messages": [
                    _tool_message(
                        runtime,
                        (
                            "Continued candidate submission rejected: "
                            f"{error}. Correct it and submit again."
                        ),
                        SUBMIT_CONTINUED_CODE_FOR_REVIEW_NAME,
                    )
                ]
            }
        )

    update: dict[str, Any] = {
        "code_candidate": submission.candidate.model_dump(mode="json"),
        "code_review_loop": updated_loop.model_dump(mode="json"),
        "code_continuation_submission": submission.model_dump(mode="json"),
        "code_agent_finished": True,
        "worker_review_requested": True,
        "messages": [
            _tool_message(
                runtime,
                (
                    "Continuation submission accepted for Reviewer with action "
                    f"{submission.action}."
                ),
                SUBMIT_CONTINUED_CODE_FOR_REVIEW_NAME,
            )
        ],
    }
    if record_value is not None:
        update["code_worker_submission"] = record_value
    runtime.stream_writer(
        {
            "type": "code_continuation_submission",
            "submission": submission.model_dump(mode="json"),
        }
    )
    return Command(update=update)


@tool(
    REQUEST_CODE_WORKER_REPAIR_NAME,
    description=load_prompt("reviewers/code_repair_tool"),
)
def request_code_worker_repair(
    summary: str,
    required_changes: list[str],
    findings: list[CodeReviewFinding],
    preserve_behaviors: list[str],
    runtime: ToolRuntime,
) -> Command:
    """Store one bounded Reviewer-to-Worker repair instruction."""

    try:
        loop = _read_loop(runtime.state)
        expected = {r.requirement_id for r in CodeTaskContract.model_validate(runtime.state.get("code_task")).requirements}
        for finding in findings:
            ids = finding.affected_requirement_ids
            if len(ids) != len(set(ids)) or not set(ids) <= expected:
                raise ValueError("Unknown or duplicate affected_requirement_ids")
        updated = advance_repair_state(
            loop,
            summary=summary,
            required_changes=tuple(required_changes),
            findings=tuple(findings),
            preserve_behaviors=tuple(preserve_behaviors),
        )
        instruction: CodeRepairInstruction = updated.pending_instruction
    except (ValueError, TypeError) as error:
        return Command(
            update={
                "messages": [
                    _tool_message(
                        runtime,
                        (
                            "Repair request rejected: "
                            f"{error}. Submit a final review instead if "
                            "budget is exhausted."
                        ),
                        REQUEST_CODE_WORKER_REPAIR_NAME,
                    )
                ]
            }
        )

    value = instruction.model_dump(mode="json")
    runtime.stream_writer(
        {"type": "code_repair_instruction", "instruction": value}
    )
    return Command(
        update={
            "code_review_loop": updated.model_dump(mode="json"),
            "code_repair_instruction": value,
            "code_agent_finished": True,
            "messages": [
                _tool_message(
                    runtime,
                    (
                        "Structured repair instruction accepted and ready "
                        "for the Code Worker."
                    ),
                    REQUEST_CODE_WORKER_REPAIR_NAME,
                )
            ],
        }
    )


@tool(
    SUBMIT_CODE_REVIEW_NAME,
    description=load_prompt("reviewers/code_review_submit_tool"),
)
def submit_code_review(
    report: CodeReviewReport,
    runtime: ToolRuntime,
) -> Command:
    """Store the final CODE review and close the Reviewer turn."""

    try:
        loop = _read_loop(runtime.state)
        validate_review_requirements(runtime.state, report)
        _validate_review_publication(runtime.state, report)
        updated = finish_code_review(loop, report)
    except (ValueError, TypeError) as error:
        return Command(
            update={
                "messages": [
                    _tool_message(
                        runtime,
                        (
                            "Code review rejected: "
                            f"{error}. Review the current candidate and "
                            "submit again."
                        ),
                        SUBMIT_CODE_REVIEW_NAME,
                    )
                ]
            }
        )

    value = report.model_dump(mode="json")
    runtime.stream_writer({"type": "code_review_report", "report": value})
    return Command(
        update={
            "code_review_loop": updated.model_dump(mode="json"),
            "code_review_report": value,
            "code_agent_finished": True,
            "messages": [
                _tool_message(
                    runtime,
                    f"Code review accepted with verdict {report.verdict}.",
                    SUBMIT_CODE_REVIEW_NAME,
                )
            ],
        }
    )


def validate_review_requirements(state, report):
    """Validate review identities against the frozen task, not model prose."""
    contract = CodeTaskContract.model_validate(state.get("code_task"))
    expected = {r.requirement_id for r in contract.requirements}
    verified = list(report.verified_requirement_ids)
    if len(verified) != len(set(verified)) or not set(verified) <= expected:
        raise ValueError("Unknown or duplicate verified_requirement_ids")
    if report.verdict == "PASSED" and set(verified) != expected:
        raise ValueError("PASSED must verify every frozen requirement")
    from workers.evidence_refs import eligible_evidence
    if not set(report.evidence_refs) <= eligible_evidence(state):
        raise ValueError("Review evidence must refer to actual registered tool results")


__all__ = [
    "CODE_CONTROL_TOOL_NAMES",
    "REQUEST_CODE_WORKER_REPAIR_NAME",
    "RESPOND_TO_CODE_REVIEW_NAME",
    "SUBMIT_CONTINUED_CODE_FOR_REVIEW_NAME",
    "SUBMIT_CODE_FOR_REVIEW_NAME",
    "SUBMIT_CODE_REVIEW_NAME",
    "CodeWorkerSubmissionRecord",
    "request_code_worker_repair",
    "respond_to_code_review",
    "submit_continued_code_for_review",
    "submit_code_for_review",
    "submit_code_review",
]
