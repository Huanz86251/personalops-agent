"""Deterministic Web StepReport projection.

The Web Worker submits one terminal ``complete_web_step`` payload. The Harness
validates that payload and projects it into the StepReport consumed by
Scheduler and Final Reviewer. No Web Reviewer or second model-authored report
round exists in this path.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from planning_models import StepCriterionResult, StepReport
from schema_normalization import derive_step_status
from workers.submission import ResolvedToolEvidence, WorkerSubmissionRecord


class WebStepReportRecord(BaseModel):
    """Harness-authored durable envelope for one Web Worker completion."""

    model_config = ConfigDict(extra="forbid")

    submitted_at: datetime
    worker_id: str | None = None
    report: StepReport
    resolved_evidence: list[ResolvedToolEvidence] = Field(default_factory=list)
    approved_artifact_candidate_ids: list[str] = Field(default_factory=list)
    normalization_notes: list[str] = Field(default_factory=list)


def _unique(values) -> list[str]:
    result: list[str] = []
    for value in values:
        item = str(value).strip()
        if item and item not in result:
            result.append(item)
    return result


def _public_evidence_refs(
    record: WorkerSubmissionRecord,
    state: Any,
) -> dict[str, str]:
    """Freeze the same short references the Worker saw into its StepReport."""

    from workers.evidence_refs import is_web_result_ref, registry

    registered = registry(state)
    public: dict[str, str] = {}
    reserved = set(registered.values())
    next_index = 1
    for evidence in record.resolved_evidence:
        raw = evidence.tool_call_id
        if is_web_result_ref(raw):
            public[raw] = raw
            continue
        short = registered.get(raw)
        if not short:
            while f"E{next_index}" in reserved:
                next_index += 1
            short = f"E{next_index}"
            reserved.add(short)
            next_index += 1
        public[raw] = short
    return public


def compile_web_step_report(
    record: WorkerSubmissionRecord,
    state: Any,
) -> WebStepReportRecord:
    """Project one validated Worker submission into a conservative StepReport.

    Only deterministic repairs are made: missing criteria become UNKNOWN,
    unsupported MET becomes UNKNOWN, aggregate status is derived from the
    criterion rows, and omitted supplementary prose reuses submission facts.
    This function never calls a model or invents business evidence.
    """

    from reporting.criteria import worker_registry

    submission = record.submission
    contract = worker_registry(state)
    expected_step = str(state.get("step_id") or record.step_id or "").strip()
    notes: list[str] = []
    public_refs = _public_evidence_refs(record, state)
    resolved_ids = {
        evidence.tool_call_id for evidence in record.resolved_evidence
    }

    indexed = {
        str(claim.criterion_id): claim
        for claim in submission.criterion_claims
        if claim.criterion_id in contract
    }
    criterion_results: list[StepCriterionResult] = []
    unresolved = _unique(submission.unresolved_items)
    used_refs: list[str] = []

    for criterion_id, criterion in contract.items():
        claim = indexed.get(criterion_id)
        if claim is None:
            status = "UNKNOWN"
            evidence: list[str] = []
            notes.append(
                f"synthesized missing criterion_id {criterion_id} as UNKNOWN"
            )
        else:
            raw_evidence = [
                reference
                for reference in _unique(claim.evidence_tool_call_ids)
                if reference in resolved_ids
            ]
            dropped = [
                reference
                for reference in _unique(claim.evidence_tool_call_ids)
                if reference not in resolved_ids
            ]
            if dropped:
                notes.append(
                    f"dropped unresolved evidence from {criterion_id}: "
                    + ", ".join(dropped)
                )
            evidence = [public_refs[reference] for reference in raw_evidence]
            status = claim.status
            if status == "MET" and not evidence:
                status = "UNKNOWN"
                notes.append(
                    f"downgraded unsupported MET criterion {criterion_id} to UNKNOWN"
                )
        if status != "MET" and criterion not in unresolved:
            unresolved.append(criterion)
        used_refs.extend(evidence)
        criterion_results.append(
            StepCriterionResult(
                criterion_id=criterion_id,
                criterion=criterion,
                evidence=evidence,
                status=status,
            )
        )

    status = derive_step_status(item.status for item in criterion_results)
    confirmed_results = _unique(submission.confirmed_results)
    if not confirmed_results:
        confirmed_results = _unique(
            claim.conclusion
            for claim in submission.criterion_claims
            if claim.status in {"MET", "PARTIAL"}
        )

    errors = _unique(submission.errors)
    if notes:
        errors.append("Harness Web report normalization: " + "; ".join(notes))

    request_replan = bool(
        submission.request_replan or submission.plan_challenge is not None
    )
    replan_reason = submission.replan_reason
    if replan_reason is None and submission.plan_challenge is not None:
        replan_reason = submission.plan_challenge.reason

    if submission.stop_reason:
        stop_reason = submission.stop_reason
    elif status == "COMPLETED":
        stop_reason = "All frozen Web Step criteria were submitted with resolved evidence."
    else:
        stop_reason = (
            "Web Worker submitted its available results; unmet or unsupported "
            "criteria were preserved conservatively."
        )

    step_id = int(expected_step) if expected_step.isdigit() else 1
    report_evidence = _unique(
        public_refs[reference]
        for reference in submission.evidence_tool_call_ids
        if reference in public_refs
    )
    report_evidence = _unique([*report_evidence, *used_refs])
    from step_execution import resolved_evidence_source_links

    report = StepReport(
        step_id=step_id,
        summary=submission.summary,
        answer_text=None,
        stop_reason=stop_reason,
        status=status,
        criterion_results=criterion_results,
        confirmed_results=confirmed_results,
        assessment_source="WEB_SELF_REPORT",
        completed_work=_unique(submission.completed_work),
        evidence=report_evidence,
        source_links=[
            link
            for link in resolved_evidence_source_links(
                record.resolved_evidence,
                public_refs,
                set(report_evidence),
            )
        ],
        errors=errors,
        unresolved_items=unresolved,
        replan_reason=replan_reason,
        request_replan=request_replan,
        next_action=submission.next_action,
        recovery_proposal=(
            submission.recovery_proposal.model_dump(mode="json")
            if submission.recovery_proposal is not None
            else None
        ),
    )
    return WebStepReportRecord(
        submitted_at=datetime.now(timezone.utc),
        worker_id=record.worker_id,
        report=report,
        resolved_evidence=record.resolved_evidence,
        approved_artifact_candidate_ids=[
            artifact.candidate_id for artifact in record.resolved_artifacts
        ],
        normalization_notes=notes,
    )


__all__ = ["WebStepReportRecord", "compile_web_step_report"]
