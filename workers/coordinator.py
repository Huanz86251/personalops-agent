"""Application service joining Worker Group state, SQLite CAS, and audit."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from eventing import (
    AsyncEventStore,
    EventConflictError,
    StoredWorkerGroup,
    WorkerGroupReviewClaim,
)
from planning_models import PlanStep
from workers.audit import WorkerAttemptAuditRecord
from workers.group import (
    StepWorkerGroup,
    WorkerAttempt,
    WorkerAttemptStatus,
    WorkerGroupMutation,
    create_step_worker_group,
    finish_worker_attempt,
    replace_worker_attempt,
    start_worker_attempt,
)


@dataclass(frozen=True, slots=True)
class CoordinatedWorkerMutation:
    """One durable Worker transition and the audit envelope that explains it."""

    group: StepWorkerGroup
    revision: int
    applied: bool
    stale: bool
    audit_id: str


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _attempt_for_identity(
    group: StepWorkerGroup,
    *,
    assignment_key: str,
    attempt_no: int,
    worker_id: str,
) -> WorkerAttempt:
    for slot in group.slots:
        if slot.assignment_key != assignment_key:
            continue
        for attempt in slot.attempts:
            if attempt.attempt_no == attempt_no and attempt.worker_id == worker_id:
                return attempt
        raise ValueError("Worker identity does not belong to the requested slot.")
    raise KeyError(f"Unknown Worker slot: {assignment_key}")


class WorkerGroupCoordinator:
    """The only runtime entry point for durable Worker lifecycle updates."""

    def __init__(
        self,
        event_store: AsyncEventStore,
        *,
        workspace_retention_minutes: int = 24 * 60,
        max_cas_retries: int = 8,
    ) -> None:
        if workspace_retention_minutes < 1:
            raise ValueError("workspace_retention_minutes must be positive.")
        if max_cas_retries < 1 or max_cas_retries > 100:
            raise ValueError("max_cas_retries must be between 1 and 100.")
        self.event_store = event_store
        self.workspace_retention_minutes = workspace_retention_minutes
        self.max_cas_retries = max_cas_retries

    async def create_group(
        self,
        step: PlanStep,
        *,
        event_id: str,
        group_id: str | None = None,
        created_at: datetime | None = None,
    ) -> StoredWorkerGroup:
        """Materialize and idempotently persist one Step Worker Group."""

        if group_id is not None:
            existing = await self.event_store.get_worker_group(group_id)
            if existing is not None:
                persisted = StepWorkerGroup.model_validate(existing.snapshot)
                expected_assignments = (
                    tuple(
                        (item.assignment_key, item.objective)
                        for item in step.worker_assignments
                    )
                    if step.execution_mode == "PARALLEL"
                    else (("primary", step.objective),)
                )
                actual_assignments = tuple(
                    (
                        slot.assignment_key,
                        slot.attempts[0].objective,
                    )
                    for slot in persisted.slots
                )
                if (
                    persisted.event_id != event_id
                    or persisted.step_id != step.step_id
                    or actual_assignments != expected_assignments
                ):
                    raise EventConflictError(
                        "Worker Group identity was reused for another Step contract."
                    )
                return existing

        group = create_step_worker_group(
            step,
            event_id=event_id,
            group_id=group_id,
            created_at=created_at,
        )
        return await self.event_store.create_worker_group(
            group.model_dump(mode="json")
        )

    async def _apply(
        self,
        *,
        group_id: str,
        assignment_key: str,
        attempt_no: int,
        worker_id: str,
        operation: str,
        occurred_at: datetime,
        summary: str,
        transition: Callable[[StepWorkerGroup], WorkerGroupMutation],
        trace_id: str | None = None,
    ) -> CoordinatedWorkerMutation:
        audit_id = f"worker-audit-{uuid4().hex}"
        for _ in range(self.max_cas_retries):
            stored = await self.event_store.require_worker_group(group_id)
            group = StepWorkerGroup.model_validate(stored.snapshot)
            before = _attempt_for_identity(
                group,
                assignment_key=assignment_key,
                attempt_no=attempt_no,
                worker_id=worker_id,
            )
            mutation = transition(group)
            if mutation.applied:
                after = _attempt_for_identity(
                    mutation.group,
                    assignment_key=assignment_key,
                    attempt_no=attempt_no,
                    worker_id=worker_id,
                )
                audit = self._audit_record(
                    audit_id=audit_id,
                    group_id=group_id,
                    assignment_key=assignment_key,
                    attempt=before,
                    operation=operation,
                    outcome="APPLIED",
                    occurred_at=occurred_at,
                    summary=summary,
                    trace_id=trace_id,
                    status_before=before.status.value,
                    status_after=after.status.value,
                )
                write = await self.event_store.compare_and_set_worker_group(
                    mutation.group.model_dump(mode="json"),
                    expected_revision=stored.revision,
                    audit_record=audit.model_dump(mode="json"),
                )
                if not write.applied:
                    continue
                return CoordinatedWorkerMutation(
                    group=StepWorkerGroup.model_validate(write.record.snapshot),
                    revision=write.record.revision,
                    applied=True,
                    stale=False,
                    audit_id=audit_id,
                )

            outcome = "STALE" if mutation.stale else "IDEMPOTENT"
            audit = self._audit_record(
                audit_id=audit_id,
                group_id=group_id,
                assignment_key=assignment_key,
                attempt=before,
                operation=("CALLBACK" if mutation.stale else operation),
                outcome=outcome,
                occurred_at=occurred_at,
                summary=summary,
                trace_id=trace_id,
                status_before=before.status.value,
                status_after=before.status.value,
            )
            await self.event_store.append_worker_attempt_audit(
                audit.model_dump(mode="json")
            )
            return CoordinatedWorkerMutation(
                group=mutation.group,
                revision=stored.revision,
                applied=False,
                stale=mutation.stale,
                audit_id=audit_id,
            )

        raise EventConflictError(
            "Worker Group remained contended after the bounded CAS retry limit."
        )

    @staticmethod
    def _audit_record(
        *,
        audit_id: str,
        group_id: str,
        assignment_key: str,
        attempt: WorkerAttempt,
        operation: str,
        outcome: str,
        occurred_at: datetime,
        summary: str,
        trace_id: str | None,
        status_before: str,
        status_after: str,
    ) -> WorkerAttemptAuditRecord:
        return WorkerAttemptAuditRecord(
            audit_id=audit_id,
            group_id=group_id,
            assignment_key=assignment_key,
            attempt_no=attempt.attempt_no,
            worker_id=attempt.worker_id,
            workspace_id=attempt.workspace.workspace_id,
            checkpoint_thread_id=attempt.workspace.checkpoint_thread_id,
            operation=operation,
            outcome=outcome,
            occurred_at=occurred_at,
            recorded_at=_utc_now(),
            summary=summary,
            trace_id=trace_id,
            status_before=status_before,
            status_after=status_after,
        )

    async def start_attempt(
        self,
        *,
        group_id: str,
        assignment_key: str,
        attempt_no: int,
        worker_id: str,
        occurred_at: datetime,
        trace_id: str | None = None,
    ) -> CoordinatedWorkerMutation:
        return await self._apply(
            group_id=group_id,
            assignment_key=assignment_key,
            attempt_no=attempt_no,
            worker_id=worker_id,
            operation="START",
            occurred_at=occurred_at,
            summary="Worker attempt entered RUNNING.",
            trace_id=trace_id,
            transition=lambda group: start_worker_attempt(
                group,
                assignment_key=assignment_key,
                attempt_no=attempt_no,
                worker_id=worker_id,
                changed_at=occurred_at,
            ),
        )

    async def finish_attempt(
        self,
        *,
        group_id: str,
        assignment_key: str,
        attempt_no: int,
        worker_id: str,
        status: WorkerAttemptStatus,
        terminal_reason: str,
        review_payload: Mapping[str, Any] | None = None,
        occurred_at: datetime,
        trace_id: str | None = None,
    ) -> CoordinatedWorkerMutation:
        return await self._apply(
            group_id=group_id,
            assignment_key=assignment_key,
            attempt_no=attempt_no,
            worker_id=worker_id,
            operation="FINISH",
            occurred_at=occurred_at,
            summary=terminal_reason,
            trace_id=trace_id,
            transition=lambda group: finish_worker_attempt(
                group,
                assignment_key=assignment_key,
                attempt_no=attempt_no,
                worker_id=worker_id,
                status=status,
                terminal_reason=terminal_reason,
                review_payload=review_payload,
                workspace_retention_minutes=self.workspace_retention_minutes,
                changed_at=occurred_at,
            ),
        )

    async def replace_attempt(
        self,
        *,
        group_id: str,
        assignment_key: str,
        attempt_no: int,
        worker_id: str,
        replacement_objective: str,
        reason: str,
        review_payload: Mapping[str, Any] | None = None,
        occurred_at: datetime,
        trace_id: str | None = None,
    ) -> CoordinatedWorkerMutation:
        return await self._apply(
            group_id=group_id,
            assignment_key=assignment_key,
            attempt_no=attempt_no,
            worker_id=worker_id,
            operation="REPLACE",
            occurred_at=occurred_at,
            summary=reason,
            trace_id=trace_id,
            transition=lambda group: replace_worker_attempt(
                group,
                assignment_key=assignment_key,
                attempt_no=attempt_no,
                worker_id=worker_id,
                replacement_objective=replacement_objective,
                reason=reason,
                review_payload=review_payload,
                workspace_retention_minutes=self.workspace_retention_minutes,
                changed_at=occurred_at,
            ),
        )

    async def claim_reporter(
        self,
        *,
        group_id: str,
        review_id: str,
        changed_at: datetime | None = None,
    ) -> WorkerGroupReviewClaim:
        return await self.event_store.claim_worker_group_review(
            group_id=group_id,
            review_id=review_id,
            changed_at=changed_at,
        )

    async def complete_reporter(
        self,
        *,
        group_id: str,
        review_id: str,
        report: Mapping[str, Any],
        changed_at: datetime | None = None,
    ) -> StoredWorkerGroup:
        return await self.event_store.complete_worker_group_review(
            group_id=group_id,
            review_id=review_id,
            report=report,
            changed_at=changed_at,
        )


__all__ = [
    "CoordinatedWorkerMutation",
    "WorkerGroupCoordinator",
]
