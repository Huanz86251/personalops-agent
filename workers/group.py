"""Pure runtime state machine for one Step's Worker slots.

This module intentionally has no asyncio, SQLite, LangGraph, or model
dependency.  Every transition returns a new immutable, JSON-serializable
snapshot so the same rules can later be enforced by the durable store.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Mapping
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from planning_models import PlanStep, StepJoinPolicy


class WorkerAttemptStatus(str, Enum):
    """Lifecycle of one concrete Worker invocation."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUBMITTED = "SUBMITTED"
    NATURAL_EXIT = "NATURAL_EXIT"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    REPLACED = "REPLACED"


ACTIVE_ATTEMPT_STATUSES = frozenset(
    {
        WorkerAttemptStatus.PENDING,
        WorkerAttemptStatus.RUNNING,
    }
)


COMPLETION_TERMINAL_STATUSES = frozenset(
    {
        WorkerAttemptStatus.SUBMITTED,
        WorkerAttemptStatus.NATURAL_EXIT,
        WorkerAttemptStatus.BUDGET_EXHAUSTED,
        WorkerAttemptStatus.FAILED,
        WorkerAttemptStatus.CANCELLED,
    }
)


class WorkerGroupReviewStatus(str, Enum):
    """Reporter coordination state for the whole Step group."""

    WAITING = "WAITING"
    JOIN_READY = "JOIN_READY"
    REVIEWING = "REVIEWING"
    REPORTED = "REPORTED"


class WorkerWorkspaceStatus(str, Enum):
    """Logical write lifecycle of one attempt's checkpoint workspace."""

    PREPARED = "PREPARED"
    ACTIVE = "ACTIVE"
    FROZEN = "FROZEN"


class WorkerGroupModel(BaseModel):
    """Strict immutable base for checkpoint-safe group state."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


def _as_utc(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include timezone information.")
    return value.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_text(value: object, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty.")
    return normalized


def _worker_id(
    *,
    group_id: str,
    assignment_key: str,
    attempt_no: int,
) -> str:
    return (
        f"worker:{group_id}:slot:{assignment_key}:attempt:{attempt_no}"
    )


class WorkerWorkspace(WorkerGroupModel):
    """A checkpoint-backed private workspace owned by one attempt generation."""

    workspace_id: str = Field(min_length=1)
    checkpoint_thread_id: str = Field(min_length=1)
    status: WorkerWorkspaceStatus
    created_at: datetime
    status_changed_at: datetime
    retain_until: datetime | None = None

    @model_validator(mode="after")
    def validate_workspace(self) -> "WorkerWorkspace":
        created_at = _as_utc(self.created_at, field_name="created_at")
        changed_at = _as_utc(
            self.status_changed_at,
            field_name="status_changed_at",
        )
        if changed_at < created_at:
            raise ValueError("Workspace status cannot change before creation.")
        retain_until = self.retain_until
        if self.status is WorkerWorkspaceStatus.FROZEN:
            if retain_until is None:
                raise ValueError("A frozen workspace requires retain_until.")
            retain_until = _as_utc(retain_until, field_name="retain_until")
            if retain_until < changed_at:
                raise ValueError("retain_until cannot be before workspace freeze time.")
        elif retain_until is not None:
            raise ValueError("Only a frozen workspace may have retain_until.")
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "status_changed_at", changed_at)
        object.__setattr__(self, "retain_until", retain_until)
        return self

    def cleanup_due(self, now: datetime | None = None) -> bool:
        """Return whether a frozen workspace passed its retention boundary."""

        if self.status is not WorkerWorkspaceStatus.FROZEN:
            return False
        timestamp = _as_utc(now or _utc_now(), field_name="now")
        return self.retain_until is not None and timestamp >= self.retain_until


def _workspace(
    *,
    event_id: str,
    group_id: str,
    assignment_key: str,
    attempt_no: int,
    created_at: datetime,
) -> WorkerWorkspace:
    identity = f"{group_id}:slot:{assignment_key}:attempt:{attempt_no}"
    return WorkerWorkspace(
        workspace_id=f"workspace:{identity}",
        checkpoint_thread_id=f"{event_id}:{identity}",
        status=WorkerWorkspaceStatus.PREPARED,
        created_at=created_at,
        status_changed_at=created_at,
    )


def _workspace_retention(minutes: int) -> timedelta:
    if isinstance(minutes, bool) or not isinstance(minutes, int) or minutes < 1:
        raise ValueError("workspace_retention_minutes must be a positive integer.")
    return timedelta(minutes=minutes)


def _active_workspace(
    workspace: WorkerWorkspace,
    *,
    changed_at: datetime,
) -> WorkerWorkspace:
    payload = workspace.model_dump()
    payload.update(
        status=WorkerWorkspaceStatus.ACTIVE,
        status_changed_at=changed_at,
    )
    return WorkerWorkspace.model_validate(payload)


def _frozen_workspace(
    workspace: WorkerWorkspace,
    *,
    changed_at: datetime,
    retention: timedelta,
) -> WorkerWorkspace:
    payload = workspace.model_dump()
    payload.update(
        status=WorkerWorkspaceStatus.FROZEN,
        status_changed_at=changed_at,
        retain_until=changed_at + retention,
    )
    return WorkerWorkspace.model_validate(payload)


class WorkerAttempt(WorkerGroupModel):
    """One immutable generation inside a logical assignment slot."""

    attempt_no: int = Field(ge=1)
    worker_id: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    workspace: WorkerWorkspace
    status: WorkerAttemptStatus
    created_at: datetime
    status_changed_at: datetime
    terminal_reason: str | None = None
    review_payload: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_attempt(self) -> "WorkerAttempt":
        created_at = _as_utc(self.created_at, field_name="created_at")
        changed_at = _as_utc(
            self.status_changed_at,
            field_name="status_changed_at",
        )
        if changed_at < created_at:
            raise ValueError("status_changed_at cannot be before created_at.")
        if self.status in ACTIVE_ATTEMPT_STATUSES and self.terminal_reason:
            raise ValueError("An active Worker attempt cannot have terminal_reason.")
        if self.status in ACTIVE_ATTEMPT_STATUSES and self.review_payload is not None:
            raise ValueError("An active Worker attempt cannot have review_payload.")
        if self.status not in ACTIVE_ATTEMPT_STATUSES and not (
            self.terminal_reason or ""
        ).strip():
            raise ValueError("A terminal Worker attempt requires terminal_reason.")
        expected_workspace_status = {
            WorkerAttemptStatus.PENDING: WorkerWorkspaceStatus.PREPARED,
            WorkerAttemptStatus.RUNNING: WorkerWorkspaceStatus.ACTIVE,
        }.get(self.status, WorkerWorkspaceStatus.FROZEN)
        if self.workspace.status is not expected_workspace_status:
            raise ValueError(
                "Worker attempt and workspace lifecycle states are inconsistent."
            )
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "status_changed_at", changed_at)
        return self


class WorkerSlot(WorkerGroupModel):
    """Stable assignment identity with append-only attempt generations."""

    assignment_key: str = Field(min_length=1)
    attempts: tuple[WorkerAttempt, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_attempt_history(self) -> "WorkerSlot":
        expected_numbers = list(range(1, len(self.attempts) + 1))
        actual_numbers = [attempt.attempt_no for attempt in self.attempts]
        if actual_numbers != expected_numbers:
            raise ValueError("Worker attempt numbers must be contiguous from 1.")
        if any(
            attempt.status is not WorkerAttemptStatus.REPLACED
            for attempt in self.attempts[:-1]
        ):
            raise ValueError(
                "Every non-current Worker attempt must have status REPLACED."
            )
        return self

    @property
    def current_attempt(self) -> WorkerAttempt:
        return self.attempts[-1]


class StepWorkerGroup(WorkerGroupModel):
    """One Step's slots plus the deterministic ALL_TERMINAL barrier."""

    group_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    step_id: int = Field(ge=1)
    join_policy: StepJoinPolicy = "ALL_TERMINAL"
    slots: tuple[WorkerSlot, ...] = Field(min_length=1, max_length=3)
    review_status: WorkerGroupReviewStatus = WorkerGroupReviewStatus.WAITING
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def validate_group(self) -> "StepWorkerGroup":
        if self.join_policy != "ALL_TERMINAL":
            raise ValueError("V1 only supports ALL_TERMINAL.")
        keys = [slot.assignment_key for slot in self.slots]
        if len(set(keys)) != len(keys):
            raise ValueError("Worker slot assignment_key values must be unique.")
        created_at = _as_utc(self.created_at, field_name="created_at")
        updated_at = _as_utc(self.updated_at, field_name="updated_at")
        if updated_at < created_at:
            raise ValueError("updated_at cannot be before created_at.")
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)

        ready = all(
            slot.current_attempt.status in COMPLETION_TERMINAL_STATUSES
            for slot in self.slots
        )
        if self.review_status is WorkerGroupReviewStatus.WAITING and ready:
            raise ValueError("A fully terminal group must be JOIN_READY.")
        if self.review_status is not WorkerGroupReviewStatus.WAITING and not ready:
            raise ValueError("A non-terminal group cannot enter Reporter state.")
        return self

    @property
    def current_attempts(self) -> tuple[WorkerAttempt, ...]:
        return tuple(slot.current_attempt for slot in self.slots)

    @property
    def active_worker_ids(self) -> tuple[str, ...]:
        return tuple(
            attempt.worker_id
            for attempt in self.current_attempts
            if attempt.status is WorkerAttemptStatus.RUNNING
        )

    @property
    def nonterminal_worker_ids(self) -> tuple[str, ...]:
        return tuple(
            attempt.worker_id
            for attempt in self.current_attempts
            if attempt.status in ACTIVE_ATTEMPT_STATUSES
        )

    @property
    def is_join_ready(self) -> bool:
        return self.review_status is WorkerGroupReviewStatus.JOIN_READY


class WorkerGroupMutation(WorkerGroupModel):
    """Outcome of an idempotent or stale-aware group update."""

    group: StepWorkerGroup
    applied: bool
    stale: bool = False


def create_step_worker_group(
    step: PlanStep,
    *,
    event_id: str,
    group_id: str | None = None,
    created_at: datetime | None = None,
) -> StepWorkerGroup:
    """Create one uniform group for either a SINGLE or PARALLEL Step."""

    normalized_event_id = _require_text(event_id, field_name="event_id")
    normalized_group_id = _require_text(
        group_id or f"group-{uuid4().hex}",
        field_name="group_id",
    )
    timestamp = _as_utc(
        created_at or _utc_now(),
        field_name="created_at",
    )

    assignments = (
        [(assignment.assignment_key, assignment.objective)
         for assignment in step.worker_assignments]
        if step.execution_mode == "PARALLEL"
        else [("primary", step.objective)]
    )
    slots = tuple(
        WorkerSlot(
            assignment_key=assignment_key,
            attempts=(
                WorkerAttempt(
                    attempt_no=1,
                    worker_id=_worker_id(
                        group_id=normalized_group_id,
                        assignment_key=assignment_key,
                        attempt_no=1,
                    ),
                    objective=objective,
                    workspace=_workspace(
                        event_id=normalized_event_id,
                        group_id=normalized_group_id,
                        assignment_key=assignment_key,
                        attempt_no=1,
                        created_at=timestamp,
                    ),
                    status=WorkerAttemptStatus.PENDING,
                    created_at=timestamp,
                    status_changed_at=timestamp,
                ),
            ),
        )
        for assignment_key, objective in assignments
    )
    return StepWorkerGroup(
        group_id=normalized_group_id,
        event_id=normalized_event_id,
        step_id=step.step_id,
        join_policy=step.join_policy,
        slots=slots,
        created_at=timestamp,
        updated_at=timestamp,
    )


def _slot_index(group: StepWorkerGroup, assignment_key: str) -> int:
    normalized_key = _require_text(
        assignment_key,
        field_name="assignment_key",
    )
    for index, slot in enumerate(group.slots):
        if slot.assignment_key == normalized_key:
            return index
    raise KeyError(f"Unknown Worker slot: {normalized_key}")


def _replace_slot(
    group: StepWorkerGroup,
    *,
    slot_index: int,
    slot: WorkerSlot,
    changed_at: datetime,
) -> StepWorkerGroup:
    slots = list(group.slots)
    slots[slot_index] = slot
    ready = all(
        candidate.current_attempt.status in COMPLETION_TERMINAL_STATUSES
        for candidate in slots
    )
    review_status = (
        WorkerGroupReviewStatus.JOIN_READY
        if ready
        else WorkerGroupReviewStatus.WAITING
    )
    payload = group.model_dump()
    payload.update(
        slots=tuple(slots),
        review_status=review_status,
        updated_at=max(group.updated_at, changed_at),
    )
    return StepWorkerGroup.model_validate(payload)


def _transition_time(
    value: datetime | None,
) -> datetime:
    return _as_utc(value or _utc_now(), field_name="changed_at")


def _callback_is_stale(
    slot: WorkerSlot,
    *,
    attempt_no: int,
    worker_id: str,
) -> bool:
    known_attempt = next(
        (
            attempt
            for attempt in slot.attempts
            if attempt.attempt_no == attempt_no
            and attempt.worker_id == worker_id
        ),
        None,
    )
    if known_attempt is None:
        raise ValueError("Worker callback identity does not belong to this slot.")
    return known_attempt is not slot.current_attempt


def start_worker_attempt(
    group: StepWorkerGroup,
    *,
    assignment_key: str,
    attempt_no: int,
    worker_id: str,
    changed_at: datetime | None = None,
) -> WorkerGroupMutation:
    """Move the current generation from PENDING to RUNNING."""

    index = _slot_index(group, assignment_key)
    slot = group.slots[index]
    attempt = slot.current_attempt
    if attempt_no != attempt.attempt_no or worker_id != attempt.worker_id:
        return WorkerGroupMutation(
            group=group,
            applied=False,
            stale=_callback_is_stale(
                slot,
                attempt_no=attempt_no,
                worker_id=worker_id,
            ),
        )
    if attempt.status is WorkerAttemptStatus.RUNNING:
        return WorkerGroupMutation(group=group, applied=False)
    if attempt.status is not WorkerAttemptStatus.PENDING:
        raise ValueError(f"Cannot start Worker attempt in {attempt.status.value}.")
    timestamp = _transition_time(changed_at)
    attempt_payload = attempt.model_dump()
    attempt_payload.update(
        workspace=_active_workspace(
            attempt.workspace,
            changed_at=timestamp,
        ),
        status=WorkerAttemptStatus.RUNNING,
        status_changed_at=timestamp,
    )
    updated_attempt = WorkerAttempt.model_validate(attempt_payload)
    updated_slot = WorkerSlot(
        assignment_key=slot.assignment_key,
        attempts=(*slot.attempts[:-1], updated_attempt),
    )
    return WorkerGroupMutation(
        group=_replace_slot(
            group,
            slot_index=index,
            slot=updated_slot,
            changed_at=timestamp,
        ),
        applied=True,
    )


def finish_worker_attempt(
    group: StepWorkerGroup,
    *,
    assignment_key: str,
    attempt_no: int,
    worker_id: str,
    status: WorkerAttemptStatus,
    terminal_reason: str,
    review_payload: Mapping[str, Any] | None = None,
    workspace_retention_minutes: int = 24 * 60,
    changed_at: datetime | None = None,
) -> WorkerGroupMutation:
    """Record one terminal callback without accepting an obsolete generation."""

    if status not in COMPLETION_TERMINAL_STATUSES:
        raise ValueError("finish_worker_attempt requires a completion terminal status.")
    normalized_reason = _require_text(
        terminal_reason,
        field_name="terminal_reason",
    )
    index = _slot_index(group, assignment_key)
    slot = group.slots[index]
    attempt = slot.current_attempt
    if attempt_no != attempt.attempt_no or worker_id != attempt.worker_id:
        return WorkerGroupMutation(
            group=group,
            applied=False,
            stale=_callback_is_stale(
                slot,
                attempt_no=attempt_no,
                worker_id=worker_id,
            ),
        )
    if attempt.status is status:
        if attempt.terminal_reason != normalized_reason:
            raise ValueError(
                "An idempotent terminal replay must keep terminal_reason unchanged."
            )
        return WorkerGroupMutation(group=group, applied=False)
    if attempt.status not in ACTIVE_ATTEMPT_STATUSES:
        raise ValueError(
            f"Worker attempt is already terminal as {attempt.status.value}."
        )
    timestamp = _transition_time(changed_at)
    retention = _workspace_retention(workspace_retention_minutes)
    attempt_payload = attempt.model_dump()
    attempt_payload.update(
        workspace=_frozen_workspace(
            attempt.workspace,
            changed_at=timestamp,
            retention=retention,
        ),
        status=status,
        status_changed_at=timestamp,
        terminal_reason=normalized_reason,
        review_payload=(
            None if review_payload is None else dict(review_payload)
        ),
    )
    updated_attempt = WorkerAttempt.model_validate(attempt_payload)
    updated_slot = WorkerSlot(
        assignment_key=slot.assignment_key,
        attempts=(*slot.attempts[:-1], updated_attempt),
    )
    return WorkerGroupMutation(
        group=_replace_slot(
            group,
            slot_index=index,
            slot=updated_slot,
            changed_at=timestamp,
        ),
        applied=True,
    )


def replace_worker_attempt(
    group: StepWorkerGroup,
    *,
    assignment_key: str,
    attempt_no: int,
    worker_id: str,
    replacement_objective: str,
    reason: str,
    review_payload: Mapping[str, Any] | None = None,
    workspace_retention_minutes: int = 24 * 60,
    changed_at: datetime | None = None,
) -> WorkerGroupMutation:
    """Close the current generation as REPLACED and append a fresh PENDING one."""

    normalized_objective = _require_text(
        replacement_objective,
        field_name="replacement_objective",
    )
    normalized_reason = _require_text(reason, field_name="reason")
    index = _slot_index(group, assignment_key)
    slot = group.slots[index]
    attempt = slot.current_attempt
    if attempt_no != attempt.attempt_no or worker_id != attempt.worker_id:
        return WorkerGroupMutation(
            group=group,
            applied=False,
            stale=_callback_is_stale(
                slot,
                attempt_no=attempt_no,
                worker_id=worker_id,
            ),
        )
    if attempt.status not in ACTIVE_ATTEMPT_STATUSES:
        raise ValueError(
            f"Cannot replace Worker attempt in {attempt.status.value}."
        )
    timestamp = _transition_time(changed_at)
    retention = _workspace_retention(workspace_retention_minutes)
    replaced_payload = attempt.model_dump()
    replaced_payload.update(
        workspace=_frozen_workspace(
            attempt.workspace,
            changed_at=timestamp,
            retention=retention,
        ),
        status=WorkerAttemptStatus.REPLACED,
        status_changed_at=timestamp,
        terminal_reason=normalized_reason,
        review_payload=(
            None if review_payload is None else dict(review_payload)
        ),
    )
    replaced_attempt = WorkerAttempt.model_validate(replaced_payload)
    next_attempt_no = attempt.attempt_no + 1
    next_attempt = WorkerAttempt(
        attempt_no=next_attempt_no,
        worker_id=_worker_id(
            group_id=group.group_id,
            assignment_key=slot.assignment_key,
            attempt_no=next_attempt_no,
        ),
        objective=normalized_objective,
        workspace=_workspace(
            event_id=group.event_id,
            group_id=group.group_id,
            assignment_key=slot.assignment_key,
            attempt_no=next_attempt_no,
            created_at=timestamp,
        ),
        status=WorkerAttemptStatus.PENDING,
        created_at=timestamp,
        status_changed_at=timestamp,
    )
    updated_slot = WorkerSlot(
        assignment_key=slot.assignment_key,
        attempts=(*slot.attempts[:-1], replaced_attempt, next_attempt),
    )
    return WorkerGroupMutation(
        group=_replace_slot(
            group,
            slot_index=index,
            slot=updated_slot,
            changed_at=timestamp,
        ),
        applied=True,
    )


__all__ = [
    "ACTIVE_ATTEMPT_STATUSES",
    "COMPLETION_TERMINAL_STATUSES",
    "StepWorkerGroup",
    "WorkerAttempt",
    "WorkerAttemptStatus",
    "WorkerGroupMutation",
    "WorkerGroupReviewStatus",
    "WorkerSlot",
    "WorkerWorkspace",
    "WorkerWorkspaceStatus",
    "create_step_worker_group",
    "finish_worker_attempt",
    "replace_worker_attempt",
    "start_worker_attempt",
]
