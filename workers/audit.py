"""Bounded audit envelopes for Worker attempt lifecycle events."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


WorkerAuditOperation = Literal[
    "START",
    "FINISH",
    "REPLACE",
    "CALLBACK",
    "ARTIFACT_PUBLISH",
]

WorkerAuditOutcome = Literal[
    "APPLIED",
    "IDEMPOTENT",
    "STALE",
    "REJECTED",
]


def _as_utc(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include timezone information.")
    return value.astimezone(timezone.utc)


class WorkerAttemptAuditRecord(BaseModel):
    """Small durable index pointing to a checkpoint or observability trace.

    Raw model messages and tool results intentionally do not live here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    audit_id: str = Field(min_length=1, max_length=160)
    group_id: str = Field(min_length=1)
    assignment_key: str = Field(min_length=1)
    attempt_no: int = Field(ge=1)
    worker_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    checkpoint_thread_id: str = Field(min_length=1)
    operation: WorkerAuditOperation
    outcome: WorkerAuditOutcome
    occurred_at: datetime
    recorded_at: datetime
    summary: str = Field(min_length=1, max_length=2000)
    trace_id: str | None = Field(default=None, max_length=160)
    status_before: str | None = Field(default=None, max_length=80)
    status_after: str | None = Field(default=None, max_length=80)

    @model_validator(mode="after")
    def validate_times(self) -> "WorkerAttemptAuditRecord":
        occurred_at = _as_utc(self.occurred_at, field_name="occurred_at")
        recorded_at = _as_utc(self.recorded_at, field_name="recorded_at")
        object.__setattr__(self, "occurred_at", occurred_at)
        object.__setattr__(self, "recorded_at", recorded_at)
        return self


__all__ = [
    "WorkerAttemptAuditRecord",
    "WorkerAuditOperation",
    "WorkerAuditOutcome",
]
