"""Harness-authored terminal records for Workers stopped without submission."""

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class WorkerCancellationRecord(BaseModel):
    """Bounded cancellation evidence created without another model call."""

    cancelled_at: datetime
    worker_id: str
    event_id: str | None = None
    step_id: str | None = None
    workspace_id: str | None = None
    reason: str = Field(min_length=1, max_length=1200)
    total_tool_calls: int = Field(ge=0)
    latest_progress: dict[str, Any] | None = None


def build_worker_cancellation_record(
    state: Mapping[str, Any],
    *,
    reason: str,
) -> WorkerCancellationRecord:
    """Materialize the latest durable state after cooperative cancellation."""

    worker_id = str(state.get("worker_id") or "").strip()
    if not worker_id:
        raise ValueError("Cancellation record requires worker_id.")
    reports = state.get("worker_progress_reports")
    latest_progress = None
    if isinstance(reports, list) and reports:
        candidate = reports[-1]
        if isinstance(candidate, Mapping):
            latest_progress = dict(candidate)
    normalized_reason = " ".join(str(reason).strip().split())
    if not normalized_reason:
        normalized_reason = "Worker was cancelled by leadership."
    return WorkerCancellationRecord(
        cancelled_at=datetime.now(timezone.utc),
        worker_id=worker_id,
        event_id=(str(state.get("event_id")) if state.get("event_id") else None),
        step_id=(str(state.get("step_id")) if state.get("step_id") else None),
        workspace_id=(
            str(state.get("workspace_id"))
            if state.get("workspace_id")
            else None
        ),
        reason=normalized_reason,
        total_tool_calls=max(
            int(state.get("worker_total_tool_calls", 0) or 0),
            0,
        ),
        latest_progress=latest_progress,
    )


__all__ = [
    "WorkerCancellationRecord",
    "build_worker_cancellation_record",
]
