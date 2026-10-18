"""Typed control-plane contracts shared by the leader model and Workers."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from schema_normalization import (
    EvidenceRefText,
    coerce_text_sequence,
    drop_invalid_auxiliary_fields,
)

LeadershipAction = Literal[
    "CONTINUE",
    "GUIDE",
    "ACCEPT",
    "CANCEL",
    "REPLACE",
]

LeadershipWakeReason = Literal[
    "SINGLE_WORKER_THRESHOLD",
    "MULTI_WORKER_BARRIER",
    "WORKER_BLOCKED",
    "WORKER_POSSIBLY_READY",
    "WORKER_FAILED",
    "WORKER_SILENT",
    "USER_CONTROL",
]


class LeadershipDecision(BaseModel):
    """The only five actions the main model may take at a safe point."""

    reason: str = Field(default="", max_length=1200, description="可选观察依据；格式错误时Harness会省略，action及条件字段仍严格校验。")
    action: LeadershipAction
    target_worker_ids: list[EvidenceRefText] = Field(default_factory=list, max_length=3)
    guidance: str | None = Field(default=None, max_length=2000)
    replacement_assignment: str | None = Field(default=None, max_length=4000)

    @field_validator("target_worker_ids", mode="before")
    @classmethod
    def normalize_target_worker_ids(cls, value):
        return coerce_text_sequence(value)

    @model_validator(mode="before")
    @classmethod
    def omit_invalid_auxiliary_fields(cls, value):
        return drop_invalid_auxiliary_fields(cls, value, ("reason",))

    @model_validator(mode="after")
    def validate_action_payload(self) -> "LeadershipDecision":
        self.target_worker_ids = list(
            dict.fromkeys(
                worker_id.strip()
                for worker_id in self.target_worker_ids
                if worker_id.strip()
            )
        )
        if self.action == "GUIDE" and not (self.guidance or "").strip():
            raise ValueError("GUIDE requires guidance.")
        if self.action == "REPLACE" and not (
            self.replacement_assignment or ""
        ).strip():
            raise ValueError("REPLACE requires replacement_assignment.")
        if self.action not in {"GUIDE", "REPLACE"}:
            self.guidance = None
            self.replacement_assignment = None
        return self


class LeadershipWorkerView(BaseModel):
    """Unread progress and assignment for one active Worker."""

    worker_id: str
    step_id: str | None = None
    assignment: str
    cursor_before: int = Field(ge=0)
    reports: list[dict[str, Any]]


class LeadershipWakeRequest(BaseModel):
    """One coalesced event presented to the main leadership model."""

    wake_id: str
    event_id: str
    reason: LeadershipWakeReason
    created_at: datetime
    workers: list[LeadershipWorkerView] = Field(min_length=1, max_length=3)
    triggering_worker_id: str | None = None


class LeadershipDecisionResult(BaseModel):
    """Decision plus accounting added by the deterministic harness."""

    decision: LeadershipDecision
    model_rounds_used: int = Field(ge=0)
    used_fallback: bool = False
    wake_id: str | None = None


__all__ = [
    "LeadershipAction",
    "LeadershipDecision",
    "LeadershipDecisionResult",
    "LeadershipWakeReason",
    "LeadershipWakeRequest",
    "LeadershipWorkerView",
]
