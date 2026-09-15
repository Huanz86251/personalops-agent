"""Bounded inputs for the independent Step Reporter."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field

from workers.submission import (
    ResolvedToolEvidence,
    WorkerCriterionClaim,
)
from artifact_models import ResolvedArtifactCandidate
from planning_models import StepArtifactOutput
from workers.plan_challenge import PlanChallenge


ReviewSubmissionSource = Literal[
    "WORKER",
    "HARNESS_FALLBACK",
]


class ReviewTaskContract(BaseModel):
    """The immutable contract against which one Step is reviewed."""

    user_request: str
    plan_objective: str
    step_id: int = Field(ge=1)
    execution_guidance: str | None = None
    step_assignment: str
    success_criteria: list[str]
    artifact_outputs: list[StepArtifactOutput] = Field(default_factory=list)

    @computed_field
    @property
    def criterion_registry(self) -> dict[str, str]:
        return {f"C{i}": text for i, text in enumerate(self.success_criteria, 1)}


class ReviewAttempt(BaseModel):
    """One Worker attempt represented without its full conversation."""

    attempt: int = Field(ge=1)
    assignment_key: str = "primary"
    assignment_objective: str = ""
    worker_id: str | None = None
    submission_source: ReviewSubmissionSource
    finish_reason: str
    stop_reason: str
    summary: str
    final_conclusion: str
    criterion_claims: list[WorkerCriterionClaim] = Field(
        default_factory=list,
    )
    resolved_evidence: list[ResolvedToolEvidence] = Field(default_factory=list)
    resolved_artifacts: list[ResolvedArtifactCandidate] = Field(
        default_factory=list,
    )
    unresolved_items: list[str] = Field(default_factory=list)
    tool_audit: list[dict[str, Any]] = Field(default_factory=list)
    tool_audit_available: bool = False
    handoff_knowledge: list[dict[str, Any]] = Field(default_factory=list)
    handoff_apis: list[dict[str, Any]] = Field(default_factory=list)
    execution_metrics: dict[str, Any] = Field(default_factory=dict)
    applied_limits: dict[str, Any] = Field(default_factory=dict)
    leadership_terminal_action: str | None = None
    final_reviewer_request: dict[str, Any] | None = None
    plan_challenge: PlanChallenge | None = None


class StepReviewPacket(BaseModel):
    """The only execution context visible to the Step Reporter."""

    created_at: datetime
    task_contract: ReviewTaskContract
    attempts: list[ReviewAttempt] = Field(min_length=1)
    previous_step_report: dict[str, Any] | None = None
    remaining_budget: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "ReviewAttempt",
    "ReviewSubmissionSource",
    "ReviewTaskContract",
    "StepReviewPacket",
]
