"""Durable resource checkpoints for one Code Worker/Reviewer attempt.

LangGraph checkpoints own model-visible state.  This module owns the separate
resource-plane record needed to find Docker containers, named volumes, host
snapshots, and the accepted integration baseline after a process restart.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from planning_models import CodeTaskContract
from workers.code_review_models import (
    CodeCandidateRef,
    CodeReviewLoopState,
    CodeReviewReport,
)
from workers.docker_sandbox import CodeSandboxPair


CodeRuntimePhase = Literal[
    "WORKER_RUNNING",
    "CANDIDATE_READY",
    "REVIEWER_RUNNING",
    "WAITING_FOR_WORKER",
    "AWAITING_SCHEDULER",
    "REVIEW_COMPLETED",
    "RECOVERY_REQUIRED",
    "TERMINAL",
]
CodePreemptionPolicy = Literal[
    "SAFE_POINT",
    "DEFER_UNTIL_ROLE_BOUNDARY",
]


class CodeRuntimeCheckpointModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class CodeSandboxPairRecord(CodeRuntimeCheckpointModel):
    """JSON-safe identity of resources created by ``CodeSandboxManager``."""

    pair_id: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    candidate_volume: str = Field(min_length=1)
    review_volume: str = Field(min_length=1)
    worker_container: str = Field(min_length=1)
    reviewer_container: str = Field(min_length=1)
    image: str = Field(min_length=1)
    active_role: Literal["WORKER", "REVIEWER"] | None
    handoff_root: str | None = None

    @classmethod
    def from_pair(cls, pair: CodeSandboxPair) -> "CodeSandboxPairRecord":
        return cls(
            pair_id=pair.pair_id,
            workspace_id=pair.workspace_id,
            candidate_volume=pair.candidate_volume,
            review_volume=pair.review_volume,
            worker_container=pair.worker_container,
            reviewer_container=pair.reviewer_container,
            image=pair.image,
            active_role=pair.active_role,
            handoff_root=pair.handoff_root,
        )

    def to_pair(self) -> CodeSandboxPair:
        return CodeSandboxPair(**self.model_dump())


class CodeRuntimeRecoveryState(CodeRuntimeCheckpointModel):
    """Serializable state required to rebuild an in-memory runtime session."""

    contract: CodeTaskContract
    candidate: CodeCandidateRef
    review_loop: CodeReviewLoopState
    worker_submission: dict[str, Any] | None = None
    last_review_details: dict[str, Any] | None = None
    last_report: CodeReviewReport | None = None
    started_at: datetime
    parent_attempt_id: str | None = None
    worker_id: str = Field(min_length=1)
    superseded_attempt_records: tuple[dict[str, Any], ...] = ()

    @model_validator(mode="after")
    def validate_recovery_identity(self) -> "CodeRuntimeRecoveryState":
        timestamp = self.started_at
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("started_at must include timezone information")
        object.__setattr__(self, "started_at", timestamp.astimezone(timezone.utc))
        if self.candidate != self.review_loop.candidate:
            raise ValueError("Recovery candidate must match the review loop")
        return self


class CodeRuntimeCheckpoint(CodeRuntimeCheckpointModel):
    """Last committed role/resource boundary for one CODE attempt."""

    schema_version: int = Field(default=1, ge=1)
    checkpoint_id: str = Field(min_length=1)
    runtime_session_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    step_id: int = Field(ge=1)
    attempt_id: str = Field(min_length=1)
    generation: int = Field(default=1, ge=1)
    phase: CodeRuntimePhase
    preemption_policy: CodePreemptionPolicy
    worker_checkpoint_id: str = Field(min_length=1)
    reviewer_checkpoint_id: str = Field(min_length=1)
    integration_root: str = Field(min_length=1)
    integration_head_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    base_revision: str = Field(min_length=1)
    attempt_root: str = Field(min_length=1)
    candidate_snapshot_path: str | None = None
    reviewer_snapshot_path: str | None = None
    sandbox: CodeSandboxPairRecord
    recovery_state: CodeRuntimeRecoveryState | None = None
    cleanup_authorized: bool = False
    reason: str = ""
    committed_at: datetime

    @model_validator(mode="after")
    def validate_boundary(self) -> "CodeRuntimeCheckpoint":
        timestamp = self.committed_at
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("committed_at must include timezone information")
        object.__setattr__(self, "committed_at", timestamp.astimezone(timezone.utc))
        expected_policy = (
            "DEFER_UNTIL_ROLE_BOUNDARY"
            if self.phase == "REVIEWER_RUNNING"
            else "SAFE_POINT"
        )
        if self.preemption_policy != expected_policy:
            raise ValueError(
                f"{self.phase} requires preemption_policy={expected_policy}"
            )
        if self.cleanup_authorized and self.phase != "TERMINAL":
            raise ValueError("Only a TERMINAL checkpoint may authorize cleanup")
        if self.cleanup_authorized and (
            not self.candidate_snapshot_path or not self.reviewer_snapshot_path
        ):
            raise ValueError(
                "Cleanup authorization requires candidate and Reviewer snapshots"
            )
        if self.phase == "AWAITING_SCHEDULER" and self.recovery_state is None:
            raise ValueError(
                "AWAITING_SCHEDULER requires a serializable recovery state"
            )
        return self


class CodeRuntimeCheckpointStore:
    """Atomically publish and verify one attempt's latest resource record."""

    filename = "runtime_checkpoint.json"

    @classmethod
    def path_for(cls, attempt_root: Path) -> Path:
        return Path(attempt_root).resolve() / cls.filename

    def load(self, attempt_root: Path) -> CodeRuntimeCheckpoint | None:
        path = self.path_for(attempt_root)
        if not path.is_file():
            return None
        return CodeRuntimeCheckpoint.model_validate_json(
            path.read_text(encoding="utf-8")
        )

    def commit(
        self,
        attempt_root: Path,
        checkpoint: CodeRuntimeCheckpoint,
    ) -> CodeRuntimeCheckpoint:
        """Write, fsync, atomically replace, and read back the checkpoint."""

        root = Path(attempt_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        destination = self.path_for(root)
        temporary = root / f".{self.filename}.{uuid4().hex}.tmp"
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(checkpoint.model_dump_json(indent=2))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

        persisted = self.load(root)
        if persisted is None or persisted.checkpoint_id != checkpoint.checkpoint_id:
            raise RuntimeError("CODE runtime checkpoint did not survive read-back")
        return persisted

    def require_cleanup_authorized(
        self,
        attempt_root: Path,
        *,
        pair_id: str,
    ) -> CodeRuntimeCheckpoint:
        checkpoint = self.load(attempt_root)
        if checkpoint is None:
            raise RuntimeError("Docker cleanup requires a committed checkpoint")
        if not checkpoint.cleanup_authorized:
            raise RuntimeError("Docker cleanup is not authorized by the checkpoint")
        if checkpoint.sandbox.pair_id != pair_id:
            raise RuntimeError("Docker cleanup checkpoint belongs to another pair")
        root = Path(attempt_root).resolve()
        required = (
            checkpoint.candidate_snapshot_path,
            checkpoint.reviewer_snapshot_path,
            "final_record.json",
        )
        for relative in required:
            candidate = (root / str(relative)).resolve()
            if not candidate.is_relative_to(root) or not candidate.exists():
                raise RuntimeError(
                    f"Docker cleanup evidence is missing: {relative}"
                )
        return checkpoint


__all__ = [
    "CodePreemptionPolicy",
    "CodeRuntimeCheckpoint",
    "CodeRuntimeCheckpointStore",
    "CodeRuntimeRecoveryState",
    "CodeRuntimePhase",
    "CodeSandboxPairRecord",
]
