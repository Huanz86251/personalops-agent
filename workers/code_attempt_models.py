"""Durable terminal records for one Code Worker/Reviewer attempt.

The review loop decides whether a candidate is acceptable. This module is a
separate control-plane contract describing what survives after the sandbox is
frozen and whether that exact candidate was actually published.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from integration_repository import IntegrationCommitReceipt
from .code_review_models import (
    CodeCandidateRef,
    CodeReviewReport,
    SchedulerCodeDecision,
)


CodeAttemptOutcome = Literal["CANCELLED", "SUPERSEDED", "FAILED", "APPLIED"]
CodeAttemptFailureStage = Literal[
    "WORKER", "REVIEWER", "FINALIZER", "PUBLISHER"
]
CodeArtifactKind = Literal[
    "SOURCE", "TEST", "CONFIG", "BUILD_OUTPUT", "DOCUMENTATION", "OTHER"
]


def _as_utc(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include timezone information.")
    return value.astimezone(timezone.utc)


def _relative_archive_path(value: str, *, field_name: str) -> str:
    normalized = value.strip().replace("\\", "/")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(value)
    if (
        not normalized
        or normalized == "."
        or posix.is_absolute()
        or windows.is_absolute()
        or ".." in posix.parts
        or any(":" in part for part in posix.parts)
    ):
        raise ValueError(f"{field_name} must be a safe relative path.")
    return posix.as_posix()


class CodeAttemptModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class CodeArtifactEntry(CodeAttemptModel):
    """One publishable file from the frozen candidate workspace."""

    path: str = Field(min_length=1)
    kind: CodeArtifactKind
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _relative_archive_path(value, field_name="path")


class CodeArtifactManifest(CodeAttemptModel):
    """Content-addressed inventory of files eligible for publication."""

    manifest_id: str = Field(min_length=1)
    candidate: CodeCandidateRef
    files: tuple[CodeArtifactEntry, ...] = ()
    created_at: datetime

    @model_validator(mode="after")
    def validate_manifest(self) -> "CodeArtifactManifest":
        created_at = _as_utc(self.created_at, field_name="created_at")
        paths = [entry.path for entry in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("Artifact manifest paths must be unique.")
        object.__setattr__(self, "created_at", created_at)
        return self


class CodeAttemptArchive(CodeAttemptModel):
    """Host-side evidence exported before containers and volumes are removed."""

    archive_id: str = Field(min_length=1)
    root_path: str = Field(min_length=1)
    candidate_snapshot_path: str = Field(min_length=1)
    reviewer_snapshot_path: str = Field(min_length=1)
    preserved_at: datetime
    retain_until: datetime

    @field_validator("candidate_snapshot_path", "reviewer_snapshot_path")
    @classmethod
    def validate_relative_snapshot_path(cls, value: str) -> str:
        return _relative_archive_path(value, field_name="snapshot path")

    @field_validator("preserved_at", "retain_until")
    @classmethod
    def validate_archive_time(cls, value: datetime) -> datetime:
        return _as_utc(value, field_name="archive time")

    @model_validator(mode="after")
    def validate_retention(self) -> "CodeAttemptArchive":
        if self.retain_until < self.preserved_at:
            raise ValueError("retain_until cannot be before preserved_at.")
        return self


class CodePublicationReceipt(CodeAttemptModel):
    """Proof that a deterministic Publisher applied one frozen manifest."""

    publication_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    candidate: CodeCandidateRef
    manifest_id: str = Field(min_length=1)
    target_root: str = Field(min_length=1)
    base_revision: str = Field(min_length=1)
    applied_revision: str = Field(min_length=1)
    applied_at: datetime

    @field_validator("applied_at")
    @classmethod
    def validate_applied_at(cls, value: datetime) -> datetime:
        return _as_utc(value, field_name="applied_at")


class CodeAttemptFinalRecord(CodeAttemptModel):
    """Immutable terminal summary; raw model/tool traces live elsewhere."""

    record_id: str = Field(min_length=1)
    candidate: CodeCandidateRef
    parent_attempt_id: str | None = Field(default=None, min_length=1)
    outcome: CodeAttemptOutcome
    terminal_reason: str = Field(min_length=1)
    failure_stage: CodeAttemptFailureStage | None = None
    started_at: datetime
    finalized_at: datetime
    worker_checkpoint_id: str = Field(min_length=1)
    reviewer_checkpoint_id: str = Field(min_length=1)
    docker_image: str = Field(min_length=1)
    archive: CodeAttemptArchive | None
    artifact_manifest: CodeArtifactManifest
    review_report: CodeReviewReport | None = None
    scheduler_decision: SchedulerCodeDecision | None = None
    publication: CodePublicationReceipt | None = None
    integration_commit: IntegrationCommitReceipt | None = None

    @model_validator(mode="after")
    def validate_terminal_record(self) -> "CodeAttemptFinalRecord":
        started_at = _as_utc(self.started_at, field_name="started_at")
        finalized_at = _as_utc(self.finalized_at, field_name="finalized_at")
        if finalized_at < started_at:
            raise ValueError("finalized_at cannot be before started_at.")
        if self.archive is not None and self.archive.preserved_at > finalized_at:
            raise ValueError("Archive cannot be preserved after finalization.")
        if self.parent_attempt_id == self.candidate.attempt_id:
            raise ValueError("An attempt cannot name itself as its parent.")
        if self.artifact_manifest.candidate != self.candidate:
            raise ValueError("Artifact manifest must describe the final candidate.")
        if (
            self.review_report is not None
            and self.review_report.candidate != self.candidate
        ):
            raise ValueError("Review report must describe the final candidate.")

        if self.outcome == "APPLIED":
            if self.failure_stage is not None:
                raise ValueError("An APPLIED attempt cannot have failure_stage.")
            if self.review_report is None or self.review_report.verdict != "PASSED":
                raise ValueError("APPLIED requires a PASSED Reviewer report.")
            if self.scheduler_decision is not None:
                raise ValueError("APPLIED does not require a Scheduler decision.")
            if self.archive is None:
                raise ValueError("APPLIED requires a durable archive.")
            if self.publication is None:
                raise ValueError("APPLIED requires a Publisher receipt.")
            if self.publication.candidate != self.candidate:
                raise ValueError("Publisher receipt references a stale candidate.")
            if self.publication.manifest_id != self.artifact_manifest.manifest_id:
                raise ValueError("Publisher receipt references a different manifest.")
            if self.publication.applied_at > finalized_at:
                raise ValueError("Publication cannot complete after finalization.")
            if self.review_report.publication_id != self.publication.publication_id:
                raise ValueError("Reviewer report references another publication.")
            if self.review_report.applied_revision != self.publication.applied_revision:
                raise ValueError("Reviewer report references another applied revision.")
        else:
            if self.integration_commit is not None:
                raise ValueError("Only APPLIED may carry an integration commit.")
            if self.publication is not None:
                raise ValueError("Only APPLIED may carry a Publisher receipt.")
            if self.outcome == "FAILED" and self.failure_stage is None:
                raise ValueError("FAILED requires failure_stage.")
            if self.outcome == "CANCELLED" and self.failure_stage is not None:
                raise ValueError("CANCELLED must not be mislabeled as a failure.")
            if self.outcome == "SUPERSEDED" and self.failure_stage is not None:
                raise ValueError("SUPERSEDED must not be mislabeled as a failure.")
            if self.archive is None and not (
                self.outcome == "FAILED"
                and self.failure_stage == "FINALIZER"
            ):
                raise ValueError(
                    "Only a FINALIZER failure may finish without an archive."
                )

        object.__setattr__(self, "started_at", started_at)
        object.__setattr__(self, "finalized_at", finalized_at)
        return self


__all__ = [
    "CodeArtifactEntry",
    "CodeArtifactKind",
    "CodeArtifactManifest",
    "CodeAttemptArchive",
    "CodeAttemptFailureStage",
    "CodeAttemptFinalRecord",
    "CodeAttemptOutcome",
    "CodePublicationReceipt",
]
