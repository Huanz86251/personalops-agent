"""Durable contracts for promoting reviewed files to a user workspace."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DeliveryApprovalMode(str, Enum):
    HUMAN = "human"
    AUTO = "auto"


class PromotionStatus(str, Enum):
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    APPROVED = "APPROVED"
    PROMOTING = "PROMOTING"
    DELIVERED = "DELIVERED"
    REJECTED = "REJECTED"
    SUPERSEDED = "SUPERSEDED"
    FAILED = "FAILED"


PROMOTION_STATUS_TRANSITIONS: dict[
    PromotionStatus,
    frozenset[PromotionStatus],
] = {
    PromotionStatus.AWAITING_APPROVAL: frozenset(
        {
            PromotionStatus.APPROVED,
            PromotionStatus.REJECTED,
            PromotionStatus.SUPERSEDED,
        }
    ),
    PromotionStatus.APPROVED: frozenset(
        {
            PromotionStatus.PROMOTING,
            PromotionStatus.REJECTED,
            PromotionStatus.SUPERSEDED,
            PromotionStatus.FAILED,
        }
    ),
    PromotionStatus.PROMOTING: frozenset(
        {
            PromotionStatus.DELIVERED,
            PromotionStatus.FAILED,
        }
    ),
    PromotionStatus.DELIVERED: frozenset(),
    PromotionStatus.REJECTED: frozenset(),
    PromotionStatus.SUPERSEDED: frozenset(),
    PromotionStatus.FAILED: frozenset(),
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_relative_path(value: str) -> str:
    normalized = str(value).strip().replace("\\", "/")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(value)
    if (
        not normalized
        or normalized == "."
        or posix.is_absolute()
        or windows.is_absolute()
        or ".." in posix.parts
        or any(":" in part for part in posix.parts)
        or any(part in {".agent", ".git"} for part in posix.parts)
    ):
        raise ValueError(f"unsafe workspace path: {value}")
    return posix.as_posix()


class DeliveryModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class WorkspaceFileManifestEntry(DeliveryModel):
    source_path: str | None = None
    path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path", "source_path")
    @classmethod
    def validate_path(cls, value: str | None) -> str | None:
        return _safe_relative_path(value) if value is not None else None


class WorkspacePromotion(DeliveryModel):
    """One immutable reviewed version moving toward user-visible delivery."""

    promotion_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    status: PromotionStatus
    approval_mode: DeliveryApprovalMode
    source_root: str = Field(min_length=1)
    source_commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    source_commit_root: str | None = None
    target_root: str = Field(min_length=1)
    target_base_revision: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    manifest_id: str = Field(min_length=1)
    files: tuple[WorkspaceFileManifestEntry, ...] = Field(min_length=1)
    review_summary: str = Field(min_length=1)
    test_summary: str = Field(min_length=1)
    created_at: datetime
    updated_at: datetime
    decided_at: datetime | None = None
    decided_by: str | None = None
    decision_reason: str | None = None
    delivered_commit: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{40}$",
    )
    failure_reason: str | None = None

    @field_validator("files")
    @classmethod
    def validate_files(
        cls,
        values: tuple[WorkspaceFileManifestEntry, ...],
    ) -> tuple[WorkspaceFileManifestEntry, ...]:
        paths = tuple(item.path for item in values)
        if len(paths) != len(set(paths)):
            raise ValueError("workspace promotion paths must be unique")
        if paths != tuple(sorted(paths)):
            raise ValueError("workspace promotion paths must be sorted")
        return values

    @model_validator(mode="after")
    def validate_state_fields(self) -> "WorkspacePromotion":
        if self.run_id != self.event_id:
            raise ValueError("first-version promotion requires run_id == event_id")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot be before created_at")
        source_commit_root = self.source_commit_root
        if self.source_commit is not None and source_commit_root is None:
            # Backward compatibility for promotions created before mixed
            # Code/Web delivery: source_root itself was the Git repository.
            source_commit_root = "."
            object.__setattr__(self, "source_commit_root", source_commit_root)
        if self.source_commit is None and source_commit_root is not None:
            raise ValueError("source_commit_root requires source_commit")
        if source_commit_root is not None and source_commit_root != ".":
            self_source_commit_root = _safe_relative_path(source_commit_root)
            object.__setattr__(self, "source_commit_root", self_source_commit_root)
        decision_required = self.status in {
            PromotionStatus.APPROVED,
            PromotionStatus.PROMOTING,
            PromotionStatus.DELIVERED,
            PromotionStatus.REJECTED,
        }
        decision_fields = (self.decided_at, self.decided_by)
        if decision_required and any(value is None for value in decision_fields):
            raise ValueError("decided promotion requires decision metadata")
        if (self.decided_at is None) != (self.decided_by is None):
            raise ValueError("promotion decision metadata must be complete")
        if self.status is PromotionStatus.DELIVERED:
            if self.delivered_commit is None:
                raise ValueError("DELIVERED promotion requires delivered_commit")
        elif self.delivered_commit is not None:
            raise ValueError("only DELIVERED promotion may contain delivered_commit")
        if self.status is PromotionStatus.FAILED:
            if not self.failure_reason:
                raise ValueError("FAILED promotion requires failure_reason")
        elif self.failure_reason is not None:
            raise ValueError("only FAILED promotion may contain failure_reason")
        return self


def _manifest_identity(
    files: tuple[WorkspaceFileManifestEntry, ...],
) -> str:
    payload = json.dumps(
        [item.model_dump(mode="json") for item in files],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def create_workspace_promotion(
    *,
    event_id: str,
    conversation_id: str,
    approval_mode: DeliveryApprovalMode,
    source_root: str,
    source_commit: str | None,
    source_commit_root: str | None = None,
    target_root: str,
    target_base_revision: str,
    files: tuple[WorkspaceFileManifestEntry, ...],
    review_summary: str,
    test_summary: str,
    created_at: datetime | None = None,
) -> WorkspacePromotion:
    ordered = tuple(sorted(files, key=lambda item: item.path))
    manifest_digest = _manifest_identity(ordered)
    identity = (
        f"{event_id}:{conversation_id}:{source_commit or '-'}:"
        f"{target_base_revision}:{manifest_digest}"
    )
    promotion_digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    timestamp = created_at or _now()
    return WorkspacePromotion(
        promotion_id=f"promotion-{promotion_digest[:24]}",
        event_id=event_id,
        run_id=event_id,
        conversation_id=conversation_id,
        status=PromotionStatus.AWAITING_APPROVAL,
        approval_mode=approval_mode,
        source_root=source_root,
        source_commit=source_commit,
        source_commit_root=source_commit_root,
        target_root=target_root,
        target_base_revision=target_base_revision,
        manifest_id=f"workspace-manifest-{manifest_digest[:24]}",
        files=ordered,
        review_summary=review_summary,
        test_summary=test_summary,
        created_at=timestamp,
        updated_at=timestamp,
    )


def transition_workspace_promotion(
    promotion: WorkspacePromotion,
    new_status: PromotionStatus,
    *,
    changed_at: datetime | None = None,
    decided_by: str | None = None,
    decision_reason: str | None = None,
    delivered_commit: str | None = None,
    failure_reason: str | None = None,
) -> WorkspacePromotion:
    if new_status not in PROMOTION_STATUS_TRANSITIONS[promotion.status]:
        raise ValueError(
            "illegal promotion transition: "
            f"{promotion.status.value} -> {new_status.value}"
        )
    timestamp = changed_at or _now()
    values: dict[str, Any] = {
        "status": new_status,
        "updated_at": timestamp,
    }
    if new_status in {
        PromotionStatus.APPROVED,
        PromotionStatus.REJECTED,
    }:
        values.update(
            decided_at=timestamp,
            decided_by=str(decided_by or "").strip(),
            decision_reason=(str(decision_reason).strip() if decision_reason else None),
        )
    elif promotion.decided_at is not None:
        values.update(
            decided_at=promotion.decided_at,
            decided_by=promotion.decided_by,
            decision_reason=promotion.decision_reason,
        )
    if new_status is PromotionStatus.DELIVERED:
        values["delivered_commit"] = delivered_commit
    if new_status is PromotionStatus.FAILED:
        values["failure_reason"] = str(failure_reason or "").strip()
    payload = promotion.model_dump(mode="python")
    payload.update(values)
    return WorkspacePromotion.model_validate(payload)


__all__ = [
    "DeliveryApprovalMode",
    "PromotionStatus",
    "WorkspaceFileManifestEntry",
    "WorkspacePromotion",
    "create_workspace_promotion",
    "transition_workspace_promotion",
]
