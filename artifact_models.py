"""Typed artifact identities shared by Web Workers and Step review."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator


WorkerArtifactKind = Literal[
    "WORKSPACE_FILE",
    "DOWNLOADED_FILE",
    "REMOTE_REPOSITORY",
]


class DownloadedArtifactRecord(BaseModel):
    """Harness-authored facts for one bounded Web download."""

    candidate_id: str = Field(min_length=1, max_length=120)
    source_url: str
    storage_path: str
    filename: str = Field(min_length=1, max_length=240)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: str | None = None
    tool_call_id: str = Field(min_length=1)
    downloaded_at: datetime
    run_id: str | None = None
    worker_id: str | None = None
    origin: Literal["WEB", "BROWSER", "EMAIL"] = "WEB"

    @field_validator("candidate_id", "source_url", "storage_path", "filename")
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("download record text cannot be empty")
        return normalized


class WorkerArtifactCandidate(BaseModel):
    """A Worker proposal. It is not a published artifact."""

    candidate_id: str = Field(min_length=1, max_length=120)
    output_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_-]*$",
    )
    kind: WorkerArtifactKind
    description: str = Field(min_length=1, max_length=600)
    path: str | None = None
    repository_url: str | None = None
    repository_ref: str | None = None
    commit: str | None = None
    relevant_paths: list[str] = Field(default_factory=list, max_length=20)
    evidence_tool_call_ids: list[str] = Field(default_factory=list)

    @field_validator("candidate_id", "description")
    @classmethod
    def normalize_candidate_text(cls, value: str) -> str:
        normalized = " ".join(str(value).strip().split())
        if not normalized:
            raise ValueError("artifact candidate text cannot be empty")
        return normalized

    @field_validator("evidence_tool_call_ids", "relevant_paths")
    @classmethod
    def normalize_unique_lists(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            normalized = str(value).strip().replace("\\", "/")
            if normalized and normalized not in result:
                result.append(normalized)
        return result

    @model_validator(mode="after")
    def validate_kind_fields(self) -> "WorkerArtifactCandidate":
        if self.kind == "WORKSPACE_FILE":
            raw_path = str(self.path or "").strip().replace("\\", "/")
            parts = [part for part in raw_path.split("/") if part]
            if not parts or raw_path.startswith(("//", "~")) or ".." in parts:
                raise ValueError("WORKSPACE_FILE requires a safe workspace path")
            self.path = "/" + "/".join(parts)
            if self.repository_url:
                raise ValueError("WORKSPACE_FILE cannot declare repository_url")
        elif self.kind == "DOWNLOADED_FILE":
            if self.path or self.repository_url:
                raise ValueError(
                    "DOWNLOADED_FILE is resolved by candidate_id, not model paths"
                )
        else:
            if self.path:
                raise ValueError("REMOTE_REPOSITORY cannot declare a local path")
            parsed = urlparse(str(self.repository_url or "").strip())
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError(
                    "REMOTE_REPOSITORY requires an absolute HTTP or HTTPS URL"
                )
            self.repository_url = parsed.geturl()
            if not self.evidence_tool_call_ids:
                raise ValueError("REMOTE_REPOSITORY requires Tool Call evidence")
        return self


class ResolvedArtifactCandidate(BaseModel):
    """Harness-resolved candidate facts visible to the Step Reporter."""

    candidate_id: str
    output_id: str | None = None
    review_ref: str | None = None
    kind: WorkerArtifactKind
    description: str
    evidence_tool_call_ids: list[str] = Field(default_factory=list)
    verified: bool = True
    location: str
    storage_path: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    sha256: str | None = None
    media_type: str | None = None
    source_url: str | None = None
    repository_ref: str | None = None
    commit: str | None = None
    relevant_paths: list[str] = Field(default_factory=list)


class WebDownloadFailure(BaseModel):
    """Machine-readable reason returned by the download Tool."""

    code: Literal[
        "INVALID_URL",
        "CREDENTIALS_IN_URL",
        "FILE_TOO_LARGE",
        "HTTP_ERROR",
        "IO_ERROR",
        "MISSING_RUNTIME_IDENTITY",
        "UNSAFE_DESTINATION",
        "UNEXPECTED_CONTENT",
    ]
    stage: Literal["VALIDATION", "RESPONSE_HEADERS", "STREAM", "WRITE", "RUNTIME"]
    message: str
    limit_bytes: int | None = Field(default=None, ge=1)
    announced_bytes: int | None = Field(default=None, ge=0)
    observed_bytes: int | None = Field(default=None, ge=0)
    retryable: bool = False


class WebDownloadToolResult(BaseModel):
    """Stable ToolMessage and custom-stream result envelope."""

    status: Literal["DOWNLOADED", "REJECTED", "FAILED"]
    artifact_candidate: WorkerArtifactCandidate | None = None
    filename: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    sha256: str | None = None
    source_url: str | None = None
    error: WebDownloadFailure | None = None


__all__ = [
    "DownloadedArtifactRecord",
    "ResolvedArtifactCandidate",
    "WebDownloadFailure",
    "WebDownloadToolResult",
    "WorkerArtifactCandidate",
    "WorkerArtifactKind",
]
