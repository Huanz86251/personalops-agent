"""Deterministic publication tool used by the independent Code Reviewer."""

from __future__ import annotations
from runtime_tracing import operation

import hashlib
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field

from prompt_loader import load_prompt
from workers.code_attempt_models import (
    CodeArtifactEntry,
    CodeArtifactManifest,
    CodePublicationReceipt,
)
from workers.code_review_models import (
    CodeCandidateRef,
    CodeReviewLoopState,
    CodeWorkerSubmission,
    record_code_publication,
)


PUBLISH_REVIEWED_CANDIDATE_NAME = "publish_reviewed_candidate"
_IGNORED_TREE_PARTS = frozenset(
    {".agent", ".git", ".pytest_cache", ".venv", "__pycache__", "node_modules"}
)


class CodePublicationContext(BaseModel):
    """Harness-owned roots and optimistic concurrency fence."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    candidate_root: str = Field(min_length=1)
    target_root: str = Field(min_length=1)
    base_revision: str = Field(min_length=1)


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
        or any(part in _IGNORED_TREE_PARTS for part in posix.parts)
    ):
        raise ValueError(f"unsafe artifact path: {value}")
    return posix.as_posix()


def compute_code_tree_revision(root: Path) -> str:
    """Hash user-visible files without depending on Git or GitHub."""

    resolved = root.resolve()
    if not resolved.is_dir():
        raise ValueError(f"code tree root is not a directory: {resolved}")
    digest = hashlib.sha256()
    files = sorted(
        item
        for item in resolved.rglob("*")
        if item.is_file()
        and not any(part in _IGNORED_TREE_PARTS for part in item.relative_to(resolved).parts)
    )
    for item in files:
        relative = item.relative_to(resolved).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _artifact_kind(path: str) -> str:
    lowered = path.lower()
    name = PurePosixPath(lowered).name
    if lowered.startswith("tests/") or name.startswith("test_"):
        return "TEST"
    if (
        name in {"dockerfile", "requirements.txt", "pyproject.toml"}
        or PurePosixPath(lowered).suffix in {".json", ".toml", ".yaml", ".yml"}
    ):
        return "CONFIG"
    if lowered.startswith("docs/") or PurePosixPath(lowered).suffix == ".md":
        return "DOCUMENTATION"
    if lowered.startswith(("build/", "dist/")):
        return "BUILD_OUTPUT"
    if PurePosixPath(lowered).suffix in {
        ".css", ".go", ".html", ".java", ".js", ".jsx", ".py", ".rs", ".ts", ".tsx"
    }:
        return "SOURCE"
    return "OTHER"


def _manifest(
    candidate: CodeCandidateRef,
    source_root: Path,
    approved_paths: tuple[str, ...],
) -> CodeArtifactManifest:
    entries: list[CodeArtifactEntry] = []
    manifest_digest = hashlib.sha256()
    for relative in approved_paths:
        source = (source_root / Path(relative)).resolve()
        try:
            source.relative_to(source_root)
        except ValueError as error:
            raise ValueError(f"artifact escapes candidate root: {relative}") from error
        if not source.is_file():
            raise ValueError(f"approved artifact does not exist as a file: {relative}")
        payload = source.read_bytes()
        sha256 = hashlib.sha256(payload).hexdigest()
        entry = CodeArtifactEntry(
            path=relative,
            kind=_artifact_kind(relative),
            sha256=sha256,
            size_bytes=len(payload),
        )
        entries.append(entry)
        manifest_digest.update(relative.encode("utf-8"))
        manifest_digest.update(sha256.encode("ascii"))
    return CodeArtifactManifest(
        manifest_id=f"manifest-{manifest_digest.hexdigest()[:24]}",
        candidate=candidate,
        files=tuple(entries),
        created_at=datetime.now(timezone.utc),
    )


def _copy_manifest_transactionally(
    source_root: Path,
    target_root: Path,
    manifest: CodeArtifactManifest,
) -> None:
    staging = Path(
        tempfile.mkdtemp(
            prefix=".personalops-publish-",
            dir=str(target_root.parent),
        )
    )
    staged_root = staging / "files"
    backup_root = staging / "backup"
    applied: list[tuple[Path, Path | None]] = []
    try:
        for entry in manifest.files:
            source = source_root / Path(entry.path)
            staged = staged_root / Path(entry.path)
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, staged)
            staged_sha256 = hashlib.sha256(staged.read_bytes()).hexdigest()
            if staged_sha256 != entry.sha256:
                raise ValueError(
                    f"candidate changed while staging artifact: {entry.path}"
                )

        for entry in manifest.files:
            relative = Path(entry.path)
            destination = target_root / relative
            try:
                destination.resolve(strict=False).relative_to(target_root)
            except ValueError as error:
                raise ValueError(
                    f"publication target escapes target root: {entry.path}"
                ) from error
            if destination.exists() and not destination.is_file():
                raise ValueError(
                    f"publication target is not a regular file: {entry.path}"
                )
            backup = None
            if destination.exists():
                backup = backup_root / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(destination, backup)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_root / relative, destination)
            applied.append((destination, backup))
    except Exception:
        for destination, backup in reversed(applied):
            if destination.exists():
                destination.unlink()
            if backup is not None and backup.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(backup, destination)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


@operation('Code Publisher / Apply Artifacts', fields=('candidate', 'approved_artifact_paths'))
def publish_code_artifacts(
    *,
    candidate: CodeCandidateRef,
    approved_artifact_paths: tuple[str, ...],
    context: CodePublicationContext,
) -> tuple[CodeArtifactManifest, CodePublicationReceipt]:
    """Apply one approved candidate with a base-revision concurrency fence."""

    source_root = Path(context.candidate_root).resolve()
    target_root = Path(context.target_root).resolve()
    if not source_root.is_dir():
        raise ValueError("candidate_root must be an existing directory")
    if not target_root.is_dir():
        raise ValueError("target_root must be an existing directory")
    if source_root == target_root:
        raise ValueError("candidate_root and target_root must be different")

    approved = tuple(_safe_relative_path(path) for path in approved_artifact_paths)
    if len(approved) != len(set(approved)):
        raise ValueError("approved artifact paths must be unique")
    manifest = _manifest(candidate, source_root, approved)

    lock_token = hashlib.sha256(str(target_root).encode("utf-8")).hexdigest()[:16]
    lock_path = target_root.parent / f".personalops-publish-{lock_token}.lock"
    lock_fd: int | None = None
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        actual_base = compute_code_tree_revision(target_root)
        if actual_base != context.base_revision:
            raise ValueError(
                "publication base revision changed; "
                f"expected={context.base_revision}, actual={actual_base}"
            )
        _copy_manifest_transactionally(source_root, target_root, manifest)
        applied_revision = compute_code_tree_revision(target_root)
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
            lock_path.unlink(missing_ok=True)

    identity = (
        f"{candidate.event_id}:{candidate.step_id}:{candidate.attempt_id}:"
        f"{candidate.candidate_revision}:{manifest.manifest_id}"
    )
    publication_hash = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    receipt = CodePublicationReceipt(
        publication_id=f"publication-{publication_hash[:24]}",
        idempotency_key=identity,
        candidate=candidate,
        manifest_id=manifest.manifest_id,
        target_root=str(target_root),
        base_revision=context.base_revision,
        applied_revision=applied_revision,
        applied_at=datetime.now(timezone.utc),
    )
    return manifest, receipt


def _tool_message(
    runtime: ToolRuntime,
    content: str,
) -> ToolMessage:
    return ToolMessage(
        content=content,
        tool_call_id=runtime.tool_call_id or PUBLISH_REVIEWED_CANDIDATE_NAME,
    )


def _current_submission(state: dict[str, Any]) -> CodeWorkerSubmission:
    raw = state.get("code_worker_submission")
    if not isinstance(raw, dict):
        raise ValueError("code_worker_submission is missing")
    return CodeWorkerSubmission.model_validate(raw.get("submission", raw))


@tool(
    PUBLISH_REVIEWED_CANDIDATE_NAME,
    description=load_prompt("reviewers/code_publish_tool"),
)
def publish_reviewed_candidate(
    approved_artifact_paths: list[str],
    runtime: ToolRuntime,
) -> Command:
    """Publish only the Worker's proposed files approved by Reviewer."""

    state = runtime.state
    try:
        candidate = CodeCandidateRef.model_validate(state.get("code_candidate"))
        loop = CodeReviewLoopState.model_validate(state.get("code_review_loop"))
        submission = _current_submission(state)
        if submission.candidate != candidate:
            raise ValueError("Worker submission references a stale candidate")

        proposed = {
            _safe_relative_path(path)
            for path in submission.proposed_artifact_paths
        }
        approved = tuple(
            _safe_relative_path(path)
            for path in approved_artifact_paths
        )
        unexpected = sorted(set(approved) - proposed)
        if unexpected:
            raise ValueError(
                "Reviewer may only approve Worker-proposed artifacts; "
                f"unexpected={unexpected}"
            )

        existing_receipt = state.get("code_publication_receipt")
        existing_manifest = state.get("code_artifact_manifest")
        if existing_receipt is not None or existing_manifest is not None:
            if existing_receipt is None or existing_manifest is None:
                raise ValueError("incomplete persisted publication state")
            receipt = CodePublicationReceipt.model_validate(existing_receipt)
            manifest = CodeArtifactManifest.model_validate(existing_manifest)
            persisted_paths = tuple(item.path for item in manifest.files)
            if (
                receipt.candidate != candidate
                or manifest.candidate != candidate
                or persisted_paths != approved
            ):
                raise ValueError("publication replay does not match persisted receipt")
            updated_loop = loop
            idempotent = True
        else:
            context = CodePublicationContext.model_validate(
                state.get("code_publication_context")
            )
            manifest, receipt = publish_code_artifacts(
                candidate=candidate,
                approved_artifact_paths=approved,
                context=context,
            )
            updated_loop = record_code_publication(
                loop,
                candidate=candidate,
            )
            idempotent = False
    except (OSError, TypeError, ValueError) as error:
        return Command(
            update={
                "messages": [
                    _tool_message(
                        runtime,
                        (
                            "Reviewed candidate was not published: "
                            f"{error}. Correct the artifact proposal, request "
                            "Worker repair, or submit an escalated review."
                        ),
                    )
                ]
            }
        )

    manifest_value = manifest.model_dump(mode="json")
    receipt_value = receipt.model_dump(mode="json")
    runtime.stream_writer(
        {
            "type": "code_publication",
            "idempotent": idempotent,
            "manifest": manifest_value,
            "receipt": receipt_value,
        }
    )
    return Command(
        update={
            "code_review_loop": updated_loop.model_dump(mode="json"),
            "code_artifact_manifest": manifest_value,
            "code_publication_receipt": receipt_value,
            "messages": [
                _tool_message(
                    runtime,
                    (
                        "Reviewed candidate publication "
                        f"{'replayed' if idempotent else 'completed'} as "
                        f"{receipt.publication_id}. Submit the final review "
                        "with the exact receipt summary."
                    ),
                )
            ],
        }
    )


__all__ = [
    "CodePublicationContext",
    "PUBLISH_REVIEWED_CANDIDATE_NAME",
    "compute_code_tree_revision",
    "publish_code_artifacts",
    "publish_reviewed_candidate",
]
