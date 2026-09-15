"""Run-local Git history for accepted CODE integration states."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from git import Actor, Repo
from pydantic import BaseModel, ConfigDict, Field, field_validator


_HARNESS_ACTOR = Actor("PersonalOps Harness", "harness@personalops.local")
_DIFF_HARD_MAX_CHARS = 50_000


class IntegrationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class IntegrationBaselineReceipt(IntegrationModel):
    run_id: str = Field(min_length=1)
    repository_root: str = Field(min_length=1)
    baseline_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    created_at: datetime


class IntegrationStatus(IntegrationModel):
    run_id: str = Field(min_length=1)
    repository_root: str = Field(min_length=1)
    baseline_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    head_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    working_tree_clean: bool
    changed_files: tuple[str, ...] = ()
    accepted_commit_count: int = Field(ge=0)


class IntegrationHistoryEntry(IntegrationModel):
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    parent_commits: tuple[str, ...] = ()
    summary: str
    committed_at: datetime


class IntegrationDiff(IntegrationModel):
    base_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    target_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    text: str
    truncated: bool = False


class IntegrationCommitView(IntegrationModel):
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    parent_commits: tuple[str, ...] = ()
    summary: str
    message: str
    committed_at: datetime
    changed_files: tuple[str, ...] = ()
    diff: str
    diff_truncated: bool = False


class IntegrationCommitReceipt(IntegrationModel):
    receipt_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    step_id: int = Field(ge=1)
    candidate_revision: int = Field(ge=1)
    manifest_id: str = Field(min_length=1)
    publication_id: str = Field(min_length=1)
    parent_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    accepted_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    changed_files: tuple[str, ...] = ()
    review_summary: str
    test_summary: str
    committed_at: datetime

    @field_validator("changed_files")
    @classmethod
    def validate_changed_files(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(str(value).strip().replace("\\", "/") for value in values)
        if any(not value or value.startswith("/") or ".." in value.split("/") for value in normalized):
            raise ValueError("changed_files must contain safe relative paths")
        if len(normalized) != len(set(normalized)):
            raise ValueError("changed_files must be unique")
        return normalized


def _write_json_atomically(path: Path, payload: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        temporary.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _bounded_line(value: str, *, max_chars: int = 1000) -> str:
    return " ".join(str(value or "").strip().split())[:max_chars]


class IntegrationRepository:
    """Thin GitPython adapter; Git remains the versioning engine."""

    def __init__(
        self,
        *,
        run_id: str,
        root: Path,
        receipts_root: Path,
    ) -> None:
        self.run_id = str(run_id).strip()
        if not self.run_id:
            raise ValueError("run_id cannot be empty")
        self.root = Path(root).resolve()
        self.receipts_root = Path(receipts_root).resolve()
        self.baseline_receipt_path = self.receipts_root / "integration-baseline.json"
        self.commit_receipts_root = self.receipts_root / "integration-commits"

    def initialize(self) -> IntegrationBaselineReceipt:
        """Create one real local Git repository and immutable baseline commit."""

        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root / ".git").exists():
            with Repo(self.root) as repo:
                if not self.baseline_receipt_path.is_file():
                    raise ValueError("integration Git repository has no baseline receipt")
                receipt = IntegrationBaselineReceipt.model_validate_json(
                    self.baseline_receipt_path.read_text(encoding="utf-8")
                )
                repo.commit(receipt.baseline_commit)
                return receipt

        repo = Repo.init(self.root, initial_branch="personalops-integration")
        try:
            repo.git.add("-A")
            commit = repo.index.commit(
                f"personalops: baseline for {self.run_id}",
                author=_HARNESS_ACTOR,
                committer=_HARNESS_ACTOR,
            )
            commit_sha = commit.hexsha
        finally:
            repo.close()
        receipt = IntegrationBaselineReceipt(
            run_id=self.run_id,
            repository_root=str(self.root),
            baseline_commit=commit_sha,
            created_at=datetime.now(timezone.utc),
        )
        _write_json_atomically(self.baseline_receipt_path, receipt)
        return receipt

    @staticmethod
    def _changed_paths(repo: Repo) -> tuple[str, ...]:
        paths = set(repo.untracked_files)
        for diff in (*repo.index.diff(None), *repo.index.diff("HEAD")):
            if diff.a_path:
                paths.add(diff.a_path)
            if diff.b_path:
                paths.add(diff.b_path)
        return tuple(sorted(path.replace("\\", "/") for path in paths))

    def status(self) -> IntegrationStatus:
        self.initialize()
        baseline = IntegrationBaselineReceipt.model_validate_json(
            self.baseline_receipt_path.read_text(encoding="utf-8")
        )
        receipts = list(self.commit_receipts_root.glob("*.json"))
        with Repo(self.root) as repo:
            changed = self._changed_paths(repo)
            return IntegrationStatus(
                run_id=self.run_id,
                repository_root=str(self.root),
                baseline_commit=baseline.baseline_commit,
                head_commit=repo.head.commit.hexsha,
                working_tree_clean=not changed,
                changed_files=changed,
                accepted_commit_count=len(receipts),
            )

    def history(self, *, max_entries: int = 10) -> tuple[IntegrationHistoryEntry, ...]:
        if max_entries < 1 or max_entries > 50:
            raise ValueError("max_entries must be between 1 and 50")
        self.initialize()
        with Repo(self.root) as repo:
            return tuple(
                IntegrationHistoryEntry(
                    commit=commit.hexsha,
                    parent_commits=tuple(parent.hexsha for parent in commit.parents),
                    summary=commit.summary,
                    committed_at=commit.committed_datetime.astimezone(timezone.utc),
                )
                for commit in repo.iter_commits(max_count=max_entries)
            )

    def diff(
        self,
        base_commit: str,
        target_commit: str = "HEAD",
        *,
        max_chars: int = 12_000,
    ) -> IntegrationDiff:
        if max_chars < 1 or max_chars > _DIFF_HARD_MAX_CHARS:
            raise ValueError(
                f"max_chars must be between 1 and {_DIFF_HARD_MAX_CHARS}"
            )
        self.initialize()
        with Repo(self.root) as repo:
            base = repo.commit(base_commit)
            target = repo.commit(target_commit)
            base_sha = base.hexsha
            target_sha = target.hexsha
            text = repo.git.diff(
                "--no-ext-diff",
                "--no-color",
                base_sha,
                target_sha,
            )
        truncated = len(text) > max_chars
        if truncated:
            marker = "\n...[diff truncated by integration context limit]...\n"
            remaining = max_chars - len(marker)
            head = max(remaining // 2, 0)
            text = text[:head] + marker + text[-(remaining - head):]
        return IntegrationDiff(
            base_commit=base_sha,
            target_commit=target_sha,
            text=text,
            truncated=truncated,
        )

    def show_commit(
        self,
        commit_id: str,
        *,
        max_diff_chars: int = 12_000,
    ) -> IntegrationCommitView:
        """Return one bounded accepted change without loading full history."""

        if max_diff_chars < 1 or max_diff_chars > _DIFF_HARD_MAX_CHARS:
            raise ValueError(
                f"max_diff_chars must be between 1 and {_DIFF_HARD_MAX_CHARS}"
            )
        self.initialize()
        with Repo(self.root) as repo:
            commit = repo.commit(commit_id)
            text = repo.git.show(
                "--format=",
                "--no-ext-diff",
                "--no-color",
                commit.hexsha,
            )
            truncated = len(text) > max_diff_chars
            if truncated:
                marker = "\n...[commit diff truncated by integration context limit]...\n"
                remaining = max_diff_chars - len(marker)
                head = max(remaining // 2, 0)
                text = text[:head] + marker + text[-(remaining - head):]
            return IntegrationCommitView(
                commit=commit.hexsha,
                parent_commits=tuple(parent.hexsha for parent in commit.parents),
                summary=commit.summary,
                message=commit.message,
                committed_at=commit.committed_datetime.astimezone(timezone.utc),
                changed_files=tuple(sorted(commit.stats.files)),
                diff=text,
                diff_truncated=truncated,
            )

    def commit_accepted(
        self,
        *,
        step_id: int,
        candidate_revision: int,
        manifest_id: str,
        publication_id: str,
        approved_paths: tuple[str, ...],
        review_summary: str,
        test_summary: str,
    ) -> IntegrationCommitReceipt:
        """Commit exactly one Reviewer-approved publication, idempotently."""

        identity = f"{self.run_id}:{publication_id}:{manifest_id}"
        receipt_hash = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        receipt_path = self.commit_receipts_root / f"accepted-{receipt_hash[:24]}.json"
        self.initialize()
        with Repo(self.root) as repo:
            if receipt_path.is_file():
                receipt = IntegrationCommitReceipt.model_validate_json(
                    receipt_path.read_text(encoding="utf-8")
                )
                repo.commit(receipt.accepted_commit)
                return receipt

            expected = tuple(dict.fromkeys(path.replace("\\", "/") for path in approved_paths))
            changed = self._changed_paths(repo)
            unexpected = sorted(set(changed) - set(expected))
            if unexpected:
                raise ValueError(
                    "integration contains changes outside the approved manifest: "
                    + ", ".join(unexpected)
                )
            missing = [path for path in expected if not (self.root / path).is_file()]
            if missing:
                raise ValueError(
                    "approved integration files are missing: " + ", ".join(missing)
                )

            parent = repo.head.commit.hexsha
            if expected:
                repo.index.add(list(expected))
            message = (
                f"personalops: accept step {step_id} revision {candidate_revision}\n\n"
                f"Run: {self.run_id}\n"
                f"Manifest: {manifest_id}\n"
                f"Publication: {publication_id}\n"
                f"Review: {_bounded_line(review_summary)}\n"
                f"Tests: {_bounded_line(test_summary)}"
            )
            commit = repo.index.commit(
                message,
                author=_HARNESS_ACTOR,
                committer=_HARNESS_ACTOR,
                parent_commits=(repo.head.commit,),
            )
            commit_sha = commit.hexsha
            if self._changed_paths(repo):
                raise RuntimeError("integration repository is dirty after accepted commit")
        receipt = IntegrationCommitReceipt(
            receipt_id=f"integration-{receipt_hash[:24]}",
            idempotency_key=identity,
            run_id=self.run_id,
            step_id=step_id,
            candidate_revision=candidate_revision,
            manifest_id=manifest_id,
            publication_id=publication_id,
            parent_commit=parent,
            accepted_commit=commit_sha,
            changed_files=expected,
            review_summary=_bounded_line(review_summary),
            test_summary=_bounded_line(test_summary),
            committed_at=datetime.now(timezone.utc),
        )
        _write_json_atomically(receipt_path, receipt)
        return receipt


__all__ = [
    "IntegrationBaselineReceipt",
    "IntegrationCommitReceipt",
    "IntegrationCommitView",
    "IntegrationDiff",
    "IntegrationHistoryEntry",
    "IntegrationRepository",
    "IntegrationStatus",
]
