"""Contract tests for durable Code attempt terminal records."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from workers.code_attempt_models import (
    CodeArtifactEntry,
    CodeArtifactManifest,
    CodeAttemptArchive,
    CodeAttemptFinalRecord,
    CodePublicationReceipt,
)
from workers.code_review_models import (
    CodeCandidateRef,
    CodeCheckResult,
    CodeReviewReport,
)


NOW = datetime(2026, 9, 4, 8, 0, tzinfo=timezone.utc)
SHA256 = "a" * 64


def candidate(revision: int = 3) -> CodeCandidateRef:
    return CodeCandidateRef(
        event_id="event-1",
        step_id=1,
        attempt_id="attempt-2",
        workspace_id="workspace-2",
        candidate_revision=revision,
    )


def manifest(ref: CodeCandidateRef | None = None) -> CodeArtifactManifest:
    return CodeArtifactManifest(
        manifest_id="manifest-1",
        candidate=ref or candidate(),
        files=(
            CodeArtifactEntry(
                path="src/app.py",
                kind="SOURCE",
                sha256=SHA256,
                size_bytes=128,
            ),
        ),
        created_at=NOW + timedelta(minutes=1),
    )


def archive() -> CodeAttemptArchive:
    return CodeAttemptArchive(
        archive_id="archive-1",
        root_path="D:/runs/event-1/attempt-2",
        candidate_snapshot_path="candidate",
        reviewer_snapshot_path="review",
        preserved_at=NOW + timedelta(minutes=2),
        retain_until=NOW + timedelta(days=7),
    )


def common_record() -> dict:
    return {
        "record_id": "record-1",
        "candidate": candidate(),
        "parent_attempt_id": "attempt-1",
        "terminal_reason": "Attempt reached a durable terminal boundary.",
        "started_at": NOW,
        "finalized_at": NOW + timedelta(minutes=3),
        "worker_checkpoint_id": "worker-thread-2",
        "reviewer_checkpoint_id": "reviewer-thread-2",
        "docker_image": "personalops-code-agent@sha256:abc",
        "archive": archive(),
        "artifact_manifest": manifest(),
    }


def passed_report() -> CodeReviewReport:
    return CodeReviewReport(
        candidate=candidate(),
        verdict="PASSED",
        summary="The frozen candidate passed focused checks.",
        verification_summary="One focused test passed.",
        check_results=(
            CodeCheckResult(
                check_id="focused-test",
                description="Run the focused feature test.",
                status="PASSED",
                summary="The feature behaved as required.",
            ),
        ),
        approved_artifact_paths=("src/app.py",),
        published_artifact_paths=("src/app.py",),
        delivery_location=str(Path("D:/PythonProject")),
        publication_id="publication-1",
        applied_revision="applied-sha",
    )


def publication() -> CodePublicationReceipt:
    return CodePublicationReceipt(
        publication_id="publication-1",
        idempotency_key="event-1:step-1:attempt-2:revision-3",
        candidate=candidate(),
        manifest_id="manifest-1",
        target_root="D:/PythonProject",
        base_revision="base-sha",
        applied_revision="applied-sha",
        applied_at=NOW + timedelta(minutes=2, seconds=30),
    )


def test_cancelled_preserves_archive_without_publication() -> None:
    record = CodeAttemptFinalRecord(
        **common_record(),
        outcome="CANCELLED",
    )

    assert record.outcome == "CANCELLED"
    assert record.archive.candidate_snapshot_path == "candidate"
    assert record.publication is None

    with pytest.raises(ValidationError, match="Only APPLIED"):
        CodeAttemptFinalRecord(
            **common_record(),
            outcome="CANCELLED",
            publication=publication(),
        )


def test_superseded_is_a_non_failure_terminal_archive() -> None:
    record = CodeAttemptFinalRecord(
        **common_record(),
        outcome="SUPERSEDED",
    )

    assert record.outcome == "SUPERSEDED"
    assert record.failure_stage is None
    assert record.publication is None


def test_failed_requires_the_failure_stage() -> None:
    with pytest.raises(ValidationError, match="failure_stage"):
        CodeAttemptFinalRecord(
            **common_record(),
            outcome="FAILED",
        )

    failed = CodeAttemptFinalRecord(
        **common_record(),
        outcome="FAILED",
        failure_stage="REVIEWER",
    )
    assert failed.failure_stage == "REVIEWER"


def test_publisher_failure_remains_failed_without_a_receipt() -> None:
    failed = CodeAttemptFinalRecord(
        **common_record(),
        outcome="FAILED",
        failure_stage="PUBLISHER",
        review_report=CodeReviewReport(
            candidate=candidate(),
            verdict="ESCALATED",
            summary="Verification passed but publication failed.",
            verification_summary="The focused test passed before publication.",
            recommended_action="CONTINUE",
        ),
    )

    assert failed.failure_stage == "PUBLISHER"
    assert failed.publication is None


def test_applied_requires_review_acceptance_and_publication_receipt() -> None:
    record = CodeAttemptFinalRecord(
        **common_record(),
        outcome="APPLIED",
        review_report=passed_report(),
        publication=publication(),
    )

    assert record.outcome == "APPLIED"
    assert record.publication.manifest_id == record.artifact_manifest.manifest_id

    with pytest.raises(ValidationError, match="Publisher receipt"):
        CodeAttemptFinalRecord(
            **common_record(),
            outcome="APPLIED",
            review_report=passed_report(),
        )


def test_applied_rejects_a_stale_publication_candidate() -> None:
    stale_receipt = publication().model_copy(
        update={"candidate": candidate(revision=2)}
    )

    with pytest.raises(ValidationError, match="stale candidate"):
        CodeAttemptFinalRecord(
            **common_record(),
            outcome="APPLIED",
            review_report=passed_report(),
            publication=stale_receipt,
        )


@pytest.mark.parametrize("unsafe_path", ["../secret", "/absolute", "C:\\temp"])
def test_manifest_rejects_paths_outside_the_candidate(unsafe_path: str) -> None:
    with pytest.raises(ValidationError, match="safe relative path"):
        CodeArtifactEntry(
            path=unsafe_path,
            kind="SOURCE",
            sha256=SHA256,
            size_bytes=1,
        )
