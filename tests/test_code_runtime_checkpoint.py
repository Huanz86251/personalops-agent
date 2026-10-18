"""Tests for durable CODE Docker ownership and cleanup authorization."""

from datetime import datetime, timezone
import json

import pytest
from pydantic import ValidationError

from planning_models import CodeRequirement, CodeTaskContract
from workers.code_review_models import CodeCandidateRef, CodeReviewLoopState
from workers.code_runtime_checkpoint import (
    CodeRuntimeCheckpoint,
    CodeRuntimeRecoveryState,
    CodeRuntimeCheckpointStore,
    CodeSandboxPairRecord,
)


def checkpoint(*, phase: str, cleanup_authorized: bool = False):
    candidate = CodeCandidateRef(
        event_id="event-1",
        step_id=1,
        attempt_id="attempt-1",
        workspace_id="workspace-1",
        candidate_revision=1,
    )
    loop = CodeReviewLoopState(
        candidate=candidate,
        worker_checkpoint_id="worker-thread",
        reviewer_checkpoint_id="reviewer-thread",
        status="REPLAN_REQUIRED",
        terminal_summary="Reviewer requested Scheduler control.",
    )
    return CodeRuntimeCheckpoint(
        checkpoint_id="checkpoint-1",
        runtime_session_id="session-1",
        run_id="event-1",
        step_id=1,
        attempt_id="attempt-1",
        generation=1,
        phase=phase,
        preemption_policy=(
            "DEFER_UNTIL_ROLE_BOUNDARY"
            if phase == "REVIEWER_RUNNING"
            else "SAFE_POINT"
        ),
        worker_checkpoint_id="worker-thread",
        reviewer_checkpoint_id="reviewer-thread",
        integration_root="integration",
        integration_head_commit="a" * 40,
        base_revision="tree-revision",
        attempt_root="attempt",
        candidate_snapshot_path="candidate/revision-1",
        reviewer_snapshot_path="reviewer/final",
        sandbox=CodeSandboxPairRecord(
            pair_id="pair-1",
            workspace_id="workspace-1",
            candidate_volume="candidate-volume",
            review_volume="review-volume",
            worker_container="worker-container",
            reviewer_container="reviewer-container",
            image="sandbox-image",
            active_role=None,
        ),
        recovery_state=CodeRuntimeRecoveryState(
            contract=CodeTaskContract(
                requirements=(
                    CodeRequirement(
                        requirement_id="feature",
                        statement="Implement the requested feature.",
                    ),
                ),
                validation_expectations=("Run a focused check.",),
            ),
            candidate=candidate,
            review_loop=loop,
            worker_submission={"submission": "saved"},
            started_at=datetime.now(timezone.utc),
            worker_id="workspace-1",
        ),
        cleanup_authorized=cleanup_authorized,
        committed_at=datetime.now(timezone.utc),
    )


def test_checkpoint_is_atomically_persisted_and_read_back(tmp_path) -> None:
    store = CodeRuntimeCheckpointStore()
    expected = checkpoint(phase="AWAITING_REPLAN")

    persisted = store.commit(tmp_path, expected)

    assert persisted == expected
    assert store.load(tmp_path) == expected
    assert not list(tmp_path.glob("*.tmp"))


def test_ownership_read_preserves_legacy_contract_and_current_strict_schema(tmp_path) -> None:
    store = CodeRuntimeCheckpointStore()
    legacy = checkpoint(phase="AWAITING_REPLAN").model_dump(mode="json")
    recovery = legacy["recovery_state"]
    recovery["contract"]["implementation_guidance"] = ["Old implementation hint."]
    recovery["review_loop"].update(
        scheduler_epoch=1,
        pending_scheduler_directive=None,
        scheduler_continuations=[],
    )
    original = json.dumps(legacy).encode("utf-8")
    path = store.path_for(tmp_path)
    path.write_bytes(original)

    assert store.read_run_id(tmp_path) == "event-1"
    with pytest.raises(ValidationError, match="implementation_guidance"):
        store.load(tmp_path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("payload", [[], {}, {"run_id": " "}, {"run_id": 123}])
def test_ownership_read_rejects_missing_or_ambiguous_identity(tmp_path, payload) -> None:
    store = CodeRuntimeCheckpointStore()
    store.path_for(tmp_path).write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        store.read_run_id(tmp_path)


def test_ownership_read_has_no_side_effect_for_missing_checkpoint(tmp_path) -> None:
    assert CodeRuntimeCheckpointStore().read_run_id(tmp_path) is None
    assert not list(tmp_path.iterdir())


def test_reviewer_phase_is_non_preemptible() -> None:
    reviewer = checkpoint(phase="REVIEWER_RUNNING")
    assert reviewer.preemption_policy == "DEFER_UNTIL_ROLE_BOUNDARY"

    invalid = reviewer.model_dump()
    invalid["preemption_policy"] = "SAFE_POINT"
    with pytest.raises(ValidationError, match="requires preemption_policy"):
        CodeRuntimeCheckpoint.model_validate(invalid)


def test_cleanup_requires_terminal_committed_record_for_same_pair(tmp_path) -> None:
    store = CodeRuntimeCheckpointStore()
    store.commit(tmp_path, checkpoint(phase="AWAITING_REPLAN"))

    with pytest.raises(RuntimeError, match="not authorized"):
        store.require_cleanup_authorized(tmp_path, pair_id="pair-1")

    store.commit(
        tmp_path,
        checkpoint(phase="TERMINAL", cleanup_authorized=True),
    )
    with pytest.raises(RuntimeError, match="evidence is missing"):
        store.require_cleanup_authorized(tmp_path, pair_id="pair-1")

    (tmp_path / "candidate" / "revision-1").mkdir(parents=True)
    (tmp_path / "reviewer" / "final").mkdir(parents=True)
    (tmp_path / "final_record.json").write_text("{}", encoding="utf-8")
    authorized = store.require_cleanup_authorized(
        tmp_path,
        pair_id="pair-1",
    )
    assert authorized.cleanup_authorized

    with pytest.raises(RuntimeError, match="another pair"):
        store.require_cleanup_authorized(tmp_path, pair_id="pair-2")
