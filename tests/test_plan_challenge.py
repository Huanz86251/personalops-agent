from __future__ import annotations

import pytest
from pydantic import ValidationError
from types import SimpleNamespace

from planning_models import ReplanDecision
from prompt_loader import load_prompt
from reporting.general_gate import general_review_reasons
from reporting.models import ReviewAttempt
from workers.code_review_models import CodeCandidateRef, CodeReviewReport
from workers.plan_challenge import PlanChallenge
from workers.submission import WorkerSubmission, submit_for_review


def _challenge() -> PlanChallenge:
    return PlanChallenge(
        reason="The Step drops the message-derived filter before a write.",
        original_requirement="Read Laura's recommendation, then select matching notes.",
        conflicting_plan_text="Reply with every note title.",
        evidence_tool_call_ids=["E2", "E2"],
        requested_revision="Restore the message-derived filter before choosing notes.",
    )


def test_plan_challenge_is_reason_first_and_deduplicates_evidence():
    challenge = _challenge()
    assert list(PlanChallenge.model_fields)[0] == "reason"
    assert challenge.evidence_tool_call_ids == ["E2"]


def test_general_challenge_forces_independent_review():
    challenge = _challenge().model_dump(mode="json")
    trace = {
        "general_result": {
            "status": "COMPLETED",
            "unresolved_items": [],
            "files": [],
            "plan_challenge": challenge,
        }
    }
    assert "plan_challenge" in general_review_reasons(
        type("Step", (), {"artifact_outputs": []})(), trace, type("Packet", (), {"attempts": []})()
    )


def test_generic_submission_and_review_packet_preserve_challenge():
    submission = WorkerSubmission(
        plan_challenge=_challenge(),
        summary="Stopped before writing.",
        final_conclusion="Plan conflict requires independent review.",
    )
    attempt = ReviewAttempt(
        attempt=1,
        submission_source="WORKER",
        finish_reason="READY_FOR_REVIEW",
        stop_reason="Plan challenge submitted.",
        summary=submission.summary,
        final_conclusion=submission.final_conclusion,
        plan_challenge=submission.plan_challenge,
    )
    assert attempt.plan_challenge == _challenge()


def test_code_reviewer_can_only_confirm_challenge_as_escalated_stop():
    candidate = CodeCandidateRef(
        event_id="event-1",
        step_id=1,
        attempt_id="attempt-1",
        workspace_id="workspace-1",
        candidate_revision=1,
    )
    common = dict(
        candidate=candidate,
        confirmed_plan_challenge=_challenge(),
        summary="The frozen task conflicts with the user request.",
        verification_summary="Compared the request and frozen code task.",
        verdict="ESCALATED",
        recommended_action="STOP",
    )
    assert CodeReviewReport(**common).confirmed_plan_challenge is not None
    with pytest.raises(ValidationError):
        CodeReviewReport(**{**common, "recommended_action": "CONTINUE"})


def test_scheduler_can_reject_challenge_with_reasoned_worker_instruction():
    decision = ReplanDecision(
        reason="The original request and Step describe the same target set.",
        action="RETURN_TO_WORKER",
        worker_instruction="Keep the existing target boundary and continue from the saved read result.",
    )
    assert decision.remaining_steps == []
    with pytest.raises(ValidationError):
        ReplanDecision(
            reason="Rejected.",
            action="RETURN_TO_WORKER",
            worker_instruction=None,
        )


def test_exhausted_challenge_budget_is_rejected_before_review():
    submission = WorkerSubmission(
        plan_challenge=_challenge(),
        summary="Stopped before writing.",
        final_conclusion="Plan conflict requires review.",
    )
    runtime = SimpleNamespace(
        state={"worker_plan_challenges_remaining": 0},
        tool_call_id="submit-1",
    )
    command = submit_for_review.func(submission, runtime)
    assert "budget is exhausted" in command.update["messages"][0].content


def test_worker_and_reviewer_prompts_explain_the_bounded_route():
    worker_prompt = load_prompt("workers/general_worker")
    reporter_prompt = load_prompt("reporters/step_report")
    scheduler_prompt = load_prompt("planning/scheduler")
    assert "不要轻易提出异议" in worker_prompt
    assert "ReviewAttempt含plan_challenge" in reporter_prompt
    assert "ScopeContract" in scheduler_prompt
    assert "execution_guidance" in scheduler_prompt
