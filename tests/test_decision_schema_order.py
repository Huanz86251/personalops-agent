"""Decision schemas must ask for evidence or reasons before outcomes."""

from evals.appworld.tool_descriptions import AppWorldBindingCheck, AppWorldCallInput
from memory import MemoryResolution
from planning_models import (
    FinalReviewDecision,
    ReplanDecision,
    StepCriterionResult,
    StepReport,
)
from reporting.criteria import CriterionReferenceResult, criterion_claim_repair_feedback
from skill_runtime.preparation import SkillChoice
from workers.code_review_models import (
    CodeCheckResult,
    CodeContinuationSubmission,
    CodeReviewReport,
    CodeSelfCheck,
    CodeWorkerRepairResponse,
    SchedulerCodeDecision,
)
from workers.leadership_models import LeadershipDecision
from workers.submission import WorkerCriterionClaim, WorkerSubmission


def _before(model, earlier: str, later: str) -> None:
    fields = list(model.model_fields)
    assert fields.index(earlier) < fields.index(later), (
        f"{model.__name__}: expected {earlier} before {later}, got {fields}"
    )


def test_reason_precedes_model_decisions() -> None:
    for model, reason, result in [
        (AppWorldBindingCheck, "reason", "assessment"),
        (AppWorldCallInput, "reason", "action_phase"),
        (MemoryResolution, "reason", "action"),
        (SkillChoice, "reason", "skill_ids"),
        (ReplanDecision, "reason", "action"),
        (FinalReviewDecision, "replan_reason", "action"),
        (LeadershipDecision, "reason", "action"),
        (SchedulerCodeDecision, "reason", "action"),
        (CodeWorkerRepairResponse, "reasons", "action"),
        (CodeContinuationSubmission, "reasons", "action"),
    ]:
        _before(model, reason, result)


def test_evidence_precedes_worker_and_reviewer_conclusions() -> None:
    _before(WorkerCriterionClaim, "evidence_tool_call_ids", "conclusion")
    _before(WorkerSubmission, "criterion_claims", "final_conclusion")
    _before(WorkerSubmission, "unresolved_items", "final_conclusion")
    _before(StepCriterionResult, "evidence", "status")
    _before(CriterionReferenceResult, "evidence", "status")
    _before(StepReport, "summary", "status")
    _before(StepReport, "stop_reason", "status")
    _before(StepReport, "replan_reason", "request_replan")
    _before(CodeSelfCheck, "summary", "outcome")
    _before(CodeCheckResult, "summary", "status")
    _before(CodeReviewReport, "verification_summary", "verdict")
    _before(CodeReviewReport, "check_results", "verdict")
    _before(CodeReviewReport, "evidence_refs", "verdict")


def test_worker_claim_repair_example_matches_real_field_names_and_order() -> None:
    feedback = criterion_claim_repair_feedback(
        {"worker_criterion_refs": {"C1": "真实验收条件"}},
        ValueError("missing"),
    )
    assert "evidence_tool_call_ids" in feedback
    assert "evidence_refs" not in feedback
    assert feedback.index("evidence_tool_call_ids") < feedback.index("conclusion")
