from pydantic import ValidationError

from planning_models import FinalReviewDecision, StepReport, SupervisorDecision
from skill_runtime.preparation import SkillChoice
from toolset_router import ModelToolsetSelection
from workers.submission import WorkerSubmission


def test_selector_drops_bad_reason_but_keeps_control_selection():
    result = ModelToolsetSelection.model_validate(
        {"reason": "x" * 501, "selected_toolsets": ["WEB_RESEARCH"]}
    )
    assert result.reason == ""
    assert result.selected_toolsets == ["WEB_RESEARCH"]


def test_selector_control_choice_remains_strict():
    try:
        ModelToolsetSelection.model_validate(
            {"reason": "valid", "selected_toolsets": []}
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("an empty required capability selection must not be hidden")


def test_skill_selector_drops_bad_reason_but_keeps_route():
    result = SkillChoice.model_validate(
        {"reason": {"bad": "shape"}, "skill_ids": [], "execution_route": "DIRECT"}
    )
    assert result.reason == ""
    assert result.execution_route == "DIRECT"


def test_step_report_drops_only_malformed_supplementary_fields():
    report = StepReport.model_validate(
        {
            "step_id": 1,
            "summary": "verified summary",
            "stop_reason": "done",
            "status": "COMPLETED",
            "criterion_results": [],
            "completed_work": {"bad": "shape"},
            "errors": {"bad": "shape"},
            "next_action": ["bad", "shape"],
        }
    )
    assert report.completed_work == []
    assert report.errors == []
    assert report.next_action is None


def test_web_report_keeps_replan_decision_strict():
    try:
        WorkerSubmission.model_validate(
            {
                "summary": "summary",
                "final_conclusion": "done",
                "stop_reason": "done",
                "request_replan": True,
                "replan_reason": None,
                "errors": {"bad": "shape"},
            }
        )
    except ValidationError as error:
        assert "replan_reason" in str(error)
    else:
        raise AssertionError("a routing-changing replan request must remain strict")


def test_final_reviewer_drops_bad_rationale_but_keeps_action_contract():
    decision = FinalReviewDecision.model_validate(
        {
            "review_reason": ["bad", "shape"],
            "action": "FINAL",
            "final_answer": "answer",
        }
    )
    assert decision.review_reason == "依据当前可见证据完成最终审核。"
    try:
        FinalReviewDecision.model_validate(
            {"action": "RETURN_TO_WORKER", "repair_request": None}
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("RETURN_TO_WORKER must still include its control payload")


def test_scheduler_plan_fields_remain_strict():
    try:
        SupervisorDecision.model_validate(
            {
                "action": "PLAN",
                "plan_objective": "objective",
                "plan_success_criteria": [],
                "steps": [],
            }
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("an empty executable plan must still be rejected")
